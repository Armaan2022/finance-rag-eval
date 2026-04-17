from langchain_core.documents import Document
from rank_bm25 import BM25Okapi


# ---------------------------------------------------------------------------
# BM25 index
# ---------------------------------------------------------------------------

def build_bm25_index(chunks: list[Document]) -> tuple[BM25Okapi, list[Document]]:
    """
    Build a BM25 index from a list of Document chunks.

    BM25 works on tokenized text — we split on whitespace and lowercase
    everything. Simple tokenization is fine here; finance documents don't
    need stemming or stop-word removal for our purposes.

    Returns:
        (bm25_index, chunks) — the index and the original chunks list
        in the same order, so index position maps back to the document.
    """
    tokenized = [
        doc.page_content.lower().split()
        for doc in chunks
    ]
    return BM25Okapi(tokenized), chunks


def bm25_search(
    query: str,
    bm25: BM25Okapi,
    chunks: list[Document],
    k: int = 20,
) -> list[tuple[Document, int]]:
    """
    Search the BM25 index and return (document, rank) pairs.

    We fetch more candidates (k=20) than the final result count because
    RRF needs a broad pool from both methods to work well. The fusion
    step will trim down to the final top-k.

    Returns:
        List of (Document, rank) tuples, ranked 1..k (1 = best match).
    """
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    indexed_scores = list(enumerate(scores))
    indexed_scores.sort(key=lambda x: x[1], reverse=True)  # sort by score desc
    top_indices = [i for i, score in indexed_scores[:k]]

    return [(chunks[i], rank + 1) for rank, i in enumerate(top_indices)]


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[Document, int]]],
    k: int = 60,
    top_n: int = 5,
) -> list[Document]:
    """
    Merge multiple ranked lists into one using Reciprocal Rank Fusion.

    Args:
        ranked_lists: Each element is a list of (Document, rank) tuples
                      from one retrieval method (vector or BM25).
        k:            RRF constant. 60 is the value from the original paper.
                      Higher k = less reward for being ranked #1 specifically.
        top_n:        How many final results to return.

    How it works:
        For each document, sum up 1/(rank + k) across all lists it appears in.
        Documents that rank well in multiple lists get the highest combined score.
        Documents only found by one method still contribute but score lower.

    Identity for deduplication:
        Two chunks are the "same document" if they share the same page_content.
        We use a dict keyed on content to accumulate scores.
    """
    scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}  # content → Document, for reconstruction

    for ranked_list in ranked_lists:
        for doc, rank in ranked_list:
            key = doc.page_content
            scores[key] = scores.get(key, 0.0) + 1.0 / (rank + k)
            doc_map[key] = doc

    # Sort by descending RRF score and return top_n Document objects
    sorted_keys = sorted(scores, key=lambda x: scores[x], reverse=True)
    return [doc_map[key] for key in sorted_keys[:top_n]]


# ---------------------------------------------------------------------------
# Hybrid retriever
# ---------------------------------------------------------------------------

class HybridRetriever:
    """
    A retriever that combines ChromaDB vector search with BM25 keyword
    search using Reciprocal Rank Fusion.

    Usage:
        retriever = HybridRetriever(vector_store, chunks, k=5)
        docs = retriever.invoke("Apple's revenue in 2025")

    The invoke() interface matches LangChain's standard retriever interface,
    so this is a drop-in replacement for the pure vector retriever used in
    pipeline.py — no changes needed to the RAG chain.

    Args:
        vector_store: A LangChain Chroma instance.
        chunks:       The same Document list used to build the vector store.
        k:            Number of final results to return.
        fetch_k:      How many candidates to fetch from each method before
                      fusion. More candidates = better fusion quality but
                      slower. 20 is a good default.
        metadata_filter: Optional ChromaDB filter dict, e.g.:
                      {"ticker": "AAPL"} to restrict search to Apple only.
                      We'll use this in the Streamlit UI (Week 4).
    """

    def __init__(
        self,
        vector_store,
        chunks: list[Document],
        k: int = 5,
        fetch_k: int = 20,
        metadata_filter: dict | None = None,
    ):
        self.vector_store = vector_store
        self.k = k
        self.fetch_k = fetch_k
        self.metadata_filter = metadata_filter
        self.bm25, self.chunks = build_bm25_index(chunks)

    def invoke(self, query: str) -> list[Document]:
        """
        Run hybrid retrieval: vector + BM25 → RRF fusion.
        """
        # --- Vector search ---
        # get_retriever() from vector_store.py with fetch_k=20 candidates.
        from src.retrieval.vector_store import get_retriever

        vector_retriever = get_retriever(
            self.vector_store,
            k=self.fetch_k,
            metadata_filter=self.metadata_filter,
        )
        vector_docs = vector_retriever.invoke(query)
        vector_ranked = [(doc, rank + 1) for rank, doc in enumerate(vector_docs)]

        # --- BM25 search ---
        # Apply metadata filter manually for BM25. 
        searchable_chunks = []

        if self.metadata_filter is not None:
            for chunk in self.chunks:
                match = True

                for key, value in self.metadata_filter.items():
                    if chunk.metadata.get(key) != value:
                        match = False
                        break

                if match:
                    searchable_chunks.append(chunk)
        else:
            searchable_chunks = self.chunks

        bm25_index, filtered_chunks = build_bm25_index(searchable_chunks)
        bm25_ranked = bm25_search(query, bm25_index, filtered_chunks, k=self.fetch_k)

        # --- RRF fusion ---
        return reciprocal_rank_fusion(
            [vector_ranked, bm25_ranked],
            top_n=self.k,
        )


# ---------------------------------------------------------------------------
# Quick comparison: vector-only vs hybrid
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from dotenv import load_dotenv
    from src.ingestion.sec_edgar import ingest_tickers
    from src.processing.chunker import chunk_documents
    from src.retrieval.vector_store import build_vector_store, load_vector_store, get_retriever
    from pathlib import Path

    load_dotenv()

    CHROMA_DIR = Path(__file__).resolve().parents[2] / "chroma_db"

    print("Loading data...")
    documents = ingest_tickers(["AAPL", "MSFT"], max_filings_per_ticker=1)
    chunks = chunk_documents(documents, config_name="medium")

    if CHROMA_DIR.exists() and any(CHROMA_DIR.iterdir()):
        vector_store = load_vector_store()
    else:
        vector_store = build_vector_store(chunks)

    vector_retriever = get_retriever(vector_store, k=5)
    hybrid_retriever = HybridRetriever(vector_store, chunks, k=5)

    test_queries = [
        "What is Apple's EBITDA?",
        "Azure OpenAI Service revenue contribution",
        "Apple dividend per share 2025",
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")

        print("\n  [Vector only]")
        for doc in vector_retriever.invoke(query):
            print(f"    {doc.metadata['ticker']} | {doc.page_content[:100]}...")

        print("\n  [Hybrid: vector + BM25 + RRF]")
        for doc in hybrid_retriever.invoke(query):
            print(f"    {doc.metadata['ticker']} | {doc.page_content[:100]}...")

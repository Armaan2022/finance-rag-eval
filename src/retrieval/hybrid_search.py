from typing import Any
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import model_validator
from rank_bm25 import BM25Okapi


# BM25 index
def build_bm25_index(chunks: list[Document]) -> tuple[BM25Okapi, list[Document]]:
    """Tokenize chunks (lowercase + whitespace split) and build a BM25 index."""

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
    """Score all chunks with BM25 and return the top-k as (doc, rank) pairs."""
    
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    indexed_scores = list(enumerate(scores))
    indexed_scores.sort(key=lambda x: x[1], reverse=True)  # sort by score desc
    top_indices = [i for i, score in indexed_scores[:k]]

    return [(chunks[i], rank + 1) for rank, i in enumerate(top_indices)]


# Reciprocal Rank Fusion
def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[Document, int]]],
    k: int = 60,
    top_n: int = 5,
) -> list[Document]:
    """
    Merge ranked lists from multiple retrieval methods into one.
    Score per doc = sum of 1/(rank + k) across all lists it appears in.
    """

    scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    for ranked_list in ranked_lists:
        for doc, rank in ranked_list:
            key = doc.page_content
            scores[key] = scores.get(key, 0.0) + 1.0 / (rank + k)
            doc_map[key] = doc

    # Sort by descending RRF score and return top_n Document objects
    sorted_keys = sorted(scores, key=lambda x: scores[x], reverse=True)
    return [doc_map[key] for key in sorted_keys[:top_n]]


# Hybrid retriever
class HybridRetriever(BaseRetriever):
    """
    Combines Qdrant vector search with BM25 keyword search via RRF.
    """

    model_config = {"arbitrary_types_allowed": True}

    vector_store: Any
    chunks: list[Document]
    k: int = 5
    fetch_k: int = 20
    metadata_filter: dict | None = None
    bm25: Any = None  # populated by model_validator after init

    @model_validator(mode="after")
    def _build_bm25(self):
        self.bm25, _ = build_bm25_index(self.chunks)
        return self

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None) -> list[Document]:
        return self._hybrid_search(query)

    def _hybrid_search(self, query: str) -> list[Document]:
        """
        Run hybrid retrieval: vector + BM25 → RRF fusion.
        """
        # Vector search
        from src.retrieval.qdrant_store import get_retriever

        vector_retriever = get_retriever(
            self.vector_store,
            k=self.fetch_k,
            metadata_filter=self.metadata_filter,
        )
        vector_docs = vector_retriever.invoke(query)
        vector_ranked = [(doc, rank + 1) for rank, doc in enumerate(vector_docs)]

        # BM25 search
        # Apply metadata filter manually for BM25. 
        searchable_chunks = []

        if self.metadata_filter is not None:
            for chunk in self.chunks:
                match = True

                for key, value in self.metadata_filter.items():
                    chunk_val = chunk.metadata.get(key)
                    if isinstance(value, list):
                        if chunk_val not in value:
                            match = False
                            break
                    elif chunk_val != value:
                        match = False
                        break

                if match:
                    searchable_chunks.append(chunk)
        else:
            searchable_chunks = self.chunks

        bm25_index, filtered_chunks = build_bm25_index(searchable_chunks)
        bm25_ranked = bm25_search(query, bm25_index, filtered_chunks, k=self.fetch_k)

        # RRF fusion
        return reciprocal_rank_fusion(
            [vector_ranked, bm25_ranked],
            top_n=self.k,
        )


# Quick comparison: vector-only vs hybrid
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
    hybrid_retriever = HybridRetriever(vector_store=vector_store, chunks=chunks, k=5)

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

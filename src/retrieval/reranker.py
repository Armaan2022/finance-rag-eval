from langchain_core.documents import Document
from sentence_transformers import CrossEncoder


# Model loading
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# The model is ~85MB and takes ~2 seconds to load. Loading it per-query would make every question take 2+ extra seconds.
reranker: CrossEncoder | None = None


def get_reranker() -> CrossEncoder:
    """Load the cross-encoder model (cached after first call)."""
    global reranker
    if reranker is None:
        print(f"  Loading reranker model: {RERANKER_MODEL}")
        reranker = CrossEncoder(RERANKER_MODEL)
    return reranker


# Reranking function
def rerank(
    query: str,
    docs: list[Document],
    top_n: int = 3,
) -> list[Document]:
    """
    Rerank a list of candidate documents using a cross-encoder.
    Returns the top_n most relevant Documents according to the cross-encoder, in descending order of relevance.
    """

    reranker = get_reranker()

    # Build input pairs: [(query, chunk1_text), (query, chunk2_text), ...]
    pairs = [(query, doc.page_content) for doc in docs]

    # scores is a numpy array of floats, one per pair
    scores = reranker.predict(pairs)

    # Zip scores with docs, sort descending, return top_n docs
    scored_docs = list(zip(scores, docs))
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored_docs[:top_n]]



# Full retrieval pipeline: hybrid → rerank
def retrieve_and_rerank(
    query: str,
    hybrid_retriever,
    top_n: int = 3,
) -> list[Document]:
    """
    Two-stage retrieval. Hybrid search for candidates, cross-encoder for precision.
    """

    # Stage 1: get broad candidate set from hybrid search
    candidates = hybrid_retriever.invoke(query)

    # Stage 2: rerank candidates with cross-encoder
    return rerank(query, candidates, top_n=top_n)



# Quick comparison: hybrid vs hybrid + rerank
if __name__ == "__main__":
    from dotenv import load_dotenv
    from src.ingestion.sec_edgar import ingest_tickers
    from src.processing.chunker import chunk_documents
    from src.retrieval.vector_store import load_vector_store, build_vector_store
    from src.retrieval.hybrid_search import HybridRetriever
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

    # fetch_k=20 gives the reranker a meaningful dataset to work with
    hybrid_retriever = HybridRetriever(vector_store=vector_store, chunks=chunks, k=20, fetch_k=20)

    test_queries = [
        "What was Apple's total revenue in fiscal year 2025?",
        "What are the main risks Microsoft faces from AI competition?",
        "How much did Apple return to shareholders through buybacks?",
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")

        candidates = hybrid_retriever.invoke(query)

        print(f"\n  [Hybrid top 5]")
        for doc in candidates[:5]:
            print(f"    {doc.metadata['ticker']} | {doc.page_content[:100]}...")

        reranked = rerank(query, candidates, top_n=3)
        print(f"\n  [After reranking → top 3]")
        for doc in reranked:
            print(f"    {doc.metadata['ticker']} | {doc.page_content[:100]}...")

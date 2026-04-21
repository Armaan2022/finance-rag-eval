from pathlib import Path

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# ChromaDB stores its data in this folder. It persists between runs so you
# don't re-embed everything every time the script runs.
CHROMA_DIR = Path(__file__).resolve().parents[2] / "chroma_db"

# text-embedding-3-small: OpenAI's efficient embedding model.
# 1,536 dimensions, cheap (~$0.00002 per 1k tokens), fast.
# text-embedding-3-large has 3,072 dims and is more accurate but costs more.
# For a portfolio project, small is the right tradeoff.
EMBEDDING_MODEL = "text-embedding-3-small"

# Each filing type gets its own ChromaDB collection so we can filter by it
# later (Week 2 metadata filtering). Think of collections like tables in SQL.
COLLECTION_NAME = "sec_10k_filings"


# ---------------------------------------------------------------------------
# Build or load the vector store
# ---------------------------------------------------------------------------

def get_embeddings() -> OpenAIEmbeddings:
    """
    Return the OpenAI embeddings model.
    Reads OPENAI_API_KEY from the environment automatically.
    """
    return OpenAIEmbeddings(model=EMBEDDING_MODEL)


def build_vector_store(chunks: list[Document], reset: bool = False) -> Chroma:
    """
    Embed a list of Document chunks and store them in ChromaDB.

    Args:
        chunks:  Output from chunk_documents() — LangChain Document objects.
        reset:   If True, delete the existing collection and rebuild from
                 scratch. Use this when you change chunk configs or add new
                 filings. If False, skip documents already in the store.

    Returns:
        A LangChain Chroma object you can use as a retriever.

    Why persist to disk?
        Embedding 1,300 chunks costs ~$0.001 and takes ~5 seconds. Without
        persistence, you'd pay that cost on every restart. ChromaDB writes
        to CHROMA_DIR so subsequent runs load vectors from disk instantly.
    """
    CHROMA_DIR.mkdir(exist_ok=True)
    embeddings = get_embeddings()

    if reset and CHROMA_DIR.exists():
        import shutil
        shutil.rmtree(CHROMA_DIR)
        CHROMA_DIR.mkdir()
        print("  Reset: cleared existing vector store.")

    print(f"  Embedding {len(chunks)} chunks with {EMBEDDING_MODEL}...")

    # Chroma.from_documents() embeds all chunks and writes them to disk in one call.
    # Under the hood: chunks → OpenAI API → 1536-dim vectors → ChromaDB storage.
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
    )

    print(f"  Stored {len(chunks)} chunks in ChromaDB at {CHROMA_DIR}")
    return vector_store


def load_vector_store() -> Chroma:
    """
    Load an existing ChromaDB collection from disk without re-embedding.
    Call this after the first build_vector_store() run.
    """
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(CHROMA_DIR),
    )


# ---------------------------------------------------------------------------
# Build a retriever
# ---------------------------------------------------------------------------

def get_retriever(vector_store: Chroma, k: int = 5, metadata_filter: dict | None = None):
    """
    Wrap the vector store as a retriever.

    Args:
        vector_store: A Chroma instance (from build or load).
        k: Number of chunks to return per query. 5 is a good default —
           enough context for the LLM, not so many that the prompt bloats.
        metadata_filter: Optional ChromaDB filter dict, e.g. {"ticker": "AAPL"}
           to restrict search to a specific company or filing year.

    Returns:
        A LangChain retriever. Call retriever.invoke("your question") to
        get back a list of the k most relevant Document objects.

    search_type="similarity" uses cosine similarity (the default).
    In Week 2 we'll switch to "mmr" (Maximal Marginal Relevance) which
    balances relevance with diversity to avoid returning 5 near-identical chunks.
    """
    search_kwargs: dict = {"k": k}
    if metadata_filter is not None:
        search_kwargs["filter"] = metadata_filter

    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs=search_kwargs,
    )


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    from src.ingestion.sec_edgar import ingest_tickers
    from src.processing.chunker import chunk_documents

    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("Set OPENAI_API_KEY in your .env file before running this.")

    print("=== Step 1: Ingest ===")
    documents = ingest_tickers(["AAPL", "MSFT"], max_filings_per_ticker=1)

    print("\n=== Step 2: Chunk (medium config) ===")
    chunks = chunk_documents(documents, config_name="medium")
    print(f"  {len(chunks)} chunks ready")

    print("\n=== Step 3: Embed + store ===")
    vector_store = build_vector_store(chunks, reset=True)

    print("\n=== Step 4: Test retrieval ===")
    retriever = get_retriever(vector_store, k=3)

    test_queries = [
        "What is Apple's annual revenue?",
        "What are Microsoft's main business segments?",
        "What risk factors does Apple mention?",
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")
        results = retriever.invoke(query)
        for i, doc in enumerate(results):
            ticker = doc.metadata.get("ticker", "?")
            date = doc.metadata.get("filing_date", "?")
            preview = doc.page_content[:120].replace("\n", " ")
            print(f"  [{i+1}] {ticker} ({date}): {preview}...")

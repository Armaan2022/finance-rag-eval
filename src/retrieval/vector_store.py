from pathlib import Path

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings


# Configuration
CHROMA_DIR = Path(__file__).resolve().parents[2] / "chroma_db"

# text-embedding-3-small: OpenAI's efficient embedding model, 1,536 dimensions, cheap, fast.
EMBEDDING_MODEL = "text-embedding-3-small"

COLLECTION_NAME = "sec_10k_filings"



def get_embeddings() -> OpenAIEmbeddings:
    """
    Return the OpenAI embeddings model.
    Reads OPENAI_API_KEY from the environment automatically.
    """
    return OpenAIEmbeddings(model=EMBEDDING_MODEL)


def build_vector_store(chunks: list[Document], reset: bool = False) -> Chroma:
    """
    Embed a list of Document chunks and store them in ChromaDB.
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
    """

    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(CHROMA_DIR),
    )



def get_retriever(vector_store: Chroma, k: int = 5, metadata_filter: dict | None = None):
    """
    Return a LangChain retriever. We call retriever.invoke("your question") to get back a list of the k most relevant Document objects.
    """

    search_kwargs: dict = {"k": k}
    if metadata_filter is not None:
        search_kwargs["filter"] = metadata_filter

    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs=search_kwargs,
    )


# Small test

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

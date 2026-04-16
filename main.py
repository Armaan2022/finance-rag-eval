"""
Finance RAG — main entry point
Run this file to execute the full Week 1 pipeline end-to-end.
"""

from pathlib import Path
from dotenv import load_dotenv

from src.ingestion.sec_edgar import ingest_tickers
from src.processing.chunker import chunk_documents
from src.retrieval.vector_store import build_vector_store, load_vector_store, get_retriever
from src.rag.pipeline import build_rag_chain, ask

load_dotenv()

# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------
TICKERS = ["AAPL", "MSFT", "GOOGL"]
MAX_FILINGS = 2
CHUNK_CONFIG = "medium"
CHROMA_DIR = Path("chroma_db")

# -------------------------------------------------------------------
# Pipeline
# -------------------------------------------------------------------

def build_pipeline(reset_store: bool = False):
    """Build or reload the full RAG pipeline. Returns (chain, retriever)."""

    if not reset_store and CHROMA_DIR.exists() and any(CHROMA_DIR.iterdir()):
        print("Loading existing vector store from disk...")
        vector_store = load_vector_store()
    else:
        print("=== Step 1: Ingest ===")
        documents = ingest_tickers(TICKERS, max_filings_per_ticker=MAX_FILINGS)

        print("\n=== Step 2: Chunk ===")
        chunks = chunk_documents(documents, config_name=CHUNK_CONFIG)
        print(f"  {len(chunks)} chunks ready")

        print("\n=== Step 3: Embed + store ===")
        vector_store = build_vector_store(chunks, reset=reset_store)

    retriever = get_retriever(vector_store, k=5)
    chain = build_rag_chain(retriever)
    return chain, retriever


if __name__ == "__main__":
    chain, retriever = build_pipeline()

    print("\n=== Finance RAG — ask questions about AAPL, MSFT, GOOGL 10-Ks ===")
    print("Type 'quit' to exit.\n")

    while True:
        question = input("Q: ").strip()
        if question.lower() in ("quit", "exit", "q"):
            break
        if not question:
            continue

        result = ask(chain, retriever, question)
        print(f"\nA: {result['answer']}")
        print("\nSources:")
        for s in result["sources"][:3]:
            print(f"  - {s['ticker']} ({s['filing_date']}): {s['preview']}...")
        print()

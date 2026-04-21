"""
One-time script: embed all filings and upload to Qdrant Cloud.

Run once locally:
    python -m scripts.upload_to_qdrant

After this, the app reads from Qdrant Cloud on every startup — no local
chroma_db/ folder needed, and Streamlit Community Cloud can connect too.
"""

from dotenv import load_dotenv
load_dotenv()

from src.ingestion.sec_edgar import ingest_tickers
from src.processing.chunker import chunk_documents
from src.retrieval.qdrant_store import build_qdrant_store

TICKERS = ["AAPL", "MSFT", "GOOGL"]

print("=== Ingesting filings ===")
documents = ingest_tickers(TICKERS, max_filings_per_ticker=2)
print(f"  {len(documents)} filings downloaded")

print("\n=== Chunking ===")
chunks = chunk_documents(documents, config_name="medium")
print(f"  {len(chunks)} chunks ready")

print("\n=== Uploading to Qdrant Cloud ===")
build_qdrant_store(chunks)

print("\nDone. Vectors are now stored in Qdrant Cloud.")

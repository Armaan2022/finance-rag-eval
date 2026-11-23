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
from src.retrieval.qdrant_store import _get_client, COLLECTION_NAME, build_qdrant_store, get_tickers_from_qdrant

TICKERS = ["AAPL", "MSFT", "GOOGL"]

# Skip tickers already in the collection: re-uploading them would add duplicate chunks.
existing_collections = [c.name for c in _get_client().get_collections().collections]
already_loaded = set(get_tickers_from_qdrant()) if COLLECTION_NAME in existing_collections else set()
to_upload = [t for t in TICKERS if t not in already_loaded]
if already_loaded & set(TICKERS):
    print(f"Skipping already-loaded tickers: {sorted(already_loaded & set(TICKERS))}")
if not to_upload:
    raise SystemExit("Nothing to upload.")

print("=== Ingesting filings ===")
documents = ingest_tickers(to_upload, max_filings_per_ticker=2)
print(f"  {len(documents)} filings downloaded")

print("\n=== Chunking ===")
chunks = chunk_documents(documents)  # always DEFAULT_CHUNK_CONFIG
print(f"  {len(chunks)} chunks ready")

print("\n=== Uploading to Qdrant Cloud ===")
build_qdrant_store(chunks)

print("\nDone. Vectors are now stored in Qdrant Cloud.")

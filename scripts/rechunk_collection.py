"""
Re-chunk every filing already in Qdrant with the current chunker settings, then replace its vectors.

Run this whenever chunking changes (size, separators, boilerplate filter) so the whole
collection stays consistent:
    python -m scripts.rechunk_collection           # dry run: shows what would change
    python -m scripts.rechunk_collection --apply   # deletes + re-uploads, one ticker at a time

It re-reads the exact filings that are stored (from their local copies in data/) instead of
downloading "the latest N" again, so the corpus doesn't silently change when a company has
filed a newer 10-K since it was loaded.
"""

import sys
from collections import Counter, defaultdict

from dotenv import load_dotenv
load_dotenv()

from src.ingestion.sec_edgar import DATA_DIR, read_filing_text
from src.processing.chunker import DEFAULT_CHUNK_CONFIG, chunk_documents
from src.retrieval.qdrant_store import delete_ticker_vectors, get_chunks_from_qdrant, upload_ticker_chunks

apply = "--apply" in sys.argv

print("=== Reading current collection ===")
stored = get_chunks_from_qdrant()
old_counts = Counter(c.metadata["ticker"] for c in stored)

# One entry per stored filing, keeping its original metadata.
filings: dict[tuple, dict] = {}
for c in stored:
    m = c.metadata
    key = (m["ticker"], m["filing_date"])
    filings.setdefault(key, {
        "ticker": m["ticker"],
        "company_name": m.get("company_name", ""),
        "filing_date": m["filing_date"],
        "form_type": m.get("form_type", "10-K"),
        "source": m["source"],
    })

# Chunk everything BEFORE deleting anything, so a missing file can't leave a ticker half-deleted.
missing = [f["source"] for f in filings.values() if not (DATA_DIR / f["source"]).exists()]
if missing:
    raise SystemExit(f"Missing local filings, nothing changed: {missing}")

new_chunks: dict[str, list] = defaultdict(list)
for f in sorted(filings.values(), key=lambda f: (f["ticker"], f["filing_date"])):
    text = read_filing_text(DATA_DIR / f["source"])
    new_chunks[f["ticker"]].extend(chunk_documents([{"text": text, "metadata": f}]))

print(f"\nChunk config: {DEFAULT_CHUNK_CONFIG}")
print(f"{'Ticker':<8}{'Filings':>8}{'Old':>8}{'New':>8}")
for ticker in sorted(new_chunks):
    n_filings = sum(1 for t, _ in filings if t == ticker)
    print(f"{ticker:<8}{n_filings:>8}{old_counts[ticker]:>8}{len(new_chunks[ticker]):>8}")

if not apply:
    raise SystemExit("\nDry run only. Re-run with --apply to replace the vectors.")

print("\n=== Replacing vectors ===")
for ticker in sorted(new_chunks):
    print(f"\n{ticker}")
    delete_ticker_vectors(ticker)
    upload_ticker_chunks(new_chunks[ticker])

print("\nDone.")

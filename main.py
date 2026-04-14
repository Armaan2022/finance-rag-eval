"""
Finance RAG — main entry point
Run this file to execute the full pipeline step by step.
Each week adds more functionality here.
"""

from src.ingestion.sec_edgar import ingest_tickers

# -------------------------------------------------------------------
# WEEK 1 — Data Ingestion
# Companies to analyze. Start small; add more once the pipeline works.
# -------------------------------------------------------------------
TICKERS = ["AAPL", "MSFT", "GOOGL"]
MAX_FILINGS = 2  # 2 years of 10-Ks per company = 6 documents total

if __name__ == "__main__":
    print("=== Step 1: Ingesting SEC 10-K filings ===")
    documents = ingest_tickers(TICKERS, max_filings_per_ticker=MAX_FILINGS)

    # documents is now a list of dicts, each with:
    #   ticker, company_name, filing_date, local_path, text, metadata
    # Next step: pass these into the chunking stage (src/processing/chunker.py)
    print(f"\nReady to process {len(documents)} documents.")

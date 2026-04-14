"""
SEC EDGAR Ingestion Module
--------------------------
SEC EDGAR (Electronic Data Gathering, Analysis, and Retrieval) is the US SEC's
public filing database. Every public company must submit filings here.

Key concepts:
- 10-K: Annual report — the most comprehensive document a company files.
  Contains business description, risk factors, financials, MD&A (Management Discussion & Analysis). This is gold for finance RAG.
- CIK: Central Index Key — SEC's unique ID for each company (like a user ID).
- Accession Number: Unique ID for each filing submission.

API used (no key required, completely free):
  https://data.sec.gov/  — SEC's official JSON API
  https://www.sec.gov/   — document downloads

Rate limit: SEC asks for max 10 requests/second and requires a User-Agent header identifying your application. We respect this with a small sleep between requests.
"""

import json
import re
import time
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# SEC requires this header — without it, you'll get 403 errors.
# It's used to identify your app in case of abuse, not for auth.
HEADERS = {
    "User-Agent": "finance-rag-project armaa@example.com",
    "Accept-Encoding": "gzip, deflate",
}

# Base URLs for the two main SEC endpoints
DATA_URL = "https://data.sec.gov"
EDGAR_URL = "https://www.sec.gov"

# Where downloaded filings are stored locally
DATA_DIR = Path(__file__).resolve().parents[2] / "data"


# ---------------------------------------------------------------------------
# Step 1: Ticker → CIK lookup
# ---------------------------------------------------------------------------

def get_cik_from_ticker(ticker: str) -> str:
    """
    Convert a stock ticker (e.g. 'AAPL') to its SEC CIK number.

    How it works:
    - SEC publishes a JSON file mapping all tickers to CIKs.
    - We download it once, find our ticker, and return the zero-padded CIK.
    - CIK must be zero-padded to 10 digits for the submissions API.
    """
    url = f"{EDGAR_URL}/files/company_tickers.json"
    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()

    tickers_data = response.json()
    ticker_upper = ticker.upper()

    for entry in tickers_data.values():
        if entry["ticker"] == ticker_upper:
            # Zero-pad to 10 digits: CIK 320193 → "0000320193"
            return str(entry["cik_str"]).zfill(10)

    raise ValueError(f"Ticker '{ticker}' not found in SEC EDGAR.")


# ---------------------------------------------------------------------------
# Step 2: Fetch the list of 10-K filings for a company
# ---------------------------------------------------------------------------

def get_10k_filings(cik: str, max_filings: int = 3) -> list[dict]:
    """
    Fetch metadata for the most recent 10-K filings for a given CIK.

    The submissions API returns a company's full filing history as JSON.
    We filter for form type '10-K' and return the most recent ones.

    Each returned dict has:
      - accession_number: unique filing ID (formatted as XXX-YY-ZZZZZZ)
      - filing_date: when it was filed
      - primary_document: the filename of the main document

    Why limit to 3 filings? For a portfolio project, 3 years of 10-Ks per
    company gives enough data without hitting rate limits or storage issues.
    """
    url = f"{DATA_URL}/submissions/CIK{cik}.json"
    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()

    data = response.json()
    company_name = data.get("name", "Unknown")
    print(f"  Company: {company_name} (CIK: {cik})")

    # The filings are stored under data["filings"]["recent"] as parallel arrays.
    # Each index i corresponds to one filing across all arrays.
    recent = data["filings"]["recent"]
    forms = recent["form"]
    accession_numbers = recent["accessionNumber"]
    filing_dates = recent["filingDate"]
    primary_documents = recent["primaryDocument"]

    filings = []
    for i, form in enumerate(forms):
        if form == "10-K":
            filings.append({
                "accession_number": accession_numbers[i],
                "filing_date": filing_dates[i],
                "primary_document": primary_documents[i],
                "company_name": company_name,
                "cik": cik,
            })
        if len(filings) >= max_filings:
            break

    print(f"  Found {len(filings)} 10-K filing(s)")
    return filings


# ---------------------------------------------------------------------------
# Step 3: Download the actual filing document
# ---------------------------------------------------------------------------

def download_filing(filing: dict, output_dir: Path) -> Path | None:
    """
    Download the primary document of a 10-K filing and save it locally.

    EDGAR stores filings at a predictable URL pattern:
      /Archives/edgar/data/{cik}/{accession_no_dashes}/{filename}

    Returns the local path to the saved file, or None if download failed.
    """
    cik = filing["cik"].lstrip("0")  # URL uses un-padded CIK
    accession_clean = filing["accession_number"].replace("-", "")
    filename = filing["primary_document"]
    company_name = filing["company_name"]
    filing_date = filing["filing_date"]

    url = f"{EDGAR_URL}/Archives/edgar/data/{cik}/{accession_clean}/{filename}"

    # Build a clean local filename: AAPL_10K_2023-10-27.htm
    ticker_name = company_name.replace(" ", "_").replace("/", "_")[:20]
    ext = Path(filename).suffix or ".htm"
    local_name = f"{ticker_name}_10K_{filing_date}{ext}"
    local_path = output_dir / local_name

    if local_path.exists():
        print(f"    Already downloaded: {local_name}")
        return local_path

    try:
        response = requests.get(url, headers=HEADERS, timeout=30)
        response.raise_for_status()
        local_path.write_bytes(response.content)
        print(f"    Saved: {local_name} ({len(response.content) // 1024} KB)")
        return local_path
    except requests.RequestException as e:
        print(f"    Failed to download {url}: {e}")
        return None


# ---------------------------------------------------------------------------
# Step 4: Strip HTML tags (10-Ks are often .htm files)
# ---------------------------------------------------------------------------

def strip_html(text: str) -> str:
    """
    Remove HTML tags and normalize whitespace from a filing document.

    Why not use BeautifulSoup? It's a heavy dependency for this one task.
    A regex is enough for our purposes — we just want the raw text for
    embedding. We're not trying to render the page.
    """
    # Remove <script> and <style> blocks entirely
    text = re.sub(r"<(script|style)[^>]*>.*?</(script|style)>", "", text, flags=re.DOTALL | re.IGNORECASE)
    # Remove all remaining HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Decode common HTML entities (e.g. &#8217; → ', &amp; → &)
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    text = text.replace("&nbsp;", " ").replace("&quot;", '"').replace("&#8217;", "'")
    text = text.replace("&#8220;", '"').replace("&#8221;", '"').replace("&#8212;", "—")
    # Remove any remaining numeric HTML entities like &#9744;
    text = re.sub(r"&#\d+;", " ", text)
    text = re.sub(r"&[a-zA-Z]+;", " ", text)
    # Collapse whitespace and blank lines
    text = re.sub(r"\s+", " ", text).strip()
    return text


def remove_xbrl_noise(text: str) -> str:
    """
    Remove XBRL inline data fragments that survive HTML stripping.

    10-K filings embed machine-readable XBRL tags as text like:
      "us-gaap:FairValueInputsLevel1Member 2025-09-27 0000320193..."
    These are useless for RAG — they're structured accounting codes, not
    natural language. A retriever trained on this noise would match XBRL
    identifiers instead of meaningful financial sentences.

    We detect these by looking for the characteristic "namespace:CamelCase"
    pattern (e.g. us-gaap:RevenueFromContractWithCustomer) and remove any
    sentence-like run that contains more than 3 such tokens.
    """
    # Pattern: word characters, a colon, then a CamelCase identifier
    xbrl_token = re.compile(r'\b[a-z][\w-]*:[A-Z][A-Za-z]+\b')

    cleaned_lines = []
    for line in text.split(". "):
        matches = xbrl_token.findall(line)
        # If more than 3 XBRL tokens in a sentence-fragment, it's machine data
        if len(matches) <= 3:
            cleaned_lines.append(line)

    return ". ".join(cleaned_lines)


def read_filing_text(filepath: Path) -> str:
    """
    Read a downloaded filing and return clean plain text.
    Handles both .htm/.html and .txt formats.
    """
    raw = filepath.read_bytes().decode("utf-8", errors="replace")

    if filepath.suffix.lower() in (".htm", ".html"):
        text = strip_html(raw)
        return remove_xbrl_noise(text)
    return raw


# ---------------------------------------------------------------------------
# Main entry point: download filings for a list of tickers
# ---------------------------------------------------------------------------

def ingest_tickers(tickers: list[str], max_filings_per_ticker: int = 3) -> list[dict]:
    """
    Full ingestion pipeline for a list of tickers.

    Returns a list of dicts with:
      - ticker
      - filing_date
      - company_name
      - local_path (Path object)
      - text (clean extracted text)

    This is what the chunking stage (src/processing/) will consume next.
    """
    DATA_DIR.mkdir(exist_ok=True)
    results = []

    for ticker in tickers:
        print(f"\nProcessing {ticker}...")
        try:
            cik = get_cik_from_ticker(ticker)
            time.sleep(0.15)  # be polite to SEC servers

            filings = get_10k_filings(cik, max_filings=max_filings_per_ticker)
            time.sleep(0.15)

            for filing in filings:
                local_path = download_filing(filing, DATA_DIR)
                time.sleep(0.15)  # rate limit: ~6-7 req/sec, well under the 10/sec limit

                if local_path is None:
                    continue

                text = read_filing_text(local_path)

                results.append({
                    "ticker": ticker.upper(),
                    "company_name": filing["company_name"],
                    "filing_date": filing["filing_date"],
                    "local_path": local_path,
                    "text": text,
                    # Metadata we'll use later for ChromaDB filtering (Week 1/2)
                    "metadata": {
                        "ticker": ticker.upper(),
                        "company_name": filing["company_name"],
                        "filing_date": filing["filing_date"],
                        "form_type": "10-K",
                        "source": str(local_path.name),
                    },
                })

        except ValueError as e:
            print(f"  Skipping {ticker}: {e}")
        except requests.RequestException as e:
            print(f"  Network error for {ticker}: {e}")

    print(f"\nIngestion complete. {len(results)} filing(s) ready for processing.")
    return results


# ---------------------------------------------------------------------------
# Quick test — run this file directly to verify it works
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Start small: 2 companies, 2 filings each = 4 documents
    # We'll expand this later once the full pipeline is working.
    TEST_TICKERS = ["AAPL", "MSFT"]

    documents = ingest_tickers(TEST_TICKERS, max_filings_per_ticker=2)

    print("\n--- Summary ---")
    for doc in documents:
        word_count = len(doc["text"].split())
        print(f"  {doc['ticker']} | {doc['filing_date']} | {word_count:,} words")

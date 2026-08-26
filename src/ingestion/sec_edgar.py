"""
Download and parse 10-K filings from SEC EDGAR.
No API key needed since everything is public. Rate limit is 10 req/sec so we add small sleeps.
"""

import json
import re
import time
from pathlib import Path

import requests

# SEC requires a User-Agent, without it we get 403s.
HEADERS = {
    "User-Agent": "finance-rag-project armaa@example.com",
    "Accept-Encoding": "gzip, deflate",
}

DATA_URL = "https://data.sec.gov"
EDGAR_URL = "https://www.sec.gov"
DATA_DIR = Path(__file__).resolve().parents[2] / "data"


# Step 1: Ticker → CIK lookup

def get_cik_from_ticker(ticker: str) -> str:
    """Look up a ticker in the SEC's public ticker→CIK map. Returns a 10-digit zero-padded CIK."""

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


# Step 2: Fetch the list of 10-K filings for a company

def get_10k_filings(cik: str, max_filings: int = 3) -> list[dict]:
    """Fetch the most recent 10-K filing metadata for a CIK from the EDGAR submissions API."""

    url = f"{DATA_URL}/submissions/CIK{cik}.json"
    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()

    data = response.json()
    company_name = data.get("name", "Unknown")
    print(f"  Company: {company_name} (CIK: {cik})")

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


# Step 3: Download the actual filing document

def download_filing(filing: dict, output_dir: Path) -> Path | None:
    """Download a filing's primary document from EDGAR and save it locally. Returns None on failure."""

    cik = filing["cik"].lstrip("0")  # URL uses un-padded CIK
    accession_clean = filing["accession_number"].replace("-", "")
    filename = filing["primary_document"]
    company_name = filing["company_name"]
    filing_date = filing["filing_date"]

    url = f"{EDGAR_URL}/Archives/edgar/data/{cik}/{accession_clean}/{filename}"

    # Builds a local filename: AAPL_10K_2023-10-27.htm
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


# Step 4: Strip HTML tags (10-Ks are often .htm files)

def strip_html(text: str) -> str:
    """Strip HTML tags and normalize whitespace since we need plain text for embedding."""

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
    Drop lines with too many XBRL tokens like "us-gaap:RevenueFromContractWithCustomer".
    These were not removed by HTML stripping and since they are machine data, they'd just corrupt retrieval.
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


# Main function: download filings for a list of tickers

def ingest_tickers(tickers: list[str], max_filings_per_ticker: int = 3) -> list[dict]:
    """Download, parse, and return clean text dicts for all 10-K filings for the given tickers."""

    DATA_DIR.mkdir(exist_ok=True)
    results = []

    for ticker in tickers:
        print(f"\nProcessing {ticker}...")
        try:
            cik = get_cik_from_ticker(ticker)
            time.sleep(0.15)  # so it doesn't exceed the rate limit

            filings = get_10k_filings(cik, max_filings=max_filings_per_ticker)
            time.sleep(0.15)

            for filing in filings:
                local_path = download_filing(filing, DATA_DIR)
                time.sleep(0.15)  

                if local_path is None:
                    continue

                text = read_filing_text(local_path)

                results.append({
                    "ticker": ticker.upper(),
                    "company_name": filing["company_name"],
                    "filing_date": filing["filing_date"],
                    "local_path": local_path,
                    "text": text,
                    # Metadata attached to every chunk for Qdrant filtering
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


# Small test

if __name__ == "__main__":

    # 2 companies, 2 filings each = 4 documents
    TEST_TICKERS = ["AAPL", "MSFT"]

    documents = ingest_tickers(TEST_TICKERS, max_filings_per_ticker=2)

    print("\n--- Summary ---")
    for doc in documents:
        word_count = len(doc["text"].split())
        print(f"  {doc['ticker']} | {doc['filing_date']} | {word_count:,} words")

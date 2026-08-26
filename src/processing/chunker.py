"""
Split raw filing text into chunks for embedding.
RecursiveCharacterTextSplitter does paragraph breaks first, then newlines, then words. Overlap (~12% of chunk_size) is used to prevent sentences from being lost when they fall exactly on a boundary.
"""

import re

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Matches 10-K section headers like "ITEM 1A. RISK FACTORS" or "Item 7. Management's Discussion"

_ITEM_RE = re.compile(
    r"(?m)^[\s]*(?:ITEM|Item)\s+(\d+[A-Z]?)\s*[.:]?\s+([A-Z][^\n]{2,80})",
)

_ITEM_NAMES = {
    "1": "Business", "1A": "Risk Factors", "1B": "Unresolved Staff Comments",
    "2": "Properties", "3": "Legal Proceedings", "4": "Mine Safety Disclosures",
    "5": "Market for Common Equity", "6": "Selected Financial Data",
    "7": "Management's Discussion and Analysis", "7A": "Quantitative and Qualitative Disclosures",
    "8": "Financial Statements", "9": "Changes in and Disagreements with Accountants",
    "9A": "Controls and Procedures", "9B": "Other Information",
    "10": "Directors and Executive Officers", "11": "Executive Compensation",
    "12": "Security Ownership", "13": "Certain Relationships",
    "14": "Principal Accountant Fees", "15": "Exhibits",
}


def _find_section_map(text: str) -> list[tuple[int, str]]:
    """Return sorted list of (char_offset, 'Item X. Title') from the filing text."""

    seen: set[str] = set()
    sections: list[tuple[int, str]] = []
    for m in _ITEM_RE.finditer(text):
        key = m.group(1).upper()
        if key in seen:
            continue
        seen.add(key)
        
        title = _ITEM_NAMES.get(key, m.group(2).strip().title())
        sections.append((m.start(), f"Item {key}. {title}"))
    return sections


def _section_for_offset(offset: int, section_map: list[tuple[int, str]]) -> str | None:
    result = None
    for pos, label in section_map:
        if pos <= offset:
            result = label
        else:
            break
    return result


# Three presets (currently we are using large for better faithfulness)
CHUNK_CONFIGS = {
    "small":  {"chunk_size": 256,  "chunk_overlap": 32},
    "medium": {"chunk_size": 512,  "chunk_overlap": 64},
    "large":  {"chunk_size": 1024, "chunk_overlap": 128},
}

DEFAULT_CHUNK_CONFIG = "medium"


def chunk_documents(
    documents: list[dict],
    config_name: str = DEFAULT_CHUNK_CONFIG,
) -> list[Document]:
    """Split filing dicts into LangChain Document chunks and then attaching 10-K section labels."""

    config = CHUNK_CONFIGS[config_name]
    chunk_size = config["chunk_size"]
    chunk_overlap = config["chunk_overlap"]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True,  # adds 'start_index' to each chunk's metadata
    )

    all_chunks: list[Document] = []

    for doc in documents:
        section_map = _find_section_map(doc["text"])
        raw_chunks = splitter.create_documents([doc["text"]])

        for i, chunk_doc in enumerate(raw_chunks):
            start_index = chunk_doc.metadata.get("start_index", 0)
            section = _section_for_offset(start_index, section_map)
            chunk_metadata = {
                **doc["metadata"],
                "chunk_index": i,
                "total_chunks": len(raw_chunks),
                "chunk_config": config_name,
            }
            if section:
                chunk_metadata["section"] = section
            all_chunks.append(Document(
                page_content=chunk_doc.page_content,
                metadata=chunk_metadata,
            ))

    return all_chunks


# Helper: print a summary of chunk statistics

def print_chunk_stats(chunks: list[Document], config_name: str) -> None:
    """Print a quick size summary — useful for comparing chunk configs."""

    lengths = [len(c.page_content) for c in chunks]
    avg = sum(lengths) / len(lengths) if lengths else 0
    tickers = set(c.metadata.get("ticker", "?") for c in chunks)

    print(f"\n--- Chunk stats [{config_name}] ---")
    print(f"  Total chunks : {len(chunks)}")
    print(f"  Avg length   : {avg:.0f} chars (~{avg/4:.0f} tokens)")
    print(f"  Min / Max    : {min(lengths)} / {max(lengths)} chars")
    print(f"  Tickers      : {', '.join(sorted(tickers))}")


# Small test

if __name__ == "__main__":
    from src.ingestion.sec_edgar import ingest_tickers

    print("Ingesting documents...")
    documents = ingest_tickers(["AAPL", "MSFT"], max_filings_per_ticker=1)

    # Run all three chunk sizes so we can compare
    for config_name in CHUNK_CONFIGS:
        chunks = chunk_documents(documents, config_name=config_name)
        print_chunk_stats(chunks, config_name)

    # Show a sample chunk so we can see what the embedder will receive
    sample_chunks = chunk_documents(documents, config_name="medium")
    print("\n--- Sample chunk (medium config) ---")
    sample = sample_chunks[10]
    print(f"  Ticker      : {sample.metadata['ticker']}")
    print(f"  Filing date : {sample.metadata['filing_date']}")
    print(f"  Chunk index : {sample.metadata['chunk_index']} / {sample.metadata['total_chunks']}")
    print(f"  Text preview: {sample.page_content[:300]}...")

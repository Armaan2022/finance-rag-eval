"""
Document Chunking Module
------------------------
Takes the raw text output from sec_edgar.py and splits it into
smaller chunks suitable for embedding.

Key concept — why we chunk:
  Embedding models convert text into a fixed-size vector (a list of numbers
  that represents meaning). If you embed an entire 50,000-word document, the
  vector averages out all the meaning and becomes too vague to retrieve
  specific facts from. Chunking gives each vector a focused topic.

Key concept — RecursiveCharacterTextSplitter:
  LangChain's splitter tries to split on natural boundaries in this order:
    1. Double newline (paragraph break)  → "\n\n"
    2. Single newline                    → "\n"
    3. Space                             → " "
    4. Character                         → ""
  It only falls back to the next separator if the current one produces a
  chunk that's still too large. This preserves sentence/paragraph structure
  much better than a naive fixed-size split.

Key concept — chunk_overlap:
  The last N tokens of chunk i are repeated at the start of chunk i+1.
  This ensures a sentence split across two chunks isn't lost in either.
  Rule of thumb: overlap = 10–20% of chunk_size.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


# ---------------------------------------------------------------------------
# Chunk size configurations to experiment with
# ---------------------------------------------------------------------------
# We define three presets so you can run all three and compare retrieval quality later in Week 3 evaluation.

CHUNK_CONFIGS = {
    "small":  {"chunk_size": 256,  "chunk_overlap": 32},
    "medium": {"chunk_size": 512,  "chunk_overlap": 64},
    "large":  {"chunk_size": 1024, "chunk_overlap": 128},
}

# Default for the main pipeline — we'll benchmark all three in Week 3
DEFAULT_CHUNK_CONFIG = "medium"


# ---------------------------------------------------------------------------
# Core chunking function
# ---------------------------------------------------------------------------

def chunk_documents(
    documents: list[dict],
    config_name: str = DEFAULT_CHUNK_CONFIG,
) -> list[Document]:
    """
    Split a list of ingested documents into LangChain Document chunks.

    Args:
        documents: Output from ingest_tickers() — list of dicts with
                   'text' and 'metadata' keys.
        config_name: One of 'small', 'medium', 'large'. Controls chunk
                     size and overlap.

    Returns:
        A flat list of LangChain Document objects, each with:
          - page_content: the chunk text
          - metadata: inherited from the source document, plus chunk index

    Why LangChain Document objects?
        ChromaDB and LangChain retrievers expect this format. Wrapping our
        text in Document objects early means the rest of the pipeline
        (embedding, storing, retrieving) needs no conversion.
    """
    config = CHUNK_CONFIGS[config_name]
    chunk_size = config["chunk_size"]
    chunk_overlap = config["chunk_overlap"]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # length_function=len measures in characters, not tokens.
        # text-embedding-3-small uses ~4 chars/token on average for English,
        # so chunk_size=512 chars ≈ 128 tokens. This is a simplification —
        # in Week 3 we could swap in a proper tokenizer, but character-based
        # splitting is standard and works well in practice.
        length_function=len,
        # These separators are tried in order. The splitter uses the first
        # one that produces chunks within the size limit.
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    all_chunks: list[Document] = []

    for doc in documents:
        raw_chunks = splitter.split_text(doc["text"])

        for i, chunk_text in enumerate(raw_chunks):
            # Inherit all metadata from the source document and add chunk index.
            # This is critical — when we retrieve a chunk later, we need to know
            # which company/filing it came from.
            chunk_metadata = {
                **doc["metadata"],
                "chunk_index": i,
                "total_chunks": len(raw_chunks),
                "chunk_config": config_name,
            }
            all_chunks.append(Document(
                page_content=chunk_text,
                metadata=chunk_metadata,
            ))

    return all_chunks


# ---------------------------------------------------------------------------
# Helper: print a summary of chunk statistics
# ---------------------------------------------------------------------------

def print_chunk_stats(chunks: list[Document], config_name: str) -> None:
    """
    Print statistics about the chunks produced by a given config.
    Useful for understanding the tradeoff between chunk sizes before
    you run full evaluation in Week 3.
    """
    lengths = [len(c.page_content) for c in chunks]
    avg = sum(lengths) / len(lengths) if lengths else 0
    tickers = set(c.metadata.get("ticker", "?") for c in chunks)

    print(f"\n--- Chunk stats [{config_name}] ---")
    print(f"  Total chunks : {len(chunks)}")
    print(f"  Avg length   : {avg:.0f} chars (~{avg/4:.0f} tokens)")
    print(f"  Min / Max    : {min(lengths)} / {max(lengths)} chars")
    print(f"  Tickers      : {', '.join(sorted(tickers))}")


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from src.ingestion.sec_edgar import ingest_tickers

    print("Ingesting documents...")
    documents = ingest_tickers(["AAPL", "MSFT"], max_filings_per_ticker=1)

    # Run all three chunk sizes so you can compare
    for config_name in CHUNK_CONFIGS:
        chunks = chunk_documents(documents, config_name=config_name)
        print_chunk_stats(chunks, config_name)

    # Show a sample chunk so you can see what the embedder will receive
    sample_chunks = chunk_documents(documents, config_name="medium")
    print("\n--- Sample chunk (medium config) ---")
    sample = sample_chunks[10]
    print(f"  Ticker      : {sample.metadata['ticker']}")
    print(f"  Filing date : {sample.metadata['filing_date']}")
    print(f"  Chunk index : {sample.metadata['chunk_index']} / {sample.metadata['total_chunks']}")
    print(f"  Text preview: {sample.page_content[:300]}...")

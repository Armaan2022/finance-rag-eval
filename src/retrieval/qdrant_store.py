import os
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams


COLLECTION_NAME = "sec_10k_filings"
EMBEDDING_MODEL = "text-embedding-3-small"
VECTOR_SIZE = 1536  # dimensions output by text-embedding-3-small


def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=EMBEDDING_MODEL)


def _get_client() -> QdrantClient:
    """Connect to Qdrant Cloud using env vars."""
    return QdrantClient(
        url=os.environ["QDRANT_URL"],
        api_key=os.environ["QDRANT_API_KEY"],
    )


def build_qdrant_store(chunks: list[Document]) -> QdrantVectorStore:
    """
    Embed chunks and upload them to Qdrant Cloud.

    Creates the collection if it doesn't exist yet, then upserts all chunks.
    Run this once locally — vectors persist in the cloud, so Streamlit Cloud
    never needs to re-embed.
    """
    client = _get_client()

    # Create collection only if it doesn't exist yet
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
        print(f"  Created Qdrant collection: {COLLECTION_NAME}")
    else:
        print(f"  Collection '{COLLECTION_NAME}' already exists — upserting.")

    print(f"  Embedding and uploading {len(chunks)} chunks...")

    vector_store = QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        url=os.environ["QDRANT_URL"],
        api_key=os.environ["QDRANT_API_KEY"],
        collection_name=COLLECTION_NAME,
    )

    print(f"  Uploaded {len(chunks)} chunks to Qdrant Cloud.")
    return vector_store


def _ensure_payload_indexes(client: QdrantClient) -> None:
    """
    Create payload indexes required for filtered search and delete.
    Qdrant needs a keyword index on metadata.ticker before any filter
    on that field can be used in a query. Safe to call repeatedly.
    """
    from qdrant_client.models import PayloadSchemaType
    for field in ("metadata.ticker", "metadata.filing_date", "metadata.form_type"):
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name=field,
            field_schema=PayloadSchemaType.KEYWORD,
        )


def load_qdrant_store() -> QdrantVectorStore:
    """
    Connect to an existing Qdrant collection without re-embedding.
    This is what the app calls on every startup.
    """
    client = _get_client()
    _ensure_payload_indexes(client)
    return QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=get_embeddings(),
    )


def upload_ticker_chunks(chunks: list[Document], batch_size: int = 32) -> None:
    """Add chunks for a new ticker to an existing Qdrant collection."""
    vector_store = load_qdrant_store()
    for i in range(0, len(chunks), batch_size):
        vector_store.add_documents(chunks[i : i + batch_size])
    print(f"  Uploaded {len(chunks)} chunks to Qdrant Cloud.")


def delete_ticker_vectors(ticker: str) -> None:
    """Delete all vectors for a given ticker from Qdrant."""
    client = _get_client()
    ticker_upper = ticker.upper()
    ids_to_delete: list = []
    offset = None

    # Scroll without a filter (no index required), collect matching IDs client-side
    while True:
        results, next_offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=1000,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in results:
            if point.payload.get("metadata", {}).get("ticker") == ticker_upper:
                ids_to_delete.append(point.id)
        if next_offset is None:
            break
        offset = next_offset

    if ids_to_delete:
        client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=ids_to_delete,
        )

    print(f"  Deleted {len(ids_to_delete)} vectors for {ticker_upper} from Qdrant.")


def get_chunks_from_qdrant() -> list[Document]:
    """
    Reconstruct all stored Document chunks from Qdrant payloads.
    LangChain stores page_content and metadata in the payload at upload time,
    so we never need to re-download from SEC EDGAR just to rebuild BM25.
    """
    client = _get_client()
    docs: list[Document] = []
    offset = None

    while True:
        results, next_offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=1000,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in results:
            payload = point.payload
            docs.append(Document(
                page_content=payload.get("page_content", ""),
                metadata=payload.get("metadata", {}),
            ))
        if next_offset is None:
            break
        offset = next_offset

    return docs


def get_tickers_from_qdrant() -> list[str]:
    """
    Query Qdrant for the distinct set of tickers stored in the collection.

    This is the authoritative source of truth — no JSON file needed.
    Scrolls through all points (without vectors) and collects unique ticker values
    from the metadata payload.
    """
    client = _get_client()
    tickers: set[str] = set()
    offset = None

    while True:
        results, next_offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=1000,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in results:
            ticker = point.payload.get("metadata", {}).get("ticker")
            if ticker:
                tickers.add(ticker)
        if next_offset is None:
            break
        offset = next_offset

    return sorted(tickers)


def get_retriever(vector_store: QdrantVectorStore, k: int = 5, metadata_filter: dict | None = None):
    """
    Wrap the Qdrant store as a LangChain retriever.

    metadata_filter: a plain dict like {"ticker": "AAPL"} or {"ticker": ["AAPL", "MSFT"]}.
    List values are treated as OR (MatchAny). This is converted to a Qdrant Filter object
    so Qdrant can apply it server-side.
    """
    search_kwargs: dict = {"k": k}
    if metadata_filter is not None:
        from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny
        must = []
        for key, val in metadata_filter.items():
            if isinstance(val, list):
                must.append(FieldCondition(key=f"metadata.{key}", match=MatchAny(any=val)))
            else:
                must.append(FieldCondition(key=f"metadata.{key}", match=MatchValue(value=val)))
        search_kwargs["filter"] = Filter(must=must)

    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs=search_kwargs,
    )

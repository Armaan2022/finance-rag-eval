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


def load_qdrant_store() -> QdrantVectorStore:
    """
    Connect to an existing Qdrant collection without re-embedding.
    This is what the app calls on every startup.
    """
    return QdrantVectorStore(
        client=_get_client(),
        collection_name=COLLECTION_NAME,
        embedding=get_embeddings(),
    )


def get_retriever(vector_store: QdrantVectorStore, k: int = 5, metadata_filter: dict | None = None):
    """
    Wrap the Qdrant store as a LangChain retriever.

    metadata_filter example: {"ticker": "AAPL"}
    Qdrant translates this into a payload filter automatically via langchain-qdrant.
    """
    search_kwargs: dict = {"k": k}
    if metadata_filter is not None:
        search_kwargs["filter"] = metadata_filter

    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs=search_kwargs,
    )

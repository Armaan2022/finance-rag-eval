import re
from typing import Any

import numpy as np
import snowballstemmer
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from rank_bm25 import BM25Okapi


# Tokenizer
# BM25 only matches exact tokens, so "Revenue," / "revenue" / "revenues" must all become the same token.
# 1. lowercase  2. pull out word/number tokens (drops punctuation, keeps decimals like "391.04")
# 3. drop stopwords ("what", "was", "the"...) that every question contains  4. stem ("revenues" -> "revenu")

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")

_STOPWORDS = frozenset("""
a an and are as at be s been being but by can could did do does for from had has have how i if in into is it
its of on or our over so than that the their them then there these they this those to under was we were
what when where which while who why will with would you your
""".split())

_stemmer = snowballstemmer.stemmer("english")


def tokenize(text: str) -> list[str]:
    tokens = [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]
    return _stemmer.stemWords(tokens)


def _matches_filter(metadata: dict, metadata_filter: dict) -> bool:
    """Same semantics as the Qdrant filter: list value = 'any of', scalar = exact match."""

    for key, value in metadata_filter.items():
        chunk_val = metadata.get(key)
        if isinstance(value, list):
            if chunk_val not in value:
                return False
        elif chunk_val != value:
            return False
    return True


# BM25 index
class BM25Index:
    """
    Built ONCE over the whole corpus, then reused for every query.
    Metadata filters are applied by masking scores, not by rebuilding the index.
    The app rebuilds it only when the corpus changes (a company is added or removed).
    """

    def __init__(self, chunks: list[Document]):
        self.chunks = chunks
        self._bm25 = BM25Okapi([tokenize(doc.page_content) for doc in chunks]) if chunks else None

    def search(
        self,
        query: str,
        k: int = 20,
        metadata_filter: dict | None = None,
    ) -> list[tuple[Document, int]]:
        """Return the top-k chunks as (doc, rank) pairs, restricted to chunks matching the filter."""

        query_tokens = tokenize(query)
        if self._bm25 is None or not query_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)
        if metadata_filter:
            mask = np.array([_matches_filter(c.metadata, metadata_filter) for c in self.chunks])
            scores = np.where(mask, scores, -np.inf)

        top = np.argsort(scores)[::-1][:k]
        # Drop filtered-out chunks and chunks sharing no terms with the query.
        top = [i for i in top if scores[i] > 0]
        return [(self.chunks[i], rank + 1) for rank, i in enumerate(top)]


# Reciprocal Rank Fusion
def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[Document, int]]],
    k: int = 60,
    top_n: int = 5,
) -> list[Document]:
    """
    Merge ranked lists from multiple retrieval methods into one.
    Score per doc = sum of 1/(rank + k) across all lists it appears in.
    """

    scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    for ranked_list in ranked_lists:
        for doc, rank in ranked_list:
            key = doc.page_content
            scores[key] = scores.get(key, 0.0) + 1.0 / (rank + k)
            doc_map[key] = doc

    # Sort by descending RRF score and return top_n Document objects
    sorted_keys = sorted(scores, key=lambda x: scores[x], reverse=True)
    return [doc_map[key] for key in sorted_keys[:top_n]]


# Hybrid retriever
class HybridRetriever(BaseRetriever):
    """
    Combines Qdrant vector search with BM25 keyword search via RRF.
    Takes a prebuilt BM25Index so that creating a retriever (e.g. one per filter combination) is cheap.
    """

    model_config = {"arbitrary_types_allowed": True}

    vector_store: Any
    bm25_index: BM25Index
    k: int = 5
    fetch_k: int = 20
    metadata_filter: dict | None = None

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None) -> list[Document]:
        from src.retrieval.qdrant_store import get_retriever

        # Vector search
        vector_retriever = get_retriever(
            self.vector_store,
            k=self.fetch_k,
            metadata_filter=self.metadata_filter,
        )
        vector_docs = vector_retriever.invoke(query)
        vector_ranked = [(doc, rank + 1) for rank, doc in enumerate(vector_docs)]

        # BM25 search (same filter, applied as a mask over the shared index)
        bm25_ranked = self.bm25_index.search(query, k=self.fetch_k, metadata_filter=self.metadata_filter)

        # RRF fusion
        return reciprocal_rank_fusion(
            [vector_ranked, bm25_ranked],
            top_n=self.k,
        )


# Quick comparison: vector-only vs hybrid
if __name__ == "__main__":
    from dotenv import load_dotenv
    from src.retrieval.qdrant_store import load_qdrant_store, get_retriever, get_chunks_from_qdrant

    load_dotenv()

    print("Loading data...")
    chunks = get_chunks_from_qdrant()
    vector_store = load_qdrant_store()

    vector_retriever = get_retriever(vector_store, k=5)
    hybrid_retriever = HybridRetriever(vector_store=vector_store, bm25_index=BM25Index(chunks), k=5)

    test_queries = [
        "What is Apple's EBITDA?",
        "Azure OpenAI Service revenue contribution",
        "Apple dividend per share 2025",
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}  ->  tokens: {tokenize(query)}")

        print("\n  [Vector only]")
        for doc in vector_retriever.invoke(query):
            print(f"    {doc.metadata['ticker']} | {doc.page_content[:100]}...")

        print("\n  [Hybrid: vector + BM25 + RRF]")
        for doc in hybrid_retriever.invoke(query):
            print(f"    {doc.metadata['ticker']} | {doc.page_content[:100]}...")

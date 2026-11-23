from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from sentence_transformers import CrossEncoder


# Model loading
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# How many chunks the reranker keeps. 5 = same context size as the other pipelines, so the
# eval compares ranking quality only.
RERANK_TOP_N = 5

# MiniLM was trained on web search passages and consistently ranks financial tables (where the
# numbers live) below prose that merely mentions the topic. E.g. for "Apple's total revenue in 2025"
# hybrid ranks the net sales table #1 and the cross-encoder drops it to #7. Raising top_n doesn't
# fix that reliably, so by default the reranker's order is fused (RRF) with the hybrid order:
# the cross-encoder can promote and demote chunks, but can't completely overrule the first stage.
FUSE_WITH_BASE_RANKING = True

# The model is ~85MB and takes ~2 seconds to load. Loading it per-query would make every question take 2+ extra seconds.
reranker: CrossEncoder | None = None


def get_reranker() -> CrossEncoder:
    """Load the cross-encoder model (cached after first call)."""
    global reranker
    if reranker is None:
        print(f"  Loading reranker model: {RERANKER_MODEL}")
        reranker = CrossEncoder(RERANKER_MODEL)
    return reranker


# Reranking function
def rerank(
    query: str,
    docs: list[Document],
    top_n: int = RERANK_TOP_N,
) -> list[Document]:
    """
    Rerank a list of candidate documents using a cross-encoder.
    Returns the top_n most relevant Documents according to the cross-encoder, in descending order of relevance.
    """

    reranker = get_reranker()

    # Build input pairs: [(query, chunk1_text), (query, chunk2_text), ...]
    pairs = [(query, doc.page_content) for doc in docs]

    # scores is a numpy array of floats, one per pair
    scores = reranker.predict(pairs)

    # Zip scores with docs, sort descending, return top_n docs
    scored_docs = list(zip(scores, docs))
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored_docs[:top_n]]



# Full retrieval pipeline: hybrid → rerank
class RerankRetriever(BaseRetriever):
    """
    Two-stage retrieval as a single retriever: the base retriever fetches a broad candidate set
    (e.g. hybrid with k=20), then the cross-encoder keeps the top_n.
    Because it's a normal retriever, the chunks it returns are exactly what the LLM reads,
    what the sources panel shows and what RAGAS scores.
    """

    model_config = {"arbitrary_types_allowed": True}

    base_retriever: Any
    top_n: int = RERANK_TOP_N
    fuse_with_base: bool = FUSE_WITH_BASE_RANKING

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None) -> list[Document]:
        candidates = self.base_retriever.invoke(query)
        if not self.fuse_with_base:
            return rerank(query, candidates, top_n=self.top_n)

        from src.retrieval.hybrid_search import reciprocal_rank_fusion

        reranked = rerank(query, candidates, top_n=len(candidates))
        return reciprocal_rank_fusion(
            [
                [(doc, rank + 1) for rank, doc in enumerate(candidates)],
                [(doc, rank + 1) for rank, doc in enumerate(reranked)],
            ],
            top_n=self.top_n,
        )


# Quick comparison: hybrid vs hybrid + rerank
if __name__ == "__main__":
    from dotenv import load_dotenv
    from src.retrieval.qdrant_store import load_qdrant_store, get_chunks_from_qdrant
    from src.retrieval.hybrid_search import BM25Index, HybridRetriever

    load_dotenv()

    print("Loading data...")
    chunks = get_chunks_from_qdrant()
    vector_store = load_qdrant_store()

    # k=20 gives the reranker a meaningful candidate set to work with
    hybrid_retriever = HybridRetriever(vector_store=vector_store, bm25_index=BM25Index(chunks), k=20, fetch_k=20)

    test_queries = [
        "What was Apple's total revenue in fiscal year 2025?",
        "What are the main risks Microsoft faces from AI competition?",
        "How much did Apple return to shareholders through buybacks?",
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")

        candidates = hybrid_retriever.invoke(query)

        print(f"\n  [Hybrid top 5]")
        for doc in candidates[:5]:
            print(f"    {doc.metadata['ticker']} | {doc.page_content[:100]}...")

        reranked = rerank(query, candidates, top_n=RERANK_TOP_N)
        print(f"\n  [After reranking → top {RERANK_TOP_N}]")
        for doc in reranked:
            print(f"    {doc.metadata['ticker']} | {doc.page_content[:100]}...")

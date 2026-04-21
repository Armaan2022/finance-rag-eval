import os
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()


# Page config
st.set_page_config(
    page_title="Finance RAG",
    page_icon="📊",
    layout="wide",
)


# Pipeline loading (cached so it only runs once per session)
@st.cache_resource(show_spinner="Loading pipeline...")
def load_pipeline():
    """
    Load all pipeline components once and cache them for the session.

    st.cache_resource caches the return value across all users and reruns.
    Without this, every interaction would re-embed documents and reload
    the cross-encoder model — making the app painfully slow.
    """
    from src.ingestion.sec_edgar import ingest_tickers
    from src.processing.chunker import chunk_documents
    from src.retrieval.vector_store import load_vector_store, build_vector_store, get_retriever
    from src.retrieval.hybrid_search import HybridRetriever
    from src.rag.pipeline import build_rag_chain

    CHROMA_DIR = Path("chroma_db")
    TICKERS = ["AAPL", "MSFT", "GOOGL"]

    # Load documents and chunks (needed for BM25 index)
    documents = ingest_tickers(TICKERS, max_filings_per_ticker=2)
    chunks = chunk_documents(documents, config_name="medium")

    # Load or build vector store
    if CHROMA_DIR.exists() and any(CHROMA_DIR.iterdir()):
        vector_store = load_vector_store()
    else:
        vector_store = build_vector_store(chunks)

    # Build all three retrievers
    retrievers = {
        "Vector only": get_retriever(vector_store, k=5),
        "Hybrid (BM25 + Vector + RRF)": HybridRetriever(
            vector_store=vector_store, chunks=chunks, k=5, fetch_k=20
        ),
        "Hybrid + Reranker": HybridRetriever(
            vector_store=vector_store, chunks=chunks, k=20, fetch_k=20
        ),
    }

    return retrievers, chunks, vector_store



# Tab 1 — Q&A Interface
def render_qa_tab(retrievers):
    st.header("Ask questions about SEC 10-K filings")
    st.caption("Covers Apple (AAPL), Microsoft (MSFT), and Google (GOOGL) — 2024 & 2025 annual reports")

    col1, col2 = st.columns([2, 1])

    with col2:
        pipeline_choice = st.selectbox(
            "Pipeline",
            options=list(retrievers.keys()),
            help="Vector only: pure semantic search\nHybrid: adds BM25 keyword search\nHybrid + Reranker: cross-encoder precision pass",
        )

        st.divider()
        st.markdown("**Example questions**")
        examples = [
            "What was Apple's total revenue in 2025?",
            "What are Microsoft's three business segments?",
            "What AI risks does Microsoft identify?",
            "How much did Apple spend on R&D?",
            "What is Microsoft's relationship with OpenAI?",
            "What are Apple's main supply chain risks?",
        ]
        for ex in examples:
            if st.button(ex, use_container_width=True, key=ex):
                st.session_state["query"] = ex

    with col1:
        query = st.text_input(
            "Your question",
            value=st.session_state.get("query", ""),
            placeholder="e.g. What was Apple's net income in fiscal year 2025?",
        )

        if st.button("Ask", type="primary", use_container_width=True) and query:
            retriever = retrievers[pipeline_choice]

            with st.spinner("Retrieving and generating answer..."):
                from src.rag.pipeline import build_rag_chain, ask

                # Use reranker for the third pipeline variant
                use_reranker = (pipeline_choice == "Hybrid + Reranker")
                chain = build_rag_chain(retriever)
                result = ask(chain, retriever, query)

                if use_reranker:
                    from src.retrieval.reranker import retrieve_and_rerank
                    reranked_docs = retrieve_and_rerank(query, retriever, top_n=3)
                    result["sources"] = [
                        {
                            "ticker": d.metadata.get("ticker", "?"),
                            "filing_date": d.metadata.get("filing_date", "?"),
                            "preview": d.page_content[:200].replace("\n", " "),
                        }
                        for d in reranked_docs
                    ]

            st.markdown("### Answer")
            st.markdown(result["answer"])

            st.markdown("### Sources")
            for i, src in enumerate(result["sources"][:5]):
                with st.expander(f"[{i+1}] {src['ticker']} — {src['filing_date']}"):
                    st.caption(src["preview"] + "...")



# Tab 2 — Evaluation Results
def render_eval_tab():
    st.header("Pipeline Evaluation Results")
    st.caption("RAGAS metrics measured on 10 hand-crafted Q&A pairs from AAPL 10-K filings")

    csv_path = Path("results/eval_metrics.csv")
    if not csv_path.exists():
        st.info("No evaluation results yet. Run `python -m src.evaluation.evaluator` to generate them.")
        return

    df = pd.read_csv(csv_path)

    # Show latest run only
    latest = df[df["timestamp"] == df["timestamp"].max()].copy()

    # Metric cards for the best pipeline
    best = latest.loc[latest["avg_score"].astype(float).idxmax()]
    st.markdown(f"**Best pipeline:** `{best['pipeline']}` — avg score `{float(best['avg_score']):.3f}`")
    st.divider()

    # Metrics comparison table
    display_cols = ["pipeline", "faithfulness", "answer_relevancy", "context_precision", "context_recall", "avg_score"]
    st.dataframe(
        latest[display_cols].style.format({
            col: "{:.4f}" for col in display_cols if col != "pipeline"
        }).background_gradient(
            subset=display_cols[1:], cmap="RdYlGn", vmin=0, vmax=1
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.divider()
    st.markdown("### What each metric means")
    st.markdown("""
| Metric | What it measures | Failure it catches |
|---|---|---|
| **Faithfulness** | Are all claims in the answer supported by the retrieved context? | Hallucination |
| **Answer Relevancy** (Factual Correctness) | Does the answer correctly state the facts? | Wrong numbers/names |
| **Context Precision** | Of retrieved chunks, how many are actually relevant? | Retriever noise |
| **Context Recall** | Did retrieval find all chunks needed to answer? | Retriever gaps |
""")

    if len(df["timestamp"].unique()) > 1:
        st.divider()
        st.markdown("### Score history across runs")
        history = df.pivot_table(index="timestamp", columns="pipeline", values="avg_score")
        st.line_chart(history)



# Tab 3 — About
def render_about_tab():
    st.header("About this project")

    st.markdown("""
### Finance RAG with Deep Evaluation

A **Retrieval-Augmented Generation (RAG)** system for querying SEC 10-K annual
filings from Apple, Microsoft, and Google.

Built over 4 weeks to demonstrate production-grade RAG engineering:
grounding LLM answers in real financial documents and measuring quality rigorously.

---

### Architecture

```
SEC EDGAR API
      ↓
  HTML parsing + XBRL noise removal
      ↓
  RecursiveCharacterTextSplitter (512 chars, 64 overlap)
      ↓
  text-embedding-3-small → ChromaDB (vector store)
      ↓
  Hybrid retrieval: BM25 + Vector + Reciprocal Rank Fusion
      ↓
  Cross-encoder reranker (ms-marco-MiniLM-L-6-v2)
      ↓
  GPT-4o-mini with grounding prompt
      ↓
  Answer + source attribution
```

### Evaluation

Measured with **RAGAS** across three pipeline variants on 20 hand-crafted
Q&A pairs derived from the actual filings.

Key finding: vector-only retrieval outperforms hybrid on a small corpus
(2 filings per company). BM25 adds noise when term frequency statistics
are thin. This was discovered through the evaluation layer — not assumed.

---

### Stack

`LangChain` · `ChromaDB` · `OpenAI` · `BM25` · `sentence-transformers` · `RAGAS` · `Streamlit`

**Source code:** [github.com/Armaan2022/finance-rag-eval](https://github.com/Armaan2022/finance-rag-eval)
""")



# Main app
def main():
    st.title("📊 Finance RAG")
    st.markdown("*Answers grounded in SEC 10-K filings from AAPL, MSFT, GOOGL*")
    st.divider()

    if not os.getenv("OPENAI_API_KEY"):
        st.error("OPENAI_API_KEY not set. Add it to your .env file.")
        st.stop()

    retrievers, chunks, vector_store = load_pipeline()

    tab1, tab2, tab3 = st.tabs(["💬 Q&A", "📈 Eval Results", "ℹ️ About"])

    with tab1:
        render_qa_tab(retrievers)
    with tab2:
        render_eval_tab()
    with tab3:
        render_about_tab()


if __name__ == "__main__":
    main()

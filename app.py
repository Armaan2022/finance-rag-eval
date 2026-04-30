import json
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


# ---------------------------------------------------------------------------
# Ticker JSON management
# ---------------------------------------------------------------------------

TICKERS_JSON = Path("data/loaded_tickers.json")
DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL"]


def load_tickers_json() -> list[str]:
    if not TICKERS_JSON.exists():
        save_tickers_json(DEFAULT_TICKERS)
        return list(DEFAULT_TICKERS)
    with open(TICKERS_JSON) as f:
        return json.load(f)["tickers"]


def save_tickers_json(tickers: list[str]) -> None:
    TICKERS_JSON.parent.mkdir(exist_ok=True)
    with open(TICKERS_JSON, "w") as f:
        json.dump({"tickers": sorted(set(t.upper() for t in tickers))}, f)


# ---------------------------------------------------------------------------
# Pipeline loading (cached per unique set of tickers)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading pipeline...")
def load_pipeline(tickers_key: str):
    """
    Load all pipeline components and cache them.

    tickers_key is a sorted comma-joined string of tickers (e.g. "AAPL,GOOGL,MSFT").
    Changing the tickers invalidates the cache and triggers a fresh load.
    Files are already downloaded locally and vectors are in Qdrant — only
    chunking (for BM25) and the Qdrant connection happen here.
    """
    from src.ingestion.sec_edgar import ingest_tickers
    from src.processing.chunker import chunk_documents
    from src.retrieval.qdrant_store import load_qdrant_store, get_retriever
    from src.retrieval.hybrid_search import HybridRetriever

    tickers = tickers_key.split(",")

    documents = ingest_tickers(tickers, max_filings_per_ticker=2)
    chunks = chunk_documents(documents, config_name="medium")
    vector_store = load_qdrant_store()

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


# ---------------------------------------------------------------------------
# Sidebar — ticker management
# ---------------------------------------------------------------------------

def render_sidebar(tickers: list[str]) -> list[str]:
    """Render the ticker management sidebar. Returns the current tickers list."""
    with st.sidebar:
        st.header("Loaded Filings")
        st.caption(f"{len(tickers)} ticker(s) in Qdrant")
        st.divider()

        for ticker in tickers:
            col1, col2 = st.columns([5, 1])
            col1.markdown(f"**{ticker}**")
            if col2.button("🗑", key=f"remove_{ticker}", help=f"Remove {ticker}"):
                with st.spinner(f"Removing {ticker}..."):
                    from src.retrieval.qdrant_store import delete_ticker_vectors
                    delete_ticker_vectors(ticker)
                    tickers = [t for t in tickers if t != ticker]
                    save_tickers_json(tickers)
                    load_pipeline.clear()
                st.rerun()

        st.divider()
        st.markdown("**Add a ticker**")

        with st.form("add_ticker_form", clear_on_submit=True):
            new_ticker = st.text_input("Ticker symbol", placeholder="e.g. NVDA").strip().upper()
            submitted = st.form_submit_button("Load filings", use_container_width=True)

        if submitted and new_ticker:
            if new_ticker in tickers:
                st.error(f"{new_ticker} is already loaded.")
            else:
                with st.spinner(f"Fetching {new_ticker} from SEC EDGAR..."):
                    try:
                        from src.ingestion.sec_edgar import get_cik_from_ticker, ingest_tickers
                        get_cik_from_ticker(new_ticker)  # raises ValueError if not found
                        documents = ingest_tickers([new_ticker], max_filings_per_ticker=2)
                        if not documents:
                            st.error(f"No 10-K filings found for {new_ticker}.")
                        else:
                            from src.processing.chunker import chunk_documents
                            from src.retrieval.qdrant_store import upload_ticker_chunks
                            chunks = chunk_documents(documents, config_name="medium")
                            upload_ticker_chunks(chunks)
                            tickers = tickers + [new_ticker]
                            save_tickers_json(tickers)
                            load_pipeline.clear()
                            st.rerun()
                    except ValueError:
                        st.error(f"'{new_ticker}' not found on SEC EDGAR. Check the ticker symbol.")

    return tickers


# ---------------------------------------------------------------------------
# Tab 1 — Q&A Interface
# ---------------------------------------------------------------------------

def render_qa_tab(retrievers, tickers):
    st.header("Ask questions about SEC 10-K filings")
    st.caption(f"Covers {', '.join(tickers)} — 2024 & 2025 annual reports")

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


# ---------------------------------------------------------------------------
# Tab 2 — Evaluation Results
# ---------------------------------------------------------------------------

def render_eval_tab():
    st.header("Pipeline Evaluation Results")
    st.caption("RAGAS metrics measured on 10 hand-crafted Q&A pairs from AAPL 10-K filings")

    csv_path = "results/eval_metrics.csv"
    if not os.path.exists(csv_path):
        st.info("No evaluation results yet. Run `python -m src.evaluation.evaluator` to generate them.")
        return

    df = pd.read_csv(csv_path)

    latest = df[df["timestamp"] == df["timestamp"].max()].copy()

    best = latest.loc[latest["avg_score"].astype(float).idxmax()]
    st.markdown(f"**Best pipeline:** `{best['pipeline']}` — avg score `{float(best['avg_score']):.3f}`")
    st.divider()

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


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main():
    st.title("📊 Finance RAG")
    st.divider()

    if not os.getenv("OPENAI_API_KEY"):
        st.error("OPENAI_API_KEY not set. Add it to your .env file.")
        st.stop()

    tickers = load_tickers_json()
    tickers = render_sidebar(tickers)

    tickers_key = ",".join(sorted(tickers))
    retrievers, chunks, vector_store = load_pipeline(tickers_key)

    tab1, tab2 = st.tabs(["Chat", "Eval Dashboard"])

    with tab1:
        render_qa_tab(retrievers, tickers)
    with tab2:
        render_eval_tab()


if __name__ == "__main__":
    main()

import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="Finance RAG",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

st.markdown("""
<style>

/* Hide sidebar collapse arrow */
[data-testid="stSidebarCollapseButton"] { display: none !important; }

/* Tighten all vertical spacing in sidebar */
section[data-testid="stSidebar"] > div { padding-top: 1.2rem !important; }
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: 0 !important; }
[data-testid="stSidebar"] hr { margin: 0.45rem 0 !important; }
[data-testid="stSidebar"] p { margin-bottom: 0 !important; font-size: 13px; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] { margin: 0 !important; }
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] { margin: 0 0 2px 0 !important; }
/* Remove form border */
[data-testid="stSidebar"] [data-testid="stForm"] {
    border: none !important;
    padding: 0 !important;
    background: transparent !important;
}

/* ── Nav tabs: radio styled as clickable items ── */
/* Hide label above radio group */
[data-testid="stSidebar"] [data-testid="stRadio"] > label { display: none !important; }
/* Hide radio circles */
[data-testid="stSidebar"] [data-testid="stRadio"] [data-baseweb="radio"] > div:first-child {
    display: none !important;
}
/* Each nav item */
[data-testid="stSidebar"] [data-testid="stRadio"] [data-baseweb="radio"] {
    padding: 7px 10px !important;
    border-radius: 6px !important;
    cursor: pointer !important;
    color: rgba(255,255,255,0.45) !important;
    transition: background 0.15s, color 0.15s !important;
    width: 100% !important;
    font-size: 13px !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] [data-baseweb="radio"]:hover {
    background: rgba(255,255,255,0.08) !important;
    color: rgba(255,255,255,0.9) !important;
}
/* Selected nav item */
[data-testid="stSidebar"] [data-testid="stRadio"] [data-checked="true"][data-baseweb="radio"],
[data-testid="stSidebar"] [data-testid="stRadio"] [aria-checked="true"] {
    background: rgba(255,255,255,0.14) !important;
    color: #fff !important;
    font-weight: 600 !important;
}

/* ── Filing cards ── */
[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(255,255,255,0.07) !important;
    border: none !important;
    border-radius: 6px !important;
    padding: 0 6px !important;
    margin: 2px 0 !important;
}
[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"] p {
    font-size: 13px !important;
    line-height: 2.2 !important;
}

/* ── Hover delete: × hidden by default, shown on card hover ── */
[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"] button {
    opacity: 0 !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    color: #ff4b4b !important;
    font-size: 13px !important;
    font-weight: bold !important;
    padding: 0 4px !important;
    min-height: 28px !important;
    height: 28px !important;
    transition: opacity 0.15s !important;
}
[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"]:hover button {
    opacity: 1 !important;
}

</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Ticker list — derived from Qdrant (single source of truth)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=30, show_spinner=False)
def fetch_loaded_tickers() -> list[str]:
    from src.retrieval.qdrant_store import get_tickers_from_qdrant
    return get_tickers_from_qdrant()


# ---------------------------------------------------------------------------
# Pipeline (cached per ticker set)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading pipeline...")
def load_pipeline(tickers_key: str):
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
# Build a filtered retriever on demand
# ---------------------------------------------------------------------------

def build_filtered_retriever(pipeline_choice, vector_store, chunks, ticker_filter):
    from src.retrieval.qdrant_store import get_retriever
    from src.retrieval.hybrid_search import HybridRetriever
    from qdrant_client.models import Filter, FieldCondition, MatchAny

    qdrant_filter = Filter(must=[
        FieldCondition(key="metadata.ticker", match=MatchAny(any=ticker_filter))
    ])
    bm25_filter = {"ticker": ticker_filter}  # dict used for BM25 chunk filtering
    k = 20 if pipeline_choice == "Hybrid + Reranker" else 5

    if pipeline_choice == "Vector only":
        return get_retriever(vector_store, k=k, metadata_filter=qdrant_filter)
    return HybridRetriever(
        vector_store=vector_store,
        chunks=chunks,
        k=k,
        fetch_k=20,
        metadata_filter=bm25_filter,
    )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar(tickers: list[str]) -> tuple[list[str], str]:
    with st.sidebar:
        st.markdown("### 📊 Finance RAG")
        st.divider()

        page = st.radio(
            "nav",
            ["💬  Q&A", "📈  Eval Dashboard", "ℹ️  About"],
            label_visibility="collapsed",
        )

        st.divider()
        st.caption(f"Loaded Filings · {len(tickers)} tickers")

        for ticker in tickers:
            with st.container(border=True):
                col1, col2 = st.columns([5, 1])
                col1.markdown(ticker)
                if col2.button("✕", key=f"remove_{ticker}", help=f"Remove {ticker}"):
                    with st.spinner(f"Removing {ticker}..."):
                        from src.retrieval.qdrant_store import delete_ticker_vectors
                        delete_ticker_vectors(ticker)
                    fetch_loaded_tickers.clear()
                    load_pipeline.clear()
                    st.rerun()

        st.divider()

        with st.form("add_ticker_form", clear_on_submit=True):
            new_ticker = st.text_input(
                "ticker", placeholder="Add ticker  e.g. NVDA",
                label_visibility="collapsed"
            ).strip().upper()
            submitted = st.form_submit_button("Load filings", use_container_width=True)

        if submitted and new_ticker:
            if new_ticker in tickers:
                st.error(f"{new_ticker} is already loaded.")
            else:
                with st.spinner(f"Fetching {new_ticker} from SEC EDGAR..."):
                    try:
                        from src.ingestion.sec_edgar import get_cik_from_ticker, ingest_tickers
                        get_cik_from_ticker(new_ticker)
                        documents = ingest_tickers([new_ticker], max_filings_per_ticker=2)
                        if not documents:
                            st.error(f"No 10-K filings found for {new_ticker}.")
                        else:
                            from src.processing.chunker import chunk_documents
                            from src.retrieval.qdrant_store import upload_ticker_chunks
                            chunks = chunk_documents(documents, config_name="medium")
                            upload_ticker_chunks(chunks)
                            fetch_loaded_tickers.clear()
                            load_pipeline.clear()
                            st.rerun()
                    except ValueError:
                        st.error(f"'{new_ticker}' not found on SEC EDGAR.")

    return tickers, page


# ---------------------------------------------------------------------------
# Q&A tab
# ---------------------------------------------------------------------------

def render_qa_tab(retrievers, tickers, chunks, vector_store):
    st.header("Ask questions about SEC 10-K filings")

    col_main, col_side = st.columns([2, 1])

    with col_side:
        pipeline_choice = st.selectbox(
            "Pipeline",
            options=list(retrievers.keys()),
            help="Vector only: pure semantic search\nHybrid: adds BM25 keyword search\nHybrid + Reranker: cross-encoder precision pass",
        )

        with st.expander("Filters", expanded=False):
            ticker_filter = st.multiselect("Company", options=tickers, placeholder="All companies")

            available_years = sorted(set(
                c.metadata["filing_date"][:4]
                for c in chunks if c.metadata.get("filing_date")
            ), reverse=True)
            year_filter = st.multiselect("Filing year", options=available_years, placeholder="All years")

            doc_types = sorted(set(c.metadata.get("form_type", "10-K") for c in chunks))
            doc_type_filter = st.multiselect("Document type", options=doc_types, placeholder="All types")

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

    with col_main:
        query = st.text_input(
            "Your question",
            value=st.session_state.get("query", ""),
            placeholder="e.g. What was Apple's net income in fiscal year 2025?",
        )

        if st.button("Ask", type="primary", use_container_width=True) and query:
            retriever = (
                build_filtered_retriever(pipeline_choice, vector_store, chunks, ticker_filter)
                if ticker_filter else retrievers[pipeline_choice]
            )

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
                            "form_type": d.metadata.get("form_type", "10-K"),
                            "preview": d.page_content[:200].replace("\n", " "),
                        }
                        for d in reranked_docs
                    ]

                # Client-side year / doc type filter on sources
                if year_filter or doc_type_filter:
                    result["sources"] = [
                        s for s in result["sources"]
                        if (not year_filter or s["filing_date"][:4] in year_filter)
                        and (not doc_type_filter or s.get("form_type", "10-K") in doc_type_filter)
                    ]

            st.markdown("### Answer")
            st.markdown(result["answer"])

            st.markdown("### Sources")
            if not result["sources"]:
                st.caption("No sources match the active filters.")
            for i, src in enumerate(result["sources"][:5]):
                with st.expander(f"[{i+1}] {src['ticker']} — {src['filing_date']} ({src.get('form_type','10-K')})"):
                    st.caption(src["preview"] + "...")


# ---------------------------------------------------------------------------
# Eval dashboard
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
        }).background_gradient(subset=display_cols[1:], cmap="RdYlGn", vmin=0, vmax=1),
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
# About
# ---------------------------------------------------------------------------

def render_about_tab():
    st.header("About this project")
    st.markdown("""
### Finance RAG with Deep Evaluation

A **Retrieval-Augmented Generation (RAG)** system for querying SEC 10-K annual
filings from Apple, Microsoft, and Google.

---

### Architecture

```
SEC EDGAR API
      ↓
  HTML parsing + XBRL noise removal
      ↓
  RecursiveCharacterTextSplitter (512 chars, 64 overlap)
      ↓
  text-embedding-3-small → Qdrant Cloud (vector store)
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

Measured with **RAGAS** across three pipeline variants on 20 hand-crafted Q&A pairs.
Key finding: vector-only outperforms hybrid on a small corpus — discovered through evaluation, not assumed.

---

### Stack

`LangChain` · `Qdrant Cloud` · `OpenAI` · `BM25` · `sentence-transformers` · `RAGAS` · `Streamlit`

**Source code:** [github.com/Armaan2022/finance-rag-eval](https://github.com/Armaan2022/finance-rag-eval)
""")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not os.getenv("OPENAI_API_KEY"):
        st.error("OPENAI_API_KEY not set. Add it to your .env file.")
        st.stop()

    tickers = fetch_loaded_tickers()
    tickers, page = render_sidebar(tickers)

    tickers_key = ",".join(sorted(tickers))
    retrievers, chunks, vector_store = load_pipeline(tickers_key)

    if page == "💬  Q&A":
        render_qa_tab(retrievers, tickers, chunks, vector_store)
    elif page == "📈  Eval Dashboard":
        render_eval_tab()
    elif page == "ℹ️  About":
        render_about_tab()


if __name__ == "__main__":
    main()

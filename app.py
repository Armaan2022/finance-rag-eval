import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from src.rag.pipeline import _sentence_preview

load_dotenv()

st.set_page_config(
    page_title="Finance RAG",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ──────────────────────────────────────────────────────────────────────

with open("style.css") as _f:
    st.markdown(f"<style>{_f.read()}</style>", unsafe_allow_html=True)


# ── Cached data & pipeline ───────────────────────────────────────────────────

@st.cache_data(ttl=30, show_spinner=False)
def fetch_loaded_tickers() -> list[str]:
    from src.retrieval.qdrant_store import get_tickers_from_qdrant
    return get_tickers_from_qdrant()


@st.cache_resource(show_spinner=False)
def load_pipeline(tickers_key: str):
    from src.retrieval.qdrant_store import load_qdrant_store, get_retriever, get_chunks_from_qdrant
    from src.retrieval.hybrid_search import HybridRetriever

    # Load chunks from Qdrant payloads — no SEC EDGAR download needed here.
    # tickers_key is only used as the cache key so stale entries are invalidated
    # when tickers are added or removed.
    chunks = get_chunks_from_qdrant()
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


def build_filtered_retriever(
    pipeline_choice, vector_store, chunks,
    ticker_filter=None, year_filter=None, doc_type_filter=None,
):
    from src.retrieval.qdrant_store import get_retriever
    from src.retrieval.hybrid_search import HybridRetriever

    # Build a plain dict filter accepted by both get_retriever and HybridRetriever.
    # List values are treated as OR (MatchAny) in both implementations.
    meta_filter: dict = {}
    if ticker_filter:
        meta_filter["ticker"] = ticker_filter
    if year_filter:
        # Translate selected years → exact filing_date strings so Qdrant can filter
        # by a keyword index rather than a string prefix (which it doesn't support).
        matching_dates = list({
            c.metadata["filing_date"]
            for c in chunks
            if c.metadata.get("filing_date", "")[:4] in year_filter
            and (not ticker_filter or c.metadata.get("ticker") in ticker_filter)
        })
        if matching_dates:
            meta_filter["filing_date"] = matching_dates
    if doc_type_filter:
        meta_filter["form_type"] = doc_type_filter

    effective_filter = meta_filter or None
    k = 20 if pipeline_choice == "Hybrid + Reranker" else 5
    if pipeline_choice == "Vector only":
        return get_retriever(vector_store, k=k, metadata_filter=effective_filter)
    return HybridRetriever(
        vector_store=vector_store,
        chunks=chunks,
        k=k,
        fetch_k=20,
        metadata_filter=effective_filter,
    )


# ── Navbar ───────────────────────────────────────────────────────────────────

def render_navbar(page: str):
    def nav_link(label: str, key: str) -> str:
        active = page == key
        color = "#f0f6fc" if active else "#7d8590"
        bg = "rgba(177,186,196,0.1)" if active else "transparent"
        weight = "500" if active else "400"
        return (
            f'<a href="?page={key}" target="_self" style="'
            f"text-decoration:none;color:{color};font-size:14px;"
            f"padding:6px 14px;border-radius:6px;margin-right:2px;"
            f"background:{bg};font-weight:{weight};"
            f'">{label}</a>'
        )

    st.markdown(
        f"""
        <div style="
            display:flex;align-items:center;
            padding:16px 0;
            border-bottom:1px solid #21262d;
            margin-bottom:28px;
        ">
            <span style="
                font-size:15px;font-weight:600;color:#f0f6fc;
                margin-right:28px;letter-spacing:-0.01em;
            ">Finance RAG</span>
            {nav_link("Q&A", "qa")}
            {nav_link("Eval Dashboard", "eval")}
            {nav_link("About", "about")}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Filings panel ────────────────────────────────────────────────────────────

# Beyond this many tickers the list switches from "size to fit" to a fixed
# height with scrolling, so it can't grow forever.
_FILINGS_MAX_VISIBLE_ROWS = 8
# Only used once the cap above is exceeded — an approximate height for ~8
# rows. It doesn't need to be pixel-exact: it just defines where scrolling
# kicks in, not a "no empty space" guarantee like the auto-fit case below.
_FILINGS_SCROLL_HEIGHT = 340


def render_filings_panel(tickers: list[str]):
    st.markdown(
        '<p style="font-size:11px;font-weight:600;color:#7d8590;'
        'letter-spacing:0.08em;text-transform:uppercase;margin:0 0 8px 0;">'
        "Loaded Filings</p>",
        unsafe_allow_html=True,
    )

    # Under the cap: no fixed height, so the container hugs its content
    # exactly — there's no leftover space below the last row because we
    # never guess a height, Streamlit just sizes it to what's rendered.
    # Over the cap: a fixed height turns scrolling on instead of letting
    # the panel grow forever.
    container_kwargs = {"key": "filings_list"}
    if len(tickers) > _FILINGS_MAX_VISIBLE_ROWS:
        container_kwargs["height"] = _FILINGS_SCROLL_HEIGHT

    with st.container(**container_kwargs):
        if not tickers:
            st.caption("No filings loaded.")
        for ticker in tickers:
            # key=f"ticker_row_{ticker}" gives this row a stable
            # "st-key-ticker_row_<ticker>" class. style.css uses it to lay
            # out the label and the remove button side by side (label
            # grows, button is a fixed 36x36 square) and to hide the
            # button until the row is hovered.
            with st.container(key=f"ticker_row_{ticker}"):
                c1, c2 = st.columns([6, 1])
                c1.markdown(
                    f'<div class="ticker-row-label">{ticker}</div>',
                    unsafe_allow_html=True,
                )
                if c2.button("×", key=f"rm_{ticker}", help=f"Remove {ticker}"):
                    with st.spinner(f"Removing {ticker}..."):
                        from src.retrieval.qdrant_store import delete_ticker_vectors
                        delete_ticker_vectors(ticker)
                    fetch_loaded_tickers.clear()
                    load_pipeline.clear()
                    st.rerun()

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    with st.form("add_ticker_form", clear_on_submit=True):
        new_ticker = st.text_input(
            "ticker",
            placeholder="Ticker, e.g. NVDA",
            label_visibility="collapsed",
        ).strip().upper()
        submitted = st.form_submit_button("Load filing", use_container_width=True)

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
                        chunks = chunk_documents(documents, config_name="large")
                        upload_ticker_chunks(chunks)
                        fetch_loaded_tickers.clear()
                        load_pipeline.clear()
                        st.rerun()
                except ValueError:
                    st.error(f"'{new_ticker}' not found on SEC EDGAR.")


# ── Q&A tab ──────────────────────────────────────────────────────────────────

def render_qa_tab(retrievers, tickers, chunks, vector_store):
    col_filings, col_main, col_pipeline = st.columns([1, 2, 1], gap="large")

    with col_filings:
        render_filings_panel(tickers)

    with col_pipeline:
        st.markdown(
            '<p style="font-size:11px;font-weight:600;color:#7d8590;'
            'letter-spacing:0.08em;text-transform:uppercase;margin:0 0 6px 0;">'
            "Pipeline</p>",
            unsafe_allow_html=True,
        )
        pipeline_choice = st.selectbox(
            "Pipeline",
            options=list(retrievers.keys()),
            label_visibility="collapsed",
            help=(
                "Vector only: pure semantic search\n"
                "Hybrid: BM25 keyword + semantic + RRF fusion\n"
                "Hybrid + Reranker: cross-encoder precision pass"
            ),
        )

        st.markdown("<div style='margin-top:16px;'></div>", unsafe_allow_html=True)

        with st.expander("Filters", expanded=False):
            ticker_filter = st.multiselect(
                "Company", options=tickers, placeholder="All companies"
            )
            available_years = sorted(
                {c.metadata["filing_date"][:4] for c in chunks if c.metadata.get("filing_date")},
                reverse=True,
            )
            year_filter = st.multiselect(
                "Filing year", options=available_years, placeholder="All years"
            )
            doc_types = sorted({c.metadata.get("form_type", "10-K") for c in chunks})
            doc_type_filter = st.multiselect(
                "Document type", options=doc_types, placeholder="All types"
            )

        st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)
        st.markdown(
            '<p style="font-size:11px;font-weight:600;color:#7d8590;'
            'letter-spacing:0.08em;text-transform:uppercase;margin:0 0 8px 0;">'
            "Example Questions</p>",
            unsafe_allow_html=True,
        )
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
            "Question",
            value=st.session_state.get("query", ""),
            placeholder="e.g. What was Apple's net income in fiscal year 2025?",
            label_visibility="collapsed",
        )

        if st.button("Ask", type="primary", use_container_width=True) and query:
            retriever = (
                build_filtered_retriever(
                    pipeline_choice, vector_store, chunks,
                    ticker_filter=ticker_filter,
                    year_filter=year_filter,
                    doc_type_filter=doc_type_filter,
                )
                if (ticker_filter or year_filter or doc_type_filter)
                else retrievers[pipeline_choice]
            )

            with st.spinner("Retrieving and generating answer..."):
                from src.rag.pipeline import build_rag_chain, ask

                use_reranker = pipeline_choice == "Hybrid + Reranker"
                chain = build_rag_chain(retriever)
                result = ask(chain, retriever, query)

                if use_reranker:
                    from src.retrieval.reranker import retrieve_and_rerank
                    reranked_docs = retrieve_and_rerank(query, retriever, top_n=3)
                    result["sources"] = [
                        {
                            "ticker": d.metadata.get("ticker", "?"),
                            "company_name": d.metadata.get("company_name", ""),
                            "filing_date": d.metadata.get("filing_date", "?"),
                            "form_type": d.metadata.get("form_type", "10-K"),
                            "section": d.metadata.get("section", ""),
                            "preview": _sentence_preview(d.page_content),
                        }
                        for d in reranked_docs
                    ]

                if year_filter or doc_type_filter:
                    result["sources"] = [
                        s
                        for s in result["sources"]
                        if (not year_filter or s["filing_date"][:4] in year_filter)
                        and (not doc_type_filter or s.get("form_type", "10-K") in doc_type_filter)
                    ]

            st.markdown("---")
            st.markdown("**Answer**")
            st.markdown(result["answer"])

            st.markdown("**Sources**")
            if not result["sources"]:
                st.caption("No sources match the active filters.")
            for i, src in enumerate(result["sources"][:5]):
                section_part = f" — {src['section']}" if src.get("section") else ""
                label = f"[{i + 1}]  {src['ticker']}{section_part} — {src['filing_date']} ({src.get('form_type', '10-K')})"
                with st.expander(label):
                    if src.get("company_name"):
                        st.caption(src["company_name"])
                    st.markdown(src["preview"])


# ── Eval dashboard ───────────────────────────────────────────────────────────

def render_eval_tab():
    st.markdown("## Pipeline Evaluation Results")
    st.caption("RAGAS metrics measured on 18 hand-crafted Q&A pairs across AAPL and MSFT 10-K filings.")

    csv_path = "results/eval_metrics.csv"
    if not os.path.exists(csv_path):
        st.info("No evaluation results yet. Run `python -m src.evaluation.evaluator` to generate them.")
        return

    df = pd.read_csv(csv_path)
    latest = df[df["timestamp"] == df["timestamp"].max()].copy()

    best = latest.loc[latest["avg_score"].astype(float).idxmax()]
    st.markdown(
        f"**Best pipeline:** `{best['pipeline']}` — avg score `{float(best['avg_score']):.3f}`"
    )
    st.divider()

    display_cols = [
        "pipeline",
        "faithfulness",
        "factual_correctness",
        "context_precision",
        "context_recall",
        "avg_score",
    ]
    st.dataframe(
        latest[display_cols]
        .style.format({col: "{:.4f}" for col in display_cols if col != "pipeline"})
        .background_gradient(subset=display_cols[1:], cmap="RdYlGn", vmin=0, vmax=1),
        use_container_width=True,
        hide_index=True,
    )

    st.divider()
    st.markdown("**What each metric measures**")
    st.markdown(
        """
| Metric | What it measures | Failure it catches |
|---|---|---|
| **Faithfulness** | Are all claims in the answer supported by retrieved context? | Hallucination |
| **Factual Correctness** | Does the answer state the correct facts (F1 vs. ground truth)? | Wrong numbers / names |
| **Context Precision** | Of retrieved chunks, how many are actually relevant? | Retriever noise |
| **Context Recall** | Did retrieval find all chunks needed to answer the question? | Retriever gaps |
"""
    )

    if len(df["timestamp"].unique()) > 1:
        st.divider()
        st.markdown("**Score history across runs**")
        history = df.pivot_table(index="timestamp", columns="pipeline", values="avg_score")
        st.line_chart(history)


# ── About ────────────────────────────────────────────────────────────────────

def render_about_tab():
    st.markdown("## About")
    st.markdown(
        """
A Retrieval-Augmented Generation system for querying SEC 10-K annual filings.
Load filings for any public company and ask questions grounded strictly in the source documents.

---

### Architecture

```
SEC EDGAR API
      |
  HTML parsing + XBRL noise removal
      |
  RecursiveCharacterTextSplitter (512 chars, 64 overlap)
      |
  text-embedding-3-small  →  Qdrant Cloud
      |
  Hybrid retrieval: BM25 + Vector + Reciprocal Rank Fusion
      |
  Cross-encoder reranker (ms-marco-MiniLM-L-6-v2)
      |
  GPT-4o-mini with grounding prompt
      |
  Answer + source attribution
```

### Evaluation

Measured with RAGAS across three pipeline variants on 20 hand-crafted Q&A pairs.
Key finding: vector-only retrieval outperforms hybrid on a small corpus — discovered through evaluation, not assumed.

---

### Stack

`LangChain` · `Qdrant Cloud` · `OpenAI` · `BM25` · `sentence-transformers` · `RAGAS` · `Streamlit`

**Source:** [github.com/Armaan2022/finance-rag-eval](https://github.com/Armaan2022/finance-rag-eval)
"""
    )


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    if not os.getenv("OPENAI_API_KEY"):
        st.error("OPENAI_API_KEY not set. Add it to your .env file.")
        st.stop()

    page = st.query_params.get("page", "qa")
    render_navbar(page)

    if page == "eval":
        render_eval_tab()
        return
    if page == "about":
        render_about_tab()
        return

    tickers = fetch_loaded_tickers()
    tickers_key = ",".join(sorted(tickers))
    retrievers, chunks, vector_store = load_pipeline(tickers_key)
    render_qa_tab(retrievers, tickers, chunks, vector_store)


if __name__ == "__main__":
    main()
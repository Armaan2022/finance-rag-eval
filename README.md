# Finance RAG with Evaluation

A RAG system for querying SEC 10-K annual filings. You can load any public company's filing, ask questions, and get answers grounded in the actual documents with source attribution. The main thing I wanted to build here wasn't just a working RAG but a proper evaluation pipeline so I could measure what actually improves retrieval quality instead of guessing.

**Live demo:** [https://finance-rag-eval-armaan.streamlit.app/]

---

## The Problem

LLMs usually hallucinate financial figures. A system that confidently states the wrong revenue number is worse than no system at all. I wanted every answer to be traceable back to a real SEC filing, and I wanted to actually measure whether the retrieval was working, not just assume it was.

---

## Architecture

```
SEC EDGAR API (free, no auth)
      │
      ▼
HTML parsing + XBRL noise removal
      │
      ▼
RecursiveCharacterTextSplitter  (1024 chars, 128 overlap, page-header chunks dropped)
      │
      ▼
text-embedding-3-small  →  Qdrant Cloud (vector store)
      │
      ▼
Hybrid retrieval: BM25 (stemmed) + Vector + Reciprocal Rank Fusion
      │
      ▼
Cross-encoder reranker  (ms-marco-MiniLM-L-6-v2, fused with the hybrid ranking)
      │
      ▼
GPT-4o-mini with grounding prompt  (reads exactly the chunks shown as sources)
      │
      ▼
Answer + source attribution (company, filing date, 10-K section)
```

---

## Evaluation Results

Measured with **RAGAS** across three pipeline variants on 18 hand-crafted Q&A pairs covering financials, business strategy, and risk factors from AAPL and MSFT 10-K filings. The collection holds two 10-Ks each for AAPL, MSFT, GOOGL, META, NVDA and AMZN, so the other four companies act as distractors.

| Pipeline | Faithfulness | Factual Correctness | Context Precision | Context Recall | **Avg** |
|---|---|---|---|---|---|
| Vector only | **0.796** | **0.426** | 0.552 | 0.583 | **0.589** |
| Hybrid (BM25 + Vector + RRF) | 0.688 | 0.377 | 0.521 | 0.602 | 0.547 |
| Hybrid + Reranker | 0.705 | 0.356 | **0.635** | **0.611** | 0.577 |

All three pipelines retrieve 5 chunks, so they're compared on ranking quality alone.

**Key findings:**

- **My first evaluation of the reranker was wrong.** The answer was generated from all 20 hybrid candidates, but RAGAS judged it against only the 3 reranked chunks. True facts from the 17 chunks the judge never saw were marked unfaithful. After fixing that, the reranker went from last place (0.470 avg, 0.479 faithfulness) to roughly tied with the others.
- **Fixing the evaluation exposed a real weakness in the reranker.** `ms-marco-MiniLM` is trained on web search passages and ranks financial tables below prose that only mentions the topic. For "What was Apple's total revenue in 2025?", hybrid ranked the net sales table #1 and the reranker dropped it to #7, so the model answered "not enough information". Fusing the reranker's order with the hybrid order (RRF) fixed this, and it now has the best precision and recall.
- **Data quality mattered more than pipeline choice.** Removing page-header chunks like `Apple Inc. | 2025 Form 10-K | 28 Apple Inc` raised every pipeline (vector only 0.535 → 0.590). Those chunks contain the company name and nothing else, so they matched company queries and filled up to 5 of hybrid's top 10 results.
- **Chunk size was the biggest single factor.** Switching from 512 to 1024 chars pushed faithfulness from ~0.47 to 0.87 in an earlier run. Larger chunks give the LLM wider context, so it stays grounded in the source. The whole collection now uses one chunk size, defined in one place.
- **The three pipelines are within noise of each other.** Hybrid's average moved by about 0.04 between two runs with no code change, which is the variance of an LLM judge on 18 questions. Claiming one pipeline is better would need more questions or averaged runs. Vector only is the most stable.

To re-run the evaluation:
```bash
python -m src.evaluation.evaluator
```

---

## Stack

| Layer | Tool |
|---|---|
| Ingestion | SEC EDGAR API, BeautifulSoup |
| Chunking | LangChain `RecursiveCharacterTextSplitter` |
| Embeddings | OpenAI `text-embedding-3-small` |
| Vector store | Qdrant Cloud |
| Keyword search | BM25 (`rank-bm25`) with Snowball stemming |
| Fusion | Reciprocal Rank Fusion (BM25 + vector, and hybrid + reranker) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| LLM | OpenAI `gpt-4o-mini` |
| Orchestration | LangChain LCEL |
| Evaluation | RAGAS |
| UI | Streamlit |

---

## Setup

```bash
git clone https://github.com/Armaan2022/finance-rag-eval.git
cd finance-rag-eval
python -m venv venv && source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file:
```
OPENAI_API_KEY=sk-...
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=your-key
```

Run the app:
```bash
streamlit run app.py
```

Use the **Load filing** form in the app to add any public company by ticker. Filings are fetched from SEC EDGAR, chunked, embedded, and uploaded to Qdrant automatically.

If you change the chunking settings, re-chunk the existing collection so it stays consistent:
```bash
python -m scripts.rechunk_collection           # dry run: shows old vs new chunk counts
python -m scripts.rechunk_collection --apply   # replaces the vectors, one ticker at a time
```
It re-reads the exact filings already stored (from `data/`) instead of downloading the latest ones, so the corpus doesn't change underneath the evaluation.

---

## Project Structure

```
src/
├── ingestion/     # SEC EDGAR download + HTML parsing
├── processing/    # Document chunking + section detection
├── retrieval/     # Qdrant store, BM25, hybrid search, reranker
├── rag/           # LangChain pipeline + prompt
└── evaluation/    # RAGAS eval dataset + evaluator
scripts/
├── upload_to_qdrant.py   # Bulk upload script (optional, skips tickers already loaded)
└── rechunk_collection.py # Re-chunk the whole collection after chunking changes
results/
└── eval_metrics.csv      # Evaluation results across runs
app.py                    # Streamlit UI
style.css                 # Theme styles
```

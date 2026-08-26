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
RecursiveCharacterTextSplitter  (1024 chars, 128 overlap)
      │
      ▼
text-embedding-3-small  →  Qdrant Cloud (vector store)
      │
      ▼
Hybrid retrieval: BM25 + Vector + Reciprocal Rank Fusion
      │
      ▼
Cross-encoder reranker  (ms-marco-MiniLM-L-6-v2)
      │
      ▼
GPT-4o-mini with grounding prompt
      │
      ▼
Answer + source attribution (company, filing date, 10-K section)
```

---

## Evaluation Results

Measured with **RAGAS** across three pipeline variants on 18 hand-crafted Q&A pairs covering financials, business strategy, and risk factors from AAPL and MSFT 10-K filings.

| Pipeline | Faithfulness | Factual Correctness | Context Precision | Context Recall | **Avg** |
|---|---|---|---|---|---|
| Vector only | **0.870** | 0.362 | 0.570 | 0.611 | **0.603** |
| Hybrid (BM25 + Vector + RRF) | 0.699 | 0.344 | 0.486 | 0.597 | 0.532 |
| Hybrid + Reranker | 0.479 | **0.409** | 0.519 | 0.472 | 0.470 |

**Key findings:**

- Vector-only retrieval won overall, which surprised me. BM25 is supposed to help but it added noise here since the dataset is too small and domain-specific for keyword statistics to be meaningful. On a larger, more diverse corpus I'd expect hybrid to perform better. 
- Chunk size made a bigger difference than pipeline complexity. Switching from 512 to 1024 chars pushed faithfulness from ~0.47 to 0.87. Larger chunks give the LLM wider context, so it stays grounded in the source.
- The reranker had the best factual correctness (0.41) but worst recall (0.47). It's precise when it gets the right chunks but too aggressive at filtering, so the LLM ends up with less context overall.
- Faithfulness is the metric I care about most here. 0.87 means the system is rarely making claims that aren't in the retrieved documents.

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
| Keyword search | BM25 (`rank-bm25`) |
| Fusion | Reciprocal Rank Fusion |
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
└── upload_to_qdrant.py   # Bulk upload script (optional)
results/
└── eval_metrics.csv      # Evaluation results across runs
app.py                    # Streamlit UI
style.css                 # Theme styles
```

# Finance RAG with Deep Evaluation

A production-grade **Retrieval-Augmented Generation (RAG)** system for querying SEC 10-K annual filings from Apple, Microsoft, and Google. Built to demonstrate that evaluation is not an afterthought — it's how you discover what actually works.

**Live demo:** [YOUR_APP_URL]

---

## The Problem

LLMs hallucinate financial figures. A chatbot that confidently states the wrong revenue number is worse than no chatbot at all. This project grounds every answer in real SEC filings and measures retrieval quality with RAGAS — so claims are traceable and quality is measurable, not assumed.

---

## Architecture

```
SEC EDGAR API (free, no auth)
      │
      ▼
HTML parsing + XBRL noise removal
      │
      ▼
RecursiveCharacterTextSplitter  (512 chars, 64 overlap)
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
Answer + source attribution
```

---

## Evaluation Results

Measured with **RAGAS** across three pipeline variants on 10 hand-crafted Q&A pairs derived from the actual filings.

| Pipeline | Faithfulness | Factual Correctness | Context Precision | Context Recall | **Avg** |
|---|---|---|---|---|---|
| Vector only | 0.867 | 0.426 | 0.695 | 0.700 | **0.672** |
| Hybrid (BM25 + Vector + RRF) | 0.804 | 0.399 | 0.584 | 0.600 | 0.597 |
| Hybrid + Reranker | 0.669 | 0.416 | 0.642 | 0.633 | 0.590 |

**Key finding:** Vector-only retrieval outperformed hybrid on this corpus. BM25 adds noise when term-frequency statistics are thin (only 2 filings per company). This was discovered through the evaluation layer — not assumed. On a larger corpus, the hybrid advantage would likely emerge.

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

Upload vectors to Qdrant Cloud (one-time):
```bash
python -m scripts.upload_to_qdrant
```

Run the app:
```bash
streamlit run app.py
```

---

## Project Structure

```
src/
├── ingestion/     # SEC EDGAR download + HTML parsing
├── processing/    # Document chunking
├── retrieval/     # Vector store, BM25, hybrid search, reranker
├── rag/           # LangChain pipeline + prompt
└── evaluation/    # RAGAS eval dataset + evaluator
scripts/
└── upload_to_qdrant.py   # One-time vector upload
results/
└── eval_metrics.csv      # Evaluation results across runs
app.py                    # Streamlit UI
```

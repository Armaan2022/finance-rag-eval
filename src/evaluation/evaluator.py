import csv
import time
from datetime import datetime
from pathlib import Path

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness, LLMContextPrecisionWithReference
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI

from src.evaluation.eval_dataset import EVAL_QUESTIONS
from src.rag.pipeline import ask, build_rag_chain

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"


def run_pipeline_on_dataset(
    retriever,
    pipeline_name: str,
    questions: list[dict],
) -> list[dict]:
    """
    Run the RAG pipeline over all eval questions and collect answers + contexts for RAGAS scoring.
    The contexts are the exact chunks the answer was generated from (one retrieval per question),
    so faithfulness/recall judge the answer against what the model actually read.
    """
    chain = build_rag_chain()
    results = []

    print(f"\n  Running {pipeline_name} ({len(questions)} questions)...")

    for i, item in enumerate(questions):
        question = item["question"]
        out = ask(chain, retriever, question)

        results.append({
            "question": question,
            "answer": out["answer"],
            "contexts": [doc.page_content for doc in out["docs"]],
            "ground_truth": item["ground_truth"],
            "pipeline": pipeline_name,
            "ticker": item["ticker"],
            "category": item["category"],
        })

        print(f"    [{i+1}/{len(questions)}] {question[:60]}...")
        time.sleep(0.5)  # gentle rate limiting for OpenAI API

    return results


def score_with_ragas(results: list[dict]) -> dict:
    """Score a set of pipeline results with RAGAS. Returns a dict of metric_name → avg float score."""
    data_list = []

    for r in results:
        item = {
            "question": r["question"],
            "answer": r["answer"],
            "contexts": r["contexts"],
            "ground_truth": r["ground_truth"],
        }
        data_list.append(item)

    dataset = Dataset.from_list(data_list)

    # RAGAS needs an LLM to judge quality and gpt-4o-mini is cheaper than gpt-4o.
    evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini", temperature=0))

    metrics = [
        LLMContextRecall(),
        Faithfulness(),
        FactualCorrectness(),
        LLMContextPrecisionWithReference(),
    ]

    scores = evaluate(dataset=dataset, metrics=metrics, llm=evaluator_llm)

    def avg(col):
        vals = scores.to_pandas()[col].dropna().tolist()
        # Some cells contain nested lists — flatten one level
        flat = []
        for v in vals:
            if isinstance(v, list):
                flat.extend(x for x in v if x is not None)
            else:
                flat.append(v)
        return sum(flat) / len(flat) if flat else 0.0

    return {
        "faithfulness": avg("faithfulness"),
        "factual_correctness": avg("factual_correctness(mode=f1)"),
        "context_precision": avg("llm_context_precision_with_reference"),
        "context_recall": avg("context_recall"),
    }


def save_results(all_scores: list[dict], run_timestamp: str) -> Path:
    """Append per-pipeline RAGAS scores to results/eval_metrics.csv"""
    RESULTS_DIR.mkdir(exist_ok=True)
    csv_path = RESULTS_DIR / "eval_metrics.csv"
    file_exists = csv_path.exists()

    fieldnames = [
        "timestamp", "pipeline",
        "faithfulness", "factual_correctness",
        "context_precision", "context_recall",
        "avg_score",
    ]

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        for row in all_scores:
            avg = (
                row["faithfulness"] + row["factual_correctness"] +
                row["context_precision"] + row["context_recall"]
            ) / 4
            writer.writerow({
                "timestamp": run_timestamp,
                "pipeline": row["pipeline"],
                "faithfulness": f"{row['faithfulness']:.4f}",
                "factual_correctness": f"{row['factual_correctness']:.4f}",
                "context_precision": f"{row['context_precision']:.4f}",
                "context_recall": f"{row['context_recall']:.4f}",
                "avg_score": f"{avg:.4f}",
            })

    print(f"\n  Results saved to {csv_path}")
    return csv_path


# Main: evaluate all 3 pipeline variants
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    from src.retrieval.qdrant_store import load_qdrant_store, get_retriever, get_chunks_from_qdrant
    from collections import Counter
    from src.retrieval.hybrid_search import BM25Index, HybridRetriever
    from src.retrieval.reranker import RERANK_TOP_N, RerankRetriever

    load_dotenv()

    run_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    QUESTIONS = [q for q in EVAL_QUESTIONS if q["ticker"] != "BOTH"]

    print("=== Loading pipeline components from Qdrant ===")
    chunks = get_chunks_from_qdrant()
    vector_store = load_qdrant_store()

    print(f"  Loaded {len(chunks)} chunks from Qdrant")

    # Scores are only comparable if the collection holds a single chunk size.
    chunk_configs = Counter(c.metadata.get("chunk_config", "unknown") for c in chunks)
    print(f"  Chunk configs in collection: {dict(chunk_configs)}")
    if len(chunk_configs) > 1:
        print("  WARNING: mixed chunk sizes in the collection; results will not be comparable.")

    # --- Define the 3 retrievers ---
    bm25_index = BM25Index(chunks)  # built once, shared by both hybrid pipelines
    vector_retriever = get_retriever(vector_store, k=5)
    hybrid_retriever = HybridRetriever(vector_store=vector_store, bm25_index=bm25_index, k=5, fetch_k=20)
    hybrid_rerank_retriever = RerankRetriever(
        base_retriever=HybridRetriever(vector_store=vector_store, bm25_index=bm25_index, k=20, fetch_k=20),
        top_n=RERANK_TOP_N,
    )

    all_scores = []

    # Pipeline 1: vector only
    print("\n=== Pipeline 1: Vector Only ===")
    v_results = run_pipeline_on_dataset(vector_retriever, "vector_only", QUESTIONS)
    print("  Scoring with RAGAS...")
    v_scores = score_with_ragas(v_results)
    all_scores.append({"pipeline": "vector_only", **v_scores})

    # Pipeline 2: hybrid
    print("\n=== Pipeline 2: Hybrid (BM25 + Vector + RRF) ===")
    h_results = run_pipeline_on_dataset(hybrid_retriever, "hybrid", QUESTIONS)
    print("  Scoring with RAGAS...")
    h_scores = score_with_ragas(h_results)
    all_scores.append({"pipeline": "hybrid", **h_scores})

    # Pipeline 3: hybrid + reranker
    print("\n=== Pipeline 3: Hybrid + Reranker ===")
    r_results = run_pipeline_on_dataset(hybrid_rerank_retriever, "hybrid_rerank", QUESTIONS)
    print("  Scoring with RAGAS...")
    r_scores = score_with_ragas(r_results)
    all_scores.append({"pipeline": "hybrid_rerank", **r_scores})

    # Save and print summary
    save_results(all_scores, run_timestamp)

    print("\n=== Results Summary ===")
    print(f"{'Pipeline':<20} {'Faithful':>9} {'Factual':>8} {'Precision':>10} {'Recall':>8} {'Avg':>7}")
    print("-" * 68)
    for row in all_scores:
        avg = sum([
            row["faithfulness"], row["factual_correctness"],
            row["context_precision"], row["context_recall"]
        ]) / 4
        print(
            f"{row['pipeline']:<20} "
            f"{row['faithfulness']:>9.4f} "
            f"{row['factual_correctness']:>8.4f} "
            f"{row['context_precision']:>10.4f} "
            f"{row['context_recall']:>8.4f} "
            f"{avg:>7.4f}"
        )

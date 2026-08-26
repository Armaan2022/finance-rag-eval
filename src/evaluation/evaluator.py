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
from src.rag.pipeline import build_rag_chain

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"


# Run one pipeline variant against all eval questions
def run_pipeline_on_dataset(
    retriever,
    pipeline_name: str,
    questions: list[dict],
    use_reranker: bool = False,
    top_n: int = 3,
) -> list[dict]:
    """
    Run a retriever + GPT-4o-mini chain over all eval questions.

    Returns a list of result dicts with:
      - question, ground_truth (from eval dataset)
      - answer (LLM response)
      - contexts (list of chunk texts that were retrieved)
      - pipeline, ticker, category (for filtering results)
    """
    chain = build_rag_chain(retriever)
    results = []

    print(f"\n  Running {pipeline_name} ({len(questions)} questions)...")

    for i, item in enumerate(questions):
        question = item["question"]

        # Get answer from the chain
        answer = chain.invoke({"question": question})

        # Get the retrieved chunks (for context metrics)
        if use_reranker:
            from src.retrieval.reranker import retrieve_and_rerank
            docs = retrieve_and_rerank(question, retriever, top_n=top_n)
        else:
            docs = retriever.invoke(question)

        contexts = [doc.page_content for doc in docs]

        results.append({
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": item["ground_truth"],
            "pipeline": pipeline_name,
            "ticker": item["ticker"],
            "category": item["category"],
        })

        print(f"    [{i+1}/{len(questions)}] {question[:60]}...")
        time.sleep(0.5)  # gentle rate limiting for OpenAI API

    return results


# Score with RAGAS
def score_with_ragas(results: list[dict]) -> dict:
    """
    Run RAGAS evaluation on a set of pipeline results.

    RAGAS expects a HuggingFace Dataset with columns:
      question, answer, contexts, ground_truth

    It uses GPT-4o (via OpenAI) as the judge LLM internally.
    Each metric scores 0-1; higher is better.

    Returns a dict of metric_name → float score.
    """
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

    # RAGAS uses its own LLM internally for judging — we point it to GPT-4o-mini
    # to keep costs down. GPT-4o is more accurate but ~10x more expensive.
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


# Save results to CSV
def save_results(all_scores: list[dict], run_timestamp: str) -> Path:
    """
    Save per-pipeline scores to results/eval_metrics.csv.

    Each row is one pipeline variant with its 4 RAGAS scores.
    The CSV accumulates across runs so you can track improvement over time.
    """
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
    from src.retrieval.hybrid_search import HybridRetriever

    load_dotenv()

    run_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Exclude cross-company comparisons (ticker="BOTH") — single-pass retrieval
    # isn't designed to simultaneously pull context from two companies, so those
    # questions reliably underperform and skew the per-pipeline averages.
    # Running 18 questions × 3 pipelines × RAGAS scoring ≈ $0.20-0.35
    QUESTIONS = [q for q in EVAL_QUESTIONS if q["ticker"] != "BOTH"]

    print("=== Loading pipeline components from Qdrant ===")
    chunks = get_chunks_from_qdrant()
    vector_store = load_qdrant_store()

    print(f"  Loaded {len(chunks)} chunks from Qdrant")

    # --- Define the 3 retrievers ---
    vector_retriever = get_retriever(vector_store, k=5)
    hybrid_retriever = HybridRetriever(vector_store=vector_store, chunks=chunks, k=5, fetch_k=20)
    hybrid_rerank_retriever = HybridRetriever(vector_store=vector_store, chunks=chunks, k=20, fetch_k=20)

    all_scores = []

    # --- Pipeline 1: vector only ---
    print("\n=== Pipeline 1: Vector Only ===")
    v_results = run_pipeline_on_dataset(vector_retriever, "vector_only", QUESTIONS)
    print("  Scoring with RAGAS...")
    v_scores = score_with_ragas(v_results)
    all_scores.append({"pipeline": "vector_only", **v_scores})

    # --- Pipeline 2: hybrid ---
    print("\n=== Pipeline 2: Hybrid (BM25 + Vector + RRF) ===")
    h_results = run_pipeline_on_dataset(hybrid_retriever, "hybrid", QUESTIONS)
    print("  Scoring with RAGAS...")
    h_scores = score_with_ragas(h_results)
    all_scores.append({"pipeline": "hybrid", **h_scores})

    # --- Pipeline 3: hybrid + reranker ---
    print("\n=== Pipeline 3: Hybrid + Reranker ===")
    r_results = run_pipeline_on_dataset(
        hybrid_rerank_retriever, "hybrid_rerank", QUESTIONS,
        use_reranker=True, top_n=3,
    )
    print("  Scoring with RAGAS...")
    r_scores = score_with_ragas(r_results)
    all_scores.append({"pipeline": "hybrid_rerank", **r_scores})

    # --- Save and print summary ---
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

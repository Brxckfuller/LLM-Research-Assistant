import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.retriever import Retriever
from src.adaptive_retrieval import adaptive_retrieve
from src.embeddings import embed_query


EVALUATION_FILE = (
    PROJECT_ROOT / "analysis" / "retrieval_eval_questions.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT / "analysis" / "retrieval_benchmark_results.csv"
)

TOP_K = 10
RAW_K = 32
REPEATS = 3

METHODS = ("bm25", "semantic", "hybrid", "adaptive")


def parse_pages(value):
    if pd.isna(value):
        return set()

    return {
        int(page.strip())
        for page in str(value).split(",")
        if page.strip()
    }


def calculate_metrics(retrieved_pages, expected_pages):
    if not expected_pages:
        raise ValueError("Expected pages cannot be empty")

    relevant_pages = set(retrieved_pages) & expected_pages

    hit_rate = float(bool(relevant_pages))
    recall = len(relevant_pages) / len(expected_pages)

    precision = (
        sum(page in expected_pages for page in retrieved_pages)
        / len(retrieved_pages)
        if retrieved_pages
        else 0.0
    )

    reciprocal_rank = 0.0

    for rank, page in enumerate(retrieved_pages, start=1):
        if page in expected_pages:
            reciprocal_rank = 1.0 / rank
            break

    return {
        "hit_rate_at_10": hit_rate,
        "recall_at_10": recall,
        "precision_at_10": precision,
        "mrr": reciprocal_rank,
    }


def run_method(method, question, document, retriever):
    if method == "bm25":
        return retriever.keyword_search(question, k=TOP_K)

    if method == "semantic":
        return retriever.semantic_search(question, k=TOP_K)

    if method == "hybrid":
        return retriever.hybrid_search(question, k=TOP_K)

    if method == "adaptive":
        response = adaptive_retrieve(
            question=question,
            document_name=document,
            top_k=TOP_K,
            raw_k=RAW_K,
        )
        return response["results"]

    raise ValueError(f"Unknown method: {method}")


def main():
    questions = pd.read_csv(EVALUATION_FILE)

    # Initialise document retrievers before timing.
    retriever_cache = {
        document: Retriever(document)
        for document in questions["document"].unique()
    }

    # Warm up the embedding model before timing retrieval.
    print("Warming up embedding model...")
    embed_query("Warm-up query for retrieval benchmarking.")

    all_results = []

    for _, row in questions.iterrows():
        question = row["question"]
        document = row["document"]
        expected_pages = parse_pages(row["expected_pages"])

        retriever = retriever_cache[document]

        for method in METHODS:
            latencies = []
            results = None

            for _ in range(REPEATS):
                start = time.perf_counter()

                results = run_method(
                    method,
                    question,
                    document,
                    retriever,
                )

                latency_ms = (
                    time.perf_counter() - start
                ) * 1000

                latencies.append(latency_ms)

            retrieved_pages = [
                int(result["page"])
                for result in results
                if result.get("page") is not None
            ][:TOP_K]

            metrics = calculate_metrics(
                retrieved_pages,
                expected_pages,
            )

            all_results.append({
                "question": question,
                "document": document,
                "method": method,
                "expected_pages": ",".join(
                    str(p) for p in sorted(expected_pages)
                ),
                "retrieved_pages": ",".join(
                    str(p) for p in retrieved_pages
                ),
                "latency_ms": sum(latencies) / len(latencies),
                **metrics,
            })

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUTPUT_FILE, index=False)

    summary = results_df.groupby("method")[
        [
            "hit_rate_at_10",
            "recall_at_10",
            "precision_at_10",
            "mrr",
            "latency_ms",
        ]
    ].mean()

    print("\nRetrieval Benchmark")
    print("=" * 75)
    print(summary.round(3).to_string())

    print(f"\nSaved detailed results to:\n{OUTPUT_FILE}")


if __name__ == "__main__":
    main()
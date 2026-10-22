
from pathlib import Path
import sys
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.adaptive_retrieval import adaptive_retrieve

EVAL_FILE = PROJECT_ROOT / "analysis" / "retrieval_eval_questions.csv"
OUTPUT_FILE = PROJECT_ROOT / "analysis" / "retrieval_eval_results.csv"
INDEX_DIR = PROJECT_ROOT / "data" / "indexes"

TOP_K = 10
RAW_K = 32


def available_documents():
    return {
        path.stem
        for path in INDEX_DIR.glob("*.faiss")
        if path.is_file()
    }


def parse_pages(value):
    if pd.isna(value):
        return set()

    return {
        int(page.strip())
        for page in str(value).split(",")
        if page.strip().isdigit()
    }


def calculate_metrics(retrieved_pages, expected_pages, k=TOP_K):
    if not expected_pages:
        raise ValueError("Expected pages cannot be empty.")

    top_pages = retrieved_pages[:k]
    retrieved_set = set(top_pages)

    relevant_pages_found = retrieved_set & expected_pages

    # Did we retrieve at least one relevant page?
    hit_rate = float(bool(relevant_pages_found))

    # What fraction of expected pages were retrieved?
    recall = len(relevant_pages_found) / len(expected_pages)

    # What fraction of retrieved passages came from relevant pages?
    relevant_results = sum(
        page in expected_pages for page in top_pages
    )
    precision = relevant_results / len(top_pages) if top_pages else 0.0

    # Rank of the first relevant retrieved passage.
    reciprocal_rank = 0.0

    for rank, page in enumerate(top_pages, start=1):
        if page in expected_pages:
            reciprocal_rank = 1.0 / rank
            break

    return {
        "hit_rate_at_10": hit_rate,
        "recall_at_10": recall,
        "precision_at_10": precision,
        "mrr": reciprocal_rank,
    }


def main():
    if not EVAL_FILE.exists():
        raise FileNotFoundError(f"Could not find {EVAL_FILE}")

    valid_docs = available_documents()
    df = pd.read_csv(EVAL_FILE)

    rows = []
    skipped = []

    for _, row in df.iterrows():
        question = row["question"]
        document = row["document"]
        expected_pages = parse_pages(row["expected_pages"])

        if not expected_pages:
            raise ValueError(
                f"No expected pages for question: {question}"
            )

        if document not in valid_docs:
            print(f"Skipping: '{document}' index not found.")
            skipped.append(document)
            continue

        retrieval_data = adaptive_retrieve(
            question=question,
            document_name=document,
            top_k=TOP_K,
            raw_k=RAW_K,
        )

        results = retrieval_data["results"]

        retrieved_pages = [
            int(result["page"])
            for result in results
            if str(result.get("page", "")).isdigit()
        ]

        metrics = calculate_metrics(
            retrieved_pages,
            expected_pages,
            k=TOP_K,
        )

        rows.append({
            "question": question,
            "document": document,
            "expected_pages": ",".join(
                map(str, sorted(expected_pages))
            ),
            "retrieved_pages": ",".join(
                map(str, retrieved_pages)
            ),
            **metrics,
        })

    if not rows:
        raise RuntimeError(
            "No questions were evaluated. Check available indexes."
        )

    results_df = pd.DataFrame(rows)
    results_df.to_csv(OUTPUT_FILE, index=False)

    print("\nRetrieval Evaluation")
    print("=" * 60)
    print(f"Questions evaluated: {len(results_df)}")
    print(f"Questions skipped: {len(skipped)}")

    for metric in [
        "hit_rate_at_10",
        "recall_at_10",
        "precision_at_10",
        "mrr",
    ]:
        print(f"{metric}: {results_df[metric].mean():.3f}")

    print("\nSaved results to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()

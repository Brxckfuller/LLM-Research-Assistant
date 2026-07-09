from pathlib import Path
import sys
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.adaptive_retrieval import adaptive_retrieve

EVAL_FILE = PROJECT_ROOT / "analysis" / "retrieval_eval_questions.csv"
OUTPUT_FILE = PROJECT_ROOT / "analysis" / "retrieval_eval_results.csv"
INDEX_DIR = PROJECT_ROOT / "data" / "indexes"

TOP_K = 10
RAW_K = 32


def available_documents():
    return {p.stem for p in INDEX_DIR.glob("*.faiss")}


def parse_pages(value):
    return {int(x.strip()) for x in str(value).split(",") if x.strip().isdigit()}


def reciprocal_rank(retrieved_pages, expected_pages):
    for rank, page in enumerate(retrieved_pages, start=1):
        if page in expected_pages:
            return 1 / rank
    return 0.0


def main():
    if not EVAL_FILE.exists():
        raise FileNotFoundError(f"Could not find {EVAL_FILE}")

    valid_docs = available_documents()
    df = pd.read_csv(EVAL_FILE)

    rows = []

    for _, row in df.iterrows():
        question = row["question"]
        document = row["document"]
        expected_pages = parse_pages(row["expected_pages"])

        if document not in valid_docs:
            print(f"Skipping: '{document}' index not found.")
            print(f"Available documents include: {sorted(valid_docs)}")
            continue

        retrieval_data = adaptive_retrieve(
            question=question,
            document_name=document,
            top_k=TOP_K,
            raw_k=RAW_K,
        )

        results = retrieval_data["results"]

        retrieved_pages = [
            int(r["page"])
            for r in results
            if str(r.get("page", "")).isdigit()
        ]

        retrieved_set = set(retrieved_pages)
        hits = retrieved_set.intersection(expected_pages)

        recall_at_10 = 1.0 if hits else 0.0
        precision_at_10 = len(hits) / TOP_K
        mrr = reciprocal_rank(retrieved_pages, expected_pages)

        rows.append(
            {
                "question": question,
                "document": document,
                "expected_pages": ",".join(map(str, sorted(expected_pages))),
                "retrieved_pages": ",".join(map(str, retrieved_pages)),
                "hit": bool(hits),
                "recall_at_10": recall_at_10,
                "precision_at_10": precision_at_10,
                "mrr": mrr,
            }
        )

    results_df = pd.DataFrame(rows)
    results_df.to_csv(OUTPUT_FILE, index=False)

    print("\nRetrieval Evaluation")
    print("=" * 60)
    print(f"Questions evaluated: {len(results_df)}")
    print(f"Recall@10: {results_df['recall_at_10'].mean():.3f}")
    print(f"Precision@10: {results_df['precision_at_10'].mean():.3f}")
    print(f"MRR: {results_df['mrr'].mean():.3f}")

    print("\nSaved results to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()
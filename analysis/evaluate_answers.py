import argparse
import csv
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from api.rag_service import answer_question

INPUT_FILE = PROJECT_ROOT / "analysis" / "answer_eval_questions.csv"
OUTPUT_FILE = PROJECT_ROOT / "analysis" / "answer_eval_results.csv"

ABSTENTION_PHRASES = (
    "i don't know",
    "i do not know",
    "cannot determine",
    "can't determine",
    "not enough information",
    "insufficient information",
    "not provided",
    "not mentioned",
    "not specified",
    "not in the document",
    "not in the provided",
    "cannot answer",
    "can't answer",
)


def keyword_coverage(answer, expected_keywords):
    keywords = [
        keyword.strip().lower()
        for keyword in expected_keywords.split(";")
        if keyword.strip()
    ]

    if not keywords:
        return None

    answer_words = set(re.findall(r"\b\w+\b", answer.lower()))

    matched = sum(keyword in answer_words for keyword in keywords)
    return round(matched / len(keywords), 3)


def detected_abstention(answer):
    answer_lower = answer.lower()
    return any(
        phrase in answer_lower
        for phrase in ABSTENTION_PHRASES
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of questions to evaluate",
    )
    args = parser.parse_args()

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")

    with INPUT_FILE.open(newline="", encoding="utf-8") as file:
        questions = list(csv.DictReader(file))

    if args.limit is not None:
        questions = questions[:args.limit]

    results = []

    for row in questions:
        question = row["question"]
        document = row["document"]
        answerable = row["answerable"].strip().lower() == "true"

        print(f"\nEvaluating: {question}", flush=True)

        try:
            response = answer_question(
                question=question,
                document_name=document,
                top_k=6,
            )

            answer = response.answer
            abstained = detected_abstention(answer)

            results.append({
                "question_id": row["question_id"],
                "document": document,
                "question": question,
                "answerable": answerable,
                "answer": answer,
                "keyword_coverage": keyword_coverage(
                    answer,
                    row["expected_keywords"],
                ),
                "detected_abstention": abstained,
                "abstention_expected": not answerable,
                "abstention_match": abstained == (not answerable),
                "source_count": len(response.sources),
                "source_pages": ";".join(
                    str(source.page)
                    for source in response.sources
                    if source.page is not None
                ),
                "retrieval_seconds": response.retrieval_time_seconds,
                "generation_seconds": response.generation_time_seconds,
                "error": "",
            })

        except Exception as exc:
            results.append({
                "question_id": row["question_id"],
                "document": document,
                "question": question,
                "answerable": answerable,
                "answer": "",
                "keyword_coverage": "",
                "detected_abstention": "",
                "abstention_expected": not answerable,
                "abstention_match": "",
                "source_count": "",
                "source_pages": "",
                "retrieval_seconds": "",
                "generation_seconds": "",
                "error": f"{type(exc).__name__}: {exc}",
            })

            print(f"Error: {exc}", flush=True)

    if not results:
        raise RuntimeError("No evaluation questions found.")

    with OUTPUT_FILE.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\nEvaluated {len(results)} question(s).")
    print(f"Results saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

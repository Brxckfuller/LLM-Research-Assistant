import argparse
import csv
from datetime import datetime
from uuid import uuid4
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from api.rag_service import answer_question
from analysis.evidence_store import save_evidence
from analysis.answer_classifier import AnswerClassification, classify_answer

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
    "do not provide information",
    "does not provide information",
    "no information on",
    "no direct evidence",
    "not directly supported",
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
    parser.add_argument(
        "--question-id",
        type=str,
        default=None,
        help="Evaluate a specific question by its ID",
    )
    args = parser.parse_args()

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")

    with INPUT_FILE.open(newline="", encoding="utf-8") as file:
        questions = list(csv.DictReader(file))

    if args.question_id is not None:
        questions = [
            row for row in questions
            if row["question_id"] == args.question_id
        ]
        if not questions:
            parser.error(
                f"Question ID {args.question_id} was not found"
            )

    if args.limit is not None:
        questions = questions[:args.limit]

    results = []

    run_id = (
        datetime.now().strftime("%Y%m%d-%H%M%S")
        + "-"
        + uuid4().hex[:8]
    )
    evidence_path = (
        PROJECT_ROOT
        / "analysis"
        / "local_evidence"
        / f"evidence_{run_id}.jsonl"
    )
    results_path = (
        PROJECT_ROOT
        / "analysis"
        / "local_evidence"
        / f"results_{run_id}.csv"
    )

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
            classification = classify_answer(answer)

            save_evidence(
                question_id=row["question_id"],
                question=question,
                response=response,
                output_path=evidence_path,
            )

            classification_match = (
                classification == AnswerClassification.ANSWERED
                if answerable
                else classification == AnswerClassification.ABSTAINED
            )

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
                "answer_classification": classification.value,
                "classification_match": classification_match,
                "manual_review_required": (
                    classification == AnswerClassification.MIXED
                ),
                "abstention_expected": not answerable,
                "abstention_match": abstained == (not answerable),
                "evidence_file": str(evidence_path),
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
                "answer_classification": "",
                "classification_match": "",
                "manual_review_required": True,
                "abstention_expected": not answerable,
                "abstention_match": "",
                "evidence_file": "",
                "source_count": "",
                "source_pages": "",
                "retrieval_seconds": "",
                "generation_seconds": "",
                "error": f"{type(exc).__name__}: {exc}",
            })

            print(f"Error: {exc}", flush=True)

    if not results:
        raise RuntimeError("No evaluation questions found.")

    output_path = (
        OUTPUT_FILE
        if OUTPUT_FILE != PROJECT_ROOT / "analysis" / "answer_eval_results.csv"
        else results_path
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\nEvaluated {len(results)} question(s).")
    print(f"Results saved to: {output_path}")
    print(f"Evidence saved to: {evidence_path}")


if __name__ == "__main__":
    main()

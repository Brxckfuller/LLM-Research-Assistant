import csv
import sys
from types import SimpleNamespace

from analysis import evaluate_answers


def test_question_id_filters_evaluation(tmp_path, monkeypatch):
    questions = tmp_path / "questions.csv"
    results = tmp_path / "results.csv"

    with questions.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "question_id",
                "document",
                "question",
                "expected_keywords",
                "answerable",
            ],
        )
        writer.writeheader()
        writer.writerow({
            "question_id": "1",
            "document": "demo",
            "question": "What is RAG?",
            "expected_keywords": "retrieval",
            "answerable": "true",
        })
        writer.writerow({
            "question_id": "2",
            "document": "demo",
            "question": "What is the capital of France?",
            "expected_keywords": "",
            "answerable": "false",
        })

    calls = []

    def fake_answer_question(question, document_name, top_k):
        calls.append(question)
        return SimpleNamespace(
            answer="The document does not provide information on this.",
            sources=[],
            retrieval_time_seconds=0.1,
            generation_time_seconds=0.2,
        )

    monkeypatch.setattr(evaluate_answers, "INPUT_FILE", questions)
    monkeypatch.setattr(evaluate_answers, "OUTPUT_FILE", results)
    monkeypatch.setattr(
        evaluate_answers, "answer_question", fake_answer_question
    )
    monkeypatch.setattr(
        sys, "argv", ["evaluate_answers.py", "--question-id", "2"]
    )

    evaluate_answers.main()

    with results.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert calls == ["What is the capital of France?"]
    assert len(rows) == 1
    assert rows[0]["question_id"] == "2"
    assert rows[0]["answer_classification"] == "ABSTAINED"
    assert rows[0]["classification_match"] == "True"
    assert rows[0]["manual_review_required"] == "False"


def test_mixed_response_requires_manual_review(tmp_path, monkeypatch):
    questions = tmp_path / "questions.csv"
    results = tmp_path / "results.csv"

    questions.write_text(
        "question_id,document,question,expected_keywords,answerable\n"
        "6,demo,What is the capital of France?,,false\n",
        encoding="utf-8",
    )

    def fake_answer_question(question, document_name, top_k):
        return SimpleNamespace(
            answer=(
                "The document does not provide information "
                "about France, but the answer is Paris."
            ),
            sources=[],
            retrieval_time_seconds=0.1,
            generation_time_seconds=0.2,
        )

    monkeypatch.setattr(evaluate_answers, "INPUT_FILE", questions)
    monkeypatch.setattr(evaluate_answers, "OUTPUT_FILE", results)
    monkeypatch.setattr(
        evaluate_answers, "answer_question", fake_answer_question
    )
    monkeypatch.setattr(sys, "argv", ["evaluate_answers.py"])

    evaluate_answers.main()

    with results.open(newline="", encoding="utf-8") as f:
        row = next(csv.DictReader(f))

    assert row["answer_classification"] == "MIXED"
    assert row["classification_match"] == "False"
    assert row["manual_review_required"] == "True"

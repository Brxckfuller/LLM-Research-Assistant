import json
from types import SimpleNamespace

from analysis.evidence_store import save_evidence


def make_response():
    return SimpleNamespace(
        document_name="demo",
        sources=[
            SimpleNamespace(
                document="demo",
                page=1,
                text="Semantic retrieval uses meaning.",
            )
        ],
    )


def test_saves_evidence(tmp_path):
    output = tmp_path / "evidence.jsonl"

    save_evidence(
        question_id="1",
        question="What is semantic retrieval?",
        response=make_response(),
        output_path=output,
    )

    records = [
        json.loads(line)
        for line in output.read_text().splitlines()
    ]

    assert len(records) == 1
    assert records[0]["question_id"] == "1"
    assert records[0]["document"] == "demo"
    assert records[0]["sources"][0]["page"] == 1
    assert records[0]["sources"][0]["text"] == (
        "Semantic retrieval uses meaning."
    )


def test_appends_multiple_records(tmp_path):
    output = tmp_path / "evidence.jsonl"

    for question_id in ("1", "2"):
        save_evidence(
            question_id=question_id,
            question="Test question",
            response=make_response(),
            output_path=output,
        )

    records = [
        json.loads(line)
        for line in output.read_text().splitlines()
    ]

    assert len(records) == 2
    assert [r["question_id"] for r in records] == ["1", "2"]

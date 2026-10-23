import json
from pathlib import Path


def save_evidence(question_id, question, response, output_path):
    """Append retrieved source passages to a local JSONL file."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "question_id": str(question_id),
        "question": question,
        "document": response.document_name,
        "sources": [
            {
                "document": source.document,
                "page": source.page,
                "text": source.text,
            }
            for source in response.sources
        ],
    }

    with output_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")

    return record

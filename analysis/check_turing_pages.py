from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.adaptive_retrieval import adaptive_retrieve

questions = [
    "What is Lady Lovelace's objection?",
    "What objections to machine intelligence does Turing discuss?",
]

for question in questions:
    print("\n" + "=" * 80)
    print("QUESTION:", question)
    print("=" * 80)

    data = adaptive_retrieve(
        question=question,
        document_name="turing",
        top_k=10,
        raw_k=32,
    )

    for rank, result in enumerate(data["results"], start=1):
        page = result.get("page", "unknown")
        text = str(result.get("text", ""))

        print(f"\nRANK {rank} | PAGE {page}")
        print(text[:450].replace("\n", " "))
from unittest.mock import MagicMock

import numpy as np

from src.retriever import Retriever


def test_hybrid_retrieval_quality_gate(monkeypatch):
    chunks = [
        {
            "chunk_id": 0,
            "page": 1,
            "text": "Machine learning models learn patterns from data.",
            "section": "Introduction",
            "total_pages": 4,
        },
        {
            "chunk_id": 1,
            "page": 2,
            "text": "Neural networks use layers of interconnected neurons.",
            "section": "Methods",
            "total_pages": 4,
        },
        {
            "chunk_id": 2,
            "page": 3,
            "text": "Reinforcement learning agents learn through rewards.",
            "section": "Results",
            "total_pages": 4,
        },
        {
            "chunk_id": 3,
            "page": 4,
            "text": "References and bibliography.",
            "section": "References",
            "total_pages": 4,
            "is_references": True,
        },
    ]

    # Question -> expected relevant chunk and simulated search signals
    cases = [
        {
            "question": "How do machine learning models learn?",
            "expected_chunk": 0,
            "semantic_scores": [0.95, 0.40, 0.20, 0.05],
            "bm25_scores": [4.0, 1.0, 0.5, 0.0],
        },
        {
            "question": "What are neural networks?",
            "expected_chunk": 1,
            "semantic_scores": [0.35, 0.95, 0.25, 0.05],
            "bm25_scores": [0.5, 4.0, 0.5, 0.0],
        },
        {
            "question": "How do reinforcement learning agents learn?",
            "expected_chunk": 2,
            "semantic_scores": [0.25, 0.30, 0.95, 0.05],
            "bm25_scores": [0.5, 0.5, 4.0, 0.0],
        },
    ]

    fake_index = MagicMock()
    fake_bm25 = MagicMock()

    monkeypatch.setattr(
        "src.retriever.faiss.read_index",
        lambda path: fake_index,
    )

    def fake_joblib_load(path):
        if str(path).endswith("_chunks.joblib"):
            return chunks
        if str(path).endswith("_bm25.joblib"):
            return fake_bm25
        raise ValueError(f"Unexpected file: {path}")

    monkeypatch.setattr(
        "src.retriever.joblib.load",
        fake_joblib_load,
    )

    monkeypatch.setattr(
        "src.retriever.embed_query",
        lambda query: np.array([0.1, 0.2, 0.3]),
    )

    retriever = Retriever("synthetic_document")

    hits = 0
    reciprocal_ranks = []

    for case in cases:
        scores = np.array(case["semantic_scores"], dtype=np.float32)
        ranked_indices = np.argsort(scores)[::-1]

        fake_index.search.return_value = (
            scores[ranked_indices].reshape(1, -1),
            ranked_indices.reshape(1, -1).astype(np.int64),
        )

        fake_bm25.get_scores.return_value = np.array(
            case["bm25_scores"], dtype=np.float32
        )

        results = retriever.hybrid_search(
            case["question"],
            k=3,
        )

        retrieved_ids = [result["chunk_id"] for result in results]
        expected_id = case["expected_chunk"]

        if expected_id in retrieved_ids:
            hits += 1
            rank = retrieved_ids.index(expected_id) + 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)

    hit_rate = hits / len(cases)
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)

    print(f"\nSynthetic Hit Rate@3: {hit_rate:.3f}")
    print(f"Synthetic MRR: {mrr:.3f}")

    assert hit_rate >= 1.0, (
        f"Retrieval quality regression: Hit Rate@3={hit_rate:.3f}"
    )

    assert mrr >= 0.90, (
        f"Retrieval quality regression: MRR={mrr:.3f}"
    )
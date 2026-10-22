from unittest.mock import MagicMock

import numpy as np

from src.retriever import Retriever


def test_hybrid_search_returns_relevant_chunks(monkeypatch):
    # Simulated document chunks
    chunks = [
        {
            "chunk_id": 0,
            "page": 1,
            "text": "Artificial intelligence uses machine learning.",
            "section": "Introduction",
            "total_pages": 3,
        },
        {
            "chunk_id": 1,
            "page": 2,
            "text": "Neural networks learn patterns from data.",
            "section": "Methods",
            "total_pages": 3,
        },
        {
            "chunk_id": 2,
            "page": 3,
            "text": "References and bibliography.",
            "section": "References",
            "total_pages": 3,
            "is_references": True,
        },
    ]

    # Fake FAISS search results
    fake_index = MagicMock()
    fake_index.search.return_value = (
        np.array([[0.95, 0.80, 0.20]], dtype=np.float32),
        np.array([[0, 1, 2]], dtype=np.int64),
    )

    # Fake BM25 scores
    fake_bm25 = MagicMock()
    fake_bm25.get_scores.return_value = np.array(
        [3.0, 2.0, 0.1]
    )

    # Replace external dependencies with deterministic values
    monkeypatch.setattr(
        "src.retriever.faiss.read_index",
        lambda path: fake_index,
    )

    def fake_joblib_load(path):
        if str(path).endswith("_chunks.joblib"):
            return chunks
        if str(path).endswith("_bm25.joblib"):
            return fake_bm25
        raise ValueError(f"Unexpected metadata file: {path}")

    monkeypatch.setattr(
        "src.retriever.joblib.load",
        fake_joblib_load,
    )

    monkeypatch.setattr(
        "src.retriever.embed_query",
        lambda query: np.array([0.1, 0.2, 0.3]),
    )

    retriever = Retriever("test_document")

    results = retriever.hybrid_search(
        "machine learning",
        k=3,
    )

    assert isinstance(results, list)
    assert len(results) > 0

    for result in results:
        assert "chunk_id" in result
        assert "page" in result
        assert "text" in result
        assert "hybrid_score" in result

    # Reference material should not outrank the strongest result
    assert results[0]["chunk_id"] != 2
import pytest

from analysis.evaluate_retrieval import calculate_metrics


def test_perfect_retrieval():
    metrics = calculate_metrics(
        retrieved_pages=[1, 2, 3],
        expected_pages={1, 2, 3},
        k=3,
    )

    assert metrics["hit_rate_at_10"] == 1.0
    assert metrics["recall_at_10"] == 1.0
    assert metrics["precision_at_10"] == 1.0
    assert metrics["mrr"] == 1.0


def test_partial_recall():
    metrics = calculate_metrics(
        retrieved_pages=[5, 2, 7, 9],
        expected_pages={2, 3},
        k=4,
    )

    assert metrics["hit_rate_at_10"] == 1.0
    assert metrics["recall_at_10"] == 0.5
    assert metrics["precision_at_10"] == 0.25
    assert metrics["mrr"] == 0.5


def test_no_relevant_results():
    metrics = calculate_metrics(
        retrieved_pages=[5, 6, 7],
        expected_pages={1, 2},
        k=3,
    )

    assert metrics["hit_rate_at_10"] == 0.0
    assert metrics["recall_at_10"] == 0.0
    assert metrics["precision_at_10"] == 0.0
    assert metrics["mrr"] == 0.0


def test_duplicate_pages():
    metrics = calculate_metrics(
        retrieved_pages=[2, 2, 3, 5],
        expected_pages={2, 3},
        k=4,
    )

    assert metrics["hit_rate_at_10"] == 1.0
    assert metrics["recall_at_10"] == 1.0
    assert metrics["precision_at_10"] == 0.75
    assert metrics["mrr"] == 1.0


def test_empty_retrieval():
    metrics = calculate_metrics(
        retrieved_pages=[],
        expected_pages={1, 2},
        k=10,
    )

    assert metrics["hit_rate_at_10"] == 0.0
    assert metrics["recall_at_10"] == 0.0
    assert metrics["precision_at_10"] == 0.0
    assert metrics["mrr"] == 0.0


def test_empty_expected_pages():
    with pytest.raises(ValueError):
        calculate_metrics(
            retrieved_pages=[1, 2],
            expected_pages=set(),
            k=10,
        )


def test_top_k_cutoff():
    metrics = calculate_metrics(
        retrieved_pages=[5, 6, 7, 1],
        expected_pages={1},
        k=3,
    )

    assert metrics["hit_rate_at_10"] == 0.0
    assert metrics["recall_at_10"] == 0.0
    assert metrics["precision_at_10"] == 0.0
    assert metrics["mrr"] == 0.0
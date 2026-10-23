from analysis.evaluate_answers import (
    detected_abstention,
    keyword_coverage,
)


def test_detects_missing_information():
    answer = (
        "The retrieved passages do not provide information "
        "on the capital of France."
    )
    assert detected_abstention(answer) is True


def test_normal_answer_is_not_abstention():
    answer = "RAG stands for Retrieval-Augmented Generation."
    assert detected_abstention(answer) is False


def test_detects_uncertainty():
    answer = "I cannot determine the answer from the document."
    assert detected_abstention(answer) is True


def test_keyword_coverage():
    answer = "RAG means Retrieval-Augmented Generation."
    keywords = "retrieval;augmented;generation"

    assert keyword_coverage(answer, keywords) == 1.0


def test_mixed_response_exposes_detector_limitation():
    answer = (
        "The document does not provide information "
        "about the capital of France, but the answer is Paris."
    )

    # This currently returns True even though the model
    # supplies an unsupported answer.
    # A future evidence-grounding evaluator must detect this.
    assert detected_abstention(answer) is True

from analysis.answer_classifier import (
    AnswerClassification,
    classify_answer,
)


def test_normal_answer():
    answer = "RAG stands for Retrieval-Augmented Generation."
    assert classify_answer(answer) == AnswerClassification.ANSWERED


def test_successful_abstention():
    answer = (
        "The retrieved passages do not provide information "
        "on the capital of France."
    )
    assert classify_answer(answer) == AnswerClassification.ABSTAINED


def test_cannot_determine():
    answer = "I cannot determine the answer from the document."
    assert classify_answer(answer) == AnswerClassification.ABSTAINED


def test_mixed_unsupported_answer():
    answer = (
        "The document does not provide information "
        "about the capital of France, but the answer is Paris."
    )
    assert classify_answer(answer) == AnswerClassification.MIXED


def test_rejects_non_string_input():
    import pytest

    with pytest.raises(TypeError):
        classify_answer(None)

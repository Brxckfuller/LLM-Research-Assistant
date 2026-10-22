import pytest

from fastapi.testclient import TestClient

from api.main import app
from api.schemas import QuestionResponse
from api import main
from api.rag_service import (
    validate_document_name,
    DocumentNotFoundError,
)


client = TestClient(app)


# --------------------------------------------------
# 1. Health Check
# --------------------------------------------------

def test_health_check():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


# --------------------------------------------------
# 2. Input Validation
# --------------------------------------------------

def test_empty_question_rejected():
    response = client.post(
        "/questions",
        json={
            "question": "",
            "document_name": "chalmers1995",
            "top_k": 6,
        },
    )

    assert response.status_code == 422


def test_invalid_top_k_rejected():
    response = client.post(
        "/questions",
        json={
            "question": "What is consciousness?",
            "document_name": "chalmers1995",
            "top_k": 100,
        },
    )

    assert response.status_code == 422


# --------------------------------------------------
# 3. Error Handling
# --------------------------------------------------

def test_missing_document_returns_404(monkeypatch):
    def fake_answer_question(**kwargs):
        raise main.DocumentNotFoundError("missing_document")

    monkeypatch.setattr(main, "answer_question", fake_answer_question)

    response = client.post(
        "/questions",
        json={
            "question": "What is this paper about?",
            "document_name": "missing_document",
            "top_k": 6,
        },
    )

    assert response.status_code == 404


def test_inference_failure_returns_503(monkeypatch):
    def fake_answer_question(**kwargs):
        raise main.InferenceError("Ollama unavailable")

    monkeypatch.setattr(main, "answer_question", fake_answer_question)

    response = client.post(
        "/questions",
        json={
            "question": "What is consciousness?",
            "document_name": "chalmers1995",
            "top_k": 6,
        },
    )

    assert response.status_code == 503


# --------------------------------------------------
# 4. Successful Question Response
# --------------------------------------------------

def test_successful_question_response(monkeypatch):
    def fake_answer_question(**kwargs):
        return QuestionResponse(
            answer="The paper discusses consciousness.",
            document_name="chalmers1995",
            question_type="definition",
            coverage_score=80,
            sources=[
                {
                    "document": "chalmers1995",
                    "page": 1,
                    "text": "Consciousness presents a hard problem.",
                }
            ],
            retrieval_time_seconds=0.5,
            generation_time_seconds=1.2,
        )

    monkeypatch.setattr(main, "answer_question", fake_answer_question)

    response = client.post(
        "/questions",
        json={
            "question": "What is consciousness?",
            "document_name": "chalmers1995",
            "top_k": 6,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["document_name"] == "chalmers1995"
    assert data["coverage_score"] == 80
    assert len(data["sources"]) == 1
    assert data["answer"] == "The paper discusses consciousness."


# --------------------------------------------------
# 5. Document Name Security Validation
# --------------------------------------------------

@pytest.mark.parametrize(
    "document_name",
    [
        "../secret",
        "../../private",
        "folder/document",
        r"folder\document",
        "",
    ],
)
def test_invalid_document_names_rejected(document_name):
    with pytest.raises(DocumentNotFoundError):
        validate_document_name(document_name)


def test_unknown_document_rejected():
    with pytest.raises(DocumentNotFoundError):
        validate_document_name("nonexistent_paper_12345")


def test_existing_document_accepted():
    validate_document_name("chalmers1995")
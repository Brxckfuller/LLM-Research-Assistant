import logging

from fastapi import FastAPI, HTTPException

from api.schemas import QuestionRequest, QuestionResponse
from api.rag_service import (
    answer_question,
    DocumentNotFoundError,
    InferenceError,
)

rag_logger = logging.getLogger("api.rag_service")
rag_logger.setLevel(logging.INFO)

if not rag_logger.handlers:
    rag_logger.addHandler(logging.StreamHandler())
app = FastAPI(
    title="Research Paper Assistant API",
    description="Production-oriented API for a RAG research assistant.",
    version="0.2.0",
)


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "research-paper-assistant",
    }


@app.post("/questions", response_model=QuestionResponse)
def ask_question(request: QuestionRequest):
    try:
        return answer_question(
            question=request.question,
            document_name=request.document_name,
            top_k=request.top_k,
        )

    except DocumentNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="Document index not found.",
        )

    except InferenceError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        )
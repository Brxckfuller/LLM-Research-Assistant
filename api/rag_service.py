import logging
import time
from pathlib import Path
import requests

from src.adaptive_retrieval import adaptive_retrieve
from src.qa_engine import build_fast_answer_prompt
from src.llm_client import generate_answer

from api.schemas import QuestionResponse, SourceReference

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
INDEX_DIR = PROJECT_ROOT / "data" / "indexes"
METADATA_DIR = PROJECT_ROOT / "data" / "metadata"


class DocumentNotFoundError(Exception):
    pass


class InferenceError(Exception):
    pass

def validate_document_name(document_name: str) -> None:
    """
    Ensure the document name refers to an existing indexed paper
    and cannot be used to access arbitrary filesystem paths.
    """
    if (
        not document_name
        or document_name in {".", ".."}
        or "/" in document_name
        or "\\" in document_name
        or "\x00" in document_name
    ):
        raise DocumentNotFoundError(document_name)

    available_documents = {
        path.stem
        for path in INDEX_DIR.glob("*.faiss")
        if path.is_file()
    }

    if document_name not in available_documents:
        raise DocumentNotFoundError(document_name)

def answer_question(
    question: str,
    document_name: str,
    top_k: int = 6,
) -> QuestionResponse:

    request_start = time.perf_counter()
    validate_document_name(document_name)

    # Validate against existing index files.
    index_path = INDEX_DIR / f"{document_name}.faiss"
    chunks_path = METADATA_DIR / f"{document_name}_chunks.joblib"
    bm25_path = METADATA_DIR / f"{document_name}_bm25.joblib"

    if not all(
        path.is_file()
        for path in (index_path, chunks_path, bm25_path)
    ):
        raise DocumentNotFoundError(document_name)

    # Stage 1: Adaptive retrieval.
    retrieval_start = time.perf_counter()

    retrieval_data = adaptive_retrieve(
        question=question,
        document_name=document_name,
        top_k=top_k,
    )

    retrieval_time = time.perf_counter() - retrieval_start

    results = retrieval_data["results"]

    if not results:
        raise InferenceError(
            "No relevant passages were retrieved."
        )

    # Stage 2: Construct the existing Fast Mode prompt.
    prompt = build_fast_answer_prompt(question, results)

    # Stage 3: Generate an answer using the configured provider.
    generation_start = time.perf_counter()

    try:
        answer = generate_answer(prompt)
    except (requests.RequestException, RuntimeError, ValueError, KeyError, IndexError, TypeError) as exc:
        raise InferenceError("LLM inference failed.") from exc

    generation_time = time.perf_counter() - generation_start

    if (
            not answer
            or answer.startswith("Error calling Ollama:")
            or answer.startswith("Ollama is not running.")
    ):
        raise InferenceError(
            "LLM returned an empty response or inference failed."
        )

    # Stage 4: Convert retrieved chunks into structured references.
    sources = [
        SourceReference(
            document=chunk.get("document", document_name),
            page=chunk.get("page"),
            text=chunk.get("text", ""),
        )
        for chunk in results
    ]
    total_time = time.perf_counter() - request_start

    logger.info(
        "rag_request_completed "
        "document=%s "
        "question_type=%s "
        "retrieval_seconds=%.3f "
        "generation_seconds=%.3f "
        "total_seconds=%.3f "
        "source_count=%d "
        "coverage_score=%d",
        document_name,
        retrieval_data["question_type"],
        retrieval_time,
        generation_time,
        total_time,
        len(sources),
        retrieval_data["coverage_score"],
    )
    return QuestionResponse(
        answer=answer,
        document_name=document_name,
        question_type=retrieval_data["question_type"],
        coverage_score=retrieval_data["coverage_score"],
        sources=sources,
        retrieval_time_seconds=round(retrieval_time, 3),
        generation_time_seconds=round(generation_time, 3),
    )
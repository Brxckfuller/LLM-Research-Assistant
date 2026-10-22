from pydantic import BaseModel, Field


class QuestionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    document_name: str = Field(min_length=1)
    top_k: int = Field(default=6, ge=1, le=20)


class SourceReference(BaseModel):
    document: str
    page: int | None = None
    text: str


class QuestionResponse(BaseModel):
    answer: str
    document_name: str
    question_type: str
    coverage_score: int
    sources: list[SourceReference]
    retrieval_time_seconds: float
    generation_time_seconds: float
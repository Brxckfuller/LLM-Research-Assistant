from analysis.citation_validator import validate_citations


SOURCES = [
    {"document": "rag_demo", "page": 1, "text": "RAG combines retrieval and generation."},
    {"document": "rag_demo", "page": 2, "text": "Evaluation requires citation accuracy."},
    {"document": "rag_demo", "page": 1, "text": "Grounded answers use source context."},
]


def test_valid_citations():
    answer = (
        "RAG combines retrieval and generation (Source 1, Page 1). "
        "Answers should be grounded (Source 3, Page 1)."
    )
    result = validate_citations(answer, SOURCES)
    assert result["citation_count"] == 2
    assert result["valid_citation_count"] == 2
    assert result["all_citations_valid"]


def test_nonexistent_source():
    result = validate_citations("Claim (Source 4, Page 1).", SOURCES)
    assert not result["all_citations_valid"]
    assert result["invalid_citations"][0]["reason"] == "source_not_found"


def test_wrong_page():
    result = validate_citations("Claim (Source 2, Page 1).", SOURCES)
    assert not result["all_citations_valid"]
    assert result["invalid_citations"][0]["reason"] == "page_mismatch"


def test_no_citations():
    result = validate_citations(
        "The documents do not contain enough information.",
        SOURCES,
    )
    assert result["citation_count"] == 0
    assert result["all_citations_valid"]

"""Validate citation references against saved retrieved evidence."""

import re


CITATION_PATTERN = re.compile(
    r"\(Source\s+(\d+),\s*Page\s+(\d+)\)",
    re.IGNORECASE,
)


def validate_citations(answer, sources):
    """Check source numbers and page numbers against retrieved passages."""
    citations = CITATION_PATTERN.findall(answer)

    invalid = []
    for source_number, page_number in citations:
        source_index = int(source_number) - 1

        if not 0 <= source_index < len(sources):
            invalid.append({
                "source": int(source_number),
                "page": int(page_number),
                "reason": "source_not_found",
            })
            continue

        source = sources[source_index]
        actual_page = source.get("page")

        if str(actual_page) != page_number:
            invalid.append({
                "source": int(source_number),
                "page": int(page_number),
                "reason": "page_mismatch",
            })

    return {
        "citation_count": len(citations),
        "valid_citation_count": len(citations) - len(invalid),
        "invalid_citations": invalid,
        "all_citations_valid": len(invalid) == 0,
    }

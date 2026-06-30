from typing import Dict, List, Tuple

from src.retriever import Retriever
from src.reranker import rerank_results
from src.query_planner import classify_question, build_retrieval_queries


UNIVERSAL_ARGUMENT_MARKERS = [
    "this paper argues",
    "this article argues",
    "this essay argues",
    "we argue",
    "i argue",
    "the argument is",
    "central argument",
    "main argument",
    "the thesis",
    "our thesis",
    "my thesis",
    "we claim",
    "i claim",
    "central claim",
    "main claim",
    "we show",
    "i show",
    "we demonstrate",
    "we find",
    "our results show",
    "i suggest",
    "we suggest",
    "i propose",
    "we propose",
    "i defend",
    "we defend",
    "i reject",
    "we reject",
    "i conclude",
    "we conclude",
    "in conclusion",
    "to conclude",
    "in summary",
    "to summarise",
    "to summarize",
]


OVERVIEW_MARKERS = [
    "abstract",
    "introduction",
    "conclusion",
    "discussion",
    "summary",
]


LIST_MARKERS = [
    "experiment",
    "experiments",
    "study",
    "studies",
    "participants",
    "subjects",
    "method",
    "methods",
    "procedure",
    "task",
    "tested",
    "measured",
    "results",
    "findings",
    "examples",
]


def chunk_key(chunk: Dict) -> Tuple:
    return (
        chunk.get("document"),
        chunk.get("page"),
        chunk.get("text", "")[:180],
    )


def merge_unique(results: List[Dict]) -> List[Dict]:
    seen = set()
    unique = []

    for result in results:
        key = chunk_key(result)

        if key not in seen:
            seen.add(key)
            unique.append(result)

    return unique


def marker_boost(chunk: Dict, question_type: str) -> float:
    text = chunk.get("text", "").lower()
    section = str(chunk.get("section", "")).lower()

    boost = 0.0

    if any(marker in section for marker in OVERVIEW_MARKERS):
        boost += 0.35

    if question_type == "paper_level_argument":
        boost += sum(0.18 for marker in UNIVERSAL_ARGUMENT_MARKERS if marker in text)

    if question_type == "list":
        boost += sum(0.12 for marker in LIST_MARKERS if marker in text)

    return min(boost, 1.2)


def diversify_by_page(chunks: List[Dict], top_k: int) -> List[Dict]:
    selected = []
    used_pages = set()

    for chunk in chunks:
        page = chunk.get("page", "unknown")

        if page not in used_pages:
            selected.append(chunk)
            used_pages.add(page)

        if len(selected) >= top_k:
            return selected

    for chunk in chunks:
        if chunk not in selected:
            selected.append(chunk)

        if len(selected) >= top_k:
            break

    return selected


def build_general_fallback_queries(question: str, question_type: str) -> List[str]:
    if question_type == "paper_level_argument":
        return [
            question,
            "abstract introduction conclusion thesis central argument main claim",
            "this paper argues this article argues we argue author argues",
            "problem solution objection response conclusion implication",
            "we show we find we demonstrate we conclude",
        ]

    if question_type == "list":
        return [
            question,
            "experiment experiments study studies examples cases methods evidence",
            "participants subjects task procedure tested measured results findings",
            "mentioned discussed cited referenced reported observed",
        ]

    if question_type == "definition":
        return [
            question,
            "definition means refers to called known as concept term",
        ]

    if question_type == "quote":
        return [
            question,
            "quote quotation passage exact words states writes says",
        ]

    return [
        question,
        "main point relevant evidence passage discussion",
    ]


def coverage_score(question: str, chunks: List[Dict], question_type: str) -> float:
    if not chunks:
        return 0.0

    joined_text = " ".join(chunk.get("text", "").lower() for chunk in chunks)
    pages = set(chunk.get("page", "unknown") for chunk in chunks)

    question_terms = [
        word.strip(".,?!:;()[]{}").lower()
        for word in question.split()
        if len(word.strip(".,?!:;()[]{}")) > 4
    ]

    score = 0.0

    if question_terms:
        matched = sum(1 for term in question_terms if term in joined_text)
        score += min(matched / len(question_terms), 1.0) * 0.35

    if question_type == "paper_level_argument":
        if any(marker in joined_text for marker in UNIVERSAL_ARGUMENT_MARKERS):
            score += 0.35
        if len(pages) >= 3:
            score += 0.20
        if any(marker in joined_text for marker in OVERVIEW_MARKERS):
            score += 0.10

    elif question_type == "list":
        matched_markers = sum(1 for marker in LIST_MARKERS if marker in joined_text)
        score += min(matched_markers / 5, 1.0) * 0.45
        if len(chunks) >= 6:
            score += 0.20

    else:
        if len(chunks) >= 4:
            score += 0.25
        if len(pages) >= 2:
            score += 0.20

    return min(score, 1.0)


def retrieve_with_queries(
    retriever: Retriever,
    queries: List[str],
    raw_k: int,
) -> List[Dict]:
    all_results = []

    for query in queries:
        all_results.extend(
            retriever.hybrid_search(
                query,
                k=raw_k,
            )
        )

    return merge_unique(all_results)


def adaptive_retrieve(
    question: str,
    document_name: str,
    top_k: int,
    raw_k: int = 60,
) -> Dict:
    retriever = Retriever(document_name)

    plan = classify_question(question)
    question_type = plan["question_type"]

    retrieval_queries = build_retrieval_queries(question)

    raw_results = retrieve_with_queries(
        retriever=retriever,
        queries=retrieval_queries,
        raw_k=raw_k,
    )

    for chunk in raw_results:
        boost = marker_boost(chunk, question_type)
        chunk["planner_boost"] = boost
        chunk["hybrid_score"] = chunk.get("hybrid_score", 0) + boost

    reranked = rerank_results(
        question=question,
        results=raw_results,
        top_k=max(top_k * 3, 24),
    )

    selected = diversify_by_page(reranked, top_k)
    score = coverage_score(question, selected, question_type)

    if score < 0.70:
        fallback_queries = build_general_fallback_queries(question, question_type)

        fallback_results = retrieve_with_queries(
            retriever=retriever,
            queries=fallback_queries,
            raw_k=raw_k,
        )

        combined_results = merge_unique(raw_results + fallback_results)

        for chunk in combined_results:
            boost = marker_boost(chunk, question_type)
            chunk["planner_boost"] = boost
            chunk["hybrid_score"] = chunk.get("hybrid_score", 0) + boost

        reranked = rerank_results(
            question=question,
            results=combined_results,
            top_k=max(top_k * 4, 32),
        )

        selected = diversify_by_page(reranked, top_k)
        score = coverage_score(question, selected, question_type)
        raw_results = combined_results

    for chunk in selected:
        chunk["document"] = document_name

    return {
        "results": selected,
        "raw_results": raw_results,
        "question_type": question_type,
        "coverage_score": int(score * 100),
        "retrieval_queries": retrieval_queries,
    }
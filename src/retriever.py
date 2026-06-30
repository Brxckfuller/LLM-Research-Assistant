from pathlib import Path
import joblib
import faiss
import numpy as np

from src.embeddings import embed_query

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INDEX_DIR = PROJECT_ROOT / "data" / "indexes"
METADATA_DIR = PROJECT_ROOT / "data" / "metadata"


class Retriever:
    def __init__(self, document_name):
        self.document_name = document_name

        self.faiss_index = faiss.read_index(
            str(INDEX_DIR / f"{document_name}.faiss")
        )

        self.chunks = joblib.load(
            METADATA_DIR / f"{document_name}_chunks.joblib"
        )

        self.bm25 = joblib.load(
            METADATA_DIR / f"{document_name}_bm25.joblib"
        )

    def rewrite_query(self, query):
        q = query.lower()

        overview_terms = [
            "what is this document about",
            "what is this paper about",
            "summary",
            "summarise",
            "summarize",
            "overview",
            "main idea",
            "main argument",
        ]

        comparison_terms = [
            "compare",
            "difference",
            "similarity",
            "solution",
            "address",
            "hard problem",
            "argument",
        ]

        if any(term in q for term in overview_terms):
            return (
                "title abstract introduction main argument thesis purpose "
                "central question conclusion"
            )

        if any(term in q for term in comparison_terms):
            return (
                query
                + " abstract introduction main argument thesis solution conclusion"
            )

        return query

    def semantic_search(self, query, k=10):
        query_embedding = embed_query(query).astype("float32")

        scores, indices = self.faiss_index.search(
            np.array([query_embedding]),
            k,
        )

        results = []

        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue

            chunk = self.chunks[idx].copy()
            chunk["semantic_score"] = float(score)
            chunk["bm25_score"] = 0.0
            results.append(chunk)

        return results

    def keyword_search(self, query, k=10):
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)

        ranked_indices = np.argsort(scores)[::-1][:k]

        results = []

        for idx in ranked_indices:
            chunk = self.chunks[idx].copy()
            chunk["semantic_score"] = 0.0
            chunk["bm25_score"] = float(scores[idx])
            results.append(chunk)

        return results

    def reference_penalty(self, chunk):
        text_lower = chunk.get("text", "").lower()
        page = chunk.get("page", 0)
        total_pages = chunk.get("total_pages", 0)

        penalty = 0.0

        reference_indicators = [
            "references",
            "bibliography",
            "works cited",
            "doi:",
            "isbn",
            "cambridge university press",
            "oxford university press",
            "journal of",
            "vol.",
            "pp.",
        ]

        indicator_count = sum(
            1 for indicator in reference_indicators if indicator in text_lower
        )

        if chunk.get("is_references"):
            penalty -= 1.0

        if indicator_count >= 2:
            penalty -= 0.6

        if total_pages and page >= total_pages - 2 and indicator_count >= 1:
            penalty -= 0.4

        return penalty

    def metadata_boost(self, chunk):
        boost = 0.0

        text_lower = chunk.get("text", "").lower()
        page = chunk.get("page", 0)
        section = chunk.get("section", "").lower()

        if page == 1:
            boost += 0.25

        if "abstract" in text_lower or "introduction" in text_lower:
            boost += 0.20

        if "conclusion" in section or "conclusion" in text_lower:
            boost += 0.15

        important_terms = [
            "hard problem",
            "consciousness",
            "experience",
            "phenomenal",
            "qualia",
            "panpsychism",
            "physicalism",
            "materialism",
            "subjective",
            "explanation",
        ]

        for term in important_terms:
            if term in text_lower:
                boost += 0.04

        boost += self.reference_penalty(chunk)

        return boost

    def hybrid_search(self, query, k=10):
        search_query = self.rewrite_query(query)

        semantic_results = self.semantic_search(search_query, k=k * 3)
        keyword_results = self.keyword_search(search_query, k=k * 3)

        merged = {}

        for result in semantic_results + keyword_results:
            chunk_id = result["chunk_id"]

            if chunk_id not in merged:
                merged[chunk_id] = result
            else:
                merged[chunk_id]["semantic_score"] = max(
                    merged[chunk_id]["semantic_score"],
                    result["semantic_score"],
                )
                merged[chunk_id]["bm25_score"] = max(
                    merged[chunk_id]["bm25_score"],
                    result["bm25_score"],
                )

        results = list(merged.values())

        max_semantic = max([r["semantic_score"] for r in results], default=1)
        max_bm25 = max([r["bm25_score"] for r in results], default=1)

        for result in results:
            semantic_norm = (
                result["semantic_score"] / max_semantic
                if max_semantic > 0
                else 0
            )

            bm25_norm = (
                result["bm25_score"] / max_bm25
                if max_bm25 > 0
                else 0
            )

            result["hybrid_score"] = (
                0.55 * semantic_norm
                + 0.30 * bm25_norm
                + self.metadata_boost(result)
            )

        results.sort(key=lambda x: x["hybrid_score"], reverse=True)

        filtered = [
            r for r in results
            if r.get("hybrid_score", 0) > -0.2
        ]

        return filtered[:k]

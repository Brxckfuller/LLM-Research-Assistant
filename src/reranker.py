from sentence_transformers import CrossEncoder

_model = None


def get_reranker():
    global _model

    if _model is None:
        _model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    return _model


def rerank_results(question, results, top_k=6):
    if not results:
        return []

    model = get_reranker()

    pairs = [
        [
            question,
            f"{r.get('title', '')} {r.get('section', '')} {r.get('text', '')}",
        ]
        for r in results
    ]

    scores = model.predict(pairs)

    reranked = []
    for result, score in zip(results, scores):
        item = result.copy()
        item["rerank_score"] = float(score)
        reranked.append(item)

    reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
    return reranked[:top_k]
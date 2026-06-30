from src.retriever import Retriever

retriever = Retriever("turing")

results = retriever.hybrid_search(
    "What is this document about?",
    k=10
)

for result in results:

    print("Page:", result["page"])
    print("Chunk:", result["chunk_id"])
    print("Hybrid:", round(result["hybrid_score"], 3))
    print("Semantic:", round(result["semantic_score"], 3))
    print("BM25:", round(result["bm25_score"], 3))
    print(result["text"][:300])
    print("-" * 60)
from src.web_search import search_web

results = search_web("What is Retrieval-Augmented Generation?")

for result in results:
    print("=" * 80)
    print(result["title"])
    print(result["url"])
    print(result["text"][:400])

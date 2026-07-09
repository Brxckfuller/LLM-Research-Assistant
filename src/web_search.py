from tavily import TavilyClient
from dotenv import load_dotenv
load_dotenv()
import os


def search_web(query: str, max_results: int = 5):
    api_key = os.getenv("TAVILY_API_KEY")

    if not api_key:
        raise ValueError("Missing TAVILY_API_KEY environment variable.")

    client = TavilyClient(api_key=api_key)

    response = client.search(
        query=query,
        max_results=max_results,
        search_depth="advanced",
        include_answer=False,
        include_raw_content=True,
    )

    chunks = []

    for i, item in enumerate(response.get("results", []), start=1):
        text = item.get("raw_content") or item.get("content") or ""

        chunks.append(
            {
                "chunk_id": f"web_{i}",
                "document": "Web Search",
                "title": item.get("title", "Untitled"),
                "author": "Unknown",
                "section": "Web result",
                "page": item.get("url", "Web"),
                "text": text,
                "url": item.get("url", ""),
                "source_type": "web",
                "hybrid_score": item.get("score", 1.0),
                "semantic_score": 0.0,
                "bm25_score": 0.0,
            }
        )

    return chunks


def choose_route(question: str) -> str:
    question_lower = question.lower()

    web_keywords = [
        "web",
        "internet",
        "online",
        "according to the web",
        "from the web",
        "on the web",
        "web search",
        "search the web",
        "google",
        "according to google",
        "latest",
        "current",
        "today",
        "yesterday",
        "tomorrow",
        "recent",
        "new",
        "news",
        "breaking",
        "published",
        "release",
        "released",
        "this week",
        "this month",
        "this year",
        "live",
        "update",
        "2025",
        "2026",
    ]

    document_keywords = [
        "this paper",
        "the paper",
        "paper",
        "uploaded paper",
        "uploaded document",
        "uploaded pdf",
        "the document",
        "the pdf",
        "according to the paper",
        "according to the pdf",
        "according to this paper",
        "according to this document",
        "in the paper",
        "in the pdf",
        "from the paper",
        "from this paper",
        "in this paper",
    ]

    needs_web = any(keyword in question_lower for keyword in web_keywords)
    needs_docs = any(keyword in question_lower for keyword in document_keywords)

    if needs_web and needs_docs:
        return "BOTH"

    if needs_web:
        return "WEB"

    return "CHROMA"


import re
from typing import List


SECTION_PATTERN = re.compile(
    r"(^|\n)\s*((\d+\.|\d+\)|[IVX]+\.)\s+)?([A-Z][A-Za-z0-9 ,:'’\-]{3,90})\s*($|\n)"
)


def split_into_sentences(text):
    return re.split(r"(?<=[.!?])\s+", text.strip())


def looks_like_section_heading(text):
    text = text.strip()

    if len(text) < 3 or len(text) > 120:
        return False

    if re.match(r"^\d+[\.\)]\s+", text):
        return True

    if text.lower() in [
        "abstract",
        "introduction",
        "conclusion",
        "discussion",
        "references",
        "methods",
        "results",
    ]:
        return True

    words = text.split()
    if 1 <= len(words) <= 10 and sum(w[:1].isupper() for w in words) >= max(1, len(words) // 2):
        return True

    return False


def chunk_text(pages: List[dict], max_chunk_chars=1600, min_chunk_chars=250):
    """Chunk within pages without discarding short paragraphs or page tails."""
    chunks = []
    current_section = "General"

    for page in pages:
        page_number = page["page"]
        text = page["text"]
        # Preserve paragraph boundaries when possible; split long paragraphs.
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        if not paragraphs:
            paragraphs = [text.strip()] if text.strip() else []
        parts = []
        for paragraph in paragraphs:
            if len(paragraph) <= max_chunk_chars:
                parts.append(paragraph)
            else:
                sentences = split_into_sentences(paragraph)
                for sentence in sentences:
                    if len(sentence) <= max_chunk_chars:
                        parts.append(sentence)
                    else:
                        parts.extend(sentence[i:i + max_chunk_chars] for i in range(0, len(sentence), max_chunk_chars))

        current = ""
        for part in parts:
            if looks_like_section_heading(part):
                current_section = part[:120]
            if current and len(current) + len(part) + 1 > max_chunk_chars:
                chunks.append({
                    "chunk_id": len(chunks), "page": page_number,
                    "text": current, "section": current_section,
                    "author": page.get("author", "Unknown author"),
                    "title": page.get("title", "Unknown title"),
                    "document": page.get("document", "Unknown document"),
                })
                current = ""
            current = f"{current}\n{part}".strip() if current else part

        if current:
            chunks.append({
                "chunk_id": len(chunks), "page": page_number,
                "text": current, "section": current_section,
                "author": page.get("author", "Unknown author"),
                "title": page.get("title", "Unknown title"),
                "document": page.get("document", "Unknown document"),
            })

    return chunks




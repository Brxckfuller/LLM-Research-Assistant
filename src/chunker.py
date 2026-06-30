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
    chunks = []
    chunk_id = 0
    current_section = "General"

    for page in pages:
        page_number = page["page"]
        text = page["text"]

        rough_parts = re.split(r"(?<=[.!?])\s+", text)

        current = ""

        for part in rough_parts:
            part = part.strip()
            if not part:
                continue

            if looks_like_section_heading(part):
                current_section = part[:120]

            if len(current) + len(part) <= max_chunk_chars:
                current += " " + part
            else:
                if len(current.strip()) >= min_chunk_chars:
                    chunks.append(
                        {
                            "chunk_id": chunk_id,
                            "page": page_number,
                            "text": current.strip(),
                            "section": current_section,
                            "author": page.get("author", "Unknown author"),
                            "title": page.get("title", "Unknown title"),
                            "document": page.get("document", "Unknown document"),
                        }
                    )
                    chunk_id += 1

                current = part

        if len(current.strip()) >= min_chunk_chars:
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "page": page_number,
                    "text": current.strip(),
                    "section": current_section,
                    "author": page.get("author", "Unknown author"),
                    "title": page.get("title", "Unknown title"),
                    "document": page.get("document", "Unknown document"),
                }
            )
            chunk_id += 1

    return chunks




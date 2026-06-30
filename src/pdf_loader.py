from pathlib import Path
import re
import pymupdf


def clean_text(text):
    text = text.replace("\x00", " ")
    text = re.sub(r"-\s+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def infer_title_author(pdf_path, pages_text):
    title = Path(pdf_path).stem
    author = "Unknown author"

    first_text = pages_text[:2500]

    lines = [line.strip() for line in first_text.split("\n") if line.strip()]

    if lines:
        title = lines[0][:180]

    for line in lines[:20]:
        lowered = line.lower()
        if (
            len(line.split()) <= 8
            and not lowered.startswith(("abstract", "introduction", "commentary"))
            and any(char.isalpha() for char in line)
            and not any(word in lowered for word in ["journal", "university", "press", "volume"])
        ):
            if line != title:
                author = line[:120]
                break

    return {
        "title": title,
        "author": author,
        "document": Path(pdf_path).stem,
    }


def extract_text_from_pdf(pdf_path):
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    document = pymupdf.open(str(pdf_path))

    raw_pages = []
    all_text = ""

    for page_number, page in enumerate(document, start=1):
        raw_text = page.get_text()
        all_text += raw_text + "\n"

        raw_pages.append(
            {
                "page": page_number,
                "text": clean_text(raw_text),
            }
        )

    metadata = infer_title_author(pdf_path, all_text)

    pages = []
    for page in raw_pages:
        pages.append(
            {
                **page,
                "title": metadata["title"],
                "author": metadata["author"],
                "document": metadata["document"],
            }
        )

    document.close()
    return pages
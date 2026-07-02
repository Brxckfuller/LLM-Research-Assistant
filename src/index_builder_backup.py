from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import faiss
from rank_bm25 import BM25Okapi

from src.pdf_loader import extract_text_from_pdf
from src.chunker import chunk_text
from src.embeddings import embed_texts


BASE_DIR = Path(__file__).resolve().parent.parent
INDEX_DIR = BASE_DIR / "data" / "indexes"
METADATA_DIR = BASE_DIR / "data" / "metadata"

INDEX_DIR.mkdir(parents=True, exist_ok=True)
METADATA_DIR.mkdir(parents=True, exist_ok=True)


def tokenize(text):
    return text.lower().split()


def guess_author_and_title(document_name: str):
    lower = document_name.lower()

    if "dennett" in lower:
        return "Daniel Dennett", document_name

    if "strawson" in lower:
        return "Galen Strawson", document_name

    if "chalmers" in lower:
        return "David Chalmers", document_name

    if "goff" in lower:
        return "Philip Goff", document_name

    return "Unknown author", document_name


def build_index(pdf_path):
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    document_name = pdf_path.stem
    author, title = guess_author_and_title(document_name)

    pages = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(pages)

    for chunk in chunks:
        chunk["document"] = document_name
        chunk["author"] = author
        chunk["title"] = title
        chunk["total_pages"] = len(pages)

    texts = [chunk["text"] for chunk in chunks]

    embeddings = embed_texts(texts).astype("float32")
    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]
    faiss_index = faiss.IndexFlatIP(dimension)
    faiss_index.add(embeddings)

    tokenized_chunks = [tokenize(text) for text in texts]
    bm25_index = BM25Okapi(tokenized_chunks)

    faiss.write_index(
        faiss_index,
        str(INDEX_DIR / f"{document_name}.faiss"),
    )

    joblib.dump(
        chunks,
        METADATA_DIR / f"{document_name}_chunks.joblib",
    )

    joblib.dump(
        bm25_index,
        METADATA_DIR / f"{document_name}_bm25.joblib",
    )

    print(f"Indexed document: {document_name}")
    print(f"Author: {author}")
    print(f"Pages: {len(pages)}")
    print(f"Chunks: {len(chunks)}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise ValueError("Usage: python src/index_builder.py path/to/file.pdf")

    build_index(sys.argv[1]) 


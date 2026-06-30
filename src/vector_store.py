from pathlib import Path

import faiss
import numpy as np
import joblib
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent.parent
INDEX_DIR = BASE_DIR / "indexes"
INDEX_DIR.mkdir(exist_ok=True)

INDEX_PATH = INDEX_DIR / "faiss_index.bin"
CHUNKS_PATH = INDEX_DIR / "chunks.joblib"

MODEL_NAME = "all-MiniLM-L6-v2"


def load_embedding_model():
    return SentenceTransformer(MODEL_NAME)


def build_vector_store(chunks):
    if not chunks:
        raise ValueError("No chunks provided.")

    model = load_embedding_model()

    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=True
    )

    embeddings = embeddings.astype("float32")

    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    faiss.write_index(index, str(INDEX_PATH))
    joblib.dump(chunks, CHUNKS_PATH)

    return index


def load_vector_store():
    if not INDEX_PATH.exists():
        raise FileNotFoundError("FAISS index not found. Build the index first.")

    if not CHUNKS_PATH.exists():
        raise FileNotFoundError("Chunks file not found. Build the index first.")

    index = faiss.read_index(str(INDEX_PATH))
    chunks = joblib.load(CHUNKS_PATH)

    return index, chunks


def search(query, top_k=3):
    model = load_embedding_model()
    index, chunks = load_vector_store()

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True
    ).astype("float32")

    faiss.normalize_L2(query_embedding)

    scores, indices = index.search(query_embedding, top_k)

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue

        chunk = chunks[idx]

        results.append(
            {
                "page": chunk["page"],
                "text": chunk["text"],
                "score": float(score)
            }
        )

    return results
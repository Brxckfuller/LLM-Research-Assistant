from src.chunker import chunk_text


def test_chunk_text_creates_chunks():
    pages = [
        {
            "page": 1,
            "text": "This is a test sentence. " * 100
        }
    ]

    chunks = chunk_text(pages)

    assert len(chunks) > 0
    assert "text" in chunks[0]
    assert "page" in chunks[0]
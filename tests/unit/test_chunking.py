import pytest

from app.ingestion.chunking import chunk_text


def test_chunk_text_with_overlap() -> None:
    chunks = chunk_text("one two three four five six", chunk_size_tokens=3, overlap_tokens=1)

    assert [chunk.content for chunk in chunks] == [
        "one two three",
        "three four five",
        "five six",
    ]


def test_chunk_text_rejects_invalid_overlap() -> None:
    with pytest.raises(ValueError, match="overlap_tokens"):
        chunk_text("one two", chunk_size_tokens=2, overlap_tokens=2)

from app.ingestion.chunking import chunk_text
from app.ingestion.synthetic import generate_documents


def test_synthetic_documents_generate_multiple_chunks() -> None:
    documents = generate_documents(target_chunks=40, seed=42, chunks_per_document=4)
    chunk_count = sum(
        len(chunk_text(document.content, chunk_size_tokens=220, overlap_tokens=40))
        for document in documents
    )

    assert chunk_count >= 40

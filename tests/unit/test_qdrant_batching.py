import asyncio
from types import SimpleNamespace
from typing import Any, cast

from qdrant_client import AsyncQdrantClient

from app.core.config import Settings
from app.embeddings.providers import HashEmbeddingProvider
from app.ingestion.pipeline import deterministic_chunk_id, ingest_documents_qdrant
from app.ingestion.synthetic import generate_documents


def test_deterministic_chunk_id_is_stable() -> None:
    import uuid

    document_id = uuid.uuid5(uuid.NAMESPACE_URL, "doc")

    assert deterministic_chunk_id(document_id, 3) == deterministic_chunk_id(document_id, 3)
    assert deterministic_chunk_id(document_id, 3) != deterministic_chunk_id(document_id, 4)


class RecordingQdrantClient:
    def __init__(self) -> None:
        self.upsert_sizes: list[int] = []
        self.created_collections: list[str] = []

    async def get_collections(self) -> Any:
        return SimpleNamespace(collections=[])

    async def create_collection(self, collection_name: str, vectors_config: Any) -> None:
        self.created_collections.append(collection_name)

    async def upsert(self, collection_name: str, points: list[Any]) -> None:
        self.upsert_sizes.append(len(points))


def test_qdrant_ingestion_batches_large_upserts() -> None:
    client = RecordingQdrantClient()
    documents = generate_documents(target_chunks=6, seed=7, chunks_per_document=3)
    settings = Settings(
        embedding_provider="hash",
        embedding_dimension=8,
        qdrant_collection="test_chunks",
    )

    indexed = asyncio.run(
        ingest_documents_qdrant(
            cast(AsyncQdrantClient, client),
            settings,
            documents,
            HashEmbeddingProvider(dimension=8),
            chunk_size_tokens=220,
            overlap_tokens=40,
            batch_size=2,
        )
    )

    assert indexed >= 6
    assert client.created_collections == ["test_chunks"]
    assert len(client.upsert_sizes) > 1
    assert max(client.upsert_sizes) <= 2
    assert sum(client.upsert_sizes) == indexed

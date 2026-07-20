import uuid
from collections.abc import Iterable

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)
from sqlalchemy import delete, func
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.db.models import BenchmarkQuery, Chunk, Document
from app.embeddings.base import EmbeddingProvider
from app.ingestion.chunking import chunk_text
from app.ingestion.synthetic import SyntheticDocument, SyntheticQuery


def deterministic_chunk_id(document_id: uuid.UUID, chunk_index: int) -> uuid.UUID:
    return uuid.uuid5(uuid.NAMESPACE_URL, f"{document_id}:{chunk_index}")


async def ingest_documents_postgres(
    session: AsyncSession,
    documents: Iterable[SyntheticDocument],
    embedding_provider: EmbeddingProvider,
    chunk_size_tokens: int,
    overlap_tokens: int,
) -> int:
    chunk_count = 0
    for document in documents:
        await session.execute(
            insert(Document)
            .values(
                id=document.id,
                title=document.title,
                source=document.source,
                version=document.version,
                language=document.language,
                tenant_id=document.tenant_id,
                department=document.department,
                access_level=document.access_level,
                checksum=document.checksum,
                metadata_=document.metadata,
            )
            .on_conflict_do_update(
                index_elements=[Document.checksum],
                set_={
                    "title": document.title,
                    "source": document.source,
                    "version": document.version,
                    "language": document.language,
                    "tenant_id": document.tenant_id,
                    "department": document.department,
                    "access_level": document.access_level,
                    "metadata": document.metadata,
                },
            )
        )
        await session.execute(delete(Chunk).where(Chunk.document_id == document.id))
        text_chunks = chunk_text(document.content, chunk_size_tokens, overlap_tokens)
        embeddings = await embedding_provider.embed_texts([chunk.content for chunk in text_chunks])
        for text_chunk, embedding in zip(text_chunks, embeddings, strict=True):
            chunk_id = deterministic_chunk_id(document.id, text_chunk.chunk_index)
            await session.execute(
                insert(Chunk).values(
                    id=chunk_id,
                    document_id=document.id,
                    chunk_index=text_chunk.chunk_index,
                    content=text_chunk.content,
                    token_count=text_chunk.token_count,
                    metadata_={
                        **document.metadata,
                        "tenant_id": document.tenant_id,
                        "department": document.department,
                        "language": document.language,
                        "version": document.version,
                        "access_level": document.access_level,
                    },
                    content_tsv=func.to_tsvector("english", text_chunk.content),
                    embedding=embedding,
                )
            )
            chunk_count += 1
    await session.commit()
    return chunk_count


async def ingest_queries_postgres(session: AsyncSession, queries: Iterable[SyntheticQuery]) -> int:
    count = 0
    for query in queries:
        await session.execute(
            insert(BenchmarkQuery)
            .values(
                id=query.id,
                query_text=query.query_text,
                query_type=query.query_type,
                language=query.language,
                filters=query.filters,
                expected_relevant_document_ids=query.expected_relevant_document_ids,
                expected_relevant_chunk_ids=query.expected_relevant_chunk_ids,
                label_source=query.label_source,
            )
            .on_conflict_do_update(
                index_elements=[BenchmarkQuery.id],
                set_={
                    "query_text": query.query_text,
                    "query_type": query.query_type,
                    "language": query.language,
                    "filters": query.filters,
                    "expected_relevant_document_ids": query.expected_relevant_document_ids,
                    "expected_relevant_chunk_ids": query.expected_relevant_chunk_ids,
                    "label_source": query.label_source,
                },
            )
        )
        count += 1
    await session.commit()
    return count


async def ensure_qdrant_collection(client: AsyncQdrantClient, settings: Settings) -> None:
    collections = await client.get_collections()
    if any(item.name == settings.qdrant_collection for item in collections.collections):
        return
    await client.create_collection(
        collection_name=settings.qdrant_collection,
        vectors_config=VectorParams(size=settings.embedding_dimension, distance=Distance.COSINE),
    )


def qdrant_filter(filters: dict[str, object] | None) -> Filter | None:
    if not filters:
        return None
    return Filter(
        must=[
            FieldCondition(key=str(key), match=MatchValue(value=value))
            for key, value in filters.items()
            if not isinstance(value, (list, dict))
        ]
    )


async def ingest_documents_qdrant(
    client: AsyncQdrantClient,
    settings: Settings,
    documents: Iterable[SyntheticDocument],
    embedding_provider: EmbeddingProvider,
    chunk_size_tokens: int,
    overlap_tokens: int,
) -> int:
    await ensure_qdrant_collection(client, settings)
    points: list[PointStruct] = []
    for document in documents:
        text_chunks = chunk_text(document.content, chunk_size_tokens, overlap_tokens)
        embeddings = await embedding_provider.embed_texts([chunk.content for chunk in text_chunks])
        for text_chunk, embedding in zip(text_chunks, embeddings, strict=True):
            chunk_id = deterministic_chunk_id(document.id, text_chunk.chunk_index)
            points.append(
                PointStruct(
                    id=str(chunk_id),
                    vector=embedding,
                    payload={
                        **document.metadata,
                        "document_id": str(document.id),
                        "text": text_chunk.content,
                        "tenant_id": document.tenant_id,
                        "department": document.department,
                        "language": document.language,
                        "version": document.version,
                        "access_level": document.access_level,
                    },
                )
            )
    if points:
        await client.upsert(collection_name=settings.qdrant_collection, points=points)
    return len(points)

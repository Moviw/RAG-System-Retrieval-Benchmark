import time
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.embeddings.base import EmbeddingProvider
from app.retrievers.filters import build_sql_filter_clause, vector_literal
from app.schemas.search import SearchResult


class LexicalPostgresRetriever:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def search(
        self,
        query: str,
        top_k: int,
        filters: dict[str, object] | None = None,
    ) -> list[SearchResult]:
        started = time.perf_counter()
        filter_clause, params = build_sql_filter_clause(filters)
        sql = text(
            f"""
            WITH parsed_query AS (
                SELECT websearch_to_tsquery('english', :query) AS tsq
            )
            SELECT
                c.id::text AS chunk_id,
                c.document_id::text AS document_id,
                c.content AS text,
                c.metadata AS metadata,
                ts_rank_cd(c.content_tsv, parsed_query.tsq) AS score
            FROM chunks c, parsed_query
            WHERE c.content_tsv @@ parsed_query.tsq
            {filter_clause}
            ORDER BY score DESC, c.id
            LIMIT :top_k
            """
        )
        rows = (
            await self.session.execute(sql, {"query": query, "top_k": top_k, **params})
        ).mappings()
        elapsed_ms = (time.perf_counter() - started) * 1000
        return [
            SearchResult(
                chunk_id=row["chunk_id"],
                document_id=row["document_id"],
                text=row["text"],
                metadata=dict(row["metadata"]),
                score=float(row["score"] or 0.0),
                rank=rank,
                strategy="lexical",
                latency_ms={"lexical": elapsed_ms},
            )
            for rank, row in enumerate(rows, start=1)
        ]


class PgVectorRetriever:
    def __init__(self, session: AsyncSession, embedding_provider: EmbeddingProvider) -> None:
        self.session = session
        self.embedding_provider = embedding_provider

    async def search(
        self,
        query: str,
        top_k: int,
        filters: dict[str, object] | None = None,
    ) -> list[SearchResult]:
        started = time.perf_counter()
        embedding_started = time.perf_counter()
        embedding = await self.embedding_provider.embed_query(query)
        embedding_ms = (time.perf_counter() - embedding_started) * 1000
        filter_clause, params = build_sql_filter_clause(filters)
        sql = text(
            f"""
            SELECT
                c.id::text AS chunk_id,
                c.document_id::text AS document_id,
                c.content AS text,
                c.metadata AS metadata,
                1 - (c.embedding <=> '{vector_literal(embedding)}'::vector) AS score
            FROM chunks c
            WHERE TRUE
            {filter_clause}
            ORDER BY c.embedding <=> '{vector_literal(embedding)}'::vector
            LIMIT :top_k
            """
        )
        rows = (await self.session.execute(sql, {"top_k": top_k, **params})).mappings()
        elapsed_ms = (time.perf_counter() - started) * 1000
        vector_ms = max(0.0, elapsed_ms - embedding_ms)
        return [
            SearchResult(
                chunk_id=row["chunk_id"],
                document_id=row["document_id"],
                text=row["text"],
                metadata=dict(row["metadata"]),
                score=float(row["score"] or 0.0),
                rank=rank,
                strategy="dense_pgvector",
                latency_ms={"embedding": embedding_ms, "vector": vector_ms},
            )
            for rank, row in enumerate(rows, start=1)
        ]


async def explain_postgres_query(
    session: AsyncSession,
    sql: str,
    params: dict[str, Any],
) -> Any:
    result = await session.execute(text(f"EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) {sql}"), params)
    return result.scalar_one()

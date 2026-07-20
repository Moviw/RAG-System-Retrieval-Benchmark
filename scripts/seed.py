import asyncio
from pathlib import Path
from typing import Any

import typer
import yaml
from qdrant_client import AsyncQdrantClient

from app.core.config import get_settings
from app.db.session import get_sessionmaker
from app.embeddings.providers import HashEmbeddingProvider, build_embedding_provider
from app.ingestion.pipeline import (
    ingest_documents_postgres,
    ingest_documents_qdrant,
    ingest_queries_postgres,
)
from app.ingestion.synthetic import generate_documents, generate_queries

app = typer.Typer(help="Seed benchmark datasets.")


@app.command()
def main(
    config: Path = typer.Option(..., "--config", exists=True),
    postgres_only: bool = typer.Option(False, "--postgres-only"),
    qdrant_only: bool = typer.Option(False, "--qdrant-only"),
    fast_hash_embeddings: bool = typer.Option(False, "--fast-hash-embeddings"),
) -> None:
    loaded_config: dict[str, Any] = yaml.safe_load(config.read_text()) or {}
    asyncio.run(_run(loaded_config, postgres_only, qdrant_only, fast_hash_embeddings))


async def _run(
    config: dict[str, Any],
    postgres_only: bool,
    qdrant_only: bool,
    fast_hash_embeddings: bool,
) -> None:
    settings = get_settings()
    dataset = config.get("dataset", {})
    embedding_config = config.get("embedding", {})
    target_chunks = int(dataset.get("target_chunks", 10000))
    seed = int(dataset.get("seed", settings.random_seed))
    chunk_size = int(dataset.get("chunk_size_tokens", 220))
    overlap = int(dataset.get("chunk_overlap_tokens", 40))

    documents = generate_documents(target_chunks=target_chunks, seed=seed)
    queries = generate_queries(documents)
    provider = (
        HashEmbeddingProvider(settings.embedding_dimension)
        if fast_hash_embeddings or embedding_config.get("provider") == "hash"
        else build_embedding_provider(settings)
    )

    if not qdrant_only:
        async with get_sessionmaker()() as session:
            chunk_count = await ingest_documents_postgres(
                session, documents, provider, chunk_size, overlap
            )
            query_count = await ingest_queries_postgres(session, queries)
        typer.echo(f"PostgreSQL seeded: documents={len(documents)} chunks={chunk_count}")
        typer.echo(f"Benchmark queries seeded: queries={query_count}")

    if not postgres_only:
        client = AsyncQdrantClient(url=settings.qdrant_url)
        try:
            qdrant_count = await ingest_documents_qdrant(
                client, settings, documents, provider, chunk_size, overlap
            )
        finally:
            await client.close()
        typer.echo(f"Qdrant seeded: chunks={qdrant_count}")


if __name__ == "__main__":
    app()

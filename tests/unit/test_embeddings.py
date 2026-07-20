import asyncio

from app.embeddings.providers import HashEmbeddingProvider


def test_hash_embeddings_are_deterministic_and_normalized() -> None:
    provider = HashEmbeddingProvider(dimension=16)
    first = asyncio.run(provider.embed_query("postgres logical replication"))
    second = asyncio.run(provider.embed_query("postgres logical replication"))

    assert first == second
    assert len(first) == 16
    assert round(sum(value * value for value in first), 8) == 1.0

import pytest

from app.embeddings.providers import HashEmbeddingProvider


@pytest.mark.anyio
async def test_hash_embeddings_are_deterministic_and_normalized() -> None:
    provider = HashEmbeddingProvider(dimension=16)
    first = await provider.embed_query("postgres logical replication")
    second = await provider.embed_query("postgres logical replication")

    assert first == second
    assert len(first) == 16
    assert sum(value * value for value in first) == pytest.approx(1.0)

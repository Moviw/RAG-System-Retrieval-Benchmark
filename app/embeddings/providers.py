import asyncio
import hashlib
import math
from collections.abc import Sequence
from typing import cast

import httpx

from app.core.config import Settings


class HashEmbeddingProvider:
    def __init__(self, dimension: int = 384) -> None:
        self.dimension = dimension

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    async def embed_query(self, query: str) -> list[float]:
        return self._embed(query)

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        for token in text.lower().split():
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "big") % self.dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[idx] += sign
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]


class LocalSentenceTransformerProvider:
    def __init__(self, model_name: str, dimension: int, batch_size: int, device: str) -> None:
        from sentence_transformers import SentenceTransformer

        self.dimension = dimension
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name, device=device)

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return await asyncio.to_thread(self._encode, texts)

    async def embed_query(self, query: str) -> list[float]:
        return (await self.embed_texts([query]))[0]

    def _encode(self, texts: Sequence[str]) -> list[list[float]]:
        embeddings = self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return cast(list[list[float]], embeddings.tolist())


class ExternalEmbeddingProvider:
    def __init__(self, url: str, api_key: str | None, dimension: int, batch_size: int) -> None:
        self.url = url
        self.api_key = api_key
        self.dimension = dimension
        self.batch_size = batch_size

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(self.url, json={"input": texts}, headers=headers)
            response.raise_for_status()
            payload = response.json()
        return [item["embedding"] for item in payload["data"]]

    async def embed_query(self, query: str) -> list[float]:
        return (await self.embed_texts([query]))[0]


EmbeddingProviderType = (
    HashEmbeddingProvider | LocalSentenceTransformerProvider | ExternalEmbeddingProvider
)


def build_embedding_provider(settings: Settings) -> EmbeddingProviderType:
    if settings.embedding_provider == "hash":
        return HashEmbeddingProvider(settings.embedding_dimension)
    if settings.embedding_provider == "external":
        if not settings.external_embedding_url:
            raise ValueError("EXTERNAL_EMBEDDING_URL is required for external embeddings")
        api_key = (
            settings.external_embedding_api_key.get_secret_value()
            if settings.external_embedding_api_key
            else None
        )
        return ExternalEmbeddingProvider(
            settings.external_embedding_url,
            api_key,
            settings.embedding_dimension,
            settings.embedding_batch_size,
        )
    return LocalSentenceTransformerProvider(
        settings.embedding_model,
        settings.embedding_dimension,
        settings.embedding_batch_size,
        settings.embedding_device,
    )

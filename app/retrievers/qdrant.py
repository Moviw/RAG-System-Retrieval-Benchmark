import time
from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue

from app.core.config import Settings
from app.embeddings.base import EmbeddingProvider
from app.retrievers.filters import ALLOWED_FILTERS
from app.schemas.search import SearchResult


def build_qdrant_filter(filters: dict[str, object] | None) -> Filter | None:
    if not filters:
        return None
    conditions = []
    for key, value in filters.items():
        if key not in ALLOWED_FILTERS:
            raise ValueError(f"Unsupported filter field: {key}")
        match = (
            MatchAny(any=[str(item) for item in value])
            if isinstance(value, list)
            else MatchValue(value=str(value))
        )
        conditions.append(FieldCondition(key=key, match=match))
    return Filter(must=conditions)


class QdrantRetriever:
    def __init__(
        self,
        client: AsyncQdrantClient,
        settings: Settings,
        embedding_provider: EmbeddingProvider,
    ) -> None:
        self.client = client
        self.settings = settings
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
        hits = await self.client.search(
            collection_name=self.settings.qdrant_collection,
            query_vector=embedding,
            query_filter=build_qdrant_filter(filters),
            limit=top_k,
            with_payload=True,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000
        vector_ms = max(0.0, elapsed_ms - embedding_ms)
        results: list[SearchResult] = []
        for rank, hit in enumerate(hits, start=1):
            payload: dict[str, Any] = dict(hit.payload or {})
            results.append(
                SearchResult(
                    chunk_id=str(hit.id),
                    document_id=str(payload.get("document_id", "")),
                    text=str(payload.get("text", "")),
                    metadata=payload,
                    score=float(hit.score),
                    rank=rank,
                    strategy="dense_qdrant",
                    latency_ms={"embedding": embedding_ms, "vector": vector_ms},
                )
            )
        return results

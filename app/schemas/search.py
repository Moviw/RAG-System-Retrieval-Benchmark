from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class RetrievalStrategy(StrEnum):
    lexical = "lexical"
    dense_pgvector = "dense_pgvector"
    dense_qdrant = "dense_qdrant"
    hybrid_pgvector = "hybrid_pgvector"
    hybrid_qdrant = "hybrid_qdrant"
    adaptive = "adaptive"


class DenseBackend(StrEnum):
    pgvector = "pgvector"
    qdrant = "qdrant"


class FusionMethod(StrEnum):
    rrf = "rrf"
    weighted = "weighted"


class SearchResult(BaseModel):
    chunk_id: str
    document_id: str
    text: str
    metadata: dict[str, Any]
    score: float
    rank: int
    strategy: str
    latency_ms: dict[str, float] = Field(default_factory=dict)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    strategy: RetrievalStrategy = RetrievalStrategy.hybrid_qdrant
    top_k: int = Field(default=10, ge=1, le=100)
    filters: dict[str, Any] | None = None
    dense_backend: DenseBackend = DenseBackend.qdrant
    fusion_method: FusionMethod = FusionMethod.rrf
    reranker_enabled: bool = False
    debug: bool = False


class SearchResponse(BaseModel):
    results: list[SearchResult]
    strategy: str
    latency_ms: dict[str, float]
    debug: dict[str, Any] | None = None

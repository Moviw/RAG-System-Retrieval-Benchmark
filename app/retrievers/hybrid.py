import asyncio
import time
from collections.abc import Iterable

from app.retrievers.base import Retriever
from app.schemas.search import SearchResult


def reciprocal_rank_fusion(
    result_sets: Iterable[list[SearchResult]],
    k: int = 60,
) -> list[SearchResult]:
    by_chunk: dict[str, SearchResult] = {}
    scores: dict[str, float] = {}
    for results in result_sets:
        for result in results:
            by_chunk.setdefault(result.chunk_id, result)
            scores[result.chunk_id] = scores.get(result.chunk_id, 0.0) + 1.0 / (k + result.rank)
    fused = sorted(by_chunk.values(), key=lambda item: scores[item.chunk_id], reverse=True)
    return [
        item.model_copy(
            update={"score": scores[item.chunk_id], "rank": rank, "strategy": "hybrid_rrf"}
        )
        for rank, item in enumerate(fused, start=1)
    ]


def min_max_normalize(results: list[SearchResult]) -> dict[str, float]:
    if not results:
        return {}
    values = [result.score for result in results]
    minimum = min(values)
    maximum = max(values)
    if maximum == minimum:
        return {result.chunk_id: 1.0 for result in results}
    return {result.chunk_id: (result.score - minimum) / (maximum - minimum) for result in results}


def weighted_score_fusion(
    lexical_results: list[SearchResult],
    dense_results: list[SearchResult],
    lexical_weight: float = 0.45,
    dense_weight: float = 0.55,
) -> list[SearchResult]:
    lexical_scores = min_max_normalize(lexical_results)
    dense_scores = min_max_normalize(dense_results)
    by_chunk = {result.chunk_id: result for result in [*lexical_results, *dense_results]}
    fused_scores = {
        chunk_id: lexical_weight * lexical_scores.get(chunk_id, 0.0)
        + dense_weight * dense_scores.get(chunk_id, 0.0)
        for chunk_id in by_chunk
    }
    fused = sorted(by_chunk.values(), key=lambda item: fused_scores[item.chunk_id], reverse=True)
    return [
        item.model_copy(
            update={
                "score": fused_scores[item.chunk_id],
                "rank": rank,
                "strategy": "hybrid_weighted",
            }
        )
        for rank, item in enumerate(fused, start=1)
    ]


class HybridRetriever:
    def __init__(
        self,
        lexical: Retriever,
        dense: Retriever,
        fusion_method: str = "rrf",
        candidate_multiplier: int = 5,
    ) -> None:
        self.lexical = lexical
        self.dense = dense
        self.fusion_method = fusion_method
        self.candidate_multiplier = candidate_multiplier

    async def search(
        self,
        query: str,
        top_k: int,
        filters: dict[str, object] | None = None,
    ) -> list[SearchResult]:
        started = time.perf_counter()
        candidate_k = max(top_k, top_k * self.candidate_multiplier)
        lexical_results, dense_results = await asyncio.gather(
            self.lexical.search(query, candidate_k, filters),
            self.dense.search(query, candidate_k, filters),
        )
        fusion_started = time.perf_counter()
        if self.fusion_method == "weighted":
            fused = weighted_score_fusion(lexical_results, dense_results)
        else:
            fused = reciprocal_rank_fusion([lexical_results, dense_results])
        fusion_ms = (time.perf_counter() - fusion_started) * 1000
        total_ms = (time.perf_counter() - started) * 1000
        return [
            result.model_copy(
                update={
                    "rank": rank,
                    "latency_ms": {
                        **result.latency_ms,
                        "fusion": fusion_ms,
                        "total": total_ms,
                    },
                }
            )
            for rank, result in enumerate(fused[:top_k], start=1)
        ]

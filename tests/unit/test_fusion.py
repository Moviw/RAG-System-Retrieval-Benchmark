from app.retrievers.hybrid import min_max_normalize, reciprocal_rank_fusion, weighted_score_fusion
from app.schemas.search import SearchResult


def result(chunk_id: str, score: float, rank: int, strategy: str = "test") -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        document_id=f"doc-{chunk_id}",
        text=f"text {chunk_id}",
        metadata={},
        score=score,
        rank=rank,
        strategy=strategy,
    )


def test_rrf_rewards_items_appearing_in_multiple_lists() -> None:
    lexical = [result("a", 10, 1), result("b", 9, 2)]
    dense = [result("b", 0.9, 1), result("c", 0.8, 2)]

    fused = reciprocal_rank_fusion([lexical, dense])

    assert fused[0].chunk_id == "b"
    assert fused[0].strategy == "hybrid_rrf"


def test_min_max_normalize_handles_equal_scores() -> None:
    scores = min_max_normalize([result("a", 2, 1), result("b", 2, 2)])

    assert scores == {"a": 1.0, "b": 1.0}


def test_weighted_fusion_normalizes_heterogeneous_scores() -> None:
    lexical = [result("a", 12.0, 1), result("b", 2.0, 2)]
    dense = [result("b", 0.9, 1), result("a", 0.2, 2)]

    fused = weighted_score_fusion(lexical, dense, lexical_weight=0.5, dense_weight=0.5)

    assert {item.chunk_id for item in fused} == {"a", "b"}
    assert all(0.0 <= item.score <= 1.0 for item in fused)

import pytest

from app.evaluation.metrics import (
    hit_rate_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


def test_retrieval_quality_metrics() -> None:
    retrieved = ["a", "b", "c"]
    relevant = {"b", "d"}

    assert recall_at_k(retrieved, relevant, 3) == 0.5
    assert precision_at_k(retrieved, relevant, 2) == 0.5
    assert hit_rate_at_k(retrieved, relevant, 1) == 0.0
    assert mean_reciprocal_rank(retrieved, relevant) == 0.5
    assert ndcg_at_k(retrieved, relevant, 3) == pytest.approx(0.38685, rel=1e-4)


def test_empty_relevance_hard_negative() -> None:
    assert recall_at_k([], set(), 10) == 1.0
    assert hit_rate_at_k(["a"], set(), 10) == 0.0

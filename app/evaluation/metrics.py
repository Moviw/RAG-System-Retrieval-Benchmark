import math
from collections.abc import Sequence


def recall_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 1.0 if not retrieved[:k] else 0.0
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def precision_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    if not retrieved[:k]:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / k


def hit_rate_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 1.0 if not retrieved[:k] else 0.0
    return 1.0 if set(retrieved[:k]) & relevant else 0.0


def mean_reciprocal_rank(retrieved: Sequence[str], relevant: set[str]) -> float:
    if not relevant:
        return 1.0 if not retrieved else 0.0
    for index, chunk_id in enumerate(retrieved, start=1):
        if chunk_id in relevant:
            return 1.0 / index
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 1.0 if not retrieved[:k] else 0.0
    dcg = 0.0
    for index, chunk_id in enumerate(retrieved[:k], start=1):
        if chunk_id in relevant:
            dcg += 1.0 / math.log2(index + 1)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(index + 1) for index in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def retrieval_metrics(retrieved: Sequence[str], relevant: set[str], k: int) -> dict[str, float]:
    return {
        f"recall_at_{k}": recall_at_k(retrieved, relevant, k),
        f"precision_at_{k}": precision_at_k(retrieved, relevant, k),
        f"hit_rate_at_{k}": hit_rate_at_k(retrieved, relevant, k),
        "mrr": mean_reciprocal_rank(retrieved, relevant),
        f"ndcg_at_{k}": ndcg_at_k(retrieved, relevant, k),
    }

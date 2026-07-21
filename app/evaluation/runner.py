import asyncio
import platform
import subprocess
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

import yaml
from qdrant_client import AsyncQdrantClient
from sqlalchemy import select

from app.core.config import get_settings
from app.db.models import BenchmarkQuery
from app.db.session import get_sessionmaker
from app.embeddings.providers import build_embedding_provider
from app.evaluation.artifacts import (
    write_basic_chart,
    write_csv,
    write_json,
    write_markdown_summary,
    write_strategy_chart,
)
from app.evaluation.metrics import retrieval_metrics
from app.retrievers.factory import build_retriever
from app.schemas.search import DenseBackend, FusionMethod, RetrievalStrategy


@dataclass(frozen=True)
class StrategySpec:
    name: str
    strategy: RetrievalStrategy
    dense_backend: DenseBackend
    fusion_method: FusionMethod


def git_commit_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def hardware_information() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": platform.python_version(),
    }


def percentile(values: list[float], percentile_value: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile_value
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    if lower == upper:
        return ordered[lower]
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def unique_preserving_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique_values: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique_values.append(value)
    return unique_values


def parse_strategy_specs(retrieval_config: dict[str, Any]) -> list[StrategySpec]:
    configured = retrieval_config.get("strategies")
    if not configured:
        configured = [retrieval_config.get("strategy", "hybrid_qdrant")]

    specs: list[StrategySpec] = []
    for item in configured:
        if isinstance(item, dict):
            raw_name = str(item["strategy"])
            dense_backend = DenseBackend(item.get("dense_backend", "qdrant"))
            fusion_method = FusionMethod(item.get("fusion_method", "rrf"))
        else:
            raw_name = str(item)
            dense_backend = DenseBackend(retrieval_config.get("dense_backend", "qdrant"))
            fusion_method = FusionMethod(retrieval_config.get("fusion_method", "rrf"))

        base_name = raw_name
        if raw_name.endswith("_rrf"):
            base_name = raw_name.removesuffix("_rrf")
            fusion_method = FusionMethod.rrf
        elif raw_name.endswith("_weighted"):
            base_name = raw_name.removesuffix("_weighted")
            fusion_method = FusionMethod.weighted

        strategy = RetrievalStrategy(base_name)
        if strategy in {RetrievalStrategy.dense_pgvector, RetrievalStrategy.hybrid_pgvector}:
            dense_backend = DenseBackend.pgvector
        elif strategy in {RetrievalStrategy.dense_qdrant, RetrievalStrategy.hybrid_qdrant}:
            dense_backend = DenseBackend.qdrant

        display_name = raw_name
        if strategy in {RetrievalStrategy.hybrid_pgvector, RetrievalStrategy.hybrid_qdrant}:
            display_name = f"{strategy.value}_{fusion_method.value}"

        specs.append(
            StrategySpec(
                name=display_name,
                strategy=strategy,
                dense_backend=dense_backend,
                fusion_method=fusion_method,
            )
        )
    return specs


def summarize_by_query_type(rows: list[dict[str, Any]], top_k: int) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["query_type"])].append(row)
    recall_key = f"recall_at_{top_k}"
    return {
        query_type: {
            "count": float(len(items)),
            "mean_recall": mean(float(item[recall_key]) for item in items),
            "mean_latency_ms": mean(float(item["latency_ms"]) for item in items),
        }
        for query_type, items in grouped.items()
    }


def summarize_by_strategy(rows: list[dict[str, Any]], top_k: int) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["strategy"])].append(row)
    recall_key = f"recall_at_{top_k}"
    precision_key = f"precision_at_{top_k}"
    hit_rate_key = f"hit_rate_at_{top_k}"
    ndcg_key = f"ndcg_at_{top_k}"
    return {
        strategy: {
            "count": float(len(items)),
            "mean_recall": mean(float(item[recall_key]) for item in items),
            "mean_precision": mean(float(item[precision_key]) for item in items),
            "mean_hit_rate": mean(float(item[hit_rate_key]) for item in items),
            "mean_mrr": mean(float(item["mrr"]) for item in items),
            "mean_ndcg": mean(float(item[ndcg_key]) for item in items),
            "mean_latency_ms": mean(float(item["latency_ms"]) for item in items),
            "p50_latency_ms": percentile([float(item["latency_ms"]) for item in items], 0.50),
            "p95_latency_ms": percentile([float(item["latency_ms"]) for item in items], 0.95),
            "p99_latency_ms": percentile([float(item["latency_ms"]) for item in items], 0.99),
            "max_latency_ms": max(float(item["latency_ms"]) for item in items),
        }
        for strategy, items in grouped.items()
    }


def summarize_by_strategy_and_query_type(
    rows: list[dict[str, Any]],
    top_k: int,
) -> dict[str, dict[str, dict[str, float]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["strategy"])].append(row)
    return {strategy: summarize_by_query_type(items, top_k) for strategy, items in grouped.items()}


async def run_benchmark(config_path: Path) -> dict[str, Any]:
    settings = get_settings()
    config_text = await asyncio.to_thread(config_path.read_text)
    config: dict[str, Any] = yaml.safe_load(config_text) or {}
    retrieval_config = config.get("retrieval", {})
    top_k = int(retrieval_config.get("top_k", settings.default_top_k))
    strategy_specs = parse_strategy_specs(retrieval_config)
    dataset_size = int(config.get("dataset", {}).get("target_chunks", 0))
    run_id = str(uuid.uuid4())

    provider = build_embedding_provider(settings)
    qdrant_client = AsyncQdrantClient(url=settings.qdrant_url)
    rows: list[dict[str, Any]] = []
    try:
        async with get_sessionmaker()() as session:
            queries = (await session.execute(select(BenchmarkQuery))).scalars().all()
            for strategy_spec in strategy_specs:
                retriever = build_retriever(
                    strategy_spec.strategy,
                    session,
                    qdrant_client,
                    settings,
                    provider,
                    strategy_spec.dense_backend,
                    strategy_spec.fusion_method,
                )
                for benchmark_query in queries:
                    started = time.perf_counter()
                    results = await retriever.search(
                        benchmark_query.query_text,
                        top_k,
                        benchmark_query.filters,
                    )
                    latency_ms = (time.perf_counter() - started) * 1000
                    retrieved_ids = [result.chunk_id for result in results]
                    relevant = set(benchmark_query.expected_relevant_chunk_ids)
                    if not relevant:
                        relevant = set(benchmark_query.expected_relevant_document_ids)
                        retrieved_ids = [result.document_id for result in results]
                    retrieved_ids = unique_preserving_order(retrieved_ids)
                    metric_values = retrieval_metrics(retrieved_ids, relevant, top_k)
                    rows.append(
                        {
                            "strategy": strategy_spec.name,
                            "base_strategy": strategy_spec.strategy.value,
                            "dense_backend": strategy_spec.dense_backend.value,
                            "fusion_method": strategy_spec.fusion_method.value,
                            "query_id": benchmark_query.id,
                            "query_type": benchmark_query.query_type,
                            "language": benchmark_query.language,
                            "latency_ms": latency_ms,
                            "retrieved_ids": retrieved_ids,
                            **metric_values,
                        }
                    )
    finally:
        await qdrant_client.close()

    payload: dict[str, Any] = {
        "run_id": run_id,
        "config_name": config.get("name", config_path.stem),
        "strategy": "multi" if len(strategy_specs) > 1 else strategy_specs[0].name,
        "strategies": [strategy_spec.name for strategy_spec in strategy_specs],
        "top_k": top_k,
        "dataset_size": dataset_size,
        "warm_cache": bool(config.get("warm_cache", True)),
        "reranker_enabled": bool(config.get("reranker_enabled", False)),
        "git_commit_hash": git_commit_hash(),
        "hardware_information": hardware_information(),
        "configuration": config,
        "results": rows,
        "by_query_type": summarize_by_query_type(rows, top_k),
        "by_strategy": summarize_by_strategy(rows, top_k),
        "by_strategy_and_query_type": summarize_by_strategy_and_query_type(rows, top_k),
    }
    output_dir = Path(settings.benchmark_results_dir) / run_id
    write_json(output_dir / "results.json", payload)
    write_csv(output_dir / "results.csv", rows)
    write_markdown_summary(output_dir / "summary.md", payload)
    write_basic_chart(output_dir / "latency_by_query_type.png", payload)
    write_strategy_chart(output_dir / "strategy_comparison.png", payload)
    return payload

import asyncio
import platform
import subprocess
import time
import uuid
from collections import defaultdict
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
)
from app.evaluation.metrics import retrieval_metrics
from app.retrievers.factory import build_retriever
from app.schemas.search import DenseBackend, FusionMethod, RetrievalStrategy


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


async def run_benchmark(config_path: Path) -> dict[str, Any]:
    settings = get_settings()
    config_text = await asyncio.to_thread(config_path.read_text)
    config: dict[str, Any] = yaml.safe_load(config_text) or {}
    top_k = int(config.get("retrieval", {}).get("top_k", settings.default_top_k))
    strategy_name = str(config.get("retrieval", {}).get("strategy", "hybrid_qdrant"))
    strategy = RetrievalStrategy(strategy_name)
    dense_backend = DenseBackend(config.get("retrieval", {}).get("dense_backend", "qdrant"))
    fusion_method = FusionMethod(config.get("retrieval", {}).get("fusion_method", "rrf"))
    dataset_size = int(config.get("dataset", {}).get("target_chunks", 0))
    run_id = str(uuid.uuid4())

    provider = build_embedding_provider(settings)
    qdrant_client = AsyncQdrantClient(url=settings.qdrant_url)
    rows: list[dict[str, Any]] = []
    try:
        async with get_sessionmaker()() as session:
            queries = (await session.execute(select(BenchmarkQuery))).scalars().all()
            retriever = build_retriever(
                strategy,
                session,
                qdrant_client,
                settings,
                provider,
                dense_backend,
                fusion_method,
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
                metric_values = retrieval_metrics(retrieved_ids, relevant, top_k)
                rows.append(
                    {
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
        "strategy": strategy.value,
        "top_k": top_k,
        "dataset_size": dataset_size,
        "warm_cache": bool(config.get("warm_cache", True)),
        "reranker_enabled": bool(config.get("reranker_enabled", False)),
        "git_commit_hash": git_commit_hash(),
        "hardware_information": hardware_information(),
        "configuration": config,
        "results": rows,
        "by_query_type": summarize_by_query_type(rows, top_k),
    }
    output_dir = Path(settings.benchmark_results_dir) / run_id
    write_json(output_dir / "results.json", payload)
    write_csv(output_dir / "results.csv", rows)
    write_markdown_summary(output_dir / "summary.md", payload)
    write_basic_chart(output_dir / "latency_by_query_type.png", payload)
    return payload

import csv
import json
from collections.abc import Iterable
from pathlib import Path
from statistics import mean
from typing import Any


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    materialized = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not materialized:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(materialized[0].keys()))
        writer.writeheader()
        writer.writerows(materialized)


def write_markdown_summary(path: Path, payload: dict[str, Any]) -> None:
    rows = payload["results"]
    latencies = [float(row["latency_ms"]) for row in rows]
    recall_key = f"recall_at_{payload['top_k']}"
    recall_values = [float(row[recall_key]) for row in rows]
    lines = [
        f"# Benchmark Run {payload['run_id']}",
        "",
        f"- Strategy set: `{payload['strategy']}`",
        f"- Strategies: `{', '.join(payload.get('strategies', [payload['strategy']]))}`",
        f"- Dataset size: `{payload['dataset_size']}` chunks",
        f"- Top K: `{payload['top_k']}`",
        f"- Warm cache: `{payload['warm_cache']}`",
        f"- Reranker enabled: `{payload['reranker_enabled']}`",
        f"- Mean latency: `{mean(latencies) if latencies else 0.0:.2f} ms`",
        f"- Mean Recall@{payload['top_k']}: `{mean(recall_values) if recall_values else 0.0:.3f}`",
        "",
        "## Strategy Summary",
        "",
        "| Strategy | Count | Recall | Precision | Hit Rate | MRR | NDCG | "
        "Mean ms | P95 ms | P99 ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for strategy, summary in payload.get("by_strategy", {}).items():
        lines.append(
            f"| {strategy} | {summary['count']} | "
            f"{summary['mean_recall']:.3f} | {summary['mean_precision']:.3f} | "
            f"{summary['mean_hit_rate']:.3f} | {summary['mean_mrr']:.3f} | "
            f"{summary['mean_ndcg']:.3f} | {summary['mean_latency_ms']:.2f} | "
            f"{summary['p95_latency_ms']:.2f} | {summary['p99_latency_ms']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Query Type Summary",
            "",
            "| Query Type | Count | Mean Recall | Mean Latency ms |",
            "|---|---:|---:|---:|",
        ]
    )
    for query_type, summary in payload["by_query_type"].items():
        lines.append(
            f"| {query_type} | {summary['count']} | "
            f"{summary['mean_recall']:.3f} | {summary['mean_latency_ms']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Context",
            "",
            f"- Config: `{payload['config_name']}`",
            f"- Git commit: `{payload['git_commit_hash']}`",
            f"- Hardware: `{json.dumps(payload['hardware_information'], sort_keys=True)}`",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_basic_chart(path: Path, payload: dict[str, Any]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    rows = payload["results"]
    labels = [row["query_type"] for row in rows]
    latencies = [row["latency_ms"] for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(len(rows)), latencies)
    ax.set_xticks(range(len(rows)), labels, rotation=45, ha="right")
    ax.set_ylabel("Latency ms")
    ax.set_title(
        f"{payload['strategy']} latency, {payload['dataset_size']} chunks, "
        f"warm_cache={payload['warm_cache']}, reranker={payload['reranker_enabled']}"
    )
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_strategy_chart(path: Path, payload: dict[str, Any]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    by_strategy = payload.get("by_strategy", {})
    if not by_strategy:
        return

    strategies = list(by_strategy)
    recall_values = [by_strategy[strategy]["mean_recall"] for strategy in strategies]
    latency_values = [by_strategy[strategy]["p95_latency_ms"] for strategy in strategies]
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, (quality_axis, latency_axis) = plt.subplots(1, 2, figsize=(12, 4))
    quality_axis.bar(range(len(strategies)), recall_values)
    quality_axis.set_xticks(range(len(strategies)), strategies, rotation=35, ha="right")
    quality_axis.set_ylabel(f"Recall@{payload['top_k']}")
    quality_axis.set_title(
        f"Quality, {payload['dataset_size']} chunks, warm_cache={payload['warm_cache']}"
    )

    latency_axis.bar(range(len(strategies)), latency_values)
    latency_axis.set_xticks(range(len(strategies)), strategies, rotation=35, ha="right")
    latency_axis.set_ylabel("P95 latency ms")
    latency_axis.set_title(f"Tail latency, reranker={payload['reranker_enabled']}")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)

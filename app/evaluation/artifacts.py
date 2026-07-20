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
        f"- Strategy: `{payload['strategy']}`",
        f"- Dataset size: `{payload['dataset_size']}` chunks",
        f"- Top K: `{payload['top_k']}`",
        f"- Warm cache: `{payload['warm_cache']}`",
        f"- Reranker enabled: `{payload['reranker_enabled']}`",
        f"- Mean latency: `{mean(latencies) if latencies else 0.0:.2f} ms`",
        f"- Mean Recall@{payload['top_k']}: `{mean(recall_values) if recall_values else 0.0:.3f}`",
        "",
        "## Query Type Summary",
        "",
        "| Query Type | Count | Mean Recall | Mean Latency ms |",
        "|---|---:|---:|---:|",
    ]
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

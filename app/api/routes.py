import asyncio
import json
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request
from qdrant_client import AsyncQdrantClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.db.session import get_session
from app.retrievers.factory import build_retriever
from app.schemas.api import (
    BenchmarkRunListResponse,
    BenchmarkRunRequest,
    BenchmarkRunResponse,
    HealthResponse,
    ReadyResponse,
)
from app.schemas.search import SearchRequest, SearchResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


@router.get("/ready", response_model=ReadyResponse)
async def ready() -> ReadyResponse:
    settings = get_settings()
    return ReadyResponse(
        status="ready",
        database_configured=bool(settings.database_url),
        qdrant_configured=bool(settings.qdrant_url),
    )


@router.post("/search", response_model=SearchResponse)
async def search(
    request_body: SearchRequest,
    request: Request,
    session: AsyncSession = Depends(get_session),
) -> SearchResponse:
    settings = get_settings()
    qdrant_client = AsyncQdrantClient(url=settings.qdrant_url)
    try:
        embedding_provider = request.app.state.embedding_provider
        retriever = build_retriever(
            request_body.strategy,
            session,
            qdrant_client,
            settings,
            embedding_provider,
            request_body.dense_backend,
            request_body.fusion_method,
        )
        results = await retriever.search(
            request_body.query,
            request_body.top_k,
            request_body.filters,
        )
    finally:
        await qdrant_client.close()

    latency = results[0].latency_ms if results else {}
    debug = {"results": [result.model_dump() for result in results]} if request_body.debug else None
    return SearchResponse(
        results=results,
        strategy=request_body.strategy.value,
        latency_ms=latency,
        debug=debug,
    )


@router.post("/benchmark/run", response_model=BenchmarkRunResponse)
async def benchmark_run(request_body: BenchmarkRunRequest) -> BenchmarkRunResponse:
    from app.evaluation.runner import run_benchmark

    config_path = Path(request_body.config_path)
    config_exists = await asyncio.to_thread(config_path.exists)
    if not config_exists:
        raise HTTPException(status_code=404, detail="Benchmark config not found")
    payload = await run_benchmark(config_path)
    results_path = f"benchmarks/results/{payload['run_id']}"
    return BenchmarkRunResponse(
        run_id=str(payload["run_id"]),
        results_path=results_path,
        summary={
            "strategy": payload["strategy"],
            "dataset_size": payload["dataset_size"],
            "top_k": payload["top_k"],
            "by_query_type": payload["by_query_type"],
        },
    )


@router.get("/benchmark/runs", response_model=BenchmarkRunListResponse)
async def benchmark_runs() -> BenchmarkRunListResponse:
    settings = get_settings()
    root = Path(settings.benchmark_results_dir)
    root_exists = await asyncio.to_thread(root.exists)
    if not root_exists:
        return BenchmarkRunListResponse(runs=[])
    paths = await asyncio.to_thread(lambda: list(root.iterdir()))
    runs = sorted(path.name for path in paths if path.is_dir())
    return BenchmarkRunListResponse(runs=runs)


@router.get("/benchmark/runs/{run_id}")
async def benchmark_run_detail(run_id: str) -> dict[str, object]:
    settings = get_settings()
    result_file = Path(settings.benchmark_results_dir) / run_id / "results.json"
    result_exists = await asyncio.to_thread(result_file.exists)
    if not result_exists:
        raise HTTPException(status_code=404, detail="Benchmark run not found")
    content = await asyncio.to_thread(result_file.read_text, encoding="utf-8")
    return json.loads(content)

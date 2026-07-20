import time
from collections.abc import Awaitable, Callable

from fastapi import Request, Response
from prometheus_client import Counter, Gauge, Histogram

REQUEST_COUNT = Counter(
    "rag_api_request_count_total",
    "Total API requests.",
    ["method", "path", "status"],
)
REQUEST_ERRORS = Counter(
    "rag_api_request_error_count_total",
    "Total API request errors.",
    ["method", "path"],
)
REQUEST_LATENCY = Histogram(
    "rag_api_request_latency_seconds",
    "API request latency.",
    ["method", "path"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10),
)
ACTIVE_REQUESTS = Gauge(
    "rag_api_active_requests",
    "Active API requests.",
    ["method", "path"],
)
RETRIEVAL_STRATEGY_COUNT = Counter(
    "rag_retrieval_strategy_count_total",
    "Retrieval strategy invocation count.",
    ["strategy"],
)
RETRIEVAL_STAGE_LATENCY = Histogram(
    "rag_retrieval_stage_latency_seconds",
    "Retrieval stage latency.",
    ["strategy", "stage", "backend"],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5),
)
RETRIEVAL_RESULT_COUNT = Histogram(
    "rag_retrieval_result_count",
    "Returned result count.",
    ["strategy", "backend"],
    buckets=(0, 1, 3, 5, 10, 20, 50, 100),
)
RETRIEVAL_EMPTY_RESULT_COUNT = Counter(
    "rag_retrieval_empty_result_count_total",
    "Empty retrieval result count.",
    ["strategy", "backend"],
)
RETRIEVAL_FILTER_SELECTIVITY = Histogram(
    "rag_retrieval_filter_selectivity",
    "Configured benchmark filter selectivity.",
    ["strategy", "backend"],
    buckets=(0.001, 0.01, 0.1, 0.5, 1.0),
)


def route_template(request: Request) -> str:
    route = request.scope.get("route")
    return getattr(route, "path", request.url.path)


async def prometheus_http_middleware(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    method = request.method
    path = route_template(request)
    ACTIVE_REQUESTS.labels(method=method, path=path).inc()
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        REQUEST_ERRORS.labels(method=method, path=path).inc()
        raise
    finally:
        elapsed = time.perf_counter() - started
        ACTIVE_REQUESTS.labels(method=method, path=path).dec()
        REQUEST_LATENCY.labels(method=method, path=path).observe(elapsed)
    REQUEST_COUNT.labels(method=method, path=path, status=str(response.status_code)).inc()
    if response.status_code >= 500:
        REQUEST_ERRORS.labels(method=method, path=path).inc()
    return response


def observe_retrieval(
    strategy: str,
    backend: str,
    result_count: int,
    latency_ms: dict[str, float],
    filter_selectivity: float | None = None,
) -> None:
    RETRIEVAL_STRATEGY_COUNT.labels(strategy=strategy).inc()
    RETRIEVAL_RESULT_COUNT.labels(strategy=strategy, backend=backend).observe(result_count)
    if result_count == 0:
        RETRIEVAL_EMPTY_RESULT_COUNT.labels(strategy=strategy, backend=backend).inc()
    if filter_selectivity is not None:
        RETRIEVAL_FILTER_SELECTIVITY.labels(strategy=strategy, backend=backend).observe(
            filter_selectivity
        )
    for stage, value_ms in latency_ms.items():
        RETRIEVAL_STAGE_LATENCY.labels(
            strategy=strategy,
            stage=stage,
            backend=backend,
        ).observe(max(value_ms, 0.0) / 1000)

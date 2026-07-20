from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str


class ReadyResponse(BaseModel):
    status: str
    database_configured: bool
    qdrant_configured: bool


class BenchmarkRunRequest(BaseModel):
    config_path: str = "benchmarks/configs/small.yaml"


class BenchmarkRunResponse(BaseModel):
    run_id: str
    results_path: str
    summary: dict[str, object]


class BenchmarkRunListResponse(BaseModel):
    runs: list[str]

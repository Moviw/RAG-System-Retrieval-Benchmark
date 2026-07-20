from pydantic import BaseModel, Field


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


class DocumentWriteRequest(BaseModel):
    document_id: str | None = None
    title: str
    content: str
    source: str = "online-api"
    version: str = "1.0"
    language: str = "en"
    tenant_id: str = "tenant-A"
    department: str = "engineering"
    access_level: str = "internal"
    metadata: dict[str, object] = Field(default_factory=dict)


class DocumentWriteResponse(BaseModel):
    document_id: str
    chunks_indexed: int

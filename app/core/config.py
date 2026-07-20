from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: str = "local"
    log_level: str = "INFO"

    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "rag_benchmark"
    postgres_user: str = "rag"
    postgres_password: SecretStr = Field(default=SecretStr("rag"))
    database_url: str = "postgresql+asyncpg://rag:rag@localhost:5432/rag_benchmark"

    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "rag_chunks"

    embedding_provider: str = "local"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    embedding_batch_size: int = 32
    embedding_device: str = "cpu"
    external_embedding_url: str | None = None
    external_embedding_api_key: SecretStr | None = None

    default_top_k: int = 10
    random_seed: int = 42
    benchmark_results_dir: str = "benchmarks/results"


@lru_cache
def get_settings() -> Settings:
    return Settings()

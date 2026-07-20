from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from prometheus_client import make_asgi_app

from app.api.routes import router
from app.core.config import get_settings
from app.core.logging import configure_logging
from app.embeddings.providers import build_embedding_provider


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings.log_level)
    app.state.embedding_provider = build_embedding_provider(settings)
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="RAG Retrieval Benchmark",
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs" if settings.app_env != "production" else None,
    )
    app.include_router(router)
    app.mount("/metrics", make_asgi_app())
    return app


app = create_app()

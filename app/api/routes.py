from fastapi import APIRouter, Depends, Request
from qdrant_client import AsyncQdrantClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.db.session import get_session
from app.retrievers.factory import build_retriever
from app.schemas.api import HealthResponse, ReadyResponse
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

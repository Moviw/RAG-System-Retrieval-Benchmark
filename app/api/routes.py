from fastapi import APIRouter

from app.core.config import get_settings
from app.schemas.api import HealthResponse, ReadyResponse

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

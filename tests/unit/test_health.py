import pytest

from app.api.routes import health, ready


@pytest.mark.anyio
async def test_health_endpoint() -> None:
    response = await health()

    assert response.status == "ok"


@pytest.mark.anyio
async def test_ready_endpoint() -> None:
    response = await ready()

    assert response.status == "ready"
    assert response.database_configured is True
    assert response.qdrant_configured is True

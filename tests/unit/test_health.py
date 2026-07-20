import asyncio

from app.api.routes import health, ready


def test_health_endpoint() -> None:
    response = asyncio.run(health())

    assert response.status == "ok"


def test_ready_endpoint() -> None:
    response = asyncio.run(ready())

    assert response.status == "ready"
    assert response.database_configured is True
    assert response.qdrant_configured is True

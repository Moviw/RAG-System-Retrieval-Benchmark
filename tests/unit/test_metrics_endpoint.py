from fastapi.testclient import TestClient

from app.main import create_app


def test_metrics_endpoint_exposes_prometheus_text() -> None:
    client = TestClient(create_app())

    response = client.get("/metrics")

    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    assert "rag_api_request_count_total" in response.text

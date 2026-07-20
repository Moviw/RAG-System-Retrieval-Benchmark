import json
from pathlib import Path


def test_grafana_dashboards_are_valid_json() -> None:
    dashboards = list(Path("infra/grafana/dashboards").glob("*.json"))

    assert {path.stem for path in dashboards} == {
        "benchmark-quality-metrics",
        "load-test-overview",
        "postgresql-performance",
        "qdrant-performance",
        "retrieval-comparison",
        "system-overview",
    }
    for path in dashboards:
        payload = json.loads(path.read_text())
        assert payload["title"]
        assert payload["panels"]

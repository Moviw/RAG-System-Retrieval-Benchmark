UV_CACHE_DIR ?= /tmp/uv-cache
PYTHON ?= uv run python

.PHONY: install lint format test typecheck up down migrate seed benchmark-small load-test compose-config

install:
	UV_CACHE_DIR=$(UV_CACHE_DIR) uv sync --extra dev --extra load --extra viz

lint:
	UV_CACHE_DIR=$(UV_CACHE_DIR) uv run ruff check .
	UV_CACHE_DIR=$(UV_CACHE_DIR) uv run ruff format --check .

format:
	UV_CACHE_DIR=$(UV_CACHE_DIR) uv run ruff format .
	UV_CACHE_DIR=$(UV_CACHE_DIR) uv run ruff check --fix .

typecheck:
	UV_CACHE_DIR=$(UV_CACHE_DIR) uv run mypy app scripts

test:
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 UV_CACHE_DIR=$(UV_CACHE_DIR) uv run pytest

up:
	docker compose up --build

down:
	docker compose down

migrate:
	UV_CACHE_DIR=$(UV_CACHE_DIR) uv run alembic upgrade head

seed:
	UV_CACHE_DIR=$(UV_CACHE_DIR) uv run python -m scripts.seed --config benchmarks/configs/small.yaml

benchmark-small:
	UV_CACHE_DIR=$(UV_CACHE_DIR) uv run python -m scripts.benchmark --config benchmarks/configs/small.yaml

load-test:
	UV_CACHE_DIR=$(UV_CACHE_DIR) uv run locust -f tests/performance/locustfile.py --host=http://localhost:8000

compose-config:
	docker compose config

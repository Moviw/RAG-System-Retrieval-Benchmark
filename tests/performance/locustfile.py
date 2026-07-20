import os
import random
import uuid

from locust import HttpUser, between, task

QUERIES = [
    ("What does error PG-204 mean?", "error_code"),
    ("asyncio.create_task usage", "api_function_name"),
    ("How can I run work without blocking the event loop?", "semantic_paraphrase"),
    ("PostgreSQL 16 logical replication timeout", "version_specific"),
    ("Find Japanese documents for tenant A about authentication", "metadata_filtered"),
    ("中文问题检索英文文档 authentication", "multilingual"),
]


class RetrievalUser(HttpUser):
    wait_time = between(0.05, 0.2)
    seed = int(os.getenv("LOAD_TEST_SEED", "42"))
    workload = os.getenv("LOAD_TEST_WORKLOAD", "read-only")

    def on_start(self) -> None:
        random.seed(self.seed)

    @task(100)
    def search(self) -> None:
        query, query_type = random.choice(QUERIES)
        filters = {"tenant_id": "tenant-A"} if query_type == "metadata_filtered" else None
        self.client.post(
            "/search",
            json={
                "query": query,
                "strategy": os.getenv("LOAD_TEST_STRATEGY", "hybrid_qdrant"),
                "top_k": int(os.getenv("LOAD_TEST_TOP_K", "10")),
                "filters": filters,
                "dense_backend": os.getenv("LOAD_TEST_DENSE_BACKEND", "qdrant"),
                "fusion_method": os.getenv("LOAD_TEST_FUSION_METHOD", "rrf"),
                "debug": False,
            },
            name="/search",
        )

    @task(5)
    def insert_or_update(self) -> None:
        if self.workload == "read-only":
            return
        suffix = uuid.uuid4().hex[:8]
        self.client.post(
            "/documents",
            json={
                "title": f"Online workload doc {suffix}",
                "content": (
                    f"Online mixed workload document {suffix} about authentication "
                    "metadata filtering and update visibility latency."
                ),
                "tenant_id": "tenant-A",
                "department": "engineering",
                "language": "en",
                "access_level": "internal",
            },
            name="/documents",
        )

    @task(5)
    def delete(self) -> None:
        if self.workload != "mixed":
            return
        self.client.delete(f"/documents/{uuid.uuid4()}", name="/documents/{id}")

import hashlib
import random
import uuid
from dataclasses import dataclass
from typing import Any

TOPICS = [
    ("authentication", "OAuth token validation, tenant login, access policy, session rotation"),
    ("postgres", "PostgreSQL 16 logical replication timeout, WAL, index scans, buffer cache"),
    ("asyncio", "asyncio.create_task usage, event loop scheduling, non-blocking background work"),
    ("qdrant", "Qdrant HNSW search, payload filter, collection optimizer, vector segment"),
    ("billing", "invoice reconciliation, idempotency key, payment retry, API error BILL-409"),
    ("kubernetes", "pod readiness probe, resource limit, service discovery, deployment rollback"),
    ("observability", "Prometheus histogram, Grafana dashboard, p95 latency, error budget"),
    ("security", "role based access control, audit log, secret rotation, policy enforcement"),
]

QUERY_TYPES = [
    "exact_identifier",
    "error_code",
    "api_function_name",
    "semantic_paraphrase",
    "conceptual_question",
    "mixed_lexical_semantic",
    "version_specific",
    "metadata_filtered",
    "multilingual",
    "hard_negative",
]


@dataclass(frozen=True)
class SyntheticDocument:
    id: uuid.UUID
    title: str
    source: str
    version: str
    language: str
    tenant_id: str
    department: str
    access_level: str
    checksum: str
    metadata: dict[str, Any]
    content: str


@dataclass(frozen=True)
class SyntheticQuery:
    id: str
    query_text: str
    query_type: str
    language: str
    filters: dict[str, Any]
    expected_relevant_document_ids: list[str]
    expected_relevant_chunk_ids: list[str]
    label_source: str


def generate_documents(
    target_chunks: int,
    seed: int,
    chunks_per_document: int = 4,
) -> list[SyntheticDocument]:
    rng = random.Random(seed)
    document_count = max(1, target_chunks // chunks_per_document)
    documents: list[SyntheticDocument] = []
    for index in range(document_count):
        topic, description = TOPICS[index % len(TOPICS)]
        tenant = f"tenant-{chr(ord('A') + index % 5)}"
        department = ["engineering", "support", "finance", "security"][index % 4]
        language = ["en", "ja", "zh"][index % 3]
        version = f"{14 + index % 4}.{index % 10}"
        access_level = ["public", "internal", "restricted"][index % 3]
        doc_uuid = uuid.uuid5(uuid.NAMESPACE_URL, f"rag-benchmark:{seed}:{index}:{topic}")
        identifier = f"{topic.upper()}-{1000 + index}"
        error_code = f"{topic[:2].upper()}-{200 + index % 50}"
        paragraphs = [
            f"{identifier} describes {topic} operations for {tenant} in {department}.",
            f"The service version is {version} and the access level is {access_level}.",
            f"Primary details include {description}.",
            f"Error code {error_code} maps to a controlled recovery procedure.",
            "Use deterministic retries, structured logs, and metadata filters "
            "for reproducible tests.",
        ]
        repeated = " ".join(paragraphs * chunks_per_document)
        checksum = hashlib.sha256(repeated.encode("utf-8")).hexdigest()
        documents.append(
            SyntheticDocument(
                id=doc_uuid,
                title=f"{topic.title()} Runbook {index}",
                source="synthetic-tech-docs",
                version=version,
                language=language,
                tenant_id=tenant,
                department=department,
                access_level=access_level,
                checksum=checksum,
                metadata={
                    "topic": topic,
                    "identifier": identifier,
                    "error_code": error_code,
                    "generated": True,
                },
                content=repeated,
            )
        )
    rng.shuffle(documents)
    return documents


def generate_queries(documents: list[SyntheticDocument]) -> list[SyntheticQuery]:
    selected = sorted(documents, key=lambda doc: str(doc.id))[: max(10, min(30, len(documents)))]
    queries: list[SyntheticQuery] = []
    for idx, query_type in enumerate(QUERY_TYPES):
        doc = selected[idx % len(selected)]
        topic = doc.metadata["topic"]
        identifier = doc.metadata["identifier"]
        error_code = doc.metadata["error_code"]
        text_by_type = {
            "exact_identifier": f"{identifier}",
            "error_code": f"What does error {error_code} mean?",
            "api_function_name": "asyncio.create_task usage",
            "semantic_paraphrase": f"How do I handle {topic} without blocking the service?",
            "conceptual_question": f"What operational practices matter for {topic} reliability?",
            "mixed_lexical_semantic": f"{identifier} recovery procedure for production incidents",
            "version_specific": f"{topic} behavior in version {doc.version}",
            "metadata_filtered": f"Find {doc.language} documents for {doc.tenant_id} about {topic}",
            "multilingual": f"中文问题检索英文文档 {topic} authentication reliability",
            "hard_negative": "nonexistent quantum compiler opcode ZX-000",
        }
        expected_docs = [] if query_type == "hard_negative" else [str(doc.id)]
        queries.append(
            SyntheticQuery(
                id=f"q-{idx:03d}-{query_type}",
                query_text=text_by_type[query_type],
                query_type=query_type,
                language="zh" if query_type == "multilingual" else "en",
                filters={
                    "tenant_id": doc.tenant_id,
                    "language": doc.language,
                }
                if query_type == "metadata_filtered"
                else {},
                expected_relevant_document_ids=expected_docs,
                expected_relevant_chunk_ids=[],
                label_source="automatic",
            )
        )
    return queries

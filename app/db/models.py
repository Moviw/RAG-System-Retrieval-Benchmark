import uuid
from datetime import datetime
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, ForeignKey, Index, Integer, String, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, TSVECTOR, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    source: Mapped[str] = mapped_column(String(300), nullable=False)
    version: Mapped[str] = mapped_column(String(80), nullable=False)
    language: Mapped[str] = mapped_column(String(16), nullable=False)
    tenant_id: Mapped[str] = mapped_column(String(80), nullable=False, index=True)
    department: Mapped[str] = mapped_column(String(80), nullable=False, index=True)
    access_level: Mapped[str] = mapped_column(String(40), nullable=False, index=True)
    checksum: Mapped[str] = mapped_column(String(128), nullable=False, unique=True)
    metadata_: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    chunks: Mapped[list["Chunk"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )


class Chunk(Base):
    __tablename__ = "chunks"
    __table_args__ = (
        UniqueConstraint("document_id", "chunk_index", name="uq_chunks_document_index"),
        Index("ix_chunks_content_tsv", "content_tsv", postgresql_using="gin"),
        Index("ix_chunks_metadata", "metadata", postgresql_using="gin"),
        Index(
            "ix_chunks_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False)
    metadata_: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )
    content_tsv: Mapped[str] = mapped_column(TSVECTOR, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(Vector(384), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    document: Mapped[Document] = relationship(back_populates="chunks")


class BenchmarkQuery(Base):
    __tablename__ = "benchmark_queries"

    id: Mapped[str] = mapped_column(String(120), primary_key=True)
    query_text: Mapped[str] = mapped_column(Text, nullable=False)
    query_type: Mapped[str] = mapped_column(String(80), nullable=False, index=True)
    language: Mapped[str] = mapped_column(String(16), nullable=False)
    filters: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    expected_relevant_document_ids: Mapped[list[str]] = mapped_column(ARRAY(String), nullable=False)
    expected_relevant_chunk_ids: Mapped[list[str]] = mapped_column(ARRAY(String), nullable=False)
    label_source: Mapped[str] = mapped_column(String(40), nullable=False)


class BenchmarkRun(Base):
    __tablename__ = "benchmark_runs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    git_commit_hash: Mapped[str] = mapped_column(String(80), nullable=False)
    configuration: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    hardware_information: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    dataset_size: Mapped[int] = mapped_column(Integer, nullable=False)
    retrieval_strategy: Mapped[str] = mapped_column(String(80), nullable=False)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class BenchmarkResult(Base):
    __tablename__ = "benchmark_results"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("benchmark_runs.id", ondelete="CASCADE"), nullable=False
    )
    query_id: Mapped[str] = mapped_column(
        String(120), ForeignKey("benchmark_queries.id", ondelete="CASCADE"), nullable=False
    )
    retrieval_latency_ms: Mapped[float] = mapped_column(nullable=False)
    retrieved_ids: Mapped[list[str]] = mapped_column(ARRAY(String), nullable=False)
    recall_at_k: Mapped[float] = mapped_column(nullable=False)
    precision_at_k: Mapped[float] = mapped_column(nullable=False)
    mrr: Mapped[float] = mapped_column(nullable=False)
    ndcg_at_k: Mapped[float] = mapped_column(nullable=False)
    hit_rate_at_k: Mapped[float] = mapped_column(nullable=False)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

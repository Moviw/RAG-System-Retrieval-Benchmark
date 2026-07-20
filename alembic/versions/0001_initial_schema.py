"""initial schema

Revision ID: 0001_initial_schema
Revises:
Create Date: 2026-07-20 00:00:00.000000
"""

from collections.abc import Sequence

import pgvector.sqlalchemy
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "0001_initial_schema"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.create_table(
        "documents",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("title", sa.String(length=500), nullable=False),
        sa.Column("source", sa.String(length=300), nullable=False),
        sa.Column("version", sa.String(length=80), nullable=False),
        sa.Column("language", sa.String(length=16), nullable=False),
        sa.Column("tenant_id", sa.String(length=80), nullable=False),
        sa.Column("department", sa.String(length=80), nullable=False),
        sa.Column("access_level", sa.String(length=40), nullable=False),
        sa.Column("checksum", sa.String(length=128), nullable=False),
        sa.Column("metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("checksum"),
    )
    op.create_index(op.f("ix_documents_access_level"), "documents", ["access_level"], unique=False)
    op.create_index(op.f("ix_documents_department"), "documents", ["department"], unique=False)
    op.create_index(op.f("ix_documents_tenant_id"), "documents", ["tenant_id"], unique=False)

    op.create_table(
        "chunks",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("document_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("token_count", sa.Integer(), nullable=False),
        sa.Column("metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("content_tsv", postgresql.TSVECTOR(), nullable=False),
        sa.Column("embedding", pgvector.sqlalchemy.Vector(dim=384), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["document_id"], ["documents.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("document_id", "chunk_index", name="uq_chunks_document_index"),
    )
    op.create_index(op.f("ix_chunks_document_id"), "chunks", ["document_id"], unique=False)
    op.create_index(
        "ix_chunks_content_tsv",
        "chunks",
        ["content_tsv"],
        unique=False,
        postgresql_using="gin",
    )
    op.create_index(
        "ix_chunks_metadata",
        "chunks",
        ["metadata"],
        unique=False,
        postgresql_using="gin",
    )
    op.create_index(
        "ix_chunks_embedding_hnsw",
        "chunks",
        ["embedding"],
        unique=False,
        postgresql_using="hnsw",
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )

    op.create_table(
        "benchmark_queries",
        sa.Column("id", sa.String(length=120), nullable=False),
        sa.Column("query_text", sa.Text(), nullable=False),
        sa.Column("query_type", sa.String(length=80), nullable=False),
        sa.Column("language", sa.String(length=16), nullable=False),
        sa.Column("filters", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("expected_relevant_document_ids", postgresql.ARRAY(sa.String()), nullable=False),
        sa.Column("expected_relevant_chunk_ids", postgresql.ARRAY(sa.String()), nullable=False),
        sa.Column("label_source", sa.String(length=40), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_benchmark_queries_query_type"), "benchmark_queries", ["query_type"], unique=False
    )

    op.create_table(
        "benchmark_runs",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("git_commit_hash", sa.String(length=80), nullable=False),
        sa.Column("configuration", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("hardware_information", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("dataset_size", sa.Integer(), nullable=False),
        sa.Column("retrieval_strategy", sa.String(length=80), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("ended_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "benchmark_results",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("run_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("query_id", sa.String(length=120), nullable=False),
        sa.Column("retrieval_latency_ms", sa.Float(), nullable=False),
        sa.Column("retrieved_ids", postgresql.ARRAY(sa.String()), nullable=False),
        sa.Column("recall_at_k", sa.Float(), nullable=False),
        sa.Column("precision_at_k", sa.Float(), nullable=False),
        sa.Column("mrr", sa.Float(), nullable=False),
        sa.Column("ndcg_at_k", sa.Float(), nullable=False),
        sa.Column("hit_rate_at_k", sa.Float(), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(["query_id"], ["benchmark_queries.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["run_id"], ["benchmark_runs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )


def downgrade() -> None:
    op.drop_table("benchmark_results")
    op.drop_table("benchmark_runs")
    op.drop_index(op.f("ix_benchmark_queries_query_type"), table_name="benchmark_queries")
    op.drop_table("benchmark_queries")
    op.drop_index("ix_chunks_embedding_hnsw", table_name="chunks", postgresql_using="hnsw")
    op.drop_index("ix_chunks_metadata", table_name="chunks", postgresql_using="gin")
    op.drop_index("ix_chunks_content_tsv", table_name="chunks", postgresql_using="gin")
    op.drop_index(op.f("ix_chunks_document_id"), table_name="chunks")
    op.drop_table("chunks")
    op.drop_index(op.f("ix_documents_tenant_id"), table_name="documents")
    op.drop_index(op.f("ix_documents_department"), table_name="documents")
    op.drop_index(op.f("ix_documents_access_level"), table_name="documents")
    op.drop_table("documents")

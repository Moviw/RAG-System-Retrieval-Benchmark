import pytest

from app.retrievers.filters import build_sql_filter_clause, vector_literal


def test_build_sql_filter_clause_uses_whitelisted_metadata_fields() -> None:
    clause, params = build_sql_filter_clause({"tenant_id": "tenant-A", "language": "en"})

    assert "tenant_id" in clause
    assert params == {"filter_tenant_id": "tenant-A", "filter_language": "en"}


def test_build_sql_filter_clause_rejects_unknown_field() -> None:
    with pytest.raises(ValueError, match="Unsupported filter"):
        build_sql_filter_clause({"query_text": "leak"})


def test_vector_literal_formats_float_list() -> None:
    assert vector_literal([1.0, -0.5]) == "[1.00000000,-0.50000000]"

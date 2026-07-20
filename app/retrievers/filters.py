from typing import Any

ALLOWED_FILTERS = {
    "tenant_id",
    "department",
    "language",
    "version",
    "access_level",
}


def build_sql_filter_clause(
    filters: dict[str, object] | None,
    *,
    alias: str = "c",
) -> tuple[str, dict[str, Any]]:
    if not filters:
        return "", {}

    clauses: list[str] = []
    params: dict[str, Any] = {}
    for key, value in filters.items():
        if key not in ALLOWED_FILTERS:
            raise ValueError(f"Unsupported filter field: {key}")
        param_name = f"filter_{key}"
        if isinstance(value, list):
            clauses.append(f"{alias}.metadata ->> '{key}' = ANY(:{param_name})")
            params[param_name] = [str(item) for item in value]
        else:
            clauses.append(f"{alias}.metadata ->> '{key}' = :{param_name}")
            params[param_name] = str(value)
    return " AND " + " AND ".join(clauses), params


def vector_literal(vector: list[float]) -> str:
    return "[" + ",".join(f"{value:.8f}" for value in vector) + "]"

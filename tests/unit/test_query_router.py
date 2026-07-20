from app.retrievers.router import RuleBasedQueryRouter, extract_query_features


def test_identifier_query_routes_to_lexical() -> None:
    decision = RuleBasedQueryRouter().route("AUTH-1001")

    assert decision.strategy == "lexical"
    assert decision.features.has_exact_identifier is True


def test_identifier_with_intent_routes_to_hybrid() -> None:
    decision = RuleBasedQueryRouter().route("What does error PG-204 mean?")

    assert decision.strategy == "hybrid"
    assert decision.features.has_error_code is True


def test_conceptual_query_routes_to_dense() -> None:
    decision = RuleBasedQueryRouter().route("How can I run work without blocking the event loop?")

    assert decision.strategy == "dense"


def test_feature_extraction_detects_function_and_version() -> None:
    features = extract_query_features("asyncio.create_task usage in PostgreSQL 16.2")

    assert features.has_function_name is True
    assert features.has_version is True

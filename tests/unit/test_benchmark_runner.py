from app.evaluation.runner import parse_strategy_specs, percentile, unique_preserving_order


def test_parse_strategy_specs_expands_hybrid_suffixes() -> None:
    specs = parse_strategy_specs(
        {
            "strategies": [
                "lexical",
                "dense_pgvector",
                "hybrid_qdrant_rrf",
                "hybrid_pgvector_weighted",
            ]
        }
    )

    assert [spec.name for spec in specs] == [
        "lexical",
        "dense_pgvector",
        "hybrid_qdrant_rrf",
        "hybrid_pgvector_weighted",
    ]
    assert specs[1].dense_backend == "pgvector"
    assert specs[2].dense_backend == "qdrant"
    assert specs[2].fusion_method == "rrf"
    assert specs[3].dense_backend == "pgvector"
    assert specs[3].fusion_method == "weighted"


def test_percentile_interpolates_sorted_values() -> None:
    assert percentile([10.0, 20.0, 30.0], 0.5) == 20.0
    assert percentile([10.0, 20.0], 0.95) == 19.5


def test_unique_preserving_order_removes_duplicate_retrieved_ids() -> None:
    assert unique_preserving_order(["doc-a", "doc-a", "doc-b", "doc-a"]) == [
        "doc-a",
        "doc-b",
    ]

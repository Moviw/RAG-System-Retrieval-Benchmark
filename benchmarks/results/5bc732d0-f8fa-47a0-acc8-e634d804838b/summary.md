# Benchmark Run 5bc732d0-f8fa-47a0-acc8-e634d804838b

- Strategy set: `multi`
- Strategies: `lexical, dense_pgvector, dense_qdrant, hybrid_pgvector_rrf, hybrid_qdrant_rrf, hybrid_pgvector_weighted, hybrid_qdrant_weighted`
- Dataset size: `10000` chunks
- Top K: `10`
- Warm cache: `True`
- Reranker enabled: `False`
- Mean latency: `12.60 ms`
- Mean Recall@10: `0.286`

## Strategy Summary

| Strategy | Count | Recall | Precision | Hit Rate | MRR | NDCG | Mean ms | P95 ms | P99 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| lexical | 10.0 | 0.200 | 0.010 | 0.200 | 0.200 | 0.200 | 1.52 | 6.26 | 8.76 |
| dense_pgvector | 10.0 | 0.300 | 0.030 | 0.300 | 0.137 | 0.174 | 9.23 | 17.23 | 19.66 |
| dense_qdrant | 10.0 | 0.300 | 0.030 | 0.300 | 0.137 | 0.174 | 12.63 | 29.79 | 41.56 |
| hybrid_pgvector_rrf | 10.0 | 0.300 | 0.030 | 0.300 | 0.137 | 0.174 | 9.91 | 16.33 | 18.48 |
| hybrid_qdrant_rrf | 10.0 | 0.300 | 0.030 | 0.300 | 0.137 | 0.174 | 14.52 | 31.23 | 43.38 |
| hybrid_pgvector_weighted | 10.0 | 0.300 | 0.030 | 0.300 | 0.137 | 0.174 | 10.44 | 16.65 | 19.13 |
| hybrid_qdrant_weighted | 10.0 | 0.300 | 0.030 | 0.300 | 0.137 | 0.174 | 29.99 | 111.11 | 154.13 |

## Query Type Summary

| Query Type | Count | Mean Recall | Mean Latency ms |
|---|---:|---:|---:|
| exact_identifier | 7.0 | 1.000 | 9.78 |
| error_code | 7.0 | 0.857 | 7.41 |
| api_function_name | 7.0 | 0.000 | 10.80 |
| semantic_paraphrase | 7.0 | 0.000 | 8.35 |
| conceptual_question | 7.0 | 0.000 | 7.89 |
| mixed_lexical_semantic | 7.0 | 0.857 | 7.52 |
| version_specific | 7.0 | 0.000 | 7.49 |
| metadata_filtered | 7.0 | 0.000 | 28.01 |
| multilingual | 7.0 | 0.000 | 30.45 |
| hard_negative | 7.0 | 0.143 | 8.33 |

## Context

- Config: `small`
- Git commit: `ce80051488d672bf2c837de6633c2557de95e21e`
- Hardware: `{"platform": "Linux-6.8.0-111-generic-x86_64-with-glibc2.35", "processor": "x86_64", "python": "3.12.13"}`

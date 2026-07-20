# Benchmark Run 0aa30fb2-970b-40d7-9a3c-41d3681ff6cc

- Strategy: `hybrid_qdrant`
- Dataset size: `10000` chunks
- Top K: `10`
- Warm cache: `True`
- Reranker enabled: `False`
- Mean latency: `10.11 ms`
- Mean Recall@10: `0.300`

## Query Type Summary

| Query Type | Count | Mean Recall | Mean Latency ms |
|---|---:|---:|---:|
| exact_identifier | 1.0 | 1.000 | 9.90 |
| error_code | 1.0 | 0.000 | 5.01 |
| api_function_name | 1.0 | 0.000 | 18.14 |
| semantic_paraphrase | 1.0 | 0.000 | 4.68 |
| conceptual_question | 1.0 | 0.000 | 4.82 |
| mixed_lexical_semantic | 1.0 | 1.000 | 4.67 |
| version_specific | 1.0 | 0.000 | 4.57 |
| metadata_filtered | 1.0 | 1.000 | 40.05 |
| multilingual | 1.0 | 0.000 | 4.67 |
| hard_negative | 1.0 | 0.000 | 4.64 |

## Context

- Config: `small`
- Git commit: `d356c7b832bd68db0fc2d62c1ca0a7a758ba77b3`
- Hardware: `{"platform": "Linux-6.8.0-111-generic-x86_64-with-glibc2.35", "processor": "x86_64", "python": "3.12.13"}`

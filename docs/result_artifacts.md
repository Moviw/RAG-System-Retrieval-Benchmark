# Result Artifacts

Each benchmark run writes:

- `results.json`
- `results.csv`
- `summary.md`
- `latency_by_query_type.png` when matplotlib is installed

Planned comparison charts:

- Dataset Size vs Recall@10
- Dataset Size vs P95 Latency
- Dataset Size vs Index Build Time
- Dataset Size vs Storage Usage
- Concurrency vs QPS
- Concurrency vs P99 Latency
- Filter Selectivity vs Recall@10
- Filter Selectivity vs P95 Latency
- Retrieval Strategy by Query Type
- Quality-Latency Pareto Frontier

Charts must include config, hardware, dataset size, cache mode, and reranker status.

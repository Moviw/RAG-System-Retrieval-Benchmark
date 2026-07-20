# Benchmark Methodology

This benchmark treats retrieval quality as ground-truth evaluation, not as LLM answer preference. Generation can be added later, but it must not replace relevance labels.

## Fair Comparisons

- Use identical documents, chunks, metadata, embeddings, query sets, and `top_k`.
- Run warm-up before measuring.
- Use the same cache mode for every strategy in a comparison.
- Keep resource limits declared in `docker-compose.yml`.
- Save the YAML config and Git commit hash with every result.

## Metadata Filtering

Filter benchmarks should cover 100 percent, 50 percent, 10 percent, 1 percent, and 0.1 percent selectivity. PostgreSQL experiments should capture `EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON)` for representative queries.

## Reporting

Every report should include quality, latency, throughput, storage, indexing time, and operational notes. A faster retriever with materially worse Recall@10 is not automatically better.

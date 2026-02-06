# llm-eval

Statistical RAG evaluation framework with golden-set metrics and drift detection.

> Most RAG systems fail silently. This catches regression before users do.

## Why llm-eval?

| Feature | llm-eval | RAGAS | DeepEval |
|---------|----------|-------|----------|
| Deterministic (no LLM-as-judge) | Yes | No | No |
| Per-query statistical tests | Yes | No | No |
| Bootstrap confidence intervals | Yes | No | No |
| McNemar/Fisher exact tests | Yes | No | No |
| Drift detection with severity | Yes | No | Partial |
| Zero torch/sklearn dependency | Yes | No | No |
| Golden-set-based | Yes | Optional | Optional |

## Quick Start

```python
from llm_eval import GoldenSet, RetrievedItem
from llm_eval.runner import run_evaluation

# 1. Load your golden set
golden = GoldenSet.from_json("golden_queries.json")

# 2. Implement the adapter protocol (2 methods)
class MyAdapter:
    @property
    def name(self) -> str:
        return "my-rag-system"

    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievedItem]:
        results = my_search(query, limit=top_k)
        return [RetrievedItem(id=r.id, rank=i+1, score=r.score)
                for i, r in enumerate(results)]

# 3. Run evaluation
run = run_evaluation(golden, MyAdapter(), top_k=10)
print(f"Hit Rate: {run.metrics['hit_rate']:.1%}")
print(f"MRR:      {run.metrics['mrr']:.3f}")
print(f"NDCG@k:   {run.metrics['ndcg_at_k']:.3f}")
```

## Installation

```bash
pip install llm-eval

# With optional scipy (faster Fisher exact test)
pip install llm-eval[scipy]
```

## CLI

```bash
# Validate a golden set
llm-eval validate golden_queries.json

# Run evaluation
llm-eval run golden_queries.json --adapter my-adapter --top-k 10

# Set baseline
llm-eval baseline set results.json --notes "v1.0 release"

# Detect drift (exits 1 on critical regression)
llm-eval drift golden_queries.json --adapter my-adapter --exit-code

# Compare two runs
llm-eval compare run_a.json run_b.json --format markdown
```

## Metrics

### Ranking Metrics
- **MRR** (Mean Reciprocal Rank): Average of 1/rank of first relevant document
- **NDCG@k**: Normalized Discounted Cumulative Gain (supports graded relevance)
- **Hit Rate@k**: Fraction of queries with at least one relevant result in top-k
- **Precision@k**: Fraction of top-k results that are relevant
- **MAP** (Mean Average Precision): Area under precision-recall curve

### Agreement Metrics
- **Cohen's Kappa**: Chance-corrected inter-rater agreement
- **Weighted Kappa**: Ordinal-aware agreement (linear/quadratic weights)

### Statistical Tests
- **Bootstrap CI**: Non-parametric confidence intervals for any metric
- **Paired permutation test**: Compare two systems on the same queries
- **Fisher's exact test**: 2x2 contingency for small samples (n < 50)
- **McNemar's test**: Paired binary outcomes (hit/miss changes)

### Drift Detection
- Compares current run against a stored baseline
- Per-metric severity classification:
  - **INFO**: No significant change
  - **WARNING**: >5% drop AND p < 0.10
  - **CRITICAL**: >10% drop AND p < 0.05

## For Statisticians

The key design choice: **per-query metric storage enables paired tests**.

Most evaluation frameworks only store aggregate metrics. This means comparing
two systems requires unpaired tests (chi-squared, two-sample z-test) that are
far less powerful. By storing per-query results, llm-eval enables:

1. **Paired permutation test** for continuous metrics (MRR, NDCG) — no
   distributional assumptions, exact p-values
2. **McNemar's test** for hit rate — proper test for paired binary outcomes
3. **Bootstrap CI** — works for any metric without normality assumption
4. **Fisher's exact test** — chosen over chi-squared because golden sets
   are typically 20-100 queries, where chi-squared expected cell counts < 5

## Architecture

```
Golden Set ─→ Adapter ─→ Runner ─→ EvalRun ─→ Reporter
                                      │
                                      ├─→ Baseline Store
                                      └─→ Drift Detector ─→ Alerts
```

Adapters implement a simple sync Protocol (2 methods). See [docs/design.md](docs/design.md).

## CI Integration

```yaml
# .github/workflows/eval.yml
- name: Run RAG evaluation
  run: llm-eval drift golden.json --adapter my-adapter --exit-code
```

The `--exit-code` flag exits 1 on CRITICAL drift, failing your CI pipeline
when retrieval quality regresses significantly.

## License

MIT

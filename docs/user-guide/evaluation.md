# Running Evaluations

ir-eval supports two evaluation paths. Both produce an `EvalRun` object with identical structure.

## Path 1: Results-First (Recommended)

Evaluate from pre-computed JSON results. This is the primary path — it's deterministic, fast, and requires no live retrieval system.

### Prepare Results

Export your retrieval results as JSON:

```json
{
  "adapter_name": "my-system",
  "results": [
    {
      "query": "What is causal inference?",
      "retrieved": [
        {"id": "doc_001", "rank": 1, "score": 0.95},
        {"id": "doc_042", "rank": 2, "score": 0.87}
      ]
    }
  ]
}
```

### Run Evaluation

**Python API:**

```python
from ir_eval import GoldenSet, ResultSet, evaluate_from_results

golden = GoldenSet.from_file("golden.json")
results = ResultSet.from_file("results.json")
run = evaluate_from_results(golden, results, top_k=10)

# Access metrics
print(f"MRR: {run.metrics['mrr']:.3f}")
print(f"NDCG@10: {run.metrics['ndcg_at_k']:.3f}")
print(f"Hit Rate: {run.metrics['hit_rate']:.1%}")
```

**CLI:**

```bash
ir-eval evaluate results.json --golden golden.json --format console
```

## Path 2: Adapter-Based (Live)

Evaluate against a running retrieval system via the `RetrievalAdapter` protocol.

```python
from ir_eval import GoldenSet, run_evaluation

golden = GoldenSet.from_file("golden.json")
adapter = MyAdapter()  # implements RetrievalAdapter
run = run_evaluation(golden, adapter, top_k=10)
```

**CLI with entry-point adapter:**

```bash
ir-eval run golden.json --adapter research-kb --top-k 10
```

## EvalRun Structure

Both paths produce an `EvalRun` with:

- `id` — Unique run identifier
- `golden_set_name` — Which golden set was used
- `adapter_name` — Which system was evaluated
- `timestamp` — When the evaluation ran
- `query_results` — Per-query metrics (enables paired tests)
- `metrics` — Aggregate metrics dictionary
- `config` — Evaluation configuration

## Output Formats

```bash
# Rich terminal tables
ir-eval evaluate results.json --golden golden.json --format console

# Markdown (for GitHub PRs)
ir-eval evaluate results.json --golden golden.json --format markdown

# JSON (for CI/CD)
ir-eval evaluate results.json --golden golden.json --format json

# Save run for later comparison/baseline
ir-eval evaluate results.json --golden golden.json --output run.json
```

## Per-Query Metrics

Each `QueryResult` in the run contains:

| Metric | Description |
|--------|-------------|
| `hit` | Whether any relevant doc appeared in top-k |
| `reciprocal_rank` | 1/rank of first relevant doc (0 if none) |
| `ndcg` | NDCG@k for this query |
| `precision_at_k` | Fraction of top-k that are relevant |
| `average_precision` | Area under precision-recall curve |
| `latency_ms` | Query latency (adapter path only) |

Storing per-query metrics is a key design decision — it enables paired statistical tests that are far more powerful than comparing aggregate numbers.

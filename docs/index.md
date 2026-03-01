# ir-eval

**Statistical retrieval evaluation framework with golden-set metrics and drift detection.**

ir-eval provides deterministic, reproducible evaluation for RAG and information retrieval systems. Instead of LLM-as-judge approaches, it uses human-curated golden sets with paired statistical tests — designed for CI/CD pipelines where regression detection matters.

## Key Features

::::{grid} 2
:gutter: 3

:::{grid-item-card} Results-First Evaluation
:link: user-guide/evaluation
:link-type: doc

Evaluate from pre-computed JSON results — no live adapter required. Deterministic, fast, CI-friendly.
:::

:::{grid-item-card} Statistical Drift Detection
:link: user-guide/drift-detection
:link-type: doc

Paired bootstrap tests, Fisher exact, and McNemar's test. 2D severity classification: magnitude × significance.
:::

:::{grid-item-card} 8 Retrieval Metrics
:link: metrics-reference
:link-type: doc

MRR, NDCG@k, Hit Rate, Precision@k, MAP, Cohen's Kappa — all with bootstrap confidence intervals.
:::

:::{grid-item-card} CLI + Python API
:link: cli
:link-type: doc

Full Typer CLI with 7 commands. Console, Markdown, and JSON reporters for CI integration.
:::
::::

## Quick Example

```python
from ir_eval import GoldenSet, ResultSet, evaluate_from_results

golden = GoldenSet.from_file("golden.json")
results = ResultSet.from_file("results.json")
run = evaluate_from_results(golden, results, top_k=10)

print(f"MRR: {run.metrics['mrr']:.3f}")
print(f"Hit Rate@10: {run.metrics['hit_rate']:.1%}")
```

```{toctree}
:maxdepth: 2
:caption: Getting Started

getting-started
```

```{toctree}
:maxdepth: 2
:caption: User Guide

user-guide/golden-sets
user-guide/evaluation
user-guide/drift-detection
user-guide/ci-integration
user-guide/custom-adapters
```

```{toctree}
:maxdepth: 2
:caption: Reference

cli
metrics-reference
architecture
```

```{toctree}
:maxdepth: 2
:caption: API Reference

api/index
api/types
api/runner
api/metrics
api/drift
api/reporters
```

```{toctree}
:maxdepth: 1
:caption: Project

changelog
glossary
```

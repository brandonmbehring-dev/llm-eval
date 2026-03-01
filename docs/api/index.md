# API Reference

Complete Python API reference for ir-eval, auto-generated from docstrings.

## Package Overview

| Module | Purpose |
|--------|---------|
| {doc}`types` | Core data types (GoldenSet, EvalRun, etc.) |
| {doc}`runner` | Evaluation orchestration |
| {doc}`metrics` | Ranking, agreement, and confidence metrics |
| {doc}`drift` | Baseline storage and drift detection |
| {doc}`reporters` | Output formatters (console, markdown, JSON) |

## Public API

The top-level `ir_eval` package exports these types and functions:

```python
from ir_eval import (
    # Data types
    GoldenSet, GoldenQuery, Difficulty,
    EvalRun, QueryResult,
    ResultSet, ResultEntry,
    RetrievedItem,
    Baseline, DriftResult, DriftSeverity,
    # Protocol
    RetrievalAdapter,
    # Functions
    run_evaluation,
    evaluate_from_results,
)
```

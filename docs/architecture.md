# Architecture & Design

## Core Architecture

```
Golden Set ──→ Adapter Protocol ──→ Runner ──→ EvalRun
                                       │
                                       ├──→ Baseline Store (.ir-eval/baselines/)
                                       ├──→ Drift Detector (statistical tests)
                                       ├──→ Reporters (console/markdown/JSON)
                                       └──→ CLI (Typer)
```

## Package Structure

```
src/ir_eval/
├── __init__.py           # Public API exports
├── types.py              # 7 frozen dataclasses (GoldenSet, EvalRun, etc.)
├── runner.py             # run_evaluation, evaluate_from_results
├── adapter.py            # RetrievalAdapter protocol
├── compare.py            # Run-to-run comparison
├── cli/main.py           # 7 CLI commands (Typer)
├── metrics/
│   ├── ranking.py        # MRR, NDCG, Hit Rate, Precision@k, MAP
│   ├── agreement.py      # Cohen's Kappa, weighted Kappa
│   └── confidence.py     # Bootstrap CI, permutation test, Fisher, McNemar
├── drift/
│   ├── baseline.py       # BaselineStore (JSON persistence)
│   ├── detector.py       # DriftDetector (statistical comparison)
│   └── alerts.py         # ExitCodeAlert (CI integration)
└── reporters/
    ├── console.py        # Rich terminal tables
    ├── markdown.py       # Markdown reports
    └── json_reporter.py  # Machine-readable JSON
```

## Key Design Decisions

### Golden-set-based, not LLM-as-judge

**Decision**: Evaluate against human-curated golden sets with known relevant documents.

**Rationale**: Deterministic, reproducible, cheap to run. LLM-as-judge (RAGAS-style) introduces non-determinism, is expensive at scale, and conflates evaluator quality with system quality.

**Tradeoff**: Requires upfront effort to build golden sets. Worth it for systems where regression detection matters.

### Per-query metric storage

**Decision**: Store all metrics at the query level, not just aggregates.

**Rationale**: Enables paired statistical tests. Comparing two systems on the same 47 queries with a paired test is far more powerful than comparing two aggregate numbers.

### No heavy dependencies

**Decision**: Pure Python implementation of all metrics.

**Rationale**: NDCG is 5 lines of code. MRR is 3 lines. The only optional dependency is scipy for a faster Fisher exact test. No numpy, sklearn, or torch required.

### Sync adapter protocol

**Decision**: `RetrievalAdapter.retrieve()` is synchronous.

**Rationale**: Most RAG systems have sync APIs. Async systems wrap with `asyncio.run()` internally. Forcing async on sync users is worse than the reverse.

### Severity as 2D classification

Drift severity combines magnitude and statistical significance to avoid two failure modes:
- Flagging tiny but statistically significant changes (noise)
- Flagging large changes from random variation (insufficient evidence)

See {doc}`user-guide/drift-detection` for the severity matrix.

## Extension Points

### Custom Adapters
Register via entry points in `pyproject.toml`. See {doc}`user-guide/custom-adapters`.

### Custom Reporters
Implement the same function signatures as the built-in reporters (console, markdown, JSON).

### Custom Alert Sinks
Implement the `AlertSink` protocol for Slack, email, PagerDuty integration.

## Plugin System

ir-eval uses Python entry points for adapter discovery:

```toml
[project.entry-points."ir_eval.adapters"]
research-kb = "ir_eval_research_kb:ResearchKBAdapter"
```

The CLI `--adapter` flag looks up adapters from installed entry points, enabling a pluggable ecosystem.

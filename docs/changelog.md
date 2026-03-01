# Changelog

## v0.1.0 (2026-02)

Initial release.

### Features

- **Results-first evaluation** — Evaluate from pre-computed JSON without a live retrieval system
- **Adapter-based evaluation** — Live evaluation via the `RetrievalAdapter` protocol
- **8 retrieval metrics** — MRR, NDCG@k, Hit Rate@k, Precision@k, MAP, Cohen's Kappa, Weighted Kappa
- **Statistical drift detection** — Bootstrap CI, paired permutation, Fisher exact, McNemar's tests
- **2D severity classification** — Magnitude × significance avoids false alarms
- **Baseline management** — Pin and track evaluation baselines over time
- **3 output formats** — Console (Rich), Markdown (GitHub PRs), JSON (CI/CD)
- **CLI with 7 commands** — `evaluate`, `run`, `baseline set/show`, `compare`, `drift`, `validate`, `history`
- **Plugin system** — Entry-point-based adapter discovery
- **Zero heavy deps** — Pure Python metrics; optional scipy for faster Fisher test

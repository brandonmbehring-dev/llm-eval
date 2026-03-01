# Drift Detection

Drift detection identifies regressions by comparing a current evaluation run against a pinned baseline using paired statistical tests.

## Setting a Baseline

Pin an evaluation run as the baseline:

```bash
# Run evaluation and save
ir-eval evaluate results.json --golden golden.json --output run.json

# Set as baseline
ir-eval baseline set run.json --notes "v1.2 release candidate"
```

**Python API:**

```python
from ir_eval.drift import BaselineStore

store = BaselineStore()
store.set_baseline(eval_run, set_by="ci", notes="v1.2 release")
```

Baselines are stored in `.ir-eval/baselines/` as JSON files, keyed by golden set name.

## Detecting Drift

Compare the current system against the baseline:

```bash
# CLI
ir-eval drift golden.json --adapter research-kb --exit-code

# With confidence intervals
ir-eval drift golden.json --adapter research-kb --ci --format markdown
```

**Python API:**

```python
from ir_eval.drift import DriftDetector, BaselineStore

store = BaselineStore()
baseline = store.get_baseline("my-golden-set")

detector = DriftDetector(ci_confidence=0.95, n_bootstrap=10_000)
results = detector.detect(baseline.eval_run, current_run)

for r in results:
    print(f"{r.metric_name}: {r.delta:+.3f} ({r.severity})")
```

## Severity Classification

Drift severity is a 2D classification based on magnitude and statistical significance:

| Magnitude | p < 0.05 | p < 0.10 | p ≥ 0.10 |
|-----------|----------|----------|----------|
| > 10% drop | CRITICAL | CRITICAL | WARNING |
| > 5% drop | WARNING | WARNING | INFO |
| ≤ 5% drop | INFO | INFO | INFO |
| Any increase | INFO | INFO | INFO |

This avoids two failure modes:
- Flagging tiny but statistically significant changes (noise)
- Flagging large changes from random variation (insufficient evidence)

## CI/CD Integration

Use `--exit-code` to fail CI pipelines on CRITICAL drift:

```yaml
# GitHub Actions
- name: Check for retrieval regression
  run: ir-eval drift golden.json --adapter my-system --exit-code
```

The `ExitCodeAlert` exits with code 1 if any metric has CRITICAL severity.

## Statistical Tests Used

| Metric Type | Test | Why |
|------------|------|-----|
| Hit Rate (binary) | Fisher's exact | Exact for small n (< 100 queries) |
| Hit Rate (paired) | McNemar's | Tests paired binary outcomes |
| MRR, NDCG (continuous) | Paired permutation | No distributional assumptions |
| All metrics | Bootstrap CI | Non-parametric confidence intervals |

## Viewing Baselines

```bash
# Show current baseline
ir-eval baseline show my-golden-set

# List all baselines
ir-eval history my-golden-set
```

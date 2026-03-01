# CI/CD Integration

ir-eval is designed for CI/CD pipelines. This guide covers GitHub Actions setup, output formats, and exit code handling.

## GitHub Actions Workflow

```yaml
name: Retrieval Evaluation
on:
  pull_request:
    branches: [main]

jobs:
  eval:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Install
        run: pip install ir-eval

      - name: Run evaluation
        run: |
          ir-eval evaluate results.json \
            --golden golden.json \
            --format markdown \
            --output run.json \
            > eval_report.md

      - name: Check for regression
        run: |
          ir-eval drift golden.json \
            --adapter my-system \
            --exit-code \
            --format markdown \
            > drift_report.md

      - name: Comment on PR
        if: always()
        uses: marocchino/sticky-pull-request-comment@v2
        with:
          path: eval_report.md
```

## Output Formats for CI

### JSON (machine-readable)

```bash
ir-eval evaluate results.json --golden golden.json --format json
```

Parse in downstream steps:

```bash
MRR=$(ir-eval evaluate results.json --golden golden.json --format json | jq '.metrics.mrr')
```

### Markdown (PR comments)

```bash
ir-eval evaluate results.json --golden golden.json --format markdown
```

Produces tables suitable for GitHub PR comments with metric summaries and per-difficulty breakdowns.

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | No CRITICAL drift detected |
| 1 | CRITICAL drift detected (with `--exit-code`) |

## Baseline Management in CI

```bash
# Set baseline on main branch merges
ir-eval baseline set run.json --notes "CI baseline $(date +%Y-%m-%d)"

# Store baselines in repo
git add .ir-eval/baselines/
git commit -m "Update evaluation baseline"
```

:::{tip}
Store baselines in your repository so drift detection works consistently across CI runs. The `.ir-eval/baselines/` directory contains JSON files that are safe to version control.
:::

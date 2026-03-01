# CLI Reference

ir-eval provides a Typer-based CLI with 7 commands for evaluation, baseline management, and drift detection.

## Global Usage

```bash
ir-eval [COMMAND] [OPTIONS]
```

## Commands

### `evaluate` — Evaluate pre-computed results

Primary evaluation path. Takes pre-computed retrieval results and evaluates against a golden set.

```bash
ir-eval evaluate RESULTS_FILE [OPTIONS]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--golden` | PATH | required | Path to golden set JSON |
| `--top-k` | INT | 10 | Cutoff for @k metrics |
| `--output` | PATH | — | Save EvalRun to JSON |
| `--format` | STR | console | Output: `console`, `markdown`, `json` |

**Example:**

```bash
ir-eval evaluate results.json --golden golden.json --format markdown --output run.json
```

### `run` — Live evaluation via adapter

Evaluate a running retrieval system using an adapter registered via entry points.

```bash
ir-eval run GOLDEN_SET [OPTIONS]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--adapter` | STR | required | Entry point name of adapter |
| `--top-k` | INT | 10 | Results to retrieve per query |
| `--output` | PATH | — | Save EvalRun to JSON |
| `--format` | STR | console | Output format |

**Example:**

```bash
ir-eval run golden.json --adapter research-kb --top-k 10
```

### `baseline set` — Pin a baseline

Pin an evaluation run as the baseline for drift detection.

```bash
ir-eval baseline set RUN_FILE [OPTIONS]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--notes` | STR | — | Notes about this baseline |
| `--store-dir` | PATH | `.ir-eval/baselines` | Storage directory |

### `baseline show` — View current baseline

```bash
ir-eval baseline show GOLDEN_SET_NAME [OPTIONS]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--store-dir` | PATH | `.ir-eval/baselines` | Storage directory |

### `compare` — Compare two runs

Side-by-side comparison of two evaluation runs.

```bash
ir-eval compare RUN_A RUN_B [OPTIONS]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--format` | STR | console | Output format |

**Example:**

```bash
ir-eval compare baseline.json current.json --format markdown
```

### `drift` — Detect regression from baseline

Run evaluation and compare against stored baseline with statistical tests.

```bash
ir-eval drift GOLDEN_SET [OPTIONS]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--adapter` | STR | required | Adapter name |
| `--top-k` | INT | 10 | Results per query |
| `--exit-code` | BOOL | false | Exit 1 on CRITICAL drift |
| `--ci` | BOOL | false | Show confidence intervals |
| `--format` | STR | console | Output format |
| `--store-dir` | PATH | `.ir-eval/baselines` | Baseline storage |

### `validate` — Validate golden set

Check golden set structure and show distribution summary.

```bash
ir-eval validate GOLDEN_SET
```

### `history` — Baseline history

Show baseline change history for a golden set.

```bash
ir-eval history GOLDEN_SET_NAME [OPTIONS]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--store-dir` | PATH | `.ir-eval/baselines` | Storage directory |

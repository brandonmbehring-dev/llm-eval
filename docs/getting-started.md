# Getting Started

## Installation

```bash
pip install ir-eval
```

For development:

```bash
git clone https://github.com/brandon-behring/llm-eval.git
cd llm-eval
pip install -e ".[dev]"
```

Optional: Install scipy for faster Fisher exact tests:

```bash
pip install ir-eval[scipy]
```

## Two Evaluation Paths

ir-eval supports two evaluation workflows:

### Path 1: Results-First (Recommended)

Evaluate from pre-computed JSON results. No live retrieval system needed.

```python
from ir_eval import GoldenSet, ResultSet, evaluate_from_results

golden = GoldenSet.from_file("golden.json")
results = ResultSet.from_file("results.json")
run = evaluate_from_results(golden, results, top_k=10)
```

### Path 2: Adapter-Based (Live)

Evaluate against a running retrieval system via the `RetrievalAdapter` protocol.

```python
from ir_eval import GoldenSet, run_evaluation
from my_adapter import MyAdapter

golden = GoldenSet.from_file("golden.json")
adapter = MyAdapter()
run = run_evaluation(golden, adapter, top_k=10)
```

## Golden Set Format

A golden set is a JSON file mapping queries to their relevant document IDs:

```json
{
  "name": "my-golden-set",
  "version": "1.0",
  "queries": [
    {
      "query": "What is causal inference?",
      "relevant_ids": ["doc_001", "doc_042"],
      "difficulty": "easy"
    },
    {
      "query": "Explain the backdoor criterion",
      "relevant_ids": ["doc_015"],
      "difficulty": "hard",
      "relevance_grades": {"doc_015": 3}
    }
  ]
}
```

## CLI Quick Start

```bash
# Evaluate pre-computed results
ir-eval evaluate results.json --golden golden.json

# Set a baseline for drift detection
ir-eval baseline set run.json

# Check for regression
ir-eval drift golden.json --adapter research-kb --exit-code

# Compare two runs
ir-eval compare run_a.json run_b.json --format markdown
```

## Next Steps

- {doc}`user-guide/golden-sets` — Creating and managing golden sets
- {doc}`user-guide/evaluation` — Running evaluations in detail
- {doc}`user-guide/drift-detection` — Setting up regression detection
- {doc}`cli` — Full CLI reference

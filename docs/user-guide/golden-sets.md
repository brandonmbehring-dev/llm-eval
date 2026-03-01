# Golden Sets

Golden sets are the foundation of ir-eval's evaluation approach. They define queries and their known relevant documents, enabling deterministic, reproducible evaluation.

## Why Golden Sets?

Unlike LLM-as-judge approaches (RAGAS, DeepEval), golden sets provide:

- **Deterministic results** — Same input always produces same output
- **Cheap to run** — No API calls during evaluation
- **Separation of concerns** — Evaluator quality ≠ system quality
- **Paired statistical tests** — Per-query metrics enable powerful comparisons

The tradeoff: golden sets require upfront curation effort. This is worth it for production systems where regression detection matters.

## Schema

```json
{
  "name": "string (required)",
  "version": "string (required)",
  "queries": [
    {
      "query": "string (required)",
      "relevant_ids": ["string", "..."],
      "difficulty": "easy | medium | hard (optional)",
      "relevance_grades": {"doc_id": 0-3}
    }
  ]
}
```

### Fields

`name`
: Unique identifier for this golden set. Used as the key in baseline storage.

`version`
: Semantic version. Increment when adding/removing queries.

`queries[].query`
: The search query text.

`queries[].relevant_ids`
: List of document IDs considered relevant for this query.

`queries[].difficulty`
: Optional difficulty tag (`easy`, `medium`, `hard`). Used for stratified analysis.

`queries[].relevance_grades`
: Optional graded relevance (0-3) for NDCG computation. If omitted, binary relevance is used.

## Building a Golden Set

### Step 1: Collect representative queries

Choose queries that reflect real usage patterns. Include a mix of:

- Simple factual queries (easy)
- Multi-concept queries (medium)
- Nuanced or ambiguous queries (hard)

### Step 2: Identify relevant documents

For each query, manually identify which documents in your corpus are relevant. Be exhaustive — missing relevant documents inflate false-negative rates.

### Step 3: Assign difficulty and grades

Optional but recommended. Difficulty tags enable stratified reporting. Graded relevance (0-3) enables NDCG with position-sensitive evaluation.

## Validation

Validate golden set structure with the CLI:

```bash
ir-eval validate golden.json
```

This checks:
- JSON schema conformance
- Unique query texts
- Non-empty relevant_ids
- Valid difficulty values
- Difficulty distribution summary

## Example

The `examples/` directory contains golden sets for the research-kb system:

- `golden_small.json` — 5 queries for quick testing
- `golden_research_kb.json` — 47 queries across causal inference domains
- `golden_research_kb_full.json` — 177 queries (comprehensive)

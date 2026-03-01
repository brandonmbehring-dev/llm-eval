# Glossary

```{glossary}
Adapter
    A class implementing the `RetrievalAdapter` protocol that connects ir-eval to a retrieval system. Provides `name` property and `retrieve(query, top_k)` method.

Baseline
    A pinned `EvalRun` used as the reference point for drift detection. Stored in `.ir-eval/baselines/` as JSON.

Bootstrap CI
    Non-parametric confidence interval computed by resampling data with replacement and taking percentiles of the statistic distribution.

Drift Detection
    Statistical comparison of a current evaluation run against a stored baseline to identify regressions.

EvalRun
    The complete output of an evaluation: per-query metrics, aggregate metrics, configuration, and metadata.

Fisher's Exact Test
    Exact statistical test for 2×2 contingency tables. Preferred over chi-squared for small sample sizes (< 100 queries).

Golden Set
    A curated collection of queries with known relevant document IDs. The foundation of deterministic, reproducible evaluation.

Hit Rate
    Fraction of queries where at least one relevant document appears in the top-k results.

MAP
    Mean Average Precision. Average of per-query AP scores across all queries.

McNemar's Test
    Test for paired binary outcomes. Only considers discordant pairs (one system correct, other wrong).

MRR
    Mean Reciprocal Rank. Average of 1/rank of the first relevant document across queries.

NDCG
    Normalized Discounted Cumulative Gain. Position-weighted metric supporting graded relevance.

Paired Permutation Test
    Non-parametric test that randomly swaps paired observations to build a null distribution. No distributional assumptions.

Precision@k
    Fraction of top-k results that are relevant for a given query.

ResultSet
    Pre-computed retrieval results in JSON format. The input for results-first evaluation.

Severity
    2D classification of drift: CRITICAL (>10% drop + significant), WARNING (>5% drop or borderline significance), INFO (small or insignificant changes).
```

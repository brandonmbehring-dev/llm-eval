# Metric Definitions

## Ranking Metrics

### Reciprocal Rank (RR)

For a single query, the reciprocal of the rank of the first relevant document:

```
RR = 1 / rank(first relevant doc)
```

If no relevant document is found: RR = 0.

### Mean Reciprocal Rank (MRR)

Average RR across all queries:

```
MRR = (1/Q) * sum(RR_i, i=1..Q)
```

- MRR = 1.0: Always rank 1
- MRR = 0.5: Average rank 2
- MRR = 0.33: Average rank 3

### Hit Rate@k

Fraction of queries where at least one relevant document appears in top-k:

```
Hit Rate@k = (# queries with hit in top-k) / (total queries)
```

### Precision@k

Fraction of top-k results that are relevant:

```
Precision@k = (# relevant in top-k) / k
```

### Average Precision (AP)

Area under precision-recall curve for a single query:

```
AP = (1/R) * sum(P(k) * rel(k), k=1..n)
```

where R = total relevant docs, P(k) = precision at rank k, rel(k) = 1 if doc at rank k is relevant.

### Mean Average Precision (MAP)

Average AP across all queries:

```
MAP = (1/Q) * sum(AP_i, i=1..Q)
```

### NDCG@k (Normalized Discounted Cumulative Gain)

Position-weighted metric supporting graded relevance:

```
DCG@k = sum(rel_i / log2(i + 1), i=1..k)
IDCG@k = DCG of ideal ranking
NDCG@k = DCG@k / IDCG@k
```

- Binary relevance: rel = 1 if relevant, 0 otherwise
- Graded relevance: rel = grade (e.g., 0-3)
- NDCG = 1.0: Perfect ranking
- Higher positions contribute more (logarithmic decay)

## Agreement Metrics

### Cohen's Kappa

Chance-corrected agreement between two raters:

```
kappa = (p_o - p_e) / (1 - p_e)
```

where p_o = observed agreement, p_e = expected agreement by chance.

| Range | Interpretation |
|-------|---------------|
| < 0 | Less than chance |
| 0.01-0.20 | Slight |
| 0.21-0.40 | Fair |
| 0.41-0.60 | Moderate |
| 0.61-0.80 | Substantial |
| 0.81-1.00 | Almost perfect |

### Weighted Kappa

For ordinal categories, penalizes distant disagreements more:

```
kappa_w = 1 - (w_o / w_e)
```

Weight schemes:
- **Linear**: w(i,j) = |i - j| / (k - 1)
- **Quadratic**: w(i,j) = (|i - j| / (k - 1))^2

## Statistical Tests

### Bootstrap Confidence Interval

Non-parametric CI using the percentile method:

1. Resample n values with replacement (B times)
2. Compute statistic for each bootstrap sample
3. CI = [percentile(alpha/2), percentile(1 - alpha/2)]

No distributional assumptions. Works for any metric.

### Paired Permutation Test

H0: No difference between systems A and B.

1. For each paired observation (a_i, b_i), randomly swap assignment
2. Compute test statistic (mean(B) - mean(A)) for each permutation
3. p-value = fraction of permutations where |diff| >= |observed diff|

More powerful than unpaired tests because it accounts for query difficulty.

### Fisher's Exact Test

Exact test for 2x2 contingency tables. Used instead of chi-squared when
expected cell counts < 5 (common with golden sets of 20-100 queries).

Uses hypergeometric distribution for exact p-value computation.

### McNemar's Test

Tests whether marginal proportions differ in paired binary outcomes.

Key insight: only "discordant" pairs matter (A correct + B wrong, or vice versa).

Under H0: discordant pairs follow Binomial(n_discordant, 0.5).

Uses exact binomial test (appropriate for small n).

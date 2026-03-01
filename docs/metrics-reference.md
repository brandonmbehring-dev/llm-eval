# Metrics Reference

Complete reference for all metrics implemented in ir-eval, including formulas and interpretation.

## Ranking Metrics

### Reciprocal Rank (RR)

For a single query, the reciprocal of the rank of the first relevant document:

$$RR = \frac{1}{\text{rank}(\text{first relevant doc})}$$

If no relevant document is found: $RR = 0$.

### Mean Reciprocal Rank (MRR)

Average RR across all queries:

$$MRR = \frac{1}{Q} \sum_{i=1}^{Q} RR_i$$

| MRR Value | Interpretation |
|-----------|---------------|
| 1.0 | Relevant doc always at rank 1 |
| 0.5 | Relevant doc on average at rank 2 |
| 0.33 | Relevant doc on average at rank 3 |

### Hit Rate@k

Fraction of queries where at least one relevant document appears in top-k:

$$\text{Hit Rate@k} = \frac{|\{q : \text{hit in top-k}\}|}{Q}$$

### Precision@k

Fraction of top-k results that are relevant:

$$\text{Precision@k} = \frac{|\text{relevant} \cap \text{top-k}|}{k}$$

### Average Precision (AP)

Area under the precision-recall curve for a single query:

$$AP = \frac{1}{R} \sum_{k=1}^{n} P(k) \cdot \text{rel}(k)$$

where $R$ = total relevant docs, $P(k)$ = precision at rank $k$, $\text{rel}(k) = 1$ if doc at rank $k$ is relevant.

### Mean Average Precision (MAP)

Average AP across all queries:

$$MAP = \frac{1}{Q} \sum_{i=1}^{Q} AP_i$$

### NDCG@k (Normalized Discounted Cumulative Gain)

Position-weighted metric supporting graded relevance:

$$DCG@k = \sum_{i=1}^{k} \frac{\text{rel}_i}{\log_2(i + 1)}$$

$$NDCG@k = \frac{DCG@k}{IDCG@k}$$

where $IDCG@k$ is the DCG of the ideal (perfect) ranking.

- **Binary relevance**: $\text{rel} = 1$ if relevant, $0$ otherwise
- **Graded relevance**: $\text{rel} = \text{grade}$ (e.g., 0–3)
- NDCG = 1.0 means perfect ranking; higher positions contribute more via logarithmic decay.

## Agreement Metrics

### Cohen's Kappa

Chance-corrected agreement between two raters:

$$\kappa = \frac{p_o - p_e}{1 - p_e}$$

where $p_o$ = observed agreement, $p_e$ = expected agreement by chance.

| Range | Interpretation |
|-------|---------------|
| < 0 | Less than chance |
| 0.01–0.20 | Slight |
| 0.21–0.40 | Fair |
| 0.41–0.60 | Moderate |
| 0.61–0.80 | Substantial |
| 0.81–1.00 | Almost perfect |

### Weighted Kappa

For ordinal categories, penalizes distant disagreements more:

$$\kappa_w = 1 - \frac{\sum w_{ij} \cdot o_{ij}}{\sum w_{ij} \cdot e_{ij}}$$

Weight schemes:
- **Linear**: $w(i,j) = \frac{|i - j|}{k - 1}$
- **Quadratic**: $w(i,j) = \left(\frac{|i - j|}{k - 1}\right)^2$

## Statistical Tests

### Bootstrap Confidence Interval

Non-parametric CI using the percentile method:

1. Resample $n$ values with replacement ($B$ times, default 10,000)
2. Compute statistic for each bootstrap sample
3. $CI = [\text{percentile}(\alpha/2), \text{percentile}(1 - \alpha/2)]$

No distributional assumptions. Works for any bounded or skewed metric.

### Paired Permutation Test

$H_0$: No difference between systems A and B.

1. For each paired observation $(a_i, b_i)$, randomly swap assignment
2. Compute test statistic $\bar{B} - \bar{A}$ for each permutation
3. $p = \frac{|\{\text{permutations where } |\text{diff}| \geq |\text{observed}|\}|}{B}$

More powerful than unpaired tests because it accounts for query difficulty.

### Fisher's Exact Test

Exact test for 2×2 contingency tables. Used instead of chi-squared when expected cell counts < 5 (common with golden sets of 20–100 queries).

Uses the hypergeometric distribution for exact $p$-value computation.

### McNemar's Test

Tests whether marginal proportions differ in paired binary outcomes.

Key insight: only discordant pairs matter (A correct + B wrong, or vice versa).

Under $H_0$: discordant pairs follow $\text{Binomial}(n_{\text{discordant}}, 0.5)$.

Uses exact binomial test, appropriate for small $n$.

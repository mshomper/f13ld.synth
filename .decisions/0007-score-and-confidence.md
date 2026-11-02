# 0007 Match is a Gaussian of z-scores against the trainer's residual σ; confidence is tree spread versus the seeds

- Status: found
- Date: 2026-11-02
- Lenses: solver
- Deliberate: yes
- Holding up: fine
- In the repo's log: README.md Architecture notes: Confidence from tree spread; docs/SESSION_RECAP_2026-10-09.md v0.4.0: Stricter penalty on low-confidence designs

## Context

12-search-core.js scores each targeted metric as z = (pred − target)/σ with σ from the bundle's test residuals, combines them as score = exp(−zRms²/6), and discounts by tree spread relative to the median spread on training seeds (no cut up to 1.25×, falling off after, ×0.7 past 2×). Labels high/medium/low come from the same ratio.

## Options

### A: z-score match plus spread penalty (current) (recommended)

Error-aware scoring with hand-set penalty breakpoints.
- Pro: Uses each metric's real model error, so weak metrics are scored loosely
- Pro: Spread penalty stopped wide searches drifting into low-confidence designs (v0.4 recap)
- Con: Breakpoints 1.25 / 2 / 0.7 are tuned by eye
- Con: Global σ per metric, not per design
- Effort: In place

### B: Per-design intervals (quantile forests or conformal)

Predict an interval per candidate and score the probability of hitting the target band.
- Pro: Honest per-design uncertainty; replaces two mechanisms with one
- Con: Trainer and bundle change; more compute per candidate
- Effort: Medium to large

### C: Nearest-neighbor distance to seeds for out-of-distribution

README v0.2 plans mention a k-NN distance check.
- Pro: Simple, model-independent
- Con: Distance in a padded feature space is a weak proxy
- Effort: Small

## Recommendation

A (no confidence given). 12-search-core.js scoreCandidate / spreadFactor / confidenceLabel; README "Confidence from tree spread"; v0.4 recap "Stricter confidence penalty" approved by Matt.

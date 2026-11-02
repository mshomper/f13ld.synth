# 0001 Random forests trained offline in Python, run in the browser

- Status: found
- Date: 2026-11-02
- Lenses: solver, architecture
- Deliberate: yes
- Holding up: fine
- In the repo's log: README.md Architecture notes: Random Forest, not MLP

## Context

train_synth.py fits scikit-learn RandomForestClassifier (validity) and one RandomForestRegressor per output metric (150 trees, max_depth 15, min_samples_leaf 2), serializes every tree to JSON, and 11-forest.js walks them in the browser. Tree spread doubles as the confidence signal (12-search-core.js scoreCandidate).

## Options

### A: Random forests (current) (recommended)

Per-metric RF regressors plus an RF validity classifier, exported as raw trees.
- Pro: Beat an MLP by 0.5+ mean R² at 257 samples (README)
- Pro: Tree disagreement gives a free confidence signal
- Pro: Exact parity between Python and JS (tests/forest-parity.js)
- Con: Big bundle: 1,650 trees of depth up to 15
- Con: Cannot extrapolate past the training range
- Con: Piecewise-constant predictions give the search flat regions
- Effort: In place

### B: Gradient-boosted trees

LightGBM or XGBoost, still exported as trees and walked by the same packed-node code.
- Pro: Usually better R² per byte; far fewer, shallower trees
- Pro: Same inference code shape
- Con: Loses tree spread as a confidence signal; would need quantile heads or an ensemble
- Con: New training dependency
- Effort: Medium: trainer change, confidence rework

### C: Small neural net via ONNX or hand-written inference

An MLP once Vault has enough rows.
- Pro: Smooth predictions, small bundle
- Pro: Gradients available for search
- Con: Lost to RF at small data sizes; still only ~2.5k rows
- Con: Confidence needs ensembles or dropout
- Effort: Large

## Recommendation

A (no confidence given). README "Random Forest, not MLP" records the empirical comparison; train_synth.py and 11-forest.js are built around it, and the confidence signal (12-search-core.js) depends on per-tree spread.

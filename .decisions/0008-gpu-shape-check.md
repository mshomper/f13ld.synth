# 0008 Measured solid fraction from a GPU field sample overrides the model's volume fraction

- Status: found
- Date: 2026-11-02
- Lenses: solver, architecture
- Deliberate: yes
- Holding up: strained. In the Mesh check every PI-TPMS design built 2 to 4 points thinner than predicted; whether the model or the 48³ sampling is wrong is an open hand check in docs/NEXT-UP.md.
- Open item in the repo's own log
- In the repo's log: docs/SESSION_RECAP_2026-10-09.md v0.4.1: Shape check, approved by Matt; docs/NEXT-UP.md Check by hand: PI-TPMS volume fraction

## Context

After the search, 20-predictor.js shapeCheck() measures the best 16 designs' solid fraction on a 48³ grid over one cell using the preview's field (25-raymarch.js), replaces predicted volume fraction, and cuts confidence when the model was off by more than 2σ (low past 3σ). Recipes record it under synth.measured.

## Options

### A: Fixed 48³ GPU sample of the top 16 (current) (recommended)

Same field as the preview and F13LD.mesh; one grid size for all modes.
- Pro: Volume-fraction targets scored on the real shape
- Pro: Catches model errors far from data (4.5% vs 20.6% case)
- Con: Thin PI-TPMS pipes may be under-sampled
- Con: Adds seconds on software renderers
- Effort: In place

### B: Adaptive or finer grid for PI-TPMS

Sample finer where walls or pipes are thin.
- Pro: Fixes the suspected PI miss
- Con: More GPU time; needs a thickness heuristic
- Effort: Small to medium

### C: Treat the check as advisory only

Show the measured value but keep the model's for scoring.
- Pro: Cheaper, simpler
- Con: Ranks designs on a number known to be wrong
- Effort: Small

## Recommendation

A (no confidence given). Approved by Matt in v0.4.1 (recap "Shape check"); README "Shape check"; 20-predictor.js shapeCheck and 12-search-core.js shapeCheck.

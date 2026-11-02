# 0006 Inverse search mutates real training designs in timed rounds and scores the exact recipe it sends

- Status: found
- Date: 2026-11-02
- Lenses: solver, architecture
- Deliberate: yes
- Holding up: fine
- In the repo's log: README.md Architecture notes: Search over real designs; Rounds, by time; docs/SESSION_RECAP_2026-10-09.md v0.3.0: Synth scores what it sends

## Context

12-search-core.js decodes training seeds into recipes, varies them the way F13LD.sweep does (nudges within training ranges, sin/cos swaps, frequency changes), encodes and scores each one. Rounds grow from the current best and from the best in each cell of a 24×24 grid, with a shrinking step, under a time budget (00-config.js SEARCH_DEPTHS) on a worker pool (20-predictor.js).

## Options

### A: Seeded mutation over real designs (current) (recommended)

Evolutionary search anchored to training seeds, scoring the exact design.
- Pro: Scored design and sent design are identical (v0.2 differed by a median 22 points)
- Pro: Stays near the data the model knows
- Pro: Grid keeps alternatives spread over the map
- Con: Cannot find designs far from any seed
- Con: Hand-tuned round schedule (ROUND constants)
- Effort: In place

### B: Optimize a continuous feature vector, then snap to a recipe

The v0.2 approach.
- Pro: Simple optimizer
- Con: Ranks a design it never sends; replaced in v0.3.0 for that reason
- Effort: n/a, already abandoned

### C: CMA-ES or Bayesian optimization over recipe parameters

A formal optimizer on the decoded recipe space.
- Pro: Better sample efficiency on hard targets
- Con: Mixed discrete and continuous structure (terms, trig choice) fits poorly
- Con: Loses the map-filling behavior
- Effort: Medium

## Recommendation

A (no confidence given). README "Search over real designs" and "Rounds, by time" describe it and the v0.3.0 recap records why the snap approach was dropped; implemented in 12-search-core.js and worker/search-worker.js.

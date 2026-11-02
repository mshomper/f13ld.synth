# 0004 The whole model ships as one JSON file of explicit trees

- Status: found
- Date: 2026-11-02
- Lenses: data-model, architecture
- Deliberate: likely
- Holding up: strained. The bundle grew from 15 MB to 36.6 MB and no longer uploads through GitHub's web UI; NEXT-UP lists serving it gzipped or split as an idea.
- In the repo's log: docs/NEXT-UP.md Ideas: Serve the model gzipped or split it

## Context

train_synth.py writes weights/tpms.json: every tree node as JSON arrays rounded to 4 decimals, plus encoding, seeds and eval. 20-predictor.js fetches it, 11-forest.js repacks it into 12-byte binary node records, then drops the parsed JSON. The file is now 38 MB (36.6 MB per docs/HOW-TO-retrain-synth.md); README still says 15 MB.

## Options

### A: One JSON bundle (current) (recommended)

Human-readable, single fetch, gzip on the wire from GitHub Pages.
- Pro: Simple to inspect and diff the meta block
- Pro: No extra tooling
- Con: Over GitHub's 25 MB browser upload, so deploys go through GitHub Desktop or a zip handed to Claude
- Con: Parse cost and memory spike before packing
- Con: Each retrain adds ~38 MB to git history
- Effort: In place

### B: JSON meta plus a binary tree file

Trainer emits the same 12-byte packed records Synth already builds, as a .bin beside a small JSON.
- Pro: Roughly a third of the size, no parse step
- Pro: Browser can use the buffer directly
- Con: Two files to version together
- Con: Trainer must replicate the packing (already specified in 11-forest.js)
- Effort: Small to medium

### C: Shrink the JSON

Fewer trees, 3 decimals, or split per metric.
- Pro: No format change
- Con: Costs R² or only delays the problem as Vault grows
- Effort: Small

## Recommendation

A (no confidence given). train_synth.py _serialize_tree, 20-predictor.js loadFamily and 11-forest.js pack show the JSON-then-pack path; README "Versioning" ties the bundle's meta block to cache busting.

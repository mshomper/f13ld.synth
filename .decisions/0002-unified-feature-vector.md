# 0002 One model per family on a zero-padded unified feature vector, encoded twice (Python and JS)

- Status: found
- Date: 2026-11-02
- Lenses: data-model, solver
- Deliberate: yes
- Holding up: strained. Two known encoder gaps (Sweep's per-axis cell_scale_x/y/z and field-pair PI-TPMS surface_b) are waiting on a coordinated Python + JS + retrain change (docs/NEXT-UP.md).
- In the repo's log: README.md Architecture notes: Unified feature vector; docs/NEXT-UP.md Encoder gaps

## Context

encode_design_unified() in train_synth.py and SynthEncoding.encode() in 10-encoding.js build the same fixed-length vector: mode one-hot, then term and factor slots padded to max_terms / max_factors. The layout is read from the bundle's encoding block, but the encoding logic itself exists in both languages and must change together (CLAUDE.md).

## Options

### A: Unified vector, mirrored by hand (current) (recommended)

One family model; Python and JS encoders kept slot-for-slot identical, checked by tests/roundtrip.js.
- Pro: Turned per-config R² 0.25 into 0.71 unified (README)
- Pro: Bundle-driven layout tolerates a larger term cap without page edits
- Con: Two implementations to keep in lockstep
- Con: New modes or fields (per-axis cell scale, surface_b) need both sides and a retrain
- Effort: In place

### B: Encoding spec as data, both sides generated

Describe slots declaratively in the bundle (or a shared JSON) and have thin interpreters in Python and JS.
- Pro: Adding a field becomes one edit
- Pro: Drift becomes impossible rather than tested-for
- Con: Upfront rework of both encoders
- Con: Interpreter must cover term structure, which is not flat
- Effort: Medium

### C: Per-mode or per-preset models

Separate models per mode or per preset family.
- Pro: Each model simpler
- Con: Exactly what failed before (R² 0.25)
- Con: Many more bundles
- Effort: Medium

## Recommendation

A (no confidence given). README "Unified feature vector" states the choice and its trade-off; CLAUDE.md requires the two encoders to change together; tests/roundtrip.js checks 600/600 seeds re-encode exactly.

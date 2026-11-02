# 0009 Intent is set on three paired pads chosen from what varies independently in the data

- Status: found
- Date: 2026-11-02
- Lenses: ui, research
- Deliberate: yes
- Holding up: fine
- In the repo's log: docs/SESSION_RECAP_2026-10-09.md v0.5.0: Pads re-paired around what varies on its own; README.md The three pads

## Context

01-defs.js and 31-pads.js define three X/Y pads (stiffness × pore size, main axis × off-axis share, porosity × pore spread), each off / prefer / require, instead of a slider per metric. Ranges are fixed from the 2026-10-09 data; derived metrics (stiff_main, stiff_ratio, porosity) are computed in 12-search-core.js addDerived.

## Options

### A: Three pads on independent axes (current) (recommended)

Pairs follow the correlation structure: density, direction, pore uniformity.
- Pro: Users cannot ask for impossible combinations as easily
- Pro: Fewer controls than nine sliders
- Con: Pad ranges and pairings depend on today's data and need re-checking after the Vault reseed
- Effort: In place

### B: One slider per metric

The v0.1 framing.
- Pro: Direct
- Con: Most combinations unreachable (old Volume × Thermal pad was two-thirds unreachable)
- Effort: Small

### C: Pads derived from the data at load

Compute principal directions or correlations from the bundle's seeds and lay out pads automatically.
- Pro: Survives retrains without hand edits
- Con: Axes stop being nameable physical quantities
- Effort: Large

## Recommendation

A (no confidence given). Matt's v0.5.0 decisions in the recap; README "The three pads" gives the correlation evidence; implemented in 01-defs.js, 31-pads.js and 12-search-core.js.

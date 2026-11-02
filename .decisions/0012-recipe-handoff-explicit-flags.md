# 0012 Results hand off as F13LD.tpms recipes in the URL, with normalization flags always written

- Status: found
- Date: 2026-11-02
- Lenses: data-model, architecture
- Deliberate: yes
- Holding up: fine
- In the repo's log: README.md Architecture notes: Recipes say how they are normalized

## Context

43-handoff.js opens F13LD.mesh with ?r=<recipe JSON> and F13LD.lab with #r=, or copies the JSON. Recipes follow F13LD.tpms v1.1.0 and always write shell_normalize / pi_normalize, because Mesh and Sweep default a missing flag to on and Lab to off.

## Options

### A: Recipe JSON in the URL, flags explicit (current) (recommended)

No server; the recipe is the whole contract.
- Pro: Works across static sites
- Pro: Explicit flags fixed PI-TPMS building at 3 to 4× predicted density in Lab
- Con: Long URLs; tools disagree on defaults, so every writer must remember the flags
- Effort: In place

### B: Recipe schema with a version and required flags

Make the flags mandatory in the recipe spec and have readers reject or warn on missing ones.
- Pro: Fixes the default mismatch at the source
- Con: Needs changes in Mesh, Lab and Sweep
- Effort: Medium

## Recommendation

A (no confidence given). README "Recipes say how they are normalized" and the v0.3.0 recap; implemented in 10-encoding.js (recipe writer) and 43-handoff.js.

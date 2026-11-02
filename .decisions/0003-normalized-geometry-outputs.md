# 0003 The model predicts material-free normalized outputs; units appear only at display

- Status: found
- Date: 2026-11-02
- Lenses: data-model, solver
- Deliberate: yes
- Holding up: fine
- In the repo's log: README.md Material card

## Context

train_synth.py extract_outputs() divides stiffness by E_solid, pore size by cell size and conductivity by k_solid per sweep. 30-material.js multiplies back using the material card shared with Vault through localStorage (f13ld.vault.globalInputs.v1).

## Options

### A: Normalized targets, scale at display (current) (recommended)

Geometry-only model; physical GPa, µm, W/mK computed from the material card.
- Pro: Changing material never invalidates the model
- Pro: Pools sweeps run on different materials
- Con: Assumes linear scaling with solid properties
- Con: Rows missing a reference value lose that metric (e.g. 414 rows with no k_solid)
- Effort: In place

### B: Material as an input feature

Feed E_solid, k_solid etc. into the model and predict physical values.
- Pro: Could capture non-linear material effects
- Con: Needs far more data per material
- Con: Model changes with every new material
- Effort: Large

## Recommendation

A (no confidence given). README "Material card" says the trainer always works in normalized space and resolution happens at display only; train_synth.py extract_outputs() and 30-material.js implement it.

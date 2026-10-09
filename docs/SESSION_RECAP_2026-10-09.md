# Session recap — 2026-10-09 · v0.4.0 (preview, UI rebuild, wider search)

## Decisions from Matt
- Result map layout (B): map larger, thumbnails smaller
- Push buttons in both the header and the inspector; candidate sort options
- UI rebuild and raymarcher in one pass
- Wider search with a loading mechanic; **Wide** is the default depth
- Depth is a small word toggle (Quick | Wide | Deep) next to Synthesize
- Stricter penalty on low-confidence designs
- Pads show the target only (the map shows the candidates)

## What changed
- **3-D preview.** WebGL2 raymarcher (`25-raymarch.js`) with F13LD.tpms field math plus Mesh's per-term phase and shell normal weights; F13LD-SHADE look and F13LD-VIEW menu shared with Sweep. Orbit, zoom, 1 or 2×2×2 tiles, X/Y/Z section plane. Every card gets a thumbnail from one hidden canvas.
- **Preview matches Mesh.** `tests/preview-parity.py` builds 12 seeds (shell, solid, PI) with the preview shader and with Mesh's own field code: 0.0 points apart in solid fraction.
- **Search in rounds.** Each round mutates half from the current best and half from cells of a 24×24 map grid, with a mutation size that shrinks each round. Quick ≈1 s, Wide ≈5 s, Deep ≈15 s; stops early once the top results stop improving, or on Stop.
- **The map is the loader.** Each round's designs fade in on the result map; numbered rings glide to the current top eight; dashed ellipse is the target ±1σ. Tabs switch the map between the active pads.
- **Stricter confidence penalty.** Score × 1 up to 1.25× normal spread, falling off past that, and × 0.7 more for low. In tests the wider search no longer drifts into low-confidence designs (hard target 64% mean with no lows; easy targets ≈90%).
- **UI rebuild** (mockup v0.5): intent column, result map, candidate strip with sort (best match / most confident / lightest), inspector with preview, fit bars and hand-off buttons; Configure drawer with Lab's 45 AM materials, model fit and search notes; status chip; phone layout in one column with no sideways scroll.
- Drawn icons throughout (`03-icons.js`); Lab materials synced by `tools/sync-materials.js`.

## Checks run
roundtrip · forest parity · load order (22 scripts) · search smoke · Mesh parity · preview parity (0.0 points) · headless page check (Wide search, preset filter, Stop, sort, recipe re-encode, phone overflow) · single-file preview.

## Open finding
Mesh parity has one mutated solid design that builds at 4.5% solid where the model predicts 20.6%. The recipe is right (the preview agrees with Mesh); the model is wrong far from its training data. Proposal: measure the solid fraction from geometry for the final eight and lower confidence when it disagrees. Not built yet.

---

# Session recap — 2026-10-09 · v0.3.0 (phase 1 of the modernization)

## Plan (approved 2026-10-09)
1. **Phase 1 — module split + correctness** (this session)
2. Phase 2 — raymarcher preview (port F13LD.tpms's field math; WebGL2 so it works in the Claude mobile app; one live viewer plus per-card thumbnails from a shared canvas; 1-cell / 2×2×2 tiling)
3. Phase 3 — UI modernization, **mockups first**: shared header with Queue / Mesh / Lab, drawn icons, Lab's AM material library, viewer as the hero, no horizontal scroll
4. Retraining — a **stopgap retrain now** on today's Vault; the real retrain after Vault is wiped and reseeded from the new Sweep

## Decisions from Matt
- Stopgap retrain until the reseeded Vault exists
- UI mockups before phase 3
- The preset menu stays and must work
- Synth may learn shear moduli once the data has them

## What changed
- **Modules.** Single file → numbered classic scripts + `worker/search-worker.js`, same layout as Mesh, Lab and Sweep. UI look unchanged apart from the card tags and buttons below.
- **Synth scores what it sends.** Candidates are real designs (decoded seeds, varied the way Sweep varies a recipe), encoded and scored exactly. The v0.2 build scored a blurred vector and snapped it afterwards; the scored and sent designs differed by a median of 22 score points (`tests/search-smoke.js`).
- **Recipes keep everything.** Per-term phases, normal weights and constant terms are no longer dropped. Recipes follow F13LD.tpms v1.1.0 and write `shell_normalize` / `pi_normalize` explicitly (Lab reads a missing flag as off, Mesh and Sweep as on; with the flags off, PI-TPMS recipes build at 3–4× the predicted density).
- **Search.** 4,000 explore + refine of the top 16, across a worker pool (main-thread fallback). At most two results per seed, topped up when a preset filter leaves few seeds.
- **Confidence.** Tree spread relative to the training seeds → high / medium / low on each card; mildly lowers the rank of far-off designs. Replaces the validity-driven "extrapolation" tag.
- **Preset filter works.** Grows candidates only from seeds of the chosen preset. The current bundle has no per-seed presets, so seeds are grouped by term skeleton (F-RD / lidinoid and Fischer-Koch S / split-P share one; two groups match no preset). Retrained bundles carry real presets.
- **Open in Lab** and **Copy recipe** on every card. Glyph icons on the cards and pads replaced with drawn ones.
- **Packed forest** (12-byte node records): ~1.8× faster scoring.
- **Fresh-browser material card** now fills the default material's values.
- **Trainer v0.3.0** (built from Matt's newer local copy; the repo copy was behind): encodes only switched-on terms; term cap 6 → 10 (lidinoid and F-RD were being cut short); terms with no per-term phase no longer drop the row; optional shear targets; stratified seeds with presets; `--max-norm`, `--solver-version`, `--max-terms`, `--n-seeds`. Bundle schema 0.2.0. Old bundles still load.
- `docs/HOW-TO-retrain-synth.md` updated (the 0.775 baseline it quoted was not the deployed bundle — production is 0.637).

## Checks run
roundtrip (200/200 seeds exact) · forest parity (≤1e-14) · load order · search smoke (8/8 recipes re-encode to the scored features) · Mesh parity (volume fraction within 5 points) · headless page check (search, preset filter, recipe) · trainer against a fake Vault · page against a new-format bundle · single-file preview from a file URL.

## Next
- Matt: click-test the branch (preview file or a local server), then merge; copy the new `train_synth.py` to the training folder and run the stopgap retrain.
- Phase 2: raymarcher.
- When the reseeded Vault exists: retrain with `--solver-version`, consider a tighter `--max-norm`; Sweep's per-axis `cell_scale_x/y/z` and field-pair PI-TPMS (`surface_b`) are not in the encoder yet — adding them changes the bundle layout (Synth reads it from the bundle, so no lockstep page edit).
- Save to Vault is still a placeholder.

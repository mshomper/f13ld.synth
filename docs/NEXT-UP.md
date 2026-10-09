# F13LD.synth — next up

Updated 2026-10-09 (v0.5.0). Newest first within each group.

## Check by hand (Matt)
- **Click-test v0.5.0** on the live site: the three new pads, map brightness, the out-of-reach note, the shape-check tag.
- **PI-TPMS volume fraction.** In the Mesh check every pipe design built 2–4 points thinner than predicted (e.g. 0.9% vs 4.9%). Build one PI-TPMS result in F13LD.lab and compare: if Lab agrees with Mesh, the model leans high on pipes; if Lab agrees with Synth, the 36³/48³ sampling misses thin pipes and the shape check should sample finer for PI.
- Time the shape check on a real GPU (console line `Shape check: 16 designs measured in …`). The VM's software renderer took about 3 s.

## When the reseeded Vault exists (from the new Sweep)
- Retrain on Matt's machine (`docs/HOW-TO-retrain-synth.md`) with `--solver-version` set and consider a tighter `--max-norm` (e.g. 1.15) for the GPU solver.
- Shear moduli train automatically once 200+ rows carry them; then a shear target could join pad 2.
- Re-run the reach check (pad ranges are set from the 2026-10-09 data: pad 1 stiffness to 0.3 Es, porosity 50–99%, CV 0.5–1.4) and the correlation check behind the pad pairing.
- Encoder gaps: Sweep's per-axis `cell_scale_x/y/z` and field-pair PI-TPMS (`surface_b`) are not encoded yet. Adding them changes the bundle layout (Synth reads it from the bundle; the trainer and `10-encoding.js` change together).

## Ideas, not started
- Orient results so the main axis is Z (the build direction): an exact axis permutation of the design, offered as a button in the inspector.
- Shape check sampling finer for PI-TPMS, or an adaptive grid.
- Show cell scale (repeats per cell) on the cards, since the thumbnails show one repeat.
- Serve the model gzipped or split it: it is 36.6 MB and over GitHub's 25 MB browser upload.
- Save to Vault is still a placeholder.
- The shared F13LD-VIEW block uses a "◐ view" glyph suite-wide; Synth swaps in a drawn icon after init. Fix it in the shared block for every tool.

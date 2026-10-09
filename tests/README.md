# F13LD.synth dev checks

None of these ship with the tool; they run from a clone.

| Check | What it proves |
|---|---|
| `node tests/roundtrip.js [bundle]` | Every training seed decodes to a design and re-encodes bit for bit, so a candidate is exactly what the model scores. Also lists the preset skeletons found. |
| `node tests/forest-parity.js [bundle]` | The packed forest predicts exactly what the v0.2 per-tree walker did. |
| `node tests/search-smoke.js [bundle]` | One explore + refine search on the main-thread path; every final recipe re-encodes to the scored features; reports how far v0.2's scored vectors were from the recipes it sent. |
| `node tests/loadorder.js` | No top-level name is declared twice across the numbered scripts, and each parses. |
| `node tests/mesh-parity.js ../f13ld.mesh` | Synth recipes built with F13LD.mesh's own TPMS field code give the volume fraction Synth predicted. |
| `python3 tests/page-check.py [port] [png]` | Loads the page in headless Chromium, runs a search and a preset-filtered search, saves a screenshot. Vault is blocked in the check. |
| `python3 tests/trainer-smoke.py SEEDS.json OUT.json` | Runs `train_synth.py`'s Vault path against a fake Vault that includes switched-off terms, phase-less PI-TPMS terms, a 9-term lidinoid, shear columns and corrupted rows. |

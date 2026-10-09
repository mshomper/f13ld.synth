# F13LD.synth dev checks

None of these ship with the tool; they run from a clone.

| Check | What it proves |
|---|---|
| `node tests/roundtrip.js [bundle]` | Every training seed decodes to a design and re-encodes bit for bit, so a candidate is exactly what the model scores. Also lists the preset skeletons found. |
| `node tests/forest-parity.js [bundle]` | The packed forest predicts exactly what the v0.2 per-tree walker did. |
| `node tests/search-smoke.js [bundle]` | One round-based search on the main-thread path; every final recipe re-encodes to the scored features; reports how far v0.2's scored vectors were from the recipes it sent. |
| `node tests/loadorder.js` | No top-level name is declared twice across the numbered scripts, and each parses. |
| `node tests/mesh-parity.js ../f13ld.mesh` | Synth recipes built with F13LD.mesh's own TPMS field code give the volume fraction Synth predicted; also shows each design's confidence before and after the shape check. |
| `python3 tests/page-check.py [port] [dir]` | Loads the page in headless Chromium with software WebGL: a search, a preset-filtered search, a stopped Deep search, the preview and thumbnails, sorting, the recipe matching the scored design, and no sideways overflow on a phone. Saves screenshots. Vault is blocked. |
| `python3 tests/preview-parity.py ../f13ld.mesh` | Evaluates the 3-D preview's field shader on the GPU over a 36³ grid and compares the solid fraction with F13LD.mesh's field code, for seeds in all three modes. |
| `python3 tests/trainer-smoke.py SEEDS.json OUT.json` | Runs `train_synth.py`'s Vault path against a fake Vault that includes switched-off terms, phase-less PI-TPMS terms, a 9-term lidinoid, shear columns, rows with no solid conductivity and corrupted rows. |

# F13LD.synth — instructions for Claude

## Commits and pull requests
- Do not add session links (e.g. `Claude-Session: https://claude.ai/...`) to commit messages, pull request descriptions or any file in this repo.
- `Co-Authored-By: Claude …` trailers are fine.
- Ask Matt before pushing to `main`.

## Code layout
- Numbered classic scripts share one global scope and load in numeric order (see README → Repo structure). Top-level code that runs at load time may only use things defined in the same or a lower-numbered file. Check with `node tests/loadorder.js`.
- `worker/search-worker.js` loads `10-encoding.js`, `11-forest.js` and `12-search-core.js` with `importScripts`; those three must not touch the DOM.
- `10-encoding.js` mirrors `encode_design_unified()` in `train_synth.py` slot for slot. Change both together.
- Bump `F13LD_SYNTH_VERSION` in `00-config.js` and the header label in `index.html` (`.fh-version`) on every release.
- Serve over http(s) to test; `file://` does not work for the multi-file build (workers). `node tools/build-single.js` makes a one-file preview that does.

## Testing
- `node tests/roundtrip.js` · `node tests/forest-parity.js` · `node tests/search-smoke.js` · `node tests/loadorder.js`
- `node tests/mesh-parity.js ../f13ld.mesh` builds Synth results with Mesh's own field code.
- `python3 tests/page-check.py` loads the page in headless Chromium (Playwright) and runs searches.
- `python3 tests/trainer-smoke.py SEED_RECIPES.json OUT.json` runs the trainer against a fake Vault.
- Retraining runs on Matt's machine (`docs/HOW-TO-retrain-synth.md`), not in the session VM.

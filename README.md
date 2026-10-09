# F13LD.synth

**Inverse-design synthesizer for the F13LD suite.** Given a design intent — *stiff but permeable, isotropic, lightweight* — F13LD.synth generates novel TPMS scaffold parameter recipes that match the intent, using a Random Forest predictor trained on the F13LD.vault community database.

[Live tool →](https://mshomper.github.io/f13ld.synth) | [F13LD suite →](https://f13ld.app) | [F13LD.vault →](https://mshomper.github.io/f13ld.vault) | [F13LD.mesh →](https://mshomper.github.io/f13ld.mesh)

---

## What it does

F13LD.synth answers a question the rest of the F13LD suite can't: *"What design produces these properties?"*

Most CAD pipelines run forward — pick parameters, simulate, see what you got. Synth runs the inverse direction. You set design intent on three X/Y pads, the predictor samples 2,000 candidate parameter sets within the trained design space, scores each against your intent, and returns the eight best with full recipes ready to open in F13LD.mesh.

The predictor is a per-family Random Forest model trained offline from F13LD.vault. As the community ingests more sweep data into Vault, the model gets retrained against a richer dataset. The model chip in the header shows this lineage — what the model was trained on, its fit, and how many designs Vault has gained since.

## The three pads

Synth deliberately doesn't expose nine independent metric sliders. Instead, three X/Y pads encode the design tradeoffs that actually matter:

**Mechanics × Pore** — the fundamental scaffold tradeoff. X-axis is overall stiffness, Y-axis is mean pore size. More material → stiffer but tighter pores. Less material → permeable but compliant. Bone scaffold designers spend their careers in this pad.

**Anisotropy × Pore Distribution** — the "what kind of material is it" pad. X-axis is anisotropy ratio (1 = isotropic, 3 = highly directional). Y-axis is pore size CV (0.4 = uniform, 0.9 = varied). Each corner is a different design philosophy: foam, directional truss, bone-like, exotic biomimicry.

**Mass × Thermal** — the optional second-tier tradeoff. X-axis is volume fraction, Y-axis is thermal conductivity. Off by default; toggle on when heat handling matters (thermal exchangers, bone cement curing, etc.).

Each pad has an `off / prefer / require` toggle. Off contributes nothing to the search. Prefer is a soft target. Require carries 2× the weight. Click any pad title to flip into precision mode. The **Preset** menu grows candidates only from training designs of one preset (listed from the loaded model with seed counts) — the SVG swaps for two number inputs with proper units, and any pinned values feed back into the search at high weight.

The 9th metric (`surface_complexity`) is dropped from controls — it's still predicted and shown on result cards, but it's a model-quality signal more than a design intent and isn't worth a knob.

## The data lineage loop

```
Sweeps → F13LD.vault → train_synth.py → weights/tpms.json → F13LD.synth
   ↑                                                               ↓
   └──────────────── Save synth candidates back to Vault ←─────────┘
```

The model is a snapshot of Vault at training time. The header status bar shows three numbers that capture this:

- **vault N** — total usable designs in Vault for this family right now
- **trained N valid** — how many designs the current model was trained on
- **+N since training** — designs added since the model was last refreshed

When the third number passes ~200, the indicator turns amber and reads "retrain warranted." A retrain pulls fresh data from Vault, trains a new model, commits the new `weights/<family>.json` file, and the tool picks it up automatically.

Synth candidates can be saved back to Vault tagged as model-generated — they're not FEA-validated until someone runs them through a sweep, but they're useful as starting points for future sweeps. (As of v0.1 the Save-to-Vault button is mocked pending a schema decision on candidate tagging.)

---

## Repo structure

```
f13ld.synth/
├── index.html              ← page shell; loads the numbered scripts in order
├── synth.css
├── 00-config.js            ← version, endpoints, storage keys, search depths
├── 01-defs.js              ← metric definitions, map ranges, the three pads
├── 02-materials.js         ← F13LD.lab's AM material library (generated: tools/sync-materials.js)
├── 03-icons.js             ← drawn icons
├── 10-encoding.js          ← design ⇄ feature vector ⇄ F13LD recipe (mirrors the trainer)
├── 11-forest.js            ← packed random-forest inference + tree spread
├── 12-search-core.js       ← mutation, scoring, rounds coordinator (worker-safe)
├── 20-predictor.js         ← bundle loading, worker pool
├── 21-vault.js             ← Vault counts for the model chip
├── 24-f13-shade.js         ← shared F13LD viewer shading + view menu (byte-identical across tools)
├── 25-raymarch.js          ← 3-D preview + thumbnails
├── 30-material.js · 31-pads.js · 35-score-viz.js · 36-state.js
├── 40-map.js · 41-strip.js · 42-inspector.js · 43-handoff.js · 50-header.js · 51-dock.js · 99-init.js
├── worker/search-worker.js
├── train_synth.py          ← offline trainer (run locally — see docs/HOW-TO-retrain-synth.md)
├── vault_client.py · vault_stats.py
├── tools/                  ← build-single.js (one-file preview), sync-materials.js, vault_diag.html
├── tests/                  ← dev checks (see tests/README.md)
└── weights/tpms.json       ← trained model bundle
```

No build step. The numbered scripts are plain classic scripts sharing one global scope, the same layout as F13LD.mesh, F13LD.lab and F13LD.sweep. Serve over http(s); for a single file you can open directly, run `node tools/build-single.js`.

---

## Deploying

GitHub Pages handles everything. From a fresh clone:

```bash
git clone https://github.com/mshomper/f13ld.synth.git
cd f13ld.synth
git push  # if you've made changes
```

Settings → Pages → main branch / root directory. Tool goes live at `https://mshomper.github.io/f13ld.synth/`.

GitHub Pages serves `weights/tpms.json` with `Content-Encoding: gzip` automatically when the file is fetched, so the 15 MB JSON arrives as ~4 MB on the wire. No manual gzipping required.

---

## Retraining

Today the pipeline is **Pattern A** — manual local script invocation. The plan is to migrate to Pattern B (GitHub Actions workflow with a button to retrain) once the trainer logic is stable.

### File-mode (works today)

When you have a directory of `sweep_results_*.json` files locally:

```bash
pip install numpy scikit-learn tensorflow
python train_synth.py tpms --from-files ~/sweeps --out weights/tpms.json
git add weights/tpms.json
git commit -m "retrain tpms · N=<sample count> · R²=<mean>"
git push
```

Synth picks up the new weights automatically next time anyone loads it (browser cache invalidates on file hash change).

### Vault-mode (stubbed; activates once vault_client.py lands)

The canonical retrain workflow once wired:

```bash
python train_synth.py tpms --from-vault --out weights/tpms.json
```

This pulls every usable design from F13LD.vault for the named family, encodes via the unified feature vector, trains validity classifier + 9-output metrics regressor, and exports. No local files to manage, no folder to keep in sync. Vault is the source of truth.

### Trainer flags

```
positional:
  family                {tpms,noise,grain}    Which family to train

source (one required):
  --from-files DIR      Read sweep_results_*.json files from this directory
  --from-vault          Pull from F13LD.vault (requires vault_client.py)

options:
  --out PATH            Output path for the trained model JSON
  --n-estimators N      Trees per forest (default 150)
  --decimals N          Float precision (default 4 decimals)
```

Reduce `--n-estimators` for smaller model files at modest R² cost. Reduce `--decimals` for further size reduction (3 decimals is roughly free; 2 is risky).

### What the trainer does

1. **Loads designs** from the chosen source, filtering to the requested family (e.g., gyroid + schwarzD + schwarzP + lidinoid + frd + iwp + Fischer-Koch S all roll up to family `tpms`)
2. **Encodes** each design via the unified feature vector (384 slots with the default 10-term cap) — works across all modes, term counts, and factor frequency configurations within a family
3. **Trains a validity classifier** on every loaded design (predicts whether a parameter set produces solver-valid geometry)
4. **Trains a metrics regressor** with 9 output heads on valid designs only (predicts each normalized output metric independently)
5. **Picks 200 seed samples** from the training set — the browser uses these as Gaussian-perturbation seeds during inverse search to keep candidates near the training distribution
6. **Evaluates** on a 20% held-out split, compares to KNN-5 baseline, reports per-metric R²
7. **Exports** a single JSON bundle containing the trees, normalization constants, encoding metadata, seed samples, and eval block

Output looks like:

```
=== F13LD.Synth trainer · family=tpms ===
Hyperparameters: n_estimators=150, threshold_precision=4 decimals
Loading 8 sweep file(s) from ~/sweeps
Designs loaded: 1405 (933 valid)
Feature dim: 233

--- Training validity classifier ---
Validity test accuracy: 0.819

--- Training metrics regressor ---
Metrics training samples: 746, test: 187
  volume_fraction        R² = +0.925   (KNN baseline +0.828) ✓
  ex_norm                R² = +0.864   (KNN baseline +0.810) ✓
  ey_norm                R² = +0.872   (KNN baseline +0.811) ✓
  ez_norm                R² = +0.864   (KNN baseline +0.803) ✓
  anisotropy             R² = +0.552   (KNN baseline +0.381) ✓
  pore_size_norm         R² = +0.638   (KNN baseline +0.649) ·
  pore_size_cv           R² = +0.316   (KNN baseline +0.140) ✓
  keff_avg_norm          R² = +0.591   (KNN baseline +0.539) ✓
  surface_complexity     R² = +0.757   (KNN baseline +0.641) ✓

Mean R²: model=0.709, KNN baseline=0.623
Exported: weights/tpms.json  (15.23 MB)
```

A `✓` means the model beat the KNN baseline on that metric — the additional complexity is earning its keep. A `·` means the model and KNN are within rounding of each other.

---

## Reading R² annotations

Every metric on every result card carries a small `R² 0.XX` badge showing how well the predictor handles that metric. Three trust bands:

- **Green (R² ≥ 0.7)** — trustworthy. Predictions land within ~10% of true FEA values for designs in the training distribution.
- **Cyan (R² 0.4–0.7)** — usable as guidance. Treat predictions as direction, not exact value.
- **Amber (R² < 0.4)** — rough. The metric varies in ways the model can't fully capture from geometry. Don't make commitments based on these numbers.

Stiffness, volume fraction, and surface complexity train cleanly. Pore distribution metrics (especially `pore_size_cv`) are the weakest because they're spatially non-smooth — small parameter changes can re-arrange pore geometry without smooth interpolation between samples. More training data helps but probably can't fully fix this.

The headline `match` percentage on each card is computed across all *targeted* metrics (the ones a pad is currently driving), so it's only as reliable as the metrics it was scored against. If a pad is targeting a low-R² metric, the headline match is correspondingly fuzzy.

The `validity` percentage is the validity classifier's confidence that the parameter set produces solver-valid geometry. Below 85% the number turns amber. Below 70% the result card gets an "extrapolation" tag — the candidate is far enough outside the training distribution that the predictor's confidence drops materially.

---

## Material card

Material context is shared across the F13LD suite via `localStorage[f13ld.vault.globalInputs.v1]`. Fill it in here, it appears in F13LD.vault automatically. Fill it in Vault, it shows up here.

The 9 metrics divide into two groups:

- **Geometry-only** (`volume_fraction`, `anisotropy`, `pore_size_cv`, `surface_complexity`): displayed as ratios/dimensionless. Material card has no effect on display.
- **Material-scaled** (`ex/ey/ez_norm`, `pore_size_norm`, `keff_avg_norm`): displayed as physical units (GPa, µm, W/mK) when the material card is filled, normalized ratios when empty. Empty fields show a small `norm` badge.

The trainer always works in normalized space — the model has no notion of any specific material. Resolution to physical units happens at display time only, which means changing the material card never invalidates the model.

Material library presets (Ti-6Al-4V, 316L SS, 17-4PH SS, H13, Al 6061, AlSi10Mg, Inconel 718, PEEK, PEKK) match Vault's. Custom values are supported — typing into any field flips the material dropdown to "Custom…".

---

## Architecture notes

**Search over real designs.** Every candidate is a real design: a training seed decoded into its recipe and then varied the way F13LD.sweep varies a recipe (continuous nudges inside the range the training data covers, occasional sin/cos swaps and frequency changes). That exact design is encoded and scored, and the same design is what the preview shows and what Open in Mesh and Open in Lab send. (Before v0.3.0 Synth scored a blurred feature vector and snapped it to a recipe afterwards; the score of what it ranked and of what it sent differed by a median of 22 points.)

**Rounds, by time (v0.4.0).** A search runs in rounds across a worker pool (all threads but one). Each round grows new designs from the best so far and from the best design in every filled cell of a 24 × 24 grid over the first active pad's axes, so it closes in on the target while keeping alternatives spread across the map. The mutation step starts wide and narrows each round. Depth sets the time limit — Quick (about 1 s), Wide (about 5 s, the default) and Deep (about 15 s) — and the search stops earlier once the best eight stop improving, or when you press Stop. At most two of the eight results come from the same seed.

**The map is the loader.** Every design that passes the filters lands on the result map as its round comes back; the eight best are numbered rings that glide to where the latest round put them; the dashed ring is 1σ of the model's own error around your target.

**Confidence from tree spread.** Each forest's trees vote separately; how much they disagree on a candidate, relative to how much they disagree on the training seeds, is shown as high, medium or low confidence. Wider searches drift toward designs where the trees disagree (likely model error), so the rank falls off steeply past the training spread, with an extra cut for low confidence.

**The preview is the build.** The 3-D preview raymarches the candidate's own field in F13LD.tpms's shader form, plus F13LD.mesh's per-term phases and anisotropic shell weights, with the suite's shared F13LD-SHADE lighting. `tests/preview-parity.py` checks its solid fraction against F13LD.mesh's field code.

**Recipes say how they are normalized.** Synth writes `shell_normalize` / `pi_normalize` explicitly. F13LD.mesh and F13LD.sweep read a missing flag as on, F13LD.lab as off; the training data was characterized normalized, and without the flag Lab would build PI-TPMS pipes at several times the predicted density.

**Random Forest, not MLP.** Original v0.1 scoping called for a small MLP. Empirical diagnostic at 257 samples showed RF outperforming MLP by 0.5+ in mean R² — small-data regime favors trees. The unified-feature trainer reaches mean R² 0.71 on 933 samples, with stiffness/volume_fraction predictions at R² > 0.85.

**Unified feature vector.** One model per family, not per (mode, term-count) configuration. The vector encodes mode as one-hot, term/factor slots as zero-padded fixed-length arrays. This was the key insight that turned the predictor from non-functional (R² 0.25 per-config) into useful (R² 0.71 unified). Trade-off: the model can't extrapolate to entirely new modes — every mode the user might query has to appear in the training data. Adding a new mode means retraining.

**Packed forests.** Each forest is one buffer of 12-byte node records (split feature, threshold or leaf value, right child; the left child is always the next node). Scoring one candidate walks 1,650 trees; on a single core that is roughly 0.3–0.6 ms, so a full search takes well under a second on a multi-core desktop.

**Validity early-rejection.** During inverse search, candidates are first scored by the validity classifier. Anything below 30% confidence is rejected before the more expensive 9-forest metric prediction runs. Saves about 30% of total compute on a typical search.

---

## Browser support

Tested on recent Chrome, Firefox, and Safari (desktop and mobile). The pad UI uses Pointer Events for unified mouse/touch handling. The model JSON is fetched once and cached aggressively (`cache: 'force-cache'`) so subsequent searches don't re-download.

The predictor consumes ~50 MB of browser memory once instantiated (15 MB JSON parsed into JS objects, ~3.5× expansion). No problem on desktop. Probably fine on tablets. Phones with <2 GB RAM may struggle.

---

## Versioning

Model bundles carry a version string in their meta block. Synth's UI shows the loaded version. When you retrain and bump the version, the URL hash on the model file changes and browsers fetch fresh.

The tool itself versions independently — `index.html`'s header shows the tool version, the model status bar shows the model version. They're decoupled by design, so a model retrain doesn't require a UI redeploy and vice versa.

---

## v0.2 plans

- **GitHub Actions retraining workflow.** Pattern B — click a button in the GitHub UI, the trainer runs on Actions, opens a PR with the updated weights file. Removes the manual local-script step.
- **Vault candidate ingest.** Wire the Save to Vault button on synth result cards to actually POST candidate-tagged designs back to Vault.
- **Multi-family support.** Train and ship `noise.json` and `grain.json` once those families have ~500 valid designs in Vault.
- **OOD detection refinement.** Currently using validity < 0.7 as a proxy for the extrapolation flag. A true k-nearest-neighbor distance check against the seed pool would be more honest.
- **Model size optimizations.** Already at 15 MB / 4 MB gzipped. Reducing to ~8 MB raw is doable with tighter quantization at minimal R² cost.

---

## Credits

Designed and built as part of the F13LD suite by [Not a Robot Engineering LLC](https://notarobot-eng.com).

The unified-feature encoding and per-family training architecture were derived empirically through diagnostic experiments — the [scoping conversation](https://github.com/mshomper/f13ld.synth/blob/main/docs/scoping.md) documents the reasoning if you want to follow how the design choices landed.

Built on the shoulders of: scikit-learn (training), numpy (math), Supabase (Vault hosting), and the F13LD suite's shared brand system.

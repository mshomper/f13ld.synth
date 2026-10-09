# How to Retrain F13LD.Synth

## When to retrain

Retrain when one or more of the following is true:

- Vault has gained meaningful new data (a few hundred+ new TPMS rows since the last bundle — Synth's status bar turns amber at +200)
- You've changed `train_synth.py` (encoder, output metrics, filters, etc.)
- You've cleaned up bad data in the Vault, or wiped and reseeded it
- Bundle's `trained_at` is more than a month or so old and you want a refresh

Retraining is cheap — a few minutes wall-clock — so when in doubt, retrain.

---

## Local environment

- Folder: `C:\Users\mshom\Desktop\f13ld-synth-train\`
- Contains: `train_synth.py`, `vault_client.py`, `vault_stats.py`, and the current `tpms.json` from the previous run
- Python 3.13 with numpy, scikit-learn and requests installed
- Run from PowerShell

**After Synth v0.3.0: copy the new `train_synth.py` from the repo into this folder.** The repo copy is now the source of truth (the old repo copy was behind your local one). `vault_client.py` is unchanged.

---

## Run the retrain

Open PowerShell. Navigate to the training folder:

```
cd C:\Users\mshom\Desktop\f13ld-synth-train
```

Run the trainer:

```
py train_synth.py tpms --from-vault --out tpms.json
```

That pulls every TPMS row from Vault (valid, partial and invalid — the invalid ones teach the validity classifier), drops rows with normalized stiffness or thermal above 1.3, trains a Random Forest per metric, and exports the bundle to `tpms.json` in the current folder.

Optional flags:
- `--limit 1000` — cap at first 1000 rows (quick iteration testing)
- `--since 2026-04-01` — only train on rows added on or after a date
- `--max-norm 1.3` — the corrupted-row cutoff. Sized for the old axial-only solver (which overestimates by up to ~10%). Data from Sweep's GPU solver may warrant something tighter, e.g. `1.15`
- `--solver-version 0.24` — keep only rows whose `solver_version` starts with this, for a Vault holding results from more than one solver. Rows with no recorded version are dropped when this is set
- `--max-terms 10` — term slots in the feature vector. 10 holds every preset Sweep expands (lidinoid is 9 terms)
- `--n-seeds 600` — training designs exported as search seeds, spread evenly across presets

---

## What changed in trainer v0.3.0 (read this before comparing numbers)

The numbers from a v0.3.0 retrain are **not directly comparable** to the previous bundle, because the encoder now sees the designs correctly:

- **Switched-off terms are no longer counted as on.** Sweep's term mask turns ~15% of terms off; the old encoder fed them to the model anyway.
- **Term cap raised from 6 to 10.** F-RD (6 terms + a constant) and lidinoid (9 terms) were being silently cut short. The trainer now warns if any row still overflows.
- **Terms without a per-term phase no longer drop the row.** Current Sweep writes no per-term phase for PI-TPMS terms; the old encoder threw on those and silently skipped the whole row. Today's Vault rows may all carry the field, but a reseed from the new Sweep would have lost every PI-TPMS row.
- **Shear moduli are trained when present.** `gxy_norm`, `gxz_norm`, `gyz_norm` are added automatically once 200+ rows carry them (Sweep v0.24+ data). Today's Vault data likely has none, and the trainer says so.
- **Seeds carry their preset**, so Synth's preset filter uses the real preset of each sweep rather than guessing from the term pattern.

---

## Verify the output before deploying

The trainer prints diagnostics during the run. Things to check:

**1. Row counts look sane:**
```
Designs loaded: 2109 (1852 usable for metric regression)
  solver_validity breakdown: valid=..., partial=..., invalid=...
  Dropped N rows for normalized stiffness/thermal > 1.3
  usable rows by preset: frd=..., gyroid=..., schwarzD=..., ...
```
- Total should match approximately what Vault has
- Outlier drop count should be small (<15% of total). If it's much larger, something upstream broke in Sweep
- There should be **no** `WARNING: N rows have more terms/factors than the encoder holds` line. If there is, add `--max-terms 12`
- `Skipped N rows whose recipe could not be encoded` should be 0 or very small

**2. Mean R² against the bundle in production:**
```
Mean R²: model=0.XXX, KNN baseline=0.XXX
```
- The bundle in production (trained 2026-05-19) is **mean R² 0.637**, KNN baseline 0.564, validity accuracy 0.879. (The 0.775 this guide used to quote was never the deployed bundle.)
- Per-metric, production is: volume_fraction 0.91 · ex 0.76 · ey 0.67 · ez 0.60 · anisotropy −0.03 · median pore 0.50 · pore CV 0.72 · thermal 0.88 · surface complexity 0.86 · directionality 0.49
- A drop of more than 0.05 = investigate before deploying. A *small* shift either way is expected from the encoder fixes above
- KNN baseline should be lower than model — if not, the model isn't learning anything useful
- If shear targets were trained, they are included in the mean

**3. Per-metric R²:**
- `volume_fraction`, `keff_avg_norm`, `surface_complexity` should be 0.85+
- `ex_norm`, `ey_norm`, `ez_norm` 0.6+ (they were 0.60–0.76 in production)
- `anisotropy` near zero is structural (its label variance is tiny after the trim), not a model problem

**4. Bundle file:**
The trainer ends with `Seeds: 600 across N presets` and `Exported: tpms.json (NN MB)`. The production bundle is about 29 MB; a v0.3.0 bundle will be a little larger (more feature slots and seeds). Open it in a text editor and check the first line for:
- `"version": "0.2.0"` and `"trainer_version": "0.3.0"`
- `"data_source": "vault"`
- `"n_valid"` matching the count printed earlier
- `"feature_dim": 384` with the default 10-term cap (it was 236 with the old 6-term cap)

`feature_dim` is no longer locked to the page: Synth v0.3.0 reads the layout from the bundle's `encoding` block and checks it on load, so a new term cap needs no page change.

---

## Deploy to GitHub

1. Open `github.com/mshomper/f13ld.synth` in browser
2. Click into the `weights/` folder
3. Click **Add file → Upload files**
4. Drag the new `tpms.json` from your desktop training folder into the upload area
5. GitHub recognizes the matching filename — it will overwrite, no warning needed
6. Commit message format: `retrain · mean R² 0.XXX · NNNN rows`
7. Click **Commit changes**

GitHub Pages auto-deploys within ~60 seconds.

---

## Verify the deployment

1. Open `mshomper.github.io/f13ld.synth/weights/tpms.json` in a new tab
2. Check the first line — `feature_dim`, `n_valid`, `data_source` should match what the trainer printed
3. Open `mshomper.github.io/f13ld.synth/` with hard refresh (**Ctrl+Shift+R**)
4. Open DevTools (F12), check the console for:
   ```
   [F13LD.synth] Predictor loaded in 0.Xs · tpms v0.2.0 · NNN valid samples · 600 seeds
   [F13LD.synth] Search pool: N workers
   ```
   That `NNN` should match the bundle's `n_valid`. If it's a different number, the browser cached the old bundle — hard refresh again.
5. The **Preset** menu should list real preset names with seed counts (gyroid, F-RD, lidinoid, …).
6. Run a test search. Click **Open in Mesh** on a PI-TPMS result. Confirm it renders as a pipe network, not a sheet or a cube. Then click **Open in Lab** on the same card and check Lab builds the same shape.

---

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| Trainer crashes with NaN error | New data row has unexpected null/NaN in a metric. Per-metric NaN masking already covers anisotropy, directionality and shear; check which metric the error names. |
| `feature_dim ... encoding block implies ...` error at Synth load | The bundle's header and its trees disagree — the file was hand-edited or truncated. Retrain. |
| Mean R² drops more than 0.05 | New bad-data batch in Vault, or a code change broke training. Use `tools/vault_diag.html` to spot-check recent rows and `vault_stats.py` for distributions. |
| Synth still loads old data after deploy | Browser cache. Hard refresh (Ctrl+Shift+R). If still wrong, check the deployed URL directly to confirm GitHub has the new file. |
| Trainer fetches fewer rows than Vault has | A `--limit` or `--since` flag is set, or `--solver-version` is filtering rows. |
| `WARNING: N rows have more terms/factors than the encoder holds` | A preset or sweep produced more terms than `--max-terms`. Raise it and retrain. |

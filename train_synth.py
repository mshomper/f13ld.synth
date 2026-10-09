"""
F13LD.Synth — Pattern A trainer (CLI, family-level)

Usage
-----
File-mode (reads sweep JSONs from a directory):
    python train_synth.py tpms --from-files ./sweeps --out ./weights/tpms.json

Vault-mode (pulls directly from F13LD.vault — requires vault_client.py
beside this file; the canonical Vault URL and key are built into it):
    python train_synth.py tpms --from-vault --out ./weights/tpms.json

Optional vault-mode filters:
    --since YYYY-MM-DD   only fetch designs created on/after this date
    --limit N            cap rows fetched (useful for quick test runs)

What it does
------------
1. Loads all sweep designs for the requested family (tpms / noise / grain / …)
2. Encodes each design via the UNIFIED FEATURE VECTOR — works across all configs
   within the family (any mode, any term count, any factor frequency)
3. Computes geometry-only normalized outputs from raw FEA values + each sweep's
   reference parameters (E_solid, sigma_ref, k_solid, cell_mm)
4. Trains:
     - Validity classifier (Random Forest, all designs) — predicts P(valid)
     - Metrics regressor   (Random Forest, valid only) — predicts 10 normalized
       outputs, plus shear moduli when enough rows carry them
5. Evaluates on a 20% held-out split, reports per-metric R² and residual σ
6. Exports model + metadata to a single JSON the browser can load

The exported bundle includes per-metric residual σ on the test split, which the
F13LD.synth predictor reads to drive its bell-curve sparkline scoring without
any browser-side approximation.

Prereqs
-------
    pip install numpy scikit-learn requests

When the trained-model JSON is committed to f13ld.synth/weights/, F13LD.Synth's
synthesized engine activates automatically for that family.
"""

import argparse
import glob
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

SEED = 42
np.random.seed(SEED)

# ============================================================
# CONSTANTS — must match F13LD.Synth UI's METRICS/STUB_MODEL_R2 shape
# ============================================================

OUTPUT_METRICS = [
    "volume_fraction",      # ratio, geometry-only
    "ex_norm", "ey_norm", "ez_norm",  # ×modulus_gpa → physical GPa at inference
    "anisotropy",           # ratio — trained directly now, with p1/p99 trim (see TRAIN_LIMITS)
    "pore_size_p50_norm",   # ×cell_µm → physical µm at inference (median pore — replaces pore_size_norm which was the mean, too heavy-tailed to train cleanly)
    "pore_size_cv",         # ratio
    "keff_avg_norm",        # ×thermal_k_wmk → physical W/mK at inference
    "surface_complexity",   # ratio (Vault percentile-ranks at display)
    "directionality",       # ratio — connectivity-percolated axis count / 3 (0, 1/3, 2/3, 1)
]

# Metrics computed at synth runtime from already-predicted values rather than
# trained directly. The op is applied to predictions of `inputs`. Empty by
# default; populate to move a metric out of direct training (e.g. when a
# derivation is cleaner than a regressor on a heavy-tailed label).
DERIVED_METRICS = {}

# Per-metric percentile-based training trims. Rows whose label falls outside
# [lower_pct, upper_pct] get masked out of THAT metric's training only. Use
# for heavy-tailed metrics where extreme outliers dominate SS_tot and prevent
# the regressor from learning the bulk distribution. Other metrics are
# unaffected. Percentiles are computed once from the training-set ground
# truth (after NaN/inf filtering) — they reflect the data, not an assumption.
TRAIN_LIMITS = {
    "anisotropy": (1, 99),  # full range hits 75+; trim restores bulk distribution
}

# Optional targets: trained only when enough rows carry them (F13LD.sweep
# v0.24+ reports shear moduli; older Vault rows don't). Each is normalized by
# the solid modulus like ex/ey/ez, so it resolves to GPa in Synth.
OPTIONAL_METRICS = ["gxy_norm", "gxz_norm", "gyz_norm"]
MIN_OPTIONAL_LABELS = 200

# Unified-feature encoding constants. Synth reads these from the bundle's
# `encoding` block, so changing them needs a retrain but no Synth edit.
MODES = ["pi-tpms", "shell", "solid", "level"]   # extend as new modes appear
TRIG_AXES = ["x", "y", "z"]
MAX_TERMS = 10     # Sweep expands lidinoid to 9 terms, F-RD to 6 + a constant
MAX_FACTORS = 4    # any preset term has at most 3 factors
TRAINER_VERSION = "0.3.1"


# ============================================================
# DATA SOURCES — file-mode and vault-mode both feed the same encoder
# ============================================================

def load_from_files(directory, family_filter):
    """Read every sweep_results_*.json in `directory`, yield designs whose
    family matches `family_filter`. Yields (design_dict, reference_params)."""
    files = sorted(glob.glob(str(Path(directory) / "sweep_results_*.json")))
    if not files:
        sys.exit(f"No sweep_results_*.json files found in {directory}")
    print(f"Loading {len(files)} sweep file(s) from {directory}")
    for fp in files:
        with open(fp) as fh:
            data = json.load(fh)
        # Family inference: most sweeps tag preset, infer family from preset family map
        preset = data.get("meta", {}).get("preset", "")
        family = _infer_family_from_preset(preset)
        if family != family_filter:
            continue
        e_solid = data["base"]["E_solid_GPa"]
        k_solid = (data["context"].get("material") or {}).get("k_W_mK")
        sigma_ref = data["context"]["sigma_ref_GPa"]
        cell_mm = data["context"].get("cellSize_mm", 2)
        if k_solid is None or k_solid == 0:
            print(f"  skipping {Path(fp).name} — missing k_solid")
            continue
        for d in data["designs"]:
            yield d, {
                "E_solid": e_solid, "k_solid": k_solid,
                "sigma_ref": sigma_ref, "cell_mm": cell_mm,
                "source_file": Path(fp).name, "preset": preset,
            }


def load_from_vault(family_filter, since=None, limit=None):
    """Yield (design_dict, reference_params) for designs in F13LD.vault matching
    the requested family.

    Vault stores each design as a flat row with columns + a `recipe` JSON blob.
    The trainer expects the older sweep_results shape: a dict with
    `design.geometry`, `design.surface`, and `browser`. This adapter rebuilds
    that shape from the recipe (geometry, surface, homogenization) + a few
    top-level columns (e_solid_gpa, sigma_ref_gpa, cell_size_mm, material).
    """
    try:
        from vault_client import VaultClient
    except ImportError:
        sys.exit(
            "vault_client.py not found. Place it next to train_synth.py, "
            "and set VAULT_SUPABASE_URL + VAULT_SUPABASE_ANON_KEY env vars."
        )
    try:
        vault = VaultClient()
    except ValueError as e:
        sys.exit(f"Vault setup error: {e}")

    print(f"Querying F13LD.vault for family={family_filter} ...")
    # include_invalid: pull solver_validity=invalid rows too. They're useless
    # for metric regression (no usable solver output) but become negative
    # examples for the validity classifier. Without them, the classifier
    # has no failure cases and gets skipped (which is what was happening).
    rows = vault.fetch_designs(
        family=family_filter,
        valid_only=False,
        exclude_degenerate=True,
        since=since,
        limit=limit,
        verbose=True,
    )
    print(f"Loaded {len(rows)} {family_filter} rows from Vault (valid+partial+invalid)")

    skipped_recipe = 0
    no_k_solid = 0
    skipped_other = 0
    yielded = 0

    for row in rows:
        recipe = _parse_recipe(row.get("recipe"))
        if recipe is None:
            skipped_recipe += 1
            continue

        # Trainer's expected per-design shape: design.{geometry,surface} + browser.
        # Vault flattens these into the recipe blob; rebuild the older shape so
        # the encoder/extractor doesn't need to know data sourced from Vault.
        homog = recipe.get("homogenization") or {}
        homog = dict(homog)  # local copy so we can patch without mutating recipe
        # Inject solver_validity from the top-level Vault column. Invalid rows
        # may not have it in their homogenization block (solver aborted), but
        # the column is authoritative.
        row_sv = row.get("solver_validity")
        if row_sv is not None:
            homog["solver_validity"] = row_sv
        # Mirror top-level Vault columns that extract_outputs needs but that
        # aren't stored in recipe.homogenization. The percentile pore metrics
        # in particular (pore_size_p10_norm, p50_norm, p90_norm) are top-level
        # columns added in a newer sweep schema. We pull them into the homog
        # dict so extract_outputs can read them via b[...] like everything else.
        for col in ("pore_size_p10_norm", "pore_size_p50_norm", "pore_size_p90_norm",
                    "directionality", "Gxy_GPa", "Gxz_GPa", "Gyz_GPa",
                    "Gxy_norm", "Gxz_norm", "Gyz_norm", "solver_version"):
            if homog.get(col) is None and row.get(col) is not None:
                homog[col] = row[col]
        design_dict = {
            "design": {
                "geometry": recipe.get("geometry") or {},
                "surface": recipe.get("surface") or {},
            },
            "browser": homog,
        }

        # Reference params: prefer top-level columns (authoritative), fall back
        # to fields inside the recipe's homogenization block.
        e_solid = row.get("e_solid_gpa")
        if e_solid is None:
            e_solid = homog.get("E_solid_GPa")
        sigma_ref = row.get("sigma_ref_gpa")
        if sigma_ref is None:
            sigma_ref = homog.get("sigma_ref_GPa")
        cell_mm = row.get("cell_size_mm") or 2  # match file-mode default
        k_solid = _extract_k_solid(row.get("material"))
        preset = ((recipe.get("meta") or {}).get("preset")
                  or (recipe.get("surface") or {}).get("preset")
                  or row.get("preset", ""))

        if k_solid is None or k_solid == 0:
            # No solid thermal conductivity: the row still trains every other
            # metric and the validity classifier; only keff_avg_norm (which
            # needs k_solid to normalize) is masked out for it.
            k_solid = None
            if row_sv != "invalid":
                no_k_solid += 1
        if e_solid is None or sigma_ref is None:
            if row_sv == "invalid":
                e_solid = e_solid or 0.0
                sigma_ref = sigma_ref or 0.0
            else:
                skipped_other += 1
                continue

        yielded += 1
        yield design_dict, {
            "E_solid": float(e_solid),
            "k_solid": float(k_solid) if k_solid is not None else None,
            "sigma_ref": float(sigma_ref),
            "cell_mm": float(cell_mm),
            "source_file": "vault",
            "preset": preset,
        }

    if skipped_recipe + skipped_other > 0:
        print(
            f"  Skipped during adapt: "
            f"{skipped_recipe} unparseable recipe · "
            f"{skipped_other} missing E_solid/sigma_ref"
        )
    if no_k_solid:
        print(f"  {no_k_solid} usable rows have no k_solid — kept; thermal (keff_avg_norm) is masked for them")
    if no_k_solid > 0 and no_k_solid == len(rows):
        print(
            "  ALL rows missing k_solid — `material` column may be a name string "
            "rather than a JSON dict. Add a materials lookup or update ingest."
        )
    print(f"  Yielded {yielded} usable designs to trainer")


def _parse_recipe(recipe_field):
    """Recipe column may be parsed jsonb (dict) or stringified JSON."""
    if isinstance(recipe_field, dict):
        return recipe_field
    if isinstance(recipe_field, str):
        try:
            parsed = json.loads(recipe_field)
            return parsed if isinstance(parsed, dict) else None
        except (json.JSONDecodeError, ValueError):
            return None
    return None


def _extract_k_solid(material_field):
    """Pull k_W_mK out of the `material` column. Tolerates dict or stringified
    JSON. Returns None if material is missing, unparseable, or doesn't carry
    a thermal conductivity field — the trainer will then skip the row."""
    if material_field is None:
        return None
    if isinstance(material_field, dict):
        return material_field.get("k_W_mK")
    if isinstance(material_field, str):
        try:
            parsed = json.loads(material_field)
            if isinstance(parsed, dict):
                return parsed.get("k_W_mK")
        except (json.JSONDecodeError, ValueError):
            pass
    return None


_PRESET_ALIASES = {
    "gyroid": "gyroid", "schwarzp": "schwarzP", "schwarz p": "schwarzP", "primitive": "schwarzP",
    "schwarzd": "schwarzD", "schwarz d": "schwarzD", "diamond": "schwarzD", "neovius": "neovius",
    "iwp": "iwp", "i-wp": "iwp", "fks": "fks", "fischer-koch s": "fks", "fischerkochs": "fks",
    "splitp": "splitP", "split-p": "splitP", "frd": "frd", "f-rd": "frd",
    "gyroidharmonic": "gyroidHarmonic", "gyroid-harmonic": "gyroidHarmonic",
    "primitivec": "primitiveC", "primitive-c (g6)": "primitiveC", "octo": "octo", "octo (g8)": "octo",
    "pharmonic": "pHarmonic", "p-harmonic": "pHarmonic", "lidinoid": "lidinoid",
}


def _canonical_preset(name):
    """Preset names as Vault rows have written them → the key F13LD.tpms uses
    (must match _PRESET_ALIASES in Synth's 10-encoding.js). Unknown → 'custom'."""
    if not name:
        return "custom"
    return _PRESET_ALIASES.get(str(name).strip().lower(), "custom")


def _infer_family_from_preset(preset):
    """Map a preset name to its family. Extend as new presets are added."""
    tpms_presets = {"gyroid", "schwarzP", "schwarzD", "lidinoid", "frd", "iwp",
                    "Fischer-Koch S", "fischerKochS", "neovius", "splitP"}
    if preset in tpms_presets:
        return "tpms"
    if preset.startswith("noise") or preset in {"perlin", "worley", "simplex"}:
        return "noise"
    if preset.startswith("grain") or preset in {"spinodoid"}:
        return "grain"
    return "unknown"


# ============================================================
# UNIFIED FEATURE ENCODER — same shape across all configs in a family
# ============================================================

def encode_design_unified(d):
    """Encode a single design into the unified feature vector. Works for any
    mode, term count, factor count within MAX_TERMS / MAX_FACTORS."""
    g = d["design"]["geometry"]
    feats = []

    # mode one-hot
    mode = g.get("mode")
    feats.extend([1.0 if mode == m else 0.0 for m in MODES])

    # geometry scalars (None → 0)
    feats.append(float(g.get("cell_scale") or 0))
    feats.append(float(g.get("wall_thickness") or 0))
    feats.append(float(g.get("pipe_radius") or 0))
    feats.append(float(g.get("offset") or 0))

    # geometry-level phase_shift {x,y,z} — used by mode='pi-tpms' to construct
    # phi_B = phi(r + Delta) from phi_A. Encoded as continuous floats; sweep
    # grid is eighths (0, 1/8, ... 7/8) but RF doesn't need that constraint.
    # For non-PI-TPMS rows the recipe holds phase_shift=null → default (0,0,0).
    ps = g.get("phase_shift") or {}
    feats.append(float(ps.get("x", 0.0)))
    feats.append(float(ps.get("y", 0.0)))
    feats.append(float(ps.get("z", 0.0)))

    # normal_weights (default 1,1,1 if absent)
    nw = g.get("normal_weights") or {}
    feats.extend([float(nw.get("wx", 1.0)),
                  float(nw.get("wy", 1.0)),
                  float(nw.get("wz", 1.0))])

    # terms — only terms that are switched on (a term Sweep turned off is
    # absent from the geometry, so it is absent from the features too),
    # padded to MAX_TERMS. A term without a per-term phase (PI-TPMS terms,
    # older sweeps) encodes phase 0, as F13LD.mesh reads it.
    terms = [t for t in d["design"]["surface"]["terms"] if t.get("on", True)]
    for t_idx in range(MAX_TERMS):
        if t_idx < len(terms):
            term = terms[t_idx]
            tps = term.get("phase_shift") or {}
            feats.append(1.0)  # term active
            feats.append(float(term["coef"]))
            feats.append(float(tps.get("x", 0.0)))
            feats.append(float(tps.get("y", 0.0)))
            feats.append(float(tps.get("z", 0.0)))
            factors = term["factors"]
            for f_idx in range(MAX_FACTORS):
                if f_idx < len(factors):
                    fact = factors[f_idx]
                    func, axis = _parse_trig(fact["trig"])
                    feats.append(1.0)                        # factor active
                    feats.append(float(fact["fx"]))
                    feats.append(float(fact["fy"]))
                    feats.append(float(fact["fz"]))
                    feats.append(1.0 if func == "cos" else 0.0)  # sin=0, cos=1
                    feats.extend([1.0 if axis == a else 0.0 for a in TRIG_AXES])
                else:
                    feats.extend([0.0] * 8)  # inactive factor
        else:
            feats.extend([0.0] * (5 + MAX_FACTORS * 8))  # inactive term

    return feats


def design_overflow(d):
    """True when a design has more on-terms or factors than the encoder holds
    (the extra ones would be silently dropped from its features)."""
    terms = [t for t in d["design"]["surface"]["terms"] if t.get("on", True)]
    return len(terms) > MAX_TERMS or any(len(t["factors"]) > MAX_FACTORS for t in terms)


def _parse_trig(s):
    """e.g. 'sin(x)' -> ('sin','x'); 'cos(z)' -> ('cos','z')"""
    func = "cos" if s.startswith("cos") else "sin"
    axis = s[s.index("(") + 1]
    return func, axis


def feature_vector_dim():
    return len(MODES) + 4 + 3 + 3 + MAX_TERMS * (5 + MAX_FACTORS * 8)


# ============================================================
# OUTPUT EXTRACTION — geometry-only normalized values
# ============================================================

def extract_outputs(d, refs):
    """Geometry-only normalized output values from the design's raw solver
    values + the sweep's reference parameters. Returns {metric: value};
    a metric the row doesn't carry is NaN, which masks that row out of that
    metric's training only."""
    b = d["browser"]
    nan = float("nan")
    keffs = [b.get(k) for k in ("keff_x", "keff_y", "keff_z")]
    keff_avg = sum(keffs) / 3.0 if all(v is not None for v in keffs) else nan
    out = {
        "volume_fraction":    b["volume_fraction"],
        "ex_norm":            b["Ex_GPa"] / refs["E_solid"],
        "ey_norm":            b["Ey_GPa"] / refs["E_solid"],
        "ez_norm":            b["Ez_GPa"] / refs["E_solid"],
        "anisotropy":         b["anisotropy"] if b.get("anisotropy") is not None else nan,
        "pore_size_p50_norm": b["pore_size_p50_norm"],
        "pore_size_cv":       b["pore_size_cv"],
        "keff_avg_norm":      keff_avg / refs["k_solid"] if refs.get("k_solid") else nan,
        "surface_complexity": b["surface_complexity"],
        # some older rows lack directionality
        "directionality":     b["directionality"] if b.get("directionality") is not None else nan,
    }
    # Shear (F13LD.sweep v0.24+): prefer GPa / E_solid so it is normalized
    # exactly like ex/ey/ez; fall back to the sweep's own *_norm column.
    for axis in ("xy", "xz", "yz"):
        g = b.get(f"G{axis}_GPa")
        if g is not None and refs["E_solid"]:
            out[f"g{axis}_norm"] = g / refs["E_solid"]
        elif b.get(f"G{axis}_norm") is not None:
            out[f"g{axis}_norm"] = b[f"G{axis}_norm"]
        else:
            out[f"g{axis}_norm"] = nan
    return out


# ============================================================
# TRAINING
# ============================================================

def train(family, design_iter, out_path, n_estimators=150, decimals=4, data_source=None,
          max_norm=1.3, n_seeds=600, solver_version=None):
    print(f"\n=== F13LD.Synth trainer v{TRAINER_VERSION} · family={family} ===")
    print(f"Hyperparameters: n_estimators={n_estimators}, threshold_precision={decimals} decimals, "
          f"max_terms={MAX_TERMS}, max_factors={MAX_FACTORS}, outlier cutoff={max_norm}")
    if data_source:
        print(f"Data source: {data_source}")
    if solver_version:
        print(f"Keeping only rows whose solver_version starts with '{solver_version}'")

    X, out_dicts, y_validity, sources, presets = [], [], [], [], []
    # Track every reason a row is dropped or changed so the load is auditable.
    n_outliers = n_encode_fail = n_overflow = n_solver_skip = 0
    validity_counts = {}
    for d, refs in design_iter:
        if solver_version:
            sv_str = str(d["browser"].get("solver_version") or "")
            if not sv_str.startswith(solver_version):
                n_solver_skip += 1
                continue
        try:
            x = encode_design_unified(d)
        except Exception:
            n_encode_fail += 1
            continue
        if design_overflow(d):
            n_overflow += 1
        X.append(x)
        sv = d["browser"].get("solver_validity")
        validity_counts[sv] = validity_counts.get(sv, 0) + 1
        # "valid" rows have all axes from the solver. "partial" rows have one
        # or more axes deliberately zeroed because connectivity check failed
        # along that axis — that 0 is the correct answer for that axis, not
        # missing data, so partial rows are usable training signal too.
        is_valid = sv in ("valid", "partial")
        y_validity.append(1 if is_valid else 0)
        outputs = None
        if is_valid:
            try:
                outputs = extract_outputs(d, refs)
                # The FFT-CG axial-only solver behind the current Vault data
                # lands ratios up to ~1.10 (no shear relaxation). The cutoff
                # allows that plus headroom but catches corrupted rows (the
                # ex_norm ~10x values that poisoned v0.1's Vault retraining).
                # Data from Sweep's newer GPU solver may warrant a tighter one.
                # (a masked NaN thermal value compares False, so it never trips this)
                if max(outputs["ex_norm"], outputs["ey_norm"], outputs["ez_norm"]) > max_norm \
                        or outputs["keff_avg_norm"] > max_norm:
                    y_validity[-1] = 0
                    outputs = None
                    n_outliers += 1
            except Exception:
                # Missing fields: drop from metrics training, keep for validity
                y_validity[-1] = 0
                outputs = None
        out_dicts.append(outputs)
        sources.append(refs.get("source_file", ""))
        presets.append(_canonical_preset(refs.get("preset")))

    # Which targets this bundle trains: the core set, plus any optional
    # target enough rows carry.
    metrics = list(OUTPUT_METRICS)
    for m in OPTIONAL_METRICS:
        n_have = sum(1 for o in out_dicts if o is not None and np.isfinite(o.get(m, float("nan"))))
        if n_have >= MIN_OPTIONAL_LABELS:
            metrics.append(m)
            print(f"  Optional target {m}: {n_have} rows carry it — training it")
        elif n_have > 0:
            print(f"  Optional target {m}: only {n_have} rows carry it (need {MIN_OPTIONAL_LABELS}) — skipped")
    nan = float("nan")
    Y_metrics = [[(o.get(m, nan) if o is not None else 0.0) for m in metrics] for o in out_dicts]

    X = np.array(X, dtype=np.float32)
    Y_metrics = np.array(Y_metrics, dtype=np.float32)
    y_validity = np.array(y_validity, dtype=np.float32)
    if len(X) < 30:
        sys.exit(f"Only {len(X)} designs loaded — need at least 30 to train. "
                 "Run more sweeps or check the family filter.")

    print(f"Designs loaded: {len(X)} ({int(y_validity.sum())} usable for metric regression)")
    print(f"Feature dim: {X.shape[1]}")
    if validity_counts:
        breakdown = ", ".join(f"{k or 'null'}={v}" for k, v in sorted(validity_counts.items(), key=lambda kv: -kv[1]))
        print(f"  solver_validity breakdown: {breakdown}")
    if n_outliers > 0:
        print(f"  Dropped {n_outliers} rows for normalized stiffness/thermal > {max_norm} (likely data corruption)")
    if n_encode_fail:
        print(f"  Skipped {n_encode_fail} rows whose recipe could not be encoded")
    if n_solver_skip:
        print(f"  Skipped {n_solver_skip} rows from other solver versions")
    if n_overflow:
        print(f"  WARNING: {n_overflow} rows have more terms/factors than the encoder holds "
              f"(max_terms={MAX_TERMS}, max_factors={MAX_FACTORS}); their extra terms are not seen. "
              f"Raise --max-terms.")
    preset_counts = {}
    for i, pz in enumerate(presets):
        if y_validity[i] == 1:
            preset_counts[pz] = preset_counts.get(pz, 0) + 1
    print("  usable rows by preset: " + ", ".join(f"{k}={v}" for k, v in sorted(preset_counts.items(), key=lambda kv: -kv[1])))

    # Split — stratified on validity if mixed, simple otherwise
    if 0.05 < y_validity.mean() < 0.95:
        idx_tr, idx_te = train_test_split(
            np.arange(len(X)), test_size=0.2, random_state=SEED, stratify=y_validity)
    else:
        idx_tr, idx_te = train_test_split(np.arange(len(X)), test_size=0.2, random_state=SEED)

    valid_mask = (y_validity == 1)
    valid_in_tr = [i for i in idx_tr if valid_mask[i]]
    valid_in_te = [i for i in idx_te if valid_mask[i]]

    # Input normalization (fit on full train set)
    in_lo = X[idx_tr].min(axis=0)
    in_hi = X[idx_tr].max(axis=0)
    in_span = np.where((in_hi - in_lo) > 1e-8, in_hi - in_lo, 1.0)

    # Validity classifier — only train if both classes present in train set
    val_meta = None
    if 0.05 < y_validity[idx_tr].mean() < 0.95:
        print("\n--- Training validity classifier ---")
        clf = RandomForestClassifier(n_estimators=n_estimators, random_state=SEED,
                                     max_depth=15, min_samples_leaf=2, n_jobs=-1)
        clf.fit(X[idx_tr], y_validity[idx_tr])
        val_acc = clf.score(X[idx_te], y_validity[idx_te])
        print(f"Validity test accuracy: {val_acc:.3f}")
        val_meta = {"accuracy": float(val_acc)}
    else:
        print(f"\n(Skipping validity classifier — valid rate {y_validity.mean()*100:.1f}% degenerate)")
        clf = None

    # Metrics regressor — RF, one tree ensemble per output
    print("\n--- Training metrics regressor ---")
    Xm_tr = X[valid_in_tr]
    Xm_te = X[valid_in_te]
    Ym_tr = Y_metrics[valid_in_tr]
    Ym_te = Y_metrics[valid_in_te]
    print(f"Metrics training samples: {len(Xm_tr)}, test: {len(Xm_te)}")

    regressors = []
    test_r2 = {}
    test_sigma = {}
    knn_r2 = {}
    for i, mname in enumerate(metrics):
        # Derived metrics don't have their own RF — they're computed from other
        # metrics' predictions at synth runtime. Skip training, leave a None
        # placeholder, evaluate after the trained loop completes.
        if mname in DERIVED_METRICS:
            regressors.append(None)
            continue
        # Per-metric NaN/inf filter. Most metrics are well-defined for both
        # 'valid' and 'partial' rows, but anisotropy = max(E)/min(E) blows up
        # to inf or NaN whenever a partial row has a zero-stiffness axis (the
        # deliberate axis-zeroing from the FFT-CG connectivity check). Mask
        # those rows out of THIS metric's train/test set without affecting
        # any other metric's training.
        tr_mask = np.isfinite(Ym_tr[:, i])
        te_mask = np.isfinite(Ym_te[:, i])
        n_drop = int((~tr_mask).sum() + (~te_mask).sum())
        # Per-metric percentile trim. Heavy-tailed metrics (e.g. anisotropy with
        # max=75 dominated by p99≈25) train poorly because the RF can't predict
        # the rare tail and SS_tot is dominated by it. Trim defined in TRAIN_LIMITS.
        # Percentiles computed from the NaN-filtered TRAINING labels only — test
        # set then trimmed using those same bounds (so the model is judged on
        # the same distribution it was trained on).
        n_trim = 0
        if mname in TRAIN_LIMITS:
            p_lo, p_hi = TRAIN_LIMITS[mname]
            tr_labels_finite = Ym_tr[tr_mask, i]
            lo_val = float(np.percentile(tr_labels_finite, p_lo))
            hi_val = float(np.percentile(tr_labels_finite, p_hi))
            tr_in_range = (Ym_tr[:, i] >= lo_val) & (Ym_tr[:, i] <= hi_val)
            te_in_range = (Ym_te[:, i] >= lo_val) & (Ym_te[:, i] <= hi_val)
            n_trim = int(((tr_mask & ~tr_in_range).sum() +
                          (te_mask & ~te_in_range).sum()))
            tr_mask = tr_mask & tr_in_range
            te_mask = te_mask & te_in_range
        Xm_tr_i = Xm_tr[tr_mask]
        Ym_tr_i = Ym_tr[tr_mask, i]
        Xm_te_i = Xm_te[te_mask]
        Ym_te_i = Ym_te[te_mask, i]
        if len(Xm_tr_i) < 10 or len(Xm_te_i) < 2:
            sys.exit(
                f"Metric '{mname}' has too few finite labels after NaN/inf filtering "
                f"(train={len(Xm_tr_i)}, test={len(Xm_te_i)}). "
                f"Investigate the data — most metrics should not produce NaN."
            )

        rf = RandomForestRegressor(n_estimators=n_estimators, random_state=SEED,
                                   max_depth=15, min_samples_leaf=2, n_jobs=-1)
        rf.fit(Xm_tr_i, Ym_tr_i)
        regressors.append(rf)
        y_pred = rf.predict(Xm_te_i)
        residuals = Ym_te_i - y_pred
        ss_tot = ((Ym_te_i - Ym_te_i.mean()) ** 2).sum()
        ss_res = (residuals ** 2).sum()
        r2 = 1 - ss_res / max(ss_tot, 1e-12)
        sigma = float(np.std(residuals))
        knn = KNeighborsRegressor(n_neighbors=min(5, len(Xm_tr_i)-1)).fit(Xm_tr_i, Ym_tr_i)
        knn_pred = knn.predict(Xm_te_i)
        ss_res_k = ((Ym_te_i - knn_pred) ** 2).sum()
        r2_k = 1 - ss_res_k / max(ss_tot, 1e-12)
        test_r2[mname] = float(r2)
        test_sigma[mname] = sigma
        knn_r2[mname] = float(r2_k)
        verdict = "✓" if r2 > r2_k else "·"
        notes = []
        if n_drop > 0: notes.append(f"dropped {n_drop} NaN/inf")
        if n_trim > 0:
            p_lo, p_hi = TRAIN_LIMITS[mname]
            notes.append(f"trimmed {n_trim} outside p{p_lo}-p{p_hi}")
        note_str = f"   ({'; '.join(notes)})" if notes else ""
        print(f"  {mname:<22} R² = {r2:>+6.3f}   σ_resid = {sigma:>7.4f}   (KNN baseline {r2_k:>+6.3f}) {verdict}{note_str}")

    # Evaluate derived metrics using the trained regressors. For each derived
    # metric, predict its inputs on the test set, apply the op, compare to
    # ground truth. Same R²/σ math as trained metrics so the bell sparkline
    # and scoring use comparable yardsticks.
    for mname, spec in DERIVED_METRICS.items():
        i = metrics.index(mname)
        te_mask = np.isfinite(Ym_te[:, i])
        n_drop_te = int((~te_mask).sum())
        if te_mask.sum() < 2:
            test_r2[mname] = float("nan")
            test_sigma[mname] = float("nan")
            knn_r2[mname] = float("nan")
            print(f"  {mname:<22} SKIPPED — insufficient finite ground truth (n={int(te_mask.sum())})")
            continue
        Xm_te_d = Xm_te[te_mask]
        Ym_te_d = Ym_te[te_mask, i]
        # Get input predictions on the derived metric's test rows
        input_arrs = []
        for inp_name in spec["inputs"]:
            inp_i = metrics.index(inp_name)
            if regressors[inp_i] is None:
                sys.exit(f"Derived metric '{mname}' depends on '{inp_name}' which has no trained regressor.")
            input_arrs.append(regressors[inp_i].predict(Xm_te_d))
        # Apply the op
        if spec["op"] == "max_over_min":
            floor = spec.get("floor", 0.01)
            stacked = np.stack(input_arrs, axis=1)
            y_pred = stacked.max(axis=1) / np.maximum(stacked.min(axis=1), floor)
        else:
            sys.exit(f"Unknown derived op '{spec['op']}' for metric '{mname}'")
        residuals = Ym_te_d - y_pred
        ss_tot = ((Ym_te_d - Ym_te_d.mean()) ** 2).sum()
        ss_res = (residuals ** 2).sum()
        r2 = 1 - ss_res / max(ss_tot, 1e-12)
        sigma = float(np.std(residuals))
        # KNN baseline on ground-truth labels (apples-to-apples with trained metrics)
        tr_mask = np.isfinite(Ym_tr[:, i])
        if tr_mask.sum() >= 5:
            knn = KNeighborsRegressor(n_neighbors=min(5, int(tr_mask.sum())-1)).fit(
                Xm_tr[tr_mask], Ym_tr[tr_mask, i]
            )
            knn_pred = knn.predict(Xm_te_d)
            ss_res_k = ((Ym_te_d - knn_pred) ** 2).sum()
            r2_k = 1 - ss_res_k / max(ss_tot, 1e-12)
        else:
            r2_k = float("nan")
        test_r2[mname] = float(r2)
        test_sigma[mname] = sigma
        knn_r2[mname] = float(r2_k)
        verdict = "✓" if (not np.isnan(r2_k) and r2 > r2_k) else "·"
        drop_note = f"   (dropped {n_drop_te} NaN/inf)" if n_drop_te > 0 else ""
        print(f"  {mname:<22} R² = {r2:>+6.3f}   σ_resid = {sigma:>7.4f}   (KNN baseline {r2_k:>+6.3f}) {verdict}{drop_note}   [DERIVED]")

    mean_r2 = float(np.mean(list(test_r2.values())))
    mean_knn = float(np.mean(list(knn_r2.values())))
    print(f"\nMean R²: model={mean_r2:.3f}, KNN baseline={mean_knn:.3f}")
    if mean_r2 < mean_knn:
        print("WARNING: Model is worse than KNN baseline. Consider more data or feature changes.")

    # ----- Seed samples for browser-side candidate generation ----
    # Synth grows candidates by mutating these real training designs, and
    # its preset filter picks seeds by preset — so seeds are drawn evenly
    # across presets (every preset gets a fair share, small ones in full).
    rs = np.random.RandomState(SEED)
    by_preset = {}
    for row_i in valid_in_tr:
        by_preset.setdefault(presets[row_i], []).append(row_i)
    budget = min(n_seeds, len(valid_in_tr))
    chosen = []
    groups = sorted(by_preset.items(), key=lambda kv: len(kv[1]))
    for gi, (pz, rows) in enumerate(groups):
        share = (budget - len(chosen)) // (len(groups) - gi)
        take = rs.choice(rows, min(share, len(rows)), replace=False).tolist()
        chosen.extend(take)
    seed_samples = np.round(X[chosen], decimals=4).tolist()
    seed_presets = [presets[i] for i in chosen]
    print(f"\nSeeds: {len(chosen)} across {len(by_preset)} presets")

    # Export bundle
    bundle = {
        "meta": {
            "family": family,
            "version": "0.2.0",
            "trainer_version": TRAINER_VERSION,
            "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "data_source": data_source or "unknown",
            "n_designs_total": int(len(X)),
            "n_valid": int(y_validity.sum()),
            "n_metrics_train": int(len(valid_in_tr)),
            "n_metrics_test": int(len(valid_in_te)),
            "feature_dim": int(X.shape[1]),
            "n_overflow": int(n_overflow),
            "max_norm": float(max_norm),
            "solver_version": solver_version,
        },
        "encoding": {
            "modes": MODES,
            "trig_axes": TRIG_AXES,
            "max_terms": MAX_TERMS,
            "max_factors": MAX_FACTORS,
            # v0.2.0 bundles encode only switched-on terms; older bundles
            # encoded every listed term. Synth reads this.
            "term_semantics": "on_terms",
        },
        "input_norm": {"lo": in_lo.tolist(), "hi": in_hi.tolist()},
        "output_metrics": metrics,
        "output_ranges": {
            m: {"min": float(np.nanmin(Ym_tr[:, i])), "max": float(np.nanmax(Ym_tr[:, i]))}
            for i, m in enumerate(metrics)
        },
        "validity_model": _serialize_rf_classifier(clf, decimals=decimals) if clf is not None else None,
        "metrics_model": [_serialize_metric_model(r, metrics[i], decimals=decimals) for i, r in enumerate(regressors)],
        "seed_samples": seed_samples,
        "seed_presets": seed_presets,
        "eval": {
            "validity": val_meta,
            "metrics_test_r2": test_r2,
            "metrics_test_sigma": test_sigma,
            "knn_baseline_r2": knn_r2,
            "mean_r2": mean_r2,
            "mean_knn_r2": mean_knn,
        },
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(bundle))
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"\nExported: {out_path}  ({size_mb:.2f} MB)")
    if size_mb > 10:
        print("Note: model file is large. RF compresses well with gzip — consider serving gzipped.")


def _serialize_rf_classifier(clf, decimals=4):
    """Serialize a RandomForestClassifier into a JSON-safe dict.
    Browser-side inference walks each tree by feature/threshold/leaf-value."""
    return {
        "kind": "rf_classifier",
        "n_classes": int(len(clf.classes_)),
        "classes": clf.classes_.tolist(),
        "trees": [_serialize_tree(t.tree_, classifier=True, decimals=decimals) for t in clf.estimators_],
    }


def _serialize_metric_model(entry, mname, decimals=4):
    """Wrap a regressors[] entry for bundle export. Trained metrics get the
    full RF serialization; derived metrics emit their op spec instead."""
    if entry is None:
        spec = DERIVED_METRICS.get(mname)
        if spec is None:
            raise ValueError(f"Metric '{mname}' has no regressor and no derived spec.")
        return {"kind": "derived", **spec}
    return _serialize_rf_regressor(entry, decimals=decimals)


def _serialize_rf_regressor(rf, decimals=4):
    return {
        "kind": "rf_regressor",
        "trees": [_serialize_tree(t.tree_, classifier=False, decimals=decimals) for t in rf.estimators_],
    }


def _serialize_tree(t, classifier=False, decimals=4):
    """Pack a sklearn Tree into compact arrays.
    Browser walks: at node i, if i is leaf, return value[i]; else compare
    feature[i] threshold[i], descend left or right.

    Float precision is rounded to `decimals` places to shrink JSON output.
    Inputs are min-max normalized to [0,1], so 4 decimals = 1e-4 resolution
    on thresholds (well below any meaningful input variation). Leaf values
    are also normalized targets, so 4 decimals is more than the trees can
    meaningfully resolve. Expected R² impact: <0.005."""
    if classifier:
        # For binary classifier, store P(class=1) per leaf
        values = (t.value[:, 0, 1] / t.value[:, 0, :].sum(axis=1))
    else:
        values = t.value[:, 0, 0]
    # Round and convert. Threshold for inactive (leaf) nodes is sklearn's
    # sentinel -2.0 — preserve it exactly so the browser-side tree walker
    # can distinguish leaves from internal nodes.
    thresholds = t.threshold.copy()
    leaf_mask = (t.children_left == -1)
    thresholds[~leaf_mask] = np.round(thresholds[~leaf_mask], decimals)
    return {
        "feature": t.feature.tolist(),
        "threshold": thresholds.tolist(),
        "left": t.children_left.tolist(),
        "right": t.children_right.tolist(),
        "value": np.round(values, decimals).tolist(),
    }


# ============================================================
# CLI
# ============================================================

def main():
    global MAX_TERMS
    p = argparse.ArgumentParser(description="F13LD.Synth offline trainer (Pattern A)")
    p.add_argument("family", choices=["tpms", "noise", "grain"],
                   help="Which family to train. The trainer pulls all designs in this family.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--from-files", metavar="DIR",
                     help="Read sweep_results_*.json files from this directory")
    src.add_argument("--from-vault", action="store_true",
                     help="Pull from F13LD.vault (requires vault_client.py)")
    p.add_argument("--out", required=True,
                   help="Output path for the trained model JSON (e.g. weights/tpms.json)")
    p.add_argument("--n-estimators", type=int, default=150,
                   help="Trees per forest (default 150 — half of sklearn default for ~2x model size reduction with small R² hit)")
    p.add_argument("--decimals", type=int, default=4,
                   help="Float precision for tree thresholds and leaf values (default 4)")
    p.add_argument("--since", default=None, metavar="YYYY-MM-DD",
                   help="(vault-mode) only fetch designs created on/after this date")
    p.add_argument("--limit", type=int, default=None,
                   help="(vault-mode) cap on rows fetched, useful for quick test runs")
    p.add_argument("--max-terms", type=int, default=MAX_TERMS,
                   help=f"term slots in the feature vector (default {MAX_TERMS}; Synth reads it from the bundle)")
    p.add_argument("--max-norm", type=float, default=1.3,
                   help="drop rows whose normalized stiffness or thermal exceeds this (default 1.3)")
    p.add_argument("--n-seeds", type=int, default=600,
                   help="training designs exported as search seeds, spread across presets (default 600)")
    p.add_argument("--solver-version", default=None, metavar="PREFIX",
                   help="keep only rows whose solver_version starts with PREFIX (for a Vault holding mixed solvers)")
    args = p.parse_args()
    MAX_TERMS = args.max_terms

    if args.from_files:
        design_iter = load_from_files(args.from_files, args.family)
        data_source = f"files:{args.from_files}"
    else:
        design_iter = load_from_vault(args.family, since=args.since, limit=args.limit)
        ds_parts = ["vault"]
        if args.since:
            ds_parts.append(f"since={args.since}")
        if args.limit:
            ds_parts.append(f"limit={args.limit}")
        data_source = "+".join(ds_parts)

    train(args.family, design_iter, args.out,
          n_estimators=args.n_estimators, decimals=args.decimals,
          data_source=data_source, max_norm=args.max_norm,
          n_seeds=args.n_seeds, solver_version=args.solver_version)


if __name__ == "__main__":
    main()

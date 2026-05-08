"""
F13LD.Synth — Pattern A trainer (CLI, family-level)

Usage
-----
File-mode (works today, reads sweep JSONs from a directory):
    python train_synth.py tpms --from-files ./sweeps --out ./weights/tpms.json

Vault-mode (stubbed; activates once vault_client.py is wired):
    python train_synth.py tpms --from-vault --out ./weights/tpms.json

What it does
------------
1. Loads all sweep designs for the requested family (tpms / noise / grain / …)
2. Encodes each design via the UNIFIED FEATURE VECTOR — works across all configs
   within the family (any mode, any term count, any factor frequency)
3. Computes geometry-only normalized outputs from raw FEA values + each sweep's
   reference parameters (E_solid, sigma_ref, k_solid, cell_mm)
4. Trains:
     - Validity classifier (Random Forest, all designs) — predicts P(valid)
     - Metrics regressor   (Random Forest, valid only) — predicts 9 normalized outputs
5. Evaluates on a 20% held-out split, reports per-metric R²
6. Exports model + metadata to a single JSON the browser can load

Prereqs
-------
    pip install numpy scikit-learn

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
    "anisotropy",           # ratio
    "pore_size_norm",       # ×cell_µm → physical µm at inference
    "pore_size_cv",         # ratio
    "keff_avg_norm",        # ×thermal_k_wmk → physical W/mK at inference
    "surface_complexity",   # ratio (Vault percentile-ranks at display)
]

# Unified-feature encoding constants — must accommodate the largest known
# sweep configuration. If sweeps ever exceed these, bump the values and retrain.
MODES = ["pi-tpms", "shell", "solid", "level"]   # extend as new modes appear
TRIG_AXES = ["x", "y", "z"]
MAX_TERMS = 6      # any single sweep has had at most 5 terms
MAX_FACTORS = 4    # any single sweep has had at most 3 factors per term


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


def load_from_vault(family_filter):
    """Yield (design_dict, reference_params) for every design in F13LD.vault
    matching the family. Stubbed until vault_client.py lands."""
    # TODO: implement once Vault client is provided
    #
    # from vault_client import VaultClient
    # vault = VaultClient()
    # for row in vault.iterByFamily(family_filter):
    #     # Pull design recipe + reference params out of JSONB
    #     yield row["design_json"], {
    #         "E_solid":   row["material_context"].get("E_solid_GPa"),
    #         "k_solid":   row["material_context"].get("k_solid_W_mK"),
    #         "sigma_ref": row["material_context"].get("sigma_ref_GPa"),
    #         "cell_mm":   row["cell_mm"],
    #         "source_file": "vault",
    #         "preset":    row["preset"],
    #     }
    raise NotImplementedError(
        "Vault-mode is not yet wired. Wait for vault_client.py, then implement this function. "
        "Use --from-files for now."
    )


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

    # normal_weights (default 1,1,1 if absent)
    nw = g.get("normal_weights") or {}
    feats.extend([float(nw.get("wx", 1.0)),
                  float(nw.get("wy", 1.0)),
                  float(nw.get("wz", 1.0))])

    # terms — pad to MAX_TERMS
    terms = d["design"]["surface"]["terms"]
    for t_idx in range(MAX_TERMS):
        if t_idx < len(terms):
            term = terms[t_idx]
            feats.append(1.0)  # term active
            feats.append(float(term["coef"]))
            feats.append(float(term["phase_shift"]["x"]))
            feats.append(float(term["phase_shift"]["y"]))
            feats.append(float(term["phase_shift"]["z"]))
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


def _parse_trig(s):
    """e.g. 'sin(x)' -> ('sin','x'); 'cos(z)' -> ('cos','z')"""
    func = "cos" if s.startswith("cos") else "sin"
    axis = s[s.index("(") + 1]
    return func, axis


def feature_vector_dim():
    return len(MODES) + 4 + 3 + MAX_TERMS * (5 + MAX_FACTORS * 8)


# ============================================================
# OUTPUT EXTRACTION — geometry-only normalized values
# ============================================================

def extract_outputs(d, refs):
    """Compute the 9 geometry-only normalized output values from the design's
    raw FEA values + the sweep's reference parameters."""
    b = d["browser"]
    cell_um = refs["cell_mm"] * 1000.0
    keff_avg = (b["keff_x"] + b["keff_y"] + b["keff_z"]) / 3.0
    return [
        b["volume_fraction"],
        b["Ex_GPa"] / refs["E_solid"],
        b["Ey_GPa"] / refs["E_solid"],
        b["Ez_GPa"] / refs["E_solid"],
        b["anisotropy"],
        b["pore_size"] / cell_um,
        b["pore_size_cv"],
        keff_avg / refs["k_solid"],
        b["surface_complexity"],
    ]


# ============================================================
# TRAINING
# ============================================================

def train(family, design_iter, out_path, n_estimators=150, decimals=4):
    print(f"\n=== F13LD.Synth trainer · family={family} ===")
    print(f"Hyperparameters: n_estimators={n_estimators}, threshold_precision={decimals} decimals")

    X, Y_metrics, y_validity, sources = [], [], [], []
    for d, refs in design_iter:
        try:
            X.append(encode_design_unified(d))
        except Exception as e:
            continue
        is_valid = d["browser"].get("solver_validity") == "valid"
        y_validity.append(1 if is_valid else 0)
        if is_valid:
            try:
                Y_metrics.append(extract_outputs(d, refs))
            except Exception:
                # If we can't extract outputs (e.g. missing fields), drop from
                # metrics training but keep in validity training set
                y_validity[-1] = 0
                Y_metrics.append([0.0] * len(OUTPUT_METRICS))
        else:
            Y_metrics.append([0.0] * len(OUTPUT_METRICS))  # placeholder; not used
        sources.append(refs.get("source_file", ""))

    X = np.array(X, dtype=np.float32)
    Y_metrics = np.array(Y_metrics, dtype=np.float32)
    y_validity = np.array(y_validity, dtype=np.float32)
    if len(X) < 30:
        sys.exit(f"Only {len(X)} designs loaded — need at least 30 to train. "
                 "Run more sweeps or check the family filter.")

    print(f"Designs loaded: {len(X)} ({int(y_validity.sum())} valid)")
    print(f"Feature dim: {X.shape[1]}")

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
    knn_r2 = {}
    for i, mname in enumerate(OUTPUT_METRICS):
        rf = RandomForestRegressor(n_estimators=n_estimators, random_state=SEED,
                                   max_depth=15, min_samples_leaf=2, n_jobs=-1)
        rf.fit(Xm_tr, Ym_tr[:, i])
        regressors.append(rf)
        y_pred = rf.predict(Xm_te)
        ss_tot = ((Ym_te[:, i] - Ym_te[:, i].mean()) ** 2).sum()
        ss_res = ((Ym_te[:, i] - y_pred) ** 2).sum()
        r2 = 1 - ss_res / max(ss_tot, 1e-12)
        knn = KNeighborsRegressor(n_neighbors=min(5, len(Xm_tr)-1)).fit(Xm_tr, Ym_tr[:, i])
        knn_pred = knn.predict(Xm_te)
        ss_res_k = ((Ym_te[:, i] - knn_pred) ** 2).sum()
        r2_k = 1 - ss_res_k / max(ss_tot, 1e-12)
        test_r2[mname] = float(r2)
        knn_r2[mname] = float(r2_k)
        verdict = "✓" if r2 > r2_k else "·"
        print(f"  {mname:<22} R² = {r2:>+6.3f}   (KNN baseline {r2_k:>+6.3f}) {verdict}")

    mean_r2 = float(np.mean(list(test_r2.values())))
    mean_knn = float(np.mean(list(knn_r2.values())))
    print(f"\nMean R²: model={mean_r2:.3f}, KNN baseline={mean_knn:.3f}")
    if mean_r2 < mean_knn:
        print("WARNING: Model is worse than KNN baseline. Consider more data or feature changes.")

    # ----- Pick seed samples for browser-side candidate generation ----
    # Browser samples candidates by perturbing these seeds with small Gaussian
    # noise — keeps the search near training distribution so predictions stay
    # reliable. Uniform [lo, hi] sampling in 233-D would be mostly OOD.
    n_seeds = min(200, len(Xm_tr))
    seed_idx = np.random.RandomState(SEED).choice(len(Xm_tr), n_seeds, replace=False)
    seed_samples = np.round(Xm_tr[seed_idx], decimals=4).tolist()

    # Export bundle
    bundle = {
        "meta": {
            "family": family,
            "version": "0.1.0",
            "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "n_designs_total": int(len(X)),
            "n_valid": int(y_validity.sum()),
            "n_metrics_train": int(len(valid_in_tr)),
            "n_metrics_test": int(len(valid_in_te)),
            "feature_dim": int(X.shape[1]),
        },
        "encoding": {
            "modes": MODES,
            "trig_axes": TRIG_AXES,
            "max_terms": MAX_TERMS,
            "max_factors": MAX_FACTORS,
        },
        "input_norm": {"lo": in_lo.tolist(), "hi": in_hi.tolist()},
        "output_metrics": OUTPUT_METRICS,
        "output_ranges": {
            m: {"min": float(Ym_tr[:, i].min()), "max": float(Ym_tr[:, i].max())}
            for i, m in enumerate(OUTPUT_METRICS)
        },
        "validity_model": _serialize_rf_classifier(clf, decimals=decimals) if clf is not None else None,
        "metrics_model": [_serialize_rf_regressor(rf, decimals=decimals) for rf in regressors],
        "seed_samples": seed_samples,
        "eval": {
            "validity": val_meta,
            "metrics_test_r2": test_r2,
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
    args = p.parse_args()

    if args.from_files:
        design_iter = load_from_files(args.from_files, args.family)
    else:
        design_iter = load_from_vault(args.family)

    train(args.family, design_iter, args.out,
          n_estimators=args.n_estimators, decimals=args.decimals)


if __name__ == "__main__":
    main()

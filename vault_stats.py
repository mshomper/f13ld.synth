#!/usr/bin/env python3
"""
vault_stats.py — Cross-design distribution statistics for any group of metrics
in F13LD.vault.

Use this to assess whether a metric is well-behaved enough to be a training
target. Tight distributions (low CV, no extreme tail) train cleanly; heavy-
tailed metrics are noisy targets that the RF can't predict well.

Add more groups below as we investigate new metric families.

Usage (from C:\\Users\\mshom\\Desktop\\f13ld-synth-train\\):
    py vault_stats.py

Requires vault_client.py in the same folder.
"""
import json
import sys
import numpy as np

try:
    from vault_client import VaultClient
except ImportError:
    sys.exit("vault_client.py not found. Run this from the training folder.")


# ----------------------------------------------------------------------
# Metric groups. Each entry: (display_name, [metric_field_names]).
# Field names match what the sweep tool writes (top-level row column OR
# nested in recipe.homogenization — the script tries both).
# ----------------------------------------------------------------------
METRIC_GROUPS = [
    ("PORE", [
        "pore_size_norm",       # currently retired training target — was the mean
        "pore_size_p10_norm",   # 10th percentile of within-design pore distribution
        "pore_size_p50_norm",   # median (currently the trained target)
        "pore_size_p90_norm",   # 90th percentile
        "pore_size_cv",         # coefficient of variation (std/mean within design)
    ]),
    ("ANISOTROPY", [
        "anisotropy",           # currently derived from Ex/Ey/Ez — heavy-tailed σ
        "directionality",       # alternative directional measure
        "aniso_efficiency",     # how efficiently the design uses material toward a preferred direction
        "ortho_contrast",       # contrast between principal stiffness axes
    ]),
]


def extract_metric(row, recipe_dict, metric_name):
    """Try top-level Vault column first, fall back to recipe.homogenization."""
    v = row.get(metric_name)
    if v is None and recipe_dict is not None:
        v = (recipe_dict.get("homogenization") or {}).get(metric_name)
    if v is None:
        return None
    try:
        fv = float(v)
        if np.isnan(fv) or np.isinf(fv):
            return None
        return fv
    except (TypeError, ValueError):
        return None


def parse_recipe(raw):
    if raw is None:
        return None
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None
    return None


def print_table(group_name, metric_names, rows):
    """Print one stats table for one metric group."""
    # Collect each metric independently — different rows may have different
    # subsets of metrics depending on which solver version produced them.
    data = {m: [] for m in metric_names}
    for row in rows:
        recipe = parse_recipe(row.get("recipe"))
        for m in metric_names:
            v = extract_metric(row, recipe, m)
            if v is not None:
                data[m].append(v)

    print(f"\n=== {group_name} ===")
    print(f"{'metric':<22} {'n':>5} {'min':>9} {'p10':>9} {'p50':>9} {'p90':>9} {'max':>9} {'mean':>9} {'std':>9} {'CV':>6}")
    print("-" * 105)
    for m in metric_names:
        vals = np.array(data[m])
        if len(vals) == 0:
            print(f"{m:<22} {0:>5}   (metric not present in any row)")
            continue
        mn, mx = vals.min(), vals.max()
        mean = vals.mean()
        std = vals.std()
        cv = std / mean if mean != 0 else float("nan")
        print(f"{m:<22} {len(vals):>5} "
              f"{mn:>9.4f} {np.percentile(vals, 10):>9.4f} "
              f"{np.percentile(vals, 50):>9.4f} {np.percentile(vals, 90):>9.4f} "
              f"{mx:>9.4f} {mean:>9.4f} {std:>9.4f} {cv:>6.3f}")


def main():
    try:
        vault = VaultClient()
    except ValueError as e:
        sys.exit(f"Vault setup error: {e}")

    print("Fetching all TPMS rows from F13LD.vault...")
    rows = vault.fetch_designs(
        family="tpms",
        valid_only=False,        # include valid + partial + invalid
        exclude_degenerate=True,
        verbose=True,
    )
    print(f"Loaded {len(rows)} rows total")

    for group_name, metric_names in METRIC_GROUPS:
        print_table(group_name, metric_names, rows)

    print()
    print("Reading guide:")
    print("  • Low CV (< ~0.5) = tight distribution = good training target")
    print("  • High CV (> ~1.0) = heavy-tailed = noisy training target")
    print("  • Compare 'mean' to 'p90' — mean > p90 indicates extreme-value skew")
    print("  • n = how many rows have that metric set; lower n = newer schema or solver path")


if __name__ == "__main__":
    main()

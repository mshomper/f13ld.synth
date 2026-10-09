"""Dev-only trainer smoke test without network access: replaces vault_client
with a fake Vault built from recipe JSON, with synthetic solver values, and
runs the real train_synth.py vault path end to end.

    python3 tests/trainer-smoke.py SEED_RECIPES.json OUT_BUNDLE.json

The fake rows deliberately include what used to break the trainer: terms
switched off, PI-TPMS terms with no per-term phase, a 9-term lidinoid, rows
with shear moduli, rows with no solid conductivity, and a few corrupted
stiffness values."""
import copy, json, math, random, sys, types, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
recipes = json.load(open(sys.argv[1]))
rnd = random.Random(5)
LIDINOID = [{"on": True, "coef": c, "factors": f} for c, f in [
    (1.1, [{"trig": "sin(x)", "fx": 2, "fy": 1, "fz": 1}, {"trig": "cos(y)", "fx": 1, "fy": 1, "fz": 1}, {"trig": "sin(z)", "fx": 1, "fy": 1, "fz": 1}])] * 3 + [
    (-0.2, [{"trig": "cos(x)", "fx": 2, "fy": 1, "fz": 1}, {"trig": "cos(y)", "fx": 1, "fy": 2, "fz": 1}])] * 3 + [
    (-0.4, [{"trig": "cos(z)", "fx": 1, "fy": 1, "fz": 2}])] * 3]

def fake_rows():
    rows = []
    for i in range(1500):
        r = copy.deepcopy(recipes[i % len(recipes)])
        g, terms = r["geometry"], r["surface"]["terms"]
        if i % 50 == 7:
            r["surface"]["terms"] = copy.deepcopy(LIDINOID); r["meta"]["preset"] = "lidinoid"; terms = r["surface"]["terms"]
        if g["mode"] != "pi-tpms" and len(terms) > 3 and i % 3 == 0:
            terms[-1]["on"] = False                                   # switched off by Sweep's term mask
        if g["mode"] == "pi-tpms":
            for t in terms: t.pop("phase_shift", None)                # Sweep writes none for PI terms
        vf = {"solid": 40, "shell": 25, "pi-tpms": 6}[g["mode"]] + rnd.gauss(0, 4)
        e = max(0.002, (vf / 100) ** 2 * 1.2 + rnd.gauss(0, 0.004))
        homog = {"volume_fraction": vf, "Ex_GPa": e * 110, "Ey_GPa": e * 110 * rnd.uniform(.8, 1.2),
                 "Ez_GPa": e * 110 * rnd.uniform(.8, 1.2), "anisotropy": rnd.uniform(1, 3),
                 "pore_size_cv": rnd.uniform(.4, .9), "keff_x": vf / 100 * 7, "keff_y": vf / 100 * 7, "keff_z": vf / 100 * 7,
                 "surface_complexity": rnd.uniform(1, 1.5)}
        if i % 97 == 3: homog["Ex_GPa"] *= 400                       # corrupted row
        no_k = i % 7 == 2                                             # no solid conductivity: thermal masked, row kept
        row = {"family": "tpms", "recipe": dict(r, homogenization=homog), "e_solid_gpa": 110, "sigma_ref_gpa": 0.9,
               "cell_size_mm": 2, "material": None if no_k else {"k_W_mK": 7.0},
               "solver_validity": "invalid" if i % 11 == 0 else ("partial" if i % 5 == 0 else "valid"),
               "pore_size_p50_norm": rnd.uniform(.05, .4), "directionality": rnd.choice([1/3, 2/3, 1.0])}
        if i % 2 == 0:
            row["Gxy_GPa"] = e * 40; row["Gxz_GPa"] = e * 38; row["Gyz_GPa"] = e * 41
        rows.append(row)
    return rows

class FakeVault:
    def fetch_designs(self, **kw):
        rows = fake_rows()
        return rows[:kw["limit"]] if kw.get("limit") else rows
sys.modules["vault_client"] = types.SimpleNamespace(VaultClient=FakeVault)
import train_synth
sys.argv = ["train_synth.py", "tpms", "--from-vault", "--out", sys.argv[2], "--n-estimators", "20"]
train_synth.main()

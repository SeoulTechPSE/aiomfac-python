"""Generate notebooks/aiomfac_validation_13_oligomer_surrogate.ipynb: alpha-pinene SOA as monomers plus dimers.

Sect. 3.3 represents alpha-pinene SOA by one monomeric proxy, and Sect. 4 argues (i) that SOA behaves like a compound
of higher Tg than its monomeric tracers and (ii) that the combinatorial weighting of G2020 would limit how far
oligomers can raise the predicted viscosity. This notebook tests both with a pre-specified surrogate, without fitting
to the viscosity data:

* monomers: pinonic and pinic acid, equal mass; dimers: pinyl-diaterpenyl ester (C17H26O8) or pinonyl-pinyl ester
  (C19H28O7), two dimers identified in alpha-pinene SOA;
* dimer mass fraction of the dry organic f_d = 0, 0.1, 0.2, 0.3, 0.5 (a scan, not a fit);
* Tg: ML predictor for every compound; for the dimers also the elemental-composition estimate of DeRieux et al. (2018);
* mixing: G2020 and the mole-fraction rule (the quantity AIOMFAC-web v3.14 reports);
* the water content at each RH from AIOMFAC with the dry organic composition fixed, on the organic-rich branch (the
  first crossing of a_w = RH from the dry end; at high RH, where the water activity along the path passes through a
  maximum, the organic-rich liquid of a phase-separated particle).

Compared with the nine alpha-pinene data sets of Gervasi et al. (2020): MAE, the horizontal RH error of notebook 12,
and the 1 h mixing-time classification for 200 nm particles. Also the effective weight of the dimer,
w_eff = d log10 eta / d log10 eta0_dimer (G2020: gamma^C x_dimer; mole-fraction rule: x_dimer). Two sensitivity
analyses: a dimer fraction that rises as the particle dries (equilibrium-like, dimer/monomer mass ratio proportional to
1/a_w, f_d = 0.3 at RH 0.5, capped at 0.5), and, as a diagnostic, the f_d that minimizes the MAE of each data set.
Results: oligomer_surrogate_results.json.

Run:  python build_nb13.py && (cd ../notebooks && PATH=/opt/miniconda3/bin:$PATH PYTHONPATH=<v1.3.1 src> \
          jupyter nbconvert --to notebook --execute --inplace aiomfac_validation_13_oligomer_surrogate.ipynb \
          --ExecutePreprocessor.timeout=-1)
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md("""
# alpha-pinene SOA as a monomer-dimer surrogate

Pre-specified scan of the dimer mass fraction (no fit), two dimer structures, two Tg sources for the dimers, and the
G2020 and mole-fraction mixing rules.
""")

code(r'''
import json, math, re, zipfile, warnings
import numpy as np
from scipy.optimize import brentq
from rdkit import Chem
from rdkit.Chem.Descriptors import MolWt
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
import aiomfac_py
from aiomfac_py import ActivityModel, Component
from aiomfac_py.s2as import smiles_to_components
from aiomfac_py.tgml_armeli import predict_tg_ml_smiles
from aiomfac_py.viscosity import organic_mixture_viscosity, pure_organic_viscosity_vtf
warnings.filterwarnings("ignore")
print("aiomfac_py", aiomfac_py.__version__, aiomfac_py.__file__)

WATER = Component(1, "Water", ((16, 1),))
MONOMERS = {"pinonic acid": "CC(=O)C1CC(C1(C)C)CC(=O)O", "pinic acid": "CC1(C(CC1C(=O)O)CC(=O)O)C"}
DIMERS = {"pinyl-diaterpenyl ester": "CC1(C)C(CC1C(=O)O)CC(=O)OC(C)(C)C(CC(=O)O)CC(=O)O",
          "pinonyl-pinyl ester": "CC1(C)C(CC1C(=O)O)CC(=O)OCC(=O)C1CC(CC(=O)O)C1(C)C"}
F_D = (0.0, 0.1, 0.2, 0.3, 0.5)

def derieux_tg(smiles):
    """Tg (K) from elemental composition, DeRieux et al. (2018), CHO compounds."""
    m = Chem.MolFromSmiles(smiles)
    n = {"C": 0, "H": 0, "O": 0}
    for a in Chem.AddHs(m).GetAtoms():
        n[a.GetSymbol()] += 1
    lc, lh, lo = math.log(n["C"]), math.log(n["H"]), math.log(n["O"])
    return (12.13 + lc) * 10.95 - 41.82 * lh + 21.61 * lc * lh + 118.96 * lo - 24.38 * lc * lo

INFO = {}
for name, smi in {**MONOMERS, **DIMERS}.items():
    INFO[name] = dict(smiles=smi, formula=CalcMolFormula(Chem.MolFromSmiles(smi)),
                      M=MolWt(Chem.MolFromSmiles(smi)), tg_ml=predict_tg_ml_smiles(smi).tg_K,
                      tg_derieux=derieux_tg(smi))
    print(f"{name:26s} {INFO[name]['formula']:10s} M={INFO[name]['M']:6.1f}  Tg ML {INFO[name]['tg_ml']:6.1f} K  "
          f"DeRieux {INFO[name]['tg_derieux']:6.1f} K")
assert INFO["pinyl-diaterpenyl ester"]["formula"] == "C17H26O8"
assert INFO["pinonyl-pinyl ester"]["formula"] == "C19H28O7"

def build_model(dimer):
    smis = list(MONOMERS.values()) + [DIMERS[dimer]]
    r = smiles_to_components(smis, water_as_component1=True)
    assert not r.removed and all(n == 0 for n in r.n_unmatched_atoms), dimer
    comps = [WATER] + [Component(k + 2, nm, c.subgroups)
                       for k, (nm, c) in enumerate(zip(list(MONOMERS) + [dimer], r.components[1:]))]
    return ActivityModel(comps)


def mass_vector(w_water, f_d):
    dry = np.array([(1 - f_d) / 2, (1 - f_d) / 2, f_d])
    return np.concatenate([[w_water], (1 - w_water) * dry])

def aw_of(model, w_water, f_d, T):
    return float(model.evaluate(mass_vector(w_water, f_d), T, basis="mass").activity[0])

W_PATH = np.concatenate([np.geomspace(1e-5, 0.05, 120), np.linspace(0.05, 0.9995, 400)[1:]])

def path_scan(model, f_d, T):
    """Water activity along the drying path; the first local maximum (spinodal of the organic-rich branch) if any."""
    a = np.array([aw_of(model, w, f_d, T) for w in W_PATH])
    dec = np.where(np.diff(a) < -1e-9)[0]
    aw_spin = float(a[dec[0]]) if len(dec) else None
    return a, aw_spin

def w_for_aw(model, aw, f_d, T, a_path):
    """Water mass fraction of the organic-rich branch: the first crossing of a_w = aw along the path from dry."""
    i = np.where(a_path >= aw)[0]
    if len(i) == 0 or i[0] == 0:
        return None
    k = i[0]
    return brentq(lambda w: aw_of(model, w, f_d, T) - aw, W_PATH[k - 1], W_PATH[k], xtol=1e-9)

def log10_eta(model, x, T, eta0, mixing):
    try:
        e = organic_mixture_viscosity(model, np.asarray(x), T, eta0, mixing=mixing).eta_pas
        return math.log10(e) if np.isfinite(e) and e > 0 else np.nan
    except (ValueError, OverflowError):
        return np.nan
''')

md("## 1. alpha-pinene data and thresholds (as in notebooks 04 and 12)")

code(r'''
ZIP = zipfile.ZipFile("gervasi_supp.zip")
def parse_soa(name):
    lines = [l.strip().strip('"') for l in ZIP.read(name).decode("utf-8-sig").splitlines()]
    T = float(re.search(r"T\s*=\s*([\d.]+)\s*K", lines[0]).group(1))
    pts = []
    for line in lines[4:]:
        p = [s for s in re.split(r"[,\t]", line) if s.strip()]
        if len(p) >= 6:
            rh, eta, _, _, yn, yp = map(float, p[:6])
            if eta > 0 and 0 < rh < 100:
                pts.append((rh / 100.0, math.log10(eta), yn, yp))
    return T, np.array(pts)

FILES = sorted(n for n in ZIP.namelist() if "aPinene_SOA" in n and n.endswith(".txt"))
DATA = {fn.split("_SOA_")[1].rsplit(".txt", 1)[0]: parse_soa(fn) for fn in FILES}
TEMPS = sorted({T for T, _ in DATA.values()})
print(len(DATA), "data sets;", sum(len(p) for _, p in DATA.values()), "points; T =", TEMPS)

kB = 1.380649e-23
TAU_1H_200 = math.log10(kB * 295.0 * 4 * math.pi**2 * 3600.0 / (6 * math.pi * 0.4e-9 * (200e-9) ** 2))
RH_GRID = np.round(np.arange(0.01, 0.991, 0.01), 3)

def horizontal(rh_grid, y_grid, rh_m, y_m):
    v = np.isfinite(y_grid)
    r, y = rh_grid[v], y_grid[v]
    d = y - y_m
    idx = np.where(np.sign(d[:-1]) * np.sign(d[1:]) <= 0)[0]
    idx = idx[d[idx] != d[idx + 1]]
    if len(idx) == 0:
        return ("above", None) if y_m > np.nanmax(y) else ("below", None)
    roots = r[idx] - d[idx] * (r[idx + 1] - r[idx]) / (d[idx + 1] - d[idx])
    return "ok", float(roots[np.argmin(np.abs(roots - rh_m))])
''')

md("## 2. Water content along RH for each dimer, f_d and T (shared by all Tg and mixing cases)")

code(r'''
MODELS = {d: build_model(d) for d in DIMERS}
COMP = {}        # (dimer, f_d, T) -> list of mole-fraction vectors on RH_GRID (None where no solution)
MONO = {}
for dimer, model in MODELS.items():
    for f_d in F_D:
        for T in TEMPS:
            a_path, aw_spin = path_scan(model, f_d, T)
            MONO[f"{dimer}|{f_d}|{T}"] = dict(aw_spinodal=aw_spin)
            xs = []
            for rh in RH_GRID:
                w = w_for_aw(model, rh, f_d, T, a_path)
                xs.append(None if w is None else model.evaluate(mass_vector(w, f_d), T, basis="mass").x)
            COMP[(dimer, f_d, T)] = xs
spin = [v["aw_spinodal"] for v in MONO.values() if v["aw_spinodal"] is not None]
print(f"organic-rich branch ends (spinodal) at a_w {min(spin):.3f}-{max(spin):.3f} in {len(spin)} of {len(MONO)} cases"
      if spin else "water activity monotonic on every path")
''')

md("## 3. Viscosity curves and comparison with the data")

code(r'''
def eta0_map(T, tg_dimer):
    return {2: pure_organic_viscosity_vtf(T, INFO["pinonic acid"]["tg_ml"]),
            3: pure_organic_viscosity_vtf(T, INFO["pinic acid"]["tg_ml"]),
            4: pure_organic_viscosity_vtf(T, tg_dimer)}

def score(curves):
    """curves: T -> log10 eta on RH_GRID. Point MAE, per-data-set MAE (as Table 4), horizontal RH errors, tau class."""
    err, ds_mae, d_rh, n_above, tau_ok, n_meas_tau, n_pred_tau, n_total = [], [], [], 0, 0, 0, 0, 0
    for study, (T, pts) in DATA.items():
        y_grid = curves[T]
        pred = np.interp(pts[:, 0], RH_GRID, y_grid, left=np.nan, right=np.nan)
        e = pred - pts[:, 1]
        v = np.isfinite(e)
        err += list(e[v]); ds_mae.append(float(np.mean(np.abs(e[v]))))
        n_total += len(pts)
        for (rh, y, _, _), p in zip(pts, pred):
            st, rs = horizontal(RH_GRID, y_grid, rh, y)
            if st == "ok":
                d_rh.append(rs - rh)
            elif st == "above":
                n_above += 1
            if np.isfinite(p):
                tau_ok += (p > TAU_1H_200) == (y > TAU_1H_200)
                n_meas_tau += y > TAU_1H_200; n_pred_tau += p > TAU_1H_200
    err, d_rh = np.array(err), np.array(d_rh)
    return dict(mae_points=float(np.mean(np.abs(err))), bias=float(np.mean(err)), mae_datasets=float(np.mean(ds_mae)),
                n_total=n_total, n_points=len(err), n_ok=len(d_rh), n_above=n_above,
                median_d_rh=float(np.median(d_rh)) if len(d_rh) else None,
                median_abs_d_rh=float(np.median(np.abs(d_rh))) if len(d_rh) else None,
                tau_agree=tau_ok / len(err), n_meas_tau=int(n_meas_tau), n_pred_tau=int(n_pred_tau))

RES, CURVES = [], {}
for dimer, model in MODELS.items():
    for tg_src in ("tg_ml", "tg_derieux"):
        tg_d = INFO[dimer][tg_src]
        for mixing in ("g2020", "mole_fraction"):
            for f_d in F_D:
                if f_d == 0.0 and (dimer != "pinyl-diaterpenyl ester" or tg_src != "tg_ml"):
                    continue                                 # f_d = 0 does not depend on the dimer or its Tg
                curves = {}
                for T in TEMPS:
                    e0 = eta0_map(T, tg_d)
                    curves[T] = np.array([np.nan if x is None else log10_eta(model, x, T, e0, mixing)
                                          for x in COMP[(dimer, f_d, T)]])
                key = f"{dimer}|{tg_src}|{mixing}|{f_d}"
                CURVES[key] = {str(T): c for T, c in curves.items()}
                r = dict(dimer=dimer, tg_source=tg_src, tg_dimer=tg_d, mixing=mixing, f_d=f_d, **score(curves))
                RES.append(r)
                print(f"{dimer:24s} {tg_src:10s} Tg_d={tg_d:5.1f} {mixing:13s} f_d={f_d:.1f} | "
                      f"MAE pts {r['mae_points']:.2f} sets {r['mae_datasets']:.2f} bias {r['bias']:+.2f} | "
                      f"reached {r['n_ok']}/{r['n_total']} med dRH {r['median_d_rh']:+.3f} | "
                      f"tau>1h agree {r['tau_agree']:.2f} (pred {r['n_pred_tau']}/meas {r['n_meas_tau']})")
''')

md("""
## 4. Effective weight of the dimer

w_eff = d log10 eta / d log10 eta0_dimer, from a factor-10 change of eta0_dimer, at RH 0.2, 0.5 and 0.8 (294 K).
For G2020 it equals gamma^C x_dimer; for the mole-fraction rule it is x_dimer.
""")

code(r'''
WEFF = []
T = 294.0
for dimer, model in MODELS.items():
    tg_d = INFO[dimer]["tg_ml"]
    for f_d in (0.1, 0.3, 0.5):
        for rh in (0.2, 0.5, 0.8):
            x = COMP[(dimer, f_d, T)][int(round(rh / 0.01)) - 1]
            if x is None:
                continue
            for mixing in ("g2020", "mole_fraction"):
                e0 = eta0_map(T, tg_d)
                y0 = log10_eta(model, x, T, e0, mixing)
                e1 = dict(e0); e1[4] = e0[4] * 10.0
                WEFF.append(dict(dimer=dimer, f_d=f_d, rh=rh, mixing=mixing, x_dimer=float(x[3]),
                                 x_water=float(x[0]), w_eff=log10_eta(model, x, T, e1, mixing) - y0))
for w in WEFF:
    print(f"{w['dimer']:24s} f_d={w['f_d']:.1f} RH={w['rh']:.1f} {w['mixing']:13s} x_water={w['x_water']:.3f} "
          f"x_dimer={w['x_dimer']:.4f} w_eff={w['w_eff']:.4f} ratio={w['w_eff'] / w['x_dimer']:.3f}")
''')

md("""
## 5. Sensitivity: dimer fraction increasing as the particle dries

Accretion products such as esters and hemiacetals form by condensation, so at equilibrium the dimer-to-monomer ratio
scales roughly as 1/a_w. Pre-specified illustration (no fit): mass ratio dimer/monomer = (0.3/0.7)(0.5/a_w), i.e.
f_d = 0.3 at RH 0.5, rising to 0.5 (the upper end of the scan, and of reported oligomer contents) at RH 0.21 and
held there in drier air; f_d = 0.19 at RH 0.9.
""")

code(r'''
PATHS = {}
def comp_at(dimer, f_d, T, rh):
    key = (dimer, round(f_d, 6), T)
    if key not in PATHS:
        PATHS[key] = path_scan(MODELS[dimer], f_d, T)[0]
    w = w_for_aw(MODELS[dimer], rh, f_d, T, PATHS[key])
    return None if w is None else MODELS[dimer].evaluate(mass_vector(w, f_d), T, basis="mass").x

F_CAP = 0.5
def f_d_of(aw):
    odds = (0.3 / 0.7) * (0.5 / aw)
    return min(F_CAP, odds / (1 + odds))

print({rh: round(f_d_of(rh), 3) for rh in (0.05, 0.2, 0.3, 0.5, 0.7, 0.9)})
RES_RH = []
for dimer in DIMERS:
    tg_d = INFO[dimer]["tg_ml"]
    xs = {T: [comp_at(dimer, f_d_of(rh), T, rh) for rh in RH_GRID] for T in TEMPS}
    for mixing in ("g2020", "mole_fraction"):
        curves = {T: np.array([np.nan if x is None else log10_eta(MODELS[dimer], x, T, eta0_map(T, tg_d), mixing)
                               for x in xs[T]]) for T in TEMPS}
        CURVES[f"{dimer}|tg_ml|{mixing}|f_d(RH)"] = {str(T): c for T, c in curves.items()}
        r = dict(dimer=dimer, tg_source="tg_ml", tg_dimer=tg_d, mixing=mixing, f_d="f_d(RH)", **score(curves))
        RES_RH.append(r)
        base = next(q for q in RES if q["dimer"] == dimer and q["tg_source"] == "tg_ml" and q["mixing"] == mixing
                    and q["f_d"] == 0.3)
        print(f"{dimer:24s} {mixing:13s} f_d(RH): MAE pts {r['mae_points']:.2f} sets {r['mae_datasets']:.2f} "
              f"bias {r['bias']:+.2f} reached {r['n_ok']}/{r['n_total']} tau agree {r['tau_agree']:.2f} | "
              f"constant 0.3: MAE pts {base['mae_points']:.2f} sets {base['mae_datasets']:.2f} bias {base['bias']:+.2f}")
''')

md("""
## 6. Diagnostic: dimer fraction that minimizes the MAE of each data set

f_d from 0 to 0.9 in steps of 0.05, ML Tg, both dimers and both mixing rules. This is an inversion on the data
(not a validation); it shows whether the data sets imply similar or different oligomer contents.
""")

code(r'''
F_GRID = np.round(np.arange(0.0, 0.901, 0.05), 2)
INV = []
for dimer in DIMERS:
    tg_d = INFO[dimer]["tg_ml"]
    for mixing in ("g2020", "mole_fraction"):
        for study, (T, pts) in DATA.items():
            e0 = eta0_map(T, tg_d)
            maes, biases = [], []
            for f_d in F_GRID:
                pred = np.array([np.nan if (x := comp_at(dimer, f_d, T, rh)) is None
                                 else log10_eta(MODELS[dimer], x, T, e0, mixing) for rh in pts[:, 0]])
                e = pred - pts[:, 1]; v = np.isfinite(e)
                maes.append(float(np.mean(np.abs(e[v]))) if v.any() else np.nan)
                biases.append(float(np.mean(e[v])) if v.any() else np.nan)
            k = int(np.nanargmin(maes))
            INV.append(dict(dimer=dimer, mixing=mixing, study=study, T_K=T, n=len(pts),
                            rh_min=float(pts[:, 0].min()), rh_max=float(pts[:, 0].max()),
                            f_best=float(F_GRID[k]), at_bound=bool(k == len(F_GRID) - 1),
                            mae_best=maes[k], bias_best=biases[k], mae_f0=maes[0], maes=maes))
for r in INV:
    if r["dimer"] == "pinonyl-pinyl ester":
        print(f"{r['mixing']:13s} {r['study']:30s} n={r['n']:2d} RH {r['rh_min']:.2f}-{r['rh_max']:.2f} "
              f"best f_d={r['f_best']:.2f}{' (bound)' if r['at_bound'] else ''} MAE {r['mae_f0']:.2f} -> {r['mae_best']:.2f}")
for dimer in DIMERS:
    for mixing in ("g2020", "mole_fraction"):
        f = [r["f_best"] for r in INV if r["dimer"] == dimer and r["mixing"] == mixing]
        print(f"{dimer:24s} {mixing:13s} best f_d median {np.median(f):.2f}, range {min(f):.2f}-{max(f):.2f}")
''')

code(r'''
def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, np.ndarray)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o
OUT = dict(aiomfac_py_version=aiomfac_py.__version__, compounds=INFO, f_d=F_D, rh_grid=RH_GRID,
           tau_1h_200nm_log10=TAU_1H_200, path_check=MONO, results=RES, w_eff=WEFF, curves=CURVES,
           f_d_of_rh=dict(anchor_rh=0.5, anchor_f_d=0.3, cap=F_CAP, results=RES_RH),
           inversion=dict(f_grid=F_GRID, results=INV))
json.dump(clean(OUT), open("oligomer_surrogate_results.json", "w"), indent=1)
print("wrote oligomer_surrogate_results.json")
''')

nb["cells"] = cells
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "../notebooks/aiomfac_validation_13_oligomer_surrogate.ipynb")
print("wrote ../notebooks/aiomfac_validation_13_oligomer_surrogate.ipynb")

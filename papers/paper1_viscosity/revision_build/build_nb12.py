"""Generate notebooks/aiomfac_validation_12_dynamics_metrics.ipynb: viscosity errors expressed as RH (water-activity)
errors and mixing-time factors, the quantities that matter for particle dynamics.

A viscosity error of Delta log10 eta is a factor 10**Delta in the bulk mixing time (Stokes-Einstein), but because log10
eta rises steeply as RH falls, the same error may correspond to a small shift in the RH at which a given viscosity, or
mixing time, is reached. For every measured point (RH_m, log10 eta_m) we find the RH* at which the predicted curve
reaches log10 eta_m and report Delta RH = RH* - RH_m (negative: the model needs a drier particle to reach the measured
viscosity, i.e. it underpredicts). Points more viscous than the prediction at the driest grid RH are censored
(Delta RH is then at most grid minimum - RH_m).

Data: the 53 single-compound curves (water activity from AIOMFAC at the measured composition, as the curves are given
against mass fraction) and the chamber SOA data of Gervasi et al. (2020) with the proxies of Table 4 (predicted Tg).
Results: dynamics_metrics_results.json.

Run:  python build_nb12.py && (cd ../notebooks && PATH=/opt/miniconda3/bin:$PATH PYTHONPATH=<v1.3.1 src> \
          jupyter nbconvert --to notebook --execute --inplace aiomfac_validation_12_dynamics_metrics.ipynb \
          --ExecutePreprocessor.timeout=-1)
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md("""
# Viscosity errors as RH errors and mixing-time factors

For each measured point, the horizontal distance to the predicted curve: the RH (or water activity) at which the
prediction reaches the measured viscosity, minus the measured RH. Phase-state thresholds: 10^2 Pa s (liquid to
semi-solid) and 10^12 Pa s (glass) (Koop et al., 2011), and the viscosity at which the bulk mixing time of an organic
molecule, tau = d^2 / (4 pi^2 D) with the Stokes-Einstein D = k T / (6 pi a eta), equals 1 h.
""")

code(r'''
import json, math, re, zipfile, warnings
import numpy as np
from scipy.optimize import brentq
import aiomfac_py
from aiomfac_py import ActivityModel, Component
from aiomfac_py.s2as import smiles_to_components
from aiomfac_py.tgml_armeli import predict_tg_ml_smiles
from aiomfac_py.viscosity import organic_mixture_viscosity, pure_organic_viscosity_vtf
from rdkit import Chem
from rdkit.Chem.Descriptors import MolWt
warnings.filterwarnings("ignore")
print("aiomfac_py", aiomfac_py.__version__, aiomfac_py.__file__)

WATER = Component(1, "Water", ((16, 1),))
OUT = dict(aiomfac_py_version=aiomfac_py.__version__)

def organic_model(smiles):
    r = smiles_to_components([smiles], water_as_component1=True)
    assert not r.removed and r.n_unmatched_atoms[0] == 0, smiles
    return ActivityModel([WATER, Component(2, "organic", r.components[1].subgroups)])

def log10_mix(model, x, T, eta0):
    try:
        e = organic_mixture_viscosity(model, np.asarray(x), T, {2: eta0}).eta_pas
        return math.log10(e) if np.isfinite(e) and e > 0 else np.nan
    except (ValueError, OverflowError):
        return np.nan

def eta_for_tau(tau_s, d_m, T, a_m):
    """Viscosity (Pa s) at which the bulk mixing time d^2/(4 pi^2 D), D = kT/(6 pi a eta), equals tau."""
    kB = 1.380649e-23
    return kB * T * 4 * math.pi**2 * tau_s / (6 * math.pi * a_m * d_m**2)

A_M = 0.4e-9                      # hydrodynamic radius of the diffusing organic molecule
THRESH = {"semisolid": 2.0, "glass": 12.0}
for d_nm in (100, 200):
    THRESH[f"tau_1h_{d_nm}nm"] = math.log10(eta_for_tau(3600.0, d_nm * 1e-9, 295.0, A_M))
print({k: round(v, 2) for k, v in THRESH.items()})
OUT["thresholds_log10_pas"] = THRESH
OUT["mixing_time"] = dict(a_m=A_M, T_K=295.0, note="threshold shifts by log10(0.4 nm / a) for another radius")

def horizontal(rh_grid, y_grid, rh_m, y_m):
    """RH* where the predicted curve equals y_m (the crossing nearest rh_m); status 'ok', 'above' (measured more
    viscous than the prediction anywhere on the grid) or 'below'."""
    v = np.isfinite(y_grid)
    r, y = rh_grid[v], y_grid[v]
    d = y - y_m
    idx = np.where(np.sign(d[:-1]) * np.sign(d[1:]) <= 0)[0]
    idx = idx[d[idx] != d[idx + 1]]
    if len(idx) == 0:
        return ("above", None) if y_m > np.nanmax(y) else ("below", None)
    roots = r[idx] - d[idx] * (r[idx + 1] - r[idx]) / (d[idx + 1] - d[idx])
    return "ok", float(roots[np.argmin(np.abs(roots - rh_m))])

def local_slope(rh_grid, y_grid, rh):
    """-d log10 eta / d RH of the predicted curve at rh (per unit RH, i.e. per 100 % RH)."""
    v = np.isfinite(y_grid)
    g = np.gradient(y_grid[v], rh_grid[v])
    return float(-np.interp(rh, rh_grid[v], g))
''')

md("## 1. Single-compound curves (water activity from AIOMFAC at the measured composition)")

code(r'''
curves = json.load(open("literature_curves.json"))
W_GRID = np.unique(np.concatenate([np.linspace(0.0005, 0.9, 300), 1 - np.logspace(-3.3, -1, 200)]))
lit = []
for c in curves:
    model = organic_model(c["smiles"])
    tg = predict_tg_ml_smiles(c["smiles"]).tg_K
    mw = MolWt(Chem.MolFromSmiles(c["smiles"]))
    regime = "concentrated" if np.mean([p["wtf_organic"] for p in c["points"]]) >= 0.5 else "dilute"
    by_T = {}
    for p in c["points"]:
        by_T.setdefault(p["T_K"], []).append(p)
    for T, pts in by_T.items():
        eta0 = pure_organic_viscosity_vtf(T, tg)
        aw_g, y_g = [], []
        for w in W_GRID:
            r = model.evaluate([1 - w, w], T, basis="mass")
            aw_g.append(float(r.activity[0])); y_g.append(log10_mix(model, r.x, T, eta0))
        aw_g, y_g = np.array(aw_g), np.array(y_g)
        o = np.argsort(aw_g); aw_g, y_g = aw_g[o], y_g[o]
        for p in pts:
            r = model.evaluate([1 - p["wtf_organic"], p["wtf_organic"]], T, basis="mass")
            aw_m = p["aw_measured"] if p["aw_measured"] is not None else float(r.activity[0])
            y_m = p["log10_eta_measured"]
            y_p = log10_mix(model, r.x, T, eta0)
            status, aw_star = horizontal(aw_g, y_g, aw_m, y_m)
            lit.append(dict(name=c["name"], source=c["source"], mw=mw, regime=regime, T_K=T, aw=aw_m, y_meas=y_m,
                            y_pred=y_p, status=status, d_aw=None if aw_star is None else aw_star - aw_m,
                            aw_min_grid=float(np.nanmin(aw_g[np.isfinite(y_g)])),
                            slope=local_slope(aw_g, y_g, aw_m)))
OUT["literature_points"] = lit
print(len(lit), "points;", sum(p["status"] != "ok" for p in lit), "without a crossing")
''')

md("## 2. Chamber SOA (proxies of Table 4, predicted Tg)")

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

def wtf_for_aw(model, aw, T, lo=0.0005, hi=0.9995):
    g = lambda w: float(model.evaluate([w, 1 - w], T, basis="mass").activity[0]) - aw
    if g(lo) > 0 or g(hi) < 0:
        return None
    return brentq(g, lo, hi, xtol=1e-7)

PROXIES = {"aPinene": {"pinonic acid": "CC(=O)C1CC(C1(C)C)CC(=O)O", "pinic acid": "CC1(C(CC1C(=O)O)CC(=O)O)C"},
           "isoprene": {"2-methyltetrol": "OCC(C)(O)C(O)CO"},
           "toluene": {"4-methylcatechol": "Cc1ccc(O)c(O)c1", "glutaric acid": "OC(=O)CCCC(=O)O",
                       "2-methylglutaric acid": "OC(=O)C(C)CCC(=O)O", "DHOPA": "CC(=O)C(O)C(O)C(=O)O",
                       "4-methyl-5-nitrocatechol": "Cc1cc(O)c(O)cc1[N+](=O)[O-]"}}
RH_GRID = np.round(np.arange(0.005, 0.9951, 0.005), 3)
soa, curves_out = [], {}
for system, proxies in PROXIES.items():
    files = sorted(n for n in ZIP.namelist() if f"{system}_SOA" in n and n.endswith(".txt"))
    sets = {fn: parse_soa(fn) for fn in files}
    for pname, smi in proxies.items():
        model = organic_model(smi)
        tg = predict_tg_ml_smiles(smi).tg_K
        grids = {}
        for T in sorted({T for T, _ in sets.values()}):
            eta0 = pure_organic_viscosity_vtf(T, tg)
            y = []
            for aw in RH_GRID:
                w = wtf_for_aw(model, aw, T)
                y.append(np.nan if w is None else log10_mix(model, model.evaluate([w, 1 - w], T, basis="mass").x, T, eta0))
            grids[T] = np.array(y)
            curves_out[f"{system}|{pname}|{T}"] = grids[T]
        for fn, (T, pts) in sets.items():
            study = fn.split("_SOA_")[1].rsplit(".txt", 1)[0]
            for rh_m, y_m, yn, yp in pts:
                status, rh_star = horizontal(RH_GRID, grids[T], rh_m, y_m)
                soa.append(dict(system=system, proxy=pname, tg_K=tg, study=study, T_K=T, rh=float(rh_m),
                                y_meas=float(y_m), lower_bound=bool(min(yn, yp) < 0.01),
                                y_pred=float(np.interp(rh_m, RH_GRID, grids[T])), status=status,
                                d_rh=None if rh_star is None else rh_star - float(rh_m),
                                slope=local_slope(RH_GRID, grids[T], rh_m)))
OUT["soa_points"] = soa
OUT["rh_grid"] = RH_GRID
OUT["soa_curves"] = curves_out
print(len(soa), "proxy-point pairs")
''')

md("## 3. Summaries")

code(r'''
def summary(rows, key):
    ok = [r for r in rows if r["status"] == "ok"]
    cen = [r for r in rows if r["status"] == "above"]
    d = np.array([r[key] for r in ok])
    err = np.array([r["y_pred"] - r["y_meas"] for r in rows if np.isfinite(r["y_pred"])])
    return dict(n=len(rows), n_ok=len(ok), n_above=len(cen), n_below=len(rows) - len(ok) - len(cen),
                median_d=float(np.median(d)) if len(d) else None,
                median_abs_d=float(np.median(np.abs(d))) if len(d) else None,
                p90_abs_d=float(np.percentile(np.abs(d), 90)) if len(d) else None,
                mae_log10=float(np.mean(np.abs(err))) if len(err) else None,
                median_tau_factor=float(10 ** np.median(np.abs(err))) if len(err) else None,
                median_slope=float(np.median([r["slope"] for r in ok])) if ok else None)

S = {}
bands = {"all": lambda y: True, "log10 eta >= 2": lambda y: y >= 2.0, "log10 eta < 2": lambda y: y < 2.0}
for regime in ("dilute", "concentrated"):
    for size, sel in (("M<342", lambda m: m < 341.5), ("M>=342", lambda m: m >= 341.5)):
        for band, f in bands.items():
            rows = [p for p in lit if p["regime"] == regime and sel(p["mw"]) and f(p["y_meas"])]
            if rows:
                S[f"literature|{regime}|{size}|{band}"] = summary(rows, "d_aw")
for system, proxies in PROXIES.items():
    for pname in proxies:
        for band, f in bands.items():
            rows = [p for p in soa if p["system"] == system and p["proxy"] == pname and f(p["y_meas"])]
            if rows:
                S[f"soa|{system}|{pname}|{band}"] = summary(rows, "d_rh")
OUT["summary"] = S
for k, v in S.items():
    if v["n_ok"]:
        print(f"{k:55s} n={v['n']:3d} ok={v['n_ok']:3d} above={v['n_above']:2d} "
              f"median dRH={v['median_d']:+.3f} |dRH| med={v['median_abs_d']:.3f} p90={v['p90_abs_d']:.3f} "
              f"MAE={v['mae_log10']:.2f} slope={v['median_slope']:.1f}/unit RH")
    else:
        print(f"{k:55s} n={v['n']:3d} ok=0 above={v['n_above']}")
''')

md("""
## 4. Phase-state class and the 1 h mixing-time threshold

Agreement of the predicted and measured class at each point: liquid (< 10^2 Pa s), semi-solid, glass (> 10^12 Pa s),
and whether the mixing time of a 200 nm particle exceeds 1 h.
""")

code(r'''
def cls(y):
    return "liquid" if y < THRESH["semisolid"] else ("glass" if y > THRESH["glass"] else "semi-solid")

agree = {}
for label, rows in (("literature (concentrated)", [p for p in lit if p["regime"] == "concentrated"]),
                    ("SOA alpha-pinene (pinonic acid)", [p for p in soa if p["proxy"] == "pinonic acid"]),
                    ("SOA alpha-pinene (pinic acid)", [p for p in soa if p["proxy"] == "pinic acid"]),
                    ("SOA isoprene (2-methyltetrol)", [p for p in soa if p["proxy"] == "2-methyltetrol"]),
                    ("SOA toluene (DHOPA)", [p for p in soa if p["proxy"] == "DHOPA"])):
    rows = [r for r in rows if np.isfinite(r["y_pred"])]
    t = THRESH["tau_1h_200nm"]
    agree[label] = dict(n=len(rows),
                        phase_class=float(np.mean([cls(r["y_pred"]) == cls(r["y_meas"]) for r in rows])),
                        tau_1h_200nm=float(np.mean([(r["y_pred"] > t) == (r["y_meas"] > t) for r in rows])),
                        n_meas_above_tau=int(sum(r["y_meas"] > t for r in rows)),
                        n_pred_above_tau=int(sum(r["y_pred"] > t for r in rows)))
    print(f"{label:34s} n={len(rows):3d} class agree {agree[label]['phase_class']:.2f}  "
          f"tau>1h agree {agree[label]['tau_1h_200nm']:.2f} (measured above: {agree[label]['n_meas_above_tau']}, "
          f"predicted above: {agree[label]['n_pred_above_tau']})")
OUT["class_agreement"] = agree
''')

code(r'''
def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, np.ndarray)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o
json.dump(clean(OUT), open("dynamics_metrics_results.json", "w"), indent=1)
print("wrote dynamics_metrics_results.json")
''')

nb["cells"] = cells
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "../notebooks/aiomfac_validation_12_dynamics_metrics.ipynb")
print("wrote ../notebooks/aiomfac_validation_12_dynamics_metrics.ipynb")

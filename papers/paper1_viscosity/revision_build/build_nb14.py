"""Generate notebooks/aiomfac_validation_14_ambient_oligomers.ipynb: the monomer-dimer surrogate of notebook 13
applied to the ambient PM2.5 viscosities of Ansan, Seoul and Beijing.

Where the organic material forms its own liquid (Sect. 3.4), that liquid contains little salt, and notebook 05 bounds
it by an organic-only liquid (salt effloresced or in the other phase). Here that organic-only liquid is the
monomer-dimer surrogate (pinonic and pinic acid in equal mass plus C17H26O8 or C19H28O7, ML Tg) at the site
temperature of notebook 05 and the RH of every measured point. Reported per site: the offset measured minus predicted
(mean, sd and median over resolved points, points within one order of magnitude, shortfall at lower bounds) for
f_d = 0-0.5, for the RH-dependent f_d of notebook 13, and the f_d at which the prediction at the median RH equals the
median measured viscosity.

The measured values are not written to the result file (see ../ambient_data/README.md); only offsets and counts are.
Run:  python build_nb14.py && (cd ../notebooks && PATH=/opt/miniconda3/bin:$PATH PYTHONPATH=<v1.3.1 src> \
          AMBIENT_XLSX=<data file> jupyter nbconvert --to notebook --execute --inplace \
          aiomfac_validation_14_ambient_oligomers.ipynb --ExecutePreprocessor.timeout=-1)
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md("""
# Ambient PM2.5 with the monomer-dimer surrogate

Organic-only liquid (the organic-rich phase of a phase-separated particle, salt-free approximation) at the site
temperature and the measured RH. Only offsets and counts are printed and saved.
""")

code(r'''
import json, math, sys, warnings
import numpy as np
from scipy.optimize import brentq
import aiomfac_py
from aiomfac_py import ActivityModel, Component
from aiomfac_py.s2as import smiles_to_components
from aiomfac_py.tgml_armeli import predict_tg_ml_smiles
from aiomfac_py.viscosity import organic_mixture_viscosity, pure_organic_viscosity_vtf
warnings.filterwarnings("ignore")
print("aiomfac_py", aiomfac_py.__version__, aiomfac_py.__file__)

sys.path.insert(0, "../ambient_data")
from ambient_data import load
PTS = load()
T_SITE = {"Ansan": 300.0, "Seoul": 290.0, "Beijing": 290.0}      # as in notebook 05

WATER = Component(1, "Water", ((16, 1),))
MONOMERS = {"pinonic acid": "CC(=O)C1CC(C1(C)C)CC(=O)O", "pinic acid": "CC1(C(CC1C(=O)O)CC(=O)O)C"}
DIMERS = {"pinyl-diaterpenyl ester": "CC1(C)C(CC1C(=O)O)CC(=O)OC(C)(C)C(CC(=O)O)CC(=O)O",
          "pinonyl-pinyl ester": "CC1(C)C(CC1C(=O)O)CC(=O)OCC(=O)C1CC(CC(=O)O)C1(C)C"}
TG = {n: predict_tg_ml_smiles(s).tg_K for n, s in {**MONOMERS, **DIMERS}.items()}
print({k: round(v, 1) for k, v in TG.items()})

def build_model(dimer):
    r = smiles_to_components(list(MONOMERS.values()) + [DIMERS[dimer]], water_as_component1=True)
    assert not r.removed and all(n == 0 for n in r.n_unmatched_atoms)
    names = list(MONOMERS) + [dimer]
    return ActivityModel([WATER] + [Component(k + 2, n, c.subgroups) for k, (n, c) in enumerate(zip(names, r.components[1:]))])
MODELS = {d: build_model(d) for d in DIMERS}

def mass_vector(w, f_d):
    return np.concatenate([[w], (1 - w) * np.array([(1 - f_d) / 2, (1 - f_d) / 2, f_d])])

W_PATH = np.concatenate([np.geomspace(1e-5, 0.05, 120), np.linspace(0.05, 0.9995, 400)[1:]])
PATHS = {}
def comp_at(dimer, f_d, T, rh):
    """Organic-rich branch: first crossing of a_w = rh from the dry end (as in notebook 13)."""
    m = MODELS[dimer]
    aw = lambda w: float(m.evaluate(mass_vector(w, f_d), T, basis="mass").activity[0])
    key = (dimer, round(f_d, 6), T)
    if key not in PATHS:
        PATHS[key] = np.array([aw(w) for w in W_PATH])
    i = np.where(PATHS[key] >= rh)[0]
    if len(i) == 0 or i[0] == 0:
        return None
    k = i[0]
    w = brentq(lambda w: aw(w) - rh, W_PATH[k - 1], W_PATH[k], xtol=1e-9)
    return m.evaluate(mass_vector(w, f_d), T, basis="mass").x

def log10_eta(dimer, x, T, mixing):
    e0 = {2: pure_organic_viscosity_vtf(T, TG["pinonic acid"]), 3: pure_organic_viscosity_vtf(T, TG["pinic acid"]),
          4: pure_organic_viscosity_vtf(T, TG[dimer])}
    try:
        e = organic_mixture_viscosity(MODELS[dimer], np.asarray(x), T, e0, mixing=mixing).eta_pas
        return math.log10(e) if np.isfinite(e) and e > 0 else np.nan
    except (ValueError, OverflowError):
        return np.nan

def f_d_of(aw):                     # notebook 13, Sect. 5
    odds = (0.3 / 0.7) * (0.5 / aw)
    return min(0.5, odds / (1 + odds))

def score(pred, pts):
    """As notebook 05: offsets measured minus predicted; lower bounds count only their shortfall."""
    y = np.array([p["log10_eta"] for p in pts]); cen = np.array([p["censored"] for p in pts])
    off = y - pred
    res = off[~cen]
    short = np.maximum(0.0, off[cen])
    return dict(n_resolved=int((~cen).sum()), mean_offset=float(np.nanmean(res)), sd_offset=float(np.nanstd(res)),
                median_offset=float(np.nanmedian(res)), n_within_1=int(np.sum(np.abs(res) <= 1.0)),
                n_censored=int(cen.sum()), censored_shortfall=float(np.nanmean(short)) if cen.any() else None)
''')

md("## 1. Offsets for a fixed and an RH-dependent dimer fraction")

code(r'''
F_D = (0.0, 0.1, 0.2, 0.3, 0.5)
RES = []
for site in ("Ansan", "Seoul", "Beijing"):
    pts = [p for p in PTS if p["site"] == site]; T = T_SITE[site]
    for dimer in DIMERS:
        for mixing in ("g2020", "mole_fraction"):
            for f_d in list(F_D) + ["f_d(RH)"]:
                if f_d == 0.0 and dimer != "pinyl-diaterpenyl ester":
                    continue
                pred = []
                for p in pts:
                    rh = p["rh_pct"] / 100
                    x = comp_at(dimer, f_d_of(rh) if f_d == "f_d(RH)" else f_d, T, rh)
                    pred.append(np.nan if x is None else log10_eta(dimer, x, T, mixing))
                r = dict(site=site, dimer=dimer, mixing=mixing, f_d=f_d, **score(np.array(pred), pts))
                RES.append(r)
for r in RES:
    fd = r["f_d"] if isinstance(r["f_d"], str) else f"{r['f_d']:.1f}"
    cs = "-" if r["censored_shortfall"] is None else f"{r['censored_shortfall']:.2f}"
    print(f"{r['site']:8s} {r['dimer']:24s} {r['mixing']:13s} f_d={fd:8s} offset {r['mean_offset']:+5.2f} "
          f"± {r['sd_offset']:.2f} (median {r['median_offset']:+5.2f}) within 1: {r['n_within_1']}/{r['n_resolved']} "
          f"shortfall at lower bounds {cs} (n={r['n_censored']})")
''')

md("""
## 2. Dimer fraction that matches the median measured viscosity at the median RH of each site

Root finding on f_d in [0, 0.95] (inversion, as the required Tg of notebook 05).
""")

code(r'''
REQ = []
for site in ("Ansan", "Seoul", "Beijing"):
    res_pts = [p for p in PTS if p["site"] == site and not p["censored"]]
    rh_med = float(np.median([p["rh_pct"] for p in res_pts])) / 100
    y_med = float(np.median([p["log10_eta"] for p in res_pts])); T = T_SITE[site]
    for dimer in DIMERS:
        for mixing in ("g2020", "mole_fraction"):
            f = lambda fd: log10_eta(dimer, comp_at(dimer, fd, T, rh_med), T, mixing) - y_med
            try:
                fd_req = float(brentq(f, 0.0, 0.95, xtol=1e-3))
            except ValueError:
                fd_req = None
            REQ.append(dict(site=site, dimer=dimer, mixing=mixing, rh_median=rh_med, f_d_required=fd_req))
            print(f"{site:8s} {dimer:24s} {mixing:13s} median RH {rh_med:.2f}: f_d required "
                  f"{'> 0.95 or < 0' if fd_req is None else f'{fd_req:.2f}'}")
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
    return o
json.dump(clean(dict(aiomfac_py_version=aiomfac_py.__version__, t_site=T_SITE, tg=TG, results=RES, f_d_required=REQ,
                     note="offsets = measured minus predicted; measured values not stored")),
          open("ambient_oligomer_results.json", "w"), indent=1)
print("wrote ambient_oligomer_results.json")
''')

nb["cells"] = cells
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "../notebooks/aiomfac_validation_14_ambient_oligomers.ipynb")
print("wrote ../notebooks/aiomfac_validation_14_ambient_oligomers.ipynb")

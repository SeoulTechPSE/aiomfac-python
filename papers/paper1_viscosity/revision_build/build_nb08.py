"""Generate notebooks/aiomfac_validation_08_parameter_sensitivity.ipynb.

One-at-a-time sensitivity of the pipeline to the AIOMFAC short-range interaction parameters a_mn, replacing the
citation of the Sobol analysis of the companion surrogate study. Every directed main-group pair (m, n) that occurs in
a molecule is scaled by 0.9 and 1.1 in turn (all subgroup entries of parA belonging to that pair), and the change of
the predicted log10 viscosity is recorded for
  * the 53 literature curves (composition given as mass fraction, so only the residual viscosity term changes), and
  * the chamber-SOA proxy curves (water activity = RH; the parameters change the water content and the residual term).

Run:  python build_nb08.py && (cd ../notebooks && PYTHONPATH=<aiomfac-python main>/src jupyter nbconvert --to notebook \
          --execute --inplace aiomfac_validation_08_parameter_sensitivity.ipynb --ExecutePreprocessor.timeout=-1)
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md("""
# Sensitivity of the pipeline to the AIOMFAC interaction parameters

AIOMFAC does not report uncertainties for its group-interaction parameters a_mn, so this is a sensitivity analysis,
not an uncertainty propagation: each directed main-group pair (m, n) present in a molecule is scaled by 0.9 and 1.1
in turn, and the change of the predicted log10 viscosity is recorded.

* Literature curves (53): the organic mass fraction is given, so a_mn changes only the residual term of G2020.
* Chamber-SOA proxies (pinonic acid, pinic acid, 2-methyltetrol, DHOPA) at the measured RH: a_mn also changes the
  water content through the water activity.

Main groups are labelled by the subgroups they contain in these molecules. Results: `parameter_sensitivity_results.json`.
""")

code(r'''
import json, math, re, zipfile, warnings
import numpy as np
from scipy.optimize import brentq
import aiomfac_py
from aiomfac_py import ActivityModel, Component
from aiomfac_py.params import load_subgroup_params
from aiomfac_py.s2as import smiles_to_components
from aiomfac_py.tgml_armeli import predict_tg_ml_smiles
from aiomfac_py.viscosity import organic_mixture_viscosity, pure_organic_viscosity_vtf
warnings.filterwarnings("ignore")
print("aiomfac_py", aiomfac_py.__version__, aiomfac_py.__file__)

SG = load_subgroup_params()
SUBNAME = {int(k): v for k, v in json.load(open("../revision_build/subgroup_names.json")).items()}
WATER = Component(1, "Water", ((16, 1),))
FACTORS = (0.9, 1.1)
curves = json.load(open("literature_curves.json"))
ZIP = zipfile.ZipFile("gervasi_supp.zip")

def organic_model(smiles):
    r = smiles_to_components([smiles], water_as_component1=True)
    assert not r.removed and r.n_unmatched_atoms[0] == 0, smiles
    return ActivityModel([WATER, Component(2, "organic", r.components[1].subgroups)])

def directed_pairs(model):
    """{(m, n): [(l, j), ...]} -- parA entries of each directed main-group pair m -> n (m != n)."""
    sr = model.mixture.sr
    mg = [SG.main_group(s) for s in sr.solv_subs]
    out = {}
    for l in range(len(mg)):
        for j in range(len(mg)):
            if mg[l] != mg[j]:
                out.setdefault((mg[l], mg[j]), []).append((l, j))
    return out

def mg_label(model):
    sr = model.mixture.sr
    lab = {}
    for s in sr.solv_subs:
        lab.setdefault(SG.main_group(s), set()).add(SUBNAME.get(s, f"sg{s}").strip("()"))
    return {m: "/".join(sorted(v)) for m, v in lab.items()}

def log10_mix(model, x, T, eta0):
    try:
        e = organic_mixture_viscosity(model, np.asarray(x), T, {2: eta0}).eta_pas
        return math.log10(e) if np.isfinite(e) and e > 0 else np.nan
    except (ValueError, OverflowError):
        return np.nan

class scaled:
    """Context manager: scale the parA entries of one directed main-group pair in place."""
    def __init__(self, model, entries, f):
        self.A, self.entries, self.f = model.mixture.sr.parA, entries, f
    def __enter__(self):
        self.saved = [self.A[l, j] for l, j in self.entries]
        for l, j in self.entries:
            self.A[l, j] *= self.f
    def __exit__(self, *a):
        for (l, j), v in zip(self.entries, self.saved):
            self.A[l, j] = v
''')

md("## 1. Literature curves")

code(r'''
LABELS = {}
lit = []          # one record per curve and directed pair: mean |Δlog10 η| over points (max over the two factors)
base_mae = {}
for ci, c in enumerate(curves):
    model = organic_model(c["smiles"])
    LABELS.update({m: v for m, v in mg_label(model).items()})
    tg = predict_tg_ml_smiles(c["smiles"]).tg_K
    pts = c["points"]
    T = [p["T_K"] for p in pts]
    xs = [model.evaluate([1 - p["wtf_organic"], p["wtf_organic"]], p["T_K"], basis="mass").x for p in pts]
    y = np.array([p["log10_eta_measured"] for p in pts])
    eta0 = [pure_organic_viscosity_vtf(t, tg) for t in T]
    pred0 = np.array([log10_mix(model, x, t, e) for x, t, e in zip(xs, T, eta0)])
    wtf_mean = float(np.mean([p["wtf_organic"] for p in pts]))
    regime = "concentrated" if wtf_mean >= 0.5 else "dilute"
    base_mae[ci] = float(np.nanmean(np.abs(pred0 - y)))
    for (m, n), entries in directed_pairs(model).items():
        d, dmae = [], []
        for f in FACTORS:
            with scaled(model, entries, f):
                p = np.array([log10_mix(model, x, t, e) for x, t, e in zip(xs, T, eta0)])
            d.append(np.nanmean(np.abs(p - pred0)))
            dmae.append(float(np.nanmean(np.abs(p - y))) - base_mae[ci])
        lit.append(dict(curve=ci, name=c["name"], source=c["source"], regime=regime, m=m, n=n,
                        mean_abs_dlog10=float(max(d)), dmae_minus=dmae[0], dmae_plus=dmae[1]))
print(len(lit), "curve-pair records")
''')

code(r'''
def table(records, key="mean_abs_dlog10", group=("m", "n")):
    agg = {}
    for r in records:
        agg.setdefault(tuple(r[g] for g in group), []).append(r[key])
    return sorted(((k, float(np.mean(v)), float(np.max(v)), len(v)) for k, v in agg.items()), key=lambda t: -t[1])

lit_summary = {}
for regime in ("dilute", "concentrated"):
    rows = table([r for r in lit if r["regime"] == regime])
    lit_summary[regime] = [dict(m=m, n=n, label=f"{LABELS.get(m, m)} -> {LABELS.get(n, n)}", mean=mu, max=mx,
                                n_curves=k) for (m, n), mu, mx, k in rows]
    print(f"--- {regime} curves: mean |Δlog10 η| for a ±10 % change of a_mn (top 10)")
    for r in lit_summary[regime][:10]:
        print(f"  {r['label']:45s} mean {r['mean']:.3f}  max {r['max']:.3f}  ({r['n_curves']} curves)")
''')

md("## 2. Chamber-SOA proxies (water content from a_w = RH)")

code(r'''
def parse_soa(name):
    txt = ZIP.read(name).decode("utf-8-sig")
    lines = [l.strip().strip('"') for l in txt.splitlines()]
    T = float(re.search(r"T\s*=\s*([\d.]+)\s*K", lines[0]).group(1))
    pts = []
    for line in lines[4:]:
        p = [s for s in re.split(r"[,\t]", line) if s.strip()]
        if len(p) >= 6:
            rh, eta = float(p[0]), float(p[1])
            if eta > 0 and 0 < rh < 100:
                pts.append((rh / 100.0, math.log10(eta)))
    return T, np.array(pts)

def wtf_for_aw(model, aw, T, lo=0.0005, hi=0.9995):
    g = lambda w: float(model.evaluate([w, 1 - w], T, basis="mass").activity[0]) - aw
    if g(lo) > 0 or g(hi) < 0:
        return None
    return brentq(g, lo, hi, xtol=1e-7)

def soa_curve(model, eta0, T, aws):
    out = []
    for aw in aws:
        w = wtf_for_aw(model, aw, T)
        out.append(np.nan if w is None else log10_mix(model, model.evaluate([w, 1 - w], T, basis="mass").x, T, eta0))
    return np.array(out)

PROXIES = {"aPinene": {"pinonic acid": "CC(=O)C1CC(C1(C)C)CC(=O)O", "pinic acid": "CC1(C(CC1C(=O)O)CC(=O)O)C"},
           "isoprene": {"2-methyltetrol": "OCC(C)(O)C(O)CO"},
           "toluene": {"DHOPA": "CC(=O)C(O)C(O)C(=O)O"}}
soa = []
for system, proxies in PROXIES.items():
    files = sorted(n for n in ZIP.namelist() if f"{system}_SOA" in n and n.endswith(".txt"))
    for pname, smi in proxies.items():
        model = organic_model(smi)
        LABELS.update(mg_label(model))
        tg = predict_tg_ml_smiles(smi).tg_K
        sets = [parse_soa(fn) for fn in files]
        base = [soa_curve(model, pure_organic_viscosity_vtf(T, tg), T, pts[:, 0]) for T, pts in sets]
        for (m, n), entries in directed_pairs(model).items():
            d = []
            for f in FACTORS:
                with scaled(model, entries, f):
                    pert = [soa_curve(model, pure_organic_viscosity_vtf(T, tg), T, pts[:, 0]) for T, pts in sets]
                d.append(np.nanmean(np.abs(np.concatenate(pert) - np.concatenate(base))))
            soa.append(dict(system=system, proxy=pname, m=m, n=n, label=f"{LABELS.get(m, m)} -> {LABELS.get(n, n)}",
                            mean_abs_dlog10=float(max(d))))
for system, proxies in PROXIES.items():
    for pname in proxies:
        sel = sorted([r for r in soa if r["proxy"] == pname], key=lambda r: -r["mean_abs_dlog10"])
        print(f"--- {system} / {pname}: mean |Δlog10 η| for a ±10 % change of a_mn (top 5)")
        for r in sel[:5]:
            print(f"  {r['label']:45s} {r['mean_abs_dlog10']:.3f}")
''')

code(r'''
def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    return o
OUT = dict(aiomfac_py_version=aiomfac_py.__version__, factors=FACTORS, labels=LABELS,
           literature_by_curve=lit, literature_summary=lit_summary, soa=soa)
json.dump(clean(OUT), open("parameter_sensitivity_results.json", "w"), indent=1)
print("wrote parameter_sensitivity_results.json")
''')

nb["cells"] = cells
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "../notebooks/aiomfac_validation_08_parameter_sensitivity.ipynb")
print("wrote ../notebooks/aiomfac_validation_08_parameter_sensitivity.ipynb")

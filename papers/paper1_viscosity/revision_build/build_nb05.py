"""Generate notebooks/aiomfac_validation_05_ambient_pointwise.ipynb (point-wise ambient comparison, single-phase rules).

Run:  python build_nb05.py && jupyter nbconvert --to notebook --execute --inplace \
          ../notebooks/aiomfac_validation_05_ambient_pointwise.ipynb --ExecutePreprocessor.timeout=-1
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md("""
# Ambient PM2.5: point-wise comparison with the data of Fig. 2a of Do et al. (2026a)

Data: the values behind Fig. 2a of Do, Lee and Song (2026a, Environ. Sci. Technol., 60, 24251-24259), provided by the
authors (`../ambient_data/ambient_data.py`; the data file is not redistributed). Ansan (300 K): mean of three particles
of one sample at one RH, with sample date and OIR. Seoul and Beijing (290 K, Ullah et al., 2026): one value per
particle, as plotted in Fig. 2a. Values >= 1e8 Pa s (no flow within 3 h) are lower bounds.

Set-up for each site: organic proxy (pinonic or pinic acid, predicted Tg) plus an inorganic fraction at the
site's mean, minimum and maximum OIR (Do et al., 2026, Table 1). Inorganic composition:
* `AS`: ammonium sulfate only (as in notebook 03);
* `AS+AN`: ammonium sulfate and ammonium nitrate in the site-mean SO4:NO3 ratio of Table 1 (minor ions ignored).
Rules: aquelec and aquorg (Lilek and Zuend, 2022). Bound: AS effloresced, i.e. an organic-only liquid
(binary proxy-water, G2020). Finally, the organic Tg that would make the aquelec prediction match the median
measured viscosity is found by root finding on Tg.
""")

code(r'''
import json, math, sys, warnings
import numpy as np
from scipy.optimize import brentq
import aiomfac_py
from aiomfac_py import ActivityModel, Component
from aiomfac_py.s2as import smiles_to_components
from aiomfac_py.tgml_armeli import predict_tg_ml_smiles
from aiomfac_py.viscosity import (pure_organic_viscosity_vtf, aquelec_viscosity, aquorg_viscosity,
                                  organic_mixture_viscosity)
warnings.filterwarnings("ignore")
print("aiomfac_py", aiomfac_py.__version__)

sys.path.insert(0, "../ambient_data")
from ambient_data import load
PTS = load()

# Do et al. (2026) Table 1, ug m-3: (SO4, NO3, NH4, minor, OM, OIR) per sample
TABLE1 = {
    "Ansan": [(2.92, .71, 1.14, .18, 7.19, 1.43), (6.54, 1.43, 2.54, .25, 9.98, .92), (3.29, .70, 1.17, .16, 5.08, .95),
              (4.50, 2.22, 2.20, .22, 12.4, 1.33), (5.40, 1.38, 2.27, .20, 8.50, .91), (4.46, 1.87, 2.36, .58, 6.38, .68),
              (3.02, .89, 1.45, .04, 9.13, 1.67), (3.79, 1.48, 1.80, .08, 9.50, 1.30), (7.09, 2.16, 3.11, .14, 10.8, .85),
              (2.13, .29, .66, .15, 4.77, 1.45)],
    "Seoul": [(2.10, .60, .90, .30, 12.9, 3.26), (1.90, 1.00, 1.00, .20, 12.3, 2.99), (3.20, 2.20, 1.60, .50, 15.5, 2.09)],
    "Beijing": [(1.37, .30, .30, .10, 14.6, 7.63), (2.50, 3.70, 1.50, .70, 23.1, 2.69), (1.00, .70, .30, .30, 4.70, 2.79)],
}
T_SITE = {"Ansan": 300.0, "Seoul": 290.0, "Beijing": 290.0}
SITE = {}
for s, rows in TABLE1.items():
    a = np.array(rows)
    so4, no3 = a[:, 0].mean(), a[:, 1].mean()
    as_mass, an_mass = so4 * 132.14 / 96.06, no3 * 80.04 / 62.00
    SITE[s] = dict(oir_mean=float(a[:, 5].mean()), oir_min=float(a[:, 5].min()), oir_max=float(a[:, 5].max()),
                   an_frac_of_salt=float(an_mass / (as_mass + an_mass)),
                   nh4_measured=float(a[:, 2].mean()), nh4_needed=float(so4 * 2 * 18.04 / 96.06 + no3 * 18.04 / 62.0))
    print(s, {k: round(v, 3) for k, v in SITE[s].items()},
          "resolved points:", sum(1 for r in PTS if r["site"] == s and not r["censored"]),
          "censored:", sum(1 for r in PTS if r["site"] == s and r["censored"]))
''')

code(r'''
WATER = Component(1, "Water", ((16, 1),))
AS = Component(3, "(NH4)2SO4", ((204, 2), (261, 1)))
AN = Component(4, "NH4NO3", ((204, 1), (245, 1)))
PROXIES = {"pinonic acid": "CC(=O)C1CC(C1(C)C)CC(=O)O", "pinic acid": "CC1(C(CC1C(=O)O)CC(=O)O)C"}

def build(smiles, with_an):
    org = Component(2, "org", smiles_to_components([smiles], water_as_component1=True).components[1].subgroups)
    return ActivityModel([WATER, org, AS, AN] if with_an else [WATER, org, AS])

def state_at(model, aw, T, f_org, an_frac, with_an):
    def fr(w):
        dry = 1 - w
        salt = dry * (1 - f_org)
        return [w, dry * f_org, salt * (1 - an_frac), salt * an_frac] if with_an else [w, dry * f_org, salt]
    g = lambda w: float(model.evaluate(fr(w), T, basis="mass").activity[0]) - aw
    lo, hi = 0.0005, 0.9995
    if g(lo) > 0 or g(hi) < 0:
        return None
    return model.evaluate(fr(brentq(g, lo, hi, xtol=1e-7)), T, basis="mass")

def predict(smiles, tg, T, rh_list, oir, rule, with_an, an_frac):
    model = build(smiles, with_an); eta0 = pure_organic_viscosity_vtf(T, tg)
    fn = aquelec_viscosity if rule == "aquelec" else aquorg_viscosity
    out = []
    for rh in rh_list:
        res = state_at(model, rh / 100, T, oir / (oir + 1), an_frac, with_an)
        out.append(np.nan if res is None else math.log10(fn(model, res, T, {2: eta0}).eta_pas))
    return np.array(out)

def predict_org_only(smiles, tg, T, rh_list):
    model = ActivityModel([WATER, Component(2, "org", smiles_to_components([smiles], water_as_component1=True).components[1].subgroups)])
    eta0 = pure_organic_viscosity_vtf(T, tg); out = []
    for rh in rh_list:
        g = lambda w: float(model.evaluate([w, 1 - w], T, basis="mass").activity[0]) - rh / 100
        w = brentq(g, 0.0005, 0.9995, xtol=1e-7)
        out.append(organic_mixture_viscosity(model, model.evaluate([w, 1 - w], T, basis="mass").x, T, {2: eta0}).log10_eta_pas)
    return np.array(out)

def score(pred, pts):
    y = np.array([p["log10_eta"] for p in pts]); cen = np.array([p["censored"] for p in pts])
    off = y - pred                                   # > 0: underprediction
    off_c = np.where(cen, np.maximum(0.0, y - pred), off)   # lower bounds: only shortfall counts
    res = off_c[~cen]
    return dict(mean_offset=float(np.nanmean(res)), sd_offset=float(np.nanstd(res)), median_offset=float(np.nanmedian(res)),
                n_resolved=int((~cen).sum()), n_within_1=int(np.sum(np.abs(res) <= 1.0)),
                censored_shortfall=float(np.nanmean(off_c[cen])) if cen.any() else None, n_censored=int(cen.sum()))

RESULTS = []
TG = {p: predict_tg_ml_smiles(s).tg_K for p, s in PROXIES.items()}
print("predicted proxy Tg:", {k: round(v, 1) for k, v in TG.items()})
for site in ("Ansan", "Seoul", "Beijing"):
    pts = [p for p in PTS if p["site"] == site]; rh = [p["rh_pct"] for p in pts]; T = T_SITE[site]; S = SITE[site]
    for pname, smi in PROXIES.items():
        for with_an in (False, True):
            for oir_label in ("oir_min", "oir_mean", "oir_max"):
                for rule in ("aquelec", "aquorg"):
                    pred = predict(smi, TG[pname], T, rh, S[oir_label], rule, with_an, S["an_frac_of_salt"])
                    r = dict(site=site, proxy=pname, inorganic="AS+AN" if with_an else "AS", oir_label=oir_label,
                             oir=S[oir_label], rule=rule, **score(pred, pts))
                    RESULTS.append(r)
        pred = predict_org_only(smi, TG[pname], T, rh)
        RESULTS.append(dict(site=site, proxy=pname, inorganic="none (AS effloresced)", oir_label="-", oir=None,
                            rule="G2020 binary", **score(pred, pts)))

print(f"{'site':8s} {'proxy':13s} {'inorg':22s} {'OIR':>6s} {'rule':12s} {'mean±sd offset':>16s} {'within1':>8s} {'cens.short':>10s}")
for r in RESULTS:
    if r["proxy"] == "pinic acid":
        oir = f"{r['oir']:.2f}" if r["oir"] else "-"
        cs = f"{r['censored_shortfall']:.2f}" if r["censored_shortfall"] is not None else "-"
        print(f"{r['site']:8s} {r['proxy']:13s} {r['inorganic']:22s} {oir:>6s} {r['rule']:12s} "
              f"{r['mean_offset']:+6.2f} ± {r['sd_offset']:.2f}  {r['n_within_1']:3d}/{r['n_resolved']:<3d} {cs:>10s}")
''')

md("""
## Organic Tg needed to match the data

For each site, aquelec at the mean OIR with `AS+AN`, the pinic-acid structure and a variable Tg: the Tg at which
the prediction equals the median measured log10 eta at the median RH of the resolved points. Repeated for the
organic-only (AS effloresced) case.
""")

code(r'''
TG_REQ = []
for site in ("Ansan", "Seoul", "Beijing"):
    pts = [p for p in PTS if p["site"] == site and not p["censored"]]
    rh_med = float(np.median([p["rh_pct"] for p in pts])); y_med = float(np.median([p["log10_eta"] for p in pts]))
    T = T_SITE[site]; S = SITE[site]; smi = PROXIES["pinic acid"]
    f_mix = lambda tg: predict(smi, tg, T, [rh_med], S["oir_mean"], "aquelec", True, S["an_frac_of_salt"])[0] - y_med
    f_org = lambda tg: predict_org_only(smi, tg, T, [rh_med])[0] - y_med
    out = dict(site=site, rh_median=rh_med, log10_eta_median=y_med)
    for key, f in (("tg_required_aquelec", f_mix), ("tg_required_org_only", f_org)):
        try:
            out[key] = float(brentq(f, 200.0, 450.0, xtol=0.1))
        except ValueError:
            out[key] = None
    TG_REQ.append(out)
    print(site, {k: (round(v, 2) if isinstance(v, float) else v) for k, v in out.items()},
          "| predicted pinic-acid Tg", round(TG["pinic acid"], 1))
json.dump(dict(site_composition=SITE, results=RESULTS, tg_required=TG_REQ, proxy_tg=TG),
          open("ambient_pointwise_results.json", "w"), indent=1)
print("wrote ambient_pointwise_results.json")
''')

nb["cells"] = cells
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "../notebooks/aiomfac_validation_05_ambient_pointwise.ipynb")
print("wrote ../notebooks/aiomfac_validation_05_ambient_pointwise.ipynb")

"""Generate notebooks/aiomfac_validation_10_proxy_oc.ipynb: sensitivity of the ambient phase-state result to the
organic proxy (O:C 0.30-0.80) and to out-of-fold Tg.

A. For six proxies (pinonic acid 0.30, pinic acid 0.44, terpenylic acid 0.50, diaterpenylic acid acetate 0.60,
   MBTCA 0.75, 2-methyltetrol 0.80) plus ammonium sulfate at each site's mean OIR, the scenarios S1 (one liquid),
   S2 (no crystallization) and S3 (equilibrium) of notebook 06 are solved on RH 5-90 %. Reported: the RH range with
   two liquids, the AS fraction in the organic-rich liquid, and the mean offsets (measured minus predicted log10 eta)
   for S1 and S3 against the ambient data.
B. Pinonic and pinic acid with their out-of-fold Tg (notebook 09): the phase equilibrium does not depend on Tg, so
   only the viscosity of the phases is recomputed.
Results: proxy_oc_results.json (model output and summary offsets only; no measured point values).

Run:  python build_nb10.py && (cd ../notebooks && PATH=/opt/miniconda3/bin:$PATH PYTHONPATH=<main export>/src \
          jupyter nbconvert --to notebook --execute --inplace aiomfac_validation_10_proxy_oc.ipynb \
          --ExecutePreprocessor.timeout=-1)
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md("""
# Ambient PM2.5: sensitivity to the organic proxy (O:C) and to out-of-fold Tg

Liquid-liquid phase separation of organic + ammonium sulfate particles is expected below an O:C of about 0.5-0.8
(e.g. Song et al., 2012; You et al., 2013). The proxies of notebook 06 (pinonic acid, O:C 0.30; pinic acid, 0.44) lie
at or below the lower end, while the measured O:C of the water-soluble organic carbon of the Seoul and Beijing samples
is 0.45-0.52. This notebook repeats the phase-state calculation of notebook 06 for proxies with O:C up to 0.80.
""")

code(r'''
import json, math, sys, time, warnings
import numpy as np
from rdkit import Chem
from rdkit.Chem.Descriptors import MolWt
import aiomfac_py
from aiomfac_py import ActivityModel, Component
from aiomfac_py.s2as import smiles_to_components
from aiomfac_py.phase_equilibrium import PhaseEquilibrium
from aiomfac_py.tgml_armeli import predict_tg_ml_smiles
from aiomfac_py.viscosity import aquelec_viscosity, pure_organic_viscosity_vtf, organic_mixture_viscosity
warnings.filterwarnings("ignore")
print("aiomfac_py", aiomfac_py.__version__, aiomfac_py.__file__)

sys.path.insert(0, "../ambient_data")
from ambient_data import load
PTS = load()
SITE = json.load(open("ambient_pointwise_results.json"))["site_composition"]
T_SITE = {"Ansan": 300.0, "Seoul": 290.0, "Beijing": 290.0}
PROXIES = {"pinonic acid": "CC(=O)C1CC(C1(C)C)CC(=O)O", "pinic acid": "CC1(C(CC1C(=O)O)CC(=O)O)C",
           "terpenylic acid": "CC1(C)OC(=O)CC1CC(=O)O",
           "diaterpenylic acid acetate": "CC(=O)OC(C)(C)C(CC(=O)O)CC(=O)O",
           "MBTCA": "CC(C)(C(=O)O)C(CC(=O)O)C(=O)O", "2-methyltetrol": "OCC(C)(O)C(O)CO"}
def oc(smi):
    m = Chem.MolFromSmiles(smi)
    return sum(a.GetSymbol() == "O" for a in m.GetAtoms()) / sum(a.GetSymbol() == "C" for a in m.GetAtoms())
OC = {p: oc(s) for p, s in PROXIES.items()}
TG = {p: predict_tg_ml_smiles(s).tg_K for p, s in PROXIES.items()}
oof = {r["proxy"]: r["tg_oof"] for r in json.load(open("oof_tg_results.json"))["literature"]["soa"]}
TG_OOF = {p: oof[p] for p in ("pinonic acid", "pinic acid")}
MW = {p: MolWt(Chem.MolFromSmiles(s)) for p, s in PROXIES.items()}
ORG = {p: Component(2, "org", smiles_to_components([s], water_as_component1=True).components[1].subgroups)
       for p, s in PROXIES.items()}
MW_AS = 132.14
RH_GRID = [round(0.05 * k, 2) for k in range(1, 19)]          # 0.05 ... 0.90
for p in PROXIES:
    print(f"{p:28s} O:C {OC[p]:.2f}  Tg {TG[p]:.1f} K" + (f"  (OOF {TG_OOF[p]:.1f} K)" if p in TG_OOF else ""))
''')

code(r'''
W = Component(1, "Water", ((16, 1),))
AS_C = Component(3, "(NH4)2SO4", ((204, 2), (261, 1)))
_vm = {}
def visc_of(L, proxy, T, tg):
    names, a = L.names, L.amounts
    w, o, so4 = a[0], a[names.index("org")], a[names.index("SO4--")]
    if proxy not in _vm:
        _vm[proxy] = (ActivityModel([W, ORG[proxy], AS_C]), ActivityModel([W, ORG[proxy]]))
    model, sub = _vm[proxy]
    eta0 = pure_organic_viscosity_vtf(T, tg)
    x = np.array([w, o, so4]); x = x / x.sum()
    if x[2] < 1e-10:
        xs = x[:2] / x[:2].sum()
        return organic_mixture_viscosity(sub, sub.evaluate(xs, T, basis="mole").x, T, {2: eta0}).log10_eta_pas
    return math.log10(aquelec_viscosity(model, model.evaluate(x, T, basis="mole"), T, {2: eta0}).eta_pas)

def feed_of(proxy, oir):
    m_org, m_as = oir / MW[proxy], 1.0 / MW_AS
    return {"org": m_org, "NH4+": 2 * m_as, "SO4--": m_as}

def as_frac(L, feed):
    return float(L.amounts[L.names.index("SO4--")] / feed["SO4--"])

def dry_org_frac(L, proxy):
    """organic mass fraction of a liquid on a water-free basis (organic / (organic + AS))"""
    m_org = L.amounts[L.names.index("org")] * MW[proxy]
    m_as = L.amounts[L.names.index("SO4--")] * MW_AS
    return float(m_org / (m_org + m_as)) if m_org + m_as > 0 else float("nan")

def org_rich(r, proxy):
    """the organic-rich liquid: the liquid with the highest organic mass fraction on a water-free basis"""
    return max(r.liquids, key=lambda L: dry_org_frac(L, proxy))

SOL = {}           # (site, proxy, rh) -> (r1, r2, r3)
t0 = time.time()
for site in ("Ansan", "Seoul", "Beijing"):
    T = T_SITE[site]; oir = SITE[site]["oir_mean"]
    for proxy in PROXIES:
        pe = PhaseEquilibrium([ORG[proxy]], ["NH4+", "SO4--"], T_K=T)
        feed = feed_of(proxy, oir)
        for rh in RH_GRID:
            SOL[(site, proxy, rh)] = (pe.solve(feed, rh, solids="none", max_liquids=1),
                                      pe.solve(feed, rh, solids="none"), pe.solve(feed, rh, solids="all"))
    print(site, f"{time.time() - t0:.0f} s", flush=True)
bad = [k for k, v in SOL.items() if v[1].status != "converged" or v[2].status != "converged"]
print("non-converged S2/S3 solves:", bad)
''')

md("## A. Phase state versus proxy O:C")

code(r'''
GRID = []
for (site, proxy, rh), (r1, r2, r3) in SOL.items():
    T = T_SITE[site]; feed = feed_of(proxy, SITE[site]["oir_mean"])
    L2, L3 = org_rich(r2, proxy), org_rich(r3, proxy)        # organic-rich liquid (highest dry organic fraction)
    g = dict(site=site, proxy=proxy, oc=OC[proxy], rh=rh,
             s1=visc_of(r1.liquids[0], proxy, T, TG[proxy]),
             s2_n_liquids=r2.n_liquids, s2=visc_of(L2, proxy, T, TG[proxy]), s2_as_in_org_liquid=as_frac(L2, feed),
             s3_n_liquids=r3.n_liquids, s3_solids=list(r3.solids), s3=visc_of(L3, proxy, T, TG[proxy]),
             s3_as_dissolved=float(sum(as_frac(L, feed) for L in r3.liquids)),
             s2_dry_org=[dry_org_frac(L, proxy) for L in r2.liquids],
             s2_water_mass_frac=[float(L.amounts[0] * 18.015 / (L.amounts[0] * 18.015 + L.amounts[L.names.index("org")] * MW[proxy]
                                       + L.amounts[L.names.index("SO4--")] * MW_AS)) for L in r2.liquids],
             s2_first_is_org_rich=bool(r2.liquids[0] is L2), s3_first_is_org_rich=bool(r3.liquids[0] is L3))
    if proxy in TG_OOF:
        g.update(s1_oof=visc_of(r1.liquids[0], proxy, T, TG_OOF[proxy]), s3_oof=visc_of(L3, proxy, T, TG_OOF[proxy]))
    GRID.append(g)

print("organic-rich liquid = solver's first liquid: S2", sum(g["s2_first_is_org_rich"] for g in GRID), "/", len(GRID),
      "; S3", sum(g["s3_first_is_org_rich"] for g in GRID), "/", len(GRID))
LLPS = []
print(f"{'site':8s} {'proxy':28s} {'O:C':>4s}  RH with two liquids (S2)          AS in org-rich liquid at RH 15-45 %")
for site in ("Ansan", "Seoul", "Beijing"):
    for proxy in PROXIES:
        rows = [g for g in GRID if g["site"] == site and g["proxy"] == proxy]
        two = [g["rh"] for g in rows if g["s2_n_liquids"] >= 2]
        meas = [g for g in rows if 0.15 <= g["rh"] <= 0.45]
        all_meas_two = all(g["s2_n_liquids"] >= 2 for g in meas)
        as_org = max(g["s2_as_in_org_liquid"] for g in meas)
        twos = [g for g in meas if g["s2_n_liquids"] >= 2]
        if twos:
            lo = min(min(g["s2_dry_org"]) for g in twos); hi = max(max(g["s2_dry_org"]) for g in twos)
            print(f"    two-liquid states at RH 15-45 %: dry organic mass fraction of the liquids {lo:.2f}-{hi:.2f}")
        LLPS.append(dict(site=site, proxy=proxy, oc=OC[proxy], rh_two_liquids=two,
                         rh_max_two=max(two) if two else None, all_measured_rh_two=all_meas_two,
                         max_as_in_org_liquid_measured_rh=as_org))
        span = f"{min(two):.2f}-{max(two):.2f} ({len(two)}/{len(rows)})" if two else "none"
        print(f"{site:8s} {proxy:28s} {OC[proxy]:4.2f}  {span:32s}  {'all' if all_meas_two else 'NOT all'}; "
              f"max AS fraction {as_org:.3f}")
''')

md("## Offsets against the ambient data (mean OIR; resolved points)")

code(r'''
def offsets(site, proxy, key):
    rows = sorted([g for g in GRID if g["site"] == site and g["proxy"] == proxy], key=lambda g: g["rh"])
    rh = np.array([g["rh"] * 100 for g in rows]); y = np.array([g[key] for g in rows])
    pts = [p for p in PTS if p["site"] == site and not p["censored"]]
    off = np.array([p["log10_eta"] for p in pts]) - np.interp([p["rh_pct"] for p in pts], rh, y)
    return float(off.mean()), float(off.std()), int(np.sum(np.abs(off) <= 1)), len(off)

OFF = []
print(f"{'site':8s} {'proxy':28s} {'O:C':>4s} {'Tg':>6s} | {'S1 offset':>12s} | {'S2 offset':>12s} | {'S3 offset':>12s}")
for site in ("Ansan", "Seoul", "Beijing"):
    for proxy in PROXIES:
        s1, s2, s3 = offsets(site, proxy, "s1"), offsets(site, proxy, "s2"), offsets(site, proxy, "s3")
        OFF.append(dict(site=site, proxy=proxy, oc=OC[proxy], tg=TG[proxy], tg_label="predicted",
                        s1_mean=s1[0], s1_sd=s1[1], s2_mean=s2[0], s2_sd=s2[1], s3_mean=s3[0], s3_sd=s3[1],
                        s3_within_1=s3[2], n=s3[3]))
        print(f"{site:8s} {proxy:28s} {OC[proxy]:4.2f} {TG[proxy]:6.1f} | {s1[0]:+6.2f} ± {s1[1]:.2f} | "
              f"{s2[0]:+6.2f} ± {s2[1]:.2f} | {s3[0]:+6.2f} ± {s3[1]:.2f}")
        if proxy in TG_OOF:
            s1o, s3o = offsets(site, proxy, "s1_oof"), offsets(site, proxy, "s3_oof")
            OFF.append(dict(site=site, proxy=proxy, oc=OC[proxy], tg=TG_OOF[proxy], tg_label="out-of-fold",
                            s1_mean=s1o[0], s1_sd=s1o[1], s3_mean=s3o[0], s3_sd=s3o[1], s3_within_1=s3o[2], n=s3o[3]))
            print(f"{'':8s} {'  ... out-of-fold Tg':28s} {'':4s} {TG_OOF[proxy]:6.1f} | {s1o[0]:+6.2f} ± {s1o[1]:.2f} | "
                  f"{s3o[0]:+6.2f} ± {s3o[1]:.2f}")
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
OUT = dict(aiomfac_py_version=aiomfac_py.__version__, proxies=PROXIES, oc=OC, tg=TG, tg_oof=TG_OOF,
           rh_grid=RH_GRID, grid=GRID, llps=LLPS, offsets=OFF)
json.dump(clean(OUT), open("proxy_oc_results.json", "w"), indent=1)
print("wrote proxy_oc_results.json")
''')

nb["cells"] = cells
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "../notebooks/aiomfac_validation_10_proxy_oc.ipynb")
print("wrote ../notebooks/aiomfac_validation_10_proxy_oc.ipynb")

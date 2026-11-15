"""Generate notebooks/aiomfac_validation_06_ambient_phase_state.ipynb (ambient PM2.5, phase state with the combined
liquid-liquid-solid solver aiomfac_py.phase_equilibrium).

Run:  python build_nb06.py && jupyter nbconvert --to notebook --execute --inplace \
          ../notebooks/aiomfac_validation_06_ambient_phase_state.ipynb --ExecutePreprocessor.timeout=-1
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md("""
# Ambient PM2.5: phase state and viscosity of the organic-rich phase

Compares the data of Do et al. (2026a, Fig. 2a; Ansan, Seoul, Beijing; provided by the authors) with three
phase-state scenarios for a proxy organic (pinonic or pinic acid, ML-predicted Tg) plus ammonium sulfate (AS). Phase
states are computed with `aiomfac_py.phase_equilibrium.PhaseEquilibrium` (combined liquid-liquid-solid solver; branch
`feature/phase-equilibrium`, commit adf3c67 of aiomfac-python), which minimizes the Gibbs energy over all liquid phases
and candidate solids at fixed RH and checks liquid-phase stability with a tangent-plane test.

* **S1, one liquid phase:** the single liquid forced (`max_liquids=1`, no solids); aquelec viscosity.
* **S2, liquid-liquid phase separation, no crystallization:** metastable solve (`solids="none"`); viscosity of the
  organic-rich liquid (the shell of a core-shell particle).
* **S3, thermodynamic equilibrium:** solids allowed (`solids="all"`); viscosity of the organic-rich liquid.

An additional section uses AS + ammonium nitrate (AN) in each site's ratio and follows a drying path on which AS
crystallizes only after a critical supersaturation calibrated to an AS efflorescence RH of 35 %.

This notebook replaces an earlier version that used `aiomfac_py.lle.solve_pep` at fixed water content; that solver's
multi-start initialization missed the liquid-liquid split below about 35 % RH (the split found here lowers the same
Gibbs function and the one-phase state fails the stability test).
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
from aiomfac_py.sle import SLESolver, implied_ln_s_crit, binary_saturation
from aiomfac_py.solids import SOLIDS
from aiomfac_py.tgml_armeli import predict_tg_ml_smiles
from aiomfac_py.viscosity import aquelec_viscosity, pure_organic_viscosity_vtf, organic_mixture_viscosity
warnings.filterwarnings("ignore")
import aiomfac_py.phase_equilibrium as _pe
print("aiomfac_py", aiomfac_py.__version__, "(feature/phase-equilibrium @ adf3c67)", _pe.__file__)

sys.path.insert(0, "../ambient_data")
from ambient_data import load
PTS = load()
NB05 = json.load(open("ambient_pointwise_results.json"))
SITE = NB05["site_composition"]
T_SITE = {"Ansan": 300.0, "Seoul": 290.0, "Beijing": 290.0}
PROXIES = {"pinonic acid": "CC(=O)C1CC(C1(C)C)CC(=O)O", "pinic acid": "CC1(C(CC1C(=O)O)CC(=O)O)C"}
TG = {p: predict_tg_ml_smiles(s).tg_K for p, s in PROXIES.items()}
MW = {p: MolWt(Chem.MolFromSmiles(s)) for p, s in PROXIES.items()}
ORG = {p: Component(2, "org", smiles_to_components([s], water_as_component1=True).components[1].subgroups)
       for p, s in PROXIES.items()}
MW_AS, MW_AN = 132.14, 80.04
RH_GRID = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]
print("Tg", {k: round(v, 1) for k, v in TG.items()})
''')

md("## 1. Inorganic phase state of each site's salt mixture (SLE)")

code(r'''
sle = SLESolver(["NH4+", "SO4--", "NO3-"])
ln_s_crit_as = {T: implied_ln_s_crit("ammonium_sulfate", T, 0.35) for T in (290.0, 300.0)}
SLE_OUT = {}
for site, S in SITE.items():
    T = T_SITE[site]
    f_an = S["an_frac_of_salt"]
    feed = {"NH4+": 2 * (1 - f_an) / MW_AS + f_an / MW_AN, "SO4--": (1 - f_an) / MW_AS, "NO3-": f_an / MW_AN}
    drh = sle.deliquescence_rh(feed, T); fdrh = sle.full_dissolution_rh(feed, T)
    erh = sle.efflorescence_rh(feed, T, ln_s_crit=ln_s_crit_as[T])
    SLE_OUT[site] = dict(T_K=T, an_mass_frac=f_an, drh=drh, full_dissolution_rh=fdrh, erh=erh["rh"],
                         erh_salt=erh["salt"], ln_s_crit_as=ln_s_crit_as[T])
    print(f"{site:8s} T={T:.0f} AN mass frac={f_an:.2f}  DRH={drh:.3f}  full-dissolution RH={fdrh:.3f}  "
          f"ERH(ln S_crit={ln_s_crit_as[T]:.2f})={erh['rh']:.3f} ({erh['salt']})")
''')

md("## 2. Organic + AS: scenarios S1, S2, S3 on an RH grid")

code(r'''
W = Component(1, "Water", ((16, 1),))
AS_C = Component(3, "(NH4)2SO4", ((204, 2), (261, 1)))
AN_C = Component(4, "NH4NO3", ((204, 1), (245, 1)))
_vm = {}
def visc_of(L, proxy, T):
    """aquelec viscosity of one liquid phase (amounts in the ion basis -> salt-unit mole fractions)."""
    names = L.names; a = L.amounts
    w, o = a[0], a[names.index("org")]
    so4 = a[names.index("SO4--")]; no3 = a[names.index("NO3-")] if "NO3-" in names else 0.0
    with_an = "NO3-" in names
    key = (proxy, T, with_an)
    if key not in _vm:
        _vm[key] = ActivityModel([W, ORG[proxy], AS_C] + ([AN_C] if with_an else []))
    model = _vm[key]
    x = np.array([w, o, so4] + ([no3] if with_an else [])); x = x / x.sum()
    eta0 = pure_organic_viscosity_vtf(T, TG[proxy])
    if x[2] + (x[3] if with_an else 0.0) < 1e-10:
        sub = ActivityModel([W, ORG[proxy]]); xs = x[:2] / x[:2].sum()
        return organic_mixture_viscosity(sub, sub.evaluate(xs, T, basis="mole").x, T, {2: eta0}).log10_eta_pas
    res = model.evaluate(x, T, basis="mole")
    return math.log10(aquelec_viscosity(model, res, T, {2: eta0}).eta_pas)

def feed_of(proxy, oir, f_an=0.0):
    m_org, m_as, m_an = oir / MW[proxy], (1 - f_an) / MW_AS, f_an / MW_AN
    f = {"org": m_org, "NH4+": 2 * m_as + m_an, "SO4--": m_as}
    if f_an > 0:
        f["NO3-"] = m_an
    return f

def summarize(r, proxy, T):
    L = r.liquids[0]                                   # liquids are sorted by organic mole fraction
    return dict(status=r.status, n_phases=r.n_liquids, solids=r.solids, log10_eta_org_phase=visc_of(L, proxy, T),
                x_org_phase=L.mole_fractions.tolist(), aw_check=L.water_activity, tpd_min=r.tpd_min,
                max_si=r.checks["max_si"], ln_aw_resid=r.checks["max_abs_ln_aw_minus_ln_rh"])

PE = {}
GRID = []
t0 = time.time()
cases = [(site, p, "oir_mean") for site in ("Ansan", "Seoul", "Beijing") for p in PROXIES]
cases += [(site, "pinic acid", lab) for site in ("Ansan", "Seoul", "Beijing") for lab in ("oir_min", "oir_max")]
for site, proxy, lab in cases:
    T = T_SITE[site]; oir = SITE[site][lab]
    key = (proxy, T, False)
    if key not in PE:
        PE[key] = PhaseEquilibrium([ORG[proxy]], ["NH4+", "SO4--"], T_K=T)
    pe = PE[key]; feed = feed_of(proxy, oir)
    for rh in RH_GRID:
        r1 = pe.solve(feed, rh, solids="none", max_liquids=1)
        r2 = pe.solve(feed, rh, solids="none")
        r3 = pe.solve(feed, rh, solids="all")
        s1 = visc_of(r1.liquids[0], proxy, T)
        s2 = summarize(r2, proxy, T)
        s3 = summarize(r3, proxy, T)
        L3 = r3.liquids[0]
        s3.update(log10_eta=s3["log10_eta_org_phase"], solid_as="ammonium_sulfate" in r3.solids,
                  frac_as_dissolved=float(sum(L.amounts[L.names.index("SO4--")] for L in r3.liquids) / feed["SO4--"]),
                  liquid_splits=r3.n_liquids > 1)
        GRID.append(dict(site=site, proxy=proxy, oir_label=lab, oir=oir, rh=rh, s1=s1, s2=s2, s3=s3))
        print(f"{site:8s} {proxy:12s} {lab:8s} OIR={oir:4.2f} RH={rh:.2f} | S1 {s1:6.2f} | S2 nph={s2['n_phases']} "
              f"{s2['log10_eta_org_phase']:6.2f} {s2['status'][:4]} | S3 nph={s3['n_phases']} solid={s3['solid_as']} "
              f"{s3['log10_eta']:6.2f} diss={s3['frac_as_dissolved']:.3f} {s3['status'][:4]}", flush=True)
print(f"elapsed {time.time() - t0:.0f} s")
bad = [(g["site"], g["proxy"], g["oir_label"], g["rh"]) for g in GRID
       for k in ("s2", "s3") if g[k]["status"] != "converged"]
print("non-converged solves:", bad)
''')

md("## 3. Point-wise comparison with the data (linear interpolation of the RH grid)")

code(r'''
def series(site, proxy, lab, key):
    rows = [g for g in GRID if g["site"] == site and g["proxy"] == proxy and g["oir_label"] == lab]
    rh = np.array([g["rh"] * 100 for g in rows])
    y = np.array([g["s1"] if key == "s1" else g[key]["log10_eta_org_phase"] for g in rows], float)
    return rh, y

def score(site, proxy, lab, key):
    pts = [p for p in PTS if p["site"] == site]
    rh, y = series(site, proxy, lab, key)
    pred = np.interp([p["rh_pct"] for p in pts], rh, y)
    meas = np.array([p["log10_eta"] for p in pts]); cen = np.array([p["censored"] for p in pts])
    off = meas - pred; res = off[~cen]
    return dict(site=site, proxy=proxy, oir_label=lab, scenario=key, n_grid_missing=0,
                mean_offset=float(res.mean()), sd=float(res.std()), n_within_1=int(np.sum(np.abs(res) <= 1)),
                n=int(len(res)), censored_shortfall=float(np.mean(np.maximum(0, off[cen]))) if cen.any() else None)

SCORES = [score(site, proxy, lab, key) for site, proxy, lab in cases for key in ("s1", "s2", "s3")]
print(f"{'site':8s} {'proxy':13s} {'OIR':8s} {'scen':4s} {'mean±sd':>14s} {'within1':>8s} {'cens.short':>10s}")
for s in SCORES:
    cs = "-" if s["censored_shortfall"] is None else f"{s['censored_shortfall']:.2f}"
    print(f"{s['site']:8s} {s['proxy']:13s} {s['oir_label']:8s} {s['scenario']:4s} {s['mean_offset']:+6.2f} ± {s['sd']:.2f} "
          f"{s['n_within_1']:4d}/{s['n']:<3d} {cs:>10s}")
''')

md("""
## 4. AS + AN: equilibrium and drying path (pinic acid, mean OIR)

Ammonium nitrate in each site's AN:AS mass ratio (Do et al., 2026a, Table 1). Drying path: AS is admitted once its
saturation index exceeds ln S_crit (calibrated to an AS efflorescence RH of 35 %); AN is not allowed to crystallize
(no efflorescence of AN-rich solutions is expected at these RH). The path starts at 80 % RH.
""")

code(r'''
PATH = []
for site in ("Ansan", "Seoul", "Beijing"):
    T = T_SITE[site]; S = SITE[site]
    pe = PhaseEquilibrium([ORG["pinic acid"]], ["NH4+", "SO4--", "NO3-"], T_K=T)
    feed = feed_of("pinic acid", S["oir_mean"], S["an_frac_of_salt"])
    grid = [0.80, 0.70, 0.60, 0.50] + sorted(RH_GRID, reverse=True)
    eq = {rh: pe.solve(feed, rh, solids="all") for rh in grid}
    dry = {round(r.rh, 2): r for r in pe.drying_path(feed, grid, ln_s_crit={"ammonium_sulfate": ln_s_crit_as[T],
                                                                                "ammonium_nitrate": 99.0})}
    for rh in grid:
        e, d = eq[rh], dry[round(rh, 2)]
        row = dict(site=site, rh=rh, eq=summarize(e, "pinic acid", T), dry=summarize(d, "pinic acid", T))
        PATH.append(row)
        print(f"{site:8s} RH={rh:.2f} | equilibrium nph={e.n_liquids} solids={sorted(e.solids)} "
              f"eta_org={row['eq']['log10_eta_org_phase']:6.2f} {e.status[:4]} | drying nph={d.n_liquids} "
              f"solids={sorted(d.solids)} eta_org={row['dry']['log10_eta_org_phase']:6.2f} {d.status[:4]}", flush=True)

def score_path(site, key):
    pts = [p for p in PTS if p["site"] == site and not p["censored"]]
    rows = sorted([r for r in PATH if r["site"] == site and r["rh"] <= 0.45], key=lambda r: r["rh"])
    pred = np.interp([p["rh_pct"] for p in pts], [r["rh"] * 100 for r in rows],
                     [r[key]["log10_eta_org_phase"] for r in rows])
    off = np.array([p["log10_eta"] for p in pts]) - pred
    return dict(site=site, path=key, mean_offset=float(off.mean()), sd=float(off.std()),
                n_within_1=int(np.sum(np.abs(off) <= 1)), n=len(off))
PATH_SCORES = [score_path(s, k) for s in ("Ansan", "Seoul", "Beijing") for k in ("eq", "dry")]
for s in PATH_SCORES:
    print(f"{s['site']:8s} {s['path']:3s} offset {s['mean_offset']:+.2f} ± {s['sd']:.2f}  within 1: {s['n_within_1']}/{s['n']}")

json.dump(dict(sle=SLE_OUT, grid=GRID, scores=SCORES, path=PATH, path_scores=PATH_SCORES, proxy_tg=TG,
               solver="aiomfac_py.phase_equilibrium (feature/phase-equilibrium @ adf3c67)"),
          open("ambient_phase_state_results.json", "w"), indent=1, default=float)
print("wrote ambient_phase_state_results.json")
''')

nb["cells"] = cells
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nb.metadata["language_info"] = {"name": "python"}
nbf.write(nb, "../notebooks/aiomfac_validation_06_ambient_phase_state.ipynb")
print("wrote ../notebooks/aiomfac_validation_06_ambient_phase_state.ipynb")

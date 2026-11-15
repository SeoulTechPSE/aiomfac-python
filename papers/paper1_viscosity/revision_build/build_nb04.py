"""Generate notebooks/aiomfac_validation_04_revision.ipynb (re-analyses for the revised manuscript).

Run:  python build_nb04.py && jupyter nbconvert --to notebook --execute --inplace \
          ../notebooks/aiomfac_validation_04_revision.ipynb --ExecutePreprocessor.timeout=-1
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md("""
# Revision analyses for the paper_1 manuscript

Re-runs, with fixed and documented settings, every number that changed in the revised manuscript:

1. Literature curves (53): ML-Tg and measured-Tg pipeline MAE, and the effect of a ±15 K Tg error
   (the out-of-sample nested-CV MAE of the Tg predictor, Armeli Iapichino et al., 2023).
2. Sensitivity of the G2020 mixture viscosity to the pure-component viscosity eta0, and a wide eta0 search
   (log10 eta0 from -6 to 40, so that the pipeline's own eta0 always lies inside the search range).
3. Ideal (mole-fraction log-linear) mixing versus G2020, each at its own best eta0.
4. Fragility parameter D required to reproduce the best-fit eta0 at the ML Tg.
5. Chamber SOA: point MAE and an interval score that respects the asymmetric (one-sided) uncertainty of the
   Gervasi et al. (2020) SOA data.
6. Ullah et al. (2026) ambient comparison restricted to the measured RH range (25-45 %), with ML and
   measured proxy Tg.

All results are written to `revision_results.json`. Requires `aiomfac_py[carbonate,smiles,tgml,tgml-smiles]`
(v1.0.1) and the Gervasi et al. (2020) Supplement as `gervasi_supp.zip` in this folder.
""")

code(r'''
import json, re, io, zipfile, math, warnings
import numpy as np
import aiomfac_py
from aiomfac_py import ActivityModel, Component
from aiomfac_py.s2as import smiles_to_components
from aiomfac_py.tgml_armeli import predict_tg_ml_smiles
from aiomfac_py.viscosity import (organic_mixture_viscosity, pure_organic_viscosity_vtf, water_viscosity_pas,
                                  aquelec_viscosity, aquorg_viscosity)
warnings.filterwarnings("ignore")
print("aiomfac_py", aiomfac_py.__version__)

WATER = Component(1, "Water", ((16, 1),))
OUT = {"aiomfac_py_version": aiomfac_py.__version__}
curves = json.load(open("literature_curves.json"))
ZIP = zipfile.ZipFile("gervasi_supp.zip")

def organic_model(smiles):
    r = smiles_to_components([smiles], water_as_component1=True)
    assert not r.removed and r.n_unmatched_atoms[0] == 0, smiles
    return ActivityModel([WATER, Component(2, "organic", r.components[1].subgroups)])

def safe_eta0(T_K, tg_K):
    try:
        return pure_organic_viscosity_vtf(T_K, tg_K)
    except ValueError:
        return np.nan

def mix_log10(model, x, T_K, eta0):
    try:
        e = organic_mixture_viscosity(model, x, T_K, {2: eta0}).eta_pas
        return math.log10(e) if np.isfinite(e) and e > 0 else np.nan
    except (ValueError, OverflowError):
        return np.nan

def ideal_log10(x, T_K, eta0):
    return (x[0] * math.log(water_viscosity_pas(T_K)) + x[1] * math.log(eta0)) / math.log(10.0)

def mae(pred, y):
    v = np.isfinite(pred) & np.isfinite(y)
    return float(np.mean(np.abs(pred[v] - y[v]))) if v.any() else None
''')

md("## 1. Literature curves: pipeline, oracle Tg and a ±15 K Tg error")

code(r'''
TG_ERR = 15.1   # nested-CV MAE of the SMILES-based Tg predictor (Armeli Iapichino et al., 2023)
tg_ml_cache, prepared = {}, []
for c in curves:
    if c["smiles"] not in tg_ml_cache:
        tg_ml_cache[c["smiles"]] = predict_tg_ml_smiles(c["smiles"]).tg_K
    model = organic_model(c["smiles"])
    pts = c["points"]
    xs = [model.evaluate([1 - p["wtf_organic"], p["wtf_organic"]], p["T_K"], basis="mass").x for p in pts]
    y = np.array([p["log10_eta_measured"] for p in pts])
    wtf_mean = float(np.mean([p["wtf_organic"] for p in pts]))
    prepared.append(dict(c=c, model=model, xs=xs, y=y, T=[p["T_K"] for p in pts],
                         regime="glassy" if wtf_mean >= 0.5 else "dilute", wtf_mean=wtf_mean))

def curve_pred(P, tg_K=None, log10_eta0=None, rule="g2020"):
    out = []
    for x, T in zip(P["xs"], P["T"]):
        eta0 = 10.0 ** log10_eta0 if log10_eta0 is not None else safe_eta0(T, tg_K)
        if not np.isfinite(eta0):
            out.append(np.nan); continue
        out.append(mix_log10(P["model"], x, T, eta0) if rule == "g2020" else ideal_log10(x, T, eta0))
    return np.array(out)

rows = []
for P in prepared:
    c = P["c"]; tg = tg_ml_cache[c["smiles"]]
    r = dict(name=c["name"], source=c["source"], n=len(P["y"]), regime=P["regime"], wtf_mean=P["wtf_mean"],
             tg_ml_K=tg, tg_measured_K=c["tg_measured_K"],
             mae_ml=mae(curve_pred(P, tg), P["y"]),
             mae_ml_plus15=mae(curve_pred(P, tg + TG_ERR), P["y"]),
             mae_ml_minus15=mae(curve_pred(P, tg - TG_ERR), P["y"]),
             mae_oracle=mae(curve_pred(P, c["tg_measured_K"]), P["y"]) if c["tg_measured_K"] else None)
    rows.append(r)

def mean_of(key, sel=lambda r: True):
    v = [r[key] for r in rows if sel(r) and r[key] is not None]
    return float(np.mean(v)), len(v)

tg_err = {}
for r in rows:
    if r["tg_measured_K"]:
        tg_err[r["name"]] = r["tg_ml_K"] - r["tg_measured_K"]
e = np.array(list(tg_err.values()))
summary = dict(
    n_curves=len(rows), n_points=int(sum(r["n"] for r in rows)),
    mae_all=mean_of("mae_ml")[0], median_all=float(np.median([r["mae_ml"] for r in rows])),
    mae_dilute=mean_of("mae_ml", lambda r: r["regime"] == "dilute"),
    mae_glassy=mean_of("mae_ml", lambda r: r["regime"] == "glassy"),
    oracle_subset_ml=mean_of("mae_ml", lambda r: r["mae_oracle"] is not None),
    oracle_subset_oracle=mean_of("mae_oracle"),
    tg_mae_K=float(np.mean(np.abs(e))), tg_bias_K=float(e.mean()), tg_n=len(e),
    tg_err_by_compound=tg_err)
for key in ("mae_ml", "mae_ml_plus15", "mae_ml_minus15"):
    for reg in ("dilute", "glassy"):
        summary[f"{key}_{reg}"] = mean_of(key, lambda r, reg=reg: r["regime"] == reg)
OUT["literature"] = dict(summary=summary, per_curve=rows)
for k, v in summary.items():
    if k != "tg_err_by_compound":
        print(f"{k:28s} {v}")
print({k: round(v, 1) for k, v in tg_err.items()})
''')

md("""
## 2. Sensitivity to eta0 and a wide eta0 search (glassy curves)

`w_eff` is d log10(eta_mix) / d log10(eta0) at the pipeline eta0, averaged over the points of a curve: the
effective weight G2020 gives the organic's pure-component viscosity. For comparison, `x_org` is the mean organic
mole fraction. The eta0 search uses a 0.25-step grid over log10 eta0 in [-6, 40] followed by a bounded
refinement around the best grid point, for G2020 and for the ideal rule separately.
""")

code(r'''
from scipy.optimize import minimize_scalar
GRID = np.arange(-6.0, 40.0001, 0.25)

def best_eta0(P, rule):
    f = lambda l: (mae(curve_pred(P, log10_eta0=l, rule=rule), P["y"]) or 1e6)
    vals = np.array([f(l) for l in GRID])
    i = int(np.argmin(vals))
    lo, hi = GRID[max(i - 1, 0)], GRID[min(i + 1, len(GRID) - 1)]
    res = minimize_scalar(f, bounds=(lo, hi), method="bounded", options={"xatol": 1e-3})
    l, v = (res.x, res.fun) if res.fun <= vals[i] else (GRID[i], vals[i])
    return float(l), float(v), bool(i in (0, len(GRID) - 1))

diag = []
for P, r in zip(prepared, rows):
    if P["regime"] != "glassy":
        continue
    tg = r["tg_ml_K"]
    l_pipe = [math.log10(safe_eta0(T, tg)) for T in P["T"]]
    w = []
    for x, T, lp in zip(P["xs"], P["T"], l_pipe):
        a, b = mix_log10(P["model"], x, T, 10 ** (lp - 1)), mix_log10(P["model"], x, T, 10 ** (lp + 1))
        w.append((b - a) / 2.0)
    l_g, m_g, edge_g = best_eta0(P, "g2020")
    l_i, m_i, edge_i = best_eta0(P, "ideal")
    d = dict(name=r["name"], source=r["source"], n=r["n"], tg_ml_K=tg,
             log10_eta0_pipeline=float(np.mean(l_pipe)), w_eff=float(np.nanmean(w)),
             x_org=float(np.mean([x[1] for x in P["xs"]])),
             mae_g2020_pipeline=r["mae_ml"], log10_eta0_best_g2020=l_g, mae_g2020_best=m_g, best_at_edge_g2020=edge_g,
             mae_ideal_pipeline=mae(curve_pred(P, tg, rule="ideal"), P["y"]),
             log10_eta0_best_ideal=l_i, mae_ideal_best=m_i, best_at_edge_ideal=edge_i)
    diag.append(d)
    print(f"{d['name']:18s} {d['source']:13s} eta0_pipe={d['log10_eta0_pipeline']:6.1f} w_eff={d['w_eff']:.3f} "
          f"x_org={d['x_org']:.3f} | G2020 {d['mae_g2020_pipeline']:.2f} -> {m_g:.2f} @ {l_g:5.1f}"
          f"{' EDGE' if edge_g else ''} | ideal {d['mae_ideal_pipeline']:.2f} -> {m_i:.2f} @ {l_i:5.1f}"
          f"{' EDGE' if edge_i else ''}")
OUT["glassy_diagnosis"] = diag
print("mean MAE glassy: G2020 pipeline %.3f, G2020 best %.3f, ideal pipeline %.3f, ideal best %.3f" % tuple(
    np.mean([d[k] for d in diag]) for k in ("mae_g2020_pipeline", "mae_g2020_best", "mae_ideal_pipeline",
                                             "mae_ideal_best")))
''')

md("""
## 3. Fragility parameter D required to reproduce the best-fit eta0 at the ML Tg

Modified VTF of Gervasi et al. (2020): T0 = 39.17 Tg / (D + 39.17), log10 eta0 = -5 + 0.434 T0 D / (T - T0).
The pipeline uses D = 10, or D = 30 when Tg > T. Evaluated for glassy curves whose best-fit eta0 is not at the
search edge and whose MAE falls by at least 0.1 log10 units.
""")

code(r'''
from scipy.optimize import brentq
frag = []
for d, P in zip([d for d in diag], [P for P in prepared if P["regime"] == "glassy"]):
    if d["best_at_edge_g2020"] or d["mae_g2020_pipeline"] - d["mae_g2020_best"] < 0.1:
        continue
    T = P["T"][0]; tg = d["tg_ml_K"]; target = d["log10_eta0_best_g2020"]
    def f(D):
        T0 = 39.17 * tg / (D + 39.17)
        return 1e6 if T <= T0 else (-5.0 + 0.434 * T0 * D / (T - T0)) - target
    try:
        D_req = float(brentq(f, 0.5, 300.0, xtol=0.01))
    except ValueError:
        D_req = None
    frag.append(dict(name=d["name"], source=d["source"], T_K=T, tg_ml_K=tg, D_pipeline=30.0 if tg > T else 10.0,
                     log10_eta0_pipeline=d["log10_eta0_pipeline"], log10_eta0_best=target, D_required=D_req,
                     mae_pipeline=d["mae_g2020_pipeline"], mae_best=d["mae_g2020_best"]))
    print(f"{d['name']:16s} {d['source']:13s} Tg={tg:6.1f} T={T:6.2f} D_pipe={frag[-1]['D_pipeline']:4.0f} "
          f"eta0 {d['log10_eta0_pipeline']:6.2f}->{target:6.2f}  D_req={D_req if D_req is None else round(D_req, 1)}"
          f"  MAE {d['mae_g2020_pipeline']:.2f}->{d['mae_g2020_best']:.2f}")
OUT["fragility"] = frag
''')

md("""
## 4. Chamber SOA: point MAE and interval score

The SOA files give log10 uncertainties (columns 'y-error neg' and 'y-error pos'). Several are strongly one-sided
(e.g. toluene 'solid', y-error neg = 0.0001, pos ≈ 3.4), i.e. the reported value is effectively a lower bound.
The interval score is the distance (log10 units) from the prediction to the interval
[log10 eta − neg, log10 eta + pos], zero inside it. Proxies use the ML-predicted Tg; for pinonic and pinic acid the
measured Tg is shown as a sensitivity case.
""")

code(r'''
def parse_soa(name):
    txt = ZIP.read(name).decode("utf-8-sig")
    lines = [l.strip().strip('"') for l in txt.splitlines()]
    T = float(re.search(r"T\s*=\s*([\d.]+)\s*K", lines[0]).group(1))
    pts = []
    for line in lines[4:]:
        p = [s for s in re.split(r"[,\t]", line) if s.strip()]
        if len(p) < 6:
            continue
        rh, eta, _, _, yn, yp = map(float, p[:6])
        if eta > 0 and 0 < rh < 100:
            pts.append((rh / 100.0, math.log10(eta), yn, yp))
    return T, np.array(pts)

def wtf_for_aw(model, aw, T, lo=0.0005, hi=0.9995):
    g = lambda w: float(model.evaluate([w, 1 - w], T, basis="mass").activity[0]) - aw
    if g(lo) > 0 or g(hi) < 0:
        return None
    return brentq(g, lo, hi, xtol=1e-6)

def soa_pred(smiles, tg, T, aws):
    model = organic_model(smiles); eta0 = safe_eta0(T, tg); out = []
    for aw in aws:
        w = wtf_for_aw(model, aw, T)
        if w is None or not np.isfinite(eta0):
            out.append(np.nan); continue
        out.append(mix_log10(model, model.evaluate([w, 1 - w], T, basis="mass").x, T, eta0))
    return np.array(out)

def interval_score(pred, y, yn, yp):
    v = np.isfinite(pred)
    d = np.maximum(0.0, np.maximum((y - yn) - pred, pred - (y + yp)))
    return float(np.mean(d[v])), int(np.sum((d == 0) & v)), int(v.sum())

PROXIES = {
    "aPinene": {"pinonic acid": ("CC(=O)C1CC(C1(C)C)CC(=O)O", 265.0), "pinic acid": ("CC1(C(CC1C(=O)O)CC(=O)O)C", 268.1)},
    "isoprene": {"2-methyltetrol": ("OCC(C)(O)C(O)CO", None)},
    "toluene": {"4-methylcatechol": ("Cc1ccc(O)c(O)c1", None), "glutaric acid": ("OC(=O)CCCC(=O)O", None),
                "2-methylglutaric acid": ("OC(=O)C(C)CCC(=O)O", None), "DHOPA": ("CC(=O)C(O)C(O)C(=O)O", None),
                "4-methyl-5-nitrocatechol": ("Cc1cc(O)c(O)cc1[N+](=O)[O-]", None)},
}
soa = []
for system, proxies in PROXIES.items():
    files = sorted(n for n in ZIP.namelist() if f"{system}_SOA" in n and n.endswith(".txt"))
    for pname, (smi, tg_meas) in proxies.items():
        tg_ml = predict_tg_ml_smiles(smi).tg_K
        for fn in files:
            T, pts = parse_soa(fn)
            study = fn.split("_SOA_")[1].rsplit(".txt", 1)[0]
            for tg_label, tg in (("ml", tg_ml), ("measured", tg_meas)):
                if tg is None:
                    continue
                pred = soa_pred(smi, tg, T, pts[:, 0])
                s, n_in, n_v = interval_score(pred, pts[:, 1], pts[:, 2], pts[:, 3])
                one_sided = int(np.sum(np.minimum(pts[:, 2], pts[:, 3]) < 0.01))
                soa.append(dict(system=system, study=study, proxy=pname, tg=tg_label, tg_K=tg, T_K=T, n=len(pts),
                                n_one_sided=one_sided, mae=mae(pred, pts[:, 1]), interval=s, n_inside=n_in,
                                n_valid=n_v, pred_mean=float(np.nanmean(pred)), meas_mean=float(pts[:, 1].mean())))
OUT["soa"] = soa
for system, proxies in PROXIES.items():
    for pname in proxies:
        for tg_label in ("ml", "measured"):
            sel = [r for r in soa if r["system"] == system and r["proxy"] == pname and r["tg"] == tg_label]
            if sel:
                print(f"{system:9s} {pname:26s} Tg={tg_label:8s} Tg_K={sel[0]['tg_K']:6.1f} curves={len(sel)} "
                      f"MAE={np.mean([r['mae'] for r in sel]):.2f} interval={np.mean([r['interval'] for r in sel]):.2f} "
                      f"inside={sum(r['n_inside'] for r in sel)}/{sum(r['n_valid'] for r in sel)}")
print()
for r in soa:
    if r["tg"] == "ml":
        print(f"  {r['system']:9s} {r['study']:32s} {r['proxy']:26s} n={r['n']:2d} one-sided={r['n_one_sided']:2d} "
              f"MAE={r['mae']:.2f} interval={r['interval']:.2f} meas={r['meas_mean']:+.2f} pred={r['pred_mean']:+.2f}")
''')

md("""
## 5. Ullah et al. (2026) ambient PM2.5, scored only over the measured RH range

Same set-up as notebook 03 (T = 290 K, ammonium sulfate, OIR 2, 3, 8), scored at RH = 25-45 % only (the range of
the reported 10^4-10^8 Pa s band). `dist` is the distance below the band's lower edge (zero inside the band);
`mid` is the band midpoint (log10 eta = 6) minus the prediction. Proxies are run with the measured Tg (as in
notebook 03) and with the ML-predicted Tg.
""")

code(r'''
AS = Component(3, "(NH4)2SO4", ((204, 2), (261, 1)))
T_AMB, RH_SCORE, OIRS = 290.0, [0.45, 0.40, 0.35, 0.30, 0.25], [2.0, 3.0, 8.0]
LO, HI = 4.0, 8.0

def wtf_for_aw_tern(model, aw, T, f_org, lo=0.0005, hi=0.9995):
    def ev(w):
        return model.evaluate([w, (1 - w) * f_org, (1 - w) * (1 - f_org)], T, basis="mass")
    g = lambda w: float(ev(w).activity[0]) - aw
    if g(lo) > 0 or g(hi) < 0:
        return None
    return ev(brentq(g, lo, hi, xtol=1e-6))

amb = []
for pname, (smi, tg_meas) in PROXIES["aPinene"].items():
    r = smiles_to_components([smi], water_as_component1=True)
    model = ActivityModel([WATER, Component(2, pname, r.components[1].subgroups), AS])
    for tg_label, tg in (("measured", tg_meas), ("ml", predict_tg_ml_smiles(smi).tg_K)):
        eta0 = pure_organic_viscosity_vtf(T_AMB, tg)
        for oir in OIRS:
            for rule, fn in (("aquelec", aquelec_viscosity), ("aquorg", aquorg_viscosity)):
                vals = []
                for rh in RH_SCORE:
                    res = wtf_for_aw_tern(model, rh, T_AMB, oir / (oir + 1))
                    vals.append(np.nan if res is None else math.log10(fn(model, res, T_AMB, {2: eta0}).eta_pas))
                vals = np.array(vals)
                amb.append(dict(proxy=pname, tg=tg_label, tg_K=tg, oir=oir, rule=rule, rh=RH_SCORE,
                                log10_eta=vals.tolist(), mid_offset=float(np.nanmean(6.0 - vals)),
                                dist_below_band=float(np.nanmean(np.maximum(0.0, LO - vals))),
                                n_in_band=int(np.sum((vals >= LO) & (vals <= HI)))))
                a = amb[-1]
                print(f"{pname:13s} Tg={tg_label:8s}({tg:5.1f}) OIR={oir:.0f} {rule:7s} mid={a['mid_offset']:+.2f} "
                      f"dist={a['dist_below_band']:.2f} in-band={a['n_in_band']}/5  {np.round(vals, 2)}")
OUT["ullah"] = amb
''')

md("## 6. Glassy-curve summary by molar mass (disaccharides and raffinose versus smaller molecules)")

code(r'''
from rdkit import Chem
from rdkit.Chem.Descriptors import MolWt
M = {c["name"]: MolWt(Chem.MolFromSmiles(c["smiles"])) for c in curves}
groups = {}
for label, sel in (("M_lt_342", lambda n: M[n] < 341.0), ("M_ge_342", lambda n: M[n] >= 341.0)):
    s = [d for d in diag if sel(d["name"])]
    g = dict(n_curves=len(s), compounds=sorted({d["name"] for d in s}),
             n_best_at_edge_g2020=int(sum(d["best_at_edge_g2020"] for d in s)),
             w_over_x_min=float(min(d["w_eff"] / d["x_org"] for d in s)),
             w_over_x_max=float(max(d["w_eff"] / d["x_org"] for d in s)))
    for k in ("mae_g2020_pipeline", "mae_g2020_best", "mae_ideal_pipeline", "mae_ideal_best", "w_eff", "x_org"):
        g[k] = float(np.mean([d[k] for d in s]))
    groups[label] = g
    print(label, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in g.items()}))
dil_big = [(r["name"], r["source"], round(r["mae_ml"], 2)) for r in rows if M[r["name"]] >= 341.0 and r["regime"] == "dilute"]
print("dilute curves of M >= 342 compounds:", dil_big)
OUT["glassy_by_molar_mass"] = groups
OUT["dilute_high_M_curves"] = dil_big
''')

code(r'''
def clean(o):
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    return o
json.dump(clean(OUT), open("revision_results.json", "w"), indent=1)
print("wrote revision_results.json")
''')

nb["cells"] = cells
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "../notebooks/aiomfac_validation_04_revision.ipynb")
print("wrote ../notebooks/aiomfac_validation_04_revision.ipynb")

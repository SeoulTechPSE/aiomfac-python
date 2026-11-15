"""Generate notebooks/aiomfac_validation_09_oof_tg.ipynb: out-of-fold Tg and Monte Carlo Tg-error propagation.

The Tg predictor (Armeli Iapichino et al., 2023, SMILES mode without Tm) was trained on BIMOG, so its predictions for
the BIMOG compounds evaluated in paper_1 are in-sample. Here the same model (ExtraTreesRegressor, 100 trees,
max_depth 9, all 208 RDKit descriptors as candidate features, no bootstrap) is refitted in grouped 10-fold
cross-validation on the 368 BIMOG compounds, repeated with five seeds. Compounds with identical descriptor vectors
(stereoisomers) are kept in the same fold. Each compound then has an out-of-fold (OOF) Tg from a model that has not
seen it.

1. OOF Tg error over BIMOG and for the evaluated compounds.
2. Literature curves with OOF Tg (mean ± sd over the five repeats), compared with the in-sample and measured Tg.
3. Monte Carlo: Tg = centre + e, e drawn from the pooled OOF residuals (centre: measured Tg where available,
   otherwise the predicted Tg), 300 draws; distribution of the mean per-curve MAE.
4. Chamber-SOA proxies with OOF Tg.
Results: oof_tg_results.json.

Run:  python build_nb09.py && (cd ../notebooks && PATH=/opt/miniconda3/bin:$PATH PYTHONPATH=<main export>/src \
          jupyter nbconvert --to notebook --execute --inplace aiomfac_validation_09_oof_tg.ipynb \
          --ExecutePreprocessor.timeout=-1)
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md("""
# Out-of-fold Tg and Monte Carlo propagation of the Tg error

The SMILES-mode Tg predictor of Armeli Iapichino et al. (2023) is refitted in grouped 10-fold cross-validation on the
368 BIMOG compounds (five seeds; stereoisomers, i.e. identical descriptor vectors, kept in one fold), with the
hyperparameters of the published model (100 trees, max_depth 9, all features, no bootstrap). The out-of-fold (OOF)
predictions replace the in-sample predictions for every evaluated compound that is in BIMOG.
""")

code(r'''
import json, math, re, zipfile, warnings
import numpy as np
from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.Descriptors import MolWt
from rdkit.ML.Cluster import Butina
from sklearn.ensemble import ExtraTreesRegressor
from scipy.optimize import brentq
import aiomfac_py
from aiomfac_py import ActivityModel, Component
from aiomfac_py.s2as import smiles_to_components
from aiomfac_py.tgml_armeli import predict_tg_ml_smiles, _rd_descriptors
from aiomfac_py.viscosity import organic_mixture_viscosity, pure_organic_viscosity_vtf
warnings.filterwarnings("ignore"); RDLogger.DisableLog("rdApp.*")
print("aiomfac_py", aiomfac_py.__version__, aiomfac_py.__file__)

SEEDS, K, N_MC = (0, 1, 2, 3, 4), 10, 300
KEYS = ("all", "dilute", "concentrated", "conc_small", "conc_large")
bimog = json.load(open("bimog_unified.json"))
X = np.array([_rd_descriptors(c["smiles"]) for c in bimog], dtype=float)
y = np.array([c["tg_K"] for c in bimog], dtype=float)
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
print("BIMOG", X.shape)

def make_model(seed):
    return ExtraTreesRegressor(n_estimators=100, max_depth=9, max_features=1.0, bootstrap=False, random_state=seed)

# refit on all data: how close is it to the published (vendored) model?
full = make_model(0).fit(X, y)
pub = np.array([predict_tg_ml_smiles(c["smiles"]).tg_K for c in bimog])
print(f"in-sample MAE: published model {np.mean(np.abs(pub - y)):.2f} K, refit {np.mean(np.abs(full.predict(X) - y)):.2f} K; "
      f"refit vs published: mean |diff| {np.mean(np.abs(full.predict(X) - pub)):.2f} K")
''')

md("## 1. Grouped K-fold out-of-fold predictions")

code(r'''
# groups: identical descriptor vectors (e.g. stereoisomers) stay together
keys = [tuple(np.round(r, 8)) for r in X]
gid = {k: i for i, k in enumerate(dict.fromkeys(keys))}
groups = np.array([gid[k] for k in keys])
print("groups:", len(gid), "for", len(y), "compounds;", sum(np.bincount(groups) > 1), "groups with >1 compound")

oof = np.empty((len(SEEDS), len(y)))
for r, seed in enumerate(SEEDS):
    rng = np.random.default_rng(seed)
    fold_of_group = rng.permutation(len(gid)) % K
    fold = fold_of_group[groups]
    for k in range(K):
        te = fold == k
        oof[r, te] = make_model(seed).fit(X[~te], y[~te]).predict(X[te])
mae_rep = np.mean(np.abs(oof - y), axis=1)
resid = (oof - y).ravel()
print(f"OOF MAE over BIMOG: {mae_rep.mean():.2f} ± {mae_rep.std(ddof=1):.2f} K (5 repeats); "
      f"bias {resid.mean():+.2f} K; median |e| {np.median(np.abs(resid)):.2f} K; 90th pct |e| {np.percentile(np.abs(resid), 90):.1f} K")
''')

md("""
### 1b. Sensitivity: cluster split

Random folds leave structurally similar compounds (e.g. other sugars when a sugar is predicted) in the training set.
As a stricter test, BIMOG is clustered by structure (Butina clustering of Morgan fingerprints, radius 2, 2048 bits,
Tanimoto distance cutoff 0.6), and whole clusters are assigned to the ten folds.
""")

code(r'''
fps = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(c["smiles"]), 2, nBits=2048) for c in bimog]
dist = []
for i in range(1, len(fps)):
    dist.extend(1.0 - np.array(DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])))
clusters = Butina.ClusterData(dist, len(fps), 0.6, isDistData=True)
cgroup = np.empty(len(fps), dtype=int)
for ci, members in enumerate(clusters):
    cgroup[list(members)] = ci
sizes = np.bincount(cgroup)
print(f"{len(clusters)} clusters; largest {sizes.max()}; singletons {int(np.sum(sizes == 1))}")
oof_cl = np.empty((len(SEEDS), len(y)))
for r, seed in enumerate(SEEDS):
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(clusters))
    fold_of_cluster = np.empty(len(clusters), dtype=int)
    load = np.zeros(K)
    for ci in sorted(order, key=lambda c: -sizes[c]):         # balance fold sizes: largest clusters first
        k = int(np.argmin(load)); fold_of_cluster[ci] = k; load[k] += sizes[ci]
    fold = fold_of_cluster[cgroup]
    for k in range(K):
        te = fold == k
        oof_cl[r, te] = make_model(seed).fit(X[~te], y[~te]).predict(X[te])
mae_cl = np.mean(np.abs(oof_cl - y), axis=1)
print(f"cluster-split OOF MAE over BIMOG: {mae_cl.mean():.2f} ± {mae_cl.std(ddof=1):.2f} K")
''')

md("## 2. Evaluated compounds: in-sample versus OOF Tg")

code(r'''
def canon(smi, iso=True):
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(m, isomericSmiles=iso) if m else None

idx_iso = {}; idx_flat = {}
for i, c in enumerate(bimog):
    idx_iso.setdefault(canon(c["smiles"]), i); idx_flat.setdefault(canon(c["smiles"], False), []).append(i)

def bimog_index(smi):
    i = idx_iso.get(canon(smi))
    if i is not None:
        return i, "exact"
    cand = idx_flat.get(canon(smi, False), [])
    return (cand[0], "stereo-free") if len(cand) == 1 else (None, "not in BIMOG" if not cand else "ambiguous")

curves = json.load(open("literature_curves.json"))
TG = {}   # smiles -> dict(ml, oof[repeats], measured, status)
for smi in sorted({c["smiles"] for c in curves}):
    i, how = bimog_index(smi)
    ml = predict_tg_ml_smiles(smi).tg_K
    TG[smi] = dict(ml=ml, how=how, measured=float(y[i]) if i is not None else None,
                   oof=list(oof[:, i]) if i is not None else [ml] * len(SEEDS),
                   oof_cluster=list(oof_cl[:, i]) if i is not None else [ml] * len(SEEDS))
names = {c["smiles"]: c["name"] for c in curves}
print(f"{'compound':20s} {'match':12s} {'meas':>6s} {'in-sample':>9s} {'OOF (mean ± sd)':>16s}")
for smi, t in TG.items():
    meas = f"{t['measured']:.1f}" if t["measured"] else "  -  "
    print(f"{names[smi]:20s} {t['how']:12s} {meas:>6s} {t['ml']:9.1f} {np.mean(t['oof']):9.1f} ± {np.std(t['oof'], ddof=1):4.1f}")
e_in = [t["ml"] - t["measured"] for t in TG.values() if t["measured"]]
e_oof = [np.mean(t["oof"]) - t["measured"] for t in TG.values() if t["measured"]]
e_cl = [np.mean(t["oof_cluster"]) - t["measured"] for t in TG.values() if t["measured"]]
print(f"cluster split, 11 compounds: MAE {np.mean(np.abs(e_cl)):.1f} K (bias {np.mean(e_cl):+.1f} K); "
      + ", ".join(f"{names[s]} {np.mean(t['oof_cluster']) - t['measured']:+.0f}" for s, t in TG.items() if t["measured"]))
print(f"{len(e_in)} compounds with measured Tg: in-sample MAE {np.mean(np.abs(e_in)):.1f} K, "
      f"OOF MAE {np.mean(np.abs(e_oof)):.1f} K (bias {np.mean(e_oof):+.1f} K)")
''')

md("## 3. Literature curves with OOF Tg and Monte Carlo Tg errors")

code(r'''
WATER = Component(1, "Water", ((16, 1),))

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

prepared = []
for c in curves:
    model = organic_model(c["smiles"])
    pts = c["points"]
    xs = [model.evaluate([1 - p["wtf_organic"], p["wtf_organic"]], p["T_K"], basis="mass").x for p in pts]
    wtf_mean = float(np.mean([p["wtf_organic"] for p in pts]))
    prepared.append(dict(c=c, model=model, xs=xs, T=[p["T_K"] for p in pts],
                         y=np.array([p["log10_eta_measured"] for p in pts]),
                         regime="concentrated" if wtf_mean >= 0.5 else "dilute",
                         mw=MolWt(Chem.MolFromSmiles(c["smiles"]))))

def curve_mae(P, tg):
    pred = []
    for x, T in zip(P["xs"], P["T"]):
        try:
            eta0 = pure_organic_viscosity_vtf(T, tg)
        except ValueError:
            pred.append(np.nan); continue
        pred.append(log10_mix(P["model"], x, T, eta0))
    pred = np.array(pred); v = np.isfinite(pred)
    return float(np.mean(np.abs(pred[v] - P["y"][v]))) if v.any() else np.nan

def summarize(maes):
    maes = np.array(maes)
    reg = np.array([P["regime"] for P in prepared])
    mw = np.array([P["mw"] for P in prepared])
    con = reg == "concentrated"
    return dict(all=float(np.nanmean(maes)), dilute=float(np.nanmean(maes[reg == "dilute"])),
                concentrated=float(np.nanmean(maes[con])),
                conc_small=float(np.nanmean(maes[con & (mw < 341.5)])),     # M < 342 g/mol (13 curves)
                conc_large=float(np.nanmean(maes[con & (mw >= 341.5)])))    # M >= 342 g/mol (7 curves)

has_meas = np.array([TG[P["c"]["smiles"]]["measured"] is not None for P in prepared])
res = {}
res["in_sample"] = summarize([curve_mae(P, TG[P["c"]["smiles"]]["ml"]) for P in prepared])
res["measured_subset"] = dict(
    in_sample=float(np.nanmean([curve_mae(P, TG[P["c"]["smiles"]]["ml"]) for P, h in zip(prepared, has_meas) if h])),
    measured=float(np.nanmean([curve_mae(P, TG[P["c"]["smiles"]]["measured"]) for P, h in zip(prepared, has_meas) if h])),
    oof=[float(np.nanmean([curve_mae(P, TG[P["c"]["smiles"]]["oof"][r]) for P, h in zip(prepared, has_meas) if h]))
         for r in range(len(SEEDS))], n_curves=int(has_meas.sum()))
oof_runs = [summarize([curve_mae(P, TG[P["c"]["smiles"]]["oof"][r]) for P in prepared]) for r in range(len(SEEDS))]
cl_runs = [summarize([curve_mae(P, TG[P["c"]["smiles"]]["oof_cluster"][r]) for P in prepared]) for r in range(len(SEEDS))]
res["oof_cluster"] = {k: (float(np.mean([o[k] for o in cl_runs])), float(np.std([o[k] for o in cl_runs], ddof=1)))
                      for k in KEYS}
res["oof"] = {k: (float(np.mean([o[k] for o in oof_runs])), float(np.std([o[k] for o in oof_runs], ddof=1)))
              for k in KEYS}
print("in-sample Tg :", {k: round(v, 3) for k, v in res["in_sample"].items()})
print("OOF Tg       :", {k: (round(m, 3), round(s, 3)) for k, (m, s) in res["oof"].items()})
print("cluster OOF  :", {k: (round(m, 3), round(s, 3)) for k, (m, s) in res["oof_cluster"].items()})
ms = res["measured_subset"]
print(f"{ms['n_curves']} curves with measured Tg: in-sample {ms['in_sample']:.3f}, measured {ms['measured']:.3f}, "
      f"OOF {np.mean(ms['oof']):.3f} ± {np.std(ms['oof'], ddof=1):.3f}")
''')

code(r'''
centre = [TG[P["c"]["smiles"]]["measured"] or TG[P["c"]["smiles"]]["ml"] for P in prepared]
resid_cl = (oof_cl - y).ravel()
patt_oh = Chem.MolFromSmarts("[OX2H]")
n_oh = np.array([len(Chem.MolFromSmiles(c["smiles"]).GetSubstructMatches(patt_oh)) for c in bimog])
mean_cl = oof_cl.mean(axis=0) - y
bias_by_class = {"n_OH>=4 (sugars, polyols)": dict(n=int(np.sum(n_oh >= 4)), bias=float(mean_cl[n_oh >= 4].mean())),
                 "n_OH<4": dict(n=int(np.sum(n_oh < 4)), bias=float(mean_cl[n_oh < 4].mean()))}
print("cluster-split bias by class:", bias_by_class)
print(f"pooled residuals: random split bias {resid.mean():+.2f} K, MAE {np.abs(resid).mean():.2f} K; "
      f"cluster split bias {resid_cl.mean():+.2f} K, MAE {np.abs(resid_cl).mean():.2f} K")

def run_mc(pool, seed):
    rng = np.random.default_rng(seed)
    mc = []
    for d in range(N_MC):
        by_smiles = {}
        maes = []
        for P, c0 in zip(prepared, centre):
            smi = P["c"]["smiles"]
            if smi not in by_smiles:
                by_smiles[smi] = rng.choice(pool)  # the same Tg error for all curves of one compound
            maes.append(curve_mae(P, c0 + by_smiles[smi]))
        mc.append(summarize(maes))
    return {k: dict(median=float(np.median([m[k] for m in mc])), p05=float(np.percentile([m[k] for m in mc], 5)),
                    p95=float(np.percentile([m[k] for m in mc], 95)), mean=float(np.mean([m[k] for m in mc])))
            for k in KEYS}

res["monte_carlo"] = run_mc(resid, 2026)                 # random-split residuals
res["monte_carlo_cluster"] = run_mc(resid_cl, 2026)      # cluster-split residuals
base = summarize([curve_mae(P, c0) for P, c0 in zip(prepared, centre)])
res["monte_carlo_centre"] = base
print("centre (measured Tg where available):", {k: round(v, 3) for k, v in base.items()})
for k, v in res["monte_carlo_cluster"].items():
    print(f"MC (cluster) {k:13s} median {v['median']:.3f}  5-95 % {v['p05']:.3f}-{v['p95']:.3f}")
for k, v in res["monte_carlo"].items():
    print(f"MC {k:13s} median {v['median']:.3f}  5-95 % {v['p05']:.3f}-{v['p95']:.3f}")
''')

md("## 4. Chamber-SOA proxies with OOF Tg")

code(r'''
ZIP = zipfile.ZipFile("gervasi_supp.zip")
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

PROXIES = {"aPinene": {"pinonic acid": "CC(=O)C1CC(C1(C)C)CC(=O)O", "pinic acid": "CC1(C(CC1C(=O)O)CC(=O)O)C"},
           "isoprene": {"2-methyltetrol": "OCC(C)(O)C(O)CO"},
           "toluene": {"4-methylcatechol": "Cc1ccc(O)c(O)c1", "glutaric acid": "OC(=O)CCCC(=O)O",
                       "2-methylglutaric acid": "OC(=O)C(C)CCC(=O)O", "DHOPA": "CC(=O)C(O)C(O)C(=O)O",
                       "4-methyl-5-nitrocatechol": "Cc1cc(O)c(O)cc1[N+](=O)[O-]"}}
soa = []
for system, proxies in PROXIES.items():
    files = sorted(n for n in ZIP.namelist() if f"{system}_SOA" in n and n.endswith(".txt"))
    sets = [parse_soa(fn) for fn in files]
    for pname, smi in proxies.items():
        model = organic_model(smi)
        i, how = bimog_index(smi)
        ml = predict_tg_ml_smiles(smi).tg_K
        oof_tg = float(np.mean(oof[:, i])) if i is not None else ml
        oof_cl_tg = float(np.mean(oof_cl[:, i])) if i is not None else ml
        wcache = [[wtf_for_aw(model, aw, T) for aw in pts[:, 0]] for T, pts in sets]
        def mae_for(tg):
            m = []
            for (T, pts), ws in zip(sets, wcache):
                eta0 = pure_organic_viscosity_vtf(T, tg)
                pred = np.array([np.nan if w is None else
                                 log10_mix(model, model.evaluate([w, 1 - w], T, basis="mass").x, T, eta0) for w in ws])
                v = np.isfinite(pred)
                m.append(float(np.mean(np.abs(pred[v] - pts[v, 1]))))
            return float(np.mean(m))
        r = dict(system=system, proxy=pname, match=how, tg_measured=float(y[i]) if i is not None else None,
                 tg_in_sample=ml, tg_oof=oof_tg, tg_oof_cluster=oof_cl_tg,
                 tg_oof_cluster_repeats=list(oof_cl[:, i]) if i is not None else None,
                 mae_in_sample=mae_for(ml), mae_oof=mae_for(oof_tg), mae_oof_cluster=mae_for(oof_cl_tg),
                 mae_measured=mae_for(float(y[i])) if i is not None else None)
        soa.append(r)
        print(f"{system:9s} {pname:26s} {how:12s} Tg in-sample {ml:6.1f} OOF {oof_tg:6.1f} | "
              f"cluster {oof_cl_tg:6.1f} | MAE in-sample {r['mae_in_sample']:.2f} OOF {r['mae_oof']:.2f} "
              f"cluster {r['mae_oof_cluster']:.2f}" + (f" measured {r['mae_measured']:.2f}" if i is not None else ""))
res["soa"] = soa
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
OUT = dict(aiomfac_py_version=aiomfac_py.__version__, seeds=SEEDS, k=K, n_mc=N_MC,
           cluster_split=dict(n_clusters=len(clusters), largest=int(sizes.max()), singletons=int(np.sum(sizes == 1)),
                              oof_mae_bimog=dict(mean=float(mae_cl.mean()), sd=float(np.std(mae_cl, ddof=1))),
                              bias_bimog=float(resid_cl.mean()), abs_p90=float(np.percentile(np.abs(resid_cl), 90)),
                              bias_by_class=bias_by_class,
                              mae_11_compounds=float(np.mean(np.abs(e_cl))), bias_11_compounds=float(np.mean(e_cl))),
           oof_mae_bimog=dict(mean=float(mae_rep.mean()), sd=float(np.std(mae_rep, ddof=1)), per_repeat=mae_rep),
           oof_bias_bimog=float(resid.mean()), oof_abs_p90=float(np.percentile(np.abs(resid), 90)),
           compounds={names[s]: t for s, t in TG.items()}, literature=res,
           bimog_oof=[dict(name=c["name"], tg=c["tg_K"], oof_mean=float(oof[:, i].mean())) for i, c in enumerate(bimog)])
json.dump(clean(OUT), open("oof_tg_results.json", "w"), indent=1)
print("wrote oof_tg_results.json")
''')

nb["cells"] = cells
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "../notebooks/aiomfac_validation_09_oof_tg.ipynb")
print("wrote ../notebooks/aiomfac_validation_09_oof_tg.ipynb")

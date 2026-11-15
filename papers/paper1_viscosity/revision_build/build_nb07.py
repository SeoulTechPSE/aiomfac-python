"""Generate notebooks/aiomfac_validation_07_ambient_identified_samples.ipynb.

Run:  python build_nb07.py && jupyter nbconvert --to notebook --execute --inplace \
          ../notebooks/aiomfac_validation_07_ambient_identified_samples.ipynb --ExecutePreprocessor.timeout=-1
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md("""
# Ambient PM2.5: OIR dependence from points with identified samples

The data provided by the authors of Do et al. (2026a) give the sample date, and hence the OIR (Do et al., 2026a,
Table 1), of every resolved point (`../ambient_data/ambient_data.py`; the data file is not redistributed):

* Seoul and Beijing (290 K): nine measurements, each the mean over the particles of one sample at one RH (the
  measurements of Ullah et al., 2026, Supplement Table S1);
* Ansan (300 K): 29 measurements, each the mean of three particles (Do et al., 2026a, Supporting Information Table S1).

S1 (one liquid phase, aquelec) is computed at each sample's own OIR; S3 (solid AS + organic liquid) is taken from
notebook 06 (it is nearly independent of OIR). A first cell checks the provided values against the mixing times tabulated
in the two supplements. The Seoul measurement at RH 32.8 % is listed under 10/12 in Ullah et al. (2026), Supplement
Table S1; the authors confirmed that it belongs to the sample of 10/15, as in the data file (typographical error in the
table).
""")

code(r'''
import json, math, sys
import numpy as np
from scipy.optimize import brentq
import aiomfac_py
from aiomfac_py import ActivityModel, Component
from aiomfac_py.s2as import smiles_to_components
from aiomfac_py.tgml_armeli import predict_tg_ml_smiles
from aiomfac_py.viscosity import aquelec_viscosity, pure_organic_viscosity_vtf
print("aiomfac_py", aiomfac_py.__version__)
sys.path.insert(0, "../ambient_data")
from ambient_data import load
ALL = load(levels=("sample mean", "group mean"))
ROWS = [dict(p, rh=p["rh_pct"]) for p in ALL if p["level"] == "group mean"]
ANSAN = [dict(p, rh=p["rh_pct"]) for p in ALL if p["site"] == "Ansan" and not p["censored"]]
for r in ROWS:
    print(f"{r['site']:8s} {r['date']} RH={r['rh']:.1f}% n={r['n_particles']} log10 eta = {r['log10_eta']:.2f}  OIR={r['oir']}")
print("Ansan resolved points:", len(ANSAN))

# --- check against the mixing times tabulated in the supplements -------------------------------------------------
DP = 40e-6
# Ullah et al. (2026) Supplement Table S1 (site, date, RH %, tau_mix h); Eqs. S3-S5: D0 = 2.12e-9 m2/s, eta_w = 1e-3 Pa s,
# xi = 0.51
S1TAB = [("Seoul", "09/30", 41, 0.13), ("Seoul", "09/30", 33, 0.54), ("Seoul", "10/12", 39, 0.21),
         ("Seoul", "10/12", 29, 0.10), ("Seoul", "10/12", 33, 0.46), ("Beijing", "09/21", 40, 0.25),
         ("Beijing", "09/21", 30, 0.99), ("Beijing", "10/03", 38, 0.04), ("Beijing", "10/14", 32, 0.22)]
def eta_ullah(tau_h):
    D = DP ** 2 / (4 * math.pi ** 2 * tau_h * 3600.0)
    return math.log10(1e-3 * (2.12e-9 / D) ** (1.0 / 0.51))
CHECK_SB = []
for site, date, rh, tau in S1TAB:
    m = min((r for r in ROWS if r["site"] == site), key=lambda r: abs(r["rh"] - rh) + (0 if r["date"] == date else 0.5))
    CHECK_SB.append(dict(site=site, si_date=date, data_date=m["date"], rh=rh, inverted=eta_ullah(tau), provided=m["log10_eta"]))
    print(f"SI {site:8s} {date} RH {rh}: inverted {eta_ullah(tau):.2f} | provided {m['date']} RH {m['rh']:.1f}: {m['log10_eta']:.2f}")
# Do et al. (2026a) SI Eqs. S1-S4 with the constants stated by the authors: R_matrix = 0.44 nm, eta_w(300 K) = 0.840 mPa s
KB, R_H2O, ETA_W300, R_M = 1.380649e-23, 1.0e-10, 0.840e-3, 0.44
XI_A = 1 - 0.73 * math.exp(-1.79 * 0.1 / R_M)
D0_A = KB * 300.0 / (6 * math.pi * ETA_W300 * R_H2O)
def eta_do(tau_h):
    D = DP ** 2 / (4 * math.pi ** 2 * tau_h * 3600.0)
    return math.log10(ETA_W300 * (D0_A / D) ** (1.0 / XI_A))
d = np.array([eta_do(r["tau_mix_h"]) - r["log10_eta"] for r in ANSAN])
print(f"Ansan: xi = {XI_A:.3f}; eta from the provided tau_mix minus provided eta: max |diff| = {np.abs(d).max():.4f}")
CHECK_A = dict(xi=XI_A, max_abs_diff=float(np.abs(d).max()))
''')

code(r'''
WATER = Component(1, "Water", ((16, 1),))
AS = Component(3, "(NH4)2SO4", ((204, 2), (261, 1)))
PROXIES = {"pinonic acid": "CC(=O)C1CC(C1(C)C)CC(=O)O", "pinic acid": "CC1(C(CC1C(=O)O)CC(=O)O)C"}
T = 290.0
NB06 = json.load(open("ambient_phase_state_results.json"))

def s1(smiles, oir, rh):
    org = smiles_to_components([smiles], water_as_component1=True).components[1]
    model = ActivityModel([WATER, Component(2, "org", org.subgroups), AS])
    eta0 = pure_organic_viscosity_vtf(T, predict_tg_ml_smiles(smiles).tg_K)
    f = oir / (oir + 1)
    fr = lambda w: [w, (1 - w) * f, (1 - w) * (1 - f)]
    w = brentq(lambda w: float(model.evaluate(fr(w), T, basis="mass").activity[0]) - rh, 1e-4, 0.9999, xtol=1e-9)
    res = model.evaluate(fr(w), T, basis="mass")
    return math.log10(aquelec_viscosity(model, res, T, {2: eta0}).eta_pas)

def s3(site, proxy, rh):
    g = [r for r in NB06["grid"] if r["site"] == site and r["proxy"] == proxy and r["oir_label"] == "oir_mean"]
    return float(np.interp(rh, [r["rh"] * 100 for r in g], [r["s3"]["log10_eta"] for r in g]))

for r in ROWS:
    for p, smi in PROXIES.items():
        r[f"s1_{p}"] = s1(smi, r["oir"], r["rh"] / 100)
        r[f"s3_{p}"] = s3(r["site"], p, r["rh"])
print(f"{'site':8s} {'date':5s} {'RH':>5s} {'OIR':>5s} {'meas':>6s} | {'S1 pinonic':>10s} {'S1 pinic':>9s} | {'S3 pinonic':>10s} {'S3 pinic':>9s}")
for r in ROWS:
    print(f"{r['site']:8s} {r['date']:5s} {r['rh']:5.1f} {r['oir']:5.2f} {r['log10_eta']:6.2f} | {r['s1_pinonic acid']:10.2f} "
          f"{r['s1_pinic acid']:9.2f} | {r['s3_pinonic acid']:10.2f} {r['s3_pinic acid']:9.2f}")

SUM = {}
for p in PROXIES:
    for key in ("s1", "s3"):
        off = np.array([r["log10_eta"] - r[f"{key}_{p}"] for r in ROWS])
        SUM[f"{key}_{p}"] = dict(mean=float(off.mean()), sd=float(off.std()), n_within_1=int(np.sum(np.abs(off) <= 1)))
        print(f"{key} {p:13s}: offset {off.mean():+.2f} ± {off.std():.2f}, within 1: {np.sum(np.abs(off) <= 1)}/{len(off)}")
''')

md("""
## OIR dependence at nearly the same RH

Pairs of measurements of different samples at similar RH: the measured difference versus the difference predicted by
S1 (which depends on OIR) and by S3 (which does not).
""")

code(r'''
PAIRS = [(("Beijing", "09/21", 40), ("Beijing", "10/03", 38)), (("Beijing", "09/21", 30), ("Beijing", "10/14", 32)),
         (("Seoul", "09/30", 42), ("Seoul", "10/12", 39)), (("Seoul", "09/30", 33), ("Seoul", "10/15", 33))]
find = lambda k: min((r for r in ROWS if (r["site"], r["date"]) == k[:2]), key=lambda r: abs(r["rh"] - k[2]))
PAIR_OUT = []
for a, b in PAIRS:
    ra, rb = find(a), find(b)
    d = dict(pair=f"{a[0]} {a[1]} (OIR {ra['oir']}, RH {ra['rh']:.1f}) - {b[1]} (OIR {rb['oir']}, RH {rb['rh']:.1f})",
             measured=ra["log10_eta"] - rb["log10_eta"],
             s1_pinic=ra["s1_pinic acid"] - rb["s1_pinic acid"], s3_pinic=ra["s3_pinic acid"] - rb["s3_pinic acid"])
    PAIR_OUT.append(d)
    print(f"{d['pair']:60s} measured {d['measured']:+.2f}  S1 {d['s1_pinic']:+.2f}  S3 {d['s3_pinic']:+.2f}")
json.dump(dict(points=ROWS, summary=SUM, pairs=PAIR_OUT, check_sb=CHECK_SB, check_ansan=CHECK_A),
          open("ambient_identified_samples_results.json", "w"), indent=1)
print("wrote ambient_identified_samples_results.json")
''')

md("""
## Ansan: 29 resolved points with identified samples
""")

code(r'''
T_A = 300.0
def s1_T(smiles, oir, rh, T):
    org = smiles_to_components([smiles], water_as_component1=True).components[1]
    model = ActivityModel([WATER, Component(2, "org", org.subgroups), AS])
    eta0 = pure_organic_viscosity_vtf(T, predict_tg_ml_smiles(smiles).tg_K)
    f = oir / (oir + 1); fr = lambda w: [w, (1 - w) * f, (1 - w) * (1 - f)]
    w = brentq(lambda w: float(model.evaluate(fr(w), T, basis="mass").activity[0]) - rh, 1e-4, 0.9999, xtol=1e-9)
    return math.log10(aquelec_viscosity(model, model.evaluate(fr(w), T, basis="mass"), T, {2: eta0}).eta_pas)

for r in ANSAN:
    for p, smi in PROXIES.items():
        r[f"s1_{p}"] = s1_T(smi, r["oir"], r["rh"] / 100, T_A)
        r[f"s3_{p}"] = s3("Ansan", p, r["rh"])

ANSAN_SUM = {}
y = np.array([r["log10_eta"] for r in ANSAN]); oir = np.array([r["oir"] for r in ANSAN]); rh = np.array([r["rh"] for r in ANSAN])
for p in PROXIES:
    for key in ("s1", "s3"):
        pred = np.array([r[f"{key}_{p}"] for r in ANSAN]); off = y - pred
        # after removing the mean offset, how much of the measured scatter does each scenario explain?
        resid = off - off.mean()
        r_oir = float(np.corrcoef(off, oir)[0, 1])
        ANSAN_SUM[f"{key}_{p}"] = dict(mean=float(off.mean()), sd=float(off.std()), corr_offset_oir=r_oir,
                                       n_within_1=int(np.sum(np.abs(off) <= 1)))
        print(f"Ansan {key} {p:13s}: offset {off.mean():+.2f} ± {off.std():.2f}  corr(offset, OIR)={r_oir:+.2f}  "
              f"within 1: {np.sum(np.abs(off) <= 1)}/{len(off)}")

# multiple regression of the measured log10 eta on RH and OIR, compared with the same regression on each scenario
X = np.c_[np.ones_like(rh), rh, oir]
def fit(v):
    c, *_ = np.linalg.lstsq(X, v, rcond=None); res = v - X @ c
    return c, float(1 - res.var() / v.var())
c_m, r2_m = fit(y)
print(f"measured: log10 eta = {c_m[0]:.2f} {c_m[1]:+.3f} RH {c_m[2]:+.2f} OIR   (R2 = {r2_m:.2f})")
REG = dict(measured=dict(coef=c_m.tolist(), r2=r2_m))
for p in PROXIES:
    for key in ("s1", "s3"):
        c, r2 = fit(np.array([r[f"{key}_{p}"] for r in ANSAN]))
        REG[f"{key}_{p}"] = dict(coef=c.tolist(), r2=r2)
        print(f"{key} {p:13s}: log10 eta = {c[0]:.2f} {c[1]:+.3f} RH {c[2]:+.2f} OIR")
# bootstrap the OIR slope of the measured data (resampling the 10 samples)
rng = np.random.default_rng(0); dates = sorted({r["date"] for r in ANSAN}); slopes = []
for _ in range(2000):
    pick = rng.choice(dates, len(dates)); idx = [i for d in pick for i, r in enumerate(ANSAN) if r["date"] == d]
    c, *_ = np.linalg.lstsq(X[idx], y[idx], rcond=None); slopes.append(c[2])
lo, hi = np.percentile(slopes, [2.5, 97.5])
REG["measured_oir_slope_ci95"] = [float(lo), float(hi)]
print(f"measured OIR slope 95% CI (bootstrap over samples): [{lo:+.2f}, {hi:+.2f}]")

json.dump(dict(points=ROWS, summary=SUM, pairs=PAIR_OUT, check_sb=CHECK_SB, check_ansan=CHECK_A, ansan_points=ANSAN,
               ansan_summary=ANSAN_SUM, ansan_regression=REG),
          open("ambient_identified_samples_results.json", "w"), indent=1)
print("wrote ambient_identified_samples_results.json (with Ansan)")
''')

nb["cells"] = cells
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nb.metadata["language_info"] = {"name": "python"}
nbf.write(nb, "../notebooks/aiomfac_validation_07_ambient_identified_samples.ipynb")
print("wrote ../notebooks/aiomfac_validation_07_ambient_identified_samples.ipynb")

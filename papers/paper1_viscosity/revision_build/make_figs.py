"""Figures for the revised paper_1 manuscript. Run from paper_1/notebooks with the analysis environment:
    python ../revision_build/make_figs.py
Reads literature_curves.json, gervasi_supp.zip and revision_results.json; writes ../figures_revised/*.png and
prints every number shown in a figure."""
import io, json, math, os, re, zipfile, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from aiomfac_py import ActivityModel, Component
from aiomfac_py.s2as import smiles_to_components
from aiomfac_py.tgml_armeli import predict_tg_ml_smiles
from aiomfac_py.viscosity import organic_mixture_viscosity, pure_organic_viscosity_vtf, water_viscosity_pas
from scipy.optimize import brentq

warnings.filterwarnings("ignore")
OUTDIR = "../figures_revised"
os.makedirs(OUTDIR, exist_ok=True)
plt.rcParams.update({"font.size": 10, "font.family": "sans-serif", "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150, "savefig.dpi": 300})
BLUE, ORANGE, GREEN, GREY = "#2b6cb0", "#c05621", "#2f855a", "#4a5568"
WATER = Component(1, "Water", ((16, 1),))
curves = json.load(open("literature_curves.json"))
R = json.load(open("revision_results.json"))
ZIP = zipfile.ZipFile("gervasi_supp.zip")
LOG10 = r"log$_{10}$($\eta$ / Pa s)"


def model_for(smiles):
    r = smiles_to_components([smiles], water_as_component1=True)
    return ActivityModel([WATER, Component(2, "o", r.components[1].subgroups)])


def mix(model, x, T, eta0, rule="g2020"):
    if rule == "ideal":
        return (x[0] * math.log(water_viscosity_pas(T)) + x[1] * math.log(eta0)) / math.log(10)
    return organic_mixture_viscosity(model, x, T, {2: eta0}).log10_eta_pas


tg_cache = {}
def tg_ml(smiles):
    if smiles not in tg_cache:
        tg_cache[smiles] = predict_tg_ml_smiles(smiles).tg_K
    return tg_cache[smiles]


# ---------- Fig. 1: Tg predictor vs measured (in-sample and out-of-fold) ----------
OOF = json.load(open("oof_tg_results.json"))["compounds"]
pairs = {n: (c["measured"], c["ml"], float(np.mean(c["oof"]))) for n, c in OOF.items() if c["measured"]}
meas = np.array([v[0] for v in pairs.values()])
pred = np.array([v[1] for v in pairs.values()])
poof = np.array([v[2] for v in pairs.values()])
fig, ax = plt.subplots(figsize=(3.6, 3.4))
ax.scatter(meas, pred, s=30, facecolors="none", edgecolors=BLUE, zorder=3, label="in-sample")
ax.scatter(meas, poof, s=26, color=ORANGE, zorder=4, label="out-of-fold")
for n, (m, p, q) in pairs.items():
    if abs(q - m) > 10:
        ax.annotate(n.replace("_", " "), (m, q), xytext=(4, -10), textcoords="offset points", fontsize=8)
lim = [170, 400]
ax.plot(lim, lim, "--", color=GREY, lw=1)
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("measured $T_g$ (K)"); ax.set_ylabel("predicted $T_g$ (K)")
ax.legend(frameon=False, fontsize=8, loc="upper left")
fig.tight_layout(); fig.savefig(f"{OUTDIR}/fig1_tg.png"); plt.close(fig)
print(f"Fig1 OOF: MAE={np.mean(np.abs(poof - meas)):.2f} K bias={np.mean(poof - meas):+.2f} K")
print(f"Fig1: n={len(pairs)} MAE={np.mean(np.abs(pred - meas)):.2f} K bias={np.mean(pred - meas):+.2f} K")

# ---------- Fig. 2: predicted vs measured, all literature points ----------
pts_by_regime = {"dilute": [], "glassy": []}
curve_pred = {}
for c in curves:
    model = model_for(c["smiles"]); tg = tg_ml(c["smiles"])
    wmean = np.mean([p["wtf_organic"] for p in c["points"]])
    reg = "glassy" if wmean >= 0.5 else "dilute"
    ys = []
    for p in c["points"]:
        x = model.evaluate([1 - p["wtf_organic"], p["wtf_organic"]], p["T_K"], basis="mass").x
        y = mix(model, x, p["T_K"], pure_organic_viscosity_vtf(p["T_K"], tg))
        ys.append(y); pts_by_regime[reg].append((p["log10_eta_measured"], y))
    curve_pred[(c["name"], c["source"], c["points"][0]["T_K"])] = (c, np.array(ys))
allp = np.array(pts_by_regime["dilute"] + pts_by_regime["glassy"])
fig, ax = plt.subplots(figsize=(4.2, 4.0))
for reg, col, lab in (("dilute", BLUE, r"dilute (mean $w_{org}$ < 0.5)"),
                      ("glassy", ORANGE, r"concentrated (mean $w_{org}$ ≥ 0.5)")):
    a = np.array(pts_by_regime[reg])
    ax.scatter(a[:, 0], a[:, 1], s=9, alpha=0.6, color=col, lw=0, label=f"{lab}, n = {len(a)}")
lim = [-3.6, 14]
ax.plot(lim, lim, "--", color=GREY, lw=1, label="1:1")
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel(f"measured {LOG10}"); ax.set_ylabel(f"predicted {LOG10}")
ax.legend(frameon=False, fontsize=7.5, loc="upper left")
fig.tight_layout(); fig.savefig(f"{OUTDIR}/fig2_parity.png"); plt.close(fig)
print(f"Fig2: n_points={len(allp)} pooled point MAE={np.mean(np.abs(allp[:, 1] - allp[:, 0])):.3f} "
      f"max pred={allp[:, 1].max():.2f}")

# ---------- Fig. 3: example curves, G2020 vs ideal at the pipeline eta0 ----------
examples = [("glycerol", "CRCHandbook", 293.15), ("sucrose", "CRCHandbook", 293.15),
            ("citric_acid", "Song2016", 293.15), ("trehalose", "Song2016", 293.0)]
labels = {"glycerol": "glycerol", "sucrose": "sucrose", "citric_acid": "citric acid", "trehalose": "trehalose"}
srcl = {"CRCHandbook": "CRC Handbook", "Song2016": "Song et al. (2016)"}
fig, axes = plt.subplots(2, 2, figsize=(6.6, 5.4))
for ax, key, tag in zip(axes.flat, examples, "abcd"):
    c, _ = curve_pred[key]
    model = model_for(c["smiles"]); T = key[2]; tg = tg_ml(c["smiles"]); eta0 = pure_organic_viscosity_vtf(T, tg)
    w = np.array([p["wtf_organic"] for p in c["points"]]); y = np.array([p["log10_eta_measured"] for p in c["points"]])
    grid = np.linspace(w.min(), w.max(), 60)
    xs = [model.evaluate([1 - g, g], T, basis="mass").x for g in grid]
    ax.plot(grid, [mix(model, x, T, eta0) for x in xs], color=BLUE, lw=2, label="G2020 (pipeline)")
    ax.plot(grid, [mix(model, x, T, eta0, "ideal") for x in xs], "--", color=GREEN, lw=1.5, label="ideal mixing")
    ax.scatter(w, y, s=14, color="k", zorder=3, label="measured")
    ax.set_title(f"({tag}) {labels[key[0]]}, {srcl[key[1]]}", fontsize=9, loc="left")
    ax.set_xlabel("organic mass fraction"); ax.set_ylabel(LOG10)
    print(f"Fig3 {key}: Tg_ML={tg:.1f} log10 eta0={math.log10(eta0):.2f}")
axes[0, 0].legend(frameon=False, fontsize=7.5)
fig.tight_layout(); fig.savefig(f"{OUTDIR}/fig3_examples.png"); plt.close(fig)

# ---------- Fig. 4: effective weight of eta0 in G2020 ----------
g = R["glassy_diagnosis"]
mm = {}
from rdkit import Chem
from rdkit.Chem.Descriptors import MolWt
for c in curves:
    mm[c["name"]] = MolWt(Chem.MolFromSmiles(c["smiles"]))
fig, (ax, bx) = plt.subplots(1, 2, figsize=(8.0, 3.6), gridspec_kw={"width_ratios": [1.15, 1]})
xo = np.array([d["x_org"] for d in g]); we = np.array([d["w_eff"] for d in g]); M = np.array([mm[d["name"]] for d in g])
sc = ax.scatter(xo, we, c=M, cmap="viridis", s=34, zorder=3, edgecolor="k", lw=0.4)
ax.plot([0, 0.9], [0, 0.9], "--", color=GREY, lw=1, label="$w_{eff}$ = $x_{org}$ (ideal mixing)")
cb = fig.colorbar(sc, ax=ax); cb.set_label("molar mass (g mol$^{-1}$)")
ax.set_xlabel("mean organic mole fraction $x_{org}$")
ax.set_ylabel(r"$w_{eff}$ = d log$_{10}\eta$ / d log$_{10}\eta_0$ = $\gamma^C_{org} x_{org}$")
ax.legend(frameon=False, fontsize=8, loc="upper left")
ax.set_title("(a)", loc="left", fontsize=10)
# (b) combinatorial activity coefficient of the organic versus its mole fraction
from aiomfac_py.sr import sr_terms
REP = [("glycerol", "OCC(O)CO"), ("glucose", "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O"),
       ("sucrose", "OC[C@H]1O[C@@](CO)(O[C@H]2O[C@H](CO)[C@@H](O)[C@H](O)[C@H]2O)[C@@H](O)[C@@H]1O"),
       ("raffinose", "OC[C@H]1O[C@@](CO)(O[C@H]2O[C@H](CO[C@H]3O[C@H](CO)[C@H](O)[C@H](O)[C@H]3O)[C@@H](O)[C@H](O)[C@H]2O)[C@@H](O)[C@@H]1O")]
xg = np.logspace(-3, 0, 120)
cols = plt.cm.viridis(np.linspace(0.05, 0.9, len(REP)))
for (name, smi), col in zip(REP, cols):
    sub = smiles_to_components([smi], water_as_component1=True).components[1].subgroups
    mdl = ActivityModel([WATER, Component(2, name, sub)])
    gc = [math.exp(sr_terms(mdl.mixture.sr, 293.15, np.array([1 - x, x])).ln_gamma_c[1]) for x in xg]
    bx.plot(xg, gc, color=col, lw=1.6, label=f"{name} (r = {mdl.mixture.sr.RS[1]:.1f})")
    print(f"Fig4b {name}: r={mdl.mixture.sr.RS[1]:.2f} gC(x=0.001)={gc[0]:.3f} gC(x=0.1)={np.interp(0.1, xg, gc):.3f}")
bx.set_xscale("log"); bx.set_yscale("log"); bx.set_ylim(5e-3, 1.5)
bx.set_xlabel("organic mole fraction $x_{org}$"); bx.set_ylabel(r"$\gamma^C_{org}$ (= $w_{eff}$ / $x_{org}$)")
bx.legend(frameon=False, fontsize=8, loc="lower right"); bx.set_title("(b)", loc="left", fontsize=10)
fig.tight_layout(); fig.savefig(f"{OUTDIR}/fig4_weight.png"); plt.close(fig)
print("Fig4: w_eff/x_org range", f"{(we / xo).min():.3f}-{(we / xo).max():.3f}",
      "M>=342:", sorted(set(round(m) for m in M if m >= 342)))

# ---------- Fig. 5: alpha-pinene SOA ----------
def parse_soa(name):
    lines = [l.strip().strip('"') for l in ZIP.read(name).decode("utf-8-sig").splitlines()]
    T = float(re.search(r"T\s*=\s*([\d.]+)\s*K", lines[0]).group(1)); pts = []
    for line in lines[4:]:
        p = [s for s in re.split(r"[,\t]", line) if s.strip()]
        if len(p) >= 6:
            rh, eta, _, _, yn, yp = map(float, p[:6])
            if eta > 0 and 0 < rh < 100:
                pts.append((rh, math.log10(eta), yn, yp))
    return T, np.array(pts)

def soa_curve(smiles, tg, T, rh):
    model = model_for(smiles); eta0 = pure_organic_viscosity_vtf(T, tg); out = []
    for aw in rh / 100:
        f = lambda w: float(model.evaluate([w, 1 - w], T, basis="mass").activity[0]) - aw
        if f(0.0005) > 0 or f(0.9995) < 0:
            out.append(np.nan); continue
        w = brentq(f, 0.0005, 0.9995, xtol=1e-6)
        out.append(mix(model, model.evaluate([w, 1 - w], T, basis="mass").x, T, eta0))
    return np.array(out)

files = sorted(n for n in ZIP.namelist() if "aPinene_SOA" in n and n.endswith(".txt"))
fig, ax = plt.subplots(figsize=(6.0, 4.3))
cmap = matplotlib.colormaps["tab10"]
for i, fn in enumerate(files):
    T, p = parse_soa(fn)
    lab = fn.split("_SOA_")[1].rsplit(".txt", 1)[0].split("_", 1)[1].replace("@", ", ").replace("ugm-3", " µg m$^{-3}$")
    lab = re.sub(r"([A-Za-z-]+)(\d{4})", r"\1 \2", lab)
    ax.errorbar(p[:, 0], p[:, 1], yerr=[p[:, 2], p[:, 3]], fmt="o", ms=4, color=cmap(i), elinewidth=0.7,
                capsize=1.5, alpha=0.85, label=lab)
rh = np.linspace(2, 95, 50)
for name, smi, col in (("pinonic acid", "CC(=O)C1CC(C1(C)C)CC(=O)O", BLUE), ("pinic acid", "CC1(C(CC1C(=O)O)CC(=O)O)C", GREEN)):
    tg = tg_ml(smi)
    ax.plot(rh, soa_curve(smi, tg, 294.0, rh), lw=2.2, color=col, label=f"{name} ($T_g$ = {tg:.0f} K)")
    print(f"Fig5: {name} Tg_ML={tg:.1f}")
ax.set_xlabel("relative humidity (%)"); ax.set_ylabel(LOG10)
ax.legend(frameon=False, fontsize=6.8, ncol=2, loc="upper right")
fig.tight_layout(); fig.savefig(f"{OUTDIR}/fig5_apinene.png"); plt.close(fig)
print("done")

# ---------- Fig. 6: ambient PM2.5, three phase-state scenarios vs measured data ----------
import sys
sys.path.insert(0, "../ambient_data")
from ambient_data import load as load_ambient
A = json.load(open("ambient_phase_state_results.json"))
pts = load_ambient()
fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.6), sharey=True)
for ax, site, tag in zip(axes, ("Ansan", "Seoul", "Beijing"), "abc"):
    def ser(lab, key):
        rows = [g for g in A["grid"] if g["site"] == site and g["proxy"] == "pinic acid" and g["oir_label"] == lab]
        rh = np.array([g["rh"] * 100 for g in rows])
        if key == "s1":
            y = [g["s1"] for g in rows]
        elif key == "s2":
            y = [g["s2"].get("log10_eta_org_phase") if abs(g["s2"].get("aw_check", 9) - g["rh"]) <= 2e-3 else None for g in rows]
        else:
            y = [g["s3"].get("log10_eta") for g in rows]
        return rh, np.array([np.nan if v is None else v for v in y], float)
    rh, lo = ser("oir_min", "s1"); _, hi = ser("oir_max", "s1"); _, mid = ser("oir_mean", "s1")
    ax.fill_between(rh, lo, hi, color=GREY, alpha=0.18, lw=0)
    ax.plot(rh, mid, color=GREY, lw=1.8, label="S1: one liquid phase")
    r2, y2 = ser("oir_mean", "s2")
    nph = np.array([g["s2"].get("n_phases") == 2 for g in A["grid"]
                    if g["site"] == site and g["proxy"] == "pinic acid" and g["oir_label"] == "oir_mean"])
    ok = np.isfinite(y2)
    ax.plot(r2[ok], y2[ok], "--", color=BLUE, lw=1.8, label="S2: no crystallization (organic-rich liquid)")
    r3, y3 = ser("oir_mean", "s3")
    ax.plot(r3, y3, color=GREEN, lw=2.2, label="S3: equilibrium (organic-rich liquid)")
    P = [p for p in pts if p["site"] == site]
    res = [p for p in P if not p["censored"]]; cen = [p for p in P if p["censored"]]
    yv = np.array([p["log10_eta"] for p in res])
    yerr = np.array([[p["log10_eta"] - p["log10_lo"] for p in res], [p["log10_hi"] - p["log10_eta"] for p in res]])
    ax.errorbar([p["rh_pct"] for p in res], yv, yerr=yerr, fmt="none", ecolor=ORANGE, elinewidth=0.6, alpha=0.6, zorder=3)
    ax.scatter([p["rh_pct"] for p in res], yv, s=16, color=ORANGE, zorder=4, edgecolor="k", lw=0.3,
               label="measured" + (" (sample means)" if site == "Ansan" else " (particles)"))
    ax.scatter([p["rh_pct"] for p in cen], [p["log10_eta"] for p in cen], s=26, marker="^", color=ORANGE,
               zorder=4, edgecolor="k", lw=0.3, label="lower bound")
    T = A["sle"][site]["T_K"]
    ax.set_title(f"({tag}) {site}, {T:.0f} K, OIR {A['grid'][0]['oir'] if False else ''}".rstrip(", OIR "), fontsize=9, loc="left")
    ax.set_xlim(0, 50); ax.set_ylim(-2, 10); ax.set_xlabel("relative humidity (%)")
axes[0].set_ylabel(LOG10)
axes[2].legend(frameon=False, fontsize=6.6, loc="lower left")
fig.tight_layout(); fig.savefig(f"{OUTDIR}/fig6_ambient.png"); plt.close(fig)
for s in A["scores"]:
    if s["oir_label"] == "oir_mean":
        print(f"Fig6/Table: {s['site']:8s} {s['proxy']:13s} {s['scenario']} {s['mean_offset']:+.2f} ± {s['sd']:.2f} "
              f"within1 {s['n_within_1']}/{s['n']} missing {s['n_grid_missing']}")

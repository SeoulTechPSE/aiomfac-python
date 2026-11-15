"""Fig. 7 of the revised paper_1 manuscript: alpha-pinene SOA with the monomer-dimer surrogate (notebook 13).
Run from paper_1/notebooks:  python ../revision_build/make_fig7.py
Reads oligomer_surrogate_results.json and gervasi_supp.zip; writes ../figures_revised/fig7_oligomers.png."""
import json, math, re, zipfile
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 10, "font.family": "sans-serif", "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150, "savefig.dpi": 300})
LOG10 = r"log$_{10}$($\eta$ / Pa s)"
O = json.load(open("oligomer_surrogate_results.json"))
ZIP = zipfile.ZipFile("gervasi_supp.zip")
DIMER, T = "pinonyl-pinyl ester", "294.0"
rh = np.array(O["rh_grid"]) * 100


def parse_soa(name):
    lines = [l.strip().strip('"') for l in ZIP.read(name).decode("utf-8-sig").splitlines()]
    pts = []
    for line in lines[4:]:
        p = [s for s in re.split(r"[,\t]", line) if s.strip()]
        if len(p) >= 6:
            r, eta, _, _, yn, yp = map(float, p[:6])
            if eta > 0 and 0 < r < 100:
                pts.append((r, math.log10(eta), yn, yp))
    return np.array(pts)


files = sorted(n for n in ZIP.namelist() if "aPinene_SOA" in n and n.endswith(".txt"))
fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.9), sharey=True)
ramp = matplotlib.colormaps["viridis"]
for ax, mixing, title in zip(axes, ("g2020", "mole_fraction"), ("(a) G2020", "(b) mole-fraction rule")):
    for k, fn in enumerate(files):
        p = parse_soa(fn)
        ax.errorbar(p[:, 0], p[:, 1], yerr=[p[:, 2], p[:, 3]], fmt="o", ms=3.2, color="0.55", elinewidth=0.6,
                    capsize=1.2, alpha=0.8, zorder=2, label="α-pinene SOA (nine data sets)" if k == 0 else None)
    for j, f_d in enumerate((0.0, 0.1, 0.3, 0.5)):
        key = f"{'pinyl-diaterpenyl ester' if f_d == 0 else DIMER}|tg_ml|{mixing}|{f_d}"
        ax.plot(rh, np.array(O["curves"][key][T], float), lw=2.0, color=ramp(j / 3.4), zorder=3,
                label=f"$f_d$ = {f_d:.1f}" + (" (monomers only)" if f_d == 0 else ""))
    y = np.array(O["curves"][f"{DIMER}|tg_ml|{mixing}|f_d(RH)"][T], float)
    ax.plot(rh, y, "--", lw=1.8, color="#c05621", zorder=3, label="$f_d$ rising as the particle dries")
    ax.axhline(O["tau_1h_200nm_log10"], color="k", lw=0.7, ls=":", zorder=1)
    ax.text(97, O["tau_1h_200nm_log10"] + 0.25, "mixing time 1 h (200 nm)", ha="right", fontsize=7)
    ax.set_title(title, fontsize=9, loc="left"); ax.set_xlabel("relative humidity (%)")
    ax.set_xlim(0, 100); ax.set_ylim(-3, 12)
axes[0].set_ylabel(LOG10)
axes[1].legend(frameon=False, fontsize=6.8, loc="upper right")
fig.tight_layout(); fig.savefig("../figures_revised/fig7_oligomers.png"); plt.close(fig)
print("wrote ../figures_revised/fig7_oligomers.png")

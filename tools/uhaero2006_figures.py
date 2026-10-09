"""Reproduce the inorganic phase diagrams of Amundson et al. (2006, Atmos. Chem. Phys. 6, 975, UHAERO) with AIOMFAC:
(NH4)2SO4 / H2SO4 / NH4NO3 / HNO3 / H2O at 298.15 K, closed system (no gas phase).

  X = NH4+ / (NH4+ + H+)  (ammonium fraction),   Y = SO4 / (SO4 + NO3)  (sulfate fraction)
  feed per mol cation: NH4+ = X, H+ = 1 - X, SO4-- = Y/(1+Y), NO3- = (1-Y)/(1+Y)

Solids (letters of the paper): A (NH4)2SO4, B (NH4)3H(SO4)2, C NH4HSO4, D NH4NO3, E (NH4)2SO4.2NH4NO3,
F (NH4)2SO4.3NH4NO3, G NH4HSO4.NH4NO3.

  python tools/uhaero2006_figures.py fig1  [out_dir]        Fig. 1: DRH contours and first solid over (X, Y)
  python tools/uhaero2006_figures.py map Y [out_dir]        Figs. 2, 4, 6, 8, 10: X-RH diagram with pH (a) and
                                                            relative particle mass (b) contours at sulfate fraction Y
  python tools/uhaero2006_figures.py curves Y [out_dir]     Figs. 3, 5, 7, 9, 11: relative particle mass vs RH
  python tools/uhaero2006_figures.py plot [out_dir]         draw all figures from the cached results

The activity model differs from the paper's (AIOMFAC instead of the Pitzer-Simonson-Clegg model), so boundaries
differ quantitatively; the double-salt constants are taken relative to the simple salts (see aiomfac_py.solids)."""
import os
import pickle
import sys
import time
import warnings

import numpy as np

from aiomfac_py.diagram import deliquescence_point, phase_map, rh_profile
from aiomfac_py.phase_equilibrium import PhaseEquilibrium

warnings.filterwarnings("ignore")
T = 298.15
LETTERS = {"ammonium_sulfate": "A", "letovicite": "B", "ammonium_bisulfate": "C", "ammonium_nitrate": "D",
           "AS_2AN": "E", "AS_3AN": "F", "AHS_AN": "G"}
CURVES_X = {1.0: [0.9, 0.73, 0.6, 0.4, 0.3, 0.1], 0.85: [0.98, 0.9, 0.77, 0.74, 0.7, 0.6],
            0.5: [0.95, 0.9, 0.86, 0.84, 0.8, 0.77, 0.75, 0.7], 0.3: [0.98, 0.95, 0.9, 0.87, 0.85, 0.8, 0.77, 0.7],
            0.2: [0.98, 0.93, 0.9, 0.85, 0.83, 0.7, 0.6, 0.3]}
FIG_NO = {1.0: (2, 3), 0.85: (4, 5), 0.5: (6, 7), 0.3: (8, 9), 0.2: (10, 11)}
# axis ranges of the deliquescence-curve panels in the paper: ((RH% a), (mass a), (RH% b), (mass b))
CURVE_AXES = {1.0: ((35, 80), (1, 2.4), (0, 35), (1, 1.8)), 0.85: ((35, 80), (1, 2.4), (25, 70), (1, 1.8)),
              0.5: ((40, 75), (1, 2.0), (35, 65), (1, 1.6)), 0.3: ((35, 75), (1, 2.0), (0, 60), (1, 1.6)),
              0.2: ((30, 70), (1, 1.8), (0, 55), (1, 1.6))}


def feed(X, Y):
    f = {"NH4+": X, "H+": 1.0 - X, "SO4--": Y / (1.0 + Y), "NO3-": (1.0 - Y) / (1.0 + Y)}
    return {k: v for k, v in f.items() if v > 1e-12}


def system(Y):
    ions = ["NH4+", "H+", "SO4--"] + (["NO3-"] if Y < 1.0 else [])
    return PhaseEquilibrium([], ions, T_K=T)


def run_fig1(out):
    pe = PhaseEquilibrium([], ["NH4+", "H+", "SO4--", "NO3-"], T_K=T)
    xs = np.linspace(0.05, 1.0, 20)
    ys = np.linspace(0.0, 1.0, 11)
    drh = np.full((len(ys), len(xs)), np.nan)
    first = np.empty((len(ys), len(xs)), dtype=object)
    t0 = time.time()
    for i, Y in enumerate(ys):
        for j, X in enumerate(xs):
            r, s = deliquescence_point(pe, feed(X, Y), step=0.05, method="si")
            drh[i, j] = np.nan if r is None else r
            first[i, j] = s
        print(f"Y = {Y:.2f} done ({time.time() - t0:.0f} s)", flush=True)
    pickle.dump({"x": xs, "y": ys, "drh": drh, "first": first}, open(os.path.join(out, "fig1.pkl"), "wb"))


def run_map(Y, out):
    pe = system(Y)
    t0 = time.time()
    xs = np.linspace(0.0, 1.0, 11)
    xs[0] = 0.01
    pm = phase_map(pe, lambda X: feed(X, Y), xs, x_label="Ammonium fraction X", n=24, rh_min=0.02, rh_max=0.98,
                   x_tol=0.02, verbose=True)
    rg = np.linspace(0.02, 0.98, 33)
    fx = np.linspace(0.0, 1.0, 15)
    fx[0] = 0.01
    fields = {"x": fx, "rh": rg, "pH": np.full((len(rg), len(fx)), np.nan), "rel_mass": np.full((len(rg), len(fx)), np.nan)}
    for j, X in enumerate(fx):
        p = rh_profile(pe, feed(X, Y), rg)
        fields["pH"][:, j] = p["pH"]
        fields["rel_mass"][:, j] = p["rel_mass"]
    print(f"Y = {Y}: {time.time() - t0:.0f} s", flush=True)
    pickle.dump({"pm": pm, "fields": fields}, open(os.path.join(out, f"map_Y{Y:.2f}.pkl"), "wb"))


def run_curves(Y, out):
    pe = system(Y)
    rg = np.round(np.arange(0.02, 0.9801, 0.01), 4)
    res = {X: rh_profile(pe, feed(X, Y), rg) for X in CURVES_X[Y]}
    pickle.dump(res, open(os.path.join(out, f"curves_Y{Y:.2f}.pkl"), "wb"))


def plot_all(out):
    import matplotlib
    import matplotlib.ticker
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from aiomfac_py.diagram import label_regions, plot_phase_map

    f1 = os.path.join(out, "fig1.pkl")
    if os.path.exists(f1):
        d = pickle.load(open(f1, "rb"))
        fig, ax = plt.subplots(figsize=(6.2, 5.6))
        keys = sorted({s for row in d["first"] for s in row if s}, key=lambda s: "".join(LETTERS.get(k, k) for k in s))
        code = np.full(d["drh"].shape, np.nan)
        for i in range(code.shape[0]):
            for j in range(code.shape[1]):
                s = d["first"][i, j]
                if s:
                    code[i, j] = keys.index(s)
        # first solid: the solid(s) present just below the full deliquescence RH; each region is outlined by the
        # 0.5 contour of its indicator on the computed grid (linear interpolation between grid points)
        for k in range(len(keys)):
            ind = (code == k).astype(float)
            if ind.any():
                ax.contour(d["x"], d["y"], ind, levels=[0.5], colors="k", linewidths=1.6)
        none = np.isnan(code)
        if none.any():
            ax.contour(d["x"], d["y"], none.astype(float), levels=[0.5], colors="k", linewidths=1.0, linestyles=":")
            jy, jx = np.nonzero(none)
            ax.text(np.median(d["x"][jx]), np.median(d["y"][jy]), "no solid", fontsize=8, ha="center", va="center")
        cs = ax.contour(d["x"], d["y"], d["drh"], levels=np.arange(0.05, 0.95, 0.05), colors="#3060a0", linewidths=0.8)
        ax.clabel(cs, fmt=lambda v: f"{100 * v:.0f}", fontsize=7)
        for k, sset in enumerate(keys):
            m = code == k
            if m.sum() >= 1:
                jy, jx = np.nonzero(m)
                ax.text(np.median(d["x"][jx]), np.median(d["y"][jy]), "+".join(LETTERS.get(t, t) for t in sset),
                        fontsize=10 if m.sum() >= 3 else 8, weight="bold", ha="center", va="center")
        ax.set_xlabel("Ammonium fraction X")
        ax.set_ylabel("Sulfate fraction Y")
        ax.set_title("Fig. 1: DRH (%) and first solid, 298.15 K (AIOMFAC)", fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(out, "uhaero06_fig01.png"), dpi=150)
        plt.close(fig)

    for Y, (fa, fc) in FIG_NO.items():
        fm = os.path.join(out, f"map_Y{Y:.2f}.pkl")
        if os.path.exists(fm):
            d = pickle.load(open(fm, "rb"))
            pm, fl = d["pm"], d["fields"]
            fig, axs = plt.subplots(1, 2, figsize=(12.0, 5.0))
            for ax, key, levels, fmt, tag in (
                    (axs[0], "pH", np.arange(-2.0, 6.01, 0.4), "%.1f", "a: pH (mole-fraction scale)"),
                    (axs[1], "rel_mass", [1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0], "%.1f",
                     "b: relative particle mass")):
                plot_phase_map(pm, ax=ax, legend=False, colors={s: "#ffffff" for s in pm.states()}, rh_points=1000,
                               x_points=800)
                label_regions(ax, pm, LETTERS, min_cells=150, fontsize=7)
                # pH on the mole-fraction scale used by the paper: a_x = a_m M_w, pH_x = pH_m - log10(0.018015)
                z = np.ma.masked_invalid(fl[key] - np.log10(0.018015) if key == "pH" else fl[key])
                cs = ax.contour(fl["x"], fl["rh"], z, levels=levels, colors="#3060a0", linewidths=0.7)
                ax.clabel(cs, fmt=fmt, fontsize=6)
                ax.set_title(f"Fig. {fa}{tag[0]}: Y = {Y}, {tag[3:]}", fontsize=10)
                ax.set_ylabel("RH (%)")
                ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{100 * v:.0f}"))
            plt.tight_layout()
            plt.savefig(os.path.join(out, f"uhaero06_fig{fa:02d}.png"), dpi=150)
            plt.close(fig)
        fcv = os.path.join(out, f"curves_Y{Y:.2f}.pkl")
        if os.path.exists(fcv):
            d = pickle.load(open(fcv, "rb"))
            xsel = list(d)
            half = (len(xsel) + 1) // 2
            fig, axs = plt.subplots(1, 2, figsize=(11.0, 4.4))
            ax_rng = CURVE_AXES[Y]
            for ax, group, tag, xr, yr in ((axs[0], xsel[:half], "a", ax_rng[0], ax_rng[1]),
                                           (axs[1], xsel[half:], "b", ax_rng[2], ax_rng[3])):
                for X in group:
                    p = d[X]
                    ax.plot(100 * p["rh"], p["rel_mass"], lw=1.4, label=f"({xsel.index(X) + 1}) X = {X}")
                ax.set_xlabel("RH (%)")
                ax.set_ylabel("Relative particle mass W$_p$/W$_{dry}$")
                ax.set_title(f"Fig. {fc}{tag}: Y = {Y}", fontsize=10)
                ax.legend(fontsize=8, frameon=False, loc="upper left")
                ax.set_xlim(*xr)
                ax.set_ylim(*yr)
                ax.grid(alpha=0.3, ls=":")
            plt.tight_layout()
            plt.savefig(os.path.join(out, f"uhaero06_fig{fc:02d}.png"), dpi=150)
            plt.close(fig)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "fig1":
        out = sys.argv[2] if len(sys.argv) > 2 else "."
        run_fig1(out)
    elif cmd in ("map", "curves"):
        Y = float(sys.argv[2])
        out = sys.argv[3] if len(sys.argv) > 3 else "."
        (run_map if cmd == "map" else run_curves)(Y, out)
    elif cmd == "plot":
        plot_all(sys.argv[2] if len(sys.argv) > 2 else ".")

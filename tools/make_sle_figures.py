"""Figures for docs/SLE.md: python tools/make_sle_figures.py  (writes docs/figs/sle_*.png)."""
from __future__ import annotations
import math, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from aiomfac_py import SOLIDS
from aiomfac_py.sle import SLESolver, binary_saturation, feed_from_salts
from solubility_data import EXP_G100
from validate_solubility import _mod, sat_molality

OUT = os.path.join(os.path.dirname(__file__), "..", "docs", "figs")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3})


def fig_solubility():
    keys = ["halite", "sylvite", "sal_ammoniac", "nitratine", "niter", "ammonium_nitrate", "ammonium_sulfate", "arcanite",
            "bischofite", "Ca_nitrate_4H2O"]
    fig, axs = plt.subplots(2, 5, figsize=(13, 5.2))
    Tgrid = np.arange(270.0, 371.0, 5.0)
    for ax, k in zip(axs.ravel(), keys):
        s = SOLIDS[k]
        mw = _mod(s)[1]
        prov, d = EXP_G100[k]
        ax.plot([273.15 + t for t in d], [g / mw * 10 for g in d.values()], "ko", ms=4, label="handbook")
        for mode, ls, col in (("anchored", "--", "tab:red"), ("fitted", "-", "tab:blue")):
            ms = [sat_molality(s, T, mode=mode) for T in Tgrid]
            ax.plot(Tgrid, ms, ls, color=col, label=mode)
        ax.set_title(f"{s.formula}", fontsize=9)
        ax.set_xlabel("T / K")
        ymax = max(g / mw * 10 for g in d.values()) * 1.5
        ax.set_ylim(0, ymax)
        ax.set_ylabel("saturation molality / mol kg$^{-1}$")
    axs[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "sle_solubility.png"), dpi=140)


def fig_drh_T():
    keys = ["halite", "sylvite", "sal_ammoniac", "nitratine", "ammonium_nitrate", "ammonium_sulfate", "arcanite", "niter"]
    Tg = np.arange(268.0, 333.0, 4.0)
    fig, ax = plt.subplots(figsize=(5.2, 4))
    for k in keys:
        ax.plot(Tg, [100 * binary_saturation(k, T)["aw"] for T in Tg], label=SOLIDS[k].formula)
    ax.set_xlabel("T / K"); ax.set_ylabel("DRH of the pure salt / %")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "sle_drh_T.png"), dpi=140)


def fig_phase_nacl_kcl():
    sol = SLESolver(["Na+", "K+", "Cl-"])
    xs = np.linspace(0.03, 0.97, 25)
    rhs = np.linspace(0.66, 0.90, 31)
    code = np.zeros((len(rhs), len(xs)))
    lab = {"aqueous": 0, "halite": 1, "sylvite": 2, "halite+sylvite(+L)": 3, "dry": 4}
    for i, rh in enumerate(rhs):
        for j, x in enumerate(xs):
            r = sol.solve(feed_from_salts({"KCl": x, "NaCl": 1 - x}), 298.15, rh)
            ks = set(r.solids)
            if r.status == "aqueous": c = 0
            elif r.status == "dry": c = 4
            elif ks == {"halite"}: c = 1
            elif ks == {"sylvite"}: c = 2
            else: c = 3
            code[i, j] = c
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(["#cfe8ff", "#ffd9a8", "#d6f5d6", "#e8c6f0", "#bdbdbd"])
    ax.pcolormesh(xs, rhs * 100, code, cmap=cmap, vmin=-0.5, vmax=4.5, shading="nearest")
    import matplotlib.patches as mp
    ax.legend(handles=[mp.Patch(color=c, label=l) for c, l in zip(cmap.colors, ["L", "L + NaCl(s)", "L + KCl(s)", "L + NaCl + KCl", "NaCl + KCl (dry)"])],
              fontsize=7, loc="lower left")
    ax.set_xlabel("KCl / (NaCl + KCl)  [mol]"); ax.set_ylabel("RH / %"); ax.set_title("NaCl-KCl-H$_2$O, 298.15 K", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "sle_phase_nacl_kcl.png"), dpi=140)


def fig_scan_mixed():
    sol = SLESolver(["Na+", "NH4+", "Cl-", "SO4--"])
    feed = feed_from_salts({"NaCl": 1.0, "(NH4)2SO4": 1.0})
    rhs = np.linspace(0.50, 0.95, 46)
    res = sol.scan_rh(feed, 298.15, rhs)
    fig, axs = plt.subplots(1, 2, figsize=(9.5, 3.8))
    axs[0].plot(rhs * 100, [r.water_kg * 1e3 for r in res], "k-")
    axs[0].set_xlabel("RH / %"); axs[0].set_ylabel("aqueous water / g  (1 mol NaCl + 1 mol (NH$_4$)$_2$SO$_4$)")
    for k in ("sal_ammoniac", "thenardite", "ammonium_sulfate", "halite", "mirabilite"):
        y = [r.solids.get(k, 0.0) for r in res]
        if max(y) > 0:
            axs[1].plot(rhs * 100, y, label=SOLIDS[k].formula)
    axs[1].set_xlabel("RH / %"); axs[1].set_ylabel("solid / mol"); axs[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "sle_scan_nacl_as.png"), dpi=140)


def fig_acid_phase_map():
    """NH4+/H+/SO4-- system at 298.15 K: stable assemblage vs. RH and acidity (H+ per sulfate, 0 = (NH4)2SO4, 2 = H2SO4)."""
    from matplotlib.colors import ListedColormap
    import matplotlib.patches as mp
    sol = SLESolver(["NH4+", "H+", "SO4--"])
    fh = np.linspace(0.04, 1.96, 17)
    rhs = np.linspace(0.30, 0.90, 17)
    short = {"ammonium_sulfate": "AS", "ammonium_bisulfate": "AHS", "letovicite": "LET"}
    labels = {}
    code = np.zeros((len(rhs), len(fh)), dtype=int)
    for i, rh in enumerate(rhs):
        for j, f in enumerate(fh):
            r = sol.solve({"SO4--": 1.0, "H+": f, "NH4+": 2.0 - f}, 298.15, rh)
            lab = "+".join(short[k] for k in sorted(r.solids, key=lambda k: list(short).index(k)))
            lab = ("L" if r.status != "dry" else "") + ("+" if lab and r.status != "dry" else "") + lab
            lab = lab or "L"
            code[i, j] = labels.setdefault(lab, len(labels))
    cols = plt.cm.tab20(np.linspace(0, 1, 20))[:len(labels)]
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.pcolormesh(fh, rhs * 100, code, cmap=ListedColormap(cols), vmin=-0.5, vmax=len(labels) - 0.5, shading="nearest")
    ax.legend(handles=[mp.Patch(color=cols[v], label=k) for k, v in labels.items()], fontsize=7, loc="upper right",
              ncol=2, title="L = aqueous", title_fontsize=7)
    ax.set_xlabel("acidity: H$^+$ per SO$_4$  [(NH$_4$)$_2$SO$_4$ = 0, NH$_4$HSO$_4$ = 1, H$_2$SO$_4$ = 2]")
    ax.set_ylabel("RH / %"); ax.set_title("NH$_4^+$-H$^+$-SO$_4^{2-}$-H$_2$O, 298.15 K", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "sle_acid_phase_map.png"), dpi=140)


if __name__ == "__main__":
    which = sys.argv[1:] or ["sol", "drh", "phase", "scan", "acid"]
    if "sol" in which: fig_solubility()
    if "drh" in which: fig_drh_T()
    if "phase" in which: fig_phase_nacl_kcl()
    if "scan" in which: fig_scan_mixed()
    if "acid" in which: fig_acid_phase_map()

"""Reproduce the organic phase diagrams of Amundson et al. (2007, Atmos. Chem. Phys. 7, 4675, UHAERO: organic systems)
with AIOMFAC (aiomfac_py), 298.15 K.  The paper used UNIFAC (three parameter sets); AIOMFAC's short-range part is a
UNIFAC with its own parameters and, through S2AS, alcohol/acid-specific subgroups, so results differ quantitatively.

  python tools/uhaero2007_figures.py fig1 [out_dir]            Fig. 1: normalized Gibbs energy of mixing and
                                                               equilibrium activities of the binary systems
  python tools/uhaero2007_figures.py ternary N [out_dir]       Figs. 2-5 (N = 2..5): ternary liquid-liquid(-liquid)
                                                               phase diagram, tie lines, activity contours
  python tools/uhaero2007_figures.py curves Y [out_dir]        Fig. 10: relative water content vs RH at X = 0.6 with
                                                               the four organic pairs (alpha = 0.2), phase pies
  python tools/uhaero2007_figures.py as [out_dir]              Fig. 11: (NH4)2SO4 + organics, water content vs a_w
  python tools/uhaero2007_figures.py drh [out_dir]             Fig. 12: DRH of (NH4)2SO4 and letovicite vs alpha
  python tools/uhaero2007_figures.py map Y P [out_dir]         Figs. 8, 9: X-RH diagram with organic pair P (1-4)
  python tools/uhaero2007_figures.py lines Y P [out_dir]       smooth boundary lines for that map (used by plot)
  python tools/uhaero2007_figures.py plot [out_dir]            draw the figures from the cached results

Inorganic + organic feeds follow the paper: per mol of cation (NH4+ + H+ = 1), sum of inorganic ions (2+Y)/(1+Y),
organics alpha/(1-alpha) times that (molar organic/inorganic mixing ratio alpha), split by f_ORG1, f_ORG2.
"""
import os
import pickle
import sys
import time
import warnings

import numpy as np

from aiomfac_py import Component
from aiomfac_py.diagram import binary_mixing_curve, plot_ternary, ternary_lle
from aiomfac_py.s2as import smiles_to_components

warnings.filterwarnings("ignore")
T = 298.15
SMILES = {"X1": ("2-hydroxy-glutaric acid", "OC(=O)CCC(O)C(=O)O"), "X2": ("adipic acid", "OC(=O)CCCCC(=O)O"),
          "X3": ("glutaraldehyde", "O=CCCCC=O"), "X4": ("palmitic acid", "CCCCCCCCCCCCCCCC(=O)O"),
          "X5": ("1-hexacosanol", "C" * 25 + "CO"), "X6": ("nonacosane", "C" * 29),
          "X7": ("pinic acid", "CC1(C)C(CC1CC(=O)O)C(=O)O"), "X8": ("pinonic acid", "CC(=O)C1CC(CC(=O)O)C1(C)C")}
TERNARY = {2: ("X5", "X7"), 3: ("X2", "X3"), 4: ("X8", "X6"), 5: ("X1", "X4")}     # (s2 on y axis, s3 on x axis)
WATER = Component(1, "Water", ((16, 1),))
PAIRS = {1: ("X5", "X7", 0.5), 2: ("X2", "X3", 0.15), 3: ("X8", "X6", 0.5), 4: ("X1", "X4", 0.5)}
LETTERS = {"ammonium_sulfate": "A", "letovicite": "B", "ammonium_bisulfate": "C", "ammonium_nitrate": "D",
           "AS_2AN": "E", "AS_3AN": "F", "AHS_AN": "G"}


def comp(key, idx):
    sub = smiles_to_components([SMILES[key][1]]).components[1].subgroups
    return Component(idx, SMILES[key][0], tuple(sub))


def binary_curve(c1, c2):
    return binary_mixing_curve([c1, c2], T)


def run_fig1(out):
    res = {}
    for k in ("X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8"):
        res[("water", k)] = binary_curve(WATER, comp(k, 2))
        print(k, res[("water", k)]["splits"], flush=True)
    for a, b in (("X5", "X7"), ("X2", "X3"), ("X8", "X6"), ("X1", "X4")):
        res[(a, b)] = binary_curve(comp(a, 1), comp(b, 2))
        print(a, b, res[(a, b)]["splits"], flush=True)
    pickle.dump(res, open(os.path.join(out, "u07_fig1.pkl"), "wb"))


def run_ternary(N, out):
    s2, s3 = TERNARY[N]
    t0 = time.time()
    td = ternary_lle([WATER, comp(s2, 2), comp(s3, 3)], T)
    print(f"Fig. {N}: {len(td.P)} points, {time.time() - t0:.0f} s; three-phase triangles: "
          f"{[np.round(v, 4).tolist() for v in td.three_phase]}", flush=True)
    pickle.dump({"s2": s2, "s3": s3, "td": td}, open(os.path.join(out, f"u07_tern{N}.pkl"), "wb"))

# ---------------------------------------------------------------------------------------------------------------
# inorganic + organic systems (Figs. 8-12)
# ---------------------------------------------------------------------------------------------------------------
def inorg_feed(X, Y):
    f = {"NH4+": X, "H+": 1.0 - X, "SO4--": Y / (1.0 + Y), "NO3-": (1.0 - Y) / (1.0 + Y)}
    return {k: v for k, v in f.items() if v > 1e-12}


def org_system(pair, ions):
    from aiomfac_py.phase_equilibrium import PhaseEquilibrium
    if pair is None:
        return PhaseEquilibrium([], ions, T_K=T), None, None, None
    k1, k2, f1 = PAIRS[pair]
    o1, o2 = comp(k1, 2), comp(k2, 3)
    return PhaseEquilibrium([o1, o2], ions, T_K=T), o1.name, o2.name, f1


def with_organics(feed, o1, o2, f1, alpha):
    feed = dict(feed)
    if o1 is not None and alpha > 0:
        b = alpha / (1.0 - alpha) * sum(feed.values())
        feed[o1], feed[o2] = f1 * b, (1.0 - f1) * b
    return feed


def phase_pies(res, o1, o2):
    """[(amount, [org1, org2, water])] of the liquids (mol)"""
    out = []
    for L in res.liquids:
        d = dict(zip(L.names, L.amounts))
        out.append((float(sum(d[k] for k in (o1, o2, "Water") if k in d)),
                    [float(d.get(o1, 0.0)), float(d.get(o2, 0.0)), float(d["Water"])]))
    return out


def profile(pe, feed, rh_grid, o1, o2, pies_at=(), **kw):
    from aiomfac_py.diagram import _solve, _ok
    inorg = sum(v for k, v in feed.items() if k[-1] in "+-")
    rh = np.asarray(rh_grid, float)
    wc = np.full(len(rh), np.nan)
    states = [None] * len(rh)
    pies = {}
    prev = None
    for i in np.argsort(-rh):
        r = _solve(pe, feed, float(rh[i]), "equilibrium", prev, dict(kw))
        if not _ok(r) and prev is not None:
            r = _solve(pe, feed, float(rh[i]), "equilibrium", None, dict(kw))
        if not _ok(r):
            prev = None
            continue
        wc[i] = sum(float(L.amounts[0]) for L in r.liquids) / inorg
        states[i] = (len(r.liquids), tuple(sorted(r.solids)))
        if any(abs(rh[i] - p) < 1e-9 for p in pies_at):
            pies[round(float(rh[i]), 4)] = phase_pies(r, o1, o2)
        prev = r if r.liquids else None
    return {"rh": rh, "wc": wc, "states": states, "pies": pies}


def run_curves(Y, out):
    ions = ["NH4+", "H+", "SO4--"] + (["NO3-"] if Y < 1 else [])
    rg = np.round(np.arange(0.02, 0.8001, 0.01), 4)
    res = {}
    for pair in (None, 1, 2, 3, 4):
        pe, o1, o2, f1 = org_system(pair, ions)
        feed = with_organics(inorg_feed(0.6, Y), o1, o2, f1, 0.2)
        t0 = time.time()
        res[pair] = profile(pe, feed, rg, o1, o2, pies_at=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7))
        print(f"Y = {Y}, pair {pair}: {time.time() - t0:.0f} s", flush=True)
    pickle.dump(res, open(os.path.join(out, f"u07_curves_Y{Y:.2f}.pkl"), "wb"))


def run_as(out):
    rg = np.round(np.arange(0.5, 0.9801, 0.01), 4)
    res = {}
    for single in (False, True):
        for pair in (None, 1, 2, 3, 4):
            pe, o1, o2, f1 = org_system(pair, ["NH4+", "SO4--"])
            feed = with_organics({"NH4+": 2.0, "SO4--": 1.0}, o1, o2, f1, 0.2)
            kw = {"max_liquids": 1} if single else {}
            res[(single, pair)] = profile(pe, feed, rg, o1, o2, pies_at=(0.5, 0.6, 0.7, 0.8, 0.9), **kw)
            print(f"AS single={single} pair {pair} done", flush=True)
    pickle.dump(res, open(os.path.join(out, "u07_as.pkl"), "wb"))


def run_drh(out):
    from aiomfac_py.diagram import deliquescence_point, _solve, _ok
    import signal

    class Slow(Exception):
        pass

    def alarm(*_):
        raise Slow()

    signal.signal(signal.SIGALRM, alarm)
    alphas = [0.0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9]
    res = {}
    for salt, ions, base in (("AS", ["NH4+", "SO4--"], {"NH4+": 2.0, "SO4--": 1.0}),
                             ("LET", ["NH4+", "H+", "SO4--"], {"NH4+": 3.0, "H+": 1.0, "SO4--": 2.0})):
        for pair in (1, 2, 3, 4):
            pe, o1, o2, f1 = org_system(pair, ions)
            drh, pies = [], {}
            for a in alphas:
                feed = with_organics(base, o1, o2, f1, a)
                signal.alarm(90)                      # a few compositions make the solver very slow: skip them
                try:
                    r, s = deliquescence_point(pe, feed, step=0.04, method="si")
                except Slow:
                    r = None
                finally:
                    signal.alarm(0)
                drh.append(np.nan if r is None else r)
                if r is not None and a in (0.2, 0.4, 0.6, 0.7, 0.9):
                    rr = _solve(pe, feed, min(r + 2e-3, 0.99), "equilibrium", None, {})
                    if _ok(rr):
                        pies[a] = phase_pies(rr, o1, o2)
            res[(salt, pair)] = {"alpha": alphas, "drh": drh, "pies": pies}
            print(salt, pair, np.round(drh, 3), flush=True)
    pickle.dump(res, open(os.path.join(out, "u07_drh.pkl"), "wb"))


def _with_time_limit(pe, seconds=60):
    """make every solve of ``pe`` give up after ``seconds`` (reported as not converged, so that trace() drops the
    point): a few compositions with nonacosane make the solver very slow"""
    import signal
    from aiomfac_py.phase_equilibrium import PhaseEquilibriumResult

    class Slow(Exception):
        pass

    def alarm(*_):
        raise Slow()

    signal.signal(signal.SIGALRM, alarm)
    solve = pe.solve

    def limited(feed, rh, **kw):
        signal.alarm(seconds)
        try:
            return solve(feed, rh, **kw)
        except Slow:
            return PhaseEquilibriumResult("not_converged", pe.T, rh, [], {}, {}, float("nan"), {}, 0.0,
                                          message="time limit")
        finally:
            signal.alarm(0)
    pe.solve = limited
    return pe


def run_map(Y, pair, out):
    from aiomfac_py.diagram import phase_map
    ions = ["NH4+", "H+", "SO4--"] + (["NO3-"] if Y < 1 else [])
    pe, o1, o2, f1 = org_system(pair, ions)
    _with_time_limit(pe, 60)
    xs = np.linspace(0.0, 1.0, 11)
    xs[0] = 0.02
    t0 = time.time()
    pm = phase_map(pe, lambda X: with_organics(inorg_feed(X, Y), o1, o2, f1, 0.2), xs,
                   x_label="Ammonium fraction X", n=16, rh_min=0.02, rh_max=0.85, x_tol=0.05, refine_x=Y >= 1,
                   verbose=True)
    print(f"Y = {Y}, pair {pair}: {time.time() - t0:.0f} s", flush=True)
    pickle.dump(pm, open(os.path.join(out, f"u07_map_Y{Y:.2f}_P{pair}.pkl"), "wb"))


def _save(obj, path):
    """pickle to a temporary file and rename it (a run killed while writing keeps the previous file)"""
    pickle.dump(obj, open(path + ".tmp", "wb"))
    os.replace(path + ".tmp", path)


def run_lines(Y, pair, out):
    """smooth boundary lines for the cached map (trace_boundaries); the solves are cached in lines_cache_*.pkl"""
    from aiomfac_py.diagram import trace_boundaries
    ions = ["NH4+", "H+", "SO4--"] + (["NO3-"] if Y < 1 else [])
    pe, o1, o2, f1 = org_system(pair, ions)
    pm = pickle.load(open(os.path.join(out, f"u07_map_Y{Y:.2f}_P{pair}.pkl"), "rb"))
    cf = os.path.join(out, f"lines_cache_Y{Y:.2f}_P{pair}.pkl")
    cache = pickle.load(open(cf, "rb")) if os.path.exists(cf) else {}
    t0 = time.time()
    try:
        bc = trace_boundaries(pe, lambda X: with_organics(inorg_feed(X, Y), o1, o2, f1, 0.2), pm, solve_timeout=20,
                              cache=cache, checkpoint=lambda c: _save(c, cf), verbose=True)
    finally:
        _save(cache, cf)
    print(f"Y = {Y}, pair {pair}: lines {time.time() - t0:.0f} s, {bc.n_solves} solves", flush=True)
    pickle.dump(bc, open(os.path.join(out, f"u07_lines_Y{Y:.2f}_P{pair}.pkl"), "wb"))


def plot_all(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    import matplotlib.colors

    f1 = os.path.join(out, "u07_fig1.pkl")
    if os.path.exists(f1):
        d = pickle.load(open(f1, "rb"))
        fig, axs = plt.subplots(3, 3, figsize=(12, 11))
        groups = [[("water", k) for k in ("X1", "X2", "X3", "X4")], [("water", k) for k in ("X5", "X6", "X7", "X8")],
                  [("X5", "X7"), ("X2", "X3"), ("X8", "X6"), ("X1", "X4")]]
        for row, grp in enumerate(groups):
            for key in grp:
                c = d[key]
                lab = "H$_2$O/" + key[1] if key[0] == "water" else f"{key[0]}/{key[1]}"
                l, = axs[row, 0].plot(c["x"], c["g"], lw=1.2, label=lab)
                for a, b in c["splits"]:
                    ga, gb = np.interp([a, b], c["x"], c["g"])
                    axs[row, 0].plot([a, b], [ga, gb], "-", color=l.get_color(), lw=0.8)
                    axs[row, 0].plot([a, b], [ga, gb], "o", color="r", ms=3)
                axs[row, 1].plot(c["x"], c["a1"], lw=1.2, color=l.get_color(), label=lab)
                axs[row, 2].plot(c["x"], c["a2"], lw=1.2, color=l.get_color(), label=lab)
            first = "water" if row < 2 else "organic 1"
            axs[row, 0].set_ylabel("normalized GFE")
            axs[row, 1].set_ylabel(f"activity of {first}")
            axs[row, 2].set_ylabel("activity of organic" + (" 2" if row == 2 else ""))
            for ax in axs[row]:
                ax.set_xlabel(f"mole fraction ({first} → organic{' 2' if row == 2 else ''})")
                ax.legend(fontsize=7, frameon=False)
                ax.grid(alpha=0.3, ls=":")
            for ax in axs[row, 1:]:
                ax.plot([0, 1], [1, 0] if ax is axs[row, 1] else [0, 1], "k-", lw=0.6)
                ax.set_ylim(0, 1.02)
        fig.suptitle("Fig. 1 (AIOMFAC): normalized Gibbs energy of mixing and equilibrium activities, 298.15 K",
                     fontsize=11)
        plt.tight_layout()
        plt.savefig(os.path.join(out, "uhaero07_fig01.png"), dpi=130)
        plt.close(fig)

    for N in TERNARY:
        f = os.path.join(out, f"u07_tern{N}.pkl")
        if not os.path.exists(f):
            continue
        d = pickle.load(open(f, "rb"))
        td = d["td"]
        n2, n3 = SMILES[d["s2"]][0], SMILES[d["s3"]][0]
        fig, axs = plt.subplots(2, 2, figsize=(10, 9.5))
        titles = ["phase diagram", "activity of water", f"activity of {n2}", f"activity of {n3}"]
        for k, ax in enumerate(axs.flat):
            plot_ternary(td, ax=ax, show="phases" if k == 0 else k - 1)
            ax.set_title(f"({'abcd'[k]}) {titles[k]}", fontsize=10)
        fig.suptitle(f"Fig. {N} (AIOMFAC): water / {n2} ({d['s2']}) / {n3} ({d['s3']}), 298.15 K", fontsize=11)
        plt.tight_layout()
        plt.savefig(os.path.join(out, f"uhaero07_fig{N:02d}.png"), dpi=130)
        plt.close(fig)


def _pie(ax, x, y, parts, r):
    """small pie at axes-fraction (x, y): parts = [org1, org2, water] (red, yellow, blue)"""
    tot = sum(parts)
    if tot <= 0:
        return
    ins = ax.inset_axes([x - r, y - r, 2 * r, 2 * r], transform=ax.transAxes)
    ins.pie(parts, colors=["#d62728", "#f2d024", "#1f4fbf"], startangle=90, counterclock=True,
            wedgeprops={"linewidth": 0.3, "edgecolor": "k"})
    ins.set_aspect("equal")


def _pie_rows(fig, gs_row, pies_by_pair, keys, label_fmt):
    """one row of pies per organic pair; each cell holds the liquids at that key (size ~ amount)"""
    import matplotlib.pyplot as plt
    pairs = [p for p in (1, 2, 3, 4) if p in pies_by_pair]
    ax = fig.add_subplot(gs_row)
    ax.set_xlim(0, len(keys)); ax.set_ylim(0, len(pairs)); ax.axis("off")
    for i, p in enumerate(pairs):
        ax.text(-0.15, len(pairs) - i - 0.5, f"({p})", ha="right", va="center", fontsize=8,
                transform=ax.transData)
        for j, k in enumerate(keys):
            phases = pies_by_pair[p].get(k, [])
            if not phases:
                continue
            tot = sum(a for a, _ in phases)
            nph = len(phases)
            for q, (amt, parts) in enumerate(phases):
                r = 0.42 * np.sqrt(amt / tot) / max(1.0, nph ** 0.5)
                cx = (j + (q + 1) / (nph + 1)) / len(keys)
                cy = (len(pairs) - i - 0.5) / len(pairs)
                _pie(ax, cx, cy, parts, r / max(len(keys), len(pairs)) * 1.6)
    for j, k in enumerate(keys):
        ax.text(j + 0.5, len(pairs) + 0.05, label_fmt(k), ha="center", va="bottom", fontsize=7)
    plt.setp(ax, xticks=[], yticks=[])


def plot_org(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    names = {None: "(0) inorganic only", 1: "(1) X5/X7", 2: "(2) X2/X3", 3: "(3) X8/X6", 4: "(4) X1/X4"}
    styles = {None: ("r", "-"), 1: ("g", "--"), 2: ("m", "--"), 3: ("b", "--"), 4: ("c", "--")}
    cur = {Y: os.path.join(out, f"u07_curves_Y{Y:.2f}.pkl") for Y in (1.0, 0.85)}
    if all(os.path.exists(f) for f in cur.values()):
        fig = plt.figure(figsize=(12, 9))
        gs = GridSpec(2, 2, height_ratios=[1.2, 1.0], figure=fig)
        for c, (Y, f) in enumerate(cur.items()):
            d = pickle.load(open(f, "rb"))
            ax = fig.add_subplot(gs[0, c])
            for p, r in d.items():
                ax.plot(100 * r["rh"], r["wc"], styles[p][0] + styles[p][1], lw=1.3, label=names[p])
            ax.set_xlim(0, 80); ax.set_ylim(0, 3)
            ax.set_xlabel("RH (%)"); ax.set_ylabel("relative water content b$_{H2O}$/Σb$_{INORG}$")
            ax.set_title(f"({'ab'[c]}) X = 0.6, Y = {Y}", fontsize=10)
            ax.legend(fontsize=7, frameon=False); ax.grid(alpha=0.3, ls=":")
            _pie_rows(fig, gs[1, c], {p: r["pies"] for p, r in d.items() if p is not None},
                      [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], lambda k: f"{100 * k:.0f}%")
        fig.suptitle("Fig. 10 (AIOMFAC): deliquescence with two organics, α = 0.2 (pies: org1 red, org2 yellow, "
                     "water blue per liquid)", fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(out, "uhaero07_fig10.png"), dpi=130)
        plt.close(fig)

    f = os.path.join(out, "u07_as.pkl")
    if os.path.exists(f):
        # panel (b) of the paper (one liquid phase assumed, water activities above 1) cannot be computed at fixed
        # RH < 1 and is not reproduced
        d = pickle.load(open(f, "rb"))
        fig = plt.figure(figsize=(7.5, 9))
        gs = GridSpec(2, 1, height_ratios=[1.2, 1.0], figure=fig)
        ax = fig.add_subplot(gs[0])
        for p in (None, 1, 2, 3, 4):
            r = d[(False, p)]
            ax.plot(r["rh"], r["wc"], styles[p][0] + styles[p][1], lw=1.3, label=names[p])
        ax.set_xlim(0.5, 0.98); ax.set_ylim(0, 9)
        ax.set_xlabel("water activity"); ax.set_ylabel("relative water content b$_{H2O}$/Σb$_{INORG}$")
        ax.set_title("(a) liquid-liquid equilibria included", fontsize=10)
        ax.legend(fontsize=7, frameon=False); ax.grid(alpha=0.3, ls=":")
        _pie_rows(fig, gs[1], {p: d[(False, p)]["pies"] for p in (1, 2, 3, 4)},
                  [0.5, 0.6, 0.7, 0.8, 0.9], lambda k: f"a$_w$ = {k}")
        fig.suptitle("Fig. 11a (AIOMFAC): (NH$_4$)$_2$SO$_4$ with two organics, α = 0.2", fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(out, "uhaero07_fig11.png"), dpi=130)
        plt.close(fig)

    f = os.path.join(out, "u07_drh.pkl")
    if os.path.exists(f):
        d = pickle.load(open(f, "rb"))
        fig = plt.figure(figsize=(12, 9))
        gs = GridSpec(2, 2, height_ratios=[1.2, 1.0], figure=fig)
        for c, salt in enumerate(("AS", "LET")):
            ax = fig.add_subplot(gs[0, c])
            for p in (1, 2, 3, 4):
                r = d[(salt, p)]
                ax.plot(r["alpha"], 100 * np.array(r["drh"]), styles[p][0] + styles[p][1], marker="o", ms=3,
                        lw=1.3, label=names[p])
            ax.set_xlim(0, 1); ax.set_ylim(0, 90)
            ax.set_xlabel("organic/inorganic mixing ratio α"); ax.set_ylabel(f"DRH of {salt} (%)")
            ax.set_title(f"({'ab'[c]}) {'(NH$_4$)$_2$SO$_4$' if salt == 'AS' else '(NH$_4$)$_3$H(SO$_4$)$_2$'}",
                         fontsize=10)
            ax.legend(fontsize=7, frameon=False); ax.grid(alpha=0.3, ls=":")
            _pie_rows(fig, gs[1, c], {p: d[(salt, p)]["pies"] for p in (1, 2, 3, 4)},
                      [0.2, 0.4, 0.6, 0.7, 0.9], lambda k: f"α = {k}")
        fig.suptitle("Fig. 12 (AIOMFAC): deliquescence RH with two organics", fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(out, "uhaero07_fig12.png"), dpi=130)
        plt.close(fig)

    from aiomfac_py.diagram import label_regions, plot_phase_map
    for Y, fno in ((1.0, 8), (0.85, 9)):
        files = [os.path.join(out, f"u07_map_Y{Y:.2f}_P{p}.pkl") for p in (1, 2, 3, 4)]
        if not any(os.path.exists(f) for f in files):
            continue
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        for p, (ax, f) in enumerate(zip(axs.flat, files), start=1):
            if not os.path.exists(f):
                ax.axis("off")
                continue
            pm = pickle.load(open(f, "rb"))
            fb = os.path.join(out, f"u07_lines_Y{Y:.2f}_P{p}.pkl")
            bc = pickle.load(open(fb, "rb")) if os.path.exists(fb) else None
            plot_phase_map(pm, ax=ax, legend=False, colors={s: c for s, c in zip(
                pm.states(), ["#ffffff"] * 99)}, curves=bc, show_failed=True)
            label_regions(ax, pm, LETTERS, min_cells=200, fontsize=7, curves=bc)
            ax.set_title(f"({'abcd'[p - 1]}) {names[p][4:]}, α = 0.2", fontsize=10)
            ax.set_ylabel("RH")
        fig.suptitle(f"Fig. {fno} (AIOMFAC): Y = {Y} with two organics (L1/L2/L3: liquid phases; letters: solids; "
                     "grey dots: solver did not converge)",
                     fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(out, f"uhaero07_fig{fno:02d}.png"), dpi=130)
        plt.close(fig)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "fig1":
        run_fig1(sys.argv[2] if len(sys.argv) > 2 else ".")
    elif cmd == "ternary":
        run_ternary(int(sys.argv[2]), sys.argv[3] if len(sys.argv) > 3 else ".")
    elif cmd == "curves":
        run_curves(float(sys.argv[2]), sys.argv[3] if len(sys.argv) > 3 else ".")
    elif cmd == "as":
        run_as(sys.argv[2] if len(sys.argv) > 2 else ".")
    elif cmd == "drh":
        run_drh(sys.argv[2] if len(sys.argv) > 2 else ".")
    elif cmd in ("map", "lines"):
        (run_map if cmd == "map" else run_lines)(float(sys.argv[2]), int(sys.argv[3]),
                                                  sys.argv[4] if len(sys.argv) > 4 else ".")
    elif cmd == "plot":
        plot_all(sys.argv[2] if len(sys.argv) > 2 else ".")
        plot_org(sys.argv[2] if len(sys.argv) > 2 else ".")

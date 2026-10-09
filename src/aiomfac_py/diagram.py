"""Phase diagrams of aerosol particles: phase-boundary tracing in relative humidity and composition--RH maps.

Built on :class:`aiomfac_py.phase_equilibrium.PhaseEquilibrium` (any liquid model, e.g. AIOMFAC or an
excess-Gibbs-energy surrogate through ``liquid_model=``), in the style of the UHAERO diagrams (Amundson et al., 2006,
2007, Atmos. Chem. Phys.).

* :func:`trace` -- for one particle composition (feed), solve on an RH grid and locate every RH at which the phase
  state (number of liquids, set of solids) changes by bisection, to a tolerance ``tol`` in RH.  Modes:
  ``"equilibrium"`` (all candidate solids) and ``"metastable"`` (no solids: supersaturated liquids, liquid-liquid
  phase separation only).
* :func:`phase_map` -- :func:`trace` for each value of a composition parameter X (e.g. the ammonium fraction or the
  organic-to-inorganic ratio), giving the data of an X--RH diagram.
* :func:`plot_phase_map` -- the diagram (regions coloured by phase state, boundary lines), optional pie charts of the
  phase composition at selected points (needs matplotlib, ``pip install aiomfac_py[plot]``).

Boundaries are located between two converged solves (``status`` other than ``"not_converged"``) whose phase states
differ; solves that do not converge are dropped and listed in ``Trace.failed``, and a bisection point that does not
converge is retried off-centre (a boundary whose interval cannot be narrowed is reported with ``resolved=False``).  A state that exists only in an RH
interval narrower than the initial grid spacing can be missed; refine the grid (``n``) where that matters.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from .phase_equilibrium import PhaseEquilibrium, PhaseEquilibriumResult

__all__ = ["PhaseState", "Boundary", "Trace", "PhaseMap", "phase_state", "trace", "phase_map", "plot_phase_map",
           "pie_composition", "particle_properties", "rh_profile", "deliquescence_point", "label_regions"]


@dataclass(frozen=True, order=True)
class PhaseState:
    """Phase state at one RH: number of liquid phases and the solids present (sorted keys)."""
    n_liquids: int
    solids: tuple

    @property
    def label(self) -> str:
        parts = []
        if self.n_liquids == 1:
            parts.append("L")
        elif self.n_liquids > 1:
            parts.append(f"{self.n_liquids}L")
        parts += list(self.solids)
        return " + ".join(parts) if parts else "empty"


def phase_state(res: PhaseEquilibriumResult) -> PhaseState:
    return PhaseState(len(res.liquids), tuple(sorted(res.solids)))


def _ok(res: PhaseEquilibriumResult) -> bool:
    return res.status != "not_converged"


@dataclass
class Boundary:
    """A change of phase state between ``rh_lo`` and ``rh_hi`` (``rh_hi - rh_lo <= tol`` when resolved);
    ``below`` is the state at ``rh_lo``, ``above`` that at ``rh_hi``."""
    rh: float
    rh_lo: float
    rh_hi: float
    below: PhaseState
    above: PhaseState
    resolved: bool = True

    @property
    def kind(self) -> str:
        """``"deliquescence"`` (a solid dissolves with increasing RH), ``"phase separation"`` (the number of
        liquids changes), ``"precipitation"`` (a solid appears with increasing RH), ``"solid transition"`` (one solid replaces another,
        e.g. a hydrate change), or ``"other"``."""
        gone = set(self.below.solids) - set(self.above.solids)
        new = set(self.above.solids) - set(self.below.solids)
        if gone and not new:
            return "deliquescence"
        if new and not gone:
            return "precipitation"
        if not gone and not new and self.below.n_liquids != self.above.n_liquids:
            return "phase separation"
        if gone and new:
            return "solid transition"
        return "other"


@dataclass
class Trace:
    feed: dict
    mode: str
    rh: np.ndarray                         # all RH values solved (grid and bisection points), increasing
    results: list                          # PhaseEquilibriumResult for each rh
    boundaries: list                       # Boundary, increasing RH
    n_solves: int = 0
    failed: list = field(default_factory=list)   # RH of solves that did not converge and were dropped
    rh_range: tuple = ()                          # requested (lowest, highest) RH

    @property
    def states(self) -> list:
        return [phase_state(r) for r in self.results]

    def state_at(self, rh: float) -> PhaseState:
        """Phase state at ``rh`` from the boundaries (exact up to ``tol``); the state of the nearest solve if
        ``rh`` lies inside an unresolved interval."""
        i = int(np.argmin(np.abs(self.rh - rh)))
        return phase_state(self.results[i])

    def intervals(self) -> list:
        """[(rh_from, rh_to, PhaseState)] covering the requested RH range, split at the boundaries (the states at
        the ends are those of the lowest and highest converged solves)."""
        out = []
        lo = float(min(self.rh[0], self.rh_range[0])) if self.rh_range else float(self.rh[0])
        for b in self.boundaries:
            out.append((lo, b.rh, b.below))
            lo = b.rh
        hi = float(max(self.rh[-1], self.rh_range[1])) if self.rh_range else float(self.rh[-1])
        out.append((lo, hi, phase_state(self.results[-1])))
        return out

    @property
    def signature(self) -> tuple:
        """Sequence of (below, above) phase states of the boundaries: equal signatures mean the same topology."""
        return tuple((b.below, b.above) for b in self.boundaries)


def _solve(pe: PhaseEquilibrium, feed: dict, rh: float, mode: str, init, solve_kw: dict) -> PhaseEquilibriumResult:
    solids = solve_kw.pop("solids", "all")
    if mode == "metastable":
        solids = "none"
    return pe.solve(feed, rh, solids=solids, init=init, **solve_kw)


def trace(pe: PhaseEquilibrium, feed: dict, *, rh_min: float = 0.05, rh_max: float = 0.98, n: int = 32,
          mode: str = "equilibrium", tol: float = 1.0e-4, rh_grid: Sequence[float] | None = None,
          warm: bool = True, **solve_kw) -> Trace:
    """Phase states of ``feed`` [mol] over RH and the RH of every change of phase state.

    The RH grid (``rh_grid``, or ``n`` points from ``rh_max`` down to ``rh_min``) is solved from high to low RH with
    warm starts (as :meth:`PhaseEquilibrium.rh_scan`).  Every pair of neighbouring solves with different phase
    states is then bisected until the interval is narrower than ``tol``; a state found in between splits the
    interval, so several changes within one grid interval are all located.  Other keywords go to
    :meth:`PhaseEquilibrium.solve` (``solids`` restricts the candidate solids in equilibrium mode)."""
    if mode not in ("equilibrium", "metastable"):
        raise ValueError("mode must be 'equilibrium' or 'metastable'")
    grid = np.sort(np.asarray(rh_grid if rh_grid is not None else np.linspace(rh_min, rh_max, n), float))[::-1]
    pts: dict[float, PhaseEquilibriumResult] = {}
    prev = None
    count = 0
    for rh in grid:
        res = _solve(pe, feed, float(rh), mode, prev if warm else None, dict(solve_kw))
        count += 1
        if not _ok(res) and prev is not None:          # a warm start that failed is retried cold inside solve();
            res2 = _solve(pe, feed, float(rh), mode, None, dict(solve_kw))   # here also when warm=False
            count += 1
            res = res2 if _ok(res2) else res
        pts[float(rh)] = res
        prev = res if (res.liquids and _ok(res)) else None

    # solves that did not converge are dropped (reported in Trace.failed); boundaries are located between converged
    # solves only, so a failure next to a boundary only widens the interval that the bisection starts from
    failed = sorted(k for k, r in pts.items() if not _ok(r))
    for k in failed:
        del pts[k]
    if not pts:
        raise RuntimeError(f"no converged solve for feed {feed}")

    boundaries: list[Boundary] = []

    def refine(lo: float, hi: float):
        nonlocal count
        r_lo, r_hi = pts[lo], pts[hi]
        s_lo, s_hi = phase_state(r_lo), phase_state(r_hi)
        if s_lo == s_hi:
            return
        if not (_ok(r_lo) and _ok(r_hi)):
            boundaries.append(Boundary(0.5 * (lo + hi), lo, hi, s_lo, s_hi, resolved=False))
            return
        if hi - lo <= tol:
            boundaries.append(Boundary(0.5 * (lo + hi), lo, hi, s_lo, s_hi))
            return
        start = r_hi if r_hi.liquids else (r_lo if r_lo.liquids else None)
        alt = r_lo if start is r_hi else r_hi
        res = None
        for frac in (0.5, 0.25, 0.75):              # a solve that fails right at a boundary is retried off-centre
            mid = lo + frac * (hi - lo)
            for init in (start, alt if (alt is not None and alt.liquids) else None):
                res = _solve(pe, feed, mid, mode, init if warm else None, dict(solve_kw))
                count += 1
                if _ok(res) or not warm:
                    break
            if _ok(res):
                break
        if not _ok(res):
            boundaries.append(Boundary(0.5 * (lo + hi), lo, hi, s_lo, s_hi, resolved=False))
            return
        pts[mid] = res
        refine(lo, mid)
        refine(mid, hi)

    keys = sorted(pts)
    for a, b in zip(keys[:-1], keys[1:]):
        refine(a, b)
    boundaries.sort(key=lambda b: b.rh)
    rhs = np.array(sorted(pts))
    return Trace(dict(feed), mode, rhs, [pts[r] for r in rhs], boundaries, count, sorted(failed),
                 (float(grid.min()), float(grid.max())))


@dataclass
class PhaseMap:
    x: np.ndarray                          # composition parameter values
    traces: list                           # Trace for each x
    x_label: str = "X"
    mode: str = "equilibrium"
    extra: dict = field(default_factory=dict)

    def states(self) -> list:
        """All phase states that occur, ordered by number of solids (descending), then liquids."""
        seen = {iv[2] for t in self.traces for iv in t.intervals()}
        return sorted(seen, key=lambda s: (-len(s.solids), s.n_liquids, s.solids))

    def boundary_lines(self, max_width: float = 1.0e-2) -> dict:
        """{(below, above): (x array, rh array)}: boundaries of the same kind connected across x (unresolved
        boundaries included when their bracket is narrower than ``max_width``)."""
        lines: dict = {}
        for x, t in zip(self.x, self.traces):
            for b in t.boundaries:
                if b.resolved or b.rh_hi - b.rh_lo <= max_width:
                    lines.setdefault((b.below, b.above), ([], []))
                    lines[(b.below, b.above)][0].append(float(x))
                    lines[(b.below, b.above)][1].append(b.rh)
        return {k: (np.array(v[0]), np.array(v[1])) for k, v in lines.items()}

    def to_records(self) -> list:
        """Flat list of boundaries (for CSV/JSON output)."""
        out = []
        for x, t in zip(self.x, self.traces):
            for b in t.boundaries:
                out.append({"x": float(x), "rh": b.rh, "rh_lo": b.rh_lo, "rh_hi": b.rh_hi, "below": b.below.label,
                            "above": b.above.label, "kind": b.kind, "resolved": b.resolved})
        return out


def phase_map(pe: PhaseEquilibrium, feed_of_x: Callable[[float], dict], x_values: Sequence[float], *,
              x_label: str = "X", refine_x: bool = True, x_tol: float | None = None, verbose: bool = False,
              **trace_kw) -> PhaseMap:
    """:func:`trace` for the feed ``feed_of_x(x)`` at every ``x`` (keywords as :func:`trace`).

    ``refine_x``: where two neighbouring x have different boundary topologies (:attr:`Trace.signature`, e.g. on
    either side of a eutonic composition), x is bisected until the interval is narrower than ``x_tol`` (default
    1/100 of the x range), so that boundary lines end close to where they meet."""
    def run(x):
        t = trace(pe, feed_of_x(float(x)), **trace_kw)
        if verbose:
            bs = ", ".join(f"{b.rh:.4f} ({b.kind})" for b in t.boundaries)
            print(f"{x_label} = {x:.4g}: {t.n_solves} solves; boundaries {bs}", flush=True)
        return t

    xs = sorted(float(x) for x in x_values)
    res = {x: run(x) for x in xs}
    if refine_x and len(xs) > 1:
        tol = x_tol if x_tol is not None else (xs[-1] - xs[0]) / 100.0
        stack = list(zip(xs[:-1], xs[1:]))
        while stack:
            a, b = stack.pop()
            if b - a <= tol or res[a].signature == res[b].signature:
                continue
            m = 0.5 * (a + b)
            if any(abs(m - k) < 1e-12 for k in res):
                continue
            res[m] = run(m)
            stack += [(a, m), (m, b)]
    xs = sorted(res)
    return PhaseMap(np.asarray(xs, float), [res[x] for x in xs], x_label, trace_kw.get("mode", "equilibrium"))


def pie_composition(res: PhaseEquilibriumResult, group: Callable[[str], str] | None = None) -> list:
    """Composition of each phase for a pie chart: [(phase label, {group: mol})].  ``group`` maps a component name
    to a display group (default: water, organics by name, ions pooled as 'ions')."""
    def default(name):
        if name == "H2O" or name.lower() == "water":
            return "water"
        return "ions" if name[-1] in "+-" else name
    g = group or default
    out = []
    for k, liq in enumerate(res.liquids):
        d: dict = {}
        for name, a in zip(liq.names, liq.amounts):
            if a > 0:
                d[g(name)] = d.get(g(name), 0.0) + float(a)
        out.append((f"L{k + 1}", d))
    for key, amt in res.solids.items():
        out.append((key, {key: float(amt)}))
    return out


def _regular_traces(pm: PhaseMap):
    """Traces used for the region grid: all except interior ones holding a phase state that occurs at no other x
    (a singular composition, e.g. the exact stoichiometry of a salt, where a solid exists alone on a line)."""
    count: dict = {}
    for t in pm.traces:
        for st in {iv[2] for iv in t.intervals()}:
            count[st] = count.get(st, 0) + 1
    keep = [i for i, t in enumerate(pm.traces)
            if i in (0, len(pm.traces) - 1) or all(count[iv[2]] > 1 for iv in t.intervals())]
    return [pm.traces[i] for i in keep], pm.x[keep]


def singular_lines(pm: PhaseMap) -> list:
    """[(x, rh_from, rh_to, PhaseState)] of the phase states that occur at a single interior x only (drawn as
    lines, as in the UHAERO diagrams)."""
    count: dict = {}
    for t in pm.traces:
        for st in {iv[2] for iv in t.intervals()}:
            count[st] = count.get(st, 0) + 1
    out = []
    for i, (xv, t) in enumerate(zip(pm.x, pm.traces)):
        if i in (0, len(pm.traces) - 1):
            continue
        for a, b, st in t.intervals():
            if count[st] == 1:
                out.append((float(xv), a, b, st))
    return out


def _region_grid(pm: PhaseMap, rh_points: int = 400, states: list | None = None):
    """State index on a fine (RH, x) grid: the boundaries of the nearer of the two neighbouring traces, each
    interpolated linearly in x towards the boundary of the same kind (same states below and above, closest in RH) of
    the other trace when it has one.  Returns (Z, x grid, RH grid, states)."""
    states = states if states is not None else pm.states()
    idx = {s: k for k, s in enumerate(states)}
    traces, x = _regular_traces(pm)
    rh_lo = min(min(t.rh[0], t.rh_range[0]) if t.rh_range else t.rh[0] for t in pm.traces)
    rh_hi = max(max(t.rh[-1], t.rh_range[1]) if t.rh_range else t.rh[-1] for t in pm.traces)
    x0, x1 = float(pm.x[0]), float(pm.x[-1])
    nx = max(4 * len(x), 300) if len(x) > 1 else 1
    xf = np.linspace(x0, x1, nx) if len(x) > 1 else x.copy()
    rg = np.linspace(rh_lo, rh_hi, rh_points)
    Z = np.full((rh_points, len(xf)), np.nan)
    for j, xv in enumerate(xf):
        k = int(np.clip(np.searchsorted(x, xv) - 1, 0, max(len(x) - 2, 0)))
        ta = traces[k]
        tb = traces[k + 1] if len(x) > 1 else ta
        if len(x) > 1:
            w = (xv - x[k]) / (x[k + 1] - x[k])
            near, far = (ta, tb) if w <= 0.5 else (tb, ta)
            # boundaries of the nearer trace; a boundary of the same kind (same states below and above) in the other
            # trace is interpolated linearly in x, so boundaries stay smooth also where the topology changes
            far_rh = {}
            for bb in far.boundaries:
                far_rh.setdefault((bb.below, bb.above), []).append(bb.rh)
            ivs = []
            lo = rh_lo
            wn = w if near is ta else 1.0 - w
            for bn in near.boundaries:
                key = (bn.below, bn.above)
                r = bn.rh
                cand = far_rh.get(key, [])
                if cand:                                       # the same kind of boundary closest in RH
                    q = int(np.argmin([abs(c - bn.rh) for c in cand]))
                    r = (1 - wn) * bn.rh + wn * cand.pop(q)
                r = max(r, lo)
                ivs.append((lo, r, bn.below))
                lo = r
            ivs.append((lo, rh_hi, near.intervals()[-1][2]))
        else:
            ivs = ta.intervals()
        for a, b, st in ivs:
            Z[(rg >= a) & (rg <= b), j] = idx[st]
    return Z, xf, rg, states


def _tex_formula(f: str) -> str:
    """'(NH4)2SO4' -> '(NH$_4$)$_2$SO$_4$' (digits after an element or a bracket become subscripts)."""
    import re
    return re.sub(r"(?<=[A-Za-z)])(\d+)", r"$_{\1}$", f).replace(".", "·")


def plot_phase_map(pm: PhaseMap, ax=None, *, colors: dict | None = None, rh_points: int = 400, legend: bool = True,
                   metastable: PhaseMap | None = None):
    """X--RH diagram: regions coloured by phase state and outlined, states that exist on a single composition only
    as dotted lines, optional metastable boundaries (``metastable``, dotted grey).  Returns the matplotlib axes."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    from .solids import SOLIDS

    if ax is None:
        _, ax = plt.subplots(figsize=(6.4, 4.8))
    states = [st for st in pm.states() if st not in {ln[3] for ln in singular_lines(pm)}]
    palette = ["#f2f0e6", "#d9e7f5", "#cfe8d4", "#f6dcc8", "#e4d6ef", "#f4e7b3", "#d6e9e7", "#ecd1d8", "#e0e0e0"]
    colors = dict(colors or {})
    for k, s in enumerate(states):
        colors.setdefault(s, palette[k % len(palette)])
    Z, xf, rg, _ = _region_grid(pm, rh_points, states)
    x = pm.x
    rh_lo, rh_hi = rg[0], rg[-1]
    cmap = ListedColormap([colors[s] for s in states])
    xe = np.concatenate([[xf[0]], 0.5 * (xf[1:] + xf[:-1]), [xf[-1]]]) if len(xf) > 1 else np.array([xf[0] - 0.5,
                                                                                                    xf[0] + 0.5])
    ax.pcolormesh(xe, np.concatenate([[rg[0]], 0.5 * (rg[1:] + rg[:-1]), [rg[-1]]]), Z, cmap=cmap,
                  vmin=-0.5, vmax=len(states) - 0.5, shading="flat")

    # boundaries: outlines of the regions of the grid (horizontal and vertical boundaries alike, consistent with
    # the colours); states that exist on a single composition line only are drawn as dotted lines
    xc, yc = np.meshgrid(xf, rg)
    for k in range(len(states)):
        ind = (Z == k).astype(float)
        if ind.any() and not ind.all():
            ax.contour(xc, yc, ind, levels=[0.5], colors="#222222", linewidths=1.3)
    for xv, a, b, st in singular_lines(pm):
        ax.plot([xv, xv], [a, b], ":", color="#222222", lw=1.3)
    if metastable is not None:
        for (below, above), (bx, br) in metastable.boundary_lines().items():
            o = np.argsort(bx)
            ax.plot(bx[o], br[o], ":", color="#555555", lw=1.2)
    ax.set_xlim(xe[0], xe[-1])
    ax.set_ylim(rh_lo, rh_hi)
    ax.set_xlabel(pm.x_label)
    ax.set_ylabel("Relative humidity")
    if legend:
        def nice(st):
            parts = (["L"] if st.n_liquids == 1 else [f"{st.n_liquids}L"] if st.n_liquids > 1 else [])
            parts += [_tex_formula(SOLIDS[k].formula) if k in SOLIDS else k for k in st.solids]
            return " + ".join(parts) or "empty"
        ax.legend(handles=[Patch(facecolor=colors[s], edgecolor="#999999", label=nice(s)) for s in states],
                  fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    return ax


# ---------------------------------------------------------------------------------------------------------------
# particle properties along RH (deliquescence curves, contour fields)
# ---------------------------------------------------------------------------------------------------------------
M_WATER = 18.015                                    # g/mol


def _molar_masses(pe: PhaseEquilibrium) -> dict:
    """g/mol of every component of ``pe`` (water, organics, ions)."""
    from .params import load_subgroup_params
    from .solids import ION_REGISTRY
    sg = load_subgroup_params()
    out = {"Water": M_WATER}
    mm = np.asarray(pe.lm._mm, float)                  # neutral molar masses of the liquid model [kg/mol]
    for k, c in enumerate(pe._organics):
        out[c.name] = 1000.0 * float(mm[1 + k])
    for ion in pe._ions:
        no, z = ION_REGISTRY[ion]
        out[ion] = float(sg.SMWC[no - 201] if z > 0 else sg.SMWA[no - 241])
    return out


def particle_properties(pe: PhaseEquilibrium, feed: dict, res: PhaseEquilibriumResult) -> dict:
    """Water content and acidity of an equilibrium state.

    ``rel_mass``: particle mass relative to the dry particle of the same feed, W_p / W_dry = 1 + W_water / W_dry
    (the 'relative particle mass' of Amundson et al., 2006); ``water_g``: liquid water [g]; ``pH``: -log10 a_H+
    (molal) in the liquid holding most water (NaN without H+ or liquid); ``n_liquids``; ``solids``."""
    mm = _molar_masses(pe)
    w_dry = sum(v * mm[k] for k, v in feed.items())
    water = sum(float(L.amounts[0]) for L in res.liquids) * M_WATER
    ph = float("nan")
    if res.liquids and "H+" in pe.names:
        L = max(res.liquids, key=lambda L_: float(L_.amounts[0]))
        ln_a = dict(zip(L.names, L.ln_a)).get("H+")
        if ln_a is not None and np.isfinite(ln_a):
            ph = -float(ln_a) / np.log(10.0)
    return {"rel_mass": 1.0 + water / w_dry, "water_g": water, "pH": ph, "n_liquids": len(res.liquids),
            "solids": tuple(sorted(res.solids)), "status": res.status}


def rh_profile(pe: PhaseEquilibrium, feed: dict, rh_grid: Sequence[float], *, mode: str = "equilibrium",
               **solve_kw) -> dict:
    """Properties (:func:`particle_properties`) on ``rh_grid``, solved from high to low RH with warm starts;
    returned in the order of ``rh_grid`` as arrays (NaN where a solve did not converge)."""
    rh = np.asarray(rh_grid, float)
    order = np.argsort(-rh)
    out = {k: np.full(len(rh), np.nan) for k in ("rel_mass", "water_g", "pH", "n_liquids")}
    out["solids"] = [()] * len(rh)
    prev = None
    for i in order:
        res = _solve(pe, feed, float(rh[i]), mode, prev, dict(solve_kw))
        if not _ok(res) and prev is not None:
            res = _solve(pe, feed, float(rh[i]), mode, None, dict(solve_kw))
        if _ok(res):
            p = particle_properties(pe, feed, res)
            for k in ("rel_mass", "water_g", "pH", "n_liquids"):
                out[k][i] = p[k]
            out["solids"][i] = p["solids"]
            prev = res if res.liquids else None
        else:
            prev = None
    out["rh"] = rh
    return out


def deliquescence_point(pe: PhaseEquilibrium, feed: dict, *, rh_max: float = 0.98, rh_min: float = 0.02,
                        step: float = 0.03, tol: float = 1.0e-4, method: str = "equilibrium", **solve_kw):
    """Highest RH at which a solid is present (the full deliquescence RH, i.e. the water activity of the saturated
    solution) and the solids just below it.  Returns ``(rh, solids)``, or ``(None, ())`` if the particle is liquid
    down to ``rh_min``.

    ``method="equilibrium"``: RH is scanned downward in steps of ``step`` with full equilibrium solves (warm starts),
    then bisected to ``tol``.  ``method="si"``: the solid-free (metastable) liquid is followed instead, and the RH at
    which the largest saturation index of the candidate solids reaches zero is located by Brent's method; the solid
    returned is the one that saturates first.  The two agree when the particle has a single liquid phase above the
    deliquescence RH (inorganic systems); ``"si"`` is several times faster."""
    if method == "si":
        return _deliquescence_point_si(pe, feed, rh_max, rh_min, step, tol, solve_kw)
    hi_res, lo_res, rh = None, None, rh_max
    prev = None
    while rh >= rh_min - 1e-12:
        res = _solve(pe, feed, rh, "equilibrium", prev, dict(solve_kw))
        if not _ok(res) and prev is not None:
            res = _solve(pe, feed, rh, "equilibrium", None, dict(solve_kw))
        if _ok(res):
            if res.solids:
                lo_res = (rh, res)
                break
            hi_res = (rh, res)
            prev = res if res.liquids else None
        rh -= step
    if lo_res is None:
        return None, ()
    if hi_res is None:
        return rh_max, tuple(sorted(lo_res[1].solids))
    lo, hi = lo_res[0], hi_res[0]
    solids = tuple(sorted(lo_res[1].solids))
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        res = _solve(pe, feed, mid, "equilibrium", hi_res[1], dict(solve_kw))
        if not _ok(res):
            res = _solve(pe, feed, mid, "equilibrium", None, dict(solve_kw))
        if not _ok(res):
            break
        if res.solids:
            lo, solids = mid, tuple(sorted(res.solids))
        else:
            hi, hi_res = mid, (mid, res)
    return 0.5 * (lo + hi), solids


def _deliquescence_point_si(pe, feed, rh_max, rh_min, step, tol, solve_kw):
    from scipy.optimize import brentq
    cache = {}
    prev = [None]

    def max_si(rh):
        if rh in cache:
            return cache[rh][0]
        res = _solve(pe, feed, rh, "metastable", prev[0], dict(solve_kw))
        if not _ok(res):
            res = _solve(pe, feed, rh, "metastable", None, dict(solve_kw))
        if not _ok(res) or not res.si:
            cache[rh] = (float("nan"), None)
            return float("nan")
        key = max(res.si, key=res.si.get)
        cache[rh] = (float(res.si[key]), key)
        prev[0] = res if res.liquids else prev[0]
        return cache[rh][0]

    hi = None
    rh = rh_max
    while rh >= rh_min - 1e-12:
        v = max_si(rh)
        if np.isfinite(v):
            if v >= 0.0:
                if hi is None:
                    return rh_max, (cache[rh][1],)
                r = brentq(lambda t: max_si(t) if np.isfinite(max_si(t)) else 0.0, rh, hi, xtol=tol)
                return float(r), (cache[min(cache, key=lambda k: abs(k - r))][1],)
            hi = rh
        rh = round(rh - step, 12)
    return None, ()


def label_regions(ax, pm: PhaseMap, labels: dict, *, min_cells: int = 40, fontsize: int = 8, rh_points: int = 400):
    """Write a short label in every phase region of ``pm`` drawn on ``ax`` (``labels``: solid key -> letter, e.g.
    {"ammonium_sulfate": "A"}; a region reads 'L+A+E').  Regions smaller than ``min_cells`` grid cells are skipped."""
    states = [st for st in pm.states() if st not in {ln[3] for ln in singular_lines(pm)}]
    Z, xf, rg, states = _region_grid(pm, rh_points, states)
    for k, st in enumerate(states):
        mask = Z == k
        if mask.sum() < min_cells:
            continue
        iy, ix = np.nonzero(mask)
        parts = (["L"] if st.n_liquids == 1 else [f"{st.n_liquids}L"] if st.n_liquids > 1 else [])
        parts += [labels.get(s, s) for s in st.solids]
        yc, xc = np.median(rg[iy]), np.median(xf[ix])
        # move the label to a cell of the region nearest to the median point
        d = (xf[ix] - xc) ** 2 / max(np.ptp(xf), 1e-12) ** 2 + (rg[iy] - yc) ** 2 / max(np.ptp(rg), 1e-12) ** 2
        q = int(np.argmin(d))
        ax.text(xf[ix[q]], rg[iy[q]], "+".join(parts) if parts else "", ha="center", va="center", fontsize=fontsize)
    for xv, a, b, st in singular_lines(pm):
        parts = (["L"] if st.n_liquids == 1 else []) + [labels.get(t, t) for t in st.solids]
        ax.text(xv, 0.5 * (a + b), " " + "+".join(parts), ha="left", va="center", fontsize=fontsize - 1)

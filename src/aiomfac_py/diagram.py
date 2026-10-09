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
           "pie_composition", "particle_properties", "rh_profile", "deliquescence_point", "label_regions",
           "singular_lines", "BoundaryCurve", "BoundaryCurves", "trace_boundaries", "plot_boundary_curves",
           "binary_mixing_curve", "TernaryLLE", "ternary_lle", "plot_ternary"]


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


def _solve_limited(pe, feed, rh, mode, init, solve_kw, timeout):
    """:func:`_solve` that gives up after ``timeout`` seconds (reported as not converged).  Uses SIGALRM, so it applies
    only in the main thread on POSIX systems; elsewhere the solve is not limited."""
    import signal
    import threading
    if not timeout or not hasattr(signal, "setitimer") or threading.current_thread() is not threading.main_thread():
        return _solve(pe, feed, rh, mode, init, solve_kw)

    class _Slow(Exception):
        pass

    def alarm(*_):
        raise _Slow()
    old = signal.signal(signal.SIGALRM, alarm)
    signal.setitimer(signal.ITIMER_REAL, float(timeout))
    try:
        return _solve(pe, feed, rh, mode, init, solve_kw)
    except _Slow:
        return PhaseEquilibriumResult("not_converged", pe.T, rh, [], {}, {}, float("nan"), {}, 0.0,
                                      message="time limit")
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, old)


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


def _region_grid(pm: PhaseMap, rh_points: int = 400, states: list | None = None, x_points: int = 300):
    """State index on a fine (RH, x) grid: the boundaries of the nearer of the two neighbouring traces, each
    interpolated linearly in x towards the boundary of the same kind (same states below and above, closest in RH) of
    the other trace when it has one.  Returns (Z, x grid, RH grid, states)."""
    states = states if states is not None else pm.states()
    idx = {s: k for k, s in enumerate(states)}
    traces, x = _regular_traces(pm)
    rh_lo = min(min(t.rh[0], t.rh_range[0]) if t.rh_range else t.rh[0] for t in pm.traces)
    rh_hi = max(max(t.rh[-1], t.rh_range[1]) if t.rh_range else t.rh[-1] for t in pm.traces)
    x0, x1 = float(pm.x[0]), float(pm.x[-1])
    nx = max(4 * len(x), x_points) if len(x) > 1 else 1
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


def plot_phase_map(pm: PhaseMap, ax=None, *, colors: dict | None = None, rh_points: int = 400, x_points: int = 300,
                   legend: bool = True, metastable: PhaseMap | None = None, curves: "BoundaryCurves | None" = None,
                   show_failed: bool = False):
    """X--RH diagram: regions coloured by phase state and outlined, states that exist on a single composition only
    as dotted lines, optional metastable boundaries (``metastable``, dotted grey).  ``rh_points`` and ``x_points``
    set the resolution of the region grid (raise them for zoomed views).  With ``curves`` (from
    :func:`trace_boundaries`) the boundaries are drawn as the traced lines and the regions are filled between them.
    ``show_failed`` marks the compositions and RH where solves did not converge (grey dots), i.e. where the
    diagram is not resolved.  Returns the matplotlib axes."""
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
    Z, xf, rg, _ = _region_grid(pm, rh_points, states, x_points)
    if curves is not None:
        Z = _curve_grid(curves, xf, rg, Z, {s: k for k, s in enumerate(states)})
    rh_lo, rh_hi = rg[0], rg[-1]
    cmap = ListedColormap([colors[s] for s in states])
    xe = np.concatenate([[xf[0]], 0.5 * (xf[1:] + xf[:-1]), [xf[-1]]]) if len(xf) > 1 else np.array([xf[0] - 0.5,
                                                                                                    xf[0] + 0.5])
    ax.pcolormesh(xe, np.concatenate([[rg[0]], 0.5 * (rg[1:] + rg[:-1]), [rg[-1]]]), Z, cmap=cmap,
                  vmin=-0.5, vmax=len(states) - 0.5, shading="flat")

    # boundaries: outlines of the regions of the grid (horizontal and vertical boundaries alike, consistent with
    # the colours); states that exist on a single composition line only are drawn as dotted lines
    if curves is not None:
        plot_boundary_curves(curves, ax)
    if show_failed:
        pts = [(float(xv), r) for xv, t in zip(pm.x, pm.traces) for r in t.failed]
        if curves is not None and len(getattr(curves, "failed", ())):
            pts += [tuple(q) for q in curves.failed]
        if pts:
            q = np.array(pts)
            ax.plot(q[:, 0], q[:, 1], ".", color="#888888", ms=2.5, zorder=3)
    else:
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
# continuation of the boundary lines in the (x, RH) plane
# ---------------------------------------------------------------------------------------------------------------
@dataclass
class BoundaryCurve:
    """A boundary line between two phase states, as a polyline in the (x, RH) plane.  ``left`` is the state on the
    left-hand side of the direction of the points (for points with increasing x: the state at higher RH)."""
    left: PhaseState
    right: PhaseState
    x: np.ndarray
    rh: np.ndarray
    ends: tuple = ("", "")                 # how each end stopped: "edge", "junction" or "merged"

    def sides(self, i: int = -1) -> tuple:
        """(state at lower RH, state at higher RH) at segment ``i`` (default: the middle segment); for a vertical
        segment (state at lower x, state at higher x)."""
        n = len(self.x)
        i = n // 2 - 1 if i == -1 else i
        i = int(np.clip(i, 0, n - 2))
        dx, dr = self.x[i + 1] - self.x[i], self.rh[i + 1] - self.rh[i]
        if abs(dx) >= 1e-12 * max(abs(dr), 1.0) and dx != 0:
            return (self.right, self.left) if dx > 0 else (self.left, self.right)
        return (self.left, self.right) if dr > 0 else (self.right, self.left)

    @property
    def kind(self) -> str:
        a, b = self.sides()
        return Boundary(0.0, 0.0, 0.0, a, b).kind


@dataclass
class BoundaryCurves:
    """Result of :func:`trace_boundaries`: the boundary lines of a phase map, traced in the (x, RH) plane."""
    curves: list
    x_range: tuple
    rh_range: tuple
    n_solves: int = 0
    seeds_skipped: int = 0
    failed: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))   # (x, rh) of solves that did not converge

    def to_records(self) -> list:
        return [{"left": c.left.label, "right": c.right.label, "kind": c.kind, "ends": list(c.ends),
                 "x": c.x.tolist(), "rh": c.rh.tolist()} for c in self.curves]


class _Tracer:
    """Continuation of a boundary line by predictor (secant along the line) and corrector (bisection along the
    normal), in coordinates scaled to the unit square."""

    def __init__(self, pe, feed_of_x, x_range, rh_range, mode, tol, h0, h_min, h_max, delta_max, solve_kw):
        self.pe, self.feed_of_x, self.mode, self.solve_kw = pe, feed_of_x, mode, solve_kw
        self.x0, self.x1 = x_range
        self.r0, self.r1 = rh_range
        self.tol, self.h0, self.h_min, self.h_max, self.delta_max = tol, h0, h_min, h_max, delta_max
        self.cache: dict = {}
        self.n_solves = 0
        self.debug = False
        self.timeout = None
        self.checkpoint = None
        self.checkpoint_every = 200

    def to_xr(self, p):
        return self.x0 + p[0] * (self.x1 - self.x0), self.r0 + p[1] * (self.r1 - self.r0)

    def to_unit(self, x, rh):
        return np.array([(x - self.x0) / (self.x1 - self.x0), (rh - self.r0) / (self.r1 - self.r0)])

    @staticmethod
    def inside(p, eps=1e-9):
        return -eps <= p[0] <= 1 + eps and -eps <= p[1] <= 1 + eps

    def state(self, p, init=None):
        """(PhaseState or None if the solve failed, result) at the unit-square point p."""
        key = (round(float(p[0]), 10), round(float(p[1]), 10))
        if key in self.cache:
            return self.cache[key]
        x, rh = self.to_xr(np.clip(p, 0.0, 1.0))
        feed = self.feed_of_x(float(x))
        res = _solve_limited(self.pe, feed, float(rh), self.mode, init, dict(self.solve_kw), self.timeout)
        self.n_solves += 1
        if not _ok(res) and init is not None and res.message != "time limit":
            res = _solve_limited(self.pe, feed, float(rh), self.mode, None, dict(self.solve_kw), self.timeout)
            self.n_solves += 1
        if not _ok(res):                                    # warm start from the nearest converged solves
            near = sorted(((np.hypot(k[0] - key[0], k[1] - key[1]), v[1]) for k, v in self.cache.items()
                           if v[0] is not None and v[1].liquids), key=lambda z: z[0])
            for d, r0 in near[:1]:
                if d > 0.1 or r0 is init:
                    break
                res = _solve_limited(self.pe, feed, float(rh), self.mode, r0, dict(self.solve_kw), self.timeout)
                self.n_solves += 1
                if _ok(res):
                    break
        out = (phase_state(res) if _ok(res) else None, res)
        self.cache[key] = out
        if self.checkpoint is not None and self.n_solves % self.checkpoint_every == 0:
            self.checkpoint(self.cache)
        return out

    def locate(self, q, n, L, R, delta, warm):
        """Point on the line q + s n where the state changes from R (s < 0) to L (s > 0); None if the bracket
        holds another state (the boundary ends) or cannot be found within ``delta_max``."""
        d = delta
        while True:
            pp, pm_ = q + d * n, q - d * n
            if not (self.inside(pp) and self.inside(pm_)):
                return None
            sp, rp = self.state(pp, warm.get(L))
            sm, rm = self.state(pm_, warm.get(R))
            if sp == L and sm == R:
                break
            if (sp is not None and sp not in (L, R)) or (sm is not None and sm not in (L, R)):
                return None
            d *= 2.0
            if d > self.delta_max:
                return None
        if rp.liquids:
            warm[L] = rp
        if rm.liquids:
            warm[R] = rm
        lo, hi = -d, d
        while hi - lo > self.tol:
            seen = set()
            for frac in (0.5, 0.3, 0.7):                    # off-centre retries: a failed solve, or a state that
                m = lo + frac * (hi - lo)                   # exists on a single line only (e.g. one salt exactly)
                s, r = self.state(q + m * n, warm.get(L) or warm.get(R))
                if s in (L, R):
                    break
                seen.add(s)
            if s not in (L, R):
                if seen == {None}:                          # the bracket cannot be narrowed further
                    break
                return None                                 # another state between L and R: the boundary ends
            if s == L:
                hi = m
                if r.liquids:
                    warm[L] = r
            elif s == R:
                lo = m
                if r.liquids:
                    warm[R] = r
            else:
                return None
        return q + 0.5 * (lo + hi) * n, 0.5 * (hi - lo)

    def follow(self, p0, t0, L, R, warm, others):
        """Points from p0 in the direction t0 until the boundary ends (junction), leaves the domain (edge) or runs
        into an already traced curve between the same states (merged)."""
        pts = [np.asarray(p0, float)]
        t = np.asarray(t0, float) / np.linalg.norm(t0)
        h, corr = self.h0, self.h0 / 4
        while True:
            if len(pts) >= 2:
                t = pts[-1] - pts[-2]
                t /= np.linalg.norm(t)
            q = pts[-1] + h * t
            edge = False
            if not self.inside(q):                          # shorten the step to end on the domain edge
                s = min(((0.0 if t[k] < 0 else 1.0) - pts[-1][k]) / t[k] for k in (0, 1) if abs(t[k]) > 1e-14)
                if s < 0.5 * self.h_min:
                    return pts, "edge"
                q, h, edge = pts[-1] + s * t, s, True
                q = np.clip(q, 0.0, 1.0)
            n = np.array([-t[1], t[0]])
            delta = float(np.clip(3.0 * corr, 2.0 * self.tol, max(0.5 * h, 2.0 * self.tol)))
            r = self.locate(q, n, L, R, delta, warm)
            ok = r is not None
            if ok:
                p_new, _ = r
                seg = p_new - pts[-1]
                cos_max = np.cos(np.radians(25 if len(pts) >= 2 else 80))   # the first tangent is only a guess
                ok = np.linalg.norm(seg) > 0.25 * h and np.dot(seg, t) / np.linalg.norm(seg) > cos_max
            if not ok:
                if self.debug:
                    print(f"    fail h={h:.4f} solves={self.n_solves}", flush=True)
                if h <= self.h_min * 1.0001:
                    return pts, "junction"
                h = max(0.5 * h, self.h_min)
                continue
            corr = float(abs(np.dot(p_new - q, n)))
            pts.append(p_new)
            if self.debug:
                xr = self.to_xr(p_new)
                print(f"    ({xr[0]:.4f}, {xr[1]:.4f}) h={h:.4f} corr={corr:.1e} solves={self.n_solves}", flush=True)
            for c in others:
                if {c[0], c[1]} == {L, R} and _dist_to_polyline(p_new, c[2]) < 2.0 * self.h_min:
                    return pts, "merged"
            if edge and not self.inside(p_new + 0.5 * self.h_min * t):
                return pts, "edge"
            h = min(1.6 * h, self.h_max) if corr < 0.25 * delta else h

    def curve(self, p0, t0, L, R, warm, others):
        """Both directions from a seed point on the boundary between L (left of t0) and R."""
        fwd, e1 = self.follow(p0, t0, L, R, dict(warm), others)
        bwd, e0 = self.follow(p0, -np.asarray(t0, float), R, L, dict(warm), others)
        pts = np.array(bwd[::-1] + fwd[1:])
        return pts, (e0, e1)


def _dist_to_polyline(p, P) -> float:
    if len(P) == 1:
        return float(np.linalg.norm(p - P[0]))
    a, b = P[:-1], P[1:]
    ab = b - a
    w = np.clip(np.einsum("ij,ij->i", p - a, ab) / np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-300), 0.0, 1.0)
    return float(np.min(np.linalg.norm(a + w[:, None] * ab - p, axis=1)))


def _crosses_h(P, xa, xb, y) -> bool:
    """Does the polyline P cross the horizontal segment from (xa, y) to (xb, y)?"""
    for (x1, y1), (x2, y2) in zip(P[:-1], P[1:]):
        if (y1 - y) * (y2 - y) <= 0 and y1 != y2:
            xc = x1 + (y - y1) / (y2 - y1) * (x2 - x1)
            if min(xa, xb) - 1e-9 <= xc <= max(xa, xb) + 1e-9:
                return True
        elif y1 == y2 == y and max(x1, x2) >= min(xa, xb) and min(x1, x2) <= max(xa, xb):
            return True
    return False


def trace_boundaries(pe: PhaseEquilibrium, feed_of_x: Callable[[float], dict], pm: PhaseMap, *, tol: float = 5.0e-4,
                     h0: float = 0.02, h_min: float = 0.002, h_max: float = 0.05, delta_max: float = 0.05,
                     rh_samples: int = 40, snap: float | None = None, solve_timeout: float | None = None,
                     x_lines: Sequence[float] = (), junction_rounds: int = 2, cache: dict | None = None,
                     checkpoint: Callable[[dict], None] | None = None, verbose: bool = False,
                     **solve_kw) -> BoundaryCurves:
    """Trace every boundary line of the phase map ``pm`` (from :func:`phase_map` with the same ``pe`` and
    ``feed_of_x``) continuously in the (x, RH) plane.

    Lengths are in units of the diagram (x and RH ranges scaled to 1).  From each boundary point of the traces,
    the line is followed in both directions: the next point is predicted along the secant of the last two points
    (step ``h``, adapted between ``h_min`` and ``h_max``) and corrected by bisection along the normal to the line
    until the bracket is narrower than ``tol``.  So boundaries of any slope, including vertical ones, are traced
    with the same accuracy, and only points close to the line are solved.  A line ends where the two states no
    longer meet (a junction with other boundaries, located to within ``h_min``), at the edge of the diagram, or
    where it runs into a line already traced.  Boundaries that cross no trace (e.g. vertical ones) are seeded by
    bisection in x between neighbouring traces at ``rh_samples`` RH levels.  Line ends at junctions closer than
    ``snap`` (default ``2.5 h_min``) are joined.  ``solve_timeout`` [s] stops solves that take longer (counted as
    not converged; POSIX main thread only), for compositions where the solver is very slow.

    Vertical lines (a boundary at fixed x, e.g. at the stoichiometric composition of a salt) are built from two RH
    scans just left and right of the line, which also resolves thin states next to it; they are detected among the
    seeds between traces, and compositions given in ``x_lines`` are scanned in any case.  After the seeds, the states
    on a small circle around every line end at a junction are checked (``junction_rounds`` times): a change of state
    that no traced line explains starts a new line (short lines between junctions that cross no trace).  ``cache``:
    a dict of solves, kept and reused between calls with the same ``pm``; ``checkpoint(cache)`` is called every 200
    solves (e.g. to save the cache of a long run).  Other keywords go to
    :meth:`PhaseEquilibrium.solve`."""
    traces, xs = _regular_traces(pm)
    x_range = (float(pm.x[0]), float(pm.x[-1]))
    rh_range = (min(t.rh_range[0] if t.rh_range else t.rh[0] for t in pm.traces),
                max(t.rh_range[1] if t.rh_range else t.rh[-1] for t in pm.traces))
    tr = _Tracer(pe, feed_of_x, x_range, rh_range, pm.mode, tol, h0, h_min, h_max, delta_max, solve_kw)
    tr.debug = verbose > 1
    tr.timeout = solve_timeout
    if cache is not None:
        tr.cache = cache
    tr.checkpoint = checkpoint
    done: list = []                                         # (L, R, points, ends)
    skipped = 0

    def report(L, R, pts, ends, what=""):
        if verbose:
            a, b = (tr.to_xr(pts[0]), tr.to_xr(pts[-1]))
            print(f"{what}{R.label} | {L.label}: {len(pts)} points from ({a[0]:.3f}, {a[1]:.3f}) to "
                  f"({b[0]:.3f}, {b[1]:.3f}), ends {ends}, {tr.n_solves} solves", flush=True)

    def run_seed(p, t0, L, R, warm, dedupe=5.0):
        nonlocal skipped
        for c in done:
            if {c[0], c[1]} == {L, R} and _dist_to_polyline(p, c[2]) < dedupe * h_min:
                skipped += 1
                return True
        pts, ends = tr.curve(p, t0, L, R, warm, done)
        if len(pts) < 2:
            return False
        done.append((L, R, pts, ends))
        report(L, R, pts, ends)
        return True

    def column(xu, n=24):
        """states along RH at the unit x ``xu``: (state at the bottom, [(v, state below, state above)])"""
        vs = np.linspace(0.0, 1.0, n)
        pts = [(v, s) for v in vs for s in [tr.state(np.array([xu, v]))[0]] if s is not None]
        out = []

        def bis(a, sa, b, sb):
            if b - a <= tol:
                out.append((0.5 * (a + b), sa, sb))
                return
            m = 0.5 * (a + b)
            sm = tr.state(np.array([xu, m]))[0]
            if sm is None:
                out.append((0.5 * (a + b), sa, sb))
                return
            if sm != sa:
                bis(a, sa, m, sm)
            if sm != sb:
                bis(m, sm, b, sb)
        for (a, sa), (b, sb) in zip(pts[:-1], pts[1:]):
            if sa != sb:
                bis(a, sa, b, sb)
        return (pts[0][1] if pts else None), sorted(out, key=lambda z: z[0])

    def state_in(col, v):
        st = col[0]
        for vb, _, above in col[1]:
            if vb < v:
                st = above
        return st

    vertical: list = []

    def vertical_line(xu):
        """the boundary at the unit x ``xu`` from RH scans just left and right of it"""
        eps = 2.0 * tol
        if any(abs(xu - v) < 3.0 * h_min for v in vertical) or not 2.0 * eps < xu < 1.0 - 2.0 * eps:
            return                                          # (a line on the edge of the diagram is not drawn)
        vertical.append(xu)
        left, right = column(max(xu - eps, 0.0)), column(min(xu + eps, 1.0))
        cuts = sorted({0.0, 1.0} | {z[0] for z in left[1]} | {z[0] for z in right[1]})
        segs: list = []                                     # [v0, v1, L, R]
        for a, b in zip(cuts[:-1], cuts[1:]):
            m = 0.5 * (a + b)
            sl, sr = state_in(left, m), state_in(right, m)
            if sl is None or sr is None or sl == sr:
                continue
            if segs and segs[-1][1] == a and segs[-1][2:] == [sl, sr]:
                segs[-1][1] = b
            else:
                segs.append([a, b, sl, sr])
        for v0, v1, sl, sr in segs:
            if v1 - v0 < 2.0 * h_min:                       # a sloped line crossing between the two scans
                continue
            pts = np.array([[xu, v0], [xu, v1]])
            ends = ("edge" if v0 <= 0.0 else "junction", "edge" if v1 >= 1.0 else "junction")
            done.append((sl, sr, pts, ends))                # upwards: the left side is at lower x
            report(sl, sr, pts, ends, "vertical: ")

    def is_vertical(xu, v):
        """does the change of state at (xu, v) continue at the same x 0.03 above or below?"""
        eps = 2.0 * tol
        for v2 in (v + 0.03, v - 0.03):
            if 0.0 <= v2 <= 1.0:
                a = tr.state(np.array([max(xu - eps, 0.0), v2]))[0]
                b = tr.state(np.array([min(xu + eps, 1.0), v2]))[0]
                if a is None or b is None or a == b:
                    return False
        return True

    for xv in x_lines:
        vertical_line(float(tr.to_unit(xv, rh_range[0])[0]))

    # seeds 1: the boundary points of the traces (followed in x)
    for xv, t in zip(xs, traces):
        for b in t.boundaries:
            if not b.resolved and b.rh_hi - b.rh_lo > 1e-2:
                continue
            i_lo = int(np.argmin(np.abs(t.rh - b.rh_lo)))
            i_hi = int(np.argmin(np.abs(t.rh - b.rh_hi)))
            warm = {b.below: t.results[i_lo], b.above: t.results[i_hi]}
            run_seed(tr.to_unit(xv, b.rh), np.array([1.0, 0.0]), b.above, b.below, warm)

    # seeds 2: changes of state between neighbouring traces at the same RH that no traced line explains
    levels = np.linspace(rh_range[0], rh_range[1], rh_samples + 2)[1:-1]
    for (xa, ta), (xb, tb) in zip(zip(xs[:-1], traces[:-1]), zip(xs[1:], traces[1:])):
        near = [b.rh for b in ta.boundaries + tb.boundaries]
        tried: set = set()                                  # (state a, state b) whose line could not be followed
        for r in levels:
            if near and min(abs(r - q) for q in near) < 2e-3 * (rh_range[1] - rh_range[0]):
                continue
            sa, sb = _state_at(ta, r), _state_at(tb, r)
            if sa == sb or (sa, sb) in tried:
                continue
            ua, ub = tr.to_unit(xa, r), tr.to_unit(xb, r)
            if any(_crosses_h(c[2], ua[0], ub[0], ua[1]) for c in done):
                continue
            lo, hi, s_hi = ua[0], ub[0], sb
            while hi - lo > tol:
                for frac in (0.5, 0.3, 0.7):
                    m = lo + frac * (hi - lo)
                    s, _ = tr.state(np.array([m, ua[1]]))
                    if s in (sa, sb):
                        break
                if s is None:
                    break
                if s == sa:
                    lo = m
                else:
                    hi, s_hi = m, s
            p = np.array([0.5 * (lo + hi), ua[1]])
            _, r_lo = tr.state(np.array([lo, ua[1]]))
            _, r_hi = tr.state(np.array([hi, ua[1]]))
            if s_hi is None or s_hi == sa:
                tried.add((sa, sb))
                continue
            if is_vertical(p[0], p[1]):
                vertical_line(p[0])
                continue
            if not run_seed(p, np.array([0.0, 1.0]), sa, s_hi, {sa: r_lo, s_hi: r_hi}):
                tried.add((sa, sb))

    # line ends at junctions: a change of state on a small circle around the end that no line explains
    rho, n_ang = 3.0 * h_min, 16
    visited: list = []
    for _ in range(junction_rounds):
        found = 0
        ends_pts = [c[2][0 if side == 0 else -1] for c in done for side in (0, 1) if c[3][side] == "junction"]
        for J in ends_pts:
            if any(np.linalg.norm(J - q) < 2.5 * h_min for q in visited):
                continue
            visited.append(J)
            ang = 2.0 * np.pi * np.arange(n_ang) / n_ang
            P = J + rho * np.c_[np.cos(ang), np.sin(ang)]
            S = [tr.state(q)[0] if tr.inside(q) else None for q in P]
            for k in range(n_ang):
                k2 = (k + 1) % n_ang
                sa, sb = S[k], S[k2]
                if sa is None or sb is None or sa == sb:
                    continue
                if any(_crosses_seg(c[2], P[k], P[k2]) for c in done):
                    continue
                a, b = ang[k], ang[k] + 2.0 * np.pi / n_ang
                s_hi = sb
                while (b - a) * rho > tol:
                    m = 0.5 * (a + b)
                    sm = tr.state(J + rho * np.array([np.cos(m), np.sin(m)]))[0]
                    if sm is None:
                        break
                    if sm == sa:
                        a = m
                    else:
                        b, s_hi = m, sm
                m = 0.5 * (a + b)
                B = J + rho * np.array([np.cos(m), np.sin(m)])
                n0 = len(done)
                run_seed(B, (B - J) / rho, s_hi, sa, {}, dedupe=1.0)    # outwards: the left side is s_hi
                found += len(done) - n0
        if not found:
            break

    polys = _join_junctions([c[2] for c in done], [c[3] for c in done], 2.5 * h_min if snap is None else snap)
    curves = []
    for (L, R, _, ends), pts in zip(done, polys):
        xr = np.array([tr.to_xr(p) for p in pts])
        curves.append(BoundaryCurve(L, R, xr[:, 0], xr[:, 1], ends))
    failed = np.array([tr.to_xr(np.array(k)) for k, v in tr.cache.items() if v[0] is None]).reshape(-1, 2)
    return BoundaryCurves(curves, x_range, rh_range, tr.n_solves, skipped, failed)


def _join_junctions(polys: list, ends: list, snap: float) -> list:
    """Close the gaps (up to ``h_min``) between line ends at junctions: ends that are closer than ``snap`` to each
    other are joined at the point closest (least squares) to their tangent lines; a single end is extended along its
    tangent to the nearest line it meets within ``snap`` (or joined to the nearest point of a line)."""
    polys = [np.asarray(P, float).copy() for P in polys]
    if snap <= 0:
        return polys
    items = [(k, side) for k, e in enumerate(ends) for side in (0, 1) if e[side] == "junction" and len(polys[k]) >= 2]

    def point(k, side):
        return polys[k][0] if side == 0 else polys[k][-1]

    def outward(k, side):
        A = polys[k]
        t = A[0] - A[1] if side == 0 else A[-1] - A[-2]
        return t / max(np.linalg.norm(t), 1e-300)

    parent = list(range(len(items)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for a in range(len(items)):
        for b in range(a + 1, len(items)):
            if np.linalg.norm(point(*items[a]) - point(*items[b])) < snap:
                parent[find(a)] = find(b)
    groups: dict = {}
    for a in range(len(items)):
        groups.setdefault(find(a), []).append(items[a])
    target: dict = {}
    for g in groups.values():
        es = np.array([point(*it) for it in g])
        if len(g) >= 2:
            M, v = np.zeros((2, 2)), np.zeros(2)
            for it, e in zip(g, es):
                t = outward(*it)
                n = np.array([-t[1], t[0]])
                M += np.outer(n, n)
                v += np.outer(n, n) @ e
            J = np.linalg.solve(M, v) if np.linalg.cond(M) < 1e6 else es.mean(axis=0)
            if np.max(np.linalg.norm(es - J, axis=1)) > 1.5 * snap:
                J = es.mean(axis=0)
            for it in g:
                target[it] = J
            continue
        k, side = g[0]
        e, t = es[0], outward(k, side)
        best = None
        for j, P in enumerate(polys):                       # extension along the tangent
            if j == k:
                continue
            a, b = P[:-1], P[1:]
            d = b - a
            den = t[0] * d[:, 1] - t[1] * d[:, 0]
            ok = np.abs(den) > 1e-14
            w = np.where(ok, ((a[:, 0] - e[0]) * d[:, 1] - (a[:, 1] - e[1]) * d[:, 0]) / np.where(ok, den, 1), -1)
            u = np.where(ok, ((a[:, 0] - e[0]) * t[1] - (a[:, 1] - e[1]) * t[0]) / np.where(ok, den, 1), -1)
            hit = ok & (w >= 0) & (w <= snap) & (u >= 0) & (u <= 1)
            if hit.any() and (best is None or w[hit].min() < best[0]):
                best = (w[hit].min(), e + w[hit].min() * t)
        if best is None:                                    # nearest point of another line
            for j, P in enumerate(polys):
                if j == k:
                    continue
                a, b = P[:-1], P[1:]
                ab = b - a
                w = np.clip(np.einsum("ij,ij->i", e - a, ab) / np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-300), 0, 1)
                proj = a + w[:, None] * ab
                dd = np.linalg.norm(proj - e, axis=1)
                q = int(np.argmin(dd))
                if dd[q] < snap and (best is None or dd[q] < best[0]):
                    best = (dd[q], proj[q])
        if best is not None:
            target[(k, side)] = best[1]
    for (k, side), J in target.items():
        polys[k] = np.vstack([J, polys[k]]) if side == 0 else np.vstack([polys[k], J])
    return polys



def _crosses_seg(P, a, b) -> bool:
    """Does the polyline P cross the segment from a to b?"""
    if len(P) < 2:
        return False
    p, q = P[:-1], P[1:]
    d1, d2 = q - p, b - a

    def cross(u, v):
        return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]
    den = cross(d1, d2)
    ok = np.abs(den) > 1e-14
    w = np.where(ok, cross(a - p, d2) / np.where(ok, den, 1.0), -1.0)
    u = np.where(ok, cross(a - p, d1) / np.where(ok, den, 1.0), -1.0)
    return bool(np.any(ok & (w >= 0) & (w <= 1) & (u >= 0) & (u <= 1)))

def _state_at(t: Trace, rh: float) -> PhaseState:
    for a, b, st in t.intervals():
        if a <= rh <= b:
            return st
    return t.intervals()[-1][2]


def _curve_grid(bc: BoundaryCurves, xf, rg, Z, idx):
    """Overwrite the columns of the region grid Z (x grid xf, RH grid rg) that are crossed by boundary lines with the
    states between the crossings."""
    sx = bc.x_range[1] - bc.x_range[0]
    sr = bc.rh_range[1] - bc.rh_range[0]
    for j, xv in enumerate(xf):
        cross = []                                          # (rh, state below, state above)
        for c in bc.curves:
            if np.ptp(c.x) / sx < 0.01 * max(np.ptp(c.rh) / sr, 1e-12):   # a vertical line: no state below/above
                continue
            for i in range(len(c.x) - 1):
                x1, x2 = c.x[i], c.x[i + 1]
                if x1 == x2 or not (min(x1, x2) <= xv <= max(x1, x2)):
                    continue
                w = (xv - x1) / (x2 - x1)
                lo, up = c.sides(i)
                cross.append((c.rh[i] + w * (c.rh[i + 1] - c.rh[i]), lo, up))
        if not cross:
            continue
        cross.sort(key=lambda z: z[0])
        # the states must match between consecutive crossings; otherwise a line is missing here and the column keeps
        # the interpolated states
        if any(a[2] != b[1] for a, b in zip(cross[:-1], cross[1:])):
            continue
        bounds = [rg[0] - 1.0] + [z[0] for z in cross] + [rg[-1] + 1.0]
        sts = [cross[0][1]] + [z[2] for z in cross]
        for k, st in enumerate(sts):
            if st in idx:
                Z[(rg >= bounds[k]) & (rg <= bounds[k + 1]), j] = idx[st]
    return Z


def plot_boundary_curves(bc: BoundaryCurves, ax, *, color: str = "#222222", lw: float = 1.3, **kw):
    """Draw the traced boundary lines on ``ax``."""
    for c in bc.curves:
        ax.plot(c.x, c.rh, "-", color=color, lw=lw, solid_capstyle="round", **kw)
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


def label_regions(ax, pm: PhaseMap, labels: dict, *, min_cells: int = 40, fontsize: int = 8, rh_points: int = 400,
                  curves: "BoundaryCurves | None" = None):
    """Write a short label in every phase region of ``pm`` drawn on ``ax`` (``labels``: solid key -> letter, e.g.
    {"ammonium_sulfate": "A"}; a region reads 'L+A+E').  Regions smaller than ``min_cells`` grid cells are skipped.
    Pass the ``curves`` used for the plot, if any."""
    states = [st for st in pm.states() if st not in {ln[3] for ln in singular_lines(pm)}]
    Z, xf, rg, states = _region_grid(pm, rh_points, states)
    if curves is not None:
        Z = _curve_grid(curves, xf, rg, Z, {s: k for k, s in enumerate(states)})
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
        ax.text(xv, 0.75 * a + 0.25 * b, "+".join(parts), ha="center", va="center", fontsize=fontsize - 1,
                rotation=90, bbox={"boxstyle": "square,pad=0.1", "fc": "white", "ec": "none"})


# ---------------------------------------------------------------------------------------------------------------
# liquid-liquid equilibrium of neutral mixtures from the convex hull of the Gibbs energy of mixing
# ---------------------------------------------------------------------------------------------------------------
def _gmix(model, x, T):
    """normalized Gibbs energy of mixing g = sum x_i ln a_i (pure liquids as reference) and the activities"""
    x = np.asarray(x, float)
    if x.max() > 1.0 - 1e-13:
        return 0.0, np.where(x > 0.5, 1.0, 0.0)
    xc = np.clip(x, 1e-300, None)
    a = np.asarray(model.evaluate(list(xc), T, "mole").activity[:len(x)], float)
    return float(np.sum(xc * np.log(np.clip(a, 1e-300, None)))), a


def binary_mixing_curve(components, T_K: float, *, n: int = 401) -> dict:
    """Normalized Gibbs energy of mixing g(x) = (1-x) ln a_1 + x ln a_2 of a binary mixture of two neutral components
    (x: mole fraction of the second), its equilibrium phase splits from the lower convex hull of g, and the
    equilibrium activities (constant across a two-phase region).

    Returns ``{"x", "g", "a1", "a2", "splits"}``; ``splits`` lists the coexisting compositions (x', x'') of each
    miscibility gap.  The compositions are resolved to the grid (``n`` points, refined towards both pure
    components down to 1e-12)."""
    from .model import ActivityModel
    model = ActivityModel(list(components))
    x = np.concatenate([np.logspace(-12, -3, 40), np.linspace(1e-3, 1 - 1e-3, n), 1 - np.logspace(-3, -12, 40)])
    g, a1, a2 = np.empty(len(x)), np.empty(len(x)), np.empty(len(x))
    for k, xv in enumerate(x):
        g[k], a = _gmix(model, [1 - xv, xv], T_K)
        a1[k], a2[k] = a
    hull = []
    for p in zip(x, g):                                   # lower convex hull (monotone chain)
        while len(hull) >= 2 and (hull[-1][0] - hull[-2][0]) * (p[1] - hull[-2][1]) - \
                (hull[-1][1] - hull[-2][1]) * (p[0] - hull[-2][0]) <= 0:
            hull.pop()
        hull.append(p)
    hx = np.array([h[0] for h in hull])
    splits = [(float(a), float(b)) for a, b in zip(hx[:-1], hx[1:])
              if np.sum((x > a) & (x < b)) >= 3 and b - a > 1e-3]
    a1e, a2e = a1.copy(), a2.copy()
    for a, b in splits:
        ia = int(np.argmin(np.abs(x - a)))
        inside = (x > a) & (x < b)
        a1e[inside], a2e[inside] = a1[ia], a2[ia]
    return {"x": x, "g": g, "a1": a1e, "a2": a2e, "splits": splits}


@dataclass
class TernaryLLE:
    """Liquid-liquid phase diagram of a ternary neutral mixture (see :func:`ternary_lle`).

    ``P``: compositions (x_2, x_3) of the grid; ``g``: normalized Gibbs energy of mixing there; ``facets``: the lower
    convex-hull triangles, each ``{"v": vertex indices, "cls": number of phases (1, 2, 3), "mu": ln a of the three
    components at equilibrium, "edges": edge lengths}``."""
    names: list
    T_K: float
    P: np.ndarray
    g: np.ndarray
    facets: list

    @property
    def three_phase(self) -> list:
        """compositions (x_2, x_3) of the three coexisting liquids of every three-phase triangle"""
        return [self.P[f["v"]] for f in self.facets if f["cls"] == 3]

    def tie_lines(self, max_lines: int | None = None, seed: int = 0) -> list:
        """[(x', x'')] end points (x_2, x_3) of tie lines of the two-phase regions (a random subset if
        ``max_lines``)"""
        out = []
        for f in self.facets:
            if f["cls"] == 2:
                V = self.P[f["v"]]
                q = int(np.argmin(f["edges"]))                # short edge -> its midpoint joins the third vertex
                out.append((0.5 * (V[q] + V[(q + 1) % 3]), V[(q + 2) % 3]))
        if max_lines is not None and len(out) > max_lines:
            idx = np.random.RandomState(seed).choice(len(out), max_lines, replace=False)
            out = [out[i] for i in sorted(idx)]
        return out

    def equilibrium(self, x2: float, x3: float) -> dict:
        """number of phases and equilibrium activities (a_1, a_2, a_3) of the overall composition (x_2, x_3)"""
        import matplotlib.tri as mtri
        tri = mtri.Triangulation(self.P[:, 0], self.P[:, 1], np.array([f["v"] for f in self.facets]))
        k = int(tri.get_trifinder()(x2, x3))
        if k < 0:
            raise ValueError("composition outside the triangle")
        return {"n_phases": self.facets[k]["cls"], "activities": np.exp(self.facets[k]["mu"])}


def ternary_lle(components, T_K: float, *, h: float = 0.005, long_edge: float = 0.06) -> TernaryLLE:
    """Liquid-liquid(-liquid) phase diagram of three neutral components (water and organics) at ``T_K``.

    The normalized Gibbs energy of mixing g(x) = sum x_i ln a_i is evaluated with AIOMFAC on a triangular grid of
    spacing ``h`` (with extra points down to 1e-12 near the edges), and the phase diagram is read off its lower
    convex hull, which is the global minimum of the Gibbs energy for every overall composition: a hull triangle
    with three edges longer than ``long_edge`` is a three-phase triangle, one with two long edges a two-phase
    (tie-line) element, the others one-phase.  The plane of a triangle gives the equilibrium chemical potentials, so
    the activities in the equilibrium state follow directly.  About 25 000 AIOMFAC evaluations (10-15 s) at the
    default spacing; compositions are resolved to the grid (use :func:`aiomfac_py.lle.solve_pep` to refine a
    particular split)."""
    from scipy.spatial import ConvexHull

    from .model import ActivityModel
    comps = list(components)
    if len(comps) != 3:
        raise ValueError("ternary_lle needs exactly three components")
    model = ActivityModel(comps)
    pts = set()
    n = int(round(1 / h))
    for i in range(n + 1):
        for j in range(n + 1 - i):
            pts.add((round(i * h, 10), round(j * h, 10)))
    for t in (1e-12, 1e-9, 1e-7, 1e-5, 1e-4, 1e-3, 3e-3):
        for k in range(n + 1):
            u = k * h * (1 - t)
            pts.update({(t, u), (u, t), (t, max(0.0, 1 - t - u)), (max(0.0, 1 - t - u), t)})
    P = np.array(sorted(pts))
    P = P[P[:, 0] + P[:, 1] <= 1 + 1e-12]
    g = np.array([_gmix(model, [max(1 - a - b, 0.0), a, b], T_K)[0] for a, b in P])
    hull = ConvexHull(np.column_stack([P, g]))
    facets = []
    for tri in hull.simplices[hull.equations[:, 2] < -1e-12]:
        V = P[tri]
        e = [float(np.linalg.norm(V[i] - V[(i + 1) % 3])) for i in range(3)]
        nlong = sum(v > long_edge for v in e)
        c = np.linalg.lstsq(np.column_stack([np.ones(3), V]), g[tri], rcond=None)[0]
        facets.append({"v": tri, "cls": 3 if nlong == 3 else (2 if nlong == 2 else 1),
                       "mu": np.array([c[0], c[0] + c[1], c[0] + c[2]]), "edges": e})
    return TernaryLLE([c.name for c in comps], float(T_K), P, g, facets)


def plot_ternary(td: TernaryLLE, ax=None, *, show: str = "phases", n_tie: int = 60, levels=None):
    """Right-triangle plot of a :class:`TernaryLLE` (x axis: mole fraction of component 3, y axis: component 2,
    component 1 at the origin).  ``show="phases"``: one-, two- and three-phase regions with tie lines; ``show`` = a
    component index (0, 1, 2): contours of its equilibrium activity.  Region boundaries and three-phase triangles
    are drawn in both cases."""
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri

    if ax is None:
        _, ax = plt.subplots(figsize=(5.2, 5.0))
    P, facets = td.P, td.facets
    tris = np.array([f["v"] for f in facets])
    cls = np.array([f["cls"] for f in facets])
    tri = mtri.Triangulation(P[:, 1], P[:, 0], tris)
    gx, gy = np.meshgrid(np.linspace(0, 1, 301), np.linspace(0, 1, 301))
    fi = tri.get_trifinder()(gx, gy)
    ax.plot([0, 1, 0, 0], [0, 0, 1, 0], "k-", lw=1)
    if show == "phases":
        ax.tripcolor(tri, facecolors=cls.astype(float), cmap=mcolors.ListedColormap(["#ffffff", "#e3edf7", "#f6dcc8"]),
                     vmin=0.5, vmax=3.5)
        for a, b in td.tie_lines(n_tie):
            ax.plot([a[1], b[1]], [a[0], b[0]], "--", color="#3060a0", lw=0.5)
        for c, name in ((2, "L2"), (3, "L3")):
            sel = np.nonzero(cls == c)[0]
            if len(sel):
                w = np.array([0.5 * abs(np.cross(P[t[1]] - P[t[0]], P[t[2]] - P[t[0]])) for t in tris[sel]])
                if w.sum() > 2e-3:
                    cc = (P[tris[sel]].mean(axis=1) * w[:, None]).sum(0) / w.sum()
                    ax.text(cc[1], cc[0], name, fontsize=11, ha="center", va="center")
    else:
        k = int(show)
        z = np.full(gx.shape, np.nan)
        ok = fi >= 0
        z[ok] = np.exp(np.array([f["mu"] for f in facets])[fi[ok], k])
        cs = ax.contour(gx, gy, np.ma.masked_invalid(z), levels=levels if levels is not None else
                        np.arange(0.1, 1.0, 0.1), cmap="jet", linewidths=0.8)
        ax.clabel(cs, fmt="%.1f", fontsize=6)
    for V in td.three_phase:
        V = np.vstack([V, V[:1]])
        ax.plot(V[:, 1], V[:, 0], "-", color="k", lw=1.8)
    inside = fi >= 0
    ax.tricontour(mtri.Triangulation(gx[inside], gy[inside]), cls[fi[inside]].astype(float), levels=[1.5, 2.5],
                  colors="k", linewidths=1.6)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.set_xlabel(f"mole fraction of {td.names[2]}")
    ax.set_ylabel(f"mole fraction of {td.names[1]}")
    return ax

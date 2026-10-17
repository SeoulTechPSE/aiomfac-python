"""Phase boundaries of a composition--RH map as zero-phase-fraction (ZPF) lines.

Every boundary line of a phase diagram is a line on which one phase is present with zero amount (Morral; Hillert's
"mapping").  For each phase p a signed function f_p(x, RH) is evaluated from an ordinary equilibrium solve:

* a solid s: its amount (mol) where it is present, its saturation index (<= 0) where it is absent;
* the k-th liquid L_k (liquids ordered by amount): the amount of the k-th largest liquid where there are at least
  k liquids; where there are k - 1, minus the smallest tangent-plane distance of a non-trivial trial phase of the
  stability test (it goes to zero where the k-th liquid is about to form); -1 otherwise.

f_p changes sign exactly where p appears or disappears, so the zero lines of all f_p are the boundary lines.  Each
is followed by continuation (predictor along the tangent, corrector along the normal by regula falsi on f_p rather
than bisection on the phase state).  A ZPF line is defined across junctions, so it continues through them: lines do
not end inside the diagram, and the law of adjoining phase regions holds by construction (each line is the
boundary of one phase).  The states on both sides are recorded at every point; a line is split where they change,
which gives the junctions.

Seeds: the boundary points of the traces; between neighbouring traces, segments halved until their ends differ
by at most one phase; and neighbouring solves whose states differ with no line between them (region consistency).

Use :func:`aiomfac_py.diagram.trace_boundaries` with ``method="zpf"``.
"""
from __future__ import annotations

import time

import numpy as np

from .diagram import (BoundaryCurve, BoundaryCurves, PhaseMap, PhaseState, _close_tips, _dist_to_polyline,
                      _phase_distance, _solve_limited,
                      _extend_to_edge, _join_junctions, _length, _ok, _regular_traces, _resample, _state_at, _Tracer, phase_state)

__all__ = ["phase_functions", "trace_zpf"]


def phase_functions(res, solid_keys, max_liquids: int) -> dict:
    """{phase: f} of a converged result (see the module docstring); phases are solid keys and "L1", "L2", ..."""
    out = {}
    for s in solid_keys:
        if s in res.solids:
            out[s] = float(res.solids[s])
        else:
            out[s] = min(float(res.si.get(s, -1.0)), -1e-12)
    tots = sorted((float(L.total) for L in res.liquids), reverse=True)
    n = len(tots)
    tn = getattr(res, "tpd_nontrivial", float("nan"))
    for k in range(1, max_liquids + 1):
        if n >= k:
            out[f"L{k}"] = tots[k - 1]
        elif k == n + 1 and np.isfinite(tn) and tn > 0:
            out[f"L{k}"] = -float(tn)
        else:
            out[f"L{k}"] = -1.0
    return out


def _phases_of(st: PhaseState) -> set:
    return set(st.solids) | {f"L{k}" for k in range(1, st.n_liquids + 1)}


def trace_zpf(pe, feed_of_x, pm: PhaseMap, *, tol: float = 5.0e-4, h0: float = 0.02, h_min: float = 0.002,
              h_max: float = 0.04, delta_max: float = 0.05, rh_samples: int = 40, max_liquids: int = 4,
              seed_subdivide: float | None = 0.004, consistency_rounds: int = 2, consistency_k: int = 8,
              consistency_radius: float = 0.01, consistency_max: int = 300, consistency_time: float = 1800.0,
              invariants: bool = True,
              solve_timeout: float | None = None, cache: dict | None = None, checkpoint=None,
              checkpoint_every: int = 200, verbose: bool = False, **solve_kw) -> BoundaryCurves:
    """ZPF tracing of the boundary lines of ``pm`` (see the module docstring).  Arguments as for
    :func:`aiomfac_py.diagram.trace_boundaries`; the cache is shared with it."""
    traces, xs = _regular_traces(pm)
    x_range = (float(pm.x[0]), float(pm.x[-1]))
    rh_range = (min(t.rh_range[0] if t.rh_range else t.rh[0] for t in pm.traces),
                max(t.rh_range[1] if t.rh_range else t.rh[-1] for t in pm.traces))
    tr = _Tracer(pe, feed_of_x, x_range, rh_range, pm.mode, tol, h0, h_min, h_max, delta_max, solve_kw)
    tr.timeout = solve_timeout
    if cache is not None:
        tr.cache = cache
    tr.checkpoint = checkpoint
    tr.checkpoint_every = max(1, int(checkpoint_every))
    for xv, t in zip(pm.x, pm.traces):
        for r, res in zip(t.rh, t.results):
            key = tuple(round(float(v), 10) for v in tr.to_unit(float(xv), float(r)))
            if _ok(res) and tr.cache.get(key, (None,))[0] is None:
                tr.cache[key] = (phase_state(res), res)
    solid_keys = sorted({s for st in pm.states() for s in st.solids})

    warm = {"res": None, "at": None}                        # the last converged solve: warm start of the next
    deadline = {"t": float("inf")}                          # time budget of the consistency seeds

    def f(q, phase):
        """(f_phase, state) at q; (None, None) where the solve failed"""
        q = np.clip(q, 0.0, 1.0)
        init = warm["res"] if warm["at"] is not None and np.linalg.norm(q - warm["at"]) < 0.05 else None
        st, res = tr.state(q, init)
        if st is None:
            return None, None
        if res.liquids:
            warm["res"], warm["at"] = res, q
        # a solid that occurs on no trace of the map (only between two traces) is not in solid_keys: it is taken
        # as well, from its saturation index where it is absent
        keys = solid_keys if (phase in solid_keys or (phase[:1] == "L" and phase[1:].isdigit())) else solid_keys + [phase]
        return phase_functions(res, keys, max_liquids)[phase], st

    def root(phase, a, b, fa, fb, sa, sb):
        """regula falsi (Illinois) for f_phase = 0 on the segment a-b (fa < 0 < fb or the reverse); returns
        (point, state on the a side, state on the b side) or None"""
        side = 0
        for _ in range(60):
            if np.linalg.norm(b - a) <= 4.0 * tol:          # f is smooth: the interpolated zero is accurate
                w = float(np.clip(fa / (fa - fb), 0.0, 1.0)) if fa != fb else 0.5
                return a + w * (b - a), sa, sb
            w = fa / (fa - fb) if fa != fb else 0.5
            w = float(np.clip(w, 0.05, 0.95))           # keep away from the ends (the two branches of f differ)
            m = a + w * (b - a)
            fm, sm = f(m, phase)
            if fm is None:
                m = a + 0.5 * (b - a)
                fm, sm = f(m, phase)
                if fm is None:
                    return None
            if (fm > 0) == (fa > 0):
                a, fa, sa = m, fm, sm
                if side == -1:
                    fb *= 0.5
                side = -1
            else:
                b, fb, sb = m, fm, sm
                if side == 1:
                    fa *= 0.5
                side = 1
        return 0.5 * (a + b), sa, sb

    def correct(phase, q, n, delta):
        """the zero of f_phase on the line q + s n near s = 0: (point, state at -n, state at +n) or None"""
        # f is also evaluated at q itself: near a corner of the line (a junction where the ZPF line turns) the
        # bracket can hold two zeros of f; the one nearer to q is taken
        d = delta
        q0 = np.clip(q, 0.0, 1.0)
        f0, s0_ = f(q0, phase)
        while True:
            pm_, pp = q - d * n, q + d * n
            if not (tr.inside(pm_) and tr.inside(pp)):
                pm_, pp = np.clip(pm_, 0.0, 1.0), np.clip(pp, 0.0, 1.0)
            fm, sm = f(pm_, phase)
            fp, sp = f(pp, phase)
            if fm is not None and fp is not None and f0 is not None:
                lo_ok, hi_ok = (fm > 0) != (f0 > 0), (fp > 0) != (f0 > 0)
                if lo_ok and hi_ok:                         # two zeros: the nearer one (linear estimate)
                    w_lo, w_hi = f0 / (f0 - fm), f0 / (f0 - fp)
                    lo_ok, hi_ok = (w_lo <= w_hi), (w_hi < w_lo)
                if lo_ok:
                    return root(phase, pm_, q0, fm, f0, sm, s0_)
                if hi_ok:
                    return root(phase, q0, pp, f0, fp, s0_, sp)
            elif fm is not None and fp is not None and (fm > 0) != (fp > 0):
                return root(phase, pm_, pp, fm, fp, sm, sp)
            d *= 2.0
            if d > delta_max:
                return None

    lines: list = []                                        # (phase, points, [(state -n, state +n)])

    def near_line(phase, p, dist):
        return any(ph == phase and _dist_to_polyline(p, P) < dist for ph, P, _ in lines if len(P) > 1)

    def kink(phase, pts, t):
        """the continuation of a ZPF line at a sharp turn: a sign change of f_phase on circles of radius 2 and 4 h_min
        around the last point, at least 15 degrees away from the way back (the line can fold back at an acute
        corner, e.g. a liquidus meeting an invariant line); (point, (left, right)) or None"""
        e = pts[-1]
        back = np.arctan2(-t[1], -t[0])
        for rho in (2.0 * h_min, 4.0 * h_min):
            ang = back + np.radians(np.linspace(15.0, 345.0, 45))
            C = e + rho * np.c_[np.cos(ang), np.sin(ang)]
            vals = [f(c, phase) if tr.inside(c) else (None, None) for c in C]
            best = None
            for k in range(len(C) - 1):
                (fa, sa), (fb, sb) = vals[k], vals[k + 1]
                if fa is None or fb is None or (fa > 0) == (fb > 0):
                    continue
                r = root(phase, C[k], C[k + 1], fa, fb, sa, sb)
                if r is None:
                    continue
                d = r[0] - e
                turn = abs(np.arctan2(t[0] * d[1] - t[1] * d[0], np.dot(t, d)))
                if best is None or turn < best[0]:
                    # going around the circle counter-clockwise, a is at the smaller angle: seen from e moving to
                    # r, the side of a is the right-hand side
                    best = (turn, r[0], (r[2], r[1]))
            if best is not None:
                return best[1], best[2]
        return None

    def follow(phase, p0, t0, s0):
        pts, sides = [np.asarray(p0, float)], [s0]
        t = np.asarray(t0, float) / np.linalg.norm(t0)
        h, corr = h0, h0 / 4
        while True:
            if time.time() > deadline["t"]:
                return pts, sides, "lost"
            if len(pts) >= 2:
                t = pts[-1] - pts[-2]
                t /= np.linalg.norm(t)
            q = pts[-1] + h * t
            if not tr.inside(q):
                s = min(((0.0 if t[k] < 0 else 1.0) - pts[-1][k]) / t[k] for k in (0, 1) if abs(t[k]) > 1e-14)
                if s < 0.5 * h_min:
                    return pts, sides, "edge"
                q, h = np.clip(pts[-1] + s * t, 0.0, 1.0), s
            n = np.array([-t[1], t[0]])
            delta = float(np.clip(3.0 * corr, 2.0 * tol, max(0.5 * h, 2.0 * tol)))
            r = correct(phase, q, n, delta)
            ok = r is not None
            if ok:
                p_new = r[0]
                seg = p_new - pts[-1]
                cos_max = np.cos(np.radians(30 if len(pts) >= 2 else 80))
                ok = np.linalg.norm(seg) > 0.25 * h and np.dot(seg, t) / np.linalg.norm(seg) > cos_max
            if not ok:
                if h <= h_min * 1.0001:
                    k = kink(phase, pts, t)                # a ZPF line turns sharply where another phase
                    if k is None:                          # appears or disappears: find where it goes on
                        return pts, sides, "lost"          # a small circle around the last point
                    pts.append(k[0])
                    sides.append(k[1])
                    h, corr = h_min, h_min / 4
                    continue
                h = max(0.5 * h, h_min)
                continue
            corr = float(abs(np.dot(r[0] - q, n)))
            pts.append(r[0])
            sides.append((r[2], r[1]))                    # (left = +n side, right = -n side)
            if len(pts) > 6 and np.linalg.norm(pts[-1] - pts[0]) < 0.75 * h:
                pts.append(pts[0].copy())
                sides.append(sides[0])
                return pts, sides, "closed"
            if near_line(phase, pts[-1], 1.5 * h_min):
                return pts, sides, "merged"
            if min(pts[-1][0], pts[-1][1], 1 - pts[-1][0], 1 - pts[-1][1]) < 1e-9:
                return pts, sides, "edge"
            h = min(1.6 * h, h_max) if corr < 0.25 * delta else h

    def trace_from(phase, p, t0):
        """the ZPF line of ``phase`` through the point near p (both directions)"""
        if near_line(phase, p, 3.0 * h_min):
            return 0
        n = np.array([-t0[1], t0[0]])
        r = correct(phase, p, n, max(4.0 * tol, 0.002))
        if r is None:
            return 0
        p0, s_neg, s_pos = r
        fwd, sf, ef = follow(phase, p0, t0, (s_pos, s_neg))
        bwd, sb, eb = (([p0], [(s_pos, s_neg)], "closed") if ef == "closed" else
                       follow(phase, p0, -np.asarray(t0, float), (s_neg, s_pos)))
        # the backward points have their sides as seen going backwards: swap to the forward orientation
        P = bwd[::-1] + fwd[1:]
        S = [(b_, a_) for a_, b_ in sb[::-1]] + sf[1:]
        if len(P) < 2:
            return 0
        lines.append((phase, np.array(P), S))
        if verbose:
            a, b = tr.to_xr(P[0]), tr.to_xr(P[-1])
            print(f"ZPF {phase}: {len(P)} points from ({a[0]:.3f}, {a[1]:.3f}) to ({b[0]:.3f}, {b[1]:.3f}), ends "
                  f"{eb}/{ef}, {tr.n_solves} solves", flush=True)
        return 1

    # seeds 1: the boundary points of the traces; every phase that differs across a boundary has a ZPF line there
    for xv, t in zip(xs, traces):
        for b in t.boundaries:
            if not b.resolved and b.rh_hi - b.rh_lo > 1e-2:
                continue
            for phase in sorted(_phases_of(b.below) ^ _phases_of(b.above)):
                trace_from(phase, tr.to_unit(xv, b.rh), np.array([1.0, 0.0]))

    # seeds 2: a phase present on one trace and absent on the neighbouring one at the same RH, with no ZPF line of
    # that phase crossing between them (lines that cross no trace, e.g. vertical ones)
    levels = np.linspace(rh_range[0], rh_range[1], rh_samples + 2)[1:-1]
    def seed_segment(ua, ub, sa, sb):
        for phase in sorted(_phases_of(sa) ^ _phases_of(sb)):
            if any(ph == phase and _crosses(P, ua, ub) for ph, P, _ in lines):
                continue
            fa, s_a = f(ua, phase)
            fb, s_b = f(ub, phase)
            if fa is None or fb is None or (fa > 0) == (fb > 0):
                continue
            rr = root(phase, ua, ub, fa, fb, s_a, s_b)
            if rr is not None:
                trace_from(phase, rr[0], np.array([0.0, 1.0]))

    def subdivide(ua, ub, sa, sb, depth=0):
        """seeds 2b: where the states at the ends of a horizontal segment differ by two or more phases, several
        boundaries lie in between, and a phase present only inside the segment has the same sign of f at both ends;
        the segment is halved until its ends differ by at most one phase (or it is shorter than seed_subdivide), and
        each piece is seeded"""
        if sa == sb or time.time() > deadline["t"]:
            return
        if (_phase_distance(sa, sb) <= 1 or np.linalg.norm(ub - ua) < seed_subdivide or depth >= 12):
            seed_segment(ua, ub, sa, sb)
            return
        m = 0.5 * (ua + ub)
        sm = tr.state(m)[0]
        if sm is None:                                     # failed solve: seed the whole segment as before
            seed_segment(ua, ub, sa, sb)
            return
        subdivide(ua, m, sa, sm, depth + 1)
        subdivide(m, ub, sm, sb, depth + 1)

    for (xa, ta), (xb, tb) in zip(zip(xs[:-1], traces[:-1]), zip(xs[1:], traces[1:])):
        for r in levels:
            sa, sb = _state_at(ta, r), _state_at(tb, r)
            if sa == sb:
                continue
            ua, ub = tr.to_unit(xa, r), tr.to_unit(xb, r)
            if seed_subdivide:
                subdivide(ua, ub, sa, sb)
            else:
                seed_segment(ua, ub, sa, sb)

    # lines that got lost (the corrector failed): look for the sign change of the same phase on circles around the
    # end, away from the line, and continue from there
    for _ in range(2):
        added = 0
        for phase, P, _ in list(lines):
            for e, back in ((P[-1], P[-2] if len(P) > 1 else P[-1]), (P[0], P[1] if len(P) > 1 else P[0])):
                if min(e[0], e[1], 1 - e[0], 1 - e[1]) < 2.0 * h_min:
                    continue
                if any(ph == phase and Q is not P and _dist_to_polyline(e, Q) < 2.0 * h_min for ph, Q, _ in lines):
                    continue
                out = e - back
                out = out / max(np.linalg.norm(out), 1e-300)
                for rho in (3.0 * h_min, 6.0 * h_min, 12.0 * h_min):
                    ang = np.linspace(-0.6 * np.pi, 0.6 * np.pi, 13) + np.arctan2(out[1], out[0])
                    C = e + rho * np.c_[np.cos(ang), np.sin(ang)]
                    vals = [f(c, phase) if tr.inside(c) else (None, None) for c in C]
                    hit = None
                    for k in range(len(C) - 1):
                        (fa, sa), (fb, sb) = vals[k], vals[k + 1]
                        if fa is not None and fb is not None and (fa > 0) != (fb > 0):
                            hit = root(phase, C[k], C[k + 1], fa, fb, sa, sb)
                            break
                    if hit is not None:
                        n0 = len(lines)
                        added += trace_from(phase, hit[0], hit[0] - e)
                        if len(lines) > n0:
                            break
        if not added:
            break

    # seeds 3 (region consistency): all solves in a region enclosed by the lines must have the same phase state.  Two
    # neighbouring converged solves (k nearest, closer than consistency_radius) whose states differ while no ZPF line
    # of a phase that differs between them crosses the segment joining them point to a missing line: both states are
    # checked against each other (re-solved from the other one, lowest Gibbs energy kept: a local minimum is no
    # boundary) and the segment is seeded as in seeds 2
    def consistency_seeds():
        from scipy.spatial import cKDTree
        tried: set = set()
        t_start = time.time()
        deadline["t"] = t_start + consistency_time
        for _ in range(max(0, int(consistency_rounds))):
            pts = [(np.array(k, float), v[0], v[1]) for k, v in tr.cache.items() if v[0] is not None]
            if len(pts) < 2:
                return 0
            X = np.array([q[0] for q in pts])
            # next to failed solves the states are not resolved, and every new solve there tends to fail slowly
            # (time limit): pairs within consistency_radius of a failed solve are left out
            Fp = np.array([k for k, v in tr.cache.items() if v[0] is None], float).reshape(-1, 2)
            Ft = cKDTree(Fp) if len(Fp) else None
            # at the junctions (where the states beside a line change, and where lines end) the states of several
            # regions meet within a few h_min: pairs there are left out too
            Ep = np.array([P[k] for _, P, S in lines if len(P) > 1
                           for k in [0, len(P) - 1] + [i for i in range(1, len(P)) if S[i] != S[i - 1]]],
                          float).reshape(-1, 2)
            Et = cKDTree(Ep) if len(Ep) else None
            by_phase: dict = {}
            for ph, P, _ in lines:
                if len(P) > 1:
                    by_phase.setdefault(ph, []).append(P)
            # a line of the phase that crosses the segment, or passes within eps of it (the line ends are joined at the
            # junctions only afterwards), accounts for the change
            eps = 2.5 * h_min
            dist, idx = cKDTree(X).query(X, k=min(int(consistency_k) + 1, len(X)))
            cand = {}
            for i in range(len(X)):
                for d, j in zip(dist[i, 1:], idx[i, 1:]):
                    if d > consistency_radius:
                        continue
                    si, sj = pts[i][1], pts[j][1]
                    if si == sj:
                        continue
                    a, b = X[i], X[j]
                    if Ft is not None and Ft.query(0.5 * (a + b))[0] < consistency_radius:
                        continue
                    if Et is not None and Et.query(0.5 * (a + b))[0] < 5.0 * h_min:
                        continue
                    if all(any(_near_seg(P, a, b, eps) for P in by_phase.get(ph, [])) for ph in _phases_of(si) ^ _phases_of(sj)):
                        continue
                    cand[(min(i, j), max(i, j))] = float(d)
            n0, n_pairs = len(lines), 0
            for (i, j), d in sorted(cand.items(), key=lambda kv: kv[1]):
                if n_pairs >= consistency_max or time.time() - t_start > consistency_time:
                    break
                a, b = X[i], X[j]
                cell = (tuple(np.round(a / (2.0 * h_min)).astype(int)), tuple(np.round(b / (2.0 * h_min)).astype(int)))
                if cell in tried:
                    continue
                tried.add(cell)
                n_pairs += 1
                sa = tr.state_min(a, [pts[j][2]])[0]
                sb = tr.state_min(b, [pts[i][2]])[0]
                if sa is None or sb is None or sa == sb:
                    continue
                nl = len(lines)
                subdivide(a, b, sa, sb)
                if verbose and len(lines) > nl:
                    xa, ra = tr.to_xr(a)
                    xb, rb = tr.to_xr(b)
                    print(f"  consistency: line between ({xa:.4f}, {ra:.4f}) {sa.label} and ({xb:.4f}, {rb:.4f}) "
                          f"{sb.label}", flush=True)
            if verbose:
                print(f"ZPF consistency: {len(cand)} inconsistent neighbours, {n_pairs} checked, "
                      f"{len(lines) - n0} lines added, {tr.n_solves} solves", flush=True)
            if len(lines) == n0:
                break
        return 0

    if consistency_rounds:
        consistency_seeds()
        deadline["t"] = float("inf")

    def sides_at(m, nrm):
        a = tr.state(np.clip(m + 3.0 * tol * nrm, 0.0, 1.0))[0]
        b = tr.state(np.clip(m - 3.0 * tol * nrm, 0.0, 1.0))[0]
        return (a, b)

    def refine_junctions(P, S):
        """where the side states change between two points, locate the change along the line by bisection and
        insert the junction point"""
        P2, S2 = [P[0]], [S[0]]
        for i in range(1, len(P)):
            if S[i] != S[i - 1] and np.linalg.norm(P[i] - P[i - 1]) > 2.0 * tol:
                a, b = P[i - 1], P[i]
                t = (b - a) / np.linalg.norm(b - a)
                nrm = np.array([-t[1], t[0]])
                lo, hi = 0.0, 1.0
                while np.linalg.norm(b - a) * (hi - lo) > tol:
                    m = 0.5 * (lo + hi)
                    sm = sides_at(a + m * (b - a), nrm)
                    if sm == S[i - 1]:
                        lo = m
                    elif sm == S[i]:
                        hi = m
                    else:
                        break
                J = a + 0.5 * (lo + hi) * (b - a)
                P2.append(J)
                S2.append(S[i])
            P2.append(P[i])
            S2.append(S[i])
        return np.array(P2), S2

    # invariant lines: a horizontal piece of a line between states that differ by two phases (one phase replaced by
    # another, e.g. 4L | 3L+C) is an invariant: all phases of both states coexist at one RH.  The minimum of F is nearly
    # flat there (the amounts jump across the line while F changes little), so solves close to it tend to fail.  Its
    # RH is found directly: the state below, imposed (its solids only, its number of liquids), is solved for the RH at
    # which the phase that appears above becomes stable (saturation index 0 for a solid, non-trivial TPD 0 for a
    # liquid), at the ends and the middle of the piece; the piece is placed there, and failed solves within 2.5 h_min of
    # it are not reported (their state is that of their side)
    inv_found = []

    def appearing_value(lo, phase, x, rh, init=None):
        """(stability of ``phase`` next to the imposed state ``lo`` at (x, rh): SI or -TPD, > 0 = stable; result),
        or (None, None) if the imposed solve fails.  ``init``: warm start (a result at the same composition)."""
        feed = feed_of_x(float(x))
        kw = dict(tr.solve_kw, solids=list(lo.solids) if lo.solids else "none", max_liquids=max(1, lo.n_liquids))
        res = _solve_limited(pe, feed, float(rh), pm.mode, init, kw, solve_timeout)
        tr.n_solves += 1
        if not _ok(res) or len(res.liquids) != lo.n_liquids:
            return None, None
        if phase.startswith("L") and phase[1:].isdigit():
            v = -float(getattr(res, "tpd_nontrivial", float("nan")))
        else:
            v = float(res.si.get(phase, np.nan))
        return (v, res) if np.isfinite(v) else (None, None)

    def invariant_rh(lo, phase, x, rh0, init=None):
        """RH at which ``phase`` becomes stable next to the imposed state ``lo`` at composition x (regula falsi).
        Every imposed solve is warm-started from the last converged one (first: ``init``, the unconstrained solution
        in the region of ``lo`` at the same composition): near an invariant the free energy is flat and the imposed
        solves started from scratch often fail.  The bracket starts close to rh0 and is widened up to +-0.01."""
        d = 3.0 * tol * (rh_range[1] - rh_range[0])
        last = [init]

        def f(r):
            v, res = appearing_value(lo, phase, x, r, last[0])
            if res is not None:
                last[0] = res
            return v
        a, b = rh0 - d, rh0 + d
        fa = f(a)
        if fa is None:
            return None
        fb = f(b)
        while fb is not None and (fa > 0) == (fb > 0) and max(rh0 - a, b - rh0) < 0.01:
            if fa > 0:                                     # stable on both ends: move the lower end down
                a = rh0 - 2.0 * (rh0 - a)
                fa = f(a)
                if fa is None:
                    return None
            else:
                b = rh0 + 2.0 * (b - rh0)
                fb = f(b)
        if fb is None or (fa > 0) == (fb > 0):
            return None
        side = 0
        for _ in range(40):
            m = b - fb * (b - a) / (fb - fa)
            fm = f(m)
            if fm is None:
                return None
            if abs(fm) < 1e-12 or abs(b - a) < 1e-9:
                return m
            if (fm > 0) == (fb > 0):
                b, fb = m, fm
                if side == -1:
                    fa *= 0.5
                side = -1
            else:
                a, fa = m, fm
                if side == 1:
                    fb *= 0.5
                side = 1
        return 0.5 * (a + b)

    def snap_invariants():
        flat = 4.0 * tol
        for li, (phase, P, S) in enumerate(lines):
            i = 0
            while i < len(P) - 1:
                j = i
                while (j + 1 < len(P) and S[j + 1] == S[i] and None not in S[i]
                       and _phase_distance(S[i][0], S[i][1]) == 2 and abs(P[j + 1][1] - P[i][1]) < flat):
                    j += 1
                if j - i >= 2 and abs(P[j][0] - P[i][0]) > 4.0 * h_min:
                    sa, sb = S[i]
                    # which side is below: the state just below the middle of the piece
                    mid = P[(i + j) // 2]
                    below = tr.state(np.clip(mid - np.array([0.0, 3.0 * tol]), 0.0, 1.0))[0]
                    lo, up = (sa, sb) if below == sa else ((sb, sa) if below == sb else (None, None))
                    new = sorted(_phases_of(up) - _phases_of(lo)) if lo is not None else []
                    if verbose:
                        print(f"  invariant candidate {sa.label} | {sb.label}: below {below.label if below else None}, "
                              f"appearing {new}", flush=True)
                    gone = sorted(_phases_of(lo) - _phases_of(up)) if lo is not None else []
                    if len(new) == 1 and len(gone) == 1:
                        xs_ = [P[i][0], mid[0], P[j][0]]
                        rs = []
                        for xu in xs_:
                            x_, r_ = tr.to_xr(np.array([xu, mid[1]]))
                            # the phase that appears, next to the state below; or (e.g. a liquid appearing next to an
                            # all-solid state, which has no stability test) the phase that goes, next to the state above
                            r_lo = tr.state(np.clip(np.array([xu, mid[1] - 3.0 * tol]), 0.0, 1.0))
                            r_up = tr.state(np.clip(np.array([xu, mid[1] + 3.0 * tol]), 0.0, 1.0))
                            rr = invariant_rh(lo, new[0], x_, r_, r_lo[1] if r_lo[0] == lo else None)
                            if rr is None:
                                rr = invariant_rh(up, gone[0], x_, r_, r_up[1] if r_up[0] == up else None)
                            if rr is not None:
                                rs.append((xu, (rr - rh_range[0]) / (rh_range[1] - rh_range[0])))
                        if verbose:
                            print(f"  invariant RH found at {len(rs)} of 3 compositions", flush=True)
                        pieces.append((li, i, j, lo, up, new[0]))
                        pooled.setdefault(frozenset((lo, up)), []).extend(rs)
                i = j + 1

    pieces, pooled, inv_fit = [], {}, {}

    def invariant_fit(rs):
        """RH* (unit) as a function of x: constant if the RH* found agree within 2 tol (an invariant is horizontal;
        the spread is the accuracy of the solves on the flat free energy), else the least-squares line"""
        cx, cr = np.array(rs).T
        if np.ptp(cr) < 2.0 * tol:
            return np.array([0.0, float(np.mean(cr))])
        return np.polyfit(cx, cr, 1)

    if invariants:
        snap_invariants()
        # the RH* found on every piece with the same two states are pooled (the coinciding ZPF lines of the
        # appearing and the disappearing phase, and pieces where the search failed)
        for li, i, j, lo, up, ph in pieces:
            rs = pooled[frozenset((lo, up))]
            if len(rs) < 2:
                continue
            k = inv_fit.setdefault(frozenset((lo, up)), invariant_fit(rs))
            P = lines[li][1]
            for q in range(i, j + 1):
                P[q][1] = np.polyval(k, P[q][0])
            inv_found.append((P[i:j + 1].copy(), lo, up))
            if verbose:
                a_, b_ = tr.to_xr(P[i]), tr.to_xr(P[j])
                print(f"ZPF invariant: {lo.label} | {up.label} ({ph} appears) X {a_[0]:.3f}-{b_[0]:.3f} "
                      f"at RH {a_[1]:.6f}-{b_[1]:.6f} ({len(rs)} RH* found)", flush=True)

    # split each line where the states on its two sides change: the pieces are the boundary curves
    curves = []
    flat_tol = 4.0 * tol
    for phase, P, S in lines:
        P, S = refine_junctions(P, S)
        cuts = [0] + [i for i in range(1, len(P)) if S[i] != S[i - 1]] + [len(P) - 1]
        for a, b in zip(cuts[:-1], cuts[1:]):
            Q = P[a:b + 1]                               # ends at the junction where the next piece starts
            left, right = S[a]
            if len(Q) >= 2 and left is not None and right is not None and left != right:
                xr = np.array([tr.to_xr(q) for q in Q])
                k = inv_fit.get(frozenset((left, right)))
                if k is not None:                        # junction points added after the snapping lie on it too
                    Q = np.array(Q, dtype=float)
                    on = np.abs(Q[:, 1] - np.polyval(k, Q[:, 0])) < flat_tol
                    Q[on, 1] = np.polyval(k, Q[on, 0])
                    xr = np.array([tr.to_xr(q) for q in Q])
                curves.append(BoundaryCurve(left, right, xr[:, 0], xr[:, 1], ("junction", "junction")))

    # a degenerate line (an invariant RH, a single composition) is the ZPF line of several phases: keep one piece
    # per pair of states where pieces coincide
    sx, sr = x_range[1] - x_range[0], rh_range[1] - rh_range[0]
    U = [np.c_[(c.x - x_range[0]) / sx, (c.rh - rh_range[0]) / sr] for c in curves]
    order = sorted(range(len(curves)), key=lambda i: -_length(U[i]))
    keep: list = []
    for i in order:
        key = frozenset((curves[i].left, curves[i].right))
        if not any(key == frozenset((curves[j].left, curves[j].right)) and
                   np.mean([_dist_to_polyline(q, U[j]) < 2.0 * h_min for q in _resample(U[i], h_min)]) >= 0.8
                   for j in keep):
            keep.append(i)
    curves = [curves[i] for i in sorted(keep)]
    # a ZPF line can run along the edge of the diagram (a phase that exists only inside, e.g. letovicite at X = 1):
    # it is no boundary inside the diagram
    e_ = 2.0 * h_min
    on_edge = lambda P: (np.all(P[:, 0] < e_) or np.all(P[:, 0] > 1 - e_) or np.all(P[:, 1] < e_) or
                         np.all(P[:, 1] > 1 - e_))
    curves = [c for c, P in zip(curves, [np.c_[(c.x - x_range[0]) / sx, (c.rh - rh_range[0]) / sr] for c in curves])
              if not on_edge(P)]
    # free ends: the two sides of the tip of a closed region are joined, others are extended along their tangent to
    # the line or edge they meet (a line ends at the edge or at a junction)
    U = [np.c_[(c.x - x_range[0]) / sx, (c.rh - rh_range[0]) / sr] for c in curves]
    U = _join_junctions(U, [c.ends for c in curves], 2.5 * h_min)
    U = _close_tips(U, [frozenset((c.left, c.right)) for c in curves], 15.0 * h_min, 2.5 * h_min)
    U = _extend_to_edge(U, 12.0 * h_min, 2.5 * h_min)
    curves = [BoundaryCurve(c.left, c.right, x_range[0] + P[:, 0] * sx, rh_range[0] + P[:, 1] * sr, c.ends)
              for c, P in zip(curves, U)]
    fk = [np.array(k) for k, v in tr.cache.items() if v[0] is None]
    fk = [k for k in fk if not any(_dist_to_polyline(k, Q) < 2.5 * h_min for Q, _, _ in inv_found)]
    failed = np.array([tr.to_xr(k) for k in fk]).reshape(-1, 2)
    return BoundaryCurves(curves, x_range, rh_range, tr.n_solves, 0, failed).cleaned(h_min)


def _near_seg(P, a, b, eps) -> bool:
    """the polyline P crosses the segment a-b or comes within eps of it"""
    if _crosses(P, a, b) or _dist_to_polyline(a, P) < eps or _dist_to_polyline(b, P) < eps:
        return True
    ab = b - a
    w = np.clip(((P - a) @ ab) / max(float(ab @ ab), 1e-300), 0.0, 1.0)
    return bool(np.min(np.linalg.norm(a + w[:, None] * ab - P, axis=1)) < eps)


def _crosses(P, a, b) -> bool:
    from .diagram import _crosses_seg
    return _crosses_seg(P, a, b)

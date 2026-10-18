"""Solver benchmark: fixed points of the UHAERO 2007 organic-inorganic systems, solved from scratch.

  python solver_bench.py run <out.json> [--timeout 20] [--kinds conv,failed,hard] [--limit N] [--warm DIR [--warm-min D]]
  python solver_bench.py compare <base.json> <new.json>

The points (tools/bench/organic_points.json) are converged (conv) and failed (failed) solves sampled from the line
caches of the UHAERO 2007 Figs. 8 and 9 (four organic pairs each) and points next to an invariant line and the tip of
a thin 4L+C region (hard).  Each point is solved with init=None under the time limit (``--warm DIR``: started
from the nearest other converged solve of the line cache DIR/lines_cache_Y<Y>_P<P>.pkl, as in the boundary tracing; ``--warm-min D``: the nearest one at least D away
(unit square) and at another composition); the result is the status, the
phase state, the Gibbs energy and the wall time.  `compare` reports, against a base run: points whose state changed,
the largest |dF| among points converged in both, the converged counts and the total and median times.
"""
import json
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def _nearest(cache_dir, p, dmin=0.0, caches={}):
    """nearest other converged result of the line cache (unit coordinates of the cache from its lines file)"""
    import pickle
    import numpy as np
    key = (p["Y"], p["P"])
    if key not in caches:
        c = pickle.load(open(os.path.join(cache_dir, f"lines_cache_Y{p['Y']:.2f}_P{p['P']}.pkl"), "rb"))
        bc = pickle.load(open(os.path.join(cache_dir, f"u07_lines_Y{p['Y']:.2f}_P{p['P']}.pkl"), "rb"))
        good = [(k, v[1]) for k, v in c.items() if v[0] is not None and v[1].liquids]
        caches[key] = (np.array([k for k, _ in good]), [r for _, r in good], bc.x_range, bc.rh_range)
    K, R, (x0, x1), (r0, r1) = caches[key]
    q = np.array([(p["X"] - x0) / (x1 - x0), (p["RH"] - r0) / (r1 - r0)])
    d = np.hypot(*(K - q).T)
    d[(d < max(dmin, 1e-9)) | (np.abs(K[:, 0] - q[0]) < (1e-6 if dmin > 0 else -1.0))] = np.inf
    j = int(np.argmin(d))
    return R[j], float(d[j])


def run(out, timeout=20.0, kinds=("conv", "failed", "hard"), limit=None, warm=None, warm_min=0.0):
    import uhaero2007_figures as u
    from aiomfac_py.diagram import _ok, _solve_limited, phase_state
    pts = [p for p in json.load(open(os.path.join(HERE, "bench", "organic_points.json"))) if p["kind"] in kinds]
    pts = pts[:limit] if limit else pts
    systems, rows = {}, []
    for i, p in enumerate(pts):
        key = (p["Y"], p["P"])
        if key not in systems:
            ions = ["NH4+", "H+", "SO4--"] + (["NO3-"] if p["Y"] < 1 else [])
            systems[key] = u.org_system(p["P"], ions)
        pe, o1, o2, f1 = systems[key]
        feed = u.with_organics(u.inorg_feed(p["X"], p["Y"]), o1, o2, f1, 0.2)
        init, dist = _nearest(warm, p, warm_min) if warm else (None, None)
        t = time.perf_counter()
        res = _solve_limited(pe, feed, p["RH"], "equilibrium", init, {}, timeout)
        dt = time.perf_counter() - t
        ok = _ok(res)
        rows.append(dict(p, ok=bool(ok), state=phase_state(res).label if ok else None,
                         gibbs=float(res.gibbs) if ok else None, time=dt, message=None if ok else res.message,
                         warm_dist=dist, warm_msg=(res.message or "") if init is not None else None))
        print(f"{i + 1:3d}/{len(pts)} {p['kind']:6s} Y {p['Y']} P{p['P']} X {p['X']:.4f} RH {p['RH']:.4f}: "
              f"{rows[-1]['state'] or 'FAIL ' + str(res.message)[:30]} {dt:.1f} s", flush=True)
    json.dump(rows, open(out, "w"), indent=0)


def compare(fa, fb):
    A, B = json.load(open(fa)), json.load(open(fb))
    assert len(A) == len(B)
    changed, dF = [], []
    for a, b in zip(A, B):
        if a["ok"] and b["ok"]:
            dF.append(abs(a["gibbs"] - b["gibbs"]))
            if a["state"] != b["state"]:
                changed.append((a, b))
        elif a["ok"] != b["ok"]:
            changed.append((a, b))
    for a, b in changed:
        print(f"  changed: {a['kind']} Y {a['Y']} P{a['P']} X {a['X']:.4f} RH {a['RH']:.4f}: "
              f"{a['state']} {a['gibbs']} -> {b['state']} {b['gibbs']}")
    for name, R in (("base", A), ("new ", B)):
        by = {}
        for r in R:
            by.setdefault(r["kind"], []).append(r)
        conv = " ".join(f"{k} {sum(r['ok'] for r in v)}/{len(v)}" for k, v in by.items())
        tc = [r["time"] for r in R if r["kind"] == "conv"]
        print(f"{name}: converged {conv}; time total {sum(r['time'] for r in R):.0f} s, "
              f"conv points total {sum(tc):.1f} s, median {statistics.median(tc):.2f} s")
    print(f"max |dF| (both converged): {max(dF) if dF else float('nan'):.2e}; changed: {len(changed)}")


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "run":
        kw = {}
        if "--timeout" in a:
            kw["timeout"] = float(a[a.index("--timeout") + 1])
        if "--kinds" in a:
            kw["kinds"] = tuple(a[a.index("--kinds") + 1].split(","))
        if "--warm" in a:
            kw["warm"] = a[a.index("--warm") + 1]
        if "--warm-min" in a:
            kw["warm_min"] = float(a[a.index("--warm-min") + 1])
        if "--limit" in a:
            kw["limit"] = int(a[a.index("--limit") + 1])
        run(a[1], **kw)
    else:
        compare(a[1], a[2])

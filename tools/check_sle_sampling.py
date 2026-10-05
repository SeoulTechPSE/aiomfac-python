"""Global-optimality spot check of aiomfac_py.sle (no optimizer involved): the Gibbs function of the active-set solution
must not exceed that of any of many random feasible solid amounts u >= 0 (including the dry corner points).
Usage:  python tools/check_sle_sampling.py [ions...]   (default: NH4+ H+ SO4--; random electroneutral acid feeds)"""
from __future__ import annotations
import math, sys, time
import numpy as np
from aiomfac_py.sle import SLESolver


def phi_samples(sol: SLESolver, feed, T, rh, nsamp, rng):
    ln_rh = math.log(rh)
    bfull = sol._vec(feed)
    present = bfull > 0
    ion_idx = np.flatnonzero(present)
    sol_idx = [j for j in range(len(sol.solids)) if np.all(sol.V[~present, j] == 0)]
    scale = bfull.max()
    b = bfull[ion_idx] / scale
    V = sol.V[np.ix_(ion_idx, sol_idx)]
    c = np.array([sol.solids[j].ln_k_eff(T, ln_rh, sol.mode) for j in sol_idx])
    aq = sol._aq_for(ion_idx)
    J = V.shape[1]
    best = np.inf
    for _ in range(nsamp):
        u = rng.random(J) * rng.choice([0.0, 0.05, 0.3, 1.0]) * (b.min() / max(V.max(), 1))
        u *= rng.random(J) > 0.3
        n = b - V @ u
        if np.any(n <= 1e-10):
            continue
        r = aq.solve_water(n, T, ln_rh)
        if r is None:
            continue
        best = min(best, float(np.sum(n * r[0]) + c @ u))
    return best * scale


if __name__ == "__main__":
    ions = sys.argv[1:] or ["NH4+", "H+", "SO4--"]
    sol = SLESolver(ions)
    rng = np.random.default_rng(11)
    bad = 0
    for k in range(14):
        fh = rng.uniform(0.02, 1.98)
        feed = {"SO4--": 1.0, "H+": fh, "NH4+": 2.0 - fh}
        rh = float(rng.uniform(0.25, 0.92))
        t = time.time(); r = sol.solve(feed, 298.15, rh); tt = time.time() - t
        pb = phi_samples(sol, feed, 298.15, rh, 600, rng)
        flag = "" if r.gibbs <= pb + 1e-7 * max(1, abs(pb)) else "  <-- SAMPLE BETTER"
        bad += bool(flag)
        print(f"{k:2d} H/SO4={fh:.2f} rh={rh:.3f} {r.status:14s} G={r.gibbs:.6f} sampled_min={pb:.6f} {tt:.1f}s "
              f"{ {a: round(b, 3) for a, b in r.solids.items()} }{flag}", flush=True)
    print("mismatches:", bad)

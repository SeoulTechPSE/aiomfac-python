"""Independent check of aiomfac_py.sle: minimize the same Gibbs function by a generic constrained optimizer (SLSQP,
multi-start) and compare with the active-set solution.  The solver logic is independent; the thermodynamics (AIOMFAC +
solids.py) are shared by construction."""
from __future__ import annotations
import math, sys
import numpy as np
from scipy.optimize import minimize
from aiomfac_py.sle import SLESolver, feed_from_salts


def brute(sol: SLESolver, feed, T, rh, nstart=6, seed=0):
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

    def phi(u):
        n = b - V @ u
        if np.any(n <= 1e-12):
            return 1e6
        r = aq.solve_water(n, T, ln_rh)
        if r is None:
            return 1e6
        return float(np.sum(n * r[0]) + c @ u)

    cons = [{"type": "ineq", "fun": lambda u: (b - V @ u) - 1e-9}]
    rng = np.random.default_rng(seed)
    best = None
    # dry-feasible direction scale
    for k in range(nstart):
        u0 = np.zeros(J) if k == 0 else rng.random(J) * 0.2 * (b.min() / max(V.max(), 1))
        r = minimize(phi, u0, method="SLSQP", bounds=[(0, None)] * J, constraints=cons,
                     options={"ftol": 1e-12, "maxiter": 400})
        if best is None or r.fun < best.fun:
            best = r
    keys = [sol.solids[j].key for j in sol_idx]
    return best.fun * scale, {k: v * scale for k, v in zip(keys, best.x) if v > 1e-8}


if __name__ == "__main__":
    ions = ["Na+", "K+", "Cl-", "NH4+", "SO4--"]
    sol = SLESolver(ions)
    cases = [({"NaCl": 1.0, "(NH4)2SO4": 1.0}, 0.66), ({"NaCl": 1.0, "(NH4)2SO4": 1.0}, 0.64),
             ({"NaCl": 1.0, "KCl": 1.0}, 0.80), ({"NaCl": 1.0, "KCl": 1.0}, 0.76), ({"NaCl": 1.0, "KCl": 1.0}, 0.72)]
    for feed, rh in cases:
        f = feed_from_salts(feed)
        r = sol.solve(f, 298.15, rh)
        phi_b, ub = brute(sol, f, 298.15, rh)
        print(f"feed={feed} RH={rh}: active-set {r.status} gibbs={r.gibbs:.6f} solids={ {k: round(v,4) for k,v in r.solids.items()} } water={r.water_kg*1e3:.2f} g"
              f"\n      brute-force      gibbs={phi_b:.6f} solids={ {k: round(v,4) for k,v in ub.items()} }")

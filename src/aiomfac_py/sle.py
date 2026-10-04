"""Solid-liquid equilibrium (SLE) at fixed temperature and relative humidity: primal-dual active-set / Newton method.

Implements, on top of this package's AIOMFAC activity model, the equilibrium formulation and the active-set strategy of

    Amundson, N. R., Caboussat, A., He, J. W., Seinfeld, J. H., Yoo, K. Y. (2006), "Primal-Dual Active-Set Algorithm for
    Chemical Equilibrium Problems Related to the Modeling of Atmospheric Inorganic Aerosols", J. Optim. Theory Appl.
    128(3), 469-498, doi:10.1007/s10957-006-9030-y.                                                  [cited as "JOTA-1"]

(the inorganic companion of the organic phase-split solver in :mod:`aiomfac_py.lle`).

Problem
-------
Minimize the Gibbs free energy of an aqueous electrolyte phase plus a set of pure solid salts/hydrates at fixed T and a
fixed water activity a_w = RH (JOTA-1 eq. (22)):

    min  G = n_c^T ln a_c + n_s^T ln k_s  - n_w ln RH          s.t.  n_c + A_s n_s = b,   n_c > 0, n_w > 0, n_s >= 0

with the aqueous *ion* amounts ``n_c`` as the components (electroneutrality is the charge "element" of the paper), the
solid stoichiometries ``A_s = V`` (columns: ions of the dissolution reaction of each salt), the solubility products
``k_s`` of :mod:`aiomfac_py.solids` and the ion activities from ``ActivityModel`` evaluated at the molality for which
a_w = RH.  The KKT conditions are exactly JOTA-1 (14)-(19)/(23)-(29):

    mass action for the saturated salts   sum_i nu_ij ln a_i + h_j ln RH = ln K_j          (j in active set, n_sj >= 0)
    dual feasibility for all others       sum_i nu_ij ln a_i + h_j ln RH <= ln K_j       (saturation index SI_j <= 0)
    complementarity                       n_sj * SI_j = 0

(the hydrate water enters the saturation condition through the fixed ln RH -- the "a_H2O in the range of A_s" case that
JOTA-1 Sec. 3.2 excludes is handled by keeping the water row out of the mass balance, since n_w is free at fixed RH).

Algorithm (JOTA-1 Table 1, Sec. 4)
----------------------------------
For an active set ``S`` of precipitated salts the KKT system is projected onto the unknown solid amounts ``u = n_s,S``
(the mass balance is used to eliminate the aqueous ion amounts, the water amount follows from a_w = RH, i.e. the
null-space reduction of Sec. 4.1 with the multipliers ``n_s`` as the remaining unknowns).  One Newton step uses the
reduced Hessian  V_S^T D V_S  (D = d ln a / d n at fixed a_w; obtained by central finite differences along the neutral
directions V_S, since AIOMFAC does not provide analytic Jacobians).  The step length is limited by (i) fraction-to-the-
boundary for the aqueous ion amounts (n_c > 0 -- JOTA-1 keeps them positive through the logarithmic reformulation) and
(ii) the blocking solid amount n_sj >= 0 (the "remove salt when its concentration becomes negative" rule, Table 1
Step 4(b)(ii)); an Armijo backtracking on the Gibbs function guards the step (the problem is convex when the aqueous
phase is stable, JOTA-1 Theorem 4.2).  After convergence the saturation index of every salt outside the active set is
checked and the most supersaturated one is added (JOTA-1 eq. (55)-(56), one constraint at a time); the loop ends when no
constraint is violated, which is the KKT test.

Differences from the paper (read before trusting the numbers)
----------------------------------------------------------------
* No gas phase / noncomponent species (NH3, HNO3, HCl partitioning; H+/HSO4- speciation) in this version: the feed is
  a closed set of ions + water at fixed RH.  Acidic systems (H+, HSO4-) are not supported yet.
* The paper keeps the iterates dual feasible (starting from an arbitrary feasible lambda^0); here the iteration is
  primal feasible (n_s >= 0, n_c > 0) and the dual feasibility (SI <= 0) is the stopping test.  For starting points
  where the fully dissolved state cannot reach the target RH within the AIOMFAC concentration range, an RH-continuation
  from high RH supplies the dual-feasible path (``_homotopy``).
* The "all salts precipitated, no aqueous phase" state, which the paper's interior formulation treats only in the
  n_l -> 0 limit, is decided explicitly before the Newton iteration: a linear program over the solids gives the dry
  assemblage and its dual (the ion potentials ``y``); the aqueous phase is absent iff the tangent-plane distance
  min_q sum_i q_i (ln a_i(q) - y_i) is non-negative (JOTA-1 Sec. 4.1, phase stability criterion).
* Jacobians by finite differences; positive-definiteness of the reduced Hessian (JOTA-1 eq. (59)-(60)) is not enforced
  by modification; a small Levenberg shift is applied if the reduced Hessian is singular.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy.optimize import brentq, linprog, minimize

from .io import Component
from .model import ActivityModel
from .params import load_subgroup_params
from .solids import ION_REGISTRY, SOLIDS, Solid

MW_WATER = 0.01801528        # kg/mol
_NEG = -1.0e300

# formula unit -> ions (for building a feed from salts)
SALT_IONS: dict[str, dict[str, int]] = {
    "NaCl": {"Na+": 1, "Cl-": 1}, "KCl": {"K+": 1, "Cl-": 1}, "NH4Cl": {"NH4+": 1, "Cl-": 1},
    "NaNO3": {"Na+": 1, "NO3-": 1}, "KNO3": {"K+": 1, "NO3-": 1}, "NH4NO3": {"NH4+": 1, "NO3-": 1},
    "Na2SO4": {"Na+": 2, "SO4--": 1}, "K2SO4": {"K+": 2, "SO4--": 1}, "(NH4)2SO4": {"NH4+": 2, "SO4--": 1},
    "MgCl2": {"Mg++": 1, "Cl-": 2}, "Mg(NO3)2": {"Mg++": 1, "NO3-": 2}, "MgSO4": {"Mg++": 1, "SO4--": 1},
    "CaCl2": {"Ca++": 1, "Cl-": 2}, "Ca(NO3)2": {"Ca++": 1, "NO3-": 2}, "CaSO4": {"Ca++": 1, "SO4--": 1},
}


def feed_from_salts(salts: dict[str, float]) -> dict[str, float]:
    """Ion amounts [mol] of a feed given as formula-unit amounts, e.g. ``{"NaCl": 1.0, "(NH4)2SO4": 0.5}``."""
    b: dict[str, float] = {}
    for salt, n in salts.items():
        for ion, nu in SALT_IONS[salt].items():
            b[ion] = b.get(ion, 0.0) + nu * n
    return b


# ============================================================================================================
# Aqueous phase: ion activities at fixed water activity
# ============================================================================================================

class AqueousIons:
    """Aqueous electrolyte solution of a fixed ion set, evaluated with ``ActivityModel`` (ions + water only)."""

    def __init__(self, ions: Sequence[str], M_cap: float = 400.0):
        bad = [i for i in ions if i in ("H+", "HSO4-")]
        if bad:
            raise NotImplementedError(f"acidic species {bad} (bisulfate speciation) are not supported by the SLE solver yet")
        self.ions = list(ions)
        self.N = len(self.ions)
        self.M_cap = M_cap
        cats = [i for i in self.ions if ION_REGISTRY[i][1] > 0]
        ans = [i for i in self.ions if ION_REGISTRY[i][1] < 0]
        if not cats or not ans:
            raise ValueError("need at least one cation and one anion")
        pairs = [(c, ans[0]) for c in cats] + [(cats[0], a) for a in ans[1:]]
        comps = [Component(1, "Water", ((16, 1),))]
        for k, (c, a) in enumerate(pairs, start=2):
            zc, za = ION_REGISTRY[c][1], -ION_REGISTRY[a][1]
            g = math.gcd(zc, za)
            comps.append(Component(k, c + a, ((ION_REGISTRY[c][0], za // g), (ION_REGISTRY[a][0], zc // g))))
        self.model = ActivityModel(comps)
        mx = self.model.mixture
        self._mx = mx
        self._ngi = mx.ngi
        self._nc, self._na = mx.sr.n_cation, mx.sr.n_anion
        self._pos = []                    # (is_cation, index in smc/sma) per ion
        for ion in self.ions:
            sid, z = ION_REGISTRY[ion]
            self._pos.append((z > 0, mx.cat_index[sid] if z > 0 else mx.an_index[sid]))
        self.charge = np.array([ION_REGISTRY[i][1] for i in self.ions], dtype=float)
        self._xn = np.array([1.0])
        self.n_eval = 0

    # -------------------------------------------------------------------------------------------------
    def ln_gamma_aw(self, m: np.ndarray, T: float):
        """ln gamma_i (molal) of all ions and ln a_w, for ion molalities ``m`` [mol/kg water]."""
        self.n_eval += 1
        smc = np.zeros(self._ngi); sma = np.zeros(self._ngi)
        for (is_cat, idx), v in zip(self._pos, m):
            (smc if is_cat else sma)[idx] = v
        mod = self.model
        x = mod._x_from_molalities(self._xn, smc, sma)
        lr, mr, sr = mod.lr_mr_sr(T, smc, sma, self._xn, x)
        nn, nc = 1, self._nc
        ln_w = mr.ln_gamma_neutral[0] + sr.ln_gamma_sr[0] + lr.ln_gamma_neutral[0] + math.log(x[0])
        out = np.empty(self.N)
        for k, (is_cat, idx) in enumerate(self._pos):
            if is_cat:
                out[k] = mr.ln_gamma_cation[idx] + sr.ln_gamma_sr[nn + idx] + lr.ln_gamma_cation[idx] - mr.tmolal
            else:
                out[k] = mr.ln_gamma_anion[idx] + sr.ln_gamma_sr[nn + nc + idx] + lr.ln_gamma_anion[idx] - mr.tmolal
        return out, ln_w

    def solve_water(self, n: np.ndarray, T: float, ln_rh: float, lnM_guess: float | None = None,
                    clip: bool = False):
        """Water mass w [kg] such that a_w(n/w) = RH.  Returns (ln a_i, w, lnM) or None if RH is out of reach.

        With ``clip=True`` an unreachable RH returns the state at the concentration cap ``M_cap`` and the extra element
        ``ln a_w - ln RH`` (> 0) as a fourth entry (used by the stability test to keep its objective continuous)."""
        S = float(np.sum(n))

        def g(lnM):
            M = math.exp(lnM)
            m = n * (M / S)
            return self.ln_gamma_aw(m, T)[1] - ln_rh

        try:
            lo, hi = math.log(1.0e-3), math.log(self.M_cap)
            if lnM_guess is not None:
                a, b = lnM_guess - 0.25, lnM_guess + 0.25
                ga, gb = g(a), g(b)
                if ga > 0 > gb:
                    lo, hi = a, b
                elif ga <= 0:
                    glo = g(lo)
                    if glo < 0:
                        return None
                    hi = a
                else:                      # gb >= 0
                    ghi = g(hi)
                    if ghi > 0:
                        return None
                    lo = b
            else:
                if g(lo) < 0:
                    return None
                if g(hi) > 0:
                    if not clip:
                        return None
                    M = self.M_cap
                    m = n * (M / S)
                    lng, ln_w = self.ln_gamma_aw(m, T)
                    return lng + np.log(m), S / M, hi, ln_w - ln_rh
            lnM = brentq(g, lo, hi, xtol=1.0e-13, rtol=1.0e-13)
        except (FloatingPointError, ValueError, ZeroDivisionError, OverflowError):
            return None
        M = math.exp(lnM)
        m = n * (M / S)
        lng, _ = self.ln_gamma_aw(m, T)
        ln_a = lng + np.log(m)
        return ln_a, S / M, lnM


# ============================================================================================================
# Results
# ============================================================================================================

@dataclass
class SLEResult:
    status: str                         # "dry" | "aqueous" | "solid+aqueous" | "failed"
    T_K: float
    rh: float
    ions: list[str]
    feed: dict[str, float]
    solids: dict[str, float]            # mol of each precipitated solid (formula units)
    aq_ions: dict[str, float]           # mol of each ion in the aqueous phase
    water_kg: float                     # aqueous water mass [kg] (0 if dry)
    molality: dict[str, float]          # ion molalities in the aqueous phase
    ln_a: dict[str, float]              # ln a_i of the ions (aqueous phase; empty if dry)
    si: dict[str, float]                # saturation index ln(IAP a_w^h / K) of every candidate solid (<=0 stable)
    active: list[str]
    gibbs: float                        # value of the minimized function (relative; see module docstring)
    n_newton: int = 0
    n_outer: int = 0
    message: str = ""
    n_eval: int = 0

    @property
    def aqueous_present(self) -> bool:
        return self.water_kg > 0.0

    def solid_mass_g(self) -> dict[str, float]:
        out = {}
        sg = load_subgroup_params()
        for k, n in self.solids.items():
            s = SOLIDS[k]
            mw = 0.0
            for ion, nu in s.ions.items():
                sid, z = ION_REGISTRY[ion]
                mw += nu * (sg.SMWC[sid - 201] if z > 0 else sg.SMWA[sid - 241])
            mw += s.h * 18.01528
            out[k] = n * mw
        return out


# ============================================================================================================
# Solver
# ============================================================================================================

class SLESolver:
    """Fixed-(T, RH) solid-liquid equilibrium solver for a given ion set and candidate solids.

    >>> sol = SLESolver(["Na+", "Cl-", "NH4+", "SO4--"])
    >>> res = sol.solve(feed_from_salts({"NaCl": 1.0, "(NH4)2SO4": 1.0}), T_K=298.15, rh=0.85)

    ``mode`` selects the K(T) representation of :meth:`aiomfac_py.solids.Solid.ln_k` ("fitted" by default).
    """

    def __init__(self, ions: Sequence[str], solids: Sequence[str] | None = None, *, mode: str | None = None,
                 M_cap: float = 400.0, verbose: bool = False):
        self.ions = list(ions)
        bad = [i for i in self.ions if i in ("H+", "HSO4-")]
        if bad:
            raise NotImplementedError(f"acidic species {bad} (bisulfate speciation, acid salts) are not supported yet")
        self.M_cap = M_cap
        keys = list(solids) if solids is not None else [k for k, s in SOLIDS.items() if set(s.ions) <= set(self.ions)]
        self.solids: list[Solid] = [SOLIDS[k] for k in keys]
        self.mode = mode
        self.verbose = verbose
        N, J = len(self.ions), len(self.solids)
        V = np.zeros((N, J))
        for j, s in enumerate(self.solids):
            for ion, nu in s.ions.items():
                V[self.ions.index(ion), j] = nu
        self.V = V
        self.hyd = np.array([s.h for s in self.solids], dtype=float)
        self.z = np.array([ION_REGISTRY[i][1] for i in self.ions], dtype=float)
        self._aq: dict[tuple, AqueousIons] = {}
        self._warm: tuple | None = None

    # ------------------------------------------------------------------------------------------------
    def _aq_for(self, ion_idx) -> AqueousIons:
        key = tuple(int(i) for i in ion_idx)
        if key not in self._aq:
            self._aq[key] = AqueousIons([self.ions[i] for i in key], M_cap=self.M_cap)
        return self._aq[key]

    def _vec(self, feed: dict[str, float]) -> np.ndarray:
        for k in feed:
            if k not in self.ions:
                raise KeyError(f"feed contains ion {k!r} that is not in the solver's ion set {self.ions}")
        return np.array([feed.get(i, 0.0) for i in self.ions], dtype=float)

    # ------------------------------------------------------------------------------------------------
    def solve(self, feed: dict[str, float], T_K: float, rh: float, *, warm: bool = False, tol: float = 1.0e-9,
              max_outer: int = 80, max_newton: int = 80) -> SLEResult:
        """Equilibrium at ``T_K`` [K] and relative humidity ``rh`` (0-1) for the ion feed ``feed`` [mol].

        ``warm=True`` starts the active-set iteration from the previous solution (use in RH scans).
        """
        if not 0.0 < rh < 1.0:
            raise ValueError("rh must be in (0, 1)")
        bfull = self._vec(feed)
        if abs(float(np.dot(self.z, bfull))) > 1.0e-8 * max(1.0, float(np.sum(np.abs(bfull)))):
            raise ValueError("feed is not electroneutral")
        present = bfull > 0.0
        if not np.any(present):
            raise ValueError("empty feed")
        ion_idx = np.flatnonzero(present)
        sol_idx = [j for j in range(len(self.solids)) if np.all(self.V[~present, j] == 0)]
        scale = float(bfull.max())
        b = bfull[ion_idx] / scale
        V = self.V[np.ix_(ion_idx, sol_idx)]
        lnk = np.array([self.solids[j].ln_k(T_K, self.mode) for j in sol_idx])
        hyd = self.hyd[sol_idx]
        res = self._solve_reduced(b, V, lnk, hyd, ion_idx, sol_idx, T_K, math.log(rh), warm, tol, max_outer, max_newton)
        res.feed = {self.ions[i]: float(bfull[i]) for i in range(len(self.ions)) if bfull[i] > 0}
        res.solids = {k: v * scale for k, v in res.solids.items()}
        res.aq_ions = {k: v * scale for k, v in res.aq_ions.items()}
        res.water_kg *= scale
        res.gibbs *= scale
        return res

    # ------------------------------------------------------------------------------------------------
    def _solve_reduced(self, b, V, lnk, hyd, ion_idx, sol_idx, T, ln_rh, warm, tol, max_outer, max_newton) -> SLEResult:
        aq = self._aq_for(ion_idx)
        N, J = V.shape
        ions_r = [self.ions[i] for i in ion_idx]
        keys_r = [self.solids[j].key for j in sol_idx]
        c = lnk - hyd * ln_rh
        ev0 = aq.n_eval
        rh = math.exp(ln_rh)

        def make(status, u, st, S, nn=0, no=0, msg=""):
            nsol = {k: float(v) for k, v in zip(keys_r, u) if v > 1e-14}
            act = [keys_r[j] for j in S]
            if st is None:
                return SLEResult(status, T, rh, ions_r, {}, nsol, {}, 0.0, {}, {}, {}, act, float(c @ u), nn, no, msg,
                                 aq.n_eval - ev0)
            ln_a, w, _ = st
            n_aq = b - V @ u
            si = V.T @ ln_a - c
            return SLEResult(status, T, rh, ions_r, {}, nsol, {i: float(v) for i, v in zip(ions_r, n_aq)}, float(w),
                             {i: float(v / w) for i, v in zip(ions_r, n_aq)}, {i: float(v) for i, v in zip(ions_r, ln_a)},
                             {k: float(v) for k, v in zip(keys_r, si)}, act, float(np.sum(n_aq * ln_a) + c @ u), nn, no,
                             msg, aq.n_eval - ev0)

        # ---- 1. dry assemblage (LP) and stability of the aqueous phase -----------------------------------------
        u_dry = None
        lp = linprog(c, A_eq=V, b_eq=b, bounds=[(0, None)] * J, method="highs")
        if lp.status == 0:
            u_dry = np.maximum(lp.x, 0.0)
            y = np.asarray(lp.eqlin.marginals, dtype=float)
            f_min = self._tangent_plane(aq, y, T, ln_rh, q_feed=b / b.sum())
            sup = [j for j in range(J) if u_dry[j] > 1e-12]
            if self.verbose:
                print(f"[sle] dry LP: {[keys_r[j] for j in sup]}  TPD_min={f_min:.3e}")
            if f_min >= -1.0e-9:
                return make("dry", u_dry, None, sup, msg="aqueous phase unstable at this RH (all salts crystalline)")
        # ---- 2. wet branch ---------------------------------------------------------------------------------------
        out = None
        if warm and self._warm is not None and self._warm[0] == (tuple(ion_idx), tuple(sol_idx)):
            out = self._wet(aq, b, V, c, T, ln_rh, tol, max_outer, max_newton, u0=self._warm[1], S0=self._warm[2])
        if out is None:
            out = self._wet(aq, b, V, c, T, ln_rh, tol, max_outer, max_newton)
        if out is None:
            out = self._homotopy(aq, b, V, lnk, hyd, T, ln_rh, tol, max_outer, max_newton)
        if out is None:
            if u_dry is not None:
                sup = [j for j in range(J) if u_dry[j] > 1e-12]
                return make("dry", u_dry, None, sup, msg="wet branch failed; dry LP assemblage returned")
            return make("failed", np.zeros(J), None, [], msg="no equilibrium found")
        u, S, st, nn, no = out
        self._warm = ((tuple(ion_idx), tuple(sol_idx)), u.copy(), list(S))
        status = "solid+aqueous" if np.any(u > 1e-14) else "aqueous"
        return make(status, u, st, S, nn, no)

    # ------------------------------------------------------------------------------------------------
    def _tangent_plane(self, aq: AqueousIons, y, T, ln_rh, q_feed=None, early_negative: bool = True) -> float:
        """min over neutral aqueous compositions n (sum n_i = 1) of  f(n) = sum_i n_i (ln a_i(n) - y_i)   [JOTA-1 Sec. 4.1].

        Cheap scheme: (1) the feed composition itself (if f < 0 there the aqueous phase is certainly stable and the
        search stops), (2) a coarse grid over mixtures of the salts that can be formed from the ions, (3) a local
        refinement from the best grid point.
        """
        z = aq.charge
        cats = [k for k in range(aq.N) if z[k] > 0]
        ans = [k for k in range(aq.N) if z[k] < 0]
        cols = []
        for cc in cats:
            for a in ans:
                zc, za = int(z[cc]), int(-z[a])
                g = math.gcd(zc, za)
                v = np.zeros(aq.N); v[cc] = za // g; v[a] = zc // g
                cols.append(v / v.sum())
        P = np.array(cols).T
        npair = P.shape[1]
        guess = {"lnM": None}

        def f_n(n):
            n = np.maximum(n / n.sum(), 1.0e-9)
            r = aq.solve_water(n, T, ln_rh, guess["lnM"], clip=True)
            if r is None:
                return 1.0e3
            ln_a, _, lnM = r[:3]
            guess["lnM"] = lnM
            val = float(np.sum(n * (ln_a - y)))
            if len(r) == 4:                      # RH not reachable below the concentration cap: penalize
                val += 5.0 * max(r[3], 0.0) + 1.0e-3
            return val

        def f_t(theta):
            t = np.exp(theta - np.max(theta)); t /= t.sum()
            return f_n(P @ t)

        best = np.inf
        scored = []
        if q_feed is not None:
            guess["lnM"] = None
            fv = f_n(q_feed)
            if fv < -1.0e-9 and early_negative:
                return fv
            best = fv
            scored.append((fv, q_feed / q_feed.sum()))
        cand = [np.eye(npair)[k] for k in range(npair)]
        for i in range(npair):
            for j in range(i + 1, npair):
                t = np.zeros(npair); t[i] = t[j] = 0.5
                cand.append(t)
        cand.append(np.full(npair, 1.0 / npair))
        for t in cand:
            guess["lnM"] = None
            fv = f_n(P @ t)
            scored.append((fv, P @ t))
            best = min(best, fv)
        if npair > 1:
            scored.sort(key=lambda p: p[0])
            for fv0, n0 in scored[:2]:
                # recover softmax coordinates (least squares in pair space)
                t0 = np.linalg.lstsq(P, n0, rcond=None)[0]
                th0 = np.log(np.clip(t0, 1.0e-4, None))
                guess["lnM"] = None
                try:
                    r = minimize(f_t, th0, method="BFGS", options={"gtol": 1e-8, "maxiter": 35})
                    best = min(best, float(r.fun))
                    if r.fun < -1.0e-9 and early_negative:
                        return float(r.fun)
                except Exception:
                    pass
        return best

    # ------------------------------------------------------------------------------------------------
    def _wet(self, aq, b, V, c, T, ln_rh, tol, max_outer, max_newton, u0=None, S0=None):
        """Active-set / Newton iteration (JOTA-1 Table 1) for the case of a present aqueous phase.

        Returns (u, S, (ln a, w, lnM), n_newton, n_outer) or None if it breaks down.
        """
        N, J = V.shape
        u = np.zeros(J) if u0 is None else np.array(u0, dtype=float)
        S = [] if S0 is None else list(S0)
        lnM = [None]

        def st_at(uu):
            n = b - V @ uu
            if np.any(n <= 0.0):
                return None
            r = aq.solve_water(n, T, ln_rh, lnM[0])
            if r is not None:
                lnM[0] = r[2]
            return r

        def phi_of(uu, st):
            return float(np.sum((b - V @ uu) * st[0]) + c @ uu)

        st = st_at(u)
        if st is None:
            return None
        n_newton = 0
        seen = set()
        bmax = float(np.max(b))
        for outer in range(max_outer):
            key = (tuple(sorted(S)), tuple(np.round(u, 9)))
            if key in seen:
                return None                      # cycling
            seen.add(key)
            # ---------------- inner Newton on the active set (reduced KKT, JOTA-1 eq. (53)) --------------------
            converged = False
            for it in range(max_newton):
                st = st_at(u)
                if st is None:
                    return None
                if not S:
                    converged = True
                    break
                si = V.T @ st[0] - c
                F = si[S]
                if float(np.max(np.abs(F))) < tol:
                    converged = True
                    break
                n_newton += 1
                s = len(S)
                n_now = b - V @ u
                H = np.zeros((s, s))
                for kk, jk in enumerate(S):
                    vk = V[:, jk]
                    pos = vk > 0
                    eps = min(1.0e-4 * bmax, 0.1 * float(np.min(n_now[pos] / vk[pos])))
                    up, um = u.copy(), u.copy()
                    up[jk] += eps; um[jk] -= eps
                    sp, sm = st_at(up), st_at(um)
                    if sp is None or sm is None:
                        return None
                    H[:, kk] = -((V.T @ sp[0] - c)[S] - (V.T @ sm[0] - c)[S]) / (2.0 * eps)
                H = 0.5 * (H + H.T)
                reg = 1.0e-12 * max(1.0, float(np.trace(np.abs(H))))
                try:
                    delta = np.linalg.solve(H + reg * np.eye(s), F)
                except np.linalg.LinAlgError:
                    delta = np.linalg.lstsq(H, F, rcond=None)[0]
                # step length: aqueous ions stay positive; solid amounts stay non-negative
                Vd = V[:, S] @ delta
                alpha = 1.0
                m_pos = Vd > 0
                if np.any(m_pos):
                    alpha = min(alpha, 0.9 * float(np.min(n_now[m_pos] / Vd[m_pos])))
                block = None
                neg = np.flatnonzero(delta < 0)
                if neg.size:
                    ratios = -u[S][neg] / delta[neg]
                    kmin = int(np.argmin(ratios))
                    if ratios[kmin] <= alpha:
                        alpha = float(max(ratios[kmin], 0.0)); block = neg[kmin]
                phi0 = phi_of(u, st)
                slope = -float(F @ delta)
                a = alpha
                ok = False
                for _ in range(40):
                    un = u.copy(); un[S] += a * delta
                    if block is not None and a == alpha:
                        un[S[block]] = 0.0
                    sn = st_at(un)
                    if sn is not None and phi_of(un, sn) <= phi0 + 1.0e-4 * a * slope + 1.0e-13 * (1.0 + abs(phi0)):
                        ok = True
                        break
                    a *= 0.5
                if not ok:
                    return None
                u = un
                if block is not None and a == alpha:
                    jb = S[block]
                    S = [j for j in S if j != jb]
                    u[jb] = 0.0
            if not converged:
                return None
            st = st_at(u)
            if st is None:
                return None
            # ---------------- Table 1 step 4: constraint test --------------------------------------------------
            zero = [j for j in S if u[j] <= 1.0e-14]
            if zero:
                S = [j for j in S if j not in zero]
                continue
            si = V.T @ st[0] - c
            inactive = [j for j in range(J) if j not in S]
            if inactive:
                jmax = max(inactive, key=lambda j: si[j])
                if si[jmax] > 1.0e-8:
                    S.append(jmax)
                    continue
            return u, S, st, n_newton, outer + 1
        return None

    # ------------------------------------------------------------------------------------------------
    def _homotopy(self, aq, b, V, lnk, hyd, T, ln_rh, tol, max_outer, max_newton):
        """RH continuation from the dilute side, carrying (u, S): the dual-feasible path of JOTA-1 Sec. 4.2."""
        rh_t = math.exp(ln_rh)
        rh_c = 0.999
        u = None; S = None
        c = lnk - hyd * math.log(rh_c)
        out = self._wet(aq, b, V, c, T, math.log(rh_c), tol, max_outer, max_newton)
        if out is None:
            return None
        u, S = out[0], out[1]
        step = 0.02
        total = out
        while rh_c > rh_t + 1e-12:
            rh_n = max(rh_t, rh_c - step * rh_c)
            ck = lnk - hyd * math.log(rh_n)
            o = self._wet(aq, b, V, ck, T, math.log(rh_n), tol, max_outer, max_newton, u0=u, S0=S)
            if o is None:
                step *= 0.5
                if step < 1e-5:
                    return None
                continue
            u, S, total = o[0], o[1], o
            rh_c = rh_n
            step = min(step * 1.5, 0.1)
        return total


    # ------------------------------------------------------------------------------------------------
    # RH scans, deliquescence / efflorescence relative humidities
    # ------------------------------------------------------------------------------------------------
    def scan_rh(self, feed: dict[str, float], T_K: float, rh_grid, *, descending: bool = False) -> list[SLEResult]:
        """Equilibrium at every RH of ``rh_grid`` (warm-started from the neighbouring point)."""
        grid = sorted(rh_grid, reverse=descending)
        self._warm = None
        out = [self.solve(feed, T_K, float(rh), warm=True) for rh in grid]
        return out

    def aqueous_state(self, feed: dict[str, float], T_K: float, rh: float) -> SLEResult:
        """Metastable (solid-free) aqueous solution at ``rh``: ion activities, water content and the saturation index of
        every candidate solid.  Used for supersaturation / efflorescence analysis."""
        bfull = self._vec(feed)
        ion_idx = np.flatnonzero(bfull > 0)
        sol_idx = [j for j in range(len(self.solids)) if np.all(self.V[np.setdiff1d(np.arange(len(self.ions)), ion_idx), j] == 0)]
        aq = self._aq_for(ion_idx)
        ln_rh = math.log(rh)
        r = aq.solve_water(bfull[ion_idx], T_K, ln_rh)
        if r is None:
            return SLEResult("failed", T_K, rh, [self.ions[i] for i in ion_idx], dict(feed), {}, {}, 0.0, {}, {}, {}, [], 0.0,
                             message="RH not reachable within the concentration cap")
        ln_a, w, _ = r
        V = self.V[np.ix_(ion_idx, sol_idx)]
        c = np.array([self.solids[j].ln_k(T_K, self.mode) - self.solids[j].h * ln_rh for j in sol_idx])
        si = V.T @ ln_a - c
        ions_r = [self.ions[i] for i in ion_idx]
        n = bfull[ion_idx]
        return SLEResult("aqueous", T_K, rh, ions_r, dict(feed), {}, {i: float(v) for i, v in zip(ions_r, n)}, float(w),
                         {i: float(v / w) for i, v in zip(ions_r, n)}, {i: float(v) for i, v in zip(ions_r, ln_a)},
                         {self.solids[j].key: float(v) for j, v in zip(sol_idx, si)}, [], 0.0)

    def deliquescence_rh(self, feed: dict[str, float], T_K: float, lo: float = 0.05, hi: float = 0.995,
                         tol: float = 2.0e-4) -> float:
        """Lowest RH at which an aqueous phase exists for this feed (mutual deliquescence RH for a mixture).
        Bisection on the stability of the aqueous phase; returns ``nan`` if it never forms in [lo, hi]."""
        if self.solve(feed, T_K, hi).status == "dry":
            return float("nan")
        if self.solve(feed, T_K, lo).status != "dry":
            return lo
        a, b = lo, hi
        while b - a > tol:
            mid = 0.5 * (a + b)
            if self.solve(feed, T_K, mid).status == "dry":
                a = mid
            else:
                b = mid
        return 0.5 * (a + b)

    def full_dissolution_rh(self, feed: dict[str, float], T_K: float, lo: float = 0.05, hi: float = 0.995,
                            tol: float = 2.0e-4) -> float:
        """Lowest RH above which no solid is left (aqueous phase only); bisection."""
        def clear(rh):
            r = self.solve(feed, T_K, rh)
            return r.status == "aqueous"
        if not clear(hi):
            return float("nan")
        if clear(lo):
            return lo
        a, b = lo, hi
        while b - a > tol:
            mid = 0.5 * (a + b)
            if clear(mid):
                b = mid
            else:
                a = mid
        return 0.5 * (a + b)

    def efflorescence_rh(self, feed: dict[str, float], T_K: float, ln_s_crit: float | dict[str, float] = 0.5,
                         lo: float = 0.05, hi: float = 0.995, tol: float = 2.0e-4) -> dict:
        """Metastable path: starting from the dilute aqueous solution, lower the RH without allowing any solid to form
        until the saturation index of some salt reaches ``ln_s_crit`` (critical supersaturation ratio ``ln S``;
        a float for all salts or a dict by solid key).  Returns the efflorescence RH and the crystallizing salt.

        The critical supersaturation is *not* a thermodynamic quantity -- it has to come from experiment (see
        :func:`implied_ln_s_crit`).  ``ln_s_crit = 0`` reproduces the saturation (deliquescence) point."""
        def crit(k):
            return ln_s_crit.get(k, 0.5) if isinstance(ln_s_crit, dict) else ln_s_crit

        def excess(rh):
            r = self.aqueous_state(feed, T_K, rh)
            if r.status == "failed":
                return 1.0e3, None
            vals = {k: v - crit(k) for k, v in r.si.items()}
            kmax = max(vals, key=vals.get)
            return vals[kmax], kmax

        if excess(hi)[0] > 0:
            return {"rh": float("nan"), "salt": None}
        a, b = lo, hi
        if excess(lo)[0] <= 0:
            return {"rh": float("nan"), "salt": None}
        while b - a > tol:
            mid = 0.5 * (a + b)
            if excess(mid)[0] > 0:
                a = mid
            else:
                b = mid
        rh = 0.5 * (a + b)
        return {"rh": rh, "salt": excess(a)[1]}


def implied_ln_s_crit(solid: str, T_K: float, rh_eff: float, mode: str | None = None) -> float:
    """ln S = ln(IAP a_w^h / K) of the binary solution of ``solid`` at the water activity ``rh_eff`` (e.g. a measured
    efflorescence RH) -- the critical supersaturation implied by that observation, for use in ``efflorescence_rh``."""
    s = SOLIDS[solid]
    ions = list(s.ions)
    aq = AqueousIons(ions)
    nu = np.array([s.ions[i] for i in ions], dtype=float)
    r = aq.solve_water(nu, T_K, math.log(rh_eff))
    if r is None:
        return float("nan")
    ln_a = r[0]
    return float(np.sum(nu * ln_a) + s.h * math.log(rh_eff) - s.ln_k(T_K, mode))


# ============================================================================================================
# Binary saturation point (single-salt solubility and DRH)
# ============================================================================================================

def binary_saturation(solid: str, T_K: float, mode: str | None = None, M_cap: float = 400.0) -> dict:
    """Saturated binary solution of ``solid`` (key of :data:`aiomfac_py.solids.SOLIDS`): molality of the formula
    unit and water activity (= the deliquescence relative humidity of the pure salt)."""
    s = SOLIDS[solid]
    ions = list(s.ions)
    aq = AqueousIons(ions, M_cap=M_cap)
    nu = np.array([s.ions[i] for i in ions], dtype=float)
    ln_k = s.ln_k(T_K, mode)

    def f(m_salt):
        m = nu * m_salt
        lng, ln_w = aq.ln_gamma_aw(m, T_K)
        return float(np.sum(nu * (lng + np.log(m))) + s.h * ln_w - ln_k), ln_w

    ms = np.geomspace(1.0e-3, M_cap / float(nu.sum()), 120)
    prev = None
    for m in ms:
        try:
            fv = f(m)[0]
        except Exception:
            continue
        if prev is not None and fv * prev[1] < 0:
            m_sat = brentq(lambda x: f(x)[0], prev[0], m, xtol=1e-12, rtol=1e-12)
            return {"molality": m_sat, "aw": math.exp(f(m_sat)[1]), "T_K": T_K}
        prev = (m, fv)
    return {"molality": float("nan"), "aw": float("nan"), "T_K": T_K}

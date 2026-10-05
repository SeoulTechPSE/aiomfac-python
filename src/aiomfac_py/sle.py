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
* Volatile gases NH3, HNO3, HCl (:mod:`aiomfac_py.gases`, Clegg et al. 1998 constants) are optional: they enter as extra
  columns of the same reduced problem, with the aqueous ions they remove (HNO3: H+ + NO3-, NH3: NH4+ - H+) -- the gas
  is a "solid" whose chemical potential is ln p + const.  ``solve(..., p_gas=...)``: open system (fixed partial
  pressures, free-sign amounts, solid growth may be unbounded -> reported as failure); ``solve_closed``: closed
  system with ideal-gas amounts as positive unknowns (ln p = ln(P n_j/(n_air + sum n))), a dual (element-potential)
  convex solve for solid + gas states and the same tangent-plane test for the aqueous phase.  Without ``p_gas`` /
  ``solve_closed`` the feed is a closed set of ions + water (strong acids stay in the particle).
* Acid sulfates (H+ with SO4--): the ions H+ and SO4-- are *stoichiometric totals*; the HSO4- <-> H+ + SO4--
  equilibrium is solved inside every activity evaluation (:class:`AqueousIons`, constant of
  :mod:`aiomfac_py.dissociation`).  Because the chemical potential of a stoichiometric component equals that of the
  free ion in equilibrium, the KKT conditions, active-set logic and tangent-plane test keep exactly the same form;
  acid solids (NH4HSO4, letovicite) are ordinary entries of :mod:`aiomfac_py.solids` whose ions are NH4+/H+/SO4--.
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
from .dissociation import ln_k_hso4_at_t
from .carbonate import gamma_co2_mr, ln_k1_hco3_at_t, ln_k2_hco3_at_t, ln_kw_at_t
from .model import ActivityModel
from .params import load_subgroup_params
from .solids import ION_REGISTRY, SOLIDS, Solid
from .gases import GASES

MW_WATER = 0.01801528        # kg/mol
_NEG = -1.0e300

# formula unit -> ions (for building a feed from salts)
SALT_IONS: dict[str, dict[str, int]] = {
    "NaCl": {"Na+": 1, "Cl-": 1}, "KCl": {"K+": 1, "Cl-": 1}, "NH4Cl": {"NH4+": 1, "Cl-": 1},
    "NaNO3": {"Na+": 1, "NO3-": 1}, "KNO3": {"K+": 1, "NO3-": 1}, "NH4NO3": {"NH4+": 1, "NO3-": 1},
    "Na2SO4": {"Na+": 2, "SO4--": 1}, "K2SO4": {"K+": 2, "SO4--": 1}, "(NH4)2SO4": {"NH4+": 2, "SO4--": 1},
    "MgCl2": {"Mg++": 1, "Cl-": 2}, "Mg(NO3)2": {"Mg++": 1, "NO3-": 2}, "MgSO4": {"Mg++": 1, "SO4--": 1},
    "CaCl2": {"Ca++": 1, "Cl-": 2}, "Ca(NO3)2": {"Ca++": 1, "NO3-": 2}, "CaSO4": {"Ca++": 1, "SO4--": 1},
    # acid sulfates (H+ / SO4-- are stoichiometric totals; HSO4- is speciated in the aqueous phase)
    "H2SO4": {"H+": 2, "SO4--": 1}, "NH4HSO4": {"NH4+": 1, "H+": 1, "SO4--": 1},
    "(NH4)3H(SO4)2": {"NH4+": 3, "H+": 1, "SO4--": 2}, "NaHSO4": {"Na+": 1, "H+": 1, "SO4--": 1},
    "KHSO4": {"K+": 1, "H+": 1, "SO4--": 1},
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
        if "HSO4-" in ions:
            raise ValueError("pass the stoichiometric ions H+ and SO4-- instead of HSO4- (bisulfate is speciated internally)")
        self.ions = list(ions)
        self.N = len(self.ions)
        self.M_cap = M_cap
        # acid-sulfate system: H+ and SO4-- are *total* (stoichiometric) amounts; the HSO4- <-> H+ + SO4-- equilibrium is
        # solved inside every activity evaluation (see ln_gamma_aw) and HSO4- is carried by the template model only.
        self._carb = "CO3--" in self.ions
        if self._carb and "H+" not in self.ions:
            raise ValueError("a carbonate system needs H+ as the (sign-free) proton-excess component")
        self._acid = ("H+" in self.ions) and ("SO4--" in self.ions) and not self._carb
        model_ions = list(self.ions)
        if self._acid:
            model_ions += ["HSO4-"]
        if self._carb:
            model_ions += ["HCO3-", "OH-"] + (["HSO4-"] if "SO4--" in self.ions else [])
        cats = [i for i in model_ions if ION_REGISTRY[i][1] > 0]
        ans = [i for i in model_ions if ION_REGISTRY[i][1] < 0]
        if not cats or not ans:
            raise ValueError("need at least one cation and one anion")
        pairs = [(c, ans[0]) for c in cats] + [(cats[0], a) for a in ans[1:]]
        comps = [Component(1, "Water", ((16, 1),))]
        for k, (c, a) in enumerate(pairs, start=2):
            zc, za = ION_REGISTRY[c][1], -ION_REGISTRY[a][1]
            g = math.gcd(zc, za)
            comps.append(Component(k, c + a, ((ION_REGISTRY[c][0], za // g), (ION_REGISTRY[a][0], zc // g))))
        self.model = ActivityModel(comps, assume_complete=self._carb)
        mx = self.model.mixture
        self._mx = mx
        self._ngi = mx.ngi
        self._nc, self._na = mx.sr.n_cation, mx.sr.n_anion
        self._pos = []                    # (is_cation, index in smc/sma) per ion
        for ion in self.ions:
            sid, z = ION_REGISTRY[ion]
            self._pos.append((z > 0, mx.cat_index[sid] if z > 0 else mx.an_index[sid]))
        if self._acid:
            self._ih = mx.cat_index[ION_REGISTRY["H+"][0]]
            self._iso = mx.an_index[ION_REGISTRY["SO4--"][0]]
            self._ihs = mx.an_index[ION_REGISTRY["HSO4-"][0]]
            self._kh = self.ions.index("H+")
            self._kso = self.ions.index("SO4--")
        if self._carb:
            self._ih = mx.cat_index[ION_REGISTRY["H+"][0]]
            self._kh = self.ions.index("H+")
            self._kc = self.ions.index("CO3--")
            self._ico3 = mx.an_index[ION_REGISTRY["CO3--"][0]]
            self._ihco3 = mx.an_index[ION_REGISTRY["HCO3-"][0]]
            self._ioh = mx.an_index[ION_REGISTRY["OH-"][0]]
            self._ks = self.ions.index("SO4--") if "SO4--" in self.ions else None
            if self._ks is not None:
                self._iso = mx.an_index[ION_REGISTRY["SO4--"][0]]
                self._ihs = mx.an_index[ION_REGISTRY["HSO4-"][0]]
            self._A_last = None
            self._lnh_last = None
        self.free_h_row = self._kh if self._carb else None      # H+ is a sign-free proton-excess amount in carbonate systems
        self._x_last = None
        self._xmax_last = 1.0
        self.charge = np.array([ION_REGISTRY[i][1] for i in self.ions], dtype=float)
        self._xn = np.array([1.0])
        self.n_eval = 0

    # -------------------------------------------------------------------------------------------------
    def _terms(self, smc, sma, T):
        mod = self.model
        x = mod._x_from_molalities(self._xn, smc, sma)
        lr, mr, sr = mod.lr_mr_sr(T, smc, sma, self._xn, x)
        return x, lr, mr, sr

    def _speciate(self, mh: float, ms: float, smc, sma, T: float):
        """Bisulfate equilibrium for total H+ (``mh``) and sulfate (``ms``) molalities by fixed-point iteration on the
        activity-coefficient ratio  Gamma = g_H g_SO4 / g_HSO4  with the exact quadratic solution at fixed Gamma
        ((mh-x)(ms-x)/x = K/Gamma).  Warm-started from the previous solution; converges in 2-5 activity evaluations.
        Fills ``smc/sma`` with the speciated molalities and returns the final (x, lr, mr, sr) and HSO4 molality."""
        K = math.exp(ln_k_hso4_at_t(T))
        xmax = min(mh, ms)
        xh = min(max(self._x_last * xmax / max(self._xmax_last, 1.0e-300), 0.0), xmax) if self._x_last is not None \
            else 0.5 * xmax
        xh = min(max(xh, 1.0e-12 * xmax), xmax * (1.0 - 1.0e-12))
        nn, nc = 1, self._nc
        lo_b, hi_b = 1.0e-12 * xmax, xmax * (1.0 - 1.0e-12)
        x_prev = f_prev = None
        for it in range(80):
            smc[self._ih] = mh - xh
            sma[self._ihs] = xh
            sma[self._iso] = ms - xh
            x, lr, mr, sr = self._terms(smc, sma, T)
            lg_h = mr.ln_gamma_cation[self._ih] + sr.ln_gamma_sr[nn + self._ih] + lr.ln_gamma_cation[self._ih]
            lg_hs = mr.ln_gamma_anion[self._ihs] + sr.ln_gamma_sr[nn + nc + self._ihs] + lr.ln_gamma_anion[self._ihs]
            lg_so = mr.ln_gamma_anion[self._iso] + sr.ln_gamma_sr[nn + nc + self._iso] + lr.ln_gamma_anion[self._iso]
            # the -tmolal conversion (molal <-> mole-fraction-based ln gamma) does not cancel: 1 + 1 - 1 = 1 net term
            q = K / math.exp(max(min(lg_h + lg_so - lg_hs - mr.tmolal, 50.0), -50.0))
            s_ = mh + ms + q
            g = 0.5 * (s_ - math.sqrt(max(s_ * s_ - 4.0 * mh * ms, 0.0)))     # fixed-point map x -> G(x)
            g = min(max(g, lo_b), hi_b)
            f = g - xh
            if abs(f) <= 1.0e-10 * xmax:
                xh = g
                break
            # plain substitution first, then secant steps on f(x) = G(x) - x (the map is slowly contracting in
            # concentrated solutions), safeguarded to stay inside the physical interval
            if x_prev is not None and abs(f - f_prev) > 1.0e-300 and it >= 2:
                xs = xh - f * (xh - x_prev) / (f - f_prev)
                xn = xs if lo_b < xs < hi_b else g
            else:
                xn = g
            x_prev, f_prev = xh, f
            xh = xn
        self._x_last, self._xmax_last = xh, xmax
        return x, lr, mr, sr, xh

    # -------------------------------------------------------------------------------------------------
    def _gam_prime(self, mr, sr, lr, is_cat: bool, idx: int) -> float:
        nn, nc = 1, self._nc
        if is_cat:
            return mr.ln_gamma_cation[idx] + sr.ln_gamma_sr[nn + idx] + lr.ln_gamma_cation[idx] - mr.tmolal
        return mr.ln_gamma_anion[idx] + sr.ln_gamma_sr[nn + nc + idx] + lr.ln_gamma_anion[idx] - mr.tmolal

    def _speciate_carb(self, m: np.ndarray, T: float, smc, sma):
        """Carbonate (and, if SO4-- is present, bisulfate) speciation for the stoichiometric totals ``m``:

        C_T = [CO2] + [HCO3-] + [CO3--]   (carried as ``CO3--``),
        P   = [H+] - [OH-] + [HCO3-] + 2[CO2] (+ [HSO4-])   (carried as ``H+``; sign-free proton excess),
        K1 = a_H a_HCO3/(a_CO2 a_w), K2 = a_H a_CO3/a_HCO3, Kw = a_H a_OH/a_w  (constants and conventions of
        :mod:`aiomfac_py.carbonate`, i.e. the AIOMFAC Fortran code; gamma(CO2) from the salting-out coefficients).
        At fixed activity-coefficient ratios the system reduces to one monotone equation in [H+]; the ratios are updated
        by fixed-point iteration.  Returns ln a of the free species of every stoichiometric ion and ln a_w."""
        CT, P = float(m[self._kc]), float(m[self._kh])
        ST = float(m[self._ks]) if self._ks is not None else 0.0
        K1, K2, Kw = math.exp(ln_k1_hco3_at_t(T)), math.exp(ln_k2_hco3_at_t(T)), math.exp(ln_kw_at_t(T))
        K3 = math.exp(ln_k_hso4_at_t(T)) if self._ks is not None else 0.0
        A = self._A_last if self._A_last is not None else (K1, K2, Kw, K3)
        A1, A2, Aw, q = A
        lnh0 = self._lnh_last
        for it in range(60):
            def f(lnh):
                h = math.exp(lnh)
                den = 1.0 + A1 / h + A1 * A2 / (h * h)
                val = h - Aw / h + CT * (A1 / h + 2.0) / den - P
                if ST > 0.0:
                    val += ST * h / (h + q)
                return val
            lo, hi = math.log(1.0e-40), math.log(1.0e4)
            if lnh0 is not None:
                a, b = lnh0 - 1.0, lnh0 + 1.0
                if f(a) < 0.0 < f(b):
                    lo, hi = a, b
            lnh = brentq(f, lo, hi, xtol=1.0e-14, rtol=1.0e-13)
            h = math.exp(lnh)
            den = 1.0 + A1 / h + A1 * A2 / (h * h)
            co2 = CT / den
            hco3 = co2 * A1 / h
            co3 = co2 * A1 * A2 / (h * h)
            oh = Aw / h
            hso4 = ST * h / (h + q) if ST > 0.0 else 0.0
            smc[self._ih] = h
            sma[self._ico3], sma[self._ihco3], sma[self._ioh] = co3, hco3, oh
            if self._ks is not None:
                sma[self._iso], sma[self._ihs] = ST - hso4, hso4
            x, lr, mr, sr = self._terms(smc, sma, T)
            g = lambda c, i: self._gam_prime(mr, sr, lr, c, i)
            lgH, lgHC, lgC, lgOH = g(True, self._ih), g(False, self._ihco3), g(False, self._ico3), g(False, self._ioh)
            lgCO2 = gamma_co2_mr(self._mx, smc, sma)
            lgW = mr.ln_gamma_neutral[0] + sr.ln_gamma_sr[0] + lr.ln_gamma_neutral[0]
            xw = float(x[0])
            nA1 = xw * math.exp(math.log(K1) - (lgH + lgHC - lgCO2 - lgW))
            nA2 = math.exp(math.log(K2) - (lgH + lgC - lgHC))
            nAw = xw * math.exp(math.log(Kw) - (lgH + lgOH - lgW))
            nq = q
            if self._ks is not None:
                # raw (no -tmolal) gamma ratio as in _speciate: the net conversion term is a single -tmolal
                lg_h = mr.ln_gamma_cation[self._ih] + sr.ln_gamma_sr[1 + self._ih] + lr.ln_gamma_cation[self._ih]
                nc = self._nc
                lg_hs = mr.ln_gamma_anion[self._ihs] + sr.ln_gamma_sr[1 + nc + self._ihs] + lr.ln_gamma_anion[self._ihs]
                lg_so = mr.ln_gamma_anion[self._iso] + sr.ln_gamma_sr[1 + nc + self._iso] + lr.ln_gamma_anion[self._iso]
                nq = K3 / math.exp(max(min(lg_h + lg_so - lg_hs - mr.tmolal, 50.0), -50.0))
            err = max(abs(math.log(nA1 / A1)), abs(math.log(nA2 / A2)), abs(math.log(nAw / Aw)),
                      abs(math.log(nq / q)) if self._ks is not None else 0.0)
            A1, A2, Aw, q = nA1, nA2, nAw, nq
            lnh0 = lnh
            if err < 1.0e-11:
                break
        self._A_last = (A1, A2, Aw, q)
        self._lnh_last = lnh
        ln_a = np.empty(self.N)
        for k, (is_cat, idx) in enumerate(self._pos):
            ln_a[k] = self._gam_prime(mr, sr, lr, is_cat, idx) + math.log(max(
                {self._kh: h, self._kc: co3}.get(k, (ST - hso4) if (self._ks is not None and k == self._ks) else float(m[k])),
                1.0e-300))
        ln_w = lgW + math.log(xw)
        return ln_a, ln_w

    def ln_a_aw(self, m: np.ndarray, T: float):
        """ln a_i of the free species behind every stoichiometric ion, and ln a_w (``m``: total molalities; in carbonate
        systems the H+ entry is the sign-free proton excess)."""
        if not self._carb:
            lng, lw = self.ln_gamma_aw(m, T)
            return lng + np.log(m), lw
        self.n_eval += 1
        smc = np.zeros(self._ngi); sma = np.zeros(self._ngi)
        for (is_cat, idx), v in zip(self._pos, m):
            (smc if is_cat else sma)[idx] = v
        return self._speciate_carb(m, T, smc, sma)

    def ln_gamma_aw(self, m: np.ndarray, T: float):
        """Effective ln gamma_i of all (stoichiometric) ions and ln a_w, for the total ion molalities ``m`` [mol/kg water].

        "Effective" means  ln gamma_i + ln m_i = ln a_i  of the *free* species.  This only differs from the ordinary
        activity coefficient for H+ and SO4-- in an acid-sulfate system, where ``m`` holds the total (stoichiometric)
        molalities and the HSO4- <-> H+ + SO4-- equilibrium (Knopf et al. 2003 constant, as in
        :mod:`aiomfac_py.dissociation`) is solved first.  The chemical potential of the stoichiometric component H (or SO4)
        equals that of the free ion in equilibrium, so ln a_i of the free ion is exactly what the solid saturation
        conditions need."""
        if self._carb:
            ln_a, lw = self.ln_a_aw(m, T)
            return ln_a - np.log(np.maximum(np.abs(m), 1.0e-300)), lw
        self.n_eval += 1
        smc = np.zeros(self._ngi); sma = np.zeros(self._ngi)
        for (is_cat, idx), v in zip(self._pos, m):
            (smc if is_cat else sma)[idx] = v
        corr_h = corr_so = 0.0
        if self._acid and m[self._kh] > 0.0 and m[self._kso] > 0.0:
            mh_t, mso_t = float(m[self._kh]), float(m[self._kso])
            x, lr, mr, sr, xh = self._speciate(mh_t, mso_t, smc, sma, T)
            corr_h = math.log(max(mh_t - xh, 1.0e-300) / mh_t)
            corr_so = math.log(max(mso_t - xh, 1.0e-300) / mso_t)
        else:
            x, lr, mr, sr = self._terms(smc, sma, T)
        nn, nc = 1, self._nc
        ln_w = mr.ln_gamma_neutral[0] + sr.ln_gamma_sr[0] + lr.ln_gamma_neutral[0] + math.log(x[0])
        out = np.empty(self.N)
        for k, (is_cat, idx) in enumerate(self._pos):
            if is_cat:
                out[k] = mr.ln_gamma_cation[idx] + sr.ln_gamma_sr[nn + idx] + lr.ln_gamma_cation[idx] - mr.tmolal
            else:
                out[k] = mr.ln_gamma_anion[idx] + sr.ln_gamma_sr[nn + nc + idx] + lr.ln_gamma_anion[idx] - mr.tmolal
        if corr_h or corr_so:
            out[self._kh] += corr_h
            out[self._kso] += corr_so
        return out, ln_w

    def solve_water(self, n: np.ndarray, T: float, ln_rh: float, lnM_guess: float | None = None,
                    clip: bool = False):
        """Water mass w [kg] such that a_w(n/w) = RH.  Returns (ln a_i, w, lnM) or None if RH is out of reach.

        With ``clip=True`` an unreachable RH returns the state at the concentration cap ``M_cap`` and the extra element
        ``ln a_w - ln RH`` (> 0) as a fourth entry (used by the stability test to keep its objective continuous)."""
        S = float(np.sum(np.abs(n)))

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
                    ln_a_, ln_w = self.ln_a_aw(m, T)
                    return ln_a_, S / M, hi, ln_w - ln_rh
            lnM = brentq(g, lo, hi, xtol=1.0e-13, rtol=1.0e-13)
        except (FloatingPointError, ValueError, ZeroDivisionError, OverflowError):
            return None
        M = math.exp(lnM)
        m = n * (M / S)
        ln_a, _ = self.ln_a_aw(m, T)
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
    gas: dict = field(default_factory=dict)      # mol of each volatile gas (HNO3, HCl, NH3) in the gas phase
    p_gas: dict = field(default_factory=dict)    # partial pressures [atm] of those gases

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
        # bisulfate is carried as its stoichiometric ions: HSO4- -> H+ + SO4-- (speciated inside AqueousIons)
        self.ions = []
        alias = {"HSO4-": ("H+", "SO4--"), "HCO3-": ("H+", "CO3--"), "OH-": ("H+",)}
        for i in ions:
            for j in alias.get(i, (i,)):
                if j not in self.ions:
                    self.ions.append(j)
        if "CO3--" in self.ions and "H+" not in self.ions:
            self.ions.append("H+")                # carbonate systems always carry the proton-excess component
        self._carb = "CO3--" in self.ions
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
        self.hyd = np.array([s.h_eff for s in self.solids], dtype=float)
        self.z = np.array([ION_REGISTRY[i][1] for i in self.ions], dtype=float)
        self._aq: dict[tuple, AqueousIons] = {}
        self._warm: tuple | None = None
        self._gcols: list[int] = []
        self._gnames: list[str] = []
        self._closed = None                     # (n_air, P) in solver units while a closed-gas wet problem is solved
        self._fast = False                      # skip the (costly) dry-state test in inner iterations of solve_closed

    # ------------------------------------------------------------------------------------------------
    def _aq_for(self, ion_idx) -> AqueousIons:
        key = tuple(int(i) for i in ion_idx)
        if key not in self._aq:
            self._aq[key] = AqueousIons([self.ions[i] for i in key], M_cap=self.M_cap)
        return self._aq[key]

    def _vec(self, feed: dict[str, float]) -> np.ndarray:
        if any(k in feed for k in ("HSO4-", "HCO3-", "OH-")):   # as-input aliases -> stoichiometric ions
            feed = dict(feed)
            if "HSO4-" in feed:                  # bisulfate = H+ + SO4--
                nb = feed.pop("HSO4-")
                feed["H+"] = feed.get("H+", 0.0) + nb
                feed["SO4--"] = feed.get("SO4--", 0.0) + nb
            if "HCO3-" in feed:                  # bicarbonate = H+ + CO3--
                nb = feed.pop("HCO3-")
                feed["H+"] = feed.get("H+", 0.0) + nb
                feed["CO3--"] = feed.get("CO3--", 0.0) + nb
            if "OH-" in feed:                    # hydroxide = -H+ (proton excess)
                feed["H+"] = feed.get("H+", 0.0) - feed.pop("OH-")
        for k in feed:
            if k not in self.ions:
                raise KeyError(f"feed contains ion {k!r} that is not in the solver's ion set {self.ions}")
        return np.array([feed.get(i, 0.0) for i in self.ions], dtype=float)

    # ------------------------------------------------------------------------------------------------
    def solve(self, feed: dict[str, float], T_K: float, rh: float, *, p_gas: dict[str, float] | None = None,
              warm: bool = False, tol: float = 1.0e-9, max_outer: int = 80, max_newton: int = 80) -> SLEResult:
        """Equilibrium at ``T_K`` [K] and relative humidity ``rh`` (0-1) for the ion feed ``feed`` [mol].

        ``p_gas`` (optional) puts the particle in contact with an infinite gas reservoir of fixed partial pressures
        [atm], e.g. ``{"HNO3": 1e-9, "NH3": 5e-9}``: the volatile species then exchange freely (the result's ``gas``
        gives the net amount transferred to the gas phase, negative = uptake).  See :meth:`solve_closed` for the
        closed system with a finite amount of air.

        ``warm=True`` starts the active-set iteration from the previous solution (use in RH scans).
        """
        if not 0.0 < rh < 1.0:
            raise ValueError("rh must be in (0, 1)")
        bfull = self._vec(feed)
        neg = bfull < 0.0
        if self._carb and "H+" in self.ions:
            neg[self.ions.index("H+")] = False         # H+ is the sign-free proton excess (OH- = -H+)
        if np.any(neg):
            raise ValueError("feed amounts must be non-negative")
        gk = list(p_gas) if p_gas else []
        lnp = np.array([math.log(p_gas[k]) for k in gk])
        res = self._solve_vec(bfull, T_K, rh, gk, lnp, warm, tol, max_outer, max_newton)
        res.feed = {self.ions[i]: float(bfull[i]) for i in range(len(self.ions)) if bfull[i] > 0}
        return res

    def _closed_dry(self, bfull, gk, G, T, rh, n_air, P, gt0) -> SLEResult | None:
        """Closed system without an aqueous phase: crystalline solids + ideal gas, solved in the dual (element-potential)
        space, plus the tangent-plane test of the aqueous phase.  Returns the dry result, or ``None`` when an aqueous
        phase is stable (the caller then solves the wet problem).

        Unknowns y (potentials of the stoichiometric ions) and the amounts u_A of the active solids satisfy
        ``V_A^T y = c_A`` and ``b - V_A u_A - G g(y) = 0`` with the gas amounts g_k = n_air q_k / (1 - sum q), where
        q_k = p_k/P = exp(G_k^T y - ln K_k)."""
        K = len(gk)
        present = bfull != 0.0
        for k in range(K):
            present = present | (G[:, k] != 0.0)
        ion_idx = np.flatnonzero(present)
        sol_idx = [j for j in range(len(self.solids)) if np.all(self.V[~present, j] == 0)]
        scale = float(np.abs(bfull).max())
        b = bfull[ion_idx] / scale
        n_air_s = n_air / scale
        Vs = self.V[np.ix_(ion_idx, sol_idx)]
        Gr = G[ion_idx]
        N, J = Vs.shape
        lnk = np.array([self.solids[j].ln_k(T, self.mode) for j in sol_idx])
        lnKg = np.array([GASES[g].ln_k(T) - GASES[g].h * math.log(rh) for g in gk])     # water activity of the gas reaction
        c = lnk                                                       # dry: hydrate water is absent, a_w^h only matters in solution
        # a dry crystalline phase has no water activity; the hydrates are therefore excluded unless RH is high enough
        # to keep them stable: solids with h > 0 enter through c_j = ln K - h ln RH (aw = RH of the surroundings)
        hyd = self.hyd[sol_idx]
        c = lnk - hyd * math.log(rh)
        if J == 0:
            return None

        def gas_of(y):
            q = np.exp(Gr.T @ y - lnKg) / P
            S = q.sum()
            return q, S, n_air_s * q / (1.0 - S)

        # starting point: relaxed LP (gas columns cost ln K + ln p_nominal)
        g0 = np.maximum(gt0 / scale, 1.0e-6)
        p_nom = np.log(np.maximum(g0 / (n_air_s + g0.sum()), 1.0e-30) * P)
        cost = np.concatenate([c, lnKg + p_nom])
        lp = linprog(cost, A_eq=np.hstack([Vs, Gr]), b_eq=b, bounds=[(0, None)] * (J + K), method="highs")
        if lp.status != 0:
            return None
        y = np.asarray(lp.eqlin.marginals, dtype=float)
        A = [j for j in range(J) if lp.x[j] > 1.0e-12]
        u = np.zeros(J); u[A] = lp.x[A]
        # active-set Newton in (y, u_A)
        seen = set()
        for outer in range(60):
            key = tuple(sorted(A))
            if key in seen and outer > 5:
                return None
            seen.add(key)
            nA = len(A)
            ok = False
            for it in range(80):
                q, S, g = gas_of(y)
                F1 = Vs[:, A].T @ y - c[A] if nA else np.zeros(0)
                F2 = b - Vs[:, A] @ u[A] - Gr @ g
                F = np.concatenate([F1, F2])
                if float(np.max(np.abs(F))) < 1.0e-11:
                    ok = True
                    break
                dg = np.zeros((K, N))
                lam = (q[:, None] * Gr.T)                              # q_k G_k
                dg = n_air_s * (lam / (1.0 - S) + q[:, None] * lam.sum(axis=0)[None, :] / (1.0 - S) ** 2)
                M = Gr @ dg
                Jm = np.zeros((nA + N, N + nA))
                if nA:
                    Jm[:nA, :N] = Vs[:, A].T
                    Jm[nA:, N:] = -Vs[:, A]
                Jm[nA:, :N] = -M
                try:
                    dz = np.linalg.lstsq(Jm, -F, rcond=None)[0]
                except np.linalg.LinAlgError:
                    return None
                dy = dz[:N]
                mx = float(np.max(np.abs(dy)))
                if mx > 5.0:
                    dz = dz * (5.0 / mx)
                t = 1.0
                nF = float(np.linalg.norm(F))
                while t > 1.0e-4:
                    yn = y + t * dz[:N]
                    qn, Sn, gn = gas_of(yn)
                    if Sn < 0.5:
                        un = u.copy(); un[A] = u[A] + t * dz[N:]
                        Fn = np.concatenate([(Vs[:, A].T @ yn - c[A]) if nA else np.zeros(0),
                                             b - Vs[:, A] @ un[A] - Gr @ gn])
                        if np.linalg.norm(Fn) < nF * (1.0 - 1.0e-4 * t) or np.linalg.norm(Fn) < 1.0e-13:
                            break
                    t *= 0.5
                else:
                    return None
                y, u = yn, un
            if not ok:
                return None
            neg = [j for j in A if u[j] < -1.0e-14]
            if neg:
                jn = min(neg, key=lambda j: u[j])
                A.remove(jn); u[jn] = 0.0
                continue
            viol = V_viol = Vs.T @ y - c
            cand = [j for j in range(J) if j not in A and viol[j] > 1.0e-9]
            if cand:
                A.append(max(cand, key=lambda j: viol[j]))
                continue
            break
        else:
            return None
        q, S, g = gas_of(y)
        # stability of an aqueous phase against this dry state (tangent-plane test, same as the open-system branch)
        aq = self._aq_for(ion_idx)
        resid_ions = b - Gr @ g
        q_feed = resid_ions / resid_ions.sum() if resid_ions.min() > 1.0e-12 and resid_ions.sum() > 0 else None
        f_min = self._tangent_plane(aq, y, T, math.log(rh), q_feed=q_feed)
        if f_min < -1.0e-9:
            return None
        ions_r = [self.ions[i] for i in ion_idx]
        keys_r = [self.solids[j].key for j in sol_idx]
        return SLEResult("dry", T, rh, ions_r, {}, {k: float(v * scale) for k, v in zip(keys_r, u) if v > 1e-14}, {}, 0.0, {}, {},
                         {k: float(v) for k, v in zip(keys_r, Vs.T @ y - c)}, [keys_r[j] for j in A], float(c @ u * scale), 0, 0,
                         "closed dry state (solids + gas); no aqueous phase stable at this RH", 0,
                         {gk[k]: float(g[k] * scale) for k in range(K)}, {gk[k]: float(P * q[k]) for k in range(K)})

    def _gas_matrix(self, gk):
        G = np.zeros((len(self.ions), len(gk)))
        for k, g in enumerate(gk):
            if g not in GASES:
                raise KeyError(f"unknown gas {g!r}; available: {sorted(GASES)}")
            for ion, nu in GASES[g].ions.items():
                if ion not in self.ions:
                    raise KeyError(f"gas {g} needs ion {ion!r}, which is not in the solver's ion set {self.ions}")
                G[self.ions.index(ion), k] = nu
        return G

    def _solve_vec(self, bfull, T_K, rh, gk, lnp, warm, tol, max_outer, max_newton, closed=None, start=None) -> SLEResult:
        if abs(float(np.dot(self.z, bfull))) > 1.0e-8 * max(1.0, float(np.sum(np.abs(bfull)))):
            raise ValueError("feed is not electroneutral")
        G = self._gas_matrix(gk)
        present = bfull != 0.0
        for k in range(len(gk)):
            present = present | (G[:, k] != 0.0)
        if self._carb and present[self.ions.index("CO3--")]:
            present[self.ions.index("H+")] = True       # proton excess is always a component of a carbonate solution
        if not np.any(present):
            raise ValueError("empty feed")
        ion_idx = np.flatnonzero(present)
        sol_idx = [j for j in range(len(self.solids)) if np.all(self.V[~present, j] == 0)]
        scale = float(np.abs(bfull).max())
        b = bfull[ion_idx] / scale
        V = np.hstack([self.V[np.ix_(ion_idx, sol_idx)], G[ion_idx]])
        lnk = np.array([self.solids[j].ln_k(T_K, self.mode) for j in sol_idx]
                       + [GASES[g].ln_k(T_K) + lp for g, lp in zip(gk, lnp)])
        hyd = np.concatenate([self.hyd[sol_idx], np.array([float(GASES[g].h) for g in gk])])
        self._closed = None if closed is None else (closed[0] / scale, closed[1])
        self._gcols = list(range(len(sol_idx), len(sol_idx) + len(gk)))
        self._gnames = list(gk)
        if start is not None:                        # initial point (reduced/scaled units) from a previous result
            u0 = np.array([start.solids.get(self.solids[j].key, 0.0) / scale for j in sol_idx]
                          + [start.gas.get(g, 0.0) / scale for g in gk])
            S0 = [k for k in range(len(sol_idx)) if u0[k] > 1.0e-14] + list(self._gcols)
            self._warm = ((tuple(ion_idx), tuple(sol_idx)), u0, S0)
        try:
            res = self._solve_reduced(b, V, lnk, hyd, ion_idx, sol_idx, T_K, math.log(rh), warm, tol, max_outer, max_newton)
        finally:
            closed_used = self._closed
            self._closed = None
        res.feed = {self.ions[i]: float(bfull[i]) for i in range(len(self.ions)) if bfull[i] != 0}
        res.solids = {k: v * scale for k, v in res.solids.items()}
        res.aq_ions = {k: v * scale for k, v in res.aq_ions.items()}
        res.gas = {g: v * scale for g, v in res.gas.items()}
        if closed is None:
            res.p_gas = {g: math.exp(lp) for g, lp in zip(gk, lnp)}
        else:
            ng = sum(res.gas.values())
            res.p_gas = {g: closed[1] * v / (closed[0] + ng) for g, v in res.gas.items()}
        res.water_kg *= scale
        res.gibbs *= scale
        return res

    def solve_closed(self, feed: dict[str, float], gas_total: dict[str, float], T_K: float, rh: float, *,
                     n_air: float, P_atm: float = 1.0, tol: float = 1.0e-9) -> SLEResult:
        """Closed system: particle phases + ideal gas phase of ``n_air`` mol of non-reacting air at total pressure ``P_atm``.

        ``feed`` are the (non-volatile or already-condensed) ion totals [mol]; ``gas_total`` the total amount of each
        volatile gas (keys of :data:`aiomfac_py.gases.GASES`) initially present [mol], e.g.
        ``{"HNO3": 2e-6, "NH3": 3e-6}``.  The gas amounts n_j are extra unknowns with the ideal-gas chemical potential
        ln p_j = ln(P n_j/(n_air + sum n)); a first convex problem without an aqueous phase (solids + gas, solved in
        element-potential space) is tested for aqueous-phase stability, and otherwise the wet problem is solved with the
        same active-set Newton method.  Totals of every ion / gas are conserved (``result.gas``, ``result.aq_ions``,
        ``result.solids``; ``result.p_gas`` in atm).
        """
        gk = list(gas_total)
        K = len(gk)
        G = self._gas_matrix(gk)
        bfull = self._vec(feed) + G @ np.array([gas_total[g] for g in gk], dtype=float)
        scale = float(np.abs(bfull).max())
        gt0 = np.array([gas_total[g] for g in gk], dtype=float)

        dry = self._closed_dry(bfull, gk, G, T_K, rh, n_air, P_atm, gt0)
        if dry is not None:
            dry.feed = {self.ions[i]: float(bfull[i]) for i in range(len(self.ions)) if bfull[i] != 0}
            return dry
        self._fast = True
        try:
            # warm start: the fixed-pressure problem at the ideal-gas pressures of the total gas amounts (reduced by
            # factors of 10 until the open solution leaves a positive amount in the gas phase)
            self._warm = None
            y = np.maximum(gt0, 1.0e-12 * scale) / (n_air + gt0.sum())
            for _ in range(12):
                r0 = self._solve_vec(bfull, T_K, rh, gk, np.log(P_atm * y), False, 1.0e-10, 80, 80)
                if r0.status != "failed" and all(r0.gas.get(g, 0.0) > 0.0 for g in gk):
                    break
                y = y * 0.1
            else:
                r0 = None
            res = self._solve_vec(bfull, T_K, rh, gk, np.zeros(K), r0 is not None, tol * 1.0e-2, 80, 80,
                                  closed=(n_air, P_atm), start=r0)
            if res.status == "failed":
                # reservoir >> particle (ill-conditioned gas variable): fixed-point on the gas mole fractions with
                # fixed-pressure (open) solves; converges when the uptake is small against the gas amounts
                fp = self._closed_fixed_point(bfull, gk, gt0, T_K, rh, n_air, P_atm, tol)
                if fp is not None:
                    res = fp
        finally:
            self._fast = False
        res.message = (res.message + " " if res.message else "") + "closed gas phase (ideal gas + air)"
        return res

    def _closed_fixed_point(self, bfull, gk, gt0, T_K, rh, n_air, P_atm, tol):
        """Closed-system fallback: iterate y_j = g_j/(n_air + sum g) with g_j = total_j + (open-solve gas amount_j; negative = uptake)."""
        y = gt0 / (n_air + gt0.sum())
        K = len(gk)
        res = None
        bfeed = bfull - self._gas_matrix(gk) @ gt0
        for _ in range(200):
            self._warm = None
            res = self._solve_vec(bfeed, T_K, rh, gk, np.log(P_atm * np.maximum(y, 1e-300)), False, 1.0e-10, 80, 80)
            if res.status == "failed":
                return None
            g = np.array([gt0[k] + res.gas.get(gk[k], 0.0) for k in range(K)])
            if np.any(g <= 0.0):
                return None
            ynew = g / (n_air + g.sum())
            if np.max(np.abs(ynew - y) / y) < max(tol, 1e-9):
                res.gas = {gk[k]: float(g[k]) for k in range(K)}
                res.p_gas = {gk[k]: float(P_atm * ynew[k]) for k in range(K)}
                res.message = (res.message + " " if res.message else "") + "[closed: fixed-point on gas pressures]"
                return res
            y = ynew
        return None

    # ------------------------------------------------------------------------------------------------
    def _solve_reduced(self, b, V, lnk, hyd, ion_idx, sol_idx, T, ln_rh, warm, tol, max_outer, max_newton) -> SLEResult:
        aq = self._aq_for(ion_idx)
        N, J = V.shape
        gc = list(self._gcols)
        Js = J - len(gc)
        gkeys = []
        ions_r = [self.ions[i] for i in ion_idx]
        keys_r = [self.solids[j].key for j in sol_idx]
        c = lnk - hyd * ln_rh
        ev0 = aq.n_eval
        rh = math.exp(ln_rh)

        def make(status, u, st, S, nn=0, no=0, msg=""):
            nsol = {k: float(v) for k, v in zip(keys_r, u[:Js]) if v > 1e-14}
            act = [keys_r[j] for j in S if j < Js]
            gas = {self._gnames[k]: float(u[Js + k]) for k in range(len(gc))}
            if st is None:
                return SLEResult(status, T, rh, ions_r, {}, nsol, {}, 0.0, {}, {}, {}, act, float(c @ u), nn, no, msg,
                                 aq.n_eval - ev0, gas)
            ln_a, w, _ = st
            n_aq = b - V @ u
            si = V.T @ ln_a - c
            return SLEResult(status, T, rh, ions_r, {}, nsol, {i: float(v) for i, v in zip(ions_r, n_aq)}, float(w),
                             {i: float(v / w) for i, v in zip(ions_r, n_aq)}, {i: float(v) for i, v in zip(ions_r, ln_a)},
                             {k: float(v) for k, v in zip(keys_r, si[:Js])}, act, float(np.sum(n_aq * ln_a) + c @ u), nn, no,
                             msg, aq.n_eval - ev0, gas)

        # ---- 0. no candidate solid for this feed (e.g. pure H2SO4 in water): single aqueous phase -----------------
        if J == 0:
            st = aq.solve_water(b, T, ln_rh)
            if st is None:
                return make("failed", np.zeros(0), None, [], msg="RH not reachable for the solid-free solution")
            return make("aqueous", np.zeros(0), st, [])
        # ---- 1. dry assemblage (LP) and stability of the aqueous phase -----------------------------------------
        u_dry = None
        if self._fast:                      # solve_closed: aqueous phase already known to be stable, wet branch only
            out = self._wet(aq, b, V, c, T, ln_rh, tol, max_outer, max_newton)
            if out is None:
                out = self._homotopy(aq, b, V, lnk, hyd, T, ln_rh, tol, max_outer, max_newton)
            if out is None:
                return make("failed", np.zeros(J), None, [], msg="wet branch failed")
            u, S, st, nn, no = out
            return make("solid+aqueous" if np.any(u[:Js] > 1e-14) else "aqueous", u, st, S, nn, no)
        lp = linprog(c, A_eq=V, b_eq=b, bounds=[(0, None)] * Js + [(None, None)] * len(gc), method="highs")
        if lp.status == 3 and gc:
            return make("failed", np.zeros(J), None, [],
                        msg="no equilibrium: the gas reservoir is supersaturated (solid growth is unbounded)")
        if lp.status == 0:
            u_dry = np.concatenate([np.maximum(lp.x[:Js], 0.0), lp.x[Js:]])
            y = self._refine_dual(aq, b, V, c, Js, np.asarray(lp.eqlin.marginals, dtype=float), float(lp.fun), T, ln_rh)
            if aq.free_h_row is not None:
                qf = b / max(float(np.abs(b).sum()), 1e-300)
            else:
                qf = np.maximum(b, 0.0) / max(float(b.sum()), 1e-300) if b.min() >= 0 else None
            f_min = self._tangent_plane(aq, y, T, ln_rh, q_feed=qf)
            sup = [j for j in range(Js) if u_dry[j] > 1e-12]
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
                sup = [j for j in range(Js) if u_dry[j] > 1e-12]
                return make("dry", u_dry, None, sup, msg="wet branch failed; dry LP assemblage returned")
            return make("failed", np.zeros(J), None, [], msg="no equilibrium found")
        u, S, st, nn, no = out
        self._warm = ((tuple(ion_idx), tuple(sol_idx)), u.copy(), list(S))
        status = "solid+aqueous" if np.any(u[:Js] > 1e-14) else "aqueous"
        return make(status, u, st, S, nn, no)

    # ------------------------------------------------------------------------------------------------
    @staticmethod
    def _refine_dual(aq, b, V, c, Js, y0, obj, T, ln_rh):
        """The dual solution of the dry LP is not unique when the dry assemblage does not fix all ion potentials (e.g.
        one solid in a 3-ion system).  Among the dual-optimal y (V_s^T y <= c_s, V_g^T y = c_g, b.y = obj) take the one
        closest (L1) to the ion activities of the metastable aqueous solution of the feed, so the tangent-plane test is
        not decided by an arbitrary vertex.  Falls back to ``y0``."""
        N = V.shape[0]
        try:
            okb = b if aq.free_h_row is None else np.delete(b, aq.free_h_row)
            if np.any(okb <= 0.0) or b.shape[0] != N:
                return y0
            r = aq.solve_water(b, T, ln_rh)
            if r is None:
                return y0
            ya = r[0]
            cost = np.concatenate([np.zeros(N), np.ones(N)])
            A_ub = [np.hstack([V[:, :Js].T, np.zeros((Js, N))]),
                    np.hstack([np.eye(N), -np.eye(N)]), np.hstack([-np.eye(N), -np.eye(N)])]
            b_ub = [c[:Js], ya, -ya]
            A_eq = [np.concatenate([b, np.zeros(N)])[None, :]]
            b_eq = [np.atleast_1d(float(obj))]
            if V.shape[1] > Js:
                A_eq.append(np.hstack([V[:, Js:].T, np.zeros((V.shape[1] - Js, N))]))
                b_eq.append(c[Js:])
            out = linprog(cost, A_ub=np.vstack(A_ub), b_ub=np.concatenate(b_ub), A_eq=np.vstack(A_eq),
                          b_eq=np.concatenate(b_eq), bounds=[(None, None)] * N + [(0, None)] * N, method="highs")
            if out.status == 0:
                return out.x[:N]
        except Exception:
            pass
        return y0

    # ------------------------------------------------------------------------------------------------
    @staticmethod
    def _interior(b, G, positive: bool = False, okrow=None):
        """Gas amounts u_g (free sign) that make every aqueous ion amount b - G u_g strictly positive."""
        K = G.shape[1]
        N = G.shape[0]
        cost = np.zeros(K + 1); cost[-1] = -1.0
        A = np.hstack([G, np.ones((N, 1))])
        if okrow is not None:
            A, b = A[okrow], b[okrow]
        bound = 20.0 * max(float(np.abs(b).max()), 1e-300)
        lo = 1.0e-4 * float(np.abs(b).max()) if positive else -bound
        out = linprog(cost, A_ub=A, b_ub=b, bounds=[(lo, bound)] * K + [(None, 1.0e-3 * float(np.abs(b).max()))],
                      method="highs")
        if out.status != 0 or out.x[-1] <= 1.0e-12 * float(np.abs(b).max()):
            return None
        return out.x[:K]

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
        fr = aq.free_h_row
        if fr is not None:                        # carbonate system: bases (cation + OH- = cation - H+) are neutral components too
            for cc in cats:
                if cc != fr:
                    v = np.zeros(aq.N); v[cc] = 1.0; v[fr] = -z[cc]
                    cols.append(v / np.abs(v).sum())
        P = np.array(cols).T
        npair = P.shape[1]
        guess = {"lnM": None}
        floor = np.ones(aq.N, dtype=bool)
        if fr is not None:
            floor[fr] = False

        def f_n(n):
            n = n / np.abs(n).sum()
            n = np.where(floor, np.maximum(n, 1.0e-9), n)
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
            scored.append((fv, q_feed / np.abs(q_feed).sum()))
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
        gc = list(self._gcols)
        isgas = np.zeros(J, dtype=bool); isgas[gc] = True
        okrow = np.ones(N, dtype=bool)                      # rows whose aqueous amount must stay positive
        if aq.free_h_row is not None:
            okrow[aq.free_h_row] = False                    # carbonate systems: H+ is a sign-free proton excess
        if u0 is None:
            u = np.zeros(J)
            if gc:
                ug = self._interior(b, V[:, gc], positive=self._closed is not None, okrow=okrow)
                if ug is None:
                    return None
                u[gc] = ug
        else:
            u = np.array(u0, dtype=float)
        S = list(gc) if S0 is None else list(dict.fromkeys(list(S0) + gc))
        lnM = [None]
        closed = self._closed

        def lnp_gas(uu):
            g = uu[gc]
            return np.log(closed[1] * g / (closed[0] + g.sum()))

        def si_of(uu, st_):
            si_ = V.T @ st_[0] - c
            if closed is not None and gc:
                si_ = si_.copy()
                si_[gc] -= lnp_gas(uu)
            return si_

        def st_at(uu):
            n = b - V @ uu
            if np.any(n[okrow] <= 0.0):
                return None
            r = aq.solve_water(n, T, ln_rh, lnM[0])
            if r is not None:
                lnM[0] = r[2]
            return r

        def phi_of(uu, st):
            val = float(np.sum((b - V @ uu) * st[0]) + c @ uu)
            if closed is not None and gc:
                g = uu[gc]
                Ng = closed[0] + g.sum()
                val += float(np.sum(g * np.log(closed[1] * g / Ng)) - closed[0] * math.log(Ng))
            return val

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
                si = si_of(u, st)
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
                    pos = (vk > 0) & okrow
                    eps = 1.0e-4 * bmax
                    if np.any(pos):
                        eps = min(eps, 0.1 * float(np.min(n_now[pos] / vk[pos])))
                    ngv = (vk < 0) & okrow
                    if np.any(ngv):
                        eps = min(eps, 0.1 * float(np.min(n_now[ngv] / -vk[ngv])))
                    if closed is not None and isgas[jk]:
                        eps = min(eps, 0.1 * float(u[jk]))
                    fh = aq.free_h_row
                    if fh is not None and vk[fh] != 0:        # keep the step small against the proton excess (pH jumps near it)
                        eps = min(eps, 0.1 * max(abs(float(n_now[fh])), 1.0e-6 * bmax) / abs(float(vk[fh])))
                    up, um = u.copy(), u.copy()
                    up[jk] += eps; um[jk] -= eps
                    sp, sm = st_at(up), st_at(um)
                    if sp is None or sm is None:
                        return None
                    H[:, kk] = -(si_of(up, sp)[S] - si_of(um, sm)[S]) / (2.0 * eps)
                H = 0.5 * (H + H.T)
                reg = 1.0e-12 * max(1.0, float(np.trace(np.abs(H))))
                try:
                    delta = np.linalg.solve(H + reg * np.eye(s), F)
                except np.linalg.LinAlgError:
                    delta = np.linalg.lstsq(H, F, rcond=None)[0]
                # step length: aqueous ions stay positive; solid amounts stay non-negative
                Vd = V[:, S] @ delta
                alpha = 1.0
                m_pos = (Vd > 0) & okrow
                if np.any(m_pos):
                    alpha = min(alpha, 0.9 * float(np.min(n_now[m_pos] / Vd[m_pos])))
                if closed is not None and gc:                         # closed gas amounts stay positive
                    for kk2, jk2 in enumerate(S):
                        if isgas[jk2] and delta[kk2] < 0:
                            alpha = min(alpha, 0.9 * float(u[jk2] / -delta[kk2]))
                block = None
                neg = np.flatnonzero(delta < 0)
                if neg.size:
                    ratios = -u[S][neg] / delta[neg]
                    ratios = np.where(isgas[np.array(S)[neg]], np.inf, ratios)       # gas amounts are free in sign
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
                    if sn is not None and aq.free_h_row is not None:
                        # carbonate speciation neglects the water consumed by CO2 + H2O <-> H+ + HCO3- and CO2(aq) in the
                        # mole fractions, so the Gibbs-Duhem identity (and with it the merit function) holds only to ~1e-3:
                        # accept a step that reduces the KKT residual instead
                        Fn = si_of(un, sn)[S]
                        if float(np.max(np.abs(Fn))) < (1.0 - 1.0e-4 * a) * float(np.max(np.abs(F))):
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
            zero = [j for j in S if u[j] <= 1.0e-14 and not isgas[j]]
            if zero:
                S = [j for j in S if j not in zero]
                continue
            si = si_of(u, st)
            inactive = [j for j in range(J) if j not in S and not isgas[j]]
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
        c = np.array([self.solids[j].ln_k(T_K, self.mode) - self.solids[j].h_eff * ln_rh for j in sol_idx])
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
    return float(np.sum(nu * ln_a) + s.h_eff * math.log(rh_eff) - s.ln_k(T_K, mode))


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
        ln_a_, ln_w = aq.ln_a_aw(m, T_K)
        return float(np.sum(nu * ln_a_) + s.h_eff * ln_w - ln_k), ln_w

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

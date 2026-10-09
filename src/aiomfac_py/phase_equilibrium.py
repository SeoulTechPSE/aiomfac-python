"""Combined liquid-liquid-solid phase equilibrium of organic-inorganic aerosol at fixed temperature and RH.

One Gibbs-energy minimization covers any number of liquid phases (each containing water, organics and ions) and all
candidate solid salts; the number of liquid phases is decided by an outer tangent-plane stability test.

Formulation
-----------
Species are water, the neutral organics and the individual ions (the "ion basis").  At fixed T and relative humidity
water is an open component (exchanged with the gas phase at a_w = RH), so the function minimized is the Legendre-
transformed Gibbs energy (in units of RT, with zero standard potentials for the ions, which cancel through the mass
balance)

    F = sum_alpha [ sum_i n_ai ln a_ai  -  n_aw ln RH ]  +  sum_s u_s ( ln K_s - h_s ln RH )

over liquid amounts n_ai > 0 and solid amounts u_s >= 0, subject to the linear constraints

    sum_alpha n_ai + sum_s nu_is u_s = b_i     for every non-water species i   (mass balance)
    sum_i z_i n_ai = 0                         for every liquid phase alpha     (electroneutrality)

``a_ai`` are AIOMFAC activities (neutrals: mole-fraction scale; ions: molal scale), ``K_s`` the solubility products
of :mod:`aiomfac_py.solids` (including the hydrate water through ``h_s``), ``nu`` the solid stoichiometries.  Because
each liquid's Gibbs energy is homogeneous of degree one, dF/dn_ai = ln a_ai (- ln RH for water); solids enter
linearly: with the dissolution  solid -> sum nu_i ion_i + h H2O  and  ln K = sum nu_i ln a_i + h ln a_w  at saturation,
the solid's potential is mu_s = sum nu_i mu0_i + h mu0_w + ln K, which with mu0_ions = 0 and the open water reservoir
gives the cost ln K_s - h_s ln RH per mole of solid.  At a stationary point
  * every liquid has a_w = RH;
  * every electroneutral combination of ions and every neutral has the same chemical potential in all liquids (single
    ion potentials may differ by z_i * psi_alpha, the phase potential);
  * every solid has saturation index SI_s = sum_i nu_is ln a_i + h_s ln RH - ln K_s <= 0, with u_s * SI_s = 0.

Solids are therefore handled *inside* the minimization (they are linear, constant-composition phases), not by an outer
loop: precipitation changes the ionic strength and with it the liquid-liquid split, and both adjust simultaneously.
This follows the structure of UHAERO's combined organic-inorganic solver (Amundson et al., 2007, ACP 7, 4675) with
the AIOMFAC activity model; see also :mod:`aiomfac_py.lle` (JOTA-2) and :mod:`aiomfac_py.sle` (JOTA-1).

Numerical method
----------------
Inner problem (fixed set of liquid phases): primal log-barrier method on the affine set of the linear constraints.
Starting from a strictly feasible point, Newton steps on  F - mu * sum(ln x)  are taken in the null space of the
constraint matrix (so the constraints hold exactly at every iterate), with a fraction-to-the-boundary rule and an
Armijo backtracking line search; mu is reduced geometrically.  Per-phase Hessians d ln a_i / d n_j are obtained by
central finite differences (AIOMFAC provides no analytic Jacobian).

Outer loop: after each inner solve, liquid phases with vanishing amount are removed and phases that converged to the
same composition are merged; then the tangent-plane distance
    TPD(w) = sum_i w_i ( ln a_i(w) - mu_i )        (mu_i: equilibrium potentials, mu_water = ln RH)
is minimized over electroneutral trial compositions w from several starting points (organic-rich, salt-rich, dilute).
A negative minimum adds a liquid phase at w; the loop ends when every trial has TPD >= -tol.

Modes
-----
``solids`` selects the candidate solids: ``"all"`` (equilibrium; every database solid whose ions are present),
``"none"`` (metastable liquids, crystallization suppressed) or an explicit list of solid keys.
:meth:`PhaseEquilibrium.drying_path` follows a decreasing-RH path on which a solid is admitted only once its
saturation index in the metastable liquid exceeds a critical supersaturation ``ln_s_crit`` and is kept afterwards.

Scope of this version
---------------------
Acid sulfate systems are supported in the same way as in :mod:`aiomfac_py.sle`: H+ and SO4-- are *stoichiometric*
(total) components, and the bisulfate equilibrium HSO4- <-> H+ + SO4-- (Knopf et al. 2003 constant, as in
:mod:`aiomfac_py.dissociation`) is solved inside every activity evaluation of every liquid; the potential of a
stoichiometric component equals that of the free ion at the speciation equilibrium, so the formulation above is
unchanged.  Pass H+ and SO4-- (never HSO4-).  Carbonate systems use CO3-- (total carbonate) and H+ (sign-free
proton excess), speciated into CO2(aq), HCO3-, CO3-- and OH- in every liquid.  NH3, HNO3, HCl and CO2 can be exchanged
with an open reservoir (``p_gas``) or a closed ideal-gas phase (``gas_total``, ``n_air``).  A state without any liquid
is reported with status "dry".  Design notes and validation: ``docs/phase_equilibrium.md``.  Like
:mod:`aiomfac_py.lle` and :mod:`aiomfac_py.sle`, this module is not part of the Fortran AIOMFAC code (which provides
activities only); every result carries its own equilibrium-condition checks (``PhaseEquilibriumResult.checks``).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy.linalg import null_space

from .io import Component
from .carbonate import gamma_co2_mr, ln_k1_hco3_at_t, ln_k2_hco3_at_t, ln_kw_at_t
from .dissociation import ln_k_hso4_at_t, solve_bisulfate
from .gases import GASES
from .model import ActivityModel
from .solids import ION_REGISTRY, SOLIDS, Solid

WATER = Component(1, "Water", ((16, 1),))
_UNSUPPORTED_IONS = ("HSO4-", "HCO3-", "OH-")


# ============================================================================================================
# Liquid-phase activities in the ion basis
# ============================================================================================================

class LiquidModel:
    """AIOMFAC activities of one liquid phase given species amounts: water, organics, individual ions."""

    def __init__(self, organics: Sequence[Component], ions: Sequence[str]):
        if "HSO4-" in ions:
            raise ValueError("pass the stoichiometric ions H+ and SO4-- instead of HSO4- (bisulfate is speciated internally)")
        bad = [i for i in ions if i in _UNSUPPORTED_IONS]
        if bad:
            raise ValueError(f"pass the stoichiometric ions H+ and CO3-- (total carbonate) instead of {bad}; "
                             "the carbonate species are speciated internally")
        if "CO3--" in ions and "H+" not in ions:
            raise ValueError("a carbonate system needs H+ as the (sign-free) proton-excess component")
        for i in ions:
            if i not in ION_REGISTRY:
                raise KeyError(f"unknown ion {i!r}")
        self.organics = list(organics)
        self.ions = list(ions)
        self.names = ["Water"] + [c.name for c in self.organics] + self.ions
        if len(set(self.names)) != len(self.names):
            raise ValueError("species names must be unique")
        self.n_neutral = 1 + len(self.organics)
        self.N = len(self.names)
        self.z = np.array([0.0] * self.n_neutral + [float(ION_REGISTRY[i][1]) for i in self.ions])

        comps = [WATER] + [Component(k + 2, c.name, tuple(c.subgroups)) for k, c in enumerate(self.organics)]
        self._carb = "CO3--" in self.ions
        self._acid = "H+" in self.ions and "SO4--" in self.ions and not self._carb
        model_ions = list(self.ions)
        if self._acid:
            model_ions += ["HSO4-"]
        if self._carb:
            model_ions += ["HCO3-", "OH-"] + (["HSO4-"] if "SO4--" in self.ions else [])
        cats = [i for i in model_ions if ION_REGISTRY[i][1] > 0]
        ans = [i for i in model_ions if ION_REGISTRY[i][1] < 0]
        if bool(cats) != bool(ans):
            raise ValueError("ions must include at least one cation and one anion (or none at all)")
        if cats:
            pairs = [(c, ans[0]) for c in cats] + [(cats[0], a) for a in ans[1:]]
            for c, a in pairs:
                zc, za = ION_REGISTRY[c][1], -ION_REGISTRY[a][1]
                g = math.gcd(zc, za)
                comps.append(Component(len(comps) + 1, c + a,
                                       ((ION_REGISTRY[c][0], za // g), (ION_REGISTRY[a][0], zc // g))))
        self.model = ActivityModel(comps, assume_complete=True)
        mx = self.model.mixture
        self._mx = mx
        self._nc = mx.sr.n_cation
        self._ngi = max(mx.ngi, 1)
        self._mm = np.asarray(mx.mmass[:self.n_neutral], dtype=float)       # kg/mol
        self._pos = []
        for ion in self.ions:
            sid, z = ION_REGISTRY[ion]
            self._pos.append((z > 0, mx.cat_index[sid] if z > 0 else mx.an_index[sid]))
        if self._acid:
            self._ih = mx.cat_index[ION_REGISTRY["H+"][0]]
            self._iso = mx.an_index[ION_REGISTRY["SO4--"][0]]
            self._ihs = mx.an_index[ION_REGISTRY["HSO4-"][0]]
        self._hso4_frac = None                       # warm start of the bisulfate speciation (HSO4- / its maximum)
        # components whose free-species fractions follow from the ratios of their amounts (internal speciation)
        self.speciated = np.zeros(self.N, dtype=bool)
        for ion in (("H+", "SO4--") if self._acid else ()) + (("CO3--", "SO4--") if self._carb else ()):
            if ion in self.ions:
                self.speciated[self.n_neutral + self.ions.index(ion)] = True
        # species whose amount may take either sign (barrier-free): the proton excess of a carbonate system
        self.free = np.zeros(self.N, dtype=bool)
        if self._carb:
            self._ih = mx.cat_index[ION_REGISTRY["H+"][0]]
            self._ico3 = mx.an_index[ION_REGISTRY["CO3--"][0]]
            self._ihco3 = mx.an_index[ION_REGISTRY["HCO3-"][0]]
            self._ioh = mx.an_index[ION_REGISTRY["OH-"][0]]
            self._has_so4 = "SO4--" in self.ions
            if self._has_so4:
                self._iso = mx.an_index[ION_REGISTRY["SO4--"][0]]
                self._ihs = mx.an_index[ION_REGISTRY["HSO4-"][0]]
            self._kh = self.n_neutral + self.ions.index("H+")
            self._kc = self.n_neutral + self.ions.index("CO3--")
            self._ks = self.n_neutral + self.ions.index("SO4--") if self._has_so4 else None
            self.free[self._kh] = True
            self._carb_cache = None
        self.n_eval = 0

    def ln_a(self, n: np.ndarray, T: float) -> np.ndarray:
        """ln a of every species for amounts ``n`` [mol] (species order ``names``); neutrals on the mole-fraction
        scale (AIOMFAC's dissociated-basis x), ions on the molal scale."""
        self.n_eval += 1
        n = np.asarray(n, dtype=float)
        nn = self.n_neutral
        nw = n[:nn]
        solv = float(np.dot(nw, self._mm))
        xn = nw / float(np.sum(nw))
        smc = np.zeros(self._ngi); sma = np.zeros(self._ngi)
        for (is_cat, idx), v in zip(self._pos, n[nn:]):
            (smc if is_cat else sma)[idx] = v / solv
        mod = self.model
        if self._carb:
            return self._ln_a_carb(n, T, xn, smc, sma, solv)
        terms = None
        if self._acid and smc[self._ih] > 0.0 and sma[self._iso] > 0.0:
            # stoichiometric H+ and SO4-- -> free H+, HSO4-, SO4-- at the bisulfate equilibrium of this phase
            terms = self._speciate_hso4(xn, smc, sma, T)
            if terms is None:                                    # fallback: the bracketing solver of dissociation.py
                r = solve_bisulfate(mod, T, xn, smc, sma, self._ih, self._ihs, self._iso)
                smc[self._ih], sma[self._ihs], sma[self._iso] = r.m_h, r.m_hso4, r.m_so4
        if terms is None:
            x = mod._x_from_molalities(xn, smc, sma)
            lr, mr, sr = mod.lr_mr_sr(T, smc, sma, xn, x)
        else:
            x, lr, mr, sr = terms
        self._last_free = (smc.copy(), sma.copy(), 0.0, solv)   # free-species molalities (start of ExplicitLiquidModel)
        out = np.empty(self.N)
        with np.errstate(divide="ignore"):                       # a species absent from the phase has ln a = -inf
            out[:nn] = lr.ln_gamma_neutral[:nn] + mr.ln_gamma_neutral[:nn] + sr.ln_gamma_sr[:nn] + np.log(x[:nn])
        nc = self._nc
        lg_ = lambda v: math.log(v) if v > 0.0 else -math.inf
        for k, (is_cat, idx) in enumerate(self._pos):
            if is_cat:
                lg = mr.ln_gamma_cation[idx] + sr.ln_gamma_sr[nn + idx] + lr.ln_gamma_cation[idx] - mr.tmolal
                out[nn + k] = lg + lg_(smc[idx])
            else:
                lg = mr.ln_gamma_anion[idx] + sr.ln_gamma_sr[nn + nc + idx] + lr.ln_gamma_anion[idx] - mr.tmolal
                out[nn + k] = lg + lg_(sma[idx])
        return out

    def _speciate_hso4(self, xn, smc, sma, T):
        """Bisulfate equilibrium HSO4- <=> H+ + SO4-- for the total H+ and SO4-- molalities in ``smc``/``sma``.

        Same condition as :func:`aiomfac_py.dissociation.solve_bisulfate` (ln g_H + ln g_SO4 - ln g_HSO4 +
        ln(m_H m_SO4/m_HSO4) = ln K), solved as in :meth:`aiomfac_py.sle.AqueousIons._speciate`: at a fixed activity-
        coefficient ratio q the HSO4- molality is the root of a quadratic; q is updated by fixed-point iteration,
        accelerated by secant steps and warm-started from the last solution (as a fraction of the maximum HSO4-).
        Typically 2-5 activity evaluations instead of ~20 for the bracketing solver.  Fills ``smc``/``sma`` with the
        speciated molalities and returns (x, lr, mr, sr) at that state, or None (caller falls back) if the iteration
        does not converge."""
        mod, nn, nc = self.model, self.n_neutral, self._nc
        ih, ihs, iso = self._ih, self._ihs, self._iso
        mh, ms = float(smc[ih]), float(sma[iso])
        K = math.exp(ln_k_hso4_at_t(T))
        xmax = min(mh, ms)
        lo_b, hi_b = 1.0e-14 * xmax, xmax * (1.0 - 1.0e-14)
        frac = self._hso4_frac if self._hso4_frac is not None else 0.5
        xh = min(max(frac * xmax, lo_b), hi_b)
        x_prev = f_prev = None
        for it in range(60):
            smc[ih] = mh - xh
            sma[ihs] = xh
            sma[iso] = ms - xh
            x = mod._x_from_molalities(xn, smc, sma)
            lr, mr, sr = mod.lr_mr_sr(T, smc, sma, xn, x)
            lg_h = mr.ln_gamma_cation[ih] + sr.ln_gamma_sr[nn + ih] + lr.ln_gamma_cation[ih]
            lg_hs = mr.ln_gamma_anion[ihs] + sr.ln_gamma_sr[nn + nc + ihs] + lr.ln_gamma_anion[ihs]
            lg_so = mr.ln_gamma_anion[iso] + sr.ln_gamma_sr[nn + nc + iso] + lr.ln_gamma_anion[iso]
            e = lg_h + lg_so - lg_hs - mr.tmolal                 # net molal conversion: one -tmolal
            if not math.isfinite(e):
                break
            q = K / math.exp(max(min(e, 50.0), -50.0))
            s_ = mh + ms + q
            g = 2.0 * mh * ms / (s_ + math.sqrt(max(s_ * s_ - 4.0 * mh * ms, 0.0)))   # smaller root, cancellation-free
            g = min(max(g, lo_b), hi_b)
            f = g - xh
            if abs(f) <= 1.0e-13 * xmax:
                self._hso4_frac = xh / xmax
                return x, lr, mr, sr                             # terms evaluated at the converged state xh
            if x_prev is not None and it >= 2 and abs(f - f_prev) > 0.0:
                xs = xh - f * (xh - x_prev) / (f - f_prev)
                xn_ = xs if lo_b < xs < hi_b else g
            else:
                xn_ = g
            x_prev, f_prev = xh, f
            xh = xn_
        self._hso4_frac = None
        smc[ih], sma[ihs], sma[iso] = mh, 0.0, ms                # restore the totals for the fallback solver
        return None

    def _gp(self, mr, sr, lr, is_cat: bool, idx: int) -> float:
        nn, nc = self.n_neutral, self._nc
        if is_cat:
            return mr.ln_gamma_cation[idx] + sr.ln_gamma_sr[nn + idx] + lr.ln_gamma_cation[idx] - mr.tmolal
        return mr.ln_gamma_anion[idx] + sr.ln_gamma_sr[nn + nc + idx] + lr.ln_gamma_anion[idx] - mr.tmolal

    def _ln_a_carb(self, n, T, xn, smc, sma, solv):
        """Carbonate (and bisulfate) speciation of one liquid, generalizing :meth:`aiomfac_py.sle.AqueousIons.
        _speciate_carb` to phases that contain organics.  Stoichiometric components: C_T = CO2 + HCO3- + CO3--
        (carried as ``CO3--``) and the sign-free proton excess P = H+ - OH- + HCO3- + 2 CO2 (+ HSO4-) (carried as
        ``H+``); K1 = a_H a_HCO3/(a_CO2 a_w), K2 = a_H a_CO3/a_HCO3, Kw = a_H a_OH/a_w with the constants of
        :mod:`aiomfac_py.carbonate`; gamma(CO2) from the salting-out coefficients.  The potential of each
        stoichiometric component equals that of its free species (H+, CO3--, SO4--) at the speciation equilibrium."""
        mod, nn = self.model, self.n_neutral
        P, CT = n[self._kh] / solv, n[self._kc] / solv
        ST = n[self._ks] / solv if self._ks is not None else 0.0
        K1, K2, Kw = math.exp(ln_k1_hco3_at_t(T)), math.exp(ln_k2_hco3_at_t(T)), math.exp(ln_kw_at_t(T))
        K3 = math.exp(ln_k_hso4_at_t(T)) if ST > 0.0 else 0.0
        cache = self._carb_cache
        if cache is not None and not all(math.isfinite(v) and v > 0 for v in cache[0][:3]):
            cache = None                                         # never warm-start from a failed evaluation
        A1, A2, Aw, q = (K1, K2, Kw, K3) if cache is None else cache[0]
        if ST > 0.0 and not (math.isfinite(q) and q > 0):
            q = K3
        lnh0 = None if cache is None else cache[1]
        from scipy.optimize import brentq
        def fractions(lnh):
            """CO2(aq), HCO3-, CO3-- fractions of C_T at ln m_H (log-sum-exp: no overflow or underflow at extreme pH)."""
            l1 = math.log(A1) - lnh
            l2 = l1 + math.log(A2) - lnh
            m_ = max(0.0, l1, l2)
            lse = m_ + math.log(math.exp(-m_) + math.exp(l1 - m_) + math.exp(l2 - m_))
            return math.exp(-lse), math.exp(l1 - lse), math.exp(l2 - lse)

        for _ in range(60):
            def f(lnh):
                h = math.exp(lnh)
                a0, a1, _a2 = fractions(lnh)
                # [OH-] capped at exp(700): far beyond any physical value, f is then hugely negative but finite
                val = h - math.exp(min(math.log(Aw) - lnh, 700.0)) + CT * (a1 + 2.0 * a0) - P
                if ST > 0.0:
                    val += ST * h / (h + q)
                return val
            lo, hi = math.log(1.0e-40), math.log(1.0e4)
            if lnh0 is not None and f(lnh0 - 1.0) < 0.0 < f(lnh0 + 1.0):
                lo, hi = lnh0 - 1.0, lnh0 + 1.0
            else:                                   # f rises monotonically in ln h: widen until it brackets the root
                while f(hi) <= 0.0 and hi < math.log(1.0e30):
                    hi += math.log(100.0)
                while f(lo) >= 0.0 and lo > math.log(1.0e-290):
                    lo = max(lo - math.log(1.0e10), math.log(1.0e-300))
            lnh = brentq(f, lo, hi, xtol=1.0e-14, rtol=1.0e-13)
            h = math.exp(lnh)
            a0, a1, a2 = fractions(lnh)
            co2, hco3, co3 = CT * a0, CT * a1, CT * a2
            oh = math.exp(min(math.log(Aw) - lnh, 700.0))
            hso4 = ST * h / (h + q) if ST > 0.0 else 0.0
            smc[self._ih] = h
            sma[self._ico3], sma[self._ihco3], sma[self._ioh] = co3, hco3, oh
            if self._ks is not None:
                sma[self._iso], sma[self._ihs] = ST - hso4, hso4
            x = mod._x_from_molalities(xn, smc, sma)
            lr, mr, sr = mod.lr_mr_sr(T, smc, sma, xn, x)
            lgH, lgHC = self._gp(mr, sr, lr, True, self._ih), self._gp(mr, sr, lr, False, self._ihco3)
            lgC, lgOH = self._gp(mr, sr, lr, False, self._ico3), self._gp(mr, sr, lr, False, self._ioh)
            lgCO2 = gamma_co2_mr(self._mx, smc, sma)
            lgW = mr.ln_gamma_neutral[0] + sr.ln_gamma_sr[0] + lr.ln_gamma_neutral[0]
            xw = float(x[0])
            clip = lambda v: min(max(v, -700.0), 700.0)          # keep the fixed-point ratios finite
            nA1 = math.exp(clip(math.log(xw) + math.log(K1) - (lgH + lgHC - lgCO2 - lgW)))
            nA2 = math.exp(clip(math.log(K2) - (lgH + lgC - lgHC)))
            nAw = math.exp(clip(math.log(xw) + math.log(Kw) - (lgH + lgOH - lgW)))
            nq = q
            if ST > 0.0:
                nc = self._nc
                lg_h = mr.ln_gamma_cation[self._ih] + sr.ln_gamma_sr[nn + self._ih] + lr.ln_gamma_cation[self._ih]
                lg_hs = mr.ln_gamma_anion[self._ihs] + sr.ln_gamma_sr[nn + nc + self._ihs] + lr.ln_gamma_anion[self._ihs]
                lg_so = mr.ln_gamma_anion[self._iso] + sr.ln_gamma_sr[nn + nc + self._iso] + lr.ln_gamma_anion[self._iso]
                nq = K3 / math.exp(max(min(lg_h + lg_so - lg_hs - mr.tmolal, 50.0), -50.0))
            err = max(abs(math.log(nA1 / A1)), abs(math.log(nA2 / A2)), abs(math.log(nAw / Aw)),
                      abs(math.log(nq / q)) if ST > 0.0 else 0.0)
            A1, A2, Aw, q = nA1, nA2, nAw, nq
            lnh0 = lnh
            if err < 1.0e-11:
                break
        if all(math.isfinite(v) and v > 0 for v in (A1, A2, Aw)) and math.isfinite(lnh):
            self._carb_cache = ((A1, A2, Aw, q), lnh)
        self._last_free = (smc.copy(), sma.copy(), float(co2), solv)
        out = np.empty(self.N)
        out[:nn] = lr.ln_gamma_neutral[:nn] + mr.ln_gamma_neutral[:nn] + sr.ln_gamma_sr[:nn] + np.log(x[:nn])
        for k, (is_cat, idx) in enumerate(self._pos):
            i = nn + k
            if i == self._kh:
                m = h
            elif i == self._kc:
                m = co3
            elif self._ks is not None and i == self._ks:
                m = ST - hso4
            else:
                m = (smc if is_cat else sma)[idx]
            out[i] = self._gp(mr, sr, lr, is_cat, idx) + math.log(max(m, 1.0e-300))
        return out

    # ---------------------------------------------------------------------------------------------------
    # ideal part of ln a and its exact Jacobian (used by the split Hessian)
    # ---------------------------------------------------------------------------------------------------
    def ln_ideal(self, n: np.ndarray, act: np.ndarray | None = None) -> np.ndarray:
        """Ideal part of ln a: ln n_i - ln(sum of the bounded amounts) for the neutrals, ln n_i - ln(solvent mass) for the
        ions, 0 for the sign-free proton excess and for absent entries.  ``ln a - ln_ideal`` is a smooth function of
        the composition (the activity coefficients, plus the speciation correction of acid and carbonate systems):
        all 1/n curvature of trace species is in the ideal part."""
        n = np.asarray(n, dtype=float)
        act = np.ones(self.N, dtype=bool) if act is None else act
        nn = self.n_neutral
        bnd = act & ~self.free
        tot = float(np.sum(n[bnd]))
        solv = float(np.dot(n[:nn], self._mm))
        out = np.zeros(self.N)
        k = bnd.copy()
        with np.errstate(divide="ignore"):
            out[k] = np.log(n[k])
        out[:nn][k[:nn]] -= math.log(tot)
        out[nn:][k[nn:]] -= math.log(solv)
        return out

    def ideal_jacobian(self, n: np.ndarray, act: np.ndarray | None = None) -> np.ndarray:
        """Exact d ln_ideal_i / d n_j (see :meth:`ln_ideal`)."""
        n = np.asarray(n, dtype=float)
        act = np.ones(self.N, dtype=bool) if act is None else act
        nn = self.n_neutral
        bnd = act & ~self.free
        tot = float(np.sum(n[bnd]))
        solv = float(np.dot(n[:nn], self._mm))
        J = np.zeros((self.N, self.N))
        idx = np.flatnonzero(bnd)
        J[idx, idx] = 1.0 / n[idx]
        rn = idx[idx < nn]
        ri = idx[idx >= nn]
        J[np.ix_(rn, idx)] -= 1.0 / tot
        cn = np.flatnonzero(act[:nn])
        J[np.ix_(ri, cn)] -= self._mm[cn] / solv
        return J

    def hessian_excess(self, n: np.ndarray, T: float, active: np.ndarray | None = None, la0: np.ndarray | None = None,
                       rel: float = 1.0e-6) -> np.ndarray:
        """d(ln a - ln_ideal)/dn by forward differences (N activity evaluations; ``la0`` = ln a(n) if already known).
        For a species that is not speciated the function differenced is smooth on the scale of the phase, so the step
        is a fraction ``rel`` of the phase size (trace species included).  The speciated components (H+, SO4-- of an
        acid system; CO3--, SO4-- of a carbonate system) are different: their speciation depends on the RATIOS of
        their amounts, which may all be traces in an organic-rich liquid, so their step is ``rel`` times their own
        amount.  The sign-free proton excess keeps the central difference of :meth:`hessian` (its ln a varies steeply
        around neutrality)."""
        n = np.asarray(n, dtype=float)
        act = np.ones(self.N, dtype=bool) if active is None else np.asarray(active, dtype=bool)
        S = float(np.sum(np.abs(n[act])))
        la0 = self.ln_a(n, T) if la0 is None else la0
        r0 = la0 - self.ln_ideal(n, act)
        R = np.zeros((self.N, self.N))
        for j in range(self.N):
            if not act[j]:
                continue
            with np.errstate(invalid="ignore", divide="ignore"):
                if self.free[j]:
                    h = 1.0e-5 * max(abs(n[j]), 1.0e-10 * S)
                    p = n.copy(); p[j] += h
                    m = n.copy(); m[j] -= h
                    col = ((self.ln_a(p, T) - self.ln_ideal(p, act)) - (self.ln_a(m, T) - self.ln_ideal(m, act))) / (2 * h)
                else:
                    h = rel * (abs(n[j]) if self.speciated[j] else S)
                    p = n.copy(); p[j] += h
                    col = ((self.ln_a(p, T) - self.ln_ideal(p, act)) - r0) / h
            R[act, j] = col[act]
        return R

    def hessian_split(self, n: np.ndarray, T: float, active: np.ndarray | None = None, la0: np.ndarray | None = None,
                      R: np.ndarray | None = None) -> np.ndarray:
        """Hessian d ln a_i/d n_j = exact ideal Jacobian + excess part (``R``: a previously computed
        :meth:`hessian_excess`, reused as long as the composition has not changed much), symmetrized."""
        act = np.ones(self.N, dtype=bool) if active is None else np.asarray(active, dtype=bool)
        if R is None:
            R = self.hessian_excess(n, T, act, la0)
        H = self.ideal_jacobian(n, act) + R
        return 0.5 * (H + H.T)

    def hessian(self, n: np.ndarray, T: float, rel: float = 1.0e-5, active: np.ndarray | None = None) -> np.ndarray:
        """d ln a_i / d n_j by central differences, symmetrized.  Rows and columns of species that are absent from the
        phase (``active`` False, amount 0) are zero."""
        n = np.asarray(n, dtype=float)
        act = np.ones(self.N, dtype=bool) if active is None else np.asarray(active, dtype=bool)
        H = np.zeros((self.N, self.N))
        floor = 1.0e-10 * float(np.sum(np.abs(n)))
        for j in range(self.N):
            if not act[j]:
                continue
            # bounded species: a step proportional to the amount keeps the perturbed state positive; only the
            # sign-free proton excess (carbonate) needs an absolute floor
            h = rel * max(abs(n[j]), floor) if self.free[j] else rel * abs(n[j])
            p = n.copy(); p[j] += h
            m = n.copy(); m[j] -= h
            with np.errstate(invalid="ignore"):                  # absent rows: -inf - -inf, discarded
                H[act, j] = ((self.ln_a(p, T) - self.ln_a(m, T)) / (2.0 * h))[act]
        return 0.5 * (H + H.T)



class ExplicitLiquidModel(LiquidModel):
    """Liquid phase with the products of the bisulfate and carbonate equilibria as explicit species.

    The components (feed, mass balance, solids, gases) are those of :class:`LiquidModel`: water, organics and the given
    ions, with H+ and SO4-- as totals in acid systems and CO3-- (total carbonate) and H+ (proton excess) in carbonate
    systems.  The species of a liquid are the components plus HSO4- (if H+ and SO4-- are present) and HCO3-, OH- and
    CO2(aq) (carbonate systems), appended after the components so that every component keeps its index.  ``E`` maps
    species to components (HSO4- = H+ + SO4--, HCO3- = H+ + CO3--, CO2 = 2 H+ + CO3-- - H2O, OH- = H2O - H+; water is
    open and has no balance).

    ``ln_a`` returns the transformed chemical potential mu_s = c_s + ln a_s of every species, with the reaction
    constants in c (water at its reservoir potential ln RH, set by :meth:`set_conditions`):

        c(HSO4-) = ln K_HSO4,  c(HCO3-) = ln K2,  c(CO2) = ln K1 + ln K2 + ln RH,  c(OH-) = -(ln Kw + ln RH),

    so that minimizing F = sum n mu - n_w ln RH under the component balances puts every reaction at equilibrium
    (mu_HSO4 = mu_H + mu_SO4, mu_HCO3 = mu_H + mu_CO3, mu_CO2 = 2 mu_H + mu_CO3, mu_OH = -mu_H).  Each activity
    evaluation is a single AIOMFAC call: no speciation iteration is nested in it.  CO2(aq) is a molal solute with
    AIOMFAC's salting-out activity coefficient (:func:`aiomfac_py.carbonate.gamma_co2_mr`); it does not enter the
    activities of the other species, which leaves the ~1 % Gibbs-Duhem inconsistency of that treatment."""

    def __init__(self, organics: Sequence[Component], ions: Sequence[str]):
        if "HSO4-" in ions:
            raise ValueError("pass the stoichiometric ions H+ and SO4-- instead of HSO4- (bisulfate is speciated internally)")
        bad = [i for i in ions if i in _UNSUPPORTED_IONS]
        if bad:
            raise ValueError(f"pass the stoichiometric ions H+ and CO3-- (total carbonate) instead of {bad}; "
                             "the carbonate species are speciated internally")
        if "CO3--" in ions and "H+" not in ions:
            raise ValueError("a carbonate system needs H+ as the (sign-free) proton-excess component")
        for i in ions:
            if i not in ION_REGISTRY:
                raise KeyError(f"unknown ion {i!r}")
        self._int = LiquidModel(organics, ions)                 # internal speciation: initial species distribution
        self.organics = list(organics)
        self.ions = list(ions)
        self.comp_names = ["Water"] + [c.name for c in self.organics] + self.ions
        self.carbonate = "CO3--" in self.ions
        self.acid = "H+" in self.ions and "SO4--" in self.ions
        extras = (["HSO4-"] if self.acid else []) + (["HCO3-", "OH-"] if self.carbonate else [])
        self.sp_ions = self.ions + extras
        self.names = self.comp_names + extras + (["CO2(aq)"] if self.carbonate else [])
        self.n_neutral = 1 + len(self.organics)
        self.N = len(self.names)
        self.z = np.array([0.0] * self.n_neutral + [float(ION_REGISTRY[i][1]) for i in self.sp_ions]
                          + ([0.0] if self.carbonate else []))
        Nc = len(self.comp_names)
        E = np.zeros((Nc, self.N))
        E[:, :Nc] = np.eye(Nc)
        col = lambda name: self.names.index(name)
        comp = lambda name: self.comp_names.index(name)
        if self.acid:
            E[comp("H+"), col("HSO4-")] = 1.0; E[comp("SO4--"), col("HSO4-")] = 1.0
        if self.carbonate:
            E[comp("H+"), col("HCO3-")] = 1.0; E[comp("CO3--"), col("HCO3-")] = 1.0
            E[comp("H+"), col("OH-")] = -1.0
            E[comp("H+"), col("CO2(aq)")] = 2.0; E[comp("CO3--"), col("CO2(aq)")] = 1.0
        self.E = E
        comps = [WATER] + [Component(k + 2, c.name, tuple(c.subgroups)) for k, c in enumerate(self.organics)]
        cats = [i for i in self.sp_ions if ION_REGISTRY[i][1] > 0]
        ans = [i for i in self.sp_ions if ION_REGISTRY[i][1] < 0]
        if bool(cats) != bool(ans):
            raise ValueError("ions must include at least one cation and one anion (or none at all)")
        if cats:
            pairs = [(c, ans[0]) for c in cats] + [(cats[0], a) for a in ans[1:]]
            for c, a in pairs:
                zc, za = ION_REGISTRY[c][1], -ION_REGISTRY[a][1]
                g = math.gcd(zc, za)
                comps.append(Component(len(comps) + 1, c + a,
                                       ((ION_REGISTRY[c][0], za // g), (ION_REGISTRY[a][0], zc // g))))
        self.model = ActivityModel(comps, assume_complete=True)
        mx = self.model.mixture
        self._mx = mx
        self._nc = mx.sr.n_cation
        self._ngi = max(mx.ngi, 1)
        self._mm = np.asarray(mx.mmass[:self.n_neutral], dtype=float)
        self._pos = []
        for ion in self.sp_ions:
            sid, zz = ION_REGISTRY[ion]
            self._pos.append((zz > 0, mx.cat_index[sid] if zz > 0 else mx.an_index[sid]))
        self._kco2 = self.N - 1 if self.carbonate else None
        self._acid = self._carb = False                          # no speciation nested in ln_a
        self.speciated = np.zeros(self.N, dtype=bool)
        self.free = np.zeros(self.N, dtype=bool)
        self.n_eval = 0
        self.n_jac = 0                                           # AD Jacobian evaluations (hess_scheme "ad")
        self._ad = None
        self._T = None
        self.set_conditions(298.15, 0.0)

    def set_conditions(self, T: float, ln_rh: float):
        """Reaction constants c_s at temperature T and water potential ln RH."""
        c = np.zeros(self.N)
        if self.acid:
            c[self.names.index("HSO4-")] = ln_k_hso4_at_t(T)
        if self.carbonate:
            k1, k2, kw = ln_k1_hco3_at_t(T), ln_k2_hco3_at_t(T), ln_kw_at_t(T)
            c[self.names.index("HCO3-")] = k2
            c[self.names.index("OH-")] = -(kw + ln_rh)
            c[self._kco2] = k1 + k2 + ln_rh
        self._T, self._ln_rh, self._c = float(T), float(ln_rh), c

    def ln_a(self, n: np.ndarray, T: float) -> np.ndarray:
        """Transformed chemical potential c_s + ln a_s of every species (neutrals on AIOMFAC's mole-fraction scale,
        ions and CO2(aq) on the molal scale)."""
        self.n_eval += 1
        if self._T != float(T):
            self.set_conditions(T, self._ln_rh)
        n = np.asarray(n, dtype=float)
        nn = self.n_neutral
        nw = n[:nn]
        solv = float(np.dot(nw, self._mm))
        xn = nw / float(np.sum(nw))
        smc = np.zeros(self._ngi); sma = np.zeros(self._ngi)
        for (is_cat, idx), v in zip(self._pos, n[nn:nn + len(self.sp_ions)]):
            (smc if is_cat else sma)[idx] = v / solv
        mod = self.model
        x = mod._x_from_molalities(xn, smc, sma)
        lr, mr, sr = mod.lr_mr_sr(T, smc, sma, xn, x)
        out = np.empty(self.N)
        with np.errstate(divide="ignore"):
            out[:nn] = lr.ln_gamma_neutral[:nn] + mr.ln_gamma_neutral[:nn] + sr.ln_gamma_sr[:nn] + np.log(x[:nn])
        lg_ = lambda v: math.log(v) if v > 0.0 else -math.inf
        for k, (is_cat, idx) in enumerate(self._pos):
            out[nn + k] = self._gp(mr, sr, lr, is_cat, idx) + lg_((smc if is_cat else sma)[idx])
        if self.carbonate:
            out[self._kco2] = gamma_co2_mr(self._mx, smc, sma) + lg_(n[self._kco2] / solv)
        return out + self._c

    def ln_a_batch(self, n: np.ndarray, T: float) -> np.ndarray:
        """:meth:`ln_a` for many compositions at one temperature: ``n`` of shape (B, N) -> (B, N).

        Uses the JAX transcription of :mod:`aiomfac_py.ad_activity` (optional ``jax``), compiled once per system,
        temperature and batch size and vectorized over the batch; it reproduces :meth:`ln_a` to round-off.  For dense
        evaluation (maps, sensitivity analyses, many states) this removes the per-call overhead of the NumPy code:
        well below 1 microsecond per composition for small systems, against about 0.1-0.4 ms per :meth:`ln_a` call."""
        from .ad_activity import batch_for
        if self._T != float(T):
            self.set_conditions(T, self._ln_rh)
        n = np.atleast_2d(np.asarray(n, dtype=float))
        self.n_eval += len(n)
        return np.asarray(batch_for(self, T)(n)) + self._c

    def hessian_ad(self, n: np.ndarray, T: float, active: np.ndarray | None = None) -> np.ndarray:
        """Exact d ln a_i/d n_j by automatic differentiation (:mod:`aiomfac_py.ad_activity`, needs jax), symmetrized;
        rows and columns of absent species are zero.  The jit-compiled Jacobian is built once per system and temperature."""
        n = np.asarray(n, dtype=float)
        act = np.ones(self.N, dtype=bool) if active is None else np.asarray(active, dtype=bool)
        if self._ad is None or self._ad[0] != float(T):
            from .ad_activity import jacobian_for
            self._ad = (float(T), jacobian_for(self, T))
        self.n_jac += 1
        J = np.zeros((self.N, self.N))
        idx = np.flatnonzero(act)
        J[np.ix_(idx, idx)] = np.asarray(self._ad[1](n))[np.ix_(idx, idx)]
        return 0.5 * (J + J.T)

    def initial_species(self, nc: np.ndarray, T: float, floor: float = 1.0e-20) -> np.ndarray:
        """Species amounts for component amounts ``nc`` (water included) from the internal speciation of
        :class:`LiquidModel` (a proton excess of either sign is allowed); species that come out zero get ``floor``
        times the phase size (the caller restores the balances)."""
        lm = self._int
        lm.ln_a(nc, T)
        smc, sma, m_co2, solv = lm._last_free
        mx = lm._mx
        out = np.zeros(self.N)
        nn = self.n_neutral
        out[:nn] = nc[:nn]
        for k, ion in enumerate(self.sp_ions):
            sid, zz = ION_REGISTRY[ion]
            m = smc[mx.cat_index[sid]] if zz > 0 else sma[mx.an_index[sid]]
            out[nn + k] = m * solv
        if self.carbonate:
            out[self._kco2] = m_co2 * solv
        size = float(np.sum(np.abs(nc)))
        return np.maximum(out, floor * size)


# ============================================================================================================
# Results
# ============================================================================================================

@dataclass
class LiquidPhase:
    amounts: np.ndarray            # mol of each component (water, organics, ions as totals)
    ln_a: np.ndarray               # ln a of each component (= that of its free species at equilibrium)
    names: list
    species_names: list | None = None          # explicit speciation: free species (HSO4-, HCO3-, OH-, CO2(aq), ...)
    species_amounts: np.ndarray | None = None
    species_ln_a: np.ndarray | None = None     # transformed potentials c + ln a of the species

    @property
    def total(self) -> float:
        return float(np.sum(self.amounts))

    @property
    def mole_fractions(self) -> np.ndarray:
        return self.amounts / self.total

    @property
    def water_activity(self) -> float:
        return math.exp(self.ln_a[0])

    def as_dict(self) -> dict:
        return {k: float(v) for k, v in zip(self.names, self.amounts)}


@dataclass
class PhaseEquilibriumResult:
    status: str                    # "converged" | "not_converged"
    T_K: float
    rh: float
    liquids: list                  # list[LiquidPhase], sorted by decreasing organic mole fraction
    solids: dict                   # solid key -> mol (only solids above the presence threshold)
    si: dict                       # saturation index of every candidate solid at the solution
    gibbs: float                   # transformed Gibbs energy F (RT units, internal scaling removed)
    checks: dict                   # equilibrium-condition residuals
    tpd_min: float                 # smallest tangent-plane distance found in the final stability test
    n_outer: int = 0
    message: str = ""
    names: list = field(default_factory=list)
    gas: dict = field(default_factory=dict)     # mol in the gas phase (closed) or net mol released, < 0 = uptake (open)
    p_gas: dict = field(default_factory=dict)   # partial pressures [atm]

    @property
    def n_liquids(self) -> int:
        return len(self.liquids)

    def summary(self) -> str:
        lines = [f"{self.status}: T={self.T_K} K, RH={self.rh:.4f}, {self.n_liquids} liquid phase(s), "
                 f"solids={ {k: round(v, 8) for k, v in self.solids.items()} }"]
        if self.gas:
            lines.append(f"  gas: { {k: float(f'{v:.4g}') for k, v in self.gas.items()} } mol, "
                         f"p = { {k: float(f'{v:.4g}') for k, v in self.p_gas.items()} } atm")
        for k, L in enumerate(self.liquids):
            xs = ", ".join(f"{n}={v:.4g}" for n, v in zip(L.names, L.mole_fractions))
            lines.append(f"  liquid {k}: n_tot={L.total:.6g} mol, a_w={L.water_activity:.5f}; x: {xs}")
        lines.append(f"  checks: { {k: float(f'{v:.3g}') for k, v in self.checks.items()} }, TPD_min={self.tpd_min:.3g}")
        if self.message:
            lines.append(f"  message: {self.message}")
        return "\n".join(lines)


# ============================================================================================================
# Solver
# ============================================================================================================

class PhaseEquilibrium:
    """Liquid-liquid-solid equilibrium of water + organics + ions at fixed T and RH.

    >>> pe = PhaseEquilibrium([pinic_acid_component], ["NH4+", "SO4--"], T_K=298.15)
    >>> res = pe.solve({"pinic_acid": 0.01, "NH4+": 0.02, "SO4--": 0.01}, rh=0.4)
    >>> print(res.summary())
    """

    def __init__(self, organics: Sequence[Component], ions: Sequence[str], T_K: float, *,
                 k_mode: str | None = None, solid_keys: Sequence[str] | None = None, speciation: str = "explicit",
                 liquid_model: "ExplicitLiquidModel | None" = None):
        """``speciation``: "explicit" (HSO4-, HCO3-, OH-, CO2(aq) as species of the liquids, reactions at
        equilibrium through the minimization; :class:`ExplicitLiquidModel`) or "internal" (stoichiometric
        components speciated inside every activity evaluation; :class:`LiquidModel`, the original formulation).

        ``liquid_model``: a liquid model of the same organics and ions to use instead of AIOMFAC, e.g. a
        :class:`aiomfac_py.gibbs_model.GibbsLiquidModel` (excess-Gibbs-energy surrogate; its exact Hessians are used,
        ``hess_scheme = "ad"``).  Child problems with fewer species get ``liquid_model.restrict(...)``."""
        if speciation not in ("explicit", "internal"):
            raise ValueError("speciation must be 'explicit' or 'internal'")
        self.speciation = speciation
        if liquid_model is not None:
            if speciation != "explicit" or not isinstance(liquid_model, ExplicitLiquidModel):
                raise ValueError("liquid_model must be an ExplicitLiquidModel (explicit speciation)")
            if [c.name for c in liquid_model.organics] != [c.name for c in organics] or list(liquid_model.ions) != list(ions):
                raise ValueError("liquid_model is built for other organics or ions")
            self.lm = liquid_model
        else:
            self.lm = ExplicitLiquidModel(organics, ions) if speciation == "explicit" else LiquidModel(organics, ions)
        self._organics, self._ions, self._solid_keys = list(organics), list(ions), solid_keys
        self._children: dict = {}
        self.T = float(T_K)
        # components (feed, mass balance, solids, gases, results) and species (the variables of a liquid); the
        # species list starts with the components, so a component's index is also that of its base species
        self.names = list(getattr(self.lm, "comp_names", self.lm.names))
        self.sp_names = list(self.lm.names)
        self.Nc = len(self.names)
        self.N = self.lm.N
        self.z = self.lm.z
        self.zc = self.z[:self.Nc]
        self.E = getattr(self.lm, "E", np.eye(self.Nc))
        # species taking part in a speciation reaction (explicit model): never removed as traces, because their small
        # amounts carry the reaction equilibria (e.g. the free H+ of a liquid where most acid is HSO4-, i.e. its pH)
        self._reactive = np.zeros(self.N, dtype=bool)
        if self.N > self.Nc:
            self._reactive[self.Nc:] = True
            self._reactive[:self.Nc] = np.any(self.E[:, self.Nc:] != 0.0, axis=1)
        self._carbonate = "CO3--" in self._ions
        # sign-free component: the proton excess of a carbonate system
        self._comp_free = np.zeros(self.Nc, dtype=bool)
        if self._carbonate:
            self._comp_free[self.names.index("H+")] = True
        if solid_keys is None:
            # hydroxide solids carry OH- as -H+ and are meaningful only when H+ is the proton excess (carbonate system)
            cand = [s for s in SOLIDS.values() if set(s.ions) <= set(ions)
                    and (not getattr(s, "n_oh", 0) or self._carbonate)
                    and not any(i in _UNSUPPORTED_IONS for i in s.ions)]
        else:
            cand = [SOLIDS[k] for k in solid_keys]
        self.all_solids: list[Solid] = [s for s in cand if s.kspec is None or True]
        self.k_mode = k_mode
        self._lnk = {s.key: s.ln_k(self.T, k_mode) for s in self.all_solids}
        self.tol_tpd = 1.0e-7
        self.seed_fractions = (0.5, 0.2, 0.05)              # size of a new liquid's seed (fraction of the most possible)
        # first seed: "linesearch" = minimize the Gibbs energy along the transfer direction (seed_fractions only for the
        # retries), "fixed" = seed_fractions throughout (the previous rule)
        self.seed_method = "linesearch"
        # Hessian of the inner Newton iteration: "split" = exact ideal Jacobian + forward-difference excess part, the
        # latter reused while no liquid amount has changed by more than hess_reuse_tol of the liquid's size;
        # "central" = central differences of ln a at every step (the original scheme); "ad" = exact Jacobian by
        # automatic differentiation at every step (explicit speciation with jax installed; "split" otherwise)
        self.hess_scheme = "ad" if getattr(self.lm, "has_exact_hessian", False) else "split"
        self.hess_reuse_tol = 0.02
        # the stability test starts far from its minima: its excess Hessian is refreshed after much smaller changes
        self.tpd_hess_reuse_tol = 0.02
        # stability test: "ss" = successive substitution (one activity evaluation per iteration; Newton polish only if
        # it does not converge), "newton" = barrier Newton from every start (the original method)
        self.tpd_method = "ss"
        # stop scanning the trial starts once a liquid with TPD below -tpd_early_stop is found (None: all starts)
        self.tpd_early_stop = 1.0e-3
        # inner problem: "newton" = Newton without a barrier, solids by an active set; "barrier" = log-barrier method
        # with mu continuation for liquids and solids (the original method); "rand" = logarithmic amounts with
        # element-potential feasibility restoration: no barrier and no trace removal (explicit speciation only;
        # "newton" otherwise)
        self.inner_method = "newton"
        self.si_tol = 1.0e-9                         # an inactive solid is added once its SI exceeds this
        self.newton_mu0, self.newton_mu_factor, self.newton_mu_min = 1.0e-4, 0.01, 1.0e-14   # liquid barrier: 6 stages
        self.trace_tol = 1.0e-9                     # scaled amount below which a liquid entry is removed
        self._trunc_max = 0.0
        self._ls_failures = 0

    # ---------------------------------------------------------------------------------------------------
    def _use_ad(self) -> bool:
        """Exact AD Hessians requested and possible (explicit liquid model, jax installed)."""
        if self.hess_scheme != "ad" or not isinstance(self.lm, ExplicitLiquidModel):
            return False
        if getattr(self.lm, "has_exact_hessian", False):      # Gibbs-function models (gibbs_model)
            return True
        from . import ad_activity
        return ad_activity.AVAILABLE

    def _solid_columns(self, solids: list[Solid]) -> np.ndarray:
        return self._columns(solids)

    def _solid_cost(self, solids: list[Solid], ln_rh: float) -> np.ndarray:
        return np.array([self._lnk[s.key] - s.h_eff * ln_rh for s in solids])

    def _si_liquids(self, liquids, ln_rh) -> dict:
        """Saturation indices from the liquids: the largest finite value over the liquids (all equal at equilibrium;
        a liquid from which an ion of the solid is absent gives -inf)."""
        out = {}
        for L in liquids:
            for k, v in self.si_of(L.ln_a, ln_rh).items():
                if not np.isnan(v):
                    out[k] = max(out.get(k, -math.inf), v)
        return out

    def si_of(self, ln_a: np.ndarray, ln_rh: float, solids: list[Solid] | None = None) -> dict:
        """Saturation index of each solid for the activities of one liquid (gauge-free: solids are neutral)."""
        solids = self.all_solids if solids is None else solids
        out = {}
        for s in solids:
            v = sum(nu * ln_a[self.names.index(ion)] for ion, nu in s.ions.items()) + s.h_eff * ln_rh - self._lnk[s.key]
            out[s.key] = float(v)
        return out

    # ---------------------------------------------------------------------------------------------------
    # Inner problem: barrier Newton on the affine set
    # ---------------------------------------------------------------------------------------------------
    def _columns(self, items) -> np.ndarray:
        """Ion columns of solids or gases: entry nu_i = amount of species i removed from the liquids per mole."""
        V = np.zeros((self.Nc, len(items)))
        for j, s in enumerate(items):
            for ion, nu in s.ions.items():
                V[self.names.index(ion), j] = nu
        return V

    def _build_A(self, n_liq: int):
        N = self.N
        W = self._W                                              # solid + gas columns (component space)
        nx = n_liq * N + W.shape[1]
        rows = []
        for i in range(1, self.Nc):                              # mass balance of non-water components
            r = np.zeros(nx)
            for a in range(n_liq):
                r[a * N:(a + 1) * N] = self.E[i]
            r[n_liq * N:] = W[i]
            rows.append(r)
        if np.any(self.z != 0):
            for a in range(n_liq):                                # electroneutrality of each liquid
                r = np.zeros(nx)
                r[a * N:(a + 1) * N] = self.z
                rows.append(r)
        return np.array(rows)

    def _free_mask(self, n_liq: int) -> np.ndarray:
        S, K = len(self._solids_now), len(self._gases_now)
        liq = np.tile(self.lm.free, n_liq)
        return np.concatenate([liq, np.zeros(S, dtype=bool), np.full(K, self._gas_mode == "open", dtype=bool)])

    def _gas_terms(self, g):
        """Objective, gradient and Hessian of the gas block (closed: ideal-gas mixing with n_air of inert air)."""
        cg = self._gas_cost
        if self._gas_mode != "closed" or len(g) == 0:
            return float(np.dot(g, cg)), cg.copy(), np.zeros((len(g), len(g)))
        ntot = self._n_air + float(np.sum(g))
        ln_y = np.log(g / ntot)
        F = float(np.dot(g, cg + ln_y)) + self._n_air * math.log(self._n_air / ntot)
        H = np.diag(1.0 / g) - 1.0 / ntot
        return F, cg + ln_y, H

    def _objective(self, x, n_liq, ln_rh, act=None):
        N, S = self.N, len(self._solids_now)
        F = 0.0
        lna = []
        for a in range(n_liq):
            n = x[a * N:(a + 1) * N]
            la = self.lm.ln_a(n, self.T)
            lna.append(la)
            ma = np.ones(N, dtype=bool) if act is None else act[a * N:(a + 1) * N]
            F += float(np.dot(n[ma], la[ma])) - n[0] * ln_rh          # absent species (n = 0) contribute nothing
        u = x[n_liq * N:n_liq * N + S]
        g = x[n_liq * N + S:]
        F += float(np.dot(u, self._solid_cost_now))
        Fg, gg, Hg = self._gas_terms(g)
        return F + Fg, lna, gg, Hg

    def _grad(self, lna, ln_rh, gg):
        g = []
        for la in lna:
            v = la.copy(); v[0] -= ln_rh
            g.append(v)
        g.append(self._solid_cost_now)
        g.append(gg)
        return np.concatenate(g)

    # ---------------------------------------------------------------------------------------------------
    # Species that are absent from one liquid (trace truncation)
    # ---------------------------------------------------------------------------------------------------
    def _restore_constraints(self, x, act, A, target, n_liq):
        """Weighted least-change correction of the active entries so that A x = target again (after amounts were moved
        between liquids); the weights are the amounts, so relative changes stay small.  A correction larger than a
        trace amount would flip its sign: such liquid entries are removed and the correction is repeated."""
        nl = n_liq * self.N
        bounded_liq = np.zeros(len(x), dtype=bool)
        bounded_liq[:nl] = np.tile(~self.lm.free, n_liq)
        for _ in range(5):
            r = A @ x - target
            if not np.any(np.abs(r) > 0.0):
                return
            Aa = A[:, act]
            w = np.abs(x[act]) + 1e-300
            M = (Aa * w) @ Aa.T
            y = np.linalg.lstsq(M, r, rcond=None)[0]
            x[act] -= w * (Aa.T @ y)
            bad = act & bounded_liq & (x <= 0.0)
            if not np.any(bad):
                return
            x[bad] = 0.0
            act[bad] = False

    def _truncate(self, x, n_liq, act, A) -> bool:
        """Remove trace entries: species i in liquid a is set to zero and excluded (act False) when its amount is below
        ``trace_tol`` (scaled units, sum |b| = 1) and another liquid holds at least 10 times more of it.  The amount is
        moved to that liquid and the linear constraints are restored.  Such an entry is the barrier's approximation of
        an amount that is negligible for the mass balance; its potential has no influence on the other species, but
        its 1/n curvature and its contribution to F, far below the floating-point resolution of F, stall the Newton
        iteration (line search) and leave the potential conditions unmet."""
        N, tol = self.N, self.trace_tol
        lm = self.lm
        target = A @ x
        changed = False
        for i in range(1, N):
            if lm.free[i] or (lm._carb and i == lm._kc) or self._reactive[i]:
                continue
            idx = [a * N + i for a in range(n_liq) if act[a * N + i]]
            if len(idx) < 2:
                continue
            keep = max(idx, key=lambda k: x[k])
            for k in idx:
                if k != keep and x[k] < tol and x[keep] >= 10.0 * x[k]:
                    x[keep] += x[k]
                    self._trunc_max = max(self._trunc_max, float(x[k]))
                    x[k] = 0.0
                    act[k] = False
                    changed = True
        if changed:
            self._restore_constraints(x, act, A, target, n_liq)
        return changed

    def _vanish(self, x, n_liq, act, A, tiny=1.0e-20) -> bool:
        """Newton solver: a species whose amount falls below ``tiny`` (scaled units) in every liquid that holds it, while
        an active gas or solid can take it up, is removed from all liquids (complete evaporation or precipitation of
        that ion).  Without a barrier such an amount would otherwise shrink geometrically until 1/n overflows.  The
        mass-balance row of the species then fixes the gas (or solid) amount; the constraints are restored."""
        N, lm = self.N, self.lm
        changed = False
        target = A @ x
        tail = np.flatnonzero(act[n_liq * N:]) + n_liq * N
        for i in range(1, N):
            if lm.free[i] or (lm._carb and i == lm._kc) or self._reactive[i]:
                continue
            idx = [a * N + i for a in range(n_liq) if act[a * N + i]]
            if not idx or max(x[k] for k in idx) >= tiny:
                continue
            if i < self.Nc:                                      # base species: its component must go somewhere
                row = i - 1                                      # mass-balance row of component i in A
                if not np.any(A[row, tail] != 0.0):
                    continue                                     # nothing can take it up: leave it
            for k in idx:
                self._trunc_max = max(self._trunc_max, float(x[k]))
                x[k] = 0.0
                act[k] = False
            changed = True
        if changed:
            self._restore_constraints(x, act, A, target, n_liq)
        return changed

    def _reenter(self, x, n_liq, act, A) -> bool:
        """Undo trace removals that are no longer justified: for every absent liquid entry, estimate its equilibrium
        amount from the potential of that species in the other liquids (neutrals: ln x = mu - ln gamma; ions:
        ln m = lambda + z psi_a - ln gamma, from the gauge fit) with the activity coefficient at infinite dilution;
        entries whose estimate exceeds 10 trace_tol are given that amount back (from the liquid holding most of the
        species) and the constraints are restored.  A species removed while it was transiently small (e.g. DLT in a
        liquid that later took up organic) would otherwise stay absent.  Returns True if an entry was restored."""
        N, lm, nn = self.N, self.lm, self.lm.n_neutral
        absent = [k for k in np.flatnonzero(~act[:n_liq * N]) if not lm.free[k % N] and k % N != 0]
        if not absent:
            return False
        phases = [x[a * N:(a + 1) * N] for a in range(n_liq)]
        masks = [act[a * N:(a + 1) * N] for a in range(n_liq)]
        with np.errstate(all="ignore"):
            lnas = [lm.ln_a(phases[a], self.T) for a in range(n_liq)]
        liqs = [LiquidPhase(phases[a].copy(), lnas[a], self.sp_names) for a in range(n_liq)]
        lam, psi = self._ion_gauge_fit(liqs) if N > nn else (None, None)
        target = A @ x
        changed = False
        for k in absent:
            a, i = divmod(k, N)
            if i < nn:
                vals = [lnas[b][i] for b in range(n_liq) if masks[b][i] and np.isfinite(lnas[b][i])]
                if not vals:
                    continue
                mu_i = max(vals)
            else:
                mu_i = lam[i - nn] + self.z[i] * psi[a]
            n = phases[a].copy()
            ma = masks[a].copy(); ma[i] = True
            tot = float(np.sum(n[ma & ~lm.free]))
            delta = 1.0e-12 * tot
            n[i] = delta
            try:
                with np.errstate(all="ignore"):
                    r_i = float(lm.ln_a(n, self.T)[i] - lm.ln_ideal(n, ma)[i])
            except (ValueError, OverflowError, FloatingPointError, ZeroDivisionError):
                continue
            if not np.isfinite(r_i):
                continue
            if i < nn:
                n_eq = math.exp(min(mu_i - r_i, 0.0)) * tot
            else:
                solv = float(np.dot(phases[a][:nn], lm._mm))
                n_eq = math.exp(min(mu_i - r_i, 50.0)) * solv
            if n_eq <= 10.0 * self.trace_tol:
                continue
            holders = [b * N + i for b in range(n_liq) if act[b * N + i]]
            if not holders:
                continue
            h = max(holders, key=lambda kk: x[kk])
            d = min(n_eq, 0.5 * x[h])
            x[h] -= d
            x[k] = d
            act[k] = True
            changed = True
        if changed:
            self._restore_constraints(x, act, A, target, n_liq)
        return changed

    def _reactivate(self, x, n_liq, act, A):
        """Give every absent entry back a trace amount (from the liquid holding most of that species) before a new liquid
        is seeded, so that all liquids again contain all species; truncation is repeated by the next inner solve."""
        N, tol = self.N, self.trace_tol
        target = A @ x
        changed = False
        for k in np.flatnonzero(~act[:n_liq * N]):
            i = k % N
            holders = [a * N + i for a in range(n_liq) if act[a * N + i]]
            if not holders:
                continue
            h = max(holders, key=lambda kk: x[kk])
            d = min(tol, 0.01 * x[h])
            if d <= 0.0:
                continue
            x[h] -= d
            x[k] = d
            act[k] = True
            changed = True
        if changed:
            self._restore_constraints(x, act, A, target, n_liq)

    def _barrier_solve(self, x, n_liq, ln_rh, *, act=None, mu0=1.0e-3, mu_min=1.0e-14, max_newton=60,
                       verbose=False):
        """Minimize F - mu sum ln x_b over {A x = A x0} (x_b: bounded active entries) from the strictly feasible x.

        A step is accepted when it decreases the barrier function (Armijo) or, failing that, the norm of the reduced
        gradient: AIOMFAC's carbonate treatment (CO2(aq) with its own salting-out activity coefficient) makes the
        stoichiometric potentials only approximately the gradient of one Gibbs function, and the equilibrium
        conditions themselves are what the final checks verify.

        ``act`` (modified in place) marks the entries that take part; trace entries are removed after every barrier
        stage (:meth:`_truncate`), which repeats the stage."""
        N, S = self.N, len(self._solids_now)
        x = np.array(x, dtype=float)
        if act is None:
            act = np.ones(len(x), dtype=bool)
        A = self._build_A(n_liq)
        free = self._free_mask(n_liq)

        def setup():
            Z = np.zeros((len(x), 0))
            if A.size:
                Za = null_space(A[:, act])
                Z = np.zeros((len(x), Za.shape[1]))
                Z[act] = Za
            else:
                Z = np.eye(len(x))[:, act]
            return Z, ~free & act

        Z, bnd = setup()
        mu = mu0
        converged = True

        def raw(xx):
            """F, ln a of every liquid, gas gradient and Hessian at xx (independent of mu, so reusable across stages)."""
            try:
                with np.errstate(all="ignore"):
                    return self._objective(xx, n_liq, ln_rh, act)
            except (ValueError, OverflowError, FloatingPointError, ZeroDivisionError):
                return None

        def barrier(xx, rv, mu_):
            if rv is None:
                return float("inf"), np.full(len(xx), np.nan)
            F, lna, gg, _ = rv
            gfull = self._grad(lna, ln_rh, gg)
            gfull[~act] = 0.0
            gfull[bnd] -= mu_ / xx[bnd]
            return F - mu_ * float(np.sum(np.log(xx[bnd]))), gfull

        # excess part of each liquid's Hessian, reused while the liquid's composition changes little
        # (the exact ideal part, which carries the 1/n curvature, is recomputed at every step)
        hcache: list = [None] * n_liq
        # carbonate systems: the stoichiometric potentials are only approximately a gradient (Sect. 3.3 of the design
        # notes), steps are often accepted on the reduced-gradient norm, and that needs a current Jacobian
        reuse_tol = 0.0 if self.lm._carb else self.hess_reuse_tol

        def liquid_hessian(a, rv):
            sl = slice(a * N, (a + 1) * N)
            n, ma = x[sl], act[sl]
            if self.hess_scheme == "central":
                return self.lm.hessian(n, self.T, active=ma)
            if self._use_ad():
                return self.lm.hessian_ad(n, self.T, ma)
            c = hcache[a]
            S = float(np.sum(np.abs(n[ma])))
            stale = (c is None or not np.array_equal(c[1], ma)
                     or float(np.max(np.abs(n - c[0]))) > reuse_tol * S)
            if stale:
                R = self.lm.hessian_excess(n, self.T, ma, la0=None if rv is None else rv[1][a])
                hcache[a] = (n.copy(), ma.copy(), R)
            return self.lm.hessian_split(n, self.T, ma, R=hcache[a][2])

        rv = raw(x)
        while True:
            for it in range(max_newton):
                phi0, gvec = barrier(x, rv, mu)
                Hg = rv[3] if rv is not None else np.zeros((0, 0))
                H = np.zeros((len(x), len(x)))
                for a in range(n_liq):
                    H[a * N:(a + 1) * N, a * N:(a + 1) * N] = liquid_hessian(a, rv)
                o = n_liq * N + S
                H[o:, o:] += Hg
                dg = np.zeros(len(x)); dg[bnd] = mu / x[bnd] ** 2
                H[np.diag_indices_from(H)] += dg
                Hr = Z.T @ H @ Z
                gr = Z.T @ gvec
                w, Q = np.linalg.eigh(0.5 * (Hr + Hr.T))
                shift = 0.0
                big = max(1.0, abs(w[-1]))
                if w[0] <= 1.0e-12 * big:
                    shift = -w[0] + 1.0e-8 * big
                d = -Q @ ((Q.T @ gr) / (w + shift))
                dx = Z @ d
                dec = float(-gr @ d)
                r0 = float(np.linalg.norm(gr))
                # converged when the decrement is small AND the Newton step is small relative to every bounded variable
                # (trace species have curvature ~1/n: a small decrement alone can hide a large error in their ln a)
                rel = float(np.max(np.abs(dx[bnd]) / x[bnd])) if np.any(bnd) else 0.0
                if dec < 1.0e-14 + 1.0e-10 * mu * len(x) and rel < 1.0e-8:
                    break
                neg = (dx < 0) & bnd
                amax = min(1.0, 0.995 * float(np.min(-x[neg] / dx[neg]))) if np.any(neg) else 1.0
                step, ok = amax, False
                for _ in range(40):
                    xt = x + step * dx
                    if np.all(xt[bnd] > 0):
                        rvt = raw(xt)
                        phit, gt = barrier(xt, rvt, mu)
                        # Armijo, or a decrease of the reduced-gradient norm: the second criterion is needed close to
                        # convergence, where the decrease of F falls below its numerical resolution (the speciation is
                        # solved iteratively), and in carbonate systems (potentials only approximately a gradient)
                        if np.isfinite(phit) and np.all(np.isfinite(gt)) and (phit <= phi0 - 1.0e-4 * step * dec
                                                  or float(np.linalg.norm(Z.T @ gt)) <= (1.0 - 1.0e-4 * step) * r0):
                            ok = True
                            break
                    step *= 0.5
                if not ok:
                    # a failed line search counts as converged only when the decrement is tiny in absolute terms:
                    # trace species have curvature ~1/n, so a moderate decrement can hide a large potential error
                    if dec < 1.0e-14:
                        break
                    converged = False
                    self._ls_failures += 1
                    break
                x, rv = xt, rvt
            else:
                converged = False
            if verbose:
                print(f"    mu={mu:.1e} it={it} dec={dec:.2e}")
            if n_liq > 1 and self._truncate(x, n_liq, act, A):
                Z, bnd = setup()
                rv = raw(x)
                # repeat this stage without the trace entries; failures before the removal stay counted in
                # self._ls_failures (reported in checks), the returned flag describes the stages after it
                converged = True
                continue
            if mu <= mu_min:
                break
            mu = max(mu * 0.1, mu_min)
        return x, converged

    # ---------------------------------------------------------------------------------------------------
    # Starting point, stability test, phase bookkeeping
    # ---------------------------------------------------------------------------------------------------
    def _newton_solve(self, x, n_liq, ln_rh, *, act=None, max_newton=60, verbose=False):
        """Inner problem with solids by an active set (``inner_method = "newton"``).

        * Solids enter F linearly and are handled by an active set, as in SLESolver: only active solids are variables;
          a step that would make an active solid negative is cut where it reaches zero and the solid is dropped; once a
          barrier stage has converged, the most supersaturated inactive solid (SI > si_tol) is added at zero amount and
          the stage is repeated.  At convergence every active solid has SI = 0 and every inactive one SI <= si_tol.
        * Liquid (and closed-gas) amounts keep a log-barrier with mu continuation from ``newton_mu0`` down to 1e-14 by
          factors of ``newton_mu_factor``.  A pure Newton iteration without a barrier (mu0 = 0) was tried: it stalls or
          takes huge steps in non-convex regions (a third liquid being seeded, a liquid disappearing), where the barrier
          curvature mu/x^2 regularizes the step.
        * Trace entries are removed (:meth:`_truncate`) and ions that left every liquid through a gas or solid are
          removed (:meth:`_vanish`) after every step; a liquid that disappears as a whole ends the inner solve.
        ``act`` is modified in place.  Returns (x, converged)."""
        N, S = self.N, len(self._solids_now)
        x = np.array(x, dtype=float)
        if act is None:
            act = np.ones(len(x), dtype=bool)
        A = self._build_A(n_liq)
        free = self._free_mask(n_liq)
        o_s = n_liq * N
        is_solid = np.zeros(len(x), dtype=bool); is_solid[o_s:o_s + S] = True
        converged = True

        def setup():
            if A.size:
                Za = null_space(A[:, act])
                Z = np.zeros((len(x), Za.shape[1]))
                Z[act] = Za
            else:
                Z = np.eye(len(x))[:, act]
            return Z, ~free & act & ~is_solid, act & is_solid

        def raw(xx):
            try:
                with np.errstate(all="ignore"):
                    return self._objective(xx, n_liq, ln_rh, act)
            except (ValueError, OverflowError, FloatingPointError, ZeroDivisionError):
                return None

        def merit(xx, rv, mu_):
            if rv is None:
                return float("inf"), np.full(len(xx), np.nan)
            F, lna, gg, _ = rv
            g = self._grad(lna, ln_rh, gg)
            g[~act] = 0.0
            if mu_ > 0:
                g[pos] -= mu_ / xx[pos]
                F = F - mu_ * float(np.sum(np.log(xx[pos])))
            return F, g

        hcache: list = [None] * n_liq
        reuse_tol = 0.0 if self.lm._carb else self.hess_reuse_tol

        def liquid_hessian(a, rv):
            sl = slice(a * N, (a + 1) * N)
            n, ma = x[sl], act[sl]
            if self.hess_scheme == "central":
                return self.lm.hessian(n, self.T, active=ma)
            if self._use_ad():
                return self.lm.hessian_ad(n, self.T, ma)
            c = hcache[a]
            Ssz = float(np.sum(np.abs(n[ma])))
            if (c is None or not np.array_equal(c[1], ma) or float(np.max(np.abs(n - c[0]))) > reuse_tol * Ssz):
                R = self.lm.hessian_excess(n, self.T, ma, la0=None if rv is None else rv[1][a])
                hcache[a] = (n.copy(), ma.copy(), R)
            return self.lm.hessian_split(n, self.T, ma, R=hcache[a][2])

        refused: set = set()

        def supersaturated(rv):
            best = (-math.inf, None)
            if rv is None:
                return best
            for j, sld in enumerate(self._solids_now):
                k = o_s + j
                if act[k] or k in refused:
                    continue
                vals = [self.si_of(la, ln_rh, [sld])[sld.key] for la in rv[1]]
                si = max((v for v in vals if np.isfinite(v)), default=-math.inf)
                if si > best[0]:
                    best = (si, k)
            return best

        Z, pos, sol = setup()
        rv = raw(x)
        mu = self.newton_mu0
        changes = 0
        reentries = 0
        dec_prev = math.inf
        while True:
            for it in range(max_newton):
                F0, gvec = merit(x, rv, mu)
                Hg = rv[3] if rv is not None else np.zeros((0, 0))
                H = np.zeros((len(x), len(x)))
                for a in range(n_liq):
                    H[a * N:(a + 1) * N, a * N:(a + 1) * N] = liquid_hessian(a, rv)
                og = o_s + S
                H[og:, og:] += Hg
                if mu > 0:
                    dg = np.zeros(len(x)); dg[pos] = mu / x[pos] ** 2
                    H[np.diag_indices_from(H)] += dg
                Hr = Z.T @ H @ Z
                gr = Z.T @ gvec
                if Hr.size == 0 or not np.all(np.isfinite(Hr)):
                    break
                w, Q = np.linalg.eigh(0.5 * (Hr + Hr.T))
                big = max(1.0, abs(w[-1]))
                shift = -w[0] + 1.0e-8 * big if w[0] <= 1.0e-12 * big else 0.0
                d = -Q @ ((Q.T @ gr) / (w + shift))
                dx = Z @ d
                dec = float(-gr @ d)
                r0 = float(np.linalg.norm(gr))
                rel = float(np.max(np.abs(dx[pos]) / x[pos])) if np.any(pos) else 0.0
                if dec < 1.0e-14 + 1.0e-10 * mu * len(x) and rel < 1.0e-8:
                    break
                # a reused excess Hessian can stall the iteration (its refresh test is absolute, so large relative
                # changes of trace ions do not trigger it): refresh it whenever the decrement stops falling
                if dec > 0.9 * dec_prev:
                    hcache[:] = [None] * n_liq
                dec_prev = dec
                amax, hit = 1.0, None
                neg = (dx < 0) & pos
                if np.any(neg):
                    amax = min(amax, 0.995 * float(np.min(-x[neg] / dx[neg])))
                negs = (dx < 0) & sol
                if np.any(negs):
                    ratios = -x[negs] / dx[negs]
                    j = int(np.argmin(ratios))
                    if ratios[j] < amax:
                        amax, hit = float(ratios[j]), int(np.flatnonzero(negs)[j])
                if hit is not None and amax <= 1.0e-14:
                    if x[hit] == 0.0:
                        refused.add(hit)                         # added at zero but driven negative at once
                    act[hit] = False; x[hit] = 0.0; changes += 1
                    Z, pos, sol = setup(); continue
                step, ok = amax, False
                for _ in range(40):
                    xt = x + step * dx
                    if hit is not None and step == amax:
                        xt[hit] = 0.0
                    if np.all(xt[pos] > 0) and np.all(xt[sol] >= 0):
                        rvt = raw(xt)
                        Ft, gt = merit(xt, rvt, mu)
                        if np.isfinite(Ft) and np.all(np.isfinite(gt)) and (Ft <= F0 - 1.0e-4 * step * dec
                                                  or float(np.linalg.norm(Z.T @ gt)) <= (1.0 - 1.0e-4 * step) * r0):
                            ok = True
                            break
                    step *= 0.5
                if not ok:
                    if dec < 1.0e-14:
                        break
                    converged = False
                    self._ls_failures += 1
                    break
                x, rv = xt, rvt
                if hit is not None and step == amax:             # the limiting solid has dissolved completely
                    act[hit] = False; x[hit] = 0.0; changes += 1
                    Z, pos, sol = setup()
                if any(float(np.sum(x[a * N:(a + 1) * N][pos[a * N:(a + 1) * N]])) < 1.0e-10 for a in range(n_liq)):
                    return x, converged                          # a liquid disappears: the outer loop removes it
                if (n_liq > 1 and self._truncate(x, n_liq, act, A)) | self._vanish(x, n_liq, act, A):
                    Z, pos, sol = setup(); rv = raw(x)
            else:
                converged = False
            if verbose:
                print(f"    mu={mu:.1e} it={it} dec={dec:.2e} active solids={int(np.sum(act & is_solid))}")
            # stage converged: restore trace entries removed too early (final stage), then add the most supersaturated
            # inactive solid, if any, and repeat the stage
            if mu <= self.newton_mu_min and n_liq > 1 and reentries < 3 and self._reenter(x, n_liq, act, A):
                reentries += 1
                Z, pos, sol = setup(); rv = raw(x); converged = True
                continue
            si, k = supersaturated(rv)
            if k is not None and si > self.si_tol and changes < 4 * S + 10:
                act[k] = True; x[k] = 0.0; changes += 1
                Z, pos, sol = setup(); converged = True
                continue
            # the barrier shifts the potential of entry j by mu / x_j: continue below newton_mu_min until that shift is
            # at most 1e-10 for every entry (explicit H+ or OH- can be traces of 1e-14 and less)
            small = float(np.min(x[pos])) if np.any(pos) else 1.0
            if mu <= self.newton_mu_min and mu <= 1.0e-10 * small:
                break
            mu = max(mu * self.newton_mu_factor, min(self.newton_mu_min, 1.0e-10 * small), 1.0e-300)
        return x, converged

    def _rand_solve(self, x, n_liq, ln_rh, *, act=None, max_newton=100, verbose=False):
        """Inner problem in logarithmic amounts (``inner_method = "rand"``; explicit speciation, no sign-free entries).

        RAND-type formulation (element-potential methods; Smith and Missen, 1982): every liquid (and closed-gas)
        amount is a logarithmic variable, n <- n exp(t delta), so no amount can become negative and a trace species
        is as well resolved as a major one.  The Newton step solves the reduced system of the scaled Hessian S H S
        (S = diag(n)), in which the 1/n curvature of the ideal part becomes the identity: trace species are well
        conditioned and need no barrier, no fraction-to-boundary rule and no removal or re-entry.  A step leaves the
        linear constraints (mass and charge balances) violated at second order; feasibility is restored before the
        objective is evaluated by the multiplicative correction x <- x exp(A^T y) of the logarithmic entries (and an
        additive one of open-gas amounts), a Newton iteration on the element potentials y.  Solids enter linearly and
        are handled by the active set of :meth:`_newton_solve`.  ``act`` is modified in place: all liquid entries take
        part (absent entries are given back a trace amount first).  Returns (x, converged)."""
        N, S = self.N, len(self._solids_now)
        x = np.array(x, dtype=float)
        if act is None:
            act = np.ones(len(x), dtype=bool)
        A = self._build_A(n_liq)
        nl = n_liq * N
        o_s, og = nl, nl + S
        K = len(x) - og
        if not np.all(act[:nl]) or np.any(x[:nl] <= 0.0):
            act[:nl] &= x[:nl] > 0.0
            self._reactivate(x, n_liq, act, A)
            act[:nl] = True
            x[:nl] = np.maximum(x[:nl], 1.0e-300)
        is_log = np.zeros(len(x), dtype=bool); is_log[:nl] = True
        if self._gas_mode == "closed":
            is_log[og:] = True
        is_solid = np.zeros(len(x), dtype=bool); is_solid[o_s:og] = True
        target = A @ x
        if A.size and np.any(self.z != 0):
            target[self.Nc - 1:] = 0.0                           # electroneutral liquids
        corr = is_log | (~is_solid & ~is_log & (np.arange(len(x)) >= og))   # entries that restore feasibility
        converged = True

        def restore(xx):
            """x exp(A^T y) on logarithmic entries (+ A^T y on open-gas entries) such that A x = target."""
            if not A.size:
                return xx
            xx = xx.copy()
            Ac = A[:, corr]
            for _ in range(30):
                r = A @ xx - target
                if float(np.max(np.abs(r))) <= 1.0e-15 * max(1.0, float(np.max(np.abs(target)))):
                    return xx
                w = np.where(is_log[corr], xx[corr], 1.0)
                J = (Ac * w) @ Ac.T
                try:
                    y = np.linalg.solve(J, r)
                except np.linalg.LinAlgError:
                    y = np.linalg.lstsq(J, r, rcond=None)[0]
                e = Ac.T @ y
                v = xx[corr]
                lg = is_log[corr]
                v[lg] = v[lg] * np.exp(np.clip(-e[lg], -50.0, 50.0))
                v[~lg] = v[~lg] - e[~lg]
                xx[corr] = v
            return None

        def raw(xx):
            try:
                with np.errstate(all="ignore"):
                    return self._objective(xx, n_liq, ln_rh, act)
            except (ValueError, OverflowError, FloatingPointError, ZeroDivisionError):
                return None

        nn = self.lm.n_neutral
        mm = self.lm._mm

        def in_range(xx):
            """Without a barrier, the water (solvent) of a liquid can be driven towards zero; AIOMFAC then returns
            arbitrarily negative values far outside its range (an anhydrous NH4NO3 "liquid" with F = -6e7 was accepted
            as a decrease).  Trial points with a total ion molality above 1e4 mol/kg are rejected (metastable salt
            liquids reach about 200 mol/kg)."""
            for a in range(n_liq):
                n = xx[a * N:(a + 1) * N]
                if np.sum(n[nn:]) > 1.0e4 * float(np.dot(n[:nn], mm)):
                    return False
            return True

        def grad(rv):
            g = self._grad(rv[1], ln_rh, rv[2])
            g[~act] = 0.0
            return g

        hcache: list = [None] * n_liq

        def liquid_hessian(a, rv):
            n = x[a * N:(a + 1) * N]
            if self.hess_scheme == "central":
                return self.lm.hessian(n, self.T)
            if self._use_ad():
                return self.lm.hessian_ad(n, self.T)
            # excess part reused as in _newton_solve; not symmetrized: CO2(aq) makes the Jacobian of the potentials
            # slightly non-symmetric (Sect. 3.4)
            c = hcache[a]
            if c is None or float(np.max(np.abs(n - c[0]))) > self.hess_reuse_tol * float(np.sum(n)):
                hcache[a] = (n.copy(), self.lm.hessian_excess(n, self.T, la0=rv[1][a]))
            return self.lm.ideal_jacobian(n) + hcache[a][1]

        refused: set = set()

        def supersaturated(rv):
            best = (-math.inf, None)
            for j, sld in enumerate(self._solids_now):
                k = o_s + j
                if act[k] or k in refused:
                    continue
                vals = [self.si_of(la, ln_rh, [sld])[sld.key] for la in rv[1]]
                si = max((v for v in vals if np.isfinite(v)), default=-math.inf)
                if si > best[0]:
                    best = (si, k)
            return best

        def setup():
            As = A[:, act] * np.where(is_log[act], np.sqrt(np.abs(x[act])), 1.0)
            if As.size:
                Za = null_space(As)
            else:
                Za = np.eye(int(np.sum(act)))
            Z = np.zeros((len(x), Za.shape[1]))
            Z[act] = Za
            return Z

        rv = raw(x)
        if rv is None:
            return x, False
        changes = 0
        dec_prev = math.inf
        short = 0
        grad_ok = bool(getattr(self.lm, "carbonate", False))
        while True:
            for it in range(max_newton):
                F0 = rv[0]
                g = grad(rv)
                s = np.where(is_log, np.sqrt(np.abs(x)), 1.0)    # D = diag(sqrt n): D (1/n) D = identity
                H = np.zeros((len(x), len(x)))
                for a in range(n_liq):
                    H[a * N:(a + 1) * N, a * N:(a + 1) * N] = liquid_hessian(a, rv)
                if K:
                    H[og:, og:] += rv[3]
                Hs = H * s[:, None] * s[None, :]
                gs = g * s
                Z = setup()
                Hr = Z.T @ Hs @ Z
                gr = Z.T @ gs
                if Hr.size == 0 or not np.all(np.isfinite(Hr)):
                    break
                w, Q = np.linalg.eigh(0.5 * (Hr + Hr.T))
                big = max(1.0, abs(w[-1]))
                shift = -w[0] + 1.0e-8 * big if w[0] <= 1.0e-12 * big else 0.0
                if shift == 0.0:
                    try:
                        d = Z @ np.linalg.solve(Hr, -gr)
                    except np.linalg.LinAlgError:
                        d = Z @ (-Q @ ((Q.T @ gr) / w))
                else:
                    d = Z @ (-Q @ ((Q.T @ gr) / (w + shift)))
                dec = float(-gr @ (Z.T @ d))
                r0 = float(np.linalg.norm(gr))
                d = s * d                                        # step in amounts
                lg = is_log & act
                dmax = float(np.max(np.abs(d[lg]) / x[lg])) if np.any(lg) else 0.0
                # relative steps of trace entries carry the round-off of the major ones (absolute 1e-16): a decrement
                # at round-off level ends the iteration as well
                if dec < 1.0e-15 and (dmax < 1.0e-8 or dec < 1.0e-24):
                    break
                if dec > 0.9 * dec_prev:                         # stalling: refresh the excess Hessian
                    hcache[:] = [None] * n_liq
                dec_prev = dec
                amax, hit = 1.0, None
                sol = act & is_solid
                negs = (d < 0) & sol
                if np.any(negs):
                    ratios = -x[negs] / d[negs]
                    j = int(np.argmin(ratios))
                    if ratios[j] < amax:
                        amax, hit = float(ratios[j]), int(np.flatnonzero(negs)[j])
                if hit is not None and amax <= 1.0e-14:
                    if x[hit] == 0.0:
                        refused.add(hit)
                    act[hit] = False; x[hit] = 0.0; changes += 1
                    continue
                step, ok = amax, False
                for _ in range(40):
                    xt = x.copy()
                    # increases additive, decreases multiplicative (n exp(t dn / n)): smooth at dn = 0, and an amount
                    # can approach zero but never cross it.  A decrease is limited to a factor exp(-7) per step: the
                    # linear step of a liquid whose composition changes a lot can predict dn = -450 n for a minor
                    # species, which the exponential would turn into 1e-195, far below its equilibrium amount, from
                    # where the additive increases recover only slowly
                    dl = step * d[is_log]
                    xl = x[is_log]
                    with np.errstate(all="ignore"):
                        xt[is_log] = np.where(dl >= 0.0, xl + dl, xl * np.exp(np.maximum(dl / xl, -7.0)))
                    xt[~is_log] = x[~is_log] + step * d[~is_log]
                    if hit is not None and step == amax:
                        xt[hit] = 0.0
                    xt[sol] = np.maximum(xt[sol], 0.0)
                    xt = restore(xt)
                    if xt is not None and np.all(xt[is_log] > 0.0) and in_range(xt):
                        rvt = raw(xt)
                        if rvt is not None and np.isfinite(rvt[0]):
                            gt = grad(rvt)
                            if np.all(np.isfinite(gt)):
                                st = np.where(is_log, np.sqrt(np.abs(xt)), 1.0)
                                At = A[:, act] * st[act]
                                Pt = gt[act] * st[act]
                                if At.size:
                                    Pt = Pt - At.T @ np.linalg.lstsq(At.T, Pt, rcond=None)[0]
                                # the reduced-gradient criterion only close to convergence, where F reaches its
                                # round-off, and in carbonate systems, whose potentials are only approximately a
                                # gradient (CO2(aq), Sect. 3.4): elsewhere it accepts uphill steps, which can empty a
                                # freshly seeded liquid (water + pinonaldehyde just below the LLPS onset)
                                if (rvt[0] <= F0 - 1.0e-4 * step * dec
                                        or ((dec < 1.0e-10 or grad_ok)
                                            and float(np.linalg.norm(Pt)) <= (1.0 - 1.0e-4 * step) * r0)):
                                    ok = True
                                    break
                    step *= 0.5
                if not ok:
                    if dec < 1.0e-14:
                        break
                    converged = False
                    self._ls_failures += 1
                    break
                if verbose:
                    print(f"      it={it} F={rvt[0]:.15e} dec={dec:.2e} dmax={dmax:.2e} step={step:.2e} shift={shift:.1e} "
                          f"r0={r0:.2e}")
                x, rv = xt, rvt
                if hit is not None and step == amax:
                    act[hit] = False; x[hit] = 0.0; changes += 1
                # a small liquid that drains only by short steps (non-convex region, where the Newton step is dominated
                # by a nearly singular direction): merge it into the liquid of the most similar composition; the outer
                # loop removes it and the stability test seeds it again if it is needed
                short = short + 1 if step < 1.0e-2 else 0
                if short >= 3 and n_liq > 1:
                    sizes = [float(np.sum(x[a * N:(a + 1) * N])) for a in range(n_liq)]
                    a = int(np.argmin(sizes))
                    if sizes[a] < 1.0e-3 * sum(sizes):
                        xa = x[a * N:(a + 1) * N] / sizes[a]
                        b = min((k for k in range(n_liq) if k != a),
                                key=lambda k: float(np.max(np.abs(x[k * N:(k + 1) * N] / sizes[k] - xa))))
                        x[b * N:(b + 1) * N] += x[a * N:(a + 1) * N]
                        x[a * N:(a + 1) * N] = 0.0
                        return x, converged
                if any(float(np.sum(x[a * N:(a + 1) * N])) < 1.0e-10 for a in range(n_liq)):
                    return x, converged                          # a liquid disappears: the outer loop removes it
            else:
                converged = False
            if verbose:
                print(f"    rand it={it} dec={dec:.2e} active solids={int(np.sum(act & is_solid))}")
            si, k = supersaturated(rv)
            if k is not None and si > self.si_tol and changes < 4 * S + 10:
                act[k] = True; x[k] = 0.0; changes += 1
                converged = True
                continue
            break
        return x, converged

    def _water_guess(self, n: np.ndarray, rh: float) -> float:
        """Water amount for a single liquid at roughly a_w = RH (bisection on ln n_w)."""
        dry = n[1:].sum()
        lo, hi = math.log(1e-6 * dry), math.log(1e4 * dry)
        lmc = self.lm._int if self.speciation == "explicit" else self.lm      # component-space model
        f = lambda lw: lmc.ln_a(np.r_[math.exp(lw), n[1:]], self.T)[0] - math.log(rh)
        try:
            flo, fhi = f(lo), f(hi)
            if flo > 0 or fhi < 0:
                return dry * rh / (1 - rh)
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                if f(mid) > 0:
                    hi = mid
                else:
                    lo = mid
            return math.exp(0.5 * (lo + hi))
        except (ValueError, FloatingPointError, OverflowError, ZeroDivisionError):
            return dry * rh / (1 - rh)

    def _tpd_minimize(self, mu_eq: np.ndarray, w0: np.ndarray, refs=None, known=None):
        """Stationary point of the tangent-plane distance from the start w0: successive substitution
        (:meth:`_tpd_ss`), polished with the barrier Newton method (:meth:`_tpd_newton`) if it does not converge.
        Carbonate systems (sign-free proton excess) use the Newton method only."""
        if self.tpd_method == "ss" and not self.lm._carb:
            w, t, ok = self._tpd_ss(mu_eq, w0, refs=refs)
            if ok:
                return w, t
            w0 = w
        return self._tpd_newton(mu_eq, w0, refs=refs, known=known)

    def _tpd_ss(self, mu_eq: np.ndarray, w0: np.ndarray, max_iter: int = 100, tol: float = 1.0e-10, refs=None):
        """Successive substitution for the stationary points of TPD(w) = sum w_i (ln a_i(w) - mu_i) (Michelsen, 1982),
        generalized to the ion basis.  With ln a = ln_ideal + r (r: activity coefficients and speciation correction,
        a smooth function of the intensive composition), the unnormalized amounts W are updated from r at the current
        composition:

            neutrals:  W_i = exp(mu_i - r_i)
            ions:      W_i = M exp(mu_i - r_i + z_i psi) / T,   M = sum_neutral W_j M_j (solvent mass),

        where T = sum W is closed-form (a quadratic) and psi makes W electroneutral (a monotonic 1-D equation).  At a
        fixed point ln a_i - mu_i = -ln T + z_i psi for every species, i.e. w = W/T is a stationary point of TPD under
        sum w = 1 and electroneutrality, with TPD = -ln T: the liquid is unstable if T > 1.  One activity evaluation
        per iteration.

        In electrolyte mixtures the plain substitution can oscillate (period two, e.g. Na+ and H+ exchanging their
        roles), so r is relaxed: r_used <- r_used + beta (r(w) - r_used), with beta halved whenever the change of ln W
        does not decrease (down to 0.05) and increased again after contracting steps.  Relaxing r keeps every iterate
        electroneutral.

        Early termination: an iterate within 1e-3 (max norm) of one of the current liquids (``refs``) with TPD above
        -tol_tpd is the trivial solution (returned as converged, it is discarded by the caller); a clearly negative TPD
        (< -1e-2) needs only a seed, so the iteration stops once ln W changes by less than 1e-4 or after 60 iterations.
        Descent guard: in strongly non-ideal electrolyte mixtures a TPD minimum can be a repelling fixed point of the
        substitution (an eigenvalue of its Jacobian above one, which relaxation cannot cure), and the iterates then
        drift uphill to the trivial solution.  The TPD of every iterate is therefore monitored; once it exceeds the
        lowest value reached (including the start) by more than max(1e-8, 1e-3 |lowest|), the substitution stops and
        returns the lowest iterate as not converged, so the Newton method continues from there.
        Returns (w, TPD(w), converged)."""
        lm, N, nn = self.lm, self.N, self.lm.n_neutral
        z = self.z
        ions = np.arange(nn, N)
        cat, an = ions[z[ions] > 0], ions[z[ions] < 0]
        mm = lm._mm
        w = np.asarray(w0, dtype=float); w = w / w.sum()
        tpd_of = lambda w_, la_: float(np.dot(w_, la_ - mu_eq))
        la = lm.ln_a(w, self.T)
        best_t, best_w = tpd_of(w, la), w.copy()
        lnW_old = None
        ok = False
        r_used = None
        beta, step_old = 1.0, math.inf
        for it in range(max_iter):
            r_new = la - lm.ln_ideal(w)
            r = r_new if r_used is None else r_used + beta * (r_new - r_used)
            r_used = r
            W = np.empty(N)
            W[:nn] = np.exp(np.clip(mu_eq[:nn] - r[:nn], -700.0, 700.0))
            Tn = float(W[:nn].sum())
            if len(ions):
                M = float(np.dot(W[:nn], mm))
                le = math.log(M) + mu_eq[nn:] - r[nn:]                       # ln e_i at psi = 0
                zi = z[nn:]
                def g(psi):                                                     # ln(cation charge) - ln(anion charge)
                    lc = le[zi > 0] + zi[zi > 0] * psi + np.log(zi[zi > 0])
                    la_ = le[zi < 0] + zi[zi < 0] * psi + np.log(-zi[zi < 0])
                    mc, ma = lc.max(), la_.max()
                    return (mc + math.log(np.exp(lc - mc).sum())) - (ma + math.log(np.exp(la_ - ma).sum()))
                lo, hi = -5.0, 5.0
                # bracket the root; g is not finite when an activity coefficient has overflowed (the bracket search
                # would then never end), and |psi| beyond ~700/|z| only means a diverging iterate
                g_lo, g_hi = g(lo), g(hi)
                for _ in range(80):
                    if not (math.isfinite(g_lo) and g_lo > 0):
                        break
                    lo -= 10.0; g_lo = g(lo)
                for _ in range(80):
                    if not (math.isfinite(g_hi) and g_hi < 0):
                        break
                    hi += 10.0; g_hi = g(hi)
                if not (math.isfinite(g_lo) and math.isfinite(g_hi) and g_lo <= 0.0 <= g_hi):
                    return best_w, best_t, False
                from scipy.optimize import brentq
                psi = brentq(g, lo, hi, xtol=1e-14)
                le = le + zi * psi
                cmax = le.max()
                if cmax > 700.0:                                 # diverging (e.g. a start far from any minimum)
                    return best_w, best_t, False
                E = math.exp(cmax) * float(np.exp(le - cmax).sum())
                T = 0.5 * (Tn + math.sqrt(Tn * Tn + 4.0 * E))
                W[nn:] = np.exp(le) / T
            else:
                T = Tn
            lnW = np.log(W)
            w = W / W.sum()
            la = lm.ln_a(w, self.T)
            if not np.all(np.isfinite(la)):
                return w, float("nan"), False
            t_now = tpd_of(w, la)
            if t_now > best_t + max(1.0e-8, 1.0e-3 * abs(best_t)):
                return best_w, best_t, False                             # uphill: hand over to Newton
            if t_now < best_t:
                best_t, best_w = t_now, w.copy()
            if refs is not None and t_now > -self.tol_tpd and \
                    min(float(np.max(np.abs(w - x))) for x in refs) < 1.0e-3:
                return w, t_now, True                                    # trivial solution
            if lnW_old is not None:
                step = float(np.max(np.abs(lnW - lnW_old)))
                if step >= step_old:
                    beta = max(0.5 * beta, 0.05)
                elif step < 0.5 * step_old:
                    beta = min(1.0, 1.5 * beta)
                step_old = step
                if step < tol:
                    ok = True
                    break
                if t_now < -1.0e-2 and (step < 1.0e-4 or it >= 60):    # clearly unstable: accurate enough to seed
                    ok = True
                    break
            lnW_old = lnW
        return w, tpd_of(w, la), ok

    def _tpd_newton(self, mu_eq: np.ndarray, w0: np.ndarray, refs=None, known=None):
        """min_w sum w_i (ln a_i(w) - mu_i) over electroneutral w with sum of the bounded entries = 1 (barrier Newton
        from w0; the proton excess of a carbonate system is a free entry).  Early exits: within 1e-3 (max norm) of a
        current liquid (``refs``) with TPD > -tol_tpd (the trivial solution, discarded by the caller), or within 1e-3
        of a minimum already found in this stability test (``known``)."""
        N = self.N
        free = self.lm.free
        bnd = ~free
        rows = [bnd.astype(float)]
        if np.any(self.z != 0):
            rows.append(self.z)
        A = np.array(rows)
        Z = null_space(A)
        w = np.asarray(w0, dtype=float) / float(np.sum(w0[bnd]))
        lm = self.lm
        la = lm.ln_a(w, self.T)
        cache = None                                         # (w, excess Hessian), reused while w changes little
        mu = 1.0e-4
        for _ in range(9):
            for it in range(40):
                f = float(np.dot(w, la - mu_eq))
                g = la - mu_eq
                g[bnd] -= mu / w[bnd]
                if self.hess_scheme == "central":
                    H = lm.hessian(w, self.T)
                elif self._use_ad():
                    H = lm.hessian_ad(w, self.T, w != 0.0)
                else:
                    tol_ = 0.0 if lm._carb else self.tpd_hess_reuse_tol
                    if cache is None or float(np.max(np.abs(w - cache[0]))) > tol_ * float(np.sum(np.abs(w))):
                        cache = (w.copy(), lm.hessian_excess(w, self.T, la0=la))
                    H = lm.hessian_split(w, self.T, R=cache[1])
                dgn = np.zeros(N); dgn[bnd] = mu / w[bnd] ** 2
                H[np.diag_indices_from(H)] += dgn
                Hr = Z.T @ H @ Z; gr = Z.T @ g
                ev, Q = np.linalg.eigh(0.5 * (Hr + Hr.T))
                big = max(1.0, float(np.max(np.abs(ev))))
                shift = -ev[0] + 1e-8 * big if ev[0] <= 1e-12 * big else 0.0
                d = Z @ (-Q @ ((Q.T @ gr) / (ev + shift)))
                dec = float(-gr @ (Z.T @ d))
                if dec < 1e-14:
                    break
                neg = (d < 0) & bnd
                amax = min(1.0, 0.995 * float(np.min(-w[neg] / d[neg]))) if np.any(neg) else 1.0
                phi0 = f - mu * float(np.sum(np.log(w[bnd])))
                step, ok = amax, False
                for _ in range(40):
                    wt = w + step * d
                    if np.all(wt[bnd] > 0):
                        lat = lm.ln_a(wt, self.T)
                        phit = float(np.dot(wt, lat - mu_eq)) - mu * float(np.sum(np.log(wt[bnd])))
                        if np.isfinite(phit) and phit <= phi0 - 1e-4 * step * dec:
                            ok = True
                            break
                    step *= 0.5
                if not ok:
                    break
                w, la = wt, lat
                if refs or known:
                    t_now = float(np.dot(w, la - mu_eq))
                    if refs and t_now > -self.tol_tpd and min(float(np.max(np.abs(w - x))) for x in refs) < 1.0e-3:
                        return w, t_now                          # trivial solution
                    if known and min(float(np.max(np.abs(w - x))) for x in known) < 1.0e-3:
                        return w, t_now                          # a minimum found from another start
            mu *= 0.1
        return w, float(np.dot(w, la - mu_eq))

    def _trial_points(self, b: np.ndarray, rh: float, liquids: list) -> list:
        """Electroneutral trial compositions (sum 1) for the stability test.  Compositions are fixed mole-fraction
        patterns of the components, independent of RH (an RH-scaled water content would make "organic-rich" trials
        water-rich close to saturation): each organic at x = 0.3, 0.7 and 0.95 in water; all organics in feed
        proportion at x = 0.5 and 0.9; salt solutions at two water-to-ion ratios; dilute water; plus perturbed copies
        of the current liquids.  With explicit speciation the component patterns are speciated with the internal model
        (a pattern with all acid as free H+ starts far from any minimum and misses, e.g., a third, sulfate-rich liquid
        of DLT + NaCl + H2SO4)."""
        Nc, N, nn = self.Nc, self.N, self.lm.n_neutral
        eps = 1e-6
        zc = self.zc
        ion_fr = np.zeros(Nc)
        if np.abs(b[nn:Nc]).sum() > 0:
            ion_fr[nn:] = np.abs(b[nn:Nc]) / np.abs(b[nn:Nc]).sum()
        org_fr = np.zeros(Nc)
        if b[1:nn].sum() > 0:
            org_fr[1:nn] = b[1:nn] / b[1:nn].sum()
        comp_trials = []

        def make(water, org, ions):
            w = np.full(Nc, eps)
            w[0] = water
            w[1:nn] += org[1:nn]
            w[nn:] = np.maximum(ions[nn:], eps * ion_fr[nn:] + 1e-12)
            return w

        for k in range(1, nn):
            for xo in (0.3, 0.7, 0.95):
                o = np.zeros(Nc); o[k] = xo
                comp_trials.append(make(1.0 - xo, o, eps * ion_fr))
        if nn > 2:
            for xo in (0.5, 0.9):
                comp_trials.append(make(1.0 - xo, xo * org_fr, eps * ion_fr))
        if ion_fr[nn:].sum() > 0:
            for water in (2.0, 10.0):                                # concentrated and moderate salt solutions
                comp_trials.append(make(water, eps * org_fr, ion_fr))
            # single-salt solutions of every cation-anion pair present: a liquid enriched in one salt (e.g. the
            # Na2SO4-rich third liquid of DLT + NaCl + H2SO4) is otherwise reached only by chance
            cats = [i for i in range(nn, Nc) if zc[i] > 0 and b[i] > 0]
            ans = [i for i in range(nn, Nc) if zc[i] < 0 and b[i] > 0]
            if len(cats) * len(ans) > 1:
                for ci in cats:
                    for ai in ans:
                        v = np.zeros(Nc); v[ci] = -zc[ai]; v[ai] = zc[ci]; v /= v.sum()
                        for water in (2.0, 10.0):
                            comp_trials.append(make(water, eps * org_fr, v))
        comp_trials.append(make(1.0, eps * org_fr, eps * ion_fr))      # dilute water
        out = []
        for w in comp_trials:                                         # electroneutral in the component basis
            w = np.maximum(w, 1e-12)
            if self._carbonate:                                       # neutralize with the proton excess
                kh = self.names.index("H+")
                w[kh] = 0.0
                w[kh] = -float(np.dot(zc, w))
            elif np.any(zc != 0):                                     # scale the anions
                qp = float(np.sum(np.where(zc > 0, zc * w, 0.0)))
                qn = float(-np.sum(np.where(zc < 0, zc * w, 0.0)))
                if qn > 0:
                    w = np.where(zc < 0, w * qp / qn, w)
            if self.speciation == "explicit":
                try:
                    ws = self.lm.initial_species(w, self.T)
                except (ValueError, FloatingPointError, OverflowError, ZeroDivisionError):
                    continue
                out.append(ws / ws.sum())
            elif self.lm._carb:
                out.append(w / float(np.sum(w[~self.lm.free])))
            else:
                out.append(w / w.sum())
        rng = np.random.default_rng(12345)
        for L in liquids:                                             # perturbed copies of the current liquids
            w = np.maximum(np.abs(L.amounts) / L.total, 1e-9) * np.exp(0.3 * rng.standard_normal(N) * (self.z == 0))
            w = np.maximum(w, 1e-12)
            if self.lm._carb:
                kh = self.lm._kh
                w[kh] = 0.0
                w[kh] = -float(np.dot(self.z, w))
                out.append(w / float(np.sum(w[~self.lm.free])))
                continue
            if np.any(self.z != 0):
                qp = float(np.sum(np.where(self.z > 0, self.z * w, 0.0)))
                qn = float(-np.sum(np.where(self.z < 0, self.z * w, 0.0)))
                if qn > 0:
                    w = np.where(self.z < 0, w * qp / qn, w)
            out.append(w / w.sum())
        return out

    def solve(self, feed: dict, rh: float, *, solids="all", p_gas: dict | None = None, gas_total: dict | None = None,
              n_air: float | None = None, P_atm: float = 1.0, max_liquids: int = 3, max_outer: int = 8,
              verbose: bool = False, init: "PhaseEquilibriumResult | None" = None) -> PhaseEquilibriumResult:
        """See :meth:`_solve`.  A warm-started solve (``init``) that does not converge is repeated from scratch, and
        the cold result is used if it converges (a start from a neighbouring state can be poor, e.g. a small salt
        liquid that disappears at the new RH)."""
        kw = dict(solids=solids, p_gas=p_gas, gas_total=gas_total, n_air=n_air, P_atm=P_atm, max_liquids=max_liquids,
                  max_outer=max_outer, verbose=verbose)
        res = self._solve(feed, rh, init=init, **kw)
        if init is not None and res.status == "not_converged":
            cold = self._solve(feed, rh, init=None, **kw)
            if cold.status == "converged":
                cold.message = ((cold.message + "; ") if cold.message else "") + "warm start discarded (not converged)"
                return cold
        return res

    def _solve(self, feed: dict, rh: float, *, solids="all", p_gas: dict | None = None, gas_total: dict | None = None,
               n_air: float | None = None, P_atm: float = 1.0, max_liquids: int = 3, max_outer: int = 8,
               verbose: bool = False, init: "PhaseEquilibriumResult | None" = None) -> PhaseEquilibriumResult:
        """Equilibrium for the non-water ``feed`` [mol] (organic names and ion keys) at relative humidity ``rh``.

        Gases (keys of :data:`aiomfac_py.gases.GASES`: NH3, HNO3, HCl, CO2) are optional:
          * ``p_gas={"HCl": 1e-9}``: open system, the particle exchanges with a reservoir of fixed partial pressure [atm];
            ``result.gas`` is the net amount released (negative = uptake).
          * ``gas_total={"HCl": 0.0}, n_air=...``: closed system with ``n_air`` mol of inert air at ``P_atm``; the
            totals of the volatile species (particle + gas) are conserved and ``result.gas`` holds the gas amounts.
        A carbonate system (``CO3--`` with ``H+`` as proton excess) can be given ``"CO2"`` in either mode.

        ``init``: a previous result for the same system and feed (e.g. at a neighbouring RH).  Its liquids, active
        solids and gas amounts are the starting point (warm start; the water of every liquid is first adjusted to
        a_w = RH); the stability test runs as usual, so the result does not depend on the start.  An incompatible
        ``init`` (other components, other feed) is ignored."""
        if not 0.0 < rh < 1.0:
            raise ValueError("rh must be in (0, 1)")
        # species that are absent from the feed and cannot be supplied by any of the given gases are removed from the
        # problem (the barrier would otherwise need an artificial positive amount for them)
        gas_keys = list((p_gas or gas_total or {}).keys())
        reach = {i for g in gas_keys if g in GASES for i in GASES[g].ions}
        present = {k for k, v in feed.items() if v != 0.0} | reach
        if gas_total:
            present |= {i for g, v in gas_total.items() if v > 0 for i in GASES[g].ions}
        if "CO3--" in present and "H+" in self._ions:
            present.add("H+")                                # proton excess of a carbonate system
        orgs = [c for c in self._organics if c.name in present]
        ions_kept = [i for i in self._ions if i in present]
        if len(orgs) < len(self._organics) or len(ions_kept) < len(self._ions):
            if not ions_kept and not orgs:
                raise ValueError("empty feed")
            key = (tuple(c.name for c in orgs), tuple(ions_kept))
            if key not in self._children:
                sk = None if self._solid_keys is None else [k for k in self._solid_keys
                                                             if set(SOLIDS[k].ions) <= set(ions_kept)]
                sub_lm = self.lm.restrict(orgs, ions_kept) if hasattr(self.lm, "restrict") else None
                child = PhaseEquilibrium(orgs, ions_kept, self.T, k_mode=self.k_mode, solid_keys=sk,
                                         speciation=self.speciation, liquid_model=sub_lm)
                child.tol_tpd, child.trace_tol, child.seed_fractions = self.tol_tpd, self.trace_tol, self.seed_fractions
                child.hess_scheme, child.hess_reuse_tol = self.hess_scheme, self.hess_reuse_tol
                child.tpd_hess_reuse_tol, child.tpd_method = self.tpd_hess_reuse_tol, self.tpd_method
                child.inner_method, child.si_tol = self.inner_method, self.si_tol
                child.seed_method, child.tpd_early_stop = self.seed_method, self.tpd_early_stop
                child.newton_mu0, child.newton_mu_factor, child.newton_mu_min = (self.newton_mu0, self.newton_mu_factor,
                                                                                  self.newton_mu_min)
                self._children[key] = child
            sub_feed = {k: v for k, v in feed.items() if k in present}
            if isinstance(solids, (list, tuple)):
                solids = [k for k in solids if set(SOLIDS[k].ions) <= set(ions_kept)] or "none"
            res = self._children[key].solve(sub_feed, rh, solids=solids, p_gas=p_gas, gas_total=gas_total, n_air=n_air,
                                            P_atm=P_atm, max_liquids=max_liquids, max_outer=max_outer, verbose=verbose,
                                            init=init)
            dropped = [n for n in self.names[1:] if n not in res.names]
            res.message = (res.message + "; " if res.message else "") + f"species absent from the problem: {dropped}"
            return res
        if p_gas and gas_total:
            raise ValueError("give either p_gas (open system) or gas_total (closed system), not both")
        ln_rh = math.log(rh)
        N, nn = self.N, self.lm.n_neutral
        if self.speciation == "explicit":
            self.lm.set_conditions(self.T, ln_rh)
        b = np.zeros(self.Nc)
        for k, v in feed.items():
            if k not in self.names or k == "Water":
                raise KeyError(f"feed species {k!r} not in {self.names[1:]}")
            b[self.names.index(k)] = float(v)

        if solids == "all":
            self._solids_now = list(self.all_solids)
        elif solids == "none":
            self._solids_now = []
        else:
            self._solids_now = [SOLIDS[k] for k in solids]
        gas_spec = p_gas or gas_total or {}
        for k in gas_spec:
            if k not in GASES:
                raise KeyError(f"unknown gas {k!r} (available: {list(GASES)})")
            if not set(GASES[k].ions) <= set(self.lm.ions):
                raise ValueError(f"gas {k} needs the ions {sorted(GASES[k].ions)} in the system")
        self._gases_now = [GASES[k] for k in gas_spec]
        self._gas_mode = "open" if p_gas else ("closed" if gas_total else "none")
        if self._gas_mode == "closed":
            if n_air is None or n_air <= 0:
                raise ValueError("a closed gas phase needs n_air > 0 (mol of inert air)")
        G = self._columns(self._gases_now)
        gt = np.array([float(gas_spec[g.key]) for g in self._gases_now]) if self._gas_mode == "closed" else np.zeros(len(self._gases_now))
        b = b + G @ gt                                            # closed: volatile totals enter the mass balance
        if abs(float(np.dot(self.zc, b))) > 1e-9 * max(1.0, float(np.abs(b).sum())):
            raise ValueError("feed is not electroneutral")
        scale = float(np.abs(b[1:]).sum())
        b = b / scale
        V = self._columns(self._solids_now)
        S, K = V.shape[1], G.shape[1]
        self._W = np.hstack([V, G])
        self._solid_cost_now = self._solid_cost(self._solids_now, ln_rh)
        if self._gas_mode == "open":
            self._gas_cost = np.array([g.ln_k(self.T) + math.log(p_gas[g.key]) - g.h * ln_rh for g in self._gases_now])
        elif self._gas_mode == "closed":
            self._gas_cost = np.array([g.ln_k(self.T) - g.h * ln_rh + math.log(P_atm) for g in self._gases_now])
            self._n_air = float(n_air) / scale
        else:
            self._gas_cost = np.zeros(0)
        free = self.lm.free

        # strictly feasible start: one liquid with (almost) everything, solids at a tiny amount, closed-system gases
        # mostly in the gas phase, open-system gases at zero net transfer
        # carbonate systems with organics keep the barrier method: their potentials are only approximately a gradient
        # (Sect. 3.3 of the design notes) and the unregularized Newton iteration wanders; the barrier stabilizes it
        use_barrier = self.inner_method == "barrier" or (self.lm._carb and self.lm.n_neutral > 1)
        u0 = np.full(S, 1e-6) if use_barrier else np.zeros(S)   # active set: no solid at the start
        g0 = self._initial_gas(b, V @ u0, G, gt / scale) if K else np.zeros(0)
        n0 = b.copy()                                            # component amounts of the starting liquid
        n0[1:] = b[1:] - V[1:] @ u0 - G[1:] @ g0
        floor = 1e-7
        cf = self._comp_free
        n0[1:] = np.where(cf[1:], n0[1:], np.maximum(n0[1:], floor))
        zc = self.zc
        if np.any(zc != 0) and np.any(b[nn:] != 0):              # re-neutralize after flooring
            if self._carbonate:
                kh = self.names.index("H+")
                n0[kh] -= float(np.dot(zc, n0))
            else:
                qp = float(np.sum(np.where(zc > 0, zc * n0, 0.0)))
                qn = float(-np.sum(np.where(zc < 0, zc * n0, 0.0)))
                n0 = np.where(zc < 0, n0 * qp / qn, n0)
        n0[0] = self._water_guess(n0, rh)
        if self.speciation == "explicit":                        # species from the internal speciation
            n0 = self.lm.initial_species(n0, self.T)
        phases = [n0]
        warm = self._warm_start(init, b, scale, rh, V, G, S, K, use_barrier)
        if warm is not None:
            phases, u0, g0, solid_act = warm
        x = np.concatenate(phases + [u0, g0])
        n_outer = 0
        tpd_min = float("nan")
        inner_ok = True
        history = []
        act = np.ones(len(x), dtype=bool)                     # entries taking part (False: species absent from a liquid)
        if not use_barrier:
            act[len(phases) * N:len(phases) * N + S] = False   # solids enter through the active set
            if warm is not None:
                act[len(phases) * N:len(phases) * N + S] = solid_act   # warm start: the previous active solids
        inner = self._barrier_solve if use_barrier else self._newton_solve
        if self.inner_method == "rand" and not use_barrier and not np.any(free):
            inner = self._rand_solve
        self._trunc_max = 0.0
        self._ls_failures = 0
        while True:
            n_outer += 1
            n_liq = len(phases)
            x, conv = inner(x, n_liq, ln_rh, act=act, verbose=verbose)
            inner_ok = inner_ok and conv
            phases = [x[a * N:(a + 1) * N] for a in range(n_liq)]
            masks = [act[a * N:(a + 1) * N].copy() for a in range(n_liq)]
            u = x[n_liq * N:]
            tail_act = act[n_liq * N:].copy()
            # remove vanished liquids, merge identical ones
            size = lambda p: float(np.sum(p[~free]))
            # a liquid that holds (practically) only water is dropped: water is open, so at RH < 1 such a liquid only
            # raises F (by n_w (0 - ln RH)); it arises when the trace removal of the inner solver takes the last
            # solute out of a small water-rich liquid, which then cannot reach a_w = RH (seen with surrogate
            # activity models, research/paper_5)
            solute = lambda p: float(np.sum(np.abs(p[1:])))
            keep = [a for a in range(n_liq) if size(phases[a]) > 1e-8
                    and not (n_liq > 1 and solute(phases[a]) <= 1e-12 * size(phases[a]))]
            if not keep:                                       # never drop the last liquid (see the "dry" status)
                keep = [int(np.argmax([size(p) for p in phases]))]
            merged = []
            for a in keep:
                xa = phases[a] / size(phases[a])
                for m in merged:
                    if np.max(np.abs(xa - phases[m] / size(phases[m]))) < 1e-4:
                        phases[m] = phases[m] + phases[a]
                        masks[m] = masks[m] | masks[a]
                        break
                else:
                    merged.append(a)
            if len(merged) < n_liq:
                phases = [phases[a] for a in merged]
                x = np.concatenate(phases + [u])
                act = np.concatenate([masks[a] for a in merged] + [tail_act])
                x, conv = inner(x, len(phases), ln_rh, act=act, verbose=verbose)
                n_liq = len(phases)
                phases = [x[a * N:(a + 1) * N] for a in range(n_liq)]
                masks = [act[a * N:(a + 1) * N].copy() for a in range(n_liq)]
                u = x[n_liq * N:]
            liquids = [LiquidPhase(p.copy(), self.lm.ln_a(p, self.T), self.sp_names) for p in phases]
            dry = len(phases) == 1 and size(phases[0]) < 1e-6
            if dry:                                            # everything crystallized: no liquid to test
                tpd_min = 0.0
                break
            # stability test of the liquid set (potentials of the largest liquid; water at ln RH)
            mu_eq = self._reference_potentials(liquids, ln_rh)
            best = (0.0, None)
            known = []                                         # non-trivial minima found in this stability test
            for w0 in self._trial_points(b, rh, liquids):
                try:
                    with np.errstate(all="ignore"):
                        w, t = self._tpd_minimize(mu_eq, w0, refs=[L.mole_fractions for L in liquids], known=known)
                except (ValueError, FloatingPointError, OverflowError, np.linalg.LinAlgError, ZeroDivisionError):
                    continue
                if not (np.all(np.isfinite(w)) and np.isfinite(t) and np.all(w > 0)):
                    continue
                dist = min(float(np.max(np.abs(w - L.mole_fractions))) for L in liquids)
                if dist > 1e-3 and t < -self.tol_tpd:
                    known.append(w)
                if t < best[0] and dist > 1e-3:
                    best = (t, w)
                # a clearly unstable liquid set needs one new liquid, not the most negative of all candidates: the
                # next outer iteration tests again (Michelsen's practice); the final, stable test runs every start, and
                # so does a test whose liquid count is at max_liquids (it reports tpd_min, e.g. for binodal searches)
                if (self.tpd_early_stop is not None and n_liq < max_liquids and best[0] < -self.tpd_early_stop):
                    break
            tpd_min = best[0]
            if verbose:
                print(f"  outer {n_outer}: {n_liq} liquid(s), TPD_min={tpd_min:.3e}")
            if best[1] is None or tpd_min > -self.tol_tpd or n_liq >= max_liquids or n_outer >= max_outer:
                break
            # an added phase that merges back gives the same phase count and the same TPD again.  Close to the
            # boundary where the new liquid appears, a large seed can make the inner solve fall back to the old state:
            # retry with smaller seeds; when all fail, stop and report the negative TPD (not converged)
            key = (n_liq, round(tpd_min, 9))
            tries = history.count(key)
            if tries >= len(self.seed_fractions):
                break
            history.append(key)
            frac = self.seed_fractions[tries]
            # all liquids contain all species again before the new one is seeded
            x = np.concatenate(phases + [u])
            act = np.concatenate(masks + [tail_act])
            if not np.all(act):
                self._reactivate(x, n_liq, act, self._build_A(n_liq))
                phases = [x[a * N:(a + 1) * N] for a in range(n_liq)]
            # add the trial phase: take a neutral amount of w out of the liquid that can supply the most
            w = best[1]
            nonw = w.copy(); nonw[0] = 0.0
            # species that are traces in the trial composition do not limit the amount moved into the new phase
            major = (nonw > 1e-4 * float(np.max(np.abs(nonw[~free])))) & ~free
            major[0] = False
            full = [float(np.min(p[major] / nonw[major])) if np.any(major) else 0.0 for p in phases]
            a = int(np.argmax(full))
            donor = phases[a]

            def split(f):
                """(donor after the transfer, new liquid) for the fraction f of the largest transfer (None if invalid)."""
                theta = f * full[a]
                tr = theta * nonw
                minor = ~major & ~free
                minor[0] = False
                tr[minor] = np.minimum(tr[minor], 0.5 * donor[minor])
                q = float(np.dot(self.z, tr))                  # keep the transferred amount electroneutral
                if abs(q) > 0.0:
                    if free.any():
                        tr[np.argmax(free)] -= q / self.z[np.argmax(free)]
                    else:
                        opp = [i for i in range(self.N) if major[i] and self.z[i] * q < 0]
                        if opp:
                            k = max(opp, key=lambda i: donor[i])
                            tr[k] -= q / self.z[k]
                newp = tr.copy()
                newp[0] = theta * w[0]
                if np.all(donor[~free] - tr[~free] > 0) and np.all(newp[~free] >= 0):
                    return donor - tr, newp
                theta = min(theta, f * float(np.min(donor[~free & (nonw > 0)] / nonw[~free & (nonw > 0)])))
                if theta <= 0:                                 # strictly proportional transfer as the fallback
                    return None
                return donor - theta * nonw, theta * w.copy()

            f_seed = self.seed_fractions[tries]
            if tries == 0 and self.seed_method == "linesearch":
                # size of the new liquid by a one-dimensional minimization of the Gibbs energy of the donor and the new
                # liquid along the transfer direction; for small f the change is f * TPD * (size) < 0, so a decrease
                # always exists
                def F_liq(n):
                    la = self.lm.ln_a(n, self.T)
                    return float(np.dot(n, la)) - n[0] * ln_rh
                try:
                    F_d = F_liq(donor)
                    def dF(f):
                        sp_ = split(f)
                        if sp_ is None:
                            return math.inf
                        try:
                            with np.errstate(all="ignore"):
                                v = F_liq(sp_[0]) + F_liq(sp_[1]) - F_d
                        except (ValueError, OverflowError, FloatingPointError, ZeroDivisionError):
                            return math.inf
                        return v if np.isfinite(v) else math.inf
                    from scipy.optimize import minimize_scalar
                    opt = minimize_scalar(lambda lf: dF(math.exp(lf)), bounds=(math.log(1e-4), math.log(0.999)),
                                          method="bounded", options={"xatol": 1e-2})
                    if np.isfinite(opt.fun) and opt.fun < 0.0:
                        f_seed = float(math.exp(opt.x))
                except (ValueError, OverflowError, FloatingPointError, ZeroDivisionError):
                    pass
            sp = split(f_seed)
            if sp is None:
                break
            donor_new, newp = sp[0].copy(), sp[1].copy()
            # floor of 1e-12 of the new liquid's size: a trial from the successive substitution can carry entries of
            # 1e-239 (exp(mu - ln gamma) of a strongly excluded organic), whose 1/x curvature overflows the inner
            # iteration; the floor is taken from the donor (mass balance) and the charges are restored below
            fl = 1.0e-12 * float(np.sum(np.abs(newp[~free])))
            add = np.where(~free & (newp < fl) & (donor_new > 10.0 * fl), fl - newp, 0.0)
            add[0] = 0.0
            newp += add
            donor_new -= add
            phases[a] = donor_new
            phases.append(newp)
            if np.any(add > 0) and np.any(self.z != 0):
                xs = np.concatenate(phases + [u])
                n_new = len(phases)
                A_new = self._build_A(n_new)
                tgt = A_new @ xs
                tgt[self.Nc - 1:] = 0.0                          # charge rows: electroneutral liquids
                act_new = np.concatenate([np.ones(n_new * N, dtype=bool), tail_act])
                self._restore_constraints(xs, act_new, A_new, tgt, n_new)
                phases = [xs[k * N:(k + 1) * N] for k in range(n_new)]
            x = np.concatenate(phases + [u])
            act = np.concatenate([np.ones(len(phases) * N, dtype=bool), tail_act])   # keeps the active solids

        liquids = [LiquidPhase(p * scale, self.lm.ln_a(p, self.T), self.sp_names) for p in phases]
        org_frac = lambda L: float(np.sum(L.amounts[1:nn]) / np.sum(L.amounts[~free]))
        liquids.sort(key=org_frac, reverse=True)
        u = x[len(phases) * N:len(phases) * N + S] * scale
        g_end = x[len(phases) * N + S:] * scale
        gas = {gs.key: float(v) for gs, v in zip(self._gases_now, g_end)}
        if self._gas_mode == "open":
            p_out = {k: float(v) for k, v in p_gas.items()}
        elif self._gas_mode == "closed":
            ntot = float(n_air) + float(np.sum(g_end))
            p_out = {gs.key: float(P_atm * v / ntot) for gs, v in zip(self._gases_now, g_end)}
        else:
            p_out = {}
        si = self._si_liquids(liquids, ln_rh)
        self._enabled_keys = {s_.key for s_ in self._solids_now}
        thresh = 1e-7 * scale
        solid_amounts = {s.key: float(v) for s, v in zip(self._solids_now, u) if v > thresh}
        F = self._objective(x, len(phases), ln_rh, act)[0]
        checks = self._checks(liquids, ln_rh, si, solid_amounts, b * scale, u, g_end, p_out)
        checks["n_absent_entries"] = float(sum(int(np.sum(~np.isfinite(L.ln_a))) for L in liquids))
        checks["max_removed_trace"] = self._trunc_max
        checks["n_line_search_failures"] = float(self._ls_failures)
        liquids = [self._to_components(L) for L in liquids]          # largest removed amount, relative to sum |feed|
        if dry:
            # all non-water material is in solids; the remaining "liquid" is a numerical remnant of the barrier
            si = self._si_liquids(liquids, ln_rh)
            return PhaseEquilibriumResult("dry", self.T, rh, [], solid_amounts, si, F * scale,
                                          {"max_mass_balance_residual": checks["max_mass_balance_residual"]},
                                          0.0, n_outer, message="no liquid phase (all solutes crystalline)",
                                          names=self.names, gas=gas, p_gas=p_out)
        # the status is decided by the equilibrium conditions of the final state, not by how the iteration ended
        status, msg = "converged", ""
        bad = [k for k, lim in (("max_abs_ln_aw_minus_ln_rh", 1e-5), ("max_neutral_mu_spread", 1e-4),
                                ("max_ion_mu_residual", 1e-4), ("max_si", 1e-4), ("max_abs_si_present_solids", 1e-4),
                                ("max_abs_gas_residual", 1e-4))
               if checks[k] > lim]
        if bad:
            status = "not_converged"
            msg = "equilibrium conditions violated: " + ", ".join(bad)
        if tpd_min < -self.tol_tpd:
            status = "not_converged"
            msg += ("; " if msg else "") + f"stability test still negative (TPD={tpd_min:.2e})"
        if not inner_ok:
            msg += ("; " if msg else "") + "inner Newton iteration stopped early (line search)"
        result = PhaseEquilibriumResult(status, self.T, rh, liquids, solid_amounts, si, F * scale, checks, tpd_min,
                                        n_outer, message=msg, names=self.names, gas=gas, p_gas=p_out)
        if self._gas_mode == "open":
            result = self._complete_evaporation(result, feed, rh, solids, p_gas, max_liquids, max_outer, verbose)
        return result

    def _complete_evaporation(self, res, feed, rh, solids, p_gas, max_liquids, max_outer, verbose):
        """Open system: a species that has (practically) all left the particle through a gas is removed exactly.

        When the amount of an ion left in all liquids is below 1e-9 of the particle's ion content and the ion belongs
        to a gas that is releasing it, the remainder is assigned to the gas as well, the feed is reduced by the total
        release and the problem is solved again without that ion and that gas.  The barrier would otherwise keep a
        trace whose potentials fail the equilibrium checks.  The removed gas is then undersaturated, i.e. the particle's
        equilibrium pressure is below the reservoir pressure (reported in ``checks``)."""
        if not res.liquids or not res.gas:
            return res
        names = res.names
        liq_tot = sum(L.amounts for L in res.liquids)
        nn = 1 + sum(1 for n in names[1:] if n not in ION_REGISTRY)
        ion_scale = float(np.sum(np.abs(liq_tot[nn:]))) + sum(abs(v) for v in res.gas.values())
        for gk, g in res.gas.items():
            if g <= 0.0:
                continue
            gas = GASES[gk]
            for ion, nu in gas.ions.items():
                if nu <= 0 or ion not in names:
                    continue
                left = float(liq_tot[names.index(ion)]) + sum(SOLIDS[k].ions.get(ion, 0) * v for k, v in res.solids.items())
                if left > 1e-9 * ion_scale:
                    continue
                extra = left / nu
                release = g + extra
                new_feed = dict(feed)
                for i2, nu2 in gas.ions.items():
                    new_feed[i2] = new_feed.get(i2, 0.0) - nu2 * release
                new_feed[ion] = 0.0
                for k in list(new_feed):
                    if abs(new_feed[k]) < 1e-14 * max(1.0, ion_scale):
                        new_feed[k] = 0.0
                rest = {k: v for k, v in p_gas.items() if k != gk}
                sub = self.solve(new_feed, rh, solids=solids, p_gas=rest or None, max_liquids=max_liquids,
                                 max_outer=max_outer, verbose=verbose)
                # undersaturation of the removed gas in the final state (gauge-free combination of the remaining ions)
                ok_ions = all(i2 in sub.names for i2, nu2 in gas.ions.items() if i2 != ion)
                sub.gas = dict(sub.gas)
                for k2, v2 in res.gas.items():
                    if k2 != gk and k2 not in sub.gas:
                        sub.gas[k2] = 0.0
                sub.gas[gk] = release + sub.gas.get(gk, 0.0)
                sub.p_gas = dict(sub.p_gas); sub.p_gas[gk] = p_gas[gk]
                sub.checks = dict(sub.checks)
                sub.checks[f"{gk}_fully_released"] = 1.0 if ok_ions else 0.0
                sub.message = ((sub.message + "; ") if sub.message else "") + \
                    f"{ion} left the particle completely as {gk} ({release:.4g} mol)"
                return sub
        return res

    def _warm_start(self, init, b, scale, rh, V, G, S, K, use_barrier):
        """Starting liquids, solid and gas amounts (scaled) from a previous result, or None if it does not fit."""
        if use_barrier:                                          # the barrier solver needs every solid > 0
            return None
        if init is None or not getattr(init, "liquids", None) or list(init.names) != list(self.names):
            return None
        if abs(init.T_K - self.T) > 1e-9:
            return None
        phases = []
        for L in init.liquids:
            n = np.array(L.species_amounts if L.species_amounts is not None else L.amounts, dtype=float)
            if len(n) != self.N:
                return None
            phases.append(n / scale)
        u0 = np.zeros(S)
        solid_act = np.zeros(S, dtype=bool)
        for j, sld in enumerate(self._solids_now):
            v = init.solids.get(sld.key, 0.0) / scale
            if v > 0:
                u0[j], solid_act[j] = v, True
        g0 = np.array([init.gas.get(gs.key, 0.0) / scale for gs in self._gases_now]) if K else np.zeros(0)
        if self._gas_mode == "closed" and K and np.any(g0 <= 0):
            return None
        # the start must carry the feed: component totals of liquids + solids (+ closed gas; open gas is released)
        tot = self.E @ sum(phases) + (V @ u0 if S else 0.0)
        if K:
            tot = tot + (G @ g0)                                 # open: released amount, closed: gas amount
        if float(np.max(np.abs(tot[1:] - b[1:]))) > 1e-8:
            return None
        free = self.lm.free
        # liquids smaller than 1 % of the largest are merged into it: such a liquid (e.g. a small salt liquid) may
        # disappear at the new RH, and its water cannot always be adjusted; the stability test re-creates it if needed
        sizes = [float(np.sum(np.abs(n[~free]))) for n in phases]
        big = int(np.argmax(sizes))
        merged = [phases[big].copy()]
        for k, n in enumerate(phases):
            if k == big:
                continue
            if sizes[k] < 0.01 * sizes[big]:
                merged[0] = merged[0] + n
            else:
                merged.append(n.copy())
        phases = merged
        out = []
        for n in phases:
            n = n.copy()
            size = float(np.sum(np.abs(n[~free])))
            n[~free] = np.maximum(n[~free], 1e-20 * size)       # removed traces start again as traces
            n[0] = self._water_guess_species(n, rh)
            out.append(n)
        return out, u0, g0, solid_act

    def _water_guess_species(self, n: np.ndarray, rh: float) -> float:
        """Water amount of a liquid (species amounts n) at roughly a_w = RH: bisection on ln n_w."""
        lw0 = math.log(max(n[0], 1e-300))
        f = lambda lw: self.lm.ln_a(np.r_[math.exp(lw), n[1:]], self.T)[0] - math.log(rh)
        try:
            lo, hi = lw0 - 5.0, lw0 + 5.0
            flo, fhi = f(lo), f(hi)
            if flo > 0 or fhi < 0:
                return n[0]
            for _ in range(30):
                mid = 0.5 * (lo + hi)
                if f(mid) > 0:
                    hi = mid
                else:
                    lo = mid
            return math.exp(0.5 * (lo + hi))
        except (ValueError, FloatingPointError, OverflowError, ZeroDivisionError):
            return n[0]

    def _to_components(self, L: LiquidPhase) -> LiquidPhase:
        """Liquid in the component basis (the species, if explicit, are kept as extra fields)."""
        if self.speciation != "explicit":
            return L
        return LiquidPhase(self.E @ L.amounts, L.ln_a[:self.Nc].copy(), list(self.names),
                           species_names=list(self.sp_names), species_amounts=L.amounts.copy(),
                           species_ln_a=L.ln_a.copy())

    def _initial_gas(self, b, vu, G, gt):
        """Initial gas amounts that leave every bounded species of the liquid strictly positive, so that species absent
        from the feed (e.g. NO3- and H+ that only enter by HNO3 uptake) need no artificial floor and the mass balance
        stays exact.  Two linear programs: (1) the largest margin t* with  b_i - vu_i - (G g)_i >= t  (bounded i);
        (2) among the g that keep a margin min(t*, 1e-6 |b|), the one closest (L1) to the reference state -- gases
        mostly in the gas phase (closed system) or no net transfer (open system) -- so the start is not an extreme
        composition."""
        from scipy.optimize import linprog
        K = G.shape[1]
        rows = [i for i in range(1, self.Nc) if not self._comp_free[i]]
        cap = float(np.abs(b).sum()) or 1.0
        Gr = G[rows]
        rhs = b[rows] - vu[rows]
        if self._gas_mode == "closed":
            bounds_g = [(1e-9 * cap, cap)] * K
            g_ref = np.maximum(gt, 1e-9 * cap) * 0.999
        else:
            bounds_g = [(-cap, cap)] * K
            g_ref = np.zeros(K)
        try:
            c = np.zeros(K + 1); c[-1] = -1.0
            r1 = linprog(c, A_ub=np.hstack([Gr, np.ones((len(rows), 1))]), b_ub=rhs,
                         bounds=bounds_g + [(None, 1e-3 * cap)], method="highs")
            if r1.status != 0 or r1.x[-1] <= 0:
                raise ValueError
            t = min(float(r1.x[-1]), 1e-6 * cap)
            # variables g (K) and e (K) with  e >= |g - g_ref| ; minimize sum e
            A2 = [np.hstack([Gr, np.zeros((len(rows), K))])]
            b2 = [rhs - t]
            I = np.eye(K)
            A2 += [np.hstack([I, -I]), np.hstack([-I, -I])]
            b2 += [g_ref, -g_ref]
            c2 = np.concatenate([np.zeros(K), np.ones(K)])
            r2 = linprog(c2, A_ub=np.vstack(A2), b_ub=np.concatenate(b2), bounds=bounds_g + [(0, None)] * K,
                         method="highs")
            if r2.status == 0:
                return r2.x[:K]
            return r1.x[:K]
        except ValueError:
            # fallback (a species cannot be supplied by any gas): reference state; absent species get a floor
            return g_ref

    def _reference_potentials(self, liquids, ln_rh):
        """Equilibrium potentials for the stability test: each neutral from the liquid that holds most of it (a
        species squeezed to a trace amount by the barrier has a depressed ln a), all ions from the liquid with the
        largest ion content (one consistent electrical gauge), water at ln RH."""
        nn = self.lm.n_neutral
        mu = np.empty(self.N)
        mu[0] = ln_rh
        for i in range(1, nn):
            mu[i] = max(liquids, key=lambda L: L.amounts[i]).ln_a[i]
        if self.N > nn:
            ref = max(range(len(liquids)), key=lambda a: float(np.sum(np.abs(liquids[a].amounts[nn:]))))
            mu[nn:] = liquids[ref].ln_a[nn:]
            missing = [i for i in range(nn, self.N) if not np.isfinite(mu[i])]
            if missing:                                    # ion absent from the reference liquid: lambda_i + z_i psi_ref
                lam, psi = self._ion_gauge_fit(liquids)
                for i in missing:
                    mu[i] = lam[i - nn] + self.z[i] * psi[ref]
        return mu

    def _ion_gauge_fit(self, liquids):
        """Least-squares fit of ln a_ai = lambda_i + z_i psi_a over the ion entries present in the liquids (psi of the
        first liquid fixed at 0)."""
        nn, K = self.lm.n_neutral, len(liquids)
        ni = self.N - nn
        rows, rhs = [], []
        for a, L in enumerate(liquids):
            for j in range(ni):
                v = L.ln_a[nn + j]
                if np.isfinite(v):
                    r = np.zeros(ni + K)
                    r[j] = 1.0
                    r[ni + a] = self.z[nn + j]
                    rows.append(r); rhs.append(v)
        r = np.zeros(ni + K); r[ni] = 1.0
        rows.append(r); rhs.append(0.0)
        sol = np.linalg.lstsq(np.array(rows), np.array(rhs), rcond=None)[0]
        return sol[:ni], sol[ni:]

    # ---------------------------------------------------------------------------------------------------
    def _checks(self, liquids, ln_rh, si, solid_amounts, b, u, g=None, p_out=None) -> dict:
        nn = self.lm.n_neutral
        c = {}
        c["max_abs_ln_aw_minus_ln_rh"] = max(abs(L.ln_a[0] - ln_rh) for L in liquids)
        # neutral organics: equal ln a in all liquids
        dn = 0.0
        n_trace = 0
        for i in range(1, nn):
            vals = [L.ln_a[i] for L in liquids if L.mole_fractions[i] > 1e-8]
            n_trace += sum(1 for L in liquids if L.mole_fractions[i] <= 1e-8)
            if len(vals) > 1:
                dn = max(dn, max(vals) - min(vals))
        c["max_neutral_mu_spread"] = dn
        c["n_trace_entries"] = float(n_trace)
        # ions: ln a_ai = lambda_i + z_i psi_a  -> fit lambda, psi by least squares, report residual
        ion_idx = [i for i in range(nn, self.N)]
        if ion_idx and len(liquids) > 1:
            K = len(liquids)
            rows, rhs = [], []
            for a, L in enumerate(liquids):
                for j, i in enumerate(ion_idx):
                    if not np.isfinite(L.ln_a[i]):            # ion absent from this liquid (trace removed)
                        continue
                    r = np.zeros(len(ion_idx) + K)
                    r[j] = 1.0
                    r[len(ion_idx) + a] = self.z[i]
                    rows.append(r); rhs.append(L.ln_a[i])
            sol, *_ = np.linalg.lstsq(np.array(rows), np.array(rhs), rcond=None)
            c["max_ion_mu_residual"] = float(np.max(np.abs(np.array(rows) @ sol - np.array(rhs))))
        else:
            c["max_ion_mu_residual"] = 0.0
        enabled = [v for k, v in si.items() if k in self._enabled_keys]
        c["max_si"] = max(enabled) if enabled else float("-inf")          # solids allowed in this solve
        c["max_si_all_candidates"] = max(si.values()) if si else float("-inf")   # > 0 in metastable solves
        present_si = [si[k] for k in solid_amounts]
        c["max_abs_si_present_solids"] = max((abs(v) for v in present_si), default=0.0)
        c["max_charge_residual"] = max(abs(float(np.dot(self.z, L.amounts))) for L in liquids)
        tot = self.E @ sum(L.amounts for L in liquids)          # species -> components
        V = self._solid_columns(self._solids_now)
        tot = tot + V @ (u if len(u) else np.zeros(0)) if V.shape[1] else tot
        G = self._columns(self._gases_now)
        if G.shape[1] and g is not None:
            tot = tot + G @ g if self._gas_mode == "closed" else tot
        bref = b.copy()
        if self._gas_mode == "open" and G.shape[1] and g is not None:
            bref = b - G @ g                                       # open: released material leaves the system
        c["max_mass_balance_residual"] = float(np.max(np.abs(tot[1:] - bref[1:]))) / max(float(np.abs(b[1:]).sum()), 1e-300)
        # gas-liquid equilibrium: sum nu ln a + h ln RH = ln K + ln p
        gr = 0.0
        for gs in self._gases_now:
            p = (p_out or {}).get(gs.key, 0.0)
            if p <= 0:
                continue
            for L in liquids:                                 # every liquid that holds the gas's ions
                lhs = sum(nu * L.ln_a[self.names.index(ion)] for ion, nu in gs.ions.items()) + gs.h * ln_rh
                if np.isfinite(lhs):
                    gr = max(gr, abs(lhs - gs.ln_k(self.T) - math.log(p)))
        c["max_abs_gas_residual"] = gr
        return c

    # ---------------------------------------------------------------------------------------------------
    def drying_path(self, feed: dict, rh_grid: Sequence[float], *, ln_s_crit: float | dict = 0.0,
                    verbose: bool = False, warm: bool = True, **gas_kw) -> list:
        """Decreasing-RH path with crystallization only after a critical supersaturation.

        Starting from the highest RH without solids, a solid becomes a candidate once its saturation index in the
        (metastable) liquid exceeds ``ln_s_crit`` (a float or a dict by solid key) and stays a candidate at all lower
        RH.  ``ln_s_crit = 0`` reproduces the equilibrium path; a large value gives the fully metastable path."""
        crit = (lambda k: ln_s_crit.get(k, 0.0)) if isinstance(ln_s_crit, dict) else (lambda k: float(ln_s_crit))
        enabled: list[str] = []
        out = []
        prev = None
        for rh in sorted(rh_grid, reverse=True):
            res = self.solve(feed, rh, solids=list(enabled) if enabled else "none", verbose=verbose,
                             init=prev if warm else None, **gas_kw)
            new = [k for k, v in res.si.items() if k not in enabled and v > crit(k)]
            if new:
                enabled += new
                res = self.solve(feed, rh, solids=list(enabled), verbose=verbose, init=res if warm else None, **gas_kw)
            res.message = f"enabled solids: {enabled}"
            out.append(res)
            prev = res if res.status in ("converged", "not_converged") else None
        return out

    def rh_scan(self, feed: dict, rh_grid: Sequence[float], *, warm: bool = True, verbose: bool = False,
                **kw) -> list:
        """Results along ``rh_grid`` (in the given order), each solve warm-started from the previous result
        (``warm=False``: every solve from scratch).  Other keywords are passed to :meth:`solve`."""
        out, prev = [], None
        for rh in rh_grid:
            res = self.solve(feed, rh, verbose=verbose, init=prev if warm else None, **kw)
            out.append(res)
            prev = res if res.liquids else None
        return out

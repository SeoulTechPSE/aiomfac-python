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
        if self._acid and smc[self._ih] > 0.0 and sma[self._iso] > 0.0:
            # stoichiometric H+ and SO4-- -> free H+, HSO4-, SO4-- at the bisulfate equilibrium of this phase
            r = solve_bisulfate(mod, T, xn, smc, sma, self._ih, self._ihs, self._iso)
            smc[self._ih], sma[self._ihs], sma[self._iso] = r.m_h, r.m_hso4, r.m_so4
        x = mod._x_from_molalities(xn, smc, sma)
        lr, mr, sr = mod.lr_mr_sr(T, smc, sma, xn, x)
        out = np.empty(self.N)
        out[:nn] = lr.ln_gamma_neutral[:nn] + mr.ln_gamma_neutral[:nn] + sr.ln_gamma_sr[:nn] + np.log(x[:nn])
        nc = self._nc
        for k, (is_cat, idx) in enumerate(self._pos):
            if is_cat:
                lg = mr.ln_gamma_cation[idx] + sr.ln_gamma_sr[nn + idx] + lr.ln_gamma_cation[idx] - mr.tmolal
                out[nn + k] = lg + math.log(smc[idx])
            else:
                lg = mr.ln_gamma_anion[idx] + sr.ln_gamma_sr[nn + nc + idx] + lr.ln_gamma_anion[idx] - mr.tmolal
                out[nn + k] = lg + math.log(sma[idx])
        return out

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
        for _ in range(60):
            def f(lnh):
                h = math.exp(lnh)
                den = 1.0 + A1 / h + A1 * A2 / (h * h)
                val = h - Aw / h + CT * (A1 / h + 2.0) / den - P
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

    def hessian(self, n: np.ndarray, T: float, rel: float = 1.0e-5) -> np.ndarray:
        """d ln a_i / d n_j by central differences, symmetrized."""
        n = np.asarray(n, dtype=float)
        H = np.empty((self.N, self.N))
        floor = 1.0e-10 * float(np.sum(np.abs(n)))
        for j in range(self.N):
            # bounded species: a step proportional to the amount keeps the perturbed state positive; only the
            # sign-free proton excess (carbonate) needs an absolute floor
            h = rel * max(abs(n[j]), floor) if self.free[j] else rel * abs(n[j])
            p = n.copy(); p[j] += h
            m = n.copy(); m[j] -= h
            H[:, j] = (self.ln_a(p, T) - self.ln_a(m, T)) / (2.0 * h)
        return 0.5 * (H + H.T)


# ============================================================================================================
# Results
# ============================================================================================================

@dataclass
class LiquidPhase:
    amounts: np.ndarray            # mol of each species (water, organics, ions)
    ln_a: np.ndarray
    names: list

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
                 k_mode: str | None = None, solid_keys: Sequence[str] | None = None):
        self.lm = LiquidModel(organics, ions)
        self._organics, self._ions, self._solid_keys = list(organics), list(ions), solid_keys
        self._children: dict = {}
        self.T = float(T_K)
        self.names = self.lm.names
        self.N = self.lm.N
        self.z = self.lm.z
        if solid_keys is None:
            # hydroxide solids carry OH- as -H+ and are meaningful only when H+ is the proton excess (carbonate system)
            cand = [s for s in SOLIDS.values() if set(s.ions) <= set(ions)
                    and (not getattr(s, "n_oh", 0) or self.lm._carb)
                    and not any(i in _UNSUPPORTED_IONS for i in s.ions)]
        else:
            cand = [SOLIDS[k] for k in solid_keys]
        self.all_solids: list[Solid] = [s for s in cand if s.kspec is None or True]
        self.k_mode = k_mode
        self._lnk = {s.key: s.ln_k(self.T, k_mode) for s in self.all_solids}
        self.tol_tpd = 1.0e-7

    # ---------------------------------------------------------------------------------------------------
    def _solid_columns(self, solids: list[Solid]) -> np.ndarray:
        return self._columns(solids)

    def _solid_cost(self, solids: list[Solid], ln_rh: float) -> np.ndarray:
        return np.array([self._lnk[s.key] - s.h_eff * ln_rh for s in solids])

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
        V = np.zeros((self.N, len(items)))
        for j, s in enumerate(items):
            for ion, nu in s.ions.items():
                V[self.names.index(ion), j] = nu
        return V

    def _build_A(self, n_liq: int):
        N = self.N
        W = self._W                                              # solid + gas columns
        nx = n_liq * N + W.shape[1]
        rows = []
        for i in range(1, N):                                     # mass balance of non-water species
            r = np.zeros(nx)
            for a in range(n_liq):
                r[a * N + i] = 1.0
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

    def _objective(self, x, n_liq, ln_rh):
        N, S = self.N, len(self._solids_now)
        F = 0.0
        lna = []
        for a in range(n_liq):
            n = x[a * N:(a + 1) * N]
            la = self.lm.ln_a(n, self.T)
            lna.append(la)
            F += float(np.dot(n, la)) - n[0] * ln_rh
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

    def _barrier_solve(self, x, n_liq, ln_rh, *, mu0=1.0e-3, mu_min=1.0e-14, max_newton=60, verbose=False):
        """Minimize F - mu sum ln x_b over {A x = A x0} (x_b: bounded entries) from the strictly feasible x.

        A step is accepted when it decreases the barrier function (Armijo) or, failing that, the norm of the reduced
        gradient: AIOMFAC's carbonate treatment (CO2(aq) with its own salting-out activity coefficient) makes the
        stoichiometric potentials only approximately the gradient of one Gibbs function, and the equilibrium
        conditions themselves are what the final checks verify."""
        N, S = self.N, len(self._solids_now)
        A = self._build_A(n_liq)
        Z = null_space(A) if A.size else np.eye(len(x))
        free = self._free_mask(n_liq)
        bnd = ~free
        mu = mu0
        converged = True

        def evaluate(xx, mu_):
            try:
                with np.errstate(all="ignore"):
                    F, lna, gg, Hg = self._objective(xx, n_liq, ln_rh)
            except (ValueError, OverflowError, FloatingPointError, ZeroDivisionError):
                return float("inf"), np.full(len(xx), np.nan), None
            gfull = self._grad(lna, ln_rh, gg)
            gfull[bnd] -= mu_ / xx[bnd]
            phi = F - mu_ * float(np.sum(np.log(xx[bnd])))
            return phi, gfull, Hg

        while True:
            for it in range(max_newton):
                phi0, gvec, Hg = evaluate(x, mu)
                H = np.zeros((len(x), len(x)))
                for a in range(n_liq):
                    H[a * N:(a + 1) * N, a * N:(a + 1) * N] = self.lm.hessian(x[a * N:(a + 1) * N], self.T)
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
                        phit, gt, _ = evaluate(xt, mu)
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
                    break
                x = xt
            else:
                converged = False
            if verbose:
                print(f"    mu={mu:.1e} it={it} dec={dec:.2e}")
            if mu <= mu_min:
                break
            mu = max(mu * 0.1, mu_min)
        return x, converged

    # ---------------------------------------------------------------------------------------------------
    # Starting point, stability test, phase bookkeeping
    # ---------------------------------------------------------------------------------------------------
    def _water_guess(self, n: np.ndarray, rh: float) -> float:
        """Water amount for a single liquid at roughly a_w = RH (bisection on ln n_w)."""
        dry = n[1:].sum()
        lo, hi = math.log(1e-6 * dry), math.log(1e4 * dry)
        f = lambda lw: self.lm.ln_a(np.r_[math.exp(lw), n[1:]], self.T)[0] - math.log(rh)
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

    def _tpd_minimize(self, mu_eq: np.ndarray, w0: np.ndarray):
        """min_w sum w_i (ln a_i(w) - mu_i) over electroneutral w with sum of the bounded entries = 1 (barrier Newton
        from w0; the proton excess of a carbonate system is a free entry)."""
        N = self.N
        free = self.lm.free
        bnd = ~free
        rows = [bnd.astype(float)]
        if np.any(self.z != 0):
            rows.append(self.z)
        A = np.array(rows)
        Z = null_space(A)
        w = np.asarray(w0, dtype=float) / float(np.sum(w0[bnd]))
        tpd = lambda w: float(np.dot(w, self.lm.ln_a(w, self.T) - mu_eq))
        mu = 1.0e-4
        for _ in range(9):
            for it in range(40):
                la = self.lm.ln_a(w, self.T)
                f = float(np.dot(w, la - mu_eq))
                g = la - mu_eq
                g[bnd] -= mu / w[bnd]
                H = self.lm.hessian(w, self.T)
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
                        phit = tpd(wt) - mu * float(np.sum(np.log(wt[bnd])))
                        if np.isfinite(phit) and phit <= phi0 - 1e-4 * step * dec:
                            ok = True
                            break
                    step *= 0.5
                if not ok:
                    break
                w = wt
            mu *= 0.1
        return w, tpd(w)

    def _trial_points(self, b: np.ndarray, rh: float, liquids: list) -> list:
        """Electroneutral trial compositions (sum 1) for the stability test.  Compositions are fixed mole-fraction
        patterns, independent of RH (an RH-scaled water content would make "organic-rich" trials water-rich close
        to saturation): each organic at x = 0.3, 0.7 and 0.95 in water; all organics in feed proportion at x = 0.5 and
        0.9; salt solutions at two water-to-ion ratios; dilute water; perturbed copies of the current liquids."""
        N, nn = self.N, self.lm.n_neutral
        eps = 1e-6
        ion_fr = np.zeros(N)
        if np.abs(b[nn:]).sum() > 0:
            ion_fr[nn:] = np.abs(b[nn:]) / np.abs(b[nn:]).sum()
        org_fr = np.zeros(N)
        if b[1:nn].sum() > 0:
            org_fr[1:nn] = b[1:nn] / b[1:nn].sum()
        trials = []

        def make(water, org, ions):
            w = np.full(N, eps)
            w[0] = water
            w[1:nn] += org[1:nn]
            w[nn:] = np.maximum(ions[nn:], eps * ion_fr[nn:] + 1e-12)
            return w

        for k in range(1, nn):
            for xo in (0.3, 0.7, 0.95):
                o = np.zeros(N); o[k] = xo
                trials.append(make(1.0 - xo, o, eps * ion_fr))
        if nn > 2:
            for xo in (0.5, 0.9):
                trials.append(make(1.0 - xo, xo * org_fr, eps * ion_fr))
        if ion_fr[nn:].sum() > 0:
            for water in (2.0, 10.0):                                # concentrated and moderate salt solutions
                trials.append(make(water, eps * org_fr, ion_fr))
        trials.append(make(1.0, eps * org_fr, eps * ion_fr))      # dilute water
        rng = np.random.default_rng(12345)
        for L in liquids:                                           # perturbed copies of the current liquids
            trials.append(np.maximum(np.abs(L.amounts) / L.total, 1e-9) * np.exp(0.3 * rng.standard_normal(N) * (self.z == 0)))
        out = []
        for w in trials:
            w = np.maximum(w, 1e-12)
            if self.lm._carb:                                       # neutralize with the free proton excess
                kh = self.lm._kh
                w[kh] = 0.0
                w[kh] = -float(np.dot(self.z, w))
                out.append(w / float(np.sum(w[~self.lm.free])))
                continue
            if np.any(self.z != 0):                                 # restore electroneutrality by scaling the anions
                qp = float(np.sum(np.where(self.z > 0, self.z * w, 0.0)))
                qn = float(-np.sum(np.where(self.z < 0, self.z * w, 0.0)))
                if qn > 0:
                    w = np.where(self.z < 0, w * qp / qn, w)
            out.append(w / w.sum())
        return out

    # ---------------------------------------------------------------------------------------------------
    def solve(self, feed: dict, rh: float, *, solids="all", p_gas: dict | None = None, gas_total: dict | None = None,
              n_air: float | None = None, P_atm: float = 1.0, max_liquids: int = 3, max_outer: int = 6,
              verbose: bool = False) -> PhaseEquilibriumResult:
        """Equilibrium for the non-water ``feed`` [mol] (organic names and ion keys) at relative humidity ``rh``.

        Gases (keys of :data:`aiomfac_py.gases.GASES`: NH3, HNO3, HCl, CO2) are optional:
          * ``p_gas={"HCl": 1e-9}``: open system, the particle exchanges with a reservoir of fixed partial pressure [atm];
            ``result.gas`` is the net amount released (negative = uptake).
          * ``gas_total={"HCl": 0.0}, n_air=...``: closed system with ``n_air`` mol of inert air at ``P_atm``; the
            totals of the volatile species (particle + gas) are conserved and ``result.gas`` holds the gas amounts.
        A carbonate system (``CO3--`` with ``H+`` as proton excess) can be given ``"CO2"`` in either mode."""
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
                child = PhaseEquilibrium(orgs, ions_kept, self.T, k_mode=self.k_mode, solid_keys=sk)
                child.tol_tpd = self.tol_tpd
                self._children[key] = child
            sub_feed = {k: v for k, v in feed.items() if k in present}
            if isinstance(solids, (list, tuple)):
                solids = [k for k in solids if set(SOLIDS[k].ions) <= set(ions_kept)] or "none"
            res = self._children[key].solve(sub_feed, rh, solids=solids, p_gas=p_gas, gas_total=gas_total, n_air=n_air,
                                            P_atm=P_atm, max_liquids=max_liquids, max_outer=max_outer, verbose=verbose)
            dropped = [n for n in self.names[1:] if n not in res.names]
            res.message = (res.message + "; " if res.message else "") + f"species absent from the problem: {dropped}"
            return res
        if p_gas and gas_total:
            raise ValueError("give either p_gas (open system) or gas_total (closed system), not both")
        ln_rh = math.log(rh)
        N, nn = self.N, self.lm.n_neutral
        b = np.zeros(N)
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
        if abs(float(np.dot(self.z, b))) > 1e-9 * max(1.0, float(np.abs(b).sum())):
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
        u0 = np.full(S, 1e-6)
        g0 = self._initial_gas(b, V @ u0, G, gt / scale) if K else np.zeros(0)
        n0 = b.copy()
        n0[1:] = b[1:] - V[1:] @ u0 - G[1:] @ g0
        floor = 1e-7
        n0[1:] = np.where(free[1:], n0[1:], np.maximum(n0[1:], floor))
        if np.any(self.z != 0) and np.any(b[nn:] != 0):           # re-neutralize after flooring
            if self.lm._carb:
                n0[self.lm._kh] -= float(np.dot(self.z, n0))
            else:
                qp = float(np.sum(np.where(self.z > 0, self.z * n0, 0.0)))
                qn = float(-np.sum(np.where(self.z < 0, self.z * n0, 0.0)))
                n0 = np.where(self.z < 0, n0 * qp / qn, n0)
        n0[0] = self._water_guess(n0, rh)
        phases = [n0]
        x = np.concatenate(phases + [u0, g0])
        n_outer = 0
        tpd_min = float("nan")
        inner_ok = True
        history = []
        while True:
            n_outer += 1
            n_liq = len(phases)
            x, conv = self._barrier_solve(x, n_liq, ln_rh, verbose=verbose)
            inner_ok = inner_ok and conv
            phases = [x[a * N:(a + 1) * N] for a in range(n_liq)]
            u = x[n_liq * N:]
            # remove vanished liquids, merge identical ones
            size = lambda p: float(np.sum(p[~free]))
            keep = [a for a in range(n_liq) if size(phases[a]) > 1e-8]
            if not keep:                                       # never drop the last liquid (see the "dry" status)
                keep = [int(np.argmax([size(p) for p in phases]))]
            merged = []
            for a in keep:
                xa = phases[a] / size(phases[a])
                for m in merged:
                    if np.max(np.abs(xa - phases[m] / size(phases[m]))) < 1e-4:
                        phases[m] = phases[m] + phases[a]
                        break
                else:
                    merged.append(a)
            if len(merged) < n_liq:
                phases = [phases[a] for a in merged]
                x = np.concatenate(phases + [u])
                x, conv = self._barrier_solve(x, len(phases), ln_rh, verbose=verbose)
                n_liq = len(phases)
                phases = [x[a * N:(a + 1) * N] for a in range(n_liq)]
                u = x[n_liq * N:]
            liquids = [LiquidPhase(p.copy(), self.lm.ln_a(p, self.T), self.names) for p in phases]
            dry = len(phases) == 1 and size(phases[0]) < 1e-6
            if dry:                                            # everything crystallized: no liquid to test
                tpd_min = 0.0
                break
            # stability test of the liquid set (potentials of the largest liquid; water at ln RH)
            mu_eq = self._reference_potentials(liquids, ln_rh)
            best = (0.0, None)
            for w0 in self._trial_points(b, rh, liquids):
                try:
                    with np.errstate(all="ignore"):
                        w, t = self._tpd_minimize(mu_eq, w0)
                except (ValueError, FloatingPointError, np.linalg.LinAlgError, ZeroDivisionError):
                    continue
                if not (np.all(np.isfinite(w)) and np.isfinite(t) and np.all(w > 0)):
                    continue
                dist = min(float(np.max(np.abs(w - L.mole_fractions))) for L in liquids)
                if t < best[0] and dist > 1e-3:
                    best = (t, w)
            tpd_min = best[0]
            if verbose:
                print(f"  outer {n_outer}: {n_liq} liquid(s), TPD_min={tpd_min:.3e}")
            if best[1] is None or tpd_min > -self.tol_tpd or n_liq >= max_liquids or n_outer >= max_outer:
                break
            # an added phase that merges back gives the same phase count and the same TPD again: stop (near-critical
            # split of two similar liquids); the result is then reported as not converged with this TPD
            if (n_liq, round(tpd_min, 9)) in history:
                break
            history.append((n_liq, round(tpd_min, 9)))
            # add the trial phase: take a small neutral amount of w out of the liquid that can supply the most
            w = best[1]
            nonw = w.copy(); nonw[0] = 0.0
            # species that are traces in the trial composition do not limit the amount moved into the new phase
            major = (nonw > 1e-4 * float(np.max(np.abs(nonw[~free])))) & ~free
            major[0] = False
            cand = []
            for a, p in enumerate(phases):
                theta = 0.5 * float(np.min(p[major] / nonw[major])) if np.any(major) else 0.0
                cand.append((theta, a))
            theta, a = max(cand)
            donor = phases[a]
            tr = theta * nonw
            minor = ~major & ~free
            minor[0] = False
            tr[minor] = np.minimum(tr[minor], 0.5 * donor[minor])
            q = float(np.dot(self.z, tr))                      # keep the transferred amount electroneutral
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
                phases[a] = donor - tr
                phases.append(newp)
            else:                                              # fall back to the strictly proportional transfer
                theta = min(theta, 0.5 * float(np.min(donor[~free & (nonw > 0)] / nonw[~free & (nonw > 0)])))
                newp = theta * w.copy()
                phases[a] = donor - theta * nonw
                phases.append(newp)
            x = np.concatenate(phases + [u])

        liquids = [LiquidPhase(p * scale, self.lm.ln_a(p, self.T), self.names) for p in phases]
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
        si = self.si_of(liquids[0].ln_a, ln_rh)
        self._enabled_keys = {s_.key for s_ in self._solids_now}
        thresh = 1e-7 * scale
        solid_amounts = {s.key: float(v) for s, v in zip(self._solids_now, u) if v > thresh}
        F = self._objective(x, len(phases), ln_rh)[0]
        checks = self._checks(liquids, ln_rh, si, solid_amounts, b * scale, u, g_end, p_out)
        if dry:
            # all non-water material is in solids; the remaining "liquid" is a numerical remnant of the barrier
            si = self.si_of(liquids[0].ln_a, ln_rh)
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

    def _initial_gas(self, b, vu, G, gt):
        """Initial gas amounts that leave every bounded species of the liquid strictly positive, so that species absent
        from the feed (e.g. NO3- and H+ that only enter by HNO3 uptake) need no artificial floor and the mass balance
        stays exact.  Two linear programs: (1) the largest margin t* with  b_i - vu_i - (G g)_i >= t  (bounded i);
        (2) among the g that keep a margin min(t*, 1e-6 |b|), the one closest (L1) to the reference state -- gases
        mostly in the gas phase (closed system) or no net transfer (open system) -- so the start is not an extreme
        composition."""
        from scipy.optimize import linprog
        K = G.shape[1]
        rows = [i for i in range(1, self.N) if not self.lm.free[i]]
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
            Li = max(liquids, key=lambda L: float(np.sum(L.amounts[nn:])))
            mu[nn:] = Li.ln_a[nn:]
        return mu

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
        tot = sum(L.amounts for L in liquids)
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
        L0 = liquids[0]
        for gs in self._gases_now:
            lhs = sum(nu * L0.ln_a[self.names.index(ion)] for ion, nu in gs.ions.items()) + gs.h * ln_rh
            p = (p_out or {}).get(gs.key, 0.0)
            if p > 0:
                gr = max(gr, abs(lhs - gs.ln_k(self.T) - math.log(p)))
        c["max_abs_gas_residual"] = gr
        return c

    # ---------------------------------------------------------------------------------------------------
    def drying_path(self, feed: dict, rh_grid: Sequence[float], *, ln_s_crit: float | dict = 0.0,
                    verbose: bool = False, **gas_kw) -> list:
        """Decreasing-RH path with crystallization only after a critical supersaturation.

        Starting from the highest RH without solids, a solid becomes a candidate once its saturation index in the
        (metastable) liquid exceeds ``ln_s_crit`` (a float or a dict by solid key) and stays a candidate at all lower
        RH.  ``ln_s_crit = 0`` reproduces the equilibrium path; a large value gives the fully metastable path."""
        crit = (lambda k: ln_s_crit.get(k, 0.0)) if isinstance(ln_s_crit, dict) else (lambda k: float(ln_s_crit))
        enabled: list[str] = []
        out = []
        for rh in sorted(rh_grid, reverse=True):
            res = self.solve(feed, rh, solids=list(enabled) if enabled else "none", verbose=verbose, **gas_kw)
            new = [k for k, v in res.si.items() if k not in enabled and v > crit(k)]
            if new:
                enabled += new
                res = self.solve(feed, rh, solids=list(enabled), verbose=verbose, **gas_kw)
            res.message = f"enabled solids: {enabled}"
            out.append(res)
        return out

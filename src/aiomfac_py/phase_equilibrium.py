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
Non-reactive ion sets only: H+, HSO4-, CO3--, HCO3- and OH- (which need dissociation equilibria) are rejected.  No gas
phase other than water.  A state without any liquid (all salts crystalline and no organic present) is not represented;
use :class:`aiomfac_py.sle.SLESolver` for purely inorganic systems below their deliquescence RH.  Like
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
from .model import ActivityModel
from .solids import ION_REGISTRY, SOLIDS, Solid

WATER = Component(1, "Water", ((16, 1),))
_UNSUPPORTED_IONS = ("H+", "HSO4-", "CO3--", "HCO3-", "OH-")


# ============================================================================================================
# Liquid-phase activities in the ion basis
# ============================================================================================================

class LiquidModel:
    """AIOMFAC activities of one liquid phase given species amounts: water, organics, individual ions."""

    def __init__(self, organics: Sequence[Component], ions: Sequence[str]):
        bad = [i for i in ions if i in _UNSUPPORTED_IONS]
        if bad:
            raise NotImplementedError(f"ions {bad} need dissociation equilibria and are not supported in this version")
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
        cats = [i for i in self.ions if ION_REGISTRY[i][1] > 0]
        ans = [i for i in self.ions if ION_REGISTRY[i][1] < 0]
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

    def hessian(self, n: np.ndarray, T: float, rel: float = 1.0e-5) -> np.ndarray:
        """d ln a_i / d n_j by central differences, symmetrized."""
        n = np.asarray(n, dtype=float)
        H = np.empty((self.N, self.N))
        for j in range(self.N):
            h = rel * n[j]
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

    @property
    def n_liquids(self) -> int:
        return len(self.liquids)

    def summary(self) -> str:
        lines = [f"{self.status}: T={self.T_K} K, RH={self.rh:.4f}, {self.n_liquids} liquid phase(s), "
                 f"solids={ {k: round(v, 8) for k, v in self.solids.items()} }"]
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
        self.T = float(T_K)
        self.names = self.lm.names
        self.N = self.lm.N
        self.z = self.lm.z
        if solid_keys is None:
            cand = [s for s in SOLIDS.values() if set(s.ions) <= set(ions) and not getattr(s, "n_oh", 0)
                    and not any(i in _UNSUPPORTED_IONS for i in s.ions)]
        else:
            cand = [SOLIDS[k] for k in solid_keys]
        self.all_solids: list[Solid] = [s for s in cand if s.kspec is None or True]
        self.k_mode = k_mode
        self._lnk = {s.key: s.ln_k(self.T, k_mode) for s in self.all_solids}
        self.tol_tpd = 1.0e-7

    # ---------------------------------------------------------------------------------------------------
    def _solid_columns(self, solids: list[Solid]) -> np.ndarray:
        V = np.zeros((self.N, len(solids)))
        for j, s in enumerate(solids):
            for ion, nu in s.ions.items():
                V[self.names.index(ion), j] = nu
        return V

    def _solid_cost(self, solids: list[Solid], ln_rh: float) -> np.ndarray:
        return np.array([self._lnk[s.key] - s.h * ln_rh for s in solids])

    def si_of(self, ln_a: np.ndarray, ln_rh: float, solids: list[Solid] | None = None) -> dict:
        """Saturation index of each solid for the activities of one liquid (gauge-free: solids are neutral)."""
        solids = self.all_solids if solids is None else solids
        out = {}
        for s in solids:
            v = sum(nu * ln_a[self.names.index(ion)] for ion, nu in s.ions.items()) + s.h * ln_rh - self._lnk[s.key]
            out[s.key] = float(v)
        return out

    # ---------------------------------------------------------------------------------------------------
    # Inner problem: barrier Newton on the affine set
    # ---------------------------------------------------------------------------------------------------
    def _build_A(self, n_liq: int, V: np.ndarray):
        N, S = self.N, V.shape[1]
        rows = []
        for i in range(1, N):                                     # mass balance of non-water species
            r = np.zeros(n_liq * N + S)
            for a in range(n_liq):
                r[a * N + i] = 1.0
            r[n_liq * N:] = V[i]
            rows.append(r)
        if np.any(self.z != 0):
            for a in range(n_liq):                                # electroneutrality of each liquid
                r = np.zeros(n_liq * N + S)
                r[a * N:(a + 1) * N] = self.z
                rows.append(r)
        return np.array(rows)

    def _objective(self, x, n_liq, ln_rh, cost):
        N = self.N
        F = 0.0
        lna = []
        for a in range(n_liq):
            n = x[a * N:(a + 1) * N]
            la = self.lm.ln_a(n, self.T)
            lna.append(la)
            F += float(np.dot(n, la)) - n[0] * ln_rh
        u = x[n_liq * N:]
        F += float(np.dot(u, cost))
        return F, lna

    def _grad(self, lna, n_liq, ln_rh, cost):
        g = []
        for la in lna:
            gg = la.copy(); gg[0] -= ln_rh
            g.append(gg)
        g.append(cost)
        return np.concatenate(g)

    def _barrier_solve(self, x, n_liq, V, ln_rh, *, mu0=1.0e-3, mu_min=1.0e-14, max_newton=60, verbose=False):
        """Minimize F - mu sum ln x over {A x = A x0} from the strictly feasible x (returns x, converged)."""
        N = self.N
        S = V.shape[1]
        cost = self._solid_cost(self._solids_now, ln_rh)
        A = self._build_A(n_liq, V)
        Z = null_space(A) if A.size else np.eye(len(x))
        mu = mu0
        converged = True
        while True:
            for it in range(max_newton):
                F, lna = self._objective(x, n_liq, ln_rh, cost)
                g = self._grad(lna, n_liq, ln_rh, cost) - mu / x
                H = np.zeros((len(x), len(x)))
                for a in range(n_liq):
                    H[a * N:(a + 1) * N, a * N:(a + 1) * N] = self.lm.hessian(x[a * N:(a + 1) * N], self.T)
                H[np.diag_indices_from(H)] += mu / x ** 2
                Hr = Z.T @ H @ Z
                gr = Z.T @ g
                w, Q = np.linalg.eigh(0.5 * (Hr + Hr.T))
                shift = 0.0
                if w[0] <= 1.0e-12 * max(1.0, abs(w[-1])):
                    shift = -w[0] + 1.0e-8 * max(1.0, abs(w[-1]))
                d = -Q @ ((Q.T @ gr) / (w + shift))
                dx = Z @ d
                dec = float(-gr @ d)
                if dec < 1.0e-14 + 1.0e-10 * mu * len(x):
                    break
                neg = dx < 0
                amax = min(1.0, 0.995 * float(np.min(-x[neg] / dx[neg]))) if np.any(neg) else 1.0
                phi0 = F - mu * float(np.sum(np.log(x)))
                step = amax
                ok = False
                for _ in range(40):
                    xt = x + step * dx
                    if np.all(xt > 0):
                        Ft, _ = self._objective(xt, n_liq, ln_rh, cost)
                        phit = Ft - mu * float(np.sum(np.log(xt)))
                        if np.isfinite(phit) and phit <= phi0 - 1.0e-4 * step * dec:
                            ok = True
                            break
                    step *= 0.5
                if not ok:
                    if dec < 1.0e-8:
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
        except (ValueError, FloatingPointError, OverflowError):
            return dry * rh / (1 - rh)

    def _tpd_minimize(self, mu_eq: np.ndarray, w0: np.ndarray):
        """min_w sum w_i (ln a_i(w) - mu_i) over w > 0, sum w = 1, sum z w = 0 (barrier Newton, from w0)."""
        N = self.N
        rows = [np.ones(N)]
        if np.any(self.z != 0):
            rows.append(self.z)
        A = np.array(rows)
        Z = null_space(A)
        w = np.asarray(w0, dtype=float) / np.sum(w0)
        tpd = lambda w: float(np.dot(w, self.lm.ln_a(w, self.T) - mu_eq))
        mu = 1.0e-4
        for _ in range(9):
            for it in range(40):
                la = self.lm.ln_a(w, self.T)
                f = float(np.dot(w, la - mu_eq))
                g = la - mu_eq - mu / w
                H = self.lm.hessian(w, self.T)
                H[np.diag_indices_from(H)] += mu / w ** 2
                Hr = Z.T @ H @ Z; gr = Z.T @ g
                ev, Q = np.linalg.eigh(0.5 * (Hr + Hr.T))
                big = max(1.0, float(np.max(np.abs(ev))))
                shift = -ev[0] + 1e-8 * big if ev[0] <= 1e-12 * big else 0.0
                d = Z @ (-Q @ ((Q.T @ gr) / (ev + shift)))
                dec = float(-gr @ (Z.T @ d))
                if dec < 1e-14:
                    break
                neg = d < 0
                amax = min(1.0, 0.995 * float(np.min(-w[neg] / d[neg]))) if np.any(neg) else 1.0
                phi0 = f - mu * float(np.sum(np.log(w)))
                step, ok = amax, False
                for _ in range(40):
                    wt = w + step * d
                    if np.all(wt > 0):
                        phit = tpd(wt) - mu * float(np.sum(np.log(wt)))
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
        if b[nn:].sum() > 0:
            ion_fr[nn:] = b[nn:] / b[nn:].sum()
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
            trials.append(np.maximum(L.amounts / L.total, 1e-9) * np.exp(0.3 * rng.standard_normal(N) * (self.z == 0)))
        out = []
        for w in trials:
            w = np.maximum(w, 1e-12)
            if np.any(self.z != 0):                                 # restore electroneutrality by scaling the anions
                qp = float(np.sum(np.where(self.z > 0, self.z * w, 0.0)))
                qn = float(-np.sum(np.where(self.z < 0, self.z * w, 0.0)))
                if qn > 0:
                    w = np.where(self.z < 0, w * qp / qn, w)
            out.append(w / w.sum())
        return out

    # ---------------------------------------------------------------------------------------------------
    def solve(self, feed: dict, rh: float, *, solids="all", max_liquids: int = 3, max_outer: int = 6,
              verbose: bool = False) -> PhaseEquilibriumResult:
        """Equilibrium for the non-water ``feed`` [mol] (organic names and ion keys) at relative humidity ``rh``."""
        if not 0.0 < rh < 1.0:
            raise ValueError("rh must be in (0, 1)")
        ln_rh = math.log(rh)
        N, nn = self.N, self.lm.n_neutral
        b = np.zeros(N)
        for k, v in feed.items():
            if k not in self.names or k == "Water":
                raise KeyError(f"feed species {k!r} not in {self.names[1:]}")
            b[self.names.index(k)] = float(v)
        if abs(float(np.dot(self.z, b))) > 1e-9 * max(1.0, b.sum()):
            raise ValueError("feed is not electroneutral")
        scale = float(b[1:].sum())
        b = b / scale

        if solids == "all":
            self._solids_now = list(self.all_solids)
        elif solids == "none":
            self._solids_now = []
        else:
            self._solids_now = [SOLIDS[k] for k in solids]
        V = self._solid_columns(self._solids_now)
        S = V.shape[1]

        # strictly feasible start: one liquid with (almost) everything, solids at a tiny neutral amount
        u0 = np.full(S, 1e-6)
        n0 = b.copy()
        n0[1:] = b[1:] - V[1:] @ u0
        floor = 1e-7
        n0[1:] = np.maximum(n0[1:], floor)
        if np.any(self.z != 0) and np.any(b[nn:] > 0):          # re-neutralize after flooring
            qp = float(np.sum(np.where(self.z > 0, self.z * n0, 0.0)))
            qn = float(-np.sum(np.where(self.z < 0, self.z * n0, 0.0)))
            n0 = np.where(self.z < 0, n0 * qp / qn, n0)
        # absorb the flooring change into the solids/feed consistency: recompute b' = n0 + V u0 for non-water
        b_eff = n0.copy(); b_eff[1:] = n0[1:] + V[1:] @ u0
        n0[0] = self._water_guess(n0, rh)
        phases = [n0]
        x = np.concatenate(phases + [u0])
        n_outer = 0
        tpd_min = float("nan")
        inner_ok = True
        while True:
            n_outer += 1
            n_liq = len(phases)
            x, conv = self._barrier_solve(x, n_liq, V, ln_rh, verbose=verbose)
            inner_ok = inner_ok and conv
            phases = [x[a * N:(a + 1) * N] for a in range(n_liq)]
            u = x[n_liq * N:]
            # remove vanished liquids, merge identical ones
            keep = [a for a in range(n_liq) if phases[a].sum() > 1e-8]
            merged = []
            for a in keep:
                xa = phases[a] / phases[a].sum()
                for m in merged:
                    if np.max(np.abs(xa - phases[m] / phases[m].sum())) < 1e-4:
                        phases[m] = phases[m] + phases[a]
                        break
                else:
                    merged.append(a)
            if len(merged) < n_liq:
                phases = [phases[a] for a in merged]
                x = np.concatenate(phases + [u])
                x, conv = self._barrier_solve(x, len(phases), V, ln_rh, verbose=verbose)
                n_liq = len(phases)
                phases = [x[a * N:(a + 1) * N] for a in range(n_liq)]
                u = x[n_liq * N:]
            liquids = [LiquidPhase(p.copy(), self.lm.ln_a(p, self.T), self.names) for p in phases]
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
            # add the trial phase: take a small neutral amount of w out of the liquid that can supply the most
            w = best[1]
            nonw = w.copy(); nonw[0] = 0.0
            cand = []
            for a, p in enumerate(phases):
                mask = nonw[1:] > 0
                theta = 0.5 * float(np.min(p[1:][mask] / nonw[1:][mask]))
                cand.append((theta, a))
            theta, a = max(cand)
            newp = theta * w.copy()
            newp[0] = theta * w[0]
            phases[a] = phases[a] - theta * nonw
            phases.append(newp)
            x = np.concatenate(phases + [u])

        liquids = [LiquidPhase(p * scale, self.lm.ln_a(p, self.T), self.names) for p in phases]
        org_frac = lambda L: float(np.sum(L.mole_fractions[1:nn]))
        liquids.sort(key=org_frac, reverse=True)
        u = x[len(phases) * N:] * scale
        si = self.si_of(liquids[0].ln_a, ln_rh)
        self._enabled_keys = {s_.key for s_ in self._solids_now}
        thresh = 1e-7 * scale
        solid_amounts = {s.key: float(v) for s, v in zip(self._solids_now, u) if v > thresh}
        F, _ = self._objective(x, len(phases), ln_rh, self._solid_cost(self._solids_now, ln_rh))
        checks = self._checks(liquids, ln_rh, si, solid_amounts, b * scale, u)
        # the status is decided by the equilibrium conditions of the final state, not by how the iteration ended
        status, msg = "converged", ""
        bad = [k for k, lim in (("max_abs_ln_aw_minus_ln_rh", 1e-5), ("max_neutral_mu_spread", 1e-4),
                                ("max_ion_mu_residual", 1e-4), ("max_si", 1e-4), ("max_abs_si_present_solids", 1e-4))
               if checks[k] > lim]
        if bad:
            status = "not_converged"
            msg = "equilibrium conditions violated: " + ", ".join(bad)
        if tpd_min < -self.tol_tpd:
            status = "not_converged"
            msg += ("; " if msg else "") + f"stability test still negative (TPD={tpd_min:.2e})"
        if not inner_ok:
            msg += ("; " if msg else "") + "inner Newton iteration stopped early (line search)"
        return PhaseEquilibriumResult(status, self.T, rh, liquids, solid_amounts, si, F * scale, checks, tpd_min,
                                      n_outer, message=msg, names=self.names)

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
    def _checks(self, liquids, ln_rh, si, solid_amounts, b, u) -> dict:
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
        c["max_mass_balance_residual"] = float(np.max(np.abs(tot[1:] - b[1:]))) / max(float(b[1:].sum()), 1e-300)
        return c

    # ---------------------------------------------------------------------------------------------------
    def drying_path(self, feed: dict, rh_grid: Sequence[float], *, ln_s_crit: float | dict = 0.0,
                    verbose: bool = False) -> list:
        """Decreasing-RH path with crystallization only after a critical supersaturation.

        Starting from the highest RH without solids, a solid becomes a candidate once its saturation index in the
        (metastable) liquid exceeds ``ln_s_crit`` (a float or a dict by solid key) and stays a candidate at all lower
        RH.  ``ln_s_crit = 0`` reproduces the equilibrium path; a large value gives the fully metastable path."""
        crit = (lambda k: ln_s_crit.get(k, 0.0)) if isinstance(ln_s_crit, dict) else (lambda k: float(ln_s_crit))
        enabled: list[str] = []
        out = []
        for rh in sorted(rh_grid, reverse=True):
            res = self.solve(feed, rh, solids=list(enabled) if enabled else "none", verbose=verbose)
            new = [k for k, v in res.si.items() if k not in enabled and v > crit(k)]
            if new:
                enabled += new
                res = self.solve(feed, rh, solids=list(enabled), verbose=verbose)
            res.message = f"enabled solids: {enabled}"
            out.append(res)
        return out

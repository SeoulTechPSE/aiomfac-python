"""Bisulfate dissociation equilibrium HSO4- <-> H+ + SO4-- (SubModDissociationEquil.f90: HSO4_dissociation,
DiffKsulfuricDissoc, fHSO4dissoc).

Only the bisulfate-*only* case is ported (Fortran calls this routine when ``bisulfsyst`` is true and
``bicarbsyst`` is false). Systems that also contain HCO3-/CO3--/CO2(aq) go through the Fortran joint 2-D solver
``HSO4_and_HCO3_dissociation`` instead, which is not ported; such systems are rejected (see model.py).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .numerics import brent_root

_DEPS = np.finfo(float).eps
_T0 = 298.15
_MOLAR_MASS_HSO4 = 0.097071
_C1, _C2, _C3, _C4, _C5 = 2.63468e-01, 2.63204e-01, 1.72067, 7.30039e01, 7.45718e-01
_DH0, _CP0, _DCPDT = -1.8554748598e4, -2.05594443486e2, -9.94992864240e-1
_FE1 = 1.0 / 8.314472
_FE2 = _DH0 - _CP0 * _T0 + 0.5 * _DCPDT * _T0 * _T0
_FE3 = 1.0 / _T0
_FE4 = _CP0 - _DCPDT * _T0


def ln_k_hso4_at_t(T_K: float) -> float:
    """ln of the 2nd dissociation constant of H2SO4 (Knopf et al., 2003, corrected eq. 16)."""
    T = min(max(T_K, 180.0), 473.0)
    return -4.54916799587 - _FE1 * (_FE2 * (1.0 / T - _FE3) - _FE4 * math.log(T / _T0) - 0.5 * _DCPDT * (T - _T0))


@dataclass
class BisulfateResult:
    m_h: float
    m_hso4: float
    m_so4: float
    alpha_hso4: float          # degree of dissociation: 1 - mHSO4/mHSO4max


def solve_bisulfate(model, T_K: float, xn, smc, sma, idx_h: int, idx_hso4: int, idx_so4: int) -> BisulfateResult:
    """Solve for the equilibrium H+/HSO4-/SO4-- molalities given the total (as-input) molalities in
    ``smc[idx_h]``, ``sma[idx_hso4]``, ``sma[idx_so4]``. Mutates and returns ``smc``/``sma`` in place is *not*
    done here; the caller applies the result (mirrors DiffKsulfuricDissoc's SMC/SMA side effects explicitly).
    """
    m_h_max = smc[idx_h] + sma[idx_hso4]
    m_sulf_max = sma[idx_hso4] + sma[idx_so4]
    m_hso4_max = sma[idx_hso4] + min(smc[idx_h], sma[idx_so4])
    m_hso4_min = 0.0
    mscale0 = 1.0e-3 * _DEPS
    if m_hso4_max <= mscale0:
        return BisulfateResult(smc[idx_h], sma[idx_hso4], sma[idx_so4], -9.999999)
    ln_k = ln_k_hso4_at_t(T_K)

    def diff_k(m_hso4_trial: float) -> float:
        mscale = 1.0e-3 * _DEPS
        m_hso4 = min(max(m_hso4_trial, m_hso4_min), m_hso4_max)
        sma_t, smc_t = sma.copy(), smc.copy()
        if m_hso4 < mscale:
            m_hso4 += min(mscale - m_hso4, m_hso4_max - m_hso4)
        sma_t[idx_hso4] = m_hso4
        m_h = max(m_h_max - m_hso4, 0.0)
        if m_h < mscale:
            madd = mscale - m_h
            if m_hso4 - madd > m_hso4_min:
                m_hso4 -= madd; sma_t[idx_hso4] = m_hso4; m_h += madd
        smc_t[idx_h] = m_h
        m_sulf = max(m_sulf_max - m_hso4, 0.0)
        if m_sulf < mscale:
            madd = mscale - m_sulf
            if m_hso4 - madd > m_hso4_min:
                m_hso4 -= madd; sma_t[idx_hso4] = m_hso4; m_sulf += madd
        sma_t[idx_so4] = m_sulf

        x = model._x_from_molalities(xn, smc_t, sma_t)
        lr, mr, sr = model.lr_mr_sr(T_K, smc_t, sma_t, xn, x)
        nn, nc = model.mixture.n_neutral, model.mixture.sr.n_cation
        ln_g_h = mr.ln_gamma_cation[idx_h] + sr.ln_gamma_sr[nn + idx_h] + lr.ln_gamma_cation[idx_h] - mr.tmolal
        ln_g_hso4 = (mr.ln_gamma_anion[idx_hso4] + sr.ln_gamma_sr[nn + nc + idx_hso4]
                    + lr.ln_gamma_anion[idx_hso4] - mr.tmolal)
        ln_g_so4 = (mr.ln_gamma_anion[idx_so4] + sr.ln_gamma_sr[nn + nc + idx_so4]
                   + lr.ln_gamma_anion[idx_so4] - mr.tmolal)
        ln_k_bisulf = ln_g_h + ln_g_so4 - ln_g_hso4
        ratio = (m_h * m_sulf / m_hso4) if m_hso4 > 0.0 else (_DEPS / max(m_hso4, _DEPS))
        ratio = max(ratio, _DEPS)
        diff_k.last = (m_h, m_hso4, m_sulf)
        return ln_k_bisulf + math.log(ratio) - ln_k

    lo, hi = max(m_hso4_min, mscale0), m_hso4_max - mscale0
    if hi <= lo:
        lo, hi = m_hso4_min, m_hso4_max
    n = 40
    grid = np.geomspace(max(lo, m_hso4_max * 1e-8), hi, n) if lo <= 0.0 else np.linspace(lo, hi, n)
    vals = [diff_k(float(g)) for g in grid]
    root = None
    for i in range(len(grid) - 1):
        if vals[i] == 0.0:
            root = float(grid[i]); break
        if vals[i] * vals[i + 1] < 0.0:
            root = brent_root(diff_k, float(grid[i]), float(grid[i + 1])); break
    if root is None:
        root = float(grid[int(np.argmin(np.abs(vals)))])
    diff_k(root)
    m_h, m_hso4, m_sulf = diff_k.last
    return BisulfateResult(m_h, m_hso4, m_sulf, 1.0 - m_hso4 / m_hso4_max)

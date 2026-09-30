"""Bicarbonate/carbonate dissociation equilibrium (SubModDissociationEquil.f90: HSO4_and_HCO3_dissociation,
DiffK_carb_sulf).

Two cases are ported:

- ``solve_carbonate`` -- **bicarbonate-only** (``bisulfsyst == False``): CO2(aq) + H2O <-> H+ + HCO3-,
  HCO3- <-> H+ + CO3--, and the water self-ionisation H2O <-> H+ + OH-, solved jointly (5 unknowns: nHCO3,
  nCarb, nCO2, nOH, nH; 3 equilibrium constants + 2 mass balances).
- ``solve_carb_sulf`` -- **joint bisulfate + bicarbonate** (``bisulfsyst == True``): the above three equilibria
  plus HSO4- <-> H+ + SO4-- (6 unknowns: nHCO3, nCarb, nCO2, nOH, nHSO4, nH; 4 equilibrium constants + 2 mass
  balances), with an optional Ca2+ + SO4-- -> CaSO4(s) precipitation pre-step (``_precipitate_ca_sulfate``,
  the Fortran ``idCa > 0`` branch) applied deterministically before the equilibrium solve.

Unlike the bisulfate-only equilibrium (dissociation.py), water is a reactant here, so the solvent mass -- and
therefore every neutral component's mass fraction -- changes with the extent of reaction.

Ported only for ``use_CO2gas_equil == False`` (a gas-phase-coupling option the Fortran web driver never enables).
The Fortran routine also special-cases *very* low HCO3-/CO3--/OH-/SO4-- concentrations with a geometric-mean
correction near machine precision; those branches are not reproduced here (the ordinary branches are used
throughout), so results at extreme dilution may deviate. CaCO3(s) precipitation (as opposed to CaSO4(s)) is not
part of this Fortran routine and is not ported here either.

Uses ``scipy.optimize.root(method="hybr")`` -- SciPy's wrapper around the same MINPACK ``hybrd`` routine the
Fortran code calls via ``Mod_MINPACK`` -- rather than a hand-written solver, since a faithful from-scratch
reimplementation of a multivariate hybrid Powell solver is out of scope here.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import dissociation
from .params import load_subgroup_params

_T0 = 298.15
_FA = (-8.204327e02, -1.4027266e-01, 5.027549e04, 1.268339e02, -3.879660e06)
_FB = (-2.484192e02, -7.489962e-02, 1.186243e04, 3.892561e01, -1.297999e06)
_FD = (-1.4816780e02, 8.933802e-01, -2.332199e-03, 2.146860e-06)


def _poly_k(T, coeffs):
    f1, f2, f3, f4, f5 = coeffs
    return f1 + f2 * T + f3 / T + f4 * math.log(T) + f5 / T ** 2


def ln_k1_hco3_at_t(T_K: float) -> float:
    return _poly_k(T_K, _FA)


def ln_k2_hco3_at_t(T_K: float) -> float:
    return _poly_k(T_K, _FB)


def ln_kw_at_t(T_K: float) -> float:
    """water self-ionisation constant; Fortran fit valid 273-323 K."""
    f1, f2, f3, f4 = _FD
    return f1 + f2 * T_K + f3 * T_K ** 2 + f4 * T_K ** 3


@dataclass
class CarbonateResult:
    n_hco3: float; n_carb: float; n_co2: float; n_oh: float; n_h: float
    mol_neutral: np.ndarray        # (n_neutral,) updated neutral mole amounts (water and CO2(aq) changed)
    residual_norm: float           # sum(|diffK|) at the returned solution (Fortran's own convergence check)


def gamma_co2_mr(mixture, smc, sma) -> float:
    """GammaCO2(): ln(gamma) of CO2(aq) from the salting-out coefficients (replaces the ordinary MR term)."""
    sg = load_subgroup_params()
    sr = mixture.sr
    lam_c = sg.lambdaIN[np.array(sr.cations) - 201]
    lam_a = sg.lambdaIN[np.array(sr.anions) - 201]
    return float(2.0 * np.sum(smc[:sr.n_cation] * lam_c) + 2.0 * np.sum(sma[:sr.n_anion] * lam_a))


def solve_carbonate(model, T_K: float, wtf0, smc, sma, idx_h: int, idx_oh: int, idx_hco3: int, idx_carb: int,
                    idx_co2_neutral: int) -> CarbonateResult:
    from scipy.optimize import root

    m = model.mixture
    nn = m.n_neutral
    Mmass = m.mmass[:nn]

    n_h2o_init = wtf0[0] / (float(np.sum(wtf0[:nn])) * Mmass[0])          # MassFrac2SolvMolalities, index 0
    mol_neutral_init = wtf0[:nn] / (float(np.sum(wtf0[:nn])) * Mmass)
    n_co2_init = mol_neutral_init[idx_co2_neutral]
    n_oh_init = sma[idx_oh]

    sum_cat = float(np.sum(smc))
    r = min(3.5 * sum_cat / n_h2o_init, 0.98)
    n_oh_max = r * n_h2o_init + n_oh_init
    n_h_max = smc[idx_h] + sma[idx_hco3] + 2.0 * n_co2_init + (n_oh_max - n_oh_init)
    n_carb_max = sma[idx_carb] + sma[idx_hco3] + n_co2_init
    if n_carb_max <= np.finfo(float).eps:
        return CarbonateResult(sma[idx_hco3], sma[idx_carb], n_co2_init, sma[idx_oh], smc[idx_h],
                               mol_neutral_init.copy(), 0.0)

    ln_k1, ln_k2, ln_kw = ln_k1_hco3_at_t(T_K), ln_k2_hco3_at_t(T_K), ln_kw_at_t(T_K)
    ngi = m.ngi
    sg = load_subgroup_params()
    smw_c = np.zeros(ngi); smw_c[:m.sr.n_cation] = [sg.SMWC[c - 201] for c in m.sr.cations]
    smw_a = np.zeros(ngi); smw_a[:m.sr.n_anion] = [sg.SMWA[a - 241] for a in m.sr.anions]
    smw_c *= 1.0e-3; smw_a *= 1.0e-3
    n_cations_molar = smc.copy(); n_anions_molar = sma.copy()   # constant molar amounts of the non-reacting ions

    def residual(log_vars):
        log_vars = np.clip(log_vars, -700.0, 700.0)
        n_hco3, n_carb, n_co2, n_oh, n_h = np.exp(log_vars)
        delta_co2 = n_co2 - n_co2_init
        mol_neutral = mol_neutral_init.copy()
        mol_neutral[0] = max(n_h2o_init + delta_co2 - (n_oh - n_oh_init), np.finfo(float).eps)
        mol_neutral[idx_co2_neutral] = n_co2
        solvmass = float(np.sum(mol_neutral * Mmass))

        smc_t, sma_t = n_cations_molar.copy(), n_anions_molar.copy()
        smc_t[idx_h] = n_h
        sma_t[idx_hco3], sma_t[idx_carb], sma_t[idx_oh] = n_hco3, n_carb, n_oh
        totmass = solvmass + float(np.sum(smc_t * smw_c)) + float(np.sum(sma_t * smw_a))
        wtf = mol_neutral * Mmass / totmass
        xn = mol_neutral / float(np.sum(mol_neutral))
        smc_m, sma_m = smc_t / solvmass, sma_t / solvmass

        x = model._x_from_molalities(xn, smc_m, sma_m)
        lr, mr, sr = model.lr_mr_sr(T_K, smc_m, sma_m, xn, x)
        nc = m.sr.n_cation
        tmolal = mr.tmolal
        ln_g_h = mr.ln_gamma_cation[idx_h] + sr.ln_gamma_sr[nn + idx_h] + lr.ln_gamma_cation[idx_h] - tmolal
        ln_g_hco3 = (mr.ln_gamma_anion[idx_hco3] + sr.ln_gamma_sr[nn + nc + idx_hco3]
                    + lr.ln_gamma_anion[idx_hco3] - tmolal)
        ln_g_carb = (mr.ln_gamma_anion[idx_carb] + sr.ln_gamma_sr[nn + nc + idx_carb]
                    + lr.ln_gamma_anion[idx_carb] - tmolal)
        ln_g_oh = mr.ln_gamma_anion[idx_oh] + sr.ln_gamma_sr[nn + nc + idx_oh] + lr.ln_gamma_anion[idx_oh] - tmolal
        ln_g_co2 = gamma_co2_mr(m, smc_m, sma_m)
        ln_g_water = sr.ln_gamma_sr[0] + mr.ln_gamma_neutral[0] + lr.ln_gamma_neutral[0]
        x_water = x[0]

        ln_k1_bicarb = ln_g_h + ln_g_hco3 - ln_g_co2 - ln_g_water
        ln_k2_bicarb = ln_g_h + ln_g_carb - ln_g_hco3
        ln_kw_bicarb = ln_g_h + ln_g_oh - ln_g_water

        m_h, m_hco3, m_carb, m_oh = smc_m[idx_h], sma_m[idx_hco3], sma_m[idx_carb], sma_m[idx_oh]
        m_co2 = mol_neutral[idx_co2_neutral] / solvmass
        d1 = ln_k1_bicarb + math.log(max(m_h * m_hco3 / (m_co2 * x_water), 1e-300)) - ln_k1
        d2 = ln_k2_bicarb + math.log(max(m_h * m_carb / max(m_hco3, 1e-300), 1e-300)) - ln_k2
        d3 = ln_kw_bicarb + math.log(max(m_h * m_oh / x_water, 1e-300)) - ln_kw
        d4 = 1.0e2 * (n_h_max - n_hco3 - n_h - 2.0 * n_co2 - n_oh_max + n_oh)
        d5 = 1.0e2 * (n_carb_max - n_hco3 - n_carb - n_co2)
        return [d1, d2, d3, d4, d5]

    n_hco3_0 = 0.13 * min(n_h_max, n_carb_max)
    n_carb_0 = 0.6 * n_carb_max
    n_co2_0 = max(1.0e-5 * n_carb_max, 1e-12)
    n_oh_0 = max(2.0e-5 * n_oh_max, 1e-12)
    n_h_0 = n_hco3_0 + 2.0 * n_carb_0 + n_oh_0
    x0 = np.log([max(n_hco3_0, 1e-12), max(n_carb_0, 1e-12), n_co2_0, n_oh_0, max(n_h_0, 1e-12)])

    best = None
    for scale in (1.0, 0.3, 3.0, 0.05, 20.0, 0.01, 100.0):
        sol = root(residual, x0 + math.log(scale), method="hybr", options={"xtol": 1e-13, "maxfev": 20000})
        rn = float(np.sum(np.abs(residual(sol.x))))
        if best is None or rn < best[1]:
            best = (sol.x, rn)
        if rn < 1e-6:
            break
    log_vars, rn = best
    n_hco3, n_carb, n_co2, n_oh, n_h = np.exp(log_vars)
    mol_neutral = mol_neutral_init.copy()
    mol_neutral[0] = max(n_h2o_init + (n_co2 - n_co2_init) - (n_oh - n_oh_init), np.finfo(float).eps)
    mol_neutral[idx_co2_neutral] = n_co2
    return CarbonateResult(n_hco3, n_carb, n_co2, n_oh, n_h, mol_neutral, rn)


def _precipitate_ca_sulfate(n_ca: float, n_hso4: float, n_sulf: float):
    """Ca2+ + SO4-- -> CaSO4(s), applied deterministically before the joint solve (Fortran ``idCa > 0`` branch
    of ``HSO4_and_HCO3_dissociation``, lines computing ``SNC(idCa)``/``nSulf``/``nH``/``nHSO4`` just before the
    solver loop). All HSO4- is converted to free H+ + (part of) the precipitating sulfate pool first, then Ca2+
    consumes as much of the combined SO4--/HSO4--derived sulfate as it can (Ksp is not modelled -- the Fortran
    routine's own precipitation is a full stoichiometric removal down to a tiny residual, not a solubility
    equilibrium).

    Returns ``(n_ca_new, n_hso4_new, n_sulf_new, n_h_delta)`` -- ``n_hso4_new`` is always 0.0; the caller adds
    ``n_h_delta`` to the free H+ amount.
    """
    deps = np.finfo(float).eps
    n_sulf_max = n_hso4 + n_sulf
    if n_sulf_max <= 0.0:
        return n_ca, n_hso4, n_sulf, 0.0
    if n_ca < n_sulf_max:
        n_ca_new = 0.0
        n_sulf_new = n_sulf + n_hso4 - n_ca
    else:
        residual = min(1.0e-5 * n_sulf_max, 1.0e3 * deps)
        n_ca_new = n_ca - n_sulf_max + residual
        n_sulf_new = residual
    return n_ca_new, 0.0, n_sulf_new, n_hso4


@dataclass
class CarbSulfResult:
    n_hco3: float; n_carb: float; n_co2: float; n_oh: float; n_hso4: float; n_h: float
    n_sulf: float                  # = nSulfmax - n_hso4 (not an independent unknown)
    mol_neutral: np.ndarray        # (n_neutral,) updated neutral mole amounts (water and CO2(aq) changed)
    n_ca: float | None             # residual Ca2+ after CaSO4(s) precipitation, or None if no Ca2+ in the system
    ca_precipitated: bool          # True if the CaSO4(s) precipitation step actually removed any Ca2+/sulfate
    residual_norm: float           # sum(|diffK|) at the returned solution (Fortran's own convergence check)


def solve_carb_sulf(model, T_K: float, wtf0, smc, sma, idx_h: int, idx_oh: int, idx_hco3: int, idx_carb: int,
                    idx_hso4: int, idx_so4: int, idx_co2_neutral: int, idx_ca: int | None) -> CarbSulfResult:
    """Joint bisulfate + bicarbonate dissociation equilibrium (6 unknowns: nHCO3, nCarb, nCO2, nOH, nHSO4, nH).

    Mirrors ``solve_carbonate`` but additionally couples in HSO4- <-> H+ + SO4-- (``dissociation.ln_k_hso4_at_t``,
    the same equilibrium constant used by the bisulfate-only solver), and -- if ``idx_ca`` is given and Ca2+ is
    present -- applies the deterministic CaSO4(s) precipitation pre-step first.
    """
    from scipy.optimize import root

    m = model.mixture
    nn = m.n_neutral
    Mmass = m.mmass[:nn]

    n_h2o_init = wtf0[0] / (float(np.sum(wtf0[:nn])) * Mmass[0])          # MassFrac2SolvMolalities, index 0
    mol_neutral_init = wtf0[:nn] / (float(np.sum(wtf0[:nn])) * Mmass)
    n_co2_init = mol_neutral_init[idx_co2_neutral]
    n_oh_init = sma[idx_oh]
    n_hco3_init = sma[idx_hco3]
    n_carb_init = sma[idx_carb]

    smc = smc.copy(); sma = sma.copy()
    n_hso4_init = sma[idx_hso4]
    n_sulf_init = sma[idx_so4]
    n_h_init = smc[idx_h]

    ca_precipitated = False
    n_ca_result = None
    if idx_ca is not None:
        n_ca_result = smc[idx_ca]
        n_ca_new, n_hso4_new, n_sulf_new, n_h_delta = _precipitate_ca_sulfate(n_ca_result, n_hso4_init, n_sulf_init)
        if n_h_delta != 0.0 or n_hso4_new != n_hso4_init:
            ca_precipitated = True
        smc[idx_ca] = n_ca_new
        n_ca_result = n_ca_new
        n_hso4_init, n_sulf_init = n_hso4_new, n_sulf_new
        n_h_init = n_h_init + n_h_delta

    sum_cat = float(np.sum(smc))
    r = min(3.5 * sum_cat / n_h2o_init, 0.98)
    n_oh_max = r * n_h2o_init + n_oh_init
    n_sulf_max = n_hso4_init + n_sulf_init
    n_h_max = n_h_init + n_hco3_init + 2.0 * n_co2_init + (n_oh_max - n_oh_init) + n_hso4_init
    n_carb_max = n_carb_init + n_hco3_init + n_co2_init
    if n_carb_max <= np.finfo(float).eps:
        return CarbSulfResult(n_hco3_init, n_carb_init, n_co2_init, n_oh_init, n_hso4_init, n_h_init, n_sulf_init,
                              mol_neutral_init.copy(), n_ca_result, ca_precipitated, 0.0)

    # If Ca2+ precipitation (above) has driven the residual sulfate pool down to a scale comparable to machine
    # epsilon relative to the rest of the system (the common case when Ca2+ is in excess: branch B of
    # _precipitate_ca_sulfate leaves only the ~1e3*deps bookkeeping residual), the coupled 6-unknown solve
    # becomes numerically degenerate: the HSO4-/SO4-- equation lives at a wildly different scale than the other
    # five, which starves scipy's finite-difference Jacobian of a usable gradient for it and prevents the whole
    # system from converging even though the other five equations are perfectly well-conditioned on their own.
    # Fortran's own "very low mSulf" branch handles this analytically (not ported, see module docstring); the
    # equivalent fix here is to recognise that a sulfate pool this small has a negligible effect on everything
    # else, decouple it, and solve the remaining bicarbonate-only 5-unknown system with the already-validated
    # ``solve_carbonate`` (all of HSO4- treated as fully dissociated into free H+ + SO4--, consistent with the
    # near-complete dissociation such a trace-level pool would show at equilibrium):
    if n_sulf_max <= 1e-8 * max(n_carb_max, n_h_max, 1e-30):
        smc2, sma2 = smc.copy(), sma.copy()
        smc2[idx_h] = n_h_init + n_hso4_init
        sma2[idx_hso4] = 0.0
        sma2[idx_so4] = n_sulf_max
        cr = solve_carbonate(model, T_K, wtf0, smc2, sma2, idx_h, idx_oh, idx_hco3, idx_carb, idx_co2_neutral)
        return CarbSulfResult(cr.n_hco3, cr.n_carb, cr.n_co2, cr.n_oh, 0.0, cr.n_h, n_sulf_max, cr.mol_neutral,
                              n_ca_result, ca_precipitated, cr.residual_norm)

    ln_k1, ln_k2, ln_kw = ln_k1_hco3_at_t(T_K), ln_k2_hco3_at_t(T_K), ln_kw_at_t(T_K)
    ln_k_hso4 = dissociation.ln_k_hso4_at_t(T_K)
    ngi = m.ngi
    sg = load_subgroup_params()
    smw_c = np.zeros(ngi); smw_c[:m.sr.n_cation] = [sg.SMWC[c - 201] for c in m.sr.cations]
    smw_a = np.zeros(ngi); smw_a[:m.sr.n_anion] = [sg.SMWA[a - 241] for a in m.sr.anions]
    smw_c *= 1.0e-3; smw_a *= 1.0e-3
    n_cations_molar = smc.copy(); n_anions_molar = sma.copy()   # constant molar amounts of the non-reacting ions

    def residual(log_vars, clip=True):
        if clip:
            log_vars = np.clip(log_vars, ln_bounds_lo, ln_bounds_hi)
        n_hco3, n_carb, n_co2, n_oh, n_hso4, n_h = np.exp(log_vars)
        n_hso4 = min(n_hso4, n_sulf_max)
        n_sulf = n_sulf_max - n_hso4
        delta_co2 = n_co2 - n_co2_init
        mol_neutral = mol_neutral_init.copy()
        mol_neutral[0] = max(n_h2o_init + delta_co2 - (n_oh - n_oh_init), np.finfo(float).eps)
        mol_neutral[idx_co2_neutral] = n_co2
        solvmass = float(np.sum(mol_neutral * Mmass))

        smc_t, sma_t = n_cations_molar.copy(), n_anions_molar.copy()
        smc_t[idx_h] = n_h
        sma_t[idx_hco3], sma_t[idx_carb], sma_t[idx_oh] = n_hco3, n_carb, n_oh
        sma_t[idx_hso4], sma_t[idx_so4] = n_hso4, n_sulf
        totmass = solvmass + float(np.sum(smc_t * smw_c)) + float(np.sum(sma_t * smw_a))
        wtf = mol_neutral * Mmass / totmass
        xn = mol_neutral / float(np.sum(mol_neutral))
        smc_m, sma_m = smc_t / solvmass, sma_t / solvmass

        x = model._x_from_molalities(xn, smc_m, sma_m)
        lr, mr, sr = model.lr_mr_sr(T_K, smc_m, sma_m, xn, x)
        nc = m.sr.n_cation
        tmolal = mr.tmolal
        ln_g_h = mr.ln_gamma_cation[idx_h] + sr.ln_gamma_sr[nn + idx_h] + lr.ln_gamma_cation[idx_h] - tmolal
        ln_g_hco3 = (mr.ln_gamma_anion[idx_hco3] + sr.ln_gamma_sr[nn + nc + idx_hco3]
                    + lr.ln_gamma_anion[idx_hco3] - tmolal)
        ln_g_carb = (mr.ln_gamma_anion[idx_carb] + sr.ln_gamma_sr[nn + nc + idx_carb]
                    + lr.ln_gamma_anion[idx_carb] - tmolal)
        ln_g_oh = mr.ln_gamma_anion[idx_oh] + sr.ln_gamma_sr[nn + nc + idx_oh] + lr.ln_gamma_anion[idx_oh] - tmolal
        ln_g_hso4 = (mr.ln_gamma_anion[idx_hso4] + sr.ln_gamma_sr[nn + nc + idx_hso4]
                    + lr.ln_gamma_anion[idx_hso4] - tmolal)
        ln_g_so4 = (mr.ln_gamma_anion[idx_so4] + sr.ln_gamma_sr[nn + nc + idx_so4]
                   + lr.ln_gamma_anion[idx_so4] - tmolal)
        ln_g_co2 = gamma_co2_mr(m, smc_m, sma_m)
        ln_g_water = sr.ln_gamma_sr[0] + mr.ln_gamma_neutral[0] + lr.ln_gamma_neutral[0]
        x_water = x[0]

        ln_k1_bicarb = ln_g_h + ln_g_hco3 - ln_g_co2 - ln_g_water
        ln_k2_bicarb = ln_g_h + ln_g_carb - ln_g_hco3
        ln_kw_bicarb = ln_g_h + ln_g_oh - ln_g_water
        ln_bisulf = ln_g_h + ln_g_so4 - ln_g_hso4

        m_h, m_hco3, m_carb, m_oh = smc_m[idx_h], sma_m[idx_hco3], sma_m[idx_carb], sma_m[idx_oh]
        m_hso4, m_sulf = sma_m[idx_hso4], sma_m[idx_so4]
        m_co2 = mol_neutral[idx_co2_neutral] / solvmass
        d1 = ln_k1_bicarb + math.log(max(m_h * m_hco3 / (m_co2 * x_water), 1e-300)) - ln_k1
        d2 = ln_k2_bicarb + math.log(max(m_h * m_carb / max(m_hco3, 1e-300), 1e-300)) - ln_k2
        d3 = ln_kw_bicarb + math.log(max(m_h * m_oh / x_water, 1e-300)) - ln_kw
        d4 = ln_bisulf + math.log(max(m_h * m_sulf / max(m_hso4, 1e-300), 1e-300)) - ln_k_hso4
        d5 = 1.0e2 * (n_carb_max - n_hco3 - n_carb - n_co2)
        d6 = 1.0e2 * (n_h_max - n_hco3 - n_h - 2.0 * n_co2 - n_hso4 - n_oh_max + n_oh)
        return [d1, d2, d3, d4, d5, d6]

    # initial guess (Fortran bisulfsyst branch of HSO4_and_HCO3_dissociation, lines computing molarinp(1:6)):
    n_hco3_max_var = min(n_h_max, n_carb_max)
    n_co2_max_var = min(0.5 * n_h_max, n_carb_max)
    deps = np.finfo(float).eps
    sum_ion_m = float(np.sum(n_cations_molar) + np.sum(n_anions_molar))
    if sum_ion_m > 0.0 and n_sulf_max > 0.0:
        mf_hso4max_in_ions = 2.0 * n_sulf_max / sum_ion_m
        wf_hso4max = (n_sulf_max * dissociation._MOLAR_MASS_HSO4
                     / (n_sulf_max * dissociation._MOLAR_MASS_HSO4
                        + mf_hso4max_in_ions * wtf0[0] / float(np.sum(wtf0[:nn]))))
        wf_hso4max = min(max(wf_hso4max, deps), 1.0 - deps)
        c1, c2, c3, c4, c5 = (dissociation._C1, dissociation._C2, dissociation._C3, dissociation._C4,
                              dissociation._C5)
        alpha_hso4_pred = (1.0 - 1.0 / (1.0 + (wf_hso4max ** -c1 - wf_hso4max ** -c2)) ** c4
                           + c3 * wf_hso4max ** c5 * (1.0 - wf_hso4max) ** 1.75)
        alpha_hso4_pred = min(max(alpha_hso4_pred, deps), 1.0 - deps)
    else:
        alpha_hso4_pred = 1.0 - deps

    n_hco3_0 = max(1.60e-6 * n_hco3_max_var, 1e-12)
    n_carb_0 = max(3.0 * deps * n_carb_max, 1e-12)
    n_co2_0 = max(0.9999 * n_co2_max_var, 1e-12)
    n_oh_0 = max(30.0 * deps * n_oh_max, 1e-12)
    n_sulf_0 = alpha_hso4_pred * n_sulf_max
    n_hso4_0 = max(n_sulf_max - n_sulf_0, 1e-12)
    n_h_0 = max(n_hco3_0 + 2.0 * n_carb_0 + n_oh_0 + n_hso4_0 + 2.0 * n_sulf_0, 1e-12)
    x0 = np.log([n_hco3_0, n_carb_0, n_co2_0, n_oh_0, n_hso4_0, n_h_0])

    # bounds-clip the trial log-variables so scipy's unconstrained hybr steps never explore physically absurd
    # molar amounts (which otherwise propagate into a negative ionic strength / math domain error in lr.py):
    ln_bounds_hi = np.log(np.array([n_carb_max, n_carb_max, n_carb_max, n_oh_max, n_sulf_max, n_h_max]) * 10.0
                          + 1e-300)
    ln_bounds_lo = math.log(deps)

    best = None
    for scale in (1.0, 0.3, 3.0, 0.05, 20.0, 0.01, 100.0):
        sol = root(residual, x0 + math.log(scale), method="hybr", options={"xtol": 1e-13, "maxfev": 20000})
        rn = float(np.sum(np.abs(residual(sol.x))))
        if best is None or rn < best[1]:
            best = (sol.x, rn)
        if rn < 1e-6:
            break
    log_vars, rn = best
    log_vars = np.clip(log_vars, ln_bounds_lo, ln_bounds_hi)
    n_hco3, n_carb, n_co2, n_oh, n_hso4, n_h = np.exp(log_vars)
    n_hso4 = min(n_hso4, n_sulf_max)
    n_sulf = n_sulf_max - n_hso4
    mol_neutral = mol_neutral_init.copy()
    mol_neutral[0] = max(n_h2o_init + (n_co2 - n_co2_init) - (n_oh - n_oh_init), np.finfo(float).eps)
    mol_neutral[idx_co2_neutral] = n_co2
    return CarbSulfResult(n_hco3, n_carb, n_co2, n_oh, n_hso4, n_h, n_sulf, mol_neutral, n_ca_result,
                          ca_precipitated, rn)

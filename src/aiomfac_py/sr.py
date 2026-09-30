"""Short-range modified-UNIFAC term (combinatorial + residual - reference), activity-coefficient part only.

Fortran counterpart: ModSRunifac.f90 -- SRunifac, SRgres, SRgcomb, SRgref (viscosity parts excluded).
Stateless: the Fortran code caches temperature-dependent and ion-reference values between calls
(``grefresh``, ``gcombrefresh``); here everything is recomputed on each call.

Notes
* ``PsiT = exp(-parA/T)`` is the original UNIFAC temperature dependence, used for the web-version dataset
  (nd = 1). The 3-parameter form (BRR, CRR) is only used for dataset numbers 500-800 / 2000-2434 and is not ported.
* ``solvmixrefnd`` (solvent mixture as ion reference state) is never enabled by the Fortran web driver; the
  branch is ported but has no Fortran reference data yet.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .system import SRSystem


@dataclass
class SRTerms:
    ln_gamma_c: np.ndarray       # combinatorial part                    (Fortran lnGaC)
    ln_gamma_r: np.ndarray       # residual part                         (lnGaR)
    ln_gamma_r_ref: np.ndarray   # residual reference (pure/infinite-dilution) part   (lnGaRref)
    ln_gamma_c_inf: np.ndarray   # combinatorial infinite-dilution value for ions     (lnGaCinf)
    ln_gamma_sr: np.ndarray      # total SR term = C + R - Rref (- Cinf for ions)     (gnsrln | gcsrln | gasrln)


def psi_t(system: SRSystem, T_K: float) -> np.ndarray:
    return np.exp(-system.parA / T_K)


def _residual(system: SRSystem, psi: np.ndarray, x: np.ndarray) -> np.ndarray:
    """SRgres: residual ln(gamma) of all species for species mole fractions ``x`` (no reference part)."""
    S1 = system.SRNY.T @ x                      # group amounts
    XG = S1 / S1.sum()
    TH = system.Q * XG / np.dot(system.Q, XG)
    S4 = psi.T @ TH                             # S4(i) = sum_k TH(k) PsiT(k, i)
    gam = system.Q * (1.0 - np.log(S4) - psi @ (TH / S4))
    return system.SRNY @ gam


def _residual_reference(system: SRSystem, psi: np.ndarray, xn: np.ndarray, solvmixrefnd: bool) -> np.ndarray:
    """SRgref: residual reference values (pure neutral components; ions only for the solvent-mixture reference)."""
    nn, nk = system.n_neutral, system.n_species
    ref = np.zeros(nk)
    for i in range(nn):
        if system.SRNY[i].sum() > 1:            # single-group components have a zero residual reference
            e = np.zeros(nk); e[i] = 1.0
            ref[i] = _residual(system, psi, e)[i]
    if solvmixrefnd and nk > nn:
        x = np.zeros(nk); x[:nn] = xn[:nn]
        ref[nn:] = _residual(system, psi, x)[nn:]
    return ref


def _combinatorial(system: SRSystem, x: np.ndarray, solvmixrefnd: bool):
    """SRgcomb: combinatorial ln(gamma) and the ion infinite-dilution values."""
    RS, QS, XL, nn = system.RS, system.QS, system.XL, system.n_neutral
    QSS, RSS = np.dot(QS, x), np.dot(RS, x)
    ViV = RS / RSS
    VQR = ViV * QSS / QS
    lnC = np.log(ViV) + 1.0 - ViV - 5.0 * QS * (np.log(VQR) + 1.0 - VQR)
    cinf = np.zeros(system.n_species)
    if system.n_species > nn:
        ion = slice(nn, None)
        if solvmixrefnd:
            xs = x[:nn] / x[:nn].sum()
            QSr, RSr, XLr = np.dot(QS[:nn], xs), np.dot(RS[:nn], xs), np.dot(XL[:nn], xs)
            B = 5.0 * QS[ion] * np.log(QS[ion] / QSr * RSr / RS[ion]) + XL[ion] - RS[ion] / RSr * XLr
            cinf[ion] = np.log(RS[ion] / RSr) + B
        else:   # water as reference solvent: RS = 0.92, QS = 1.40, l = -2.32
            B = 5.0 * QS[ion] * np.log(QS[ion] / 1.40 * 0.92 / RS[ion]) + XL[ion] - RS[ion] / 0.92 * (-2.32)
            cinf[ion] = np.log(RS[ion] / 0.92) + B
    return lnC, cinf


def sr_terms(system: SRSystem, T_K: float, x, xn=None, *, solvmixrefnd: bool = False) -> SRTerms:
    """SR contribution to ln(activity coefficients) of all species.

    ``x``  : species mole fractions (neutrals, cations, anions), Fortran ``X``
    ``xn`` : neutral-component mole fractions, Fortran ``XN`` (only used for ``solvmixrefnd``)
    """
    x = np.asarray(x, dtype=float)
    if x.shape != (system.n_species,):
        raise ValueError(f"x must have {system.n_species} entries, got {x.shape}")
    xn = x[:system.n_neutral] / x[:system.n_neutral].sum() if xn is None else np.asarray(xn, dtype=float)
    psi = psi_t(system, T_K)
    ref = _residual_reference(system, psi, xn, solvmixrefnd)
    lnR = _residual(system, psi, x)
    lnC, cinf = _combinatorial(system, x, solvmixrefnd)
    return SRTerms(lnC, lnR, ref, cinf, lnC + lnR - ref - cinf)

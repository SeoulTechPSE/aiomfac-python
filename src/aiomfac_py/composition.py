"""Composition conversions (ModCompScaleConversion.f90): input fractions -> mass fractions -> ion molalities."""
from __future__ import annotations

import numpy as np

_LOWVAL = 1.0e2 * np.finfo(float).eps


def mole_frac_to_mass_frac(x, mmass):
    """MoleFrac2MassFrac: the most abundant component takes up the rounding residual so that sum(w) == 1."""
    x, mmass = np.asarray(x, dtype=float), np.asarray(mmass, dtype=float)
    nc = x.size
    imax = int(np.argmax(x))                    # first maximum, like Fortran maxloc
    totmass = float(np.sum(x * mmass))
    w = np.zeros(nc)
    if totmass > 0.0:
        if imax > 0:
            w[:imax] = x[:imax] * mmass[:imax] / totmass
            sum1 = float(np.sum(w[:imax]))
        else:
            sum1 = 0.0
        if imax < nc - 1:
            w[imax + 1:] = x[imax + 1:] * mmass[imax + 1:] / totmass
            sum2 = float(np.sum(w[imax + 1:]))
        else:
            sum2 = 0.0
    else:
        w[:] = 1.0 / nc
        sum1, sum2 = float(np.sum(w[:nc - 1])), 0.0
    w[imax] = max(1.0 - sum1 - sum2, 0.0)
    return w


def input_to_mass_frac(inputconc, mmass, n_neutral, *, mole_basis: bool):
    """Inputconc_to_wtf, default case (the dataset-specific special cases are not part of the web version).

    ``inputconc`` has one entry per independent component (neutrals, then electrolytes); component 1 is
    obtained by difference. Returns mass fractions of all independent components.
    """
    inputconc = np.asarray(inputconc, dtype=float)
    if mole_basis:
        x = inputconc.copy()
        x[0] = 1.0 - float(np.sum(x[1:]))
        wtf = mole_frac_to_mass_frac(x, mmass)
    else:
        wtf = inputconc.copy()
        wtf[0] = 1.0 - float(np.sum(wtf[1:]))
    if np.any(wtf < 0.0):
        if abs(wtf.min()) < 1.0e-8:             # floating-point rounding problem: move the deficit to the largest
            imin, imax = int(np.argmin(wtf)), int(np.argmax(wtf))
            wtf[imax] += wtf[imin]
            wtf[imin] = 0.0
        else:
            raise ValueError(f"mass fraction of a component is negative ({wtf.min():.3e})")
    if wtf[:n_neutral].sum() < _LOWVAL and wtf[n_neutral:].sum() > _LOWVAL:
        wtf[1:] *= (1.0 - _LOWVAL)
        wtf[0] = 1.0 - float(np.sum(wtf[1:]))
    return wtf


def mass_frac_to_ion_molalities(wtf, mixture):
    """MassFrac2IonMolalities: cation/anion molalities [mol/kg solvent], zero-padded to NGI entries."""
    wtf = np.asarray(wtf, dtype=float)
    nn, ngi = mixture.n_neutral, mixture.ngi
    smc, sma = np.zeros(ngi), np.zeros(ngi)
    sum_wn = float(np.sum(wtf[:nn]))
    for k in range(mixture.n_electrol):
        ic, ia = mixture.elect_comps[k]
        cn, an = float(mixture.elect_nues[k, 0]), float(mixture.elect_nues[k, 1])
        ii = nn + k
        smc[mixture.cat_index[int(ic)]] += (wtf[ii] / mixture.mmass[ii]) * (cn / sum_wn)
        sma[mixture.an_index[int(ia)]] += (wtf[ii] / mixture.mmass[ii]) * (an / sum_wn)
    return smc, sma

"""Long-range (Debye-Hueckel type) term.

Fortran counterpart: the LR part of ModMRpart.f90 (LR_MR_activity). As in the Fortran code, all LR properties are
those of pure water at 298.15 K (density 997 kg/m3, relative permittivity 78.54); only the molar masses of the
neutral components enter through ``Mmass``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

DENS_WATER = 997.0          # [kg/m3] at 298.15 K
DIEL_WATER = 78.54          # static relative permittivity of water at 298.15 K


def debye_huckel_parameters(T_K: float) -> tuple[float, float]:
    """A, b of the extended Debye-Hueckel expression (Fortran A_DebyeHw, B_DebyeHw)."""
    sq = math.sqrt(DENS_WATER)
    return (1.327757e5 * sq / ((DIEL_WATER * T_K) ** 1.5),
            6.359696 * sq / ((DIEL_WATER * T_K) ** 0.5))


@dataclass
class LRTerms:
    ln_gamma_neutral: np.ndarray     # (n_neutral,)  gnlrln
    ln_gamma_cation: np.ndarray      # (NGI,)        gclrln (zero-padded beyond the number of cations)
    ln_gamma_anion: np.ndarray       # (NGI,)        galrln
    ionic_strength: float            # molality basis


def lr_terms(mmass_neutral, cation_z, anion_z, smc, sma, T_K: float) -> LRTerms:
    """cation_z/anion_z/smc/sma are zero-padded to NGI entries; ``anion_z`` may be negative."""
    A, b = debye_huckel_parameters(T_K)
    za2 = np.abs(anion_z) ** 2
    zc2 = np.asarray(cation_z) ** 2
    SI = 0.5 * float(np.sum(sma * za2 + smc * zc2))
    SI2 = math.sqrt(SI)
    bb = 1.0 + b * SI2
    term_i = bb - 1.0 / bb - 2.0 * math.log(bb)
    lrw1 = 2.0 * A / (b * b * b) * term_i
    gn = np.asarray(mmass_neutral) * lrw1
    lrw1 = A * SI2 / bb
    return LRTerms(gn, -zc2 * lrw1, -za2 * lrw1, SI)

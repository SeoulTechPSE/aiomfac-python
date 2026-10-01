"""AIOMFAC-VISC: predictive viscosity model for aqueous electrolyte solutions.

Ports the **aqueous-electrolyte** part only of:

    Lilek, J. and Zuend, A. (2022), "A predictive viscosity model for aqueous electrolytes and mixed
    organic-inorganic aerosol phases", Atmos. Chem. Phys., 22, 3203-3233,
    doi:10.5194/acp-22-3203-2022. [cited below as "LZ2022"]

What this implements, and how closely (read this before trusting the numbers)
-------------------------------------------------------------------------------
LZ2022 Sect. 2 derives, from Eyring's absolute-rate-theory expression for viscosity (their Eq. 2,
``eta = (h*NA/V) * exp(dg*/(RT))``), a molar Gibbs energy of activation for viscous flow, ``dg*``, built as an
*additive* combination (their Eq. 16) of: a water contribution (``dg*_w``, fixed by the known viscosity of pure
water, Eq. 3), a per-ion contribution (``dg*_i``, two fitted coefficients ``c0,i``/``c1,i`` per ion applied to
that ion's own AIOMFAC-computed molal activity, Eq. 13), and a per-cation-anion-pair contribution (``dg*_c,a``,
one fitted coefficient ``c_c,a`` per pair, scaled by the square root of molal ionic strength, Eq. 14), the last
weighted by a charge- and abundance-based fractional term ``tau_c,a`` (Eq. 17-18) so that an arbitrary mixture
of cations and anions is treated as a weighted combination of all its possible binary salts. This module
implements exactly that: Eq. (2)-(19) of LZ2022, reusing this package's own ``ActivityModel`` for the ion molal
activities and activity coefficients the formulas need (LZ2022 Sect. 3.3 notes this is exactly how the
viscosity module is meant to plug into AIOMFAC: "A number of input quantities are needed prior to calling the
aqueous electrolyte solution viscosity module within AIOMFAC[:] the mole fractions of water and the ions, the
activity coefficients, and the relative ionic volumes[, which] are all available through the AIOMFAC
interface").

**Not implemented: LZ2022 Sect. 3's organic-inorganic mixing models.** LZ2022 extends AIOMFAC-VISC beyond
pure aqueous electrolytes to mixtures that also contain organic compounds, by combining this electrolyte model
with a separate aqueous-organic viscosity model (Gervasi et al., 2020, not part of this port) through one of
several competing mixing rules the paper evaluates (a log-viscosity mass-weighted rule, a Vignes-type rule, and
a ZSR-based rule -- LZ2022 Sect. 3.4.1-3.4.4) -- a substantially larger scope addition (a second, separate
organic-viscosity parameterization, plus a choice among mixing rules the paper itself does not settle on a
single winner for) than this module attempts. This module is therefore usable only for **organic-free aqueous
electrolyte mixtures** -- water plus one or more of the 17 ions in ``ION_SUBGROUP`` below -- not for
organic-containing aerosol phases, which is the paper's own eventual application (its title notwithstanding).

Supported ions, and parameter provenance
-------------------------------------------
LZ2022 fitted 58 coefficients describing 17 ions (the same 7 cations and 10 anions AIOMFAC's activity-
coefficient part already supports: H+, Li+, Na+, K+, NH4+, Mg2+, Ca2+, Cl-, Br-, NO3-, HSO4-, SO4(2-), I-,
CO3(2-), HCO3-, OH-, IO3-) and 42 (of 70 possible) independently fitted cation-anion pair coefficients -- the
rest are substituted from a chemically similar pair per LZ2022's own Table 7 footnote (e.g. Ca2+/Br- uses the
Ca2+/Cl- value) and already carried as such in ``CATION_ANION_C`` below (``SUBSTITUTED_PAIRS`` records which).
All ion coefficients (``ION_C0_C1``), ionic volumes (``ION_RTH``), and cation-anion coefficients
(``CATION_ANION_C``) below are transcribed directly from LZ2022's Tables 6, 1, and 7 respectively -- not
re-fitted or approximated. The fitted volume-correction term ``CV = 1.679827`` and the water-viscosity
correlation (Dehaoui, Issenmann, Caupin, 2015, PNAS 112, 12020-12025, the parameterization LZ2022 Sect. 3.3
cites for pure-water viscosity) are likewise taken directly from their sources.

The ionic and water molar volumes (``ION_RTH`` times, and water's own 0.92, the reference subgroup volume
``V_REF_M3_PER_MOL``) are **not** physical liquid molar volumes -- they are AIOMFAC's own (temperature-
independent) relative van der Waals group-contribution volumes, the same ones the UNIFAC-based short-range
activity-coefficient term already uses (LZ2022 Sect. 2.4: "AIOMFAC uses relative van der Waals volumes ... The
reference subgroup is used to calculate relative volumes for neutral molecules as well. For example, the
relative volume for H2O is 0.92"). Using these (rather than, say, a temperature-dependent liquid-water density
correlation) is confirmed correct by direct back-calculation: LZ2022 states that pure water's
``dg*_w/(RT) = 3.44`` at 298.15 K; reproducing that from Eq. (3) with this module's water viscosity and
``V_w = 0.92 * V_REF_M3_PER_MOL`` gives ``3.441`` -- matching to the precision LZ2022 gives, and confirming
both the water viscosity correlation and the (T-independent) volume convention used here. This also matches
LZ2022's own design philosophy that "AIOMFAC-VISC coefficients are assumed to be temperature-independent ...
the temperature-dependent pure-component viscosity of water already sufficiently captures the temperature
dependence" -- so there would be no reason for the ionic/water volumes to carry their own T-dependence either.

Units: SI throughout (Pa s for viscosity, m^3/mol for molar volumes, mol/kg for molality, K for temperature).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .model import ActivityModel, ActivityTerms

R_GAS = 8.314462618        # J / (mol K)
H_PLANCK = 6.62607015e-34  # J s
N_AVOGADRO = 6.02214076e23  # 1 / mol

#: Abrams & Prausnitz (1975) / Fredenslund et al. (1975) reference subgroup volume, used throughout AIOMFAC to
#: define *relative* van der Waals volumes (LZ2022 Sect. 2.4).
V_REF_M3_PER_MOL = 15.17e-6

#: AIOMFAC's own relative van der Waals volume for water (the same value the short-range/UNIFAC term uses).
WATER_RTH = 0.92

#: Fitted volume-correction term, LZ2022 Sect. 4.1 ("The fitted cv value is 1.679827").
CV = 1.679827

#: AIOMFAC/UNIFAC cation subgroup IDs (same convention as the rest of this package, e.g. gp_partition.py's
#: notebook examples) -> ion name.
CATION_SUBGROUP = {205: "H+", 201: "Li+", 202: "Na+", 203: "K+", 204: "NH4+", 223: "Mg2+", 221: "Ca2+"}
#: AIOMFAC/UNIFAC anion subgroup IDs -> ion name.
ANION_SUBGROUP = {242: "Cl-", 243: "Br-", 245: "NO3-", 248: "HSO4-", 261: "SO4--", 244: "I-",
                  262: "CO3--", 250: "HCO3-", 247: "OH-", 246: "IO3-"}
#: name -> subgroup ID, both cations and anions (inverse of the two dicts above).
ION_SUBGROUP = {name: sub for sub, name in {**CATION_SUBGROUP, **ANION_SUBGROUP}.items()}

#: LZ2022 Table 1: relative van der Waals ionic volume (R_tH), by ion name. Adapted from Zuend et al. (2008)
#: and Yin et al. (2022); identical to the values AIOMFAC's own activity-coefficient part uses for these ions.
ION_RTH = {
    "H+": 1.78, "Li+": 0.61, "Na+": 0.38, "K+": 0.44, "NH4+": 0.69, "Mg2+": 5.44, "Ca2+": 2.24,
    "Cl-": 0.99, "Br-": 1.25, "NO3-": 0.95, "HSO4-": 1.65, "SO4--": 3.34, "I-": 1.77,
    "CO3--": 4.62, "HCO3-": 2.89, "OH-": 2.99, "IO3-": 1.52,
}

#: LZ2022 Table 6: AIOMFAC-VISC ion coefficients (c0,i, c1,i), by ion name.
ION_C0_C1 = {
    "H+": (1.737603e-3, 8.200895), "Li+": (1.574561e-1, 9.571695), "K+": (4.355008e-1, 3.521816),
    "Na+": (2.320511e-1, 8.708063), "Ca2+": (4.902515e-2, 1.224856e1), "NH4+": (2.206851e-1, 4.396357),
    "Mg2+": (3.250157e-2, 2.975466e1), "Cl-": (2.580008e-2, 3.834502), "Br-": (9.704206e-3, 9.000332e-1),
    "NO3-": (1.424428, 1.633573), "SO4--": (1.010392e-11, 1.845397e1), "HSO4-": (2.147275e-2, 7.640526),
    "I-": (2.130622e-1, 1.925830e-2), "CO3--": (1.010392e-11, 2.421959e1), "HCO3-": (1.174492, 1.519666e1),
    "OH-": (1.010392e-11, 1.521291e1), "IO3-": (1.600203e1, 3.393813e1),
}

#: LZ2022 Table 7: AIOMFAC-VISC cation-anion pair coefficient c_c,a, by (cation name, anion name). Transcribed
#: in full (all 7 x 10 = 70 entries LZ2022 reports), including the entries LZ2022 flags with "*" as substituted
#: from a chemically similar pair rather than independently fitted (``SUBSTITUTED_PAIRS`` below records which).
CATION_ANION_C = {
    ("H+", "Cl-"): 3.264145e-1, ("Li+", "Cl-"): 1.444564, ("K+", "Cl-"): 3.392420e-1,
    ("Na+", "Cl-"): 4.594814e-1, ("Ca2+", "Cl-"): 4.694481, ("NH4+", "Cl-"): 1.010392e-11,
    ("Mg2+", "Cl-"): 2.478116,
    ("H+", "Br-"): 1.432635, ("Li+", "Br-"): 1.702737, ("K+", "Br-"): 1.428021,
    ("Na+", "Br-"): 1.554556, ("Ca2+", "Br-"): 4.694481, ("NH4+", "Br-"): 1.010392e-11,
    ("Mg2+", "Br-"): 2.478116,
    ("H+", "NO3-"): 1.010392e-11, ("Li+", "NO3-"): 1.822536e-2, ("K+", "NO3-"): 1.749285,
    ("Na+", "NO3-"): 1.804247, ("Ca2+", "NO3-"): 6.777004, ("NH4+", "NO3-"): 5.675336e-1,
    ("Mg2+", "NO3-"): 8.193433e-1,
    ("H+", "SO4--"): 2.531472e-1, ("Li+", "SO4--"): 3.707555, ("K+", "SO4--"): 6.871033e-2,
    ("Na+", "SO4--"): 3.178487e-1, ("Ca2+", "SO4--"): 3.397288, ("NH4+", "SO4--"): 1.010392e-11,
    ("Mg2+", "SO4--"): 3.397288,
    ("H+", "I-"): 1.105883, ("Li+", "I-"): 1.345442, ("K+", "I-"): 1.712015,
    ("Na+", "I-"): 1.858026, ("Ca2+", "I-"): 4.694481, ("NH4+", "I-"): 1.010392e-11,
    ("Mg2+", "I-"): 2.478116,
    ("H+", "IO3-"): 3.406453, ("Li+", "IO3-"): 1.822536e-2, ("K+", "IO3-"): 1.749285,
    ("Na+", "IO3-"): 1.804247, ("Ca2+", "IO3-"): 6.777004, ("NH4+", "IO3-"): 5.675336e-1,
    ("Mg2+", "IO3-"): 8.193433e-1,
    ("H+", "CO3--"): 2.307056e1, ("Li+", "CO3--"): 3.803271, ("K+", "CO3--"): 1.525729,
    ("Na+", "CO3--"): 3.803271, ("Ca2+", "CO3--"): 3.397288, ("NH4+", "CO3--"): 1.525729,
    ("Mg2+", "CO3--"): 3.397288,
    ("H+", "OH-"): 2.458673e1, ("Li+", "OH-"): 1.705202, ("K+", "OH-"): 3.450629e-2,
    ("Na+", "OH-"): 1.010392e-11, ("Ca2+", "OH-"): 4.694481, ("NH4+", "OH-"): 3.450629e-2,
    ("Mg2+", "OH-"): 2.478116,
    ("H+", "HCO3-"): 2.372604e1, ("Li+", "HCO3-"): 1.393572e-2, ("K+", "HCO3-"): 1.010392e-11,
    ("Na+", "HCO3-"): 1.393572e-2, ("Ca2+", "HCO3-"): 6.777004, ("NH4+", "HCO3-"): 1.010392e-11,
    ("Mg2+", "HCO3-"): 8.193433e-1,
    ("H+", "HSO4-"): 2.504231e-1, ("Li+", "HSO4-"): 1.444564, ("K+", "HSO4-"): 3.392420e-1,
    ("Na+", "HSO4-"): 1.319013, ("Ca2+", "HSO4-"): 4.694481, ("NH4+", "HSO4-"): 1.010392e-11,
    ("Mg2+", "HSO4-"): 2.478116,
}

#: Pairs LZ2022's Table 7 flags with "*" (no direct measurements; coefficient substituted from a chemically
#: similar pair, e.g. Ca2+/Br- <- Ca2+/Cl-). Informational only -- ``CATION_ANION_C`` already carries the
#: substituted numeric values, so no special handling is needed to use them.
SUBSTITUTED_PAIRS = frozenset({
    ("Ca2+", "Br-"), ("NH4+", "Br-"), ("Mg2+", "Br-"),
    ("Ca2+", "SO4--"),
    ("Ca2+", "I-"), ("NH4+", "I-"), ("Mg2+", "I-"),
    ("Li+", "IO3-"), ("K+", "IO3-"), ("Na+", "IO3-"), ("Ca2+", "IO3-"), ("NH4+", "IO3-"), ("Mg2+", "IO3-"),
    ("Li+", "CO3--"), ("Ca2+", "CO3--"), ("NH4+", "CO3--"), ("Mg2+", "CO3--"),
    ("Ca2+", "OH-"), ("NH4+", "OH-"), ("Mg2+", "OH-"),
    ("K+", "HCO3-"), ("Ca2+", "HCO3-"), ("NH4+", "HCO3-"), ("Mg2+", "HCO3-"),
    ("Li+", "HSO4-"), ("K+", "HSO4-"), ("Ca2+", "HSO4-"), ("NH4+", "HSO4-"), ("Mg2+", "HSO4-"),
})


def water_viscosity_pas(T_K: float) -> float:
    """Pure-water dynamic viscosity (Pa s), Dehaoui, Issenmann & Caupin (2015, PNAS 112, 12020), the power-law
    fit LZ2022 Sect. 3.3 cites: ``eta = eta0 * (T/Ts - 1)^(-gamma)``, valid 239.15-373.15 K (fit parameters
    eta0 = 1.3788e-4 Pa s, Ts = 225.66 K, gamma = 1.6438). Gives 8.93e-4 Pa s at 298.15 K, matching the
    commonly cited ~8.9e-4 Pa s value for water at room temperature."""
    eta0, T_s, gamma = 1.3788e-4, 225.66, 1.6438
    return eta0 * (T_K / T_s - 1.0) ** (-gamma)


@dataclass
class ViscosityResult:
    eta_pas: float          # predicted dynamic viscosity, Pa s
    log10_eta_pas: float    # log10(eta / 1 Pa s) -- the scale LZ2022 itself plots on, given the huge dynamic range
    dg_star_over_RT: float  # the full mixture's dimensionless molar Gibbs energy of activation for viscous flow
    V_m3_per_mol: float     # the mixture's effective mean molar volume (Eq. 19), m^3/mol


def electrolyte_viscosity(model: ActivityModel, result: ActivityTerms, T_K: float, *,
                           cv: float = CV) -> ViscosityResult:
    """AIOMFAC-VISC predicted dynamic viscosity of an **organic-free** aqueous electrolyte mixture (LZ2022
    Eq. 2-19), given an :class:`ActivityModel` built from exactly ``[water, *electrolytes]`` (water must be
    the first, and only neutral, component) and the :class:`ActivityTerms` from calling
    ``model.evaluate(..., basis=...)`` on it at temperature ``T_K``.

    Every ion subgroup present must be one of the 17 in ``ION_SUBGROUP``; a ``KeyError`` naming the unsupported
    subgroup is raised otherwise. See the module docstring for everything this does and does not cover.
    """
    sr = model.mixture.sr
    if sr.n_neutral != 1:
        raise ValueError("electrolyte_viscosity requires exactly one neutral component (water) -- this module "
                          "does not implement LZ2022's organic-inorganic mixing models (see module docstring)")

    nn = sr.n_neutral
    x_w = float(result.x[0])
    eta_w = water_viscosity_pas(T_K)
    V_w = WATER_RTH * V_REF_M3_PER_MOL
    dg_w_over_RT = math.log(eta_w * V_w / (H_PLANCK * N_AVOGADRO))

    dg_over_RT = x_w * dg_w_over_RT
    V = x_w * V_w

    # Per-ion contributions (Eq. 13) and the mixture's mean molar volume (Eq. 19); also collect per-ion
    # (name, mole fraction, molality, charge) for the cation-anion pair term below.
    ions = []  # (name, x_i, m_i, z_i, is_cation)
    for k, sub in enumerate(sr.cations):
        name = CATION_SUBGROUP.get(sub)
        if name is None:
            raise KeyError(f"cation subgroup {sub} is not one of the 17 ions AIOMFAC-VISC supports")
        idx = nn + k
        x_i, m_i = float(result.x[idx]), float(result.smc[k])
        a_i_ref = m_i * math.exp(result.ln_gamma[idx])
        c0, c1 = ION_C0_C1[name]
        dg_i_over_RT = c0 * math.log(a_i_ref) + c1
        dg_over_RT += x_i * dg_i_over_RT
        V += cv * x_i * ION_RTH[name] * V_REF_M3_PER_MOL
        ions.append((name, x_i, m_i, sr.cation_z[k], True))

    for k, sub in enumerate(sr.anions):
        name = ANION_SUBGROUP.get(sub)
        if name is None:
            raise KeyError(f"anion subgroup {sub} is not one of the 17 ions AIOMFAC-VISC supports")
        idx = nn + sr.n_cation + k
        x_i, m_i = float(result.x[idx]), float(result.sma[k])
        a_i_ref = m_i * math.exp(result.ln_gamma[idx])
        c0, c1 = ION_C0_C1[name]
        dg_i_over_RT = c0 * math.log(a_i_ref) + c1
        dg_over_RT += x_i * dg_i_over_RT
        V += cv * x_i * ION_RTH[name] * V_REF_M3_PER_MOL
        ions.append((name, x_i, m_i, sr.anion_z[k], False))

    # Molal ionic strength (Eq. 15) and the cation-anion pair term (Eq. 14, 16-18).
    ionic_strength = 0.5 * sum(m_i * z_i ** 2 for _, _, m_i, z_i, _ in ions)
    sqrt_I = math.sqrt(ionic_strength)

    cations = [(name, x_i, z_i) for name, x_i, _, z_i, is_cat in ions if is_cat]
    anions = [(name, x_i, z_i) for name, x_i, _, z_i, is_cat in ions if not is_cat]
    total_neg_charge = sum(x_a * abs(z_a) for _, x_a, z_a in anions)

    if total_neg_charge > 0.0:
        for name_c, x_c, z_c in cations:
            for name_a, x_a, z_a in anions:
                psi_a = (x_a * abs(z_a)) / total_neg_charge
                nu_c = abs(z_a) // math.gcd(int(round(z_c)), int(round(abs(z_a))))
                tau_ca = (x_c / nu_c) * psi_a
                c_ca = CATION_ANION_C[(name_c, name_a)]
                dg_ca_over_RT = c_ca * sqrt_I
                dg_over_RT += tau_ca * dg_ca_over_RT

    eta = (H_PLANCK * N_AVOGADRO / V) * math.exp(dg_over_RT)
    return ViscosityResult(eta_pas=eta, log10_eta_pas=math.log10(eta), dg_star_over_RT=dg_over_RT, V_m3_per_mol=V)

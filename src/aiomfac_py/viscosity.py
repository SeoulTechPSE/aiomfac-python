"""AIOMFAC-VISC: predictive viscosity model for aqueous electrolyte, aqueous organic, and mixed
organic-inorganic solutions.

Ports:

    Lilek, J. and Zuend, A. (2022), "A predictive viscosity model for aqueous electrolytes and mixed
    organic-inorganic aerosol phases", Atmos. Chem. Phys., 22, 3203-3233,
    doi:10.5194/acp-22-3203-2022. [cited below as "LZ2022"]

    Gervasi, N. R., Topping, D. O., and Zuend, A. (2020), "A predictive group-contribution model for the
    viscosity of aqueous organic aerosol", Atmos. Chem. Phys., 20, 2987-3008, doi:10.5194/acp-20-2987-2020.
    [cited below as "G2020"]

    DeRieux, W.-S. W., Li, Y., Lin, P., Laskin, J., Laskin, A., Bertram, A. K., Nizkorodov, S. A., and
    Shiraiwa, M. (2018), "Predicting the glass transition temperature and viscosity of secondary organic
    material using molecular composition", Atmos. Chem. Phys., 18, 6331-6351, doi:10.5194/acp-18-6331-2018.
    [cited below as "D2018"]

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

Organic-inorganic mixtures: what's implemented, and what still isn't
---------------------------------------------------------------------
LZ2022 Sect. 3 extends AIOMFAC-VISC to mixtures that also contain organic compounds by combining the
electrolyte model above with a separate aqueous-organic viscosity model (G2020) through one of three mixing
rules. This module now implements:

* **G2020's aqueous organic viscosity model itself** (``organic_mixture_viscosity``, ``pure_organic_viscosity_
  vtf``) -- the "GC-UNIMOD"/AIOMFAC-VISC-organic combinatorial-plus-residual mixture viscosity equations
  (G2020 Eq. 1-9). These need no new fitted parameters beyond what this package's existing short-range/UNIFAC
  machinery (``sr.py``) already has (the subgroup R, Q and interaction tables). Validated (see
  ``organic_mixture_viscosity``'s docstring for details and the PEG workaround) against G2020's own published
  error statistics (Supplement Table S5) and against the Fortran AIOMFAC-VISC organic code.
* **Two of LZ2022's three mixing rules**, "aquelec" and "aquorg" (``aquelec_viscosity``, ``aquorg_viscosity``;
  LZ2022 Sect. 3.4.1-3.4.2) -- both need only a fixed, non-iterative sequence of calls into the electrolyte and
  organic models above, with ion molalities/mole fractions or organic mole fractions rescaled as LZ2022
  describes; neither requires any additional fitted parameters.
* **D2018's predictive glass-transition-temperature (Tg) model** (``predict_tg_derieux2018``; D2018 Eq. 2,
  Table 1), the parameterization G2020's own Eq. 11 relies on for estimating an organic's pure-component
  viscosity from molecular formula alone (number of C, H, and O atoms) when no measured Tg is available. This
  is a closed-form equation with published fitted coefficients, so it needed no new fitted parameters of its
  own; validated directly against D2018's own worked example (stachyose, C24H42O21: this implementation gives
  394.3 K, matching D2018's own quoted 394 K to the precision given). Feed its result into
  ``pure_organic_viscosity_vtf`` to get a pure-component viscosity for ``organic_mixture_viscosity``.
  G2020 itself flags this Tg-to-viscosity route as the model's single largest source of uncertainty (their
  Fig. 2: several orders of magnitude off for some compounds) -- prefer a measured Tg or pure-component
  viscosity when one is available, as G2020's own validation figures do.

* **The machine-learning-based Tg predictor of Armeli, Peters, and Koop (2023), ACS Omega 8, 12298-12309**
  (an "extra trees" ensemble regressor, trained on the ~355-compound Bielefeld Molecular Organic Glasses
  database; reported MAE ~12-13 K, somewhat better than D2018's ~21 K prediction band). Unlike D2018's Eq. 2,
  this model has no closed-form expression to port -- it is implemented as a *separate*, opt-in subpackage,
  ``aiomfac_py.tgml_armeli``, that loads the authors' own trained model files directly (not reimplemented or
  retrained). It needs ``scikit-learn`` (the ``tgml`` extra) and, for its SMILES-input mode, ``rdkit`` as
  well (the ``tgml-smiles`` extra) -- both install normally alongside the rest of this package, with no
  special version pin, since the vendored model files were migrated to a format any reasonably current
  ``scikit-learn`` can load and its SMILES-mode descriptors are looked up by name rather than position. See
  that subpackage's own module docstring and ``PROVENANCE.md`` for the full story, including a small known
  accuracy caveat for SMILES mode vs. the exact RDKit version the original authors trained on.
  ``predict_tg_derieux2018`` above remains the dependency-free option that works directly alongside the rest
  of this package.

**Not implemented:**

* **LZ2022's third mixing rule, the ZSR-style rule** (Sect. 3.4.3) -- unlike aquelec/aquorg, it requires
  solving a nonlinear equation (matching each subsystem's water activity to the full mixture's RH) iteratively
  for every evaluation, which is a materially larger piece of numerical machinery than this port adds here.
  LZ2022 itself does not single out ZSR, aquelec, or aquorg as uniformly best (Sect. 3.4.4), so having two of
  the three is a reasonable, honestly-documented partial implementation rather than an arbitrary omission.

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
import warnings
from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .model import ActivityModel, ActivityTerms
from .sr import psi_t, sr_terms

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


def _collect_ions(sr, result: ActivityTerms, nn: int):
    """Per-ion (name, mole fraction, molality, charge, reference activity, is_cation) tuples for every ion in
    ``sr``, read off an already-evaluated ``result`` (LZ2022 Eq. 9's ion activity, ``a_i = m_i * gamma_i``, the
    same product ``electrolyte_viscosity`` and the mixing functions below all reuse)."""
    ions = []
    for k, sub in enumerate(sr.cations):
        name = CATION_SUBGROUP.get(sub)
        if name is None:
            raise KeyError(f"cation subgroup {sub} is not one of the 17 ions AIOMFAC-VISC supports")
        idx = nn + k
        x_i, m_i = float(result.x[idx]), float(result.smc[k])
        a_i_ref = m_i * math.exp(result.ln_gamma[idx])
        ions.append((name, x_i, m_i, sr.cation_z[k], a_i_ref, True))
    for k, sub in enumerate(sr.anions):
        name = ANION_SUBGROUP.get(sub)
        if name is None:
            raise KeyError(f"anion subgroup {sub} is not one of the 17 ions AIOMFAC-VISC supports")
        idx = nn + sr.n_cation + k
        x_i, m_i = float(result.x[idx]), float(result.sma[k])
        a_i_ref = m_i * math.exp(result.ln_gamma[idx])
        ions.append((name, x_i, m_i, sr.anion_z[k], a_i_ref, False))
    return ions


def _electrolyte_core(x_w: float, ions, eta_w: float, cv: float) -> ViscosityResult:
    """LZ2022 Eq. 2-19 given the water mole fraction, a per-ion tuple list (as ``_collect_ions`` returns), and
    the pure-component ("solvent") viscosity to use for water -- ordinarily ``water_viscosity_pas(T_K)``, but
    the mixing functions below substitute an "aware" pseudo-pure-water value instead (LZ2022 Sect. 3.4.1-3.4.2)
    without otherwise changing this formula."""
    V_w = WATER_RTH * V_REF_M3_PER_MOL
    dg_w_over_RT = math.log(eta_w * V_w / (H_PLANCK * N_AVOGADRO))

    dg_over_RT = x_w * dg_w_over_RT
    V = x_w * V_w

    for name, x_i, _m_i, _z_i, a_i_ref, _is_cat in ions:
        c0, c1 = ION_C0_C1[name]
        dg_over_RT += x_i * (c0 * math.log(a_i_ref) + c1)
        V += cv * x_i * ION_RTH[name] * V_REF_M3_PER_MOL

    ionic_strength = 0.5 * sum(m_i * z_i ** 2 for _, _, m_i, z_i, _, _ in ions)
    sqrt_I = math.sqrt(ionic_strength)

    cations = [(name, x_i, z_i) for name, x_i, _, z_i, _, is_cat in ions if is_cat]
    anions = [(name, x_i, z_i) for name, x_i, _, z_i, _, is_cat in ions if not is_cat]
    total_neg_charge = sum(x_a * abs(z_a) for _, x_a, z_a in anions)

    if total_neg_charge > 0.0:
        for name_c, x_c, z_c in cations:
            for name_a, x_a, z_a in anions:
                psi_a = (x_a * abs(z_a)) / total_neg_charge
                nu_c = abs(z_a) // math.gcd(int(round(z_c)), int(round(abs(z_a))))
                tau_ca = (x_c / nu_c) * psi_a
                c_ca = CATION_ANION_C[(name_c, name_a)]
                dg_over_RT += tau_ca * c_ca * sqrt_I

    eta = (H_PLANCK * N_AVOGADRO / V) * math.exp(dg_over_RT)
    return ViscosityResult(eta_pas=eta, log10_eta_pas=math.log10(eta), dg_star_over_RT=dg_over_RT, V_m3_per_mol=V)


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
        raise ValueError("electrolyte_viscosity requires exactly one neutral component (water) -- for mixtures "
                          "that also contain organics, use aquelec_viscosity or aquorg_viscosity instead")
    ions = _collect_ions(sr, result, sr.n_neutral)
    return _electrolyte_core(float(result.x[0]), ions, water_viscosity_pas(T_K), cv)


# ---------------------------------------------------------------------------------------------------------------
# Organic-inorganic extension: Gervasi et al. (2020) aqueous organic viscosity model ("GC-UNIMOD"/AIOMFAC-VISC
# organic part) plus two of Lilek and Zuend (2022) Sect. 3.4's three mixing rules for combining it with the
# electrolyte model above. See the module docstring for exactly what is and is not covered.
# ---------------------------------------------------------------------------------------------------------------

#: UNIFAC/AIOMFAC lattice coordination number (Zuend et al., 2008; Abrams and Prausnitz, 1975). Appears
#: elsewhere in this package only implicitly, as the factor ``5.0 = Z_COORD / 2`` in ``sr._combinatorial``.
Z_COORD = 10.0


#: DeRieux et al. (2018, ACP 18, 6331-6351, "D2018") Table 1: fitted coefficients of their Eq. (2) predictive
#: glass-transition-temperature parameterization, by elemental composition class. The CH class has no oxygen
#: term (b_O = b_CO = 0.0, simply unused whenever n_O = 0).
_DERIEUX2018_TG_COEFFS = {
    "CH":  dict(n_C0=1.96, b_C=61.99, b_H=-113.33, b_CH=28.74, b_O=0.0, b_CO=0.0),
    "CHO": dict(n_C0=12.13, b_C=10.95, b_H=-41.82, b_CH=21.61, b_O=118.96, b_CO=-24.38),
}


def predict_tg_derieux2018(n_C: int, n_H: int, n_O: int = 0) -> float:
    """Predict an organic compound's glass transition temperature, Tg (K), from its elemental composition
    (numbers of carbon, hydrogen, and oxygen atoms) via DeRieux et al. (2018, ACP 18, 6331-6351, "D2018")
    Eq. (2), the predictive Tg model that Gervasi, Topping and Zuend (2020)'s Eq. (11) builds on:

        Tg = (n_C^0 + ln(n_C)) b_C + ln(n_H) b_H + ln(n_C) ln(n_H) b_CH + ln(n_O) b_O + ln(n_C) ln(n_O) b_CO

    with fitted coefficients (D2018 Table 1) depending on whether the compound contains oxygen (the "CHO"
    class, ``n_O > 0``) or not (the "CH" class, ``n_O == 0``). D2018 fit this to compounds with molar mass up
    to ~1100 g/mol; it supersedes the lower-molar-mass-only parameterization of Shiraiwa et al. (2017, D2018
    Eq. 1) that AIOMFAC-VISC/G2020 does not use. D2018 report individual-compound predictions accurate to
    within about +/-21 K (their Fig. 1c prediction band) and recommend using the *median* Tg across a
    multi-component mixture's compounds for a more precise mixture-level estimate.

    This only covers CH and CHO compounds (D2018's own stated scope); it raises ``ValueError`` for an
    unphysical atom count rather than silently extrapolating to compounds containing N, S, or halogens, for
    which D2018 do not provide coefficients (a CHON extension exists in Li et al., 2020, ACP 20, 8103-8122,
    which is a different paper not ported here).
    """
    if n_C <= 0 or n_H <= 0 or n_O < 0:
        raise ValueError(f"predict_tg_derieux2018 requires n_C > 0 and n_H > 0 (got n_C={n_C}, n_H={n_H}, "
                          f"n_O={n_O})")
    c = _DERIEUX2018_TG_COEFFS["CHO" if n_O > 0 else "CH"]
    ln_nC, ln_nH = math.log(n_C), math.log(n_H)
    tg = (c["n_C0"] + ln_nC) * c["b_C"] + ln_nH * c["b_H"] + ln_nC * ln_nH * c["b_CH"]
    if n_O > 0:
        ln_nO = math.log(n_O)
        tg += ln_nO * c["b_O"] + ln_nC * ln_nO * c["b_CO"]
    return tg


def pure_organic_viscosity_vtf(T_K: float, Tg_K: float, D: float | None = None) -> float:
    """Pure-component organic viscosity from a glass transition temperature via the modified Vogel-Tammann-
    Fulcher equation (Gervasi, Topping and Zuend, 2020, ACP 20, 2987-3008, Eq. 11-12; "G2020" below), given ``Tg_K``
    (e.g. a measured or literature-estimated glass transition temperature -- **not** predicted here, see the
    module docstring) and a fragility parameter ``D`` (dimensionless; G2020 Sect. 2.2.1: typically 5-30, with
    10 a reasonable default at/above Tg). If ``D`` is omitted, G2020's own simulation convention is used:
    ``D = 10`` normally, or ``D = 30`` if ``Tg_K`` is above ``T_K`` (i.e. the system is simulated below its own
    glass transition, where a fragility of 10 would drastically overestimate the pure-component viscosity;
    G2020 Sect. 2.3: "we choose to assign a fragility parameter of D = 10 for all organic compounds, with the
    exception of those whose Tg is warmer than the simulation temperature[, for which] the FSC provides us with
    the theoretical basis to assign a fragility parameter of D = 30").

    G2020 itself flags this pure-component viscosity estimate as the single largest source of uncertainty in
    the whole model (their Fig. 2: predicted vs. experimental pure-component viscosity can disagree by several
    orders of magnitude for some compounds, e.g. glycerol) -- prefer a measured pure-component viscosity over
    this function wherever one is available (as G2020's own validation figures do).
    """
    if D is None:
        D = 30.0 if Tg_K > T_K else 10.0
    T0 = 39.17 * Tg_K / (D + 39.17)
    if T_K <= T0:
        raise ValueError(f"T_K={T_K:g} K is at or below the Vogel temperature T0={T0:.2f} K implied by "
                          f"Tg={Tg_K:g} K, D={D:g} -- the VTF viscosity diverges there and is not meaningful")
    log10_eta0 = -5.0 + 0.434 * T0 * D / (T_K - T0)
    return 10.0 ** log10_eta0


@dataclass
class OrganicViscosityResult:
    eta_pas: float          # predicted dynamic viscosity, Pa s
    log10_eta_pas: float    # log10(eta / 1 Pa s)
    ln_eta: float           # ln(eta / 1 Pa s) -- the left-hand side of G2020 Eq. 1 directly


def _is_pure_water(component) -> bool:
    return tuple(component.subgroups) == ((16, 1),)


def _group_residual_L(sr, psi: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Per-group ``L_k = sum_m Gamma_{m,k} ln(Psi_{m,k})`` (G2020 Eq. 4's sum, with Eq. 7-8's Gamma/Psi) for
    neutral-species mole fractions ``x``, evaluated at whatever composition is passed in (the real mixture for
    the "mixture" term of Eq. 3, or a single-component unit vector for the "ref" term -- mirroring exactly how
    ``sr._residual``/``_residual_reference`` compute the analogous thermodynamic quantity)."""
    S1 = sr.SRNY.T @ x
    total = S1.sum()
    if total <= 0.0:
        return np.zeros_like(sr.Q)
    XG = S1 / total
    TH = sr.Q * XG / np.dot(sr.Q, XG)
    S4 = psi.T @ TH                                    # S4[k] = sum_m TH[m] * Psi[m, k]
    with np.errstate(divide="ignore", invalid="ignore"):
        Gamma = (TH[:, None] * psi) / S4[None, :]       # Gamma[m, k]
        ln_psi = np.log(psi)
        term = Gamma * ln_psi
    return np.where(np.isfinite(term), term, 0.0).sum(axis=0)


PEG_TREATMENTS = ("aiomfac_web_v3.14", None)


def _peg_components(components) -> list[int]:
    """Indices of components with more than one CH2OCH2[PEG] subgroup (154) -- the Fortran's
    ``ITAB(i,154) > 1`` test for ``is_PEG_molec`` (AIOMFAC-web v3.14, ``ModSRunifac.f90``)."""
    return [i for i, comp in enumerate(components) if dict(comp.subgroups).get(154, 0) > 1]


def organic_mixture_viscosity(model: ActivityModel, x_neutral, T_K: float,
                               eta0_pas: Mapping[int, float] | None = None, *,
                               peg_treatment: str | None = "aiomfac_web_v3.14") -> OrganicViscosityResult:
    """AIOMFAC-VISC/"GC-UNIMOD" predicted dynamic viscosity of an **ion-free** aqueous organic mixture (Gervasi,
    Topping and Zuend, 2020, ACP 20, 2987-3008, Eq. 1-9), given an :class:`ActivityModel` built from water plus one
    or more organic components (no electrolytes), the neutral-component mole fractions ``x_neutral`` (need not
    match ``model``'s own last-evaluated composition -- the mixing functions below call this at a re-normalized
    sub-composition), and ``eta0_pas``: pure-component viscosities (Pa s) at ``T_K``, keyed by each organic
    component's ``Component.number`` (from a measurement, or ``pure_organic_viscosity_vtf``). Water's own
    pure-component viscosity is ``water_viscosity_pas(T_K)`` unless ``eta0_pas`` gives an explicit override for
    it too (the mixing functions use this to substitute an "aware" pseudo-pure-water value).

    Eq. 1-9 need only quantities this package's existing short-range/UNIFAC machinery (``sr.py``) already
    computes from the thermodynamic subgroup tables (R, Q, the interaction parameters a_{m,k}, and the
    combinatorial activity coefficients) -- no additional fitted viscosity-specific parameters are needed for
    this part of the model (unlike the per-ion/per-pair coefficients ``electrolyte_viscosity`` uses).

    Validation. G2020 Supplement Table S5 lists, per binary aqueous data set, the mean absolute and mean bias
    error (log10 units, with each point weighted by its stated measurement error, their Eq. S6) of the model run
    with fixed pure-component viscosities. Rerun on the G2020 Supplement data with the same eta0, this function
    reproduces every Song et al. (2016) row whose error weighting can be reconstructed from the supplement files
    (the mole-fraction mixing-rule column of the same row then also matches) to within 0.004: e.g.
    1,2,4-butanetriol 0.0152/0.0052 (G2020: 0.0152/0.0052), erythritol 0.2921/-0.2915 (0.2921/-0.2915),
    sucrose 1.3780/-0.1886 (1.3781/-0.1887), maleic acid 0.0000/0.0000 (0.0000/0.0000), citric acid
    0.4132/0.2791 (0.4144/0.2829) (``tests/test_viscosity.py::TestOrganicMixtureViscosityGervasi2020``). Against
    the Fortran AIOMFAC-VISC organic code of AIOMFAC-web v3.14 (``SRgres``/``SRgcomb``/``SRcalcvisc`` in
    ``ModSRunifac.f90``, the G2020 value ``sum(lneta_cpn)`` printed before ``SRcalcvisc`` overwrites it, see
    below) for water + glycerol, citric acid, sucrose, diethylene glycol and 1,2-dimethoxyethane (x_org =
    0.01-0.9, 293.15 K, with the Fortran's own pure-component viscosities), the difference is < 1e-14 log10
    units. Up to aiomfac_py 1.3.0, Eq. 5 was coded with Q_k multiplying only its first term; that gave
    differences of 0.01-0.09 log10 units for the same systems.

    Model limitation: PEG oligomers (subgroup 154, CH2OCH2[PEG]). AIOMFAC refits R = 1.381, Q = 3.0 for this
    subgroup (for activities), so for a PEG chain q_i - r_i is large and positive (PEG-400: q = 27.5, r = 14.4),
    and the residual term (Eq. 3-5, which scales with q_i - r_i) grows with chain length: G2020 Eq. 1-9 then give
    unphysical mixture viscosities far above both pure components (water + PEG-400 at 290 K with eta0 = 0.12 Pa s:
    up to ~1e38 Pa s near x_PEG = 0.1; water + triethylene glycol at 293.15 K: up to 1.6 log10 units above
    measured values, and above pure triethylene glycol). This is what the published equations give -- the old
    Fortran code shows the same divergence -- not a porting error; G2020 did not fit or test PEG. Ordinary ether
    groups are not affected (water + diethylene glycol, subgroups 25/150/153: mean absolute error 0.24 log10
    against Hoga et al., 2018, J. Chem. Thermodyn. 122, 38-64, at 293.15 K).

    ``peg_treatment`` (keyword). The default, ``"aiomfac_web_v3.14"``, applies the workaround of the AIOMFAC-web
    v3.14 Fortran (``SRgres``/``SRgcomb`` in ``ModSRunifac.f90``) to every component with more than one
    subgroup 154: its residual contribution is set to zero (xi_R,i = 0, mixture and reference alike) and in
    Eq. 2 gamma_i^C x_i is capped at 1. All other components are computed as published. With it, this function
    reproduces the v3.14 Fortran's internal G2020 value for water + triethylene glycol and + PEG-400 (x_org =
    0.01-0.9) to < 1e-14 log10 units, and water + triethylene glycol at 293.15 K is within 0.11 log10 units (mean
    absolute) of Hoga et al. (2018) instead of 1.0. The workaround is not published (the Fortran comment calls it
    "not an ideal approach but a feasible one for now"); it changed between versions (v3.10-v3.13 set xi_R = 0
    for *all* components of a PEG-containing mixture and capped gamma_i^C at 100), it has been checked against one
    measured data set only, and for PEG-400 it still gives a weak maximum slightly above the pure-PEG viscosity
    (water + PEG-400, 290.15 K, eta0 = 0.12 Pa s: log10 eta = -0.86 at x_PEG = 0.6 vs. -0.92 for pure PEG).
    A ``UserWarning`` names the components it was applied to. ``peg_treatment=None`` gives the published
    equations unchanged (and warns that they are not valid for these components).

    Also not ported: AIOMFAC-web v3.14 reports as its mixture viscosity not Eq. 1 but the mole-fraction mixing
    rule ``ln eta = sum_i x_i ln eta0_i`` (with electrolyte-aware water in the aquelec case); it still computes
    Eq. 1 internally and overwrites it in ``SRcalcvisc``. AIOMFAC-web v3.13 and earlier report Eq. 1.
    """
    if peg_treatment not in PEG_TREATMENTS:
        raise ValueError(f"peg_treatment must be one of {PEG_TREATMENTS}, got {peg_treatment!r}")
    sr = model.mixture.sr
    if sr.n_species != sr.n_neutral:
        raise ValueError("organic_mixture_viscosity requires an ion-free mixture (water + organics only); for "
                          "mixtures that also contain electrolytes, use aquelec_viscosity or aquorg_viscosity")
    nn = sr.n_neutral
    x = np.asarray(x_neutral, dtype=float)
    if x.shape != (nn,):
        raise ValueError(f"x_neutral must have {nn} entries, got {x.shape}")
    components = model.mixture.components[:nn]
    eta0_pas = {} if eta0_pas is None else eta0_pas

    eta0 = np.empty(nn)
    for i, comp in enumerate(components):
        if comp.number in eta0_pas:
            eta0[i] = eta0_pas[comp.number]
        elif _is_pure_water(comp):
            eta0[i] = water_viscosity_pas(T_K)
        else:
            raise ValueError(f"no pure-component viscosity given for organic component {comp.number} "
                              f"({comp.name!r}) -- pass it in eta0_pas (measured, or via "
                              f"pure_organic_viscosity_vtf with a known/estimated Tg)")
    peg = _peg_components(components)
    if peg and peg_treatment is None:
        warnings.warn(f"organic_mixture_viscosity: {[components[i].name for i in peg]} contain more than one "
                      f"CH2OCH2[PEG] subgroup (154); Gervasi et al. (2020) Eq. 1-9 are not valid for PEG oligomers "
                      f"and can give mixture viscosities orders of magnitude above both pure components (see the "
                      f"docstring; the default peg_treatment avoids this)", UserWarning, stacklevel=2)
    elif peg:
        warnings.warn(f"organic_mixture_viscosity: applied the unpublished AIOMFAC-web v3.14 PEG workaround "
                      f"(residual term zero, gamma^C x capped at 1) to {[components[i].name for i in peg]}; "
                      f"peg_treatment=None gives the published Gervasi et al. (2020) equations",
                      UserWarning, stacklevel=2)

    psi = psi_t(sr, T_K)
    L_mix = _group_residual_L(sr, psi, x)
    RS, QS, R, Q, SRNY = sr.RS, sr.QS, sr.R, sr.Q, sr.SRNY
    with np.errstate(divide="ignore", invalid="ignore"):
        Q_over_R = np.where(R > 0.0, Q / np.where(R > 0.0, R, 1.0), 0.0)

    Phi = x * RS / np.dot(RS, x)                        # Eq. 9
    ln_gamma_c = sr_terms(sr, T_K, x).ln_gamma_c         # Eq. 2's gamma_i^C (combinatorial activity coefficient)

    ln_eta = 0.0
    for i in range(nn):
        if peg_treatment is not None and i in peg:                         # AIOMFAC-web v3.14 PEG workaround
            ln_eta += min(math.exp(ln_gamma_c[i]) * x[i], 1.0) * math.log(eta0[i])
            continue
        xi_C = math.exp(ln_gamma_c[i]) * x[i] * math.log(eta0[i])          # Eq. 2

        e_i = np.zeros(nn); e_i[i] = 1.0
        L_ref = _group_residual_L(sr, psi, e_i)                            # pure component i (Eq. 4's "ref")
        N_vis = Q * ((QS[i] - RS[i]) / 2.0 - (1.0 - RS[i]) / Z_COORD)      # Eq. 5, vs. group k
        xi_R = Phi[i] * np.sum(SRNY[i] * Q_over_R * N_vis * (L_mix - L_ref))  # Eq. 3-4

        ln_eta += xi_C + xi_R

    eta = math.exp(ln_eta)
    return OrganicViscosityResult(eta_pas=eta, log10_eta_pas=math.log10(eta), ln_eta=ln_eta)


def _organic_submodel_and_renorm_x(model: ActivityModel, result: ActivityTerms):
    """The water+organics-only (ion-free) sub-mixture of a full organic-inorganic ``model``, and its neutral
    mole fractions re-normalized to exclude ions -- shared by both ``aquelec_viscosity`` (its step 6-7) and
    ``aquorg_viscosity`` (its step 1), which both need exactly this sub-mixture/composition."""
    nn = model.mixture.sr.n_neutral
    organic_model = ActivityModel(list(model.mixture.components[:nn]))
    x_neutral = np.asarray(result.x[:nn], dtype=float)
    return organic_model, x_neutral / x_neutral.sum()


def aquelec_viscosity(model: ActivityModel, result: ActivityTerms, T_K: float,
                       eta0_pas: Mapping[int, float] | None = None, *, cv: float = CV,
                       peg_treatment: str | None = "aiomfac_web_v3.14") -> OrganicViscosityResult:
    """"aquelec" mixing rule (LZ2022 Sect. 3.4.1): treats inorganic ions as dissolving exclusively in water,
    computes that organic-free electrolyte subsystem's viscosity (rescaling ion molalities/mole fractions to
    exclude the organics, Eq. 20-21), then uses it as an "electrolyte-aware" pseudo-pure-water property in a
    run of ``organic_mixture_viscosity`` for the full (renormalized, ion-free) water+organics system.

    ``model``/``result`` are the full organic-inorganic mixture (water first, then organics, then ions);
    ``eta0_pas`` gives pure-component viscosities for the organics exactly as in ``organic_mixture_viscosity``
    (water's own entry, if given, is ignored -- it is always the "electrolyte-aware" value computed here);
    ``peg_treatment`` is passed on to ``organic_mixture_viscosity``.
    """
    sr = model.mixture.sr
    nn = sr.n_neutral
    if nn < 2:
        raise ValueError("aquelec_viscosity requires at least one organic component besides water")

    x = result.x
    wtf = result.wtf
    ww, worg = float(wtf[0]), float(np.sum(wtf[1:nn]))
    if ww + worg <= 0.0:
        raise ValueError("water + organic mass fraction is zero")
    lam = ww / (ww + worg)                                               # Eq. 21

    ions = _collect_ions(sr, result, nn)
    denom = float(x[0]) + sum(x_i for _, x_i, _, _, _, _ in ions)
    ions_aquelec = [(name, x_i / denom, lam * m_i, z_i, lam * a_i_ref, is_cat)
                    for name, x_i, m_i, z_i, a_i_ref, is_cat in ions]     # Eq. 20, step 2-3
    eta1 = _electrolyte_core(float(x[0]) / denom, ions_aquelec, water_viscosity_pas(T_K), cv).eta_pas

    organic_model, x_renorm = _organic_submodel_and_renorm_x(model, result)   # step 6
    eta0_full = dict(eta0_pas) if eta0_pas else {}
    eta0_full[model.mixture.components[0].number] = eta1                 # step 5: electrolyte-aware water
    return organic_mixture_viscosity(organic_model, x_renorm, T_K, eta0_full,  # step 7
                                     peg_treatment=peg_treatment)


def aquorg_viscosity(model: ActivityModel, result: ActivityTerms, T_K: float,
                      eta0_pas: Mapping[int, float] | None = None, *, cv: float = CV,
                      peg_treatment: str | None = "aiomfac_web_v3.14") -> ViscosityResult:
    """"aquorg" mixing rule (LZ2022 Sect. 3.4.2): the mirror image of ``aquelec_viscosity`` -- first computes
    the ion-free aqueous organic subsystem's viscosity (ordinary Dehaoui-model water), uses it as an
    "organics-aware" pseudo-pure-water property, folds water and organics into one combined mole-fraction
    pool, and runs the electrolyte model on that combined pool plus the (unmodified) ions.

    Arguments as in ``aquelec_viscosity``. Returns a :class:`ViscosityResult` (like ``electrolyte_viscosity``),
    since the final step here is the electrolyte model rather than the organic one.
    """
    organic_model, x_renorm = _organic_submodel_and_renorm_x(model, result)   # step 1
    eta2 = organic_mixture_viscosity(organic_model, x_renorm, T_K, eta0_pas, peg_treatment=peg_treatment).eta_pas

    sr = model.mixture.sr
    nn = sr.n_neutral
    x_w_combined = float(np.sum(result.x[:nn]))                          # step 3: organics-aware water
    ions = _collect_ions(sr, result, nn)                                 # unmodified (step: "not necessary to
    return _electrolyte_core(x_w_combined, ions, eta2, cv)               # modify the ion molalities")

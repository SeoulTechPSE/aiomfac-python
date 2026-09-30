"""mr.py's two numerical-safety guard branches (README.md: "No very high ionic strength: the
omega*sqrt(I) > 300 and sqrt(I) > 250 guard branches in mr.py are untested"):

    oexp1 = st.omega * SI2                          (SI2 = sqrt(ionic strength))
    Bca = np.where(oexp1 > 300.0, st.BAC, ...)       -- avoids exp() overflow for very large arguments
    ...
    if SI2 > 250.0: Bkc, Bka = st.bnc.copy(), st.bna.copy()

Unlike every other case in this test suite, these branches cannot be validated against Fortran: they guard
against sqrt(ionic strength) > 250-375 mol/kg (depending on omega, which is ~0.6-1.2 for the tabulated
electrolytes), i.e. ionic strength upward of tens of thousands of mol/kg. No real electrolyte solution -- not
even a hypothetical one -- reaches that; saturated CaCl2/LiCl (the most concentrated cases in test_cases.py)
sit around sqrt(I) ~ 4-6. Constructing an AIOMFAC-web input file that reaches sqrt(I) > 250 is not possible
through the normal input pathway (composition fractions cannot represent it), so there is no way to ask the
reference Fortran binary for a comparison point either -- this is unreachable dead code from a chemistry
standpoint, kept only so `mr_terms` doesn't raise `OverflowError`/emit `inf` from `exp()` if it is ever called
with pathological inputs (e.g. directly, bypassing the normal composition pipeline, as a defensive numerics
test does).

What *is* tested here, directly against `mr_terms` rather than through an input file: with a realistic mixture
(dilute aqueous NaCl) but an ``si`` argument deliberately set far above both thresholds, the guarded code paths
execute without raising and return finite output -- i.e. the branches are at least exercised and don't crash,
even though "matches Fortran" isn't a meaningful claim to make about them.
"""
import numpy as np

from aiomfac_py import ActivityModel, Component
from aiomfac_py.mr import mr_terms


def _real_mixture_state():
    water = Component(1, "Water", ((16, 1),))
    nacl = Component(2, "NaCl", ((202, 1), (242, 1)))
    model = ActivityModel([water, nacl])
    r = model.evaluate([0.9, 0.1], 298.15, "mass")
    m = model.mixture
    sr = m.sr
    nc, na = sr.n_cation, sr.n_anion
    mean_mw = float(np.sum(m.mmass[: m.n_neutral] * r.xn))
    sum_ion_m = float(np.sum(r.sma[:na]) + np.sum(r.smc[:nc]))
    return model, r, mean_mw, sum_ion_m


def test_omega_sqrtI_over_300_guard_is_finite():
    """oexp1 = omega*sqrt(I) > 300 branch (both the cation<->anion Bca/Cnca and, via SI2>250, the
    main-group<->ion Bkc/Bka fallback -- omega is ~0.6-1.2 for NaCl, so sqrt(I)=400 clears omega*sqrt(I)=300 too)."""
    model, r, mean_mw, sum_ion_m = _real_mixture_state()
    si_huge = 400.0 ** 2                                        # sqrt(si) = 400 > 250, and > 300/omega
    terms = mr_terms(model.mixture, model._mr, model.cation_z, model.anion_z, r.smc, r.sma, r.xn,
                      si_huge, sum_ion_m, mean_mw)
    assert np.all(np.isfinite(terms.ln_gamma_neutral))
    assert np.all(np.isfinite(terms.ln_gamma_cation))
    assert np.all(np.isfinite(terms.ln_gamma_anion))


def test_sqrtI_over_250_but_under_omega_sqrtI_300_guard_is_finite():
    """the SI2>250 branch alone (Bkc/Bka fall back to bnc/bna), at an ionic strength where omega*sqrt(I) is
    still under 300 for NaCl's omega (~0.8), so only the second guard -- not the first -- is exercised."""
    model, r, mean_mw, sum_ion_m = _real_mixture_state()
    si_mid = 260.0 ** 2                                          # sqrt(si) = 260: > 250, omega*260 ~ 208 < 300
    terms = mr_terms(model.mixture, model._mr, model.cation_z, model.anion_z, r.smc, r.sma, r.xn,
                      si_mid, sum_ion_m, mean_mw)
    assert np.all(np.isfinite(terms.ln_gamma_neutral))
    assert np.all(np.isfinite(terms.ln_gamma_cation))
    assert np.all(np.isfinite(terms.ln_gamma_anion))

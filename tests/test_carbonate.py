"""Bicarbonate-only dissociation equilibrium (carbonate.py), validated against the instrumented Fortran model.

The reference dumps come from the unpatched Fortran code, whose Gammas() does not refresh the sum of ion molalities
after the carbonate speciation (it does so only for bisulfate systems; see README, "Bicarbonate-only systems").
The resulting small, per-species differences (~1e-6 for HCO3-, ~3e-4 for the trace ion H+) set the tolerances
of the first test. ``test_matches_fortran_with_ion_sum_refresh`` checks against the Fortran code with that one
line corrected, where the agreement is at the printed precision.
"""
import numpy as np
import pytest

from aiomfac_py import ActivityModel, read_input_file
from reference_utils import REF, parse_dump

D = REF / "carbonate"
CASES = {"c002": "NaHCO3 (2 points)", "c004": "KHCO3 (2 points)"}


@pytest.mark.parametrize("n,name", CASES.items())
def test_bicarbonate_dissociation_approximately_matches_fortran(n, name):
    case = read_input_file(D / "inputs" / f"input_{n}.txt")
    pts = parse_dump(D / "dumps" / f"debug_terms_{n}.txt.gz")
    model = ActivityModel(case.components)
    assert model._carbonate is not None, name
    for k, p in enumerate(pts):
        r = model.evaluate(case.fractions[k], case.T_K[k], case.basis)
        nn, nc, na = p["dims"]
        assert (model.mixture.n_neutral, nc, na) == p["dims"]
        # dominant species (Na+/K+, HCO3-) are held to a tight tolerance; the trace ion H+ gets a looser one
        # for the documented conditioning reason -- this asymmetry is itself a regression check: if the tight
        # tolerance below ever needs loosening, that is a signal something else has changed.
        idx_h = model._carbonate[0]
        major_c = [j for j in range(nc) if j != idx_h]
        np.testing.assert_allclose(r.smc[major_c], p["SMC"][major_c], rtol=1e-4, err_msg=f"{name} pt{k + 1}: SMC major")
        np.testing.assert_allclose(r.smc[idx_h], p["SMC"][idx_h], rtol=2e-3, err_msg=f"{name} pt{k + 1}: SMC(H+)")
        np.testing.assert_allclose(r.sma[:na], p["SMA"], rtol=1e-3, err_msg=f"{name} point {k + 1}: SMA")
        lnG = np.concatenate([p["lnactcoeff_n"], p["lnactcoeff_c"], p["lnactcoeff_a"]])
        np.testing.assert_allclose(r.ln_gamma, lnG, atol=1e-3, err_msg=f"{name} point {k + 1}: ln gamma")


def test_pure_bicarbonate_dims_match_fortran():
    """cross-check against the independent structural test in test_completion.py (same NaHCO3 case, DIMS only)."""
    case = read_input_file(D / "inputs" / "input_c002.txt")
    pts = parse_dump(D / "dumps" / "debug_terms_c002.txt.gz")
    model = ActivityModel(case.components)
    assert (model.mixture.n_neutral, model.mixture.sr.n_cation, model.mixture.sr.n_anion) == pts[0]["dims"]


# Fortran with `if (bisulfsyst .or. bicarbsyst)` in Gammas(): (a_w, m_H+, m_HCO3-, m_CO3--) for NaHCO3, input_c002
FIXED_FORTRAN_C002 = [(9.9210745e-01, 1.4545730e-08, 2.3601423e-01, 3.4348369e-03),
                      (9.8027960e-01, 2.2445202e-08, 6.0671835e-01, 9.7124472e-03)]


def test_matches_fortran_with_ion_sum_refresh():
    case = read_input_file(D / "inputs" / "input_c002.txt")
    model = ActivityModel(case.components)
    idx_h, idx_oh, idx_hco3, idx_carb, _ = model._carbonate
    for k, (aw, mh, mhco3, mco3) in enumerate(FIXED_FORTRAN_C002):
        r = model.evaluate(case.fractions[k], case.T_K[k], case.basis)
        np.testing.assert_allclose([r.activity[0], r.smc[idx_h], r.sma[idx_hco3], r.sma[idx_carb]],
                                   [aw, mh, mhco3, mco3], rtol=2e-6)

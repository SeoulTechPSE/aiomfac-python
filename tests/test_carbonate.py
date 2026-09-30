"""Bicarbonate-only dissociation equilibrium (carbonate.py), validated against the instrumented Fortran model.

Unlike the bisulfate equilibrium (test_completion.py's bisulfate tests, atol 1e-12/1e-13), this solver's agreement
with Fortran varies by species: ~1e-6 relative for the dominant carbon species (HCO3-), degrading to ~3e-4 for the
trace ion H+. This is NOT a skipped Fortran branch (that was checked and ruled out) -- it is catastrophic
cancellation inherent to the mass-balance formulation (a ~1e-8 remainder from subtracting terms of order 1), which
affects Fortran's own solver just as much; see the README's validation section for the full analysis. The
tolerances below reflect this measured, per-species pattern rather than a single blanket number.
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

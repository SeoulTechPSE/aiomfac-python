"""Sanity checks of the Fortran reference data itself + the (currently failing-by-design) port check."""
from pathlib import Path

import numpy as np
import pytest

from reference_utils import REF, parse_dump, parse_output_table

CASES = ["0001", "0003"]


@pytest.mark.parametrize("n", CASES)
def test_dump_identities_and_match_with_printed_output(n):
    pts = parse_dump(REF / "debug_terms" / f"debug_terms_{n}.txt")
    out = parse_output_table(REF / "outputs" / f"AIOMFAC_output_{n}.txt")
    nn, nc, na = pts[0]["dims"]
    for p in pts:
        # ln(gamma) of neutrals is the sum of the three range contributions
        np.testing.assert_allclose(p["gnlrln"] + p["gnmrln"] + p["gnsrln"], p["lnactcoeff_n"], rtol=0, atol=1e-14)
        # SR = combinatorial + residual - reference (ions: additionally - Cinf); CO2(aq) is overwritten by GammaCO2
        sr = p["SR_lnGaC"] + p["SR_lnGaR"] - p["SR_lnGaRref"]
        if len(p["SR_lnGaCinf"]) == len(sr):
            sr = sr - p["SR_lnGaCinf"]
        # bicarbsyst: CO2(aq) is overwritten by GammaCO2() after the SR call; in the reference case it is component 2,
        # so only the other neutrals (here: water) are checked for such systems
        chk = slice(0, 1) if p["flags"][1] else slice(0, nn)
        np.testing.assert_allclose(sr[chk], p["gnsrln"][chk], rtol=0, atol=1e-14)
        np.testing.assert_allclose(sr[nn:nn + nc], p["gcsrln"], rtol=0, atol=1e-14)
        np.testing.assert_allclose(sr[nn + nc:nn + nc + na], p["gasrln"], rtol=0, atol=1e-14)
    # printed a_coeff_x has 6 significant digits
    for cp in range(1, nn + 1):
        for k, p in enumerate(pts):
            if p["actcoeff_n"][cp - 1] > 0:
                assert out[cp][k, 5] == pytest.approx(p["actcoeff_n"][cp - 1], rel=5e-6)

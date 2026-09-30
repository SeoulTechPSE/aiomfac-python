"""SR port vs. the instrumented Fortran model (every composition point of the reference cases)."""
import numpy as np
import pytest

from aiomfac_py import Component, build_sr_system, read_input_file, sr_terms
from reference_utils import REF, parse_dump

RTOL, ATOL = 1e-12, 1e-13


def _system(n):
    case = read_input_file(REF / "inputs" / f"input_{n}.txt")
    comps = list(case.components)
    if n == "0003":
        # Fortran SetSystem adds H+ + OH- as an extra electrolyte component for bicarbonate systems (and CO2(aq),
        # which is already an input here); the completion step itself is not ported, so it is spelled out.
        comps.append(Component(6, "H+OH- (added by SetSystem)", ((205, 1), (247, 1))))
        return build_sr_system(comps, assume_complete=True)
    return build_sr_system(comps)


@pytest.mark.parametrize("n", ["0001", "0003"])
def test_sr_terms_match_fortran(n):
    system = _system(n)
    pts = parse_dump(REF / "debug_terms" / f"debug_terms_{n}.txt")
    nn, nc, na = pts[0]["dims"]
    assert (system.n_neutral, system.n_cation, system.n_anion) == (nn, nc, na)
    for k, p in enumerate(pts):
        r = sr_terms(system, p["T_K"][0], p["X"], p["XN"], solvmixrefnd=p["flags"][2])
        msg = f"example {n}, point {k + 1}"
        np.testing.assert_allclose(r.ln_gamma_c, p["SR_lnGaC"], rtol=RTOL, atol=ATOL, err_msg=msg + " lnGaC")
        np.testing.assert_allclose(r.ln_gamma_r, p["SR_lnGaR"], rtol=RTOL, atol=ATOL, err_msg=msg + " lnGaR")
        np.testing.assert_allclose(r.ln_gamma_r_ref, p["SR_lnGaRref"], rtol=RTOL, atol=ATOL, err_msg=msg + " lnGaRref")
        np.testing.assert_allclose(r.ln_gamma_c_inf, p["SR_lnGaCinf"], rtol=RTOL, atol=ATOL, err_msg=msg + " lnGaCinf")
        # totals as stored by the model (CO2(aq) in carbonate systems is overwritten later by GammaCO2)
        chk = slice(0, 1) if p["flags"][1] else slice(0, nn)
        np.testing.assert_allclose(r.ln_gamma_sr[chk], p["gnsrln"][chk], rtol=RTOL, atol=ATOL, err_msg=msg + " gnsrln")
        np.testing.assert_allclose(r.ln_gamma_sr[nn:nn + nc], p["gcsrln"], rtol=RTOL, atol=ATOL, err_msg=msg + " gcsrln")
        np.testing.assert_allclose(r.ln_gamma_sr[nn + nc:], p["gasrln"], rtol=RTOL, atol=ATOL, err_msg=msg + " gasrln")


def test_system_structure_example_0001():
    s = _system("0001")
    assert s.solv_subs == (1, 2, 3, 16, 25, 141, 145, 146, 150, 151, 153)
    assert s.cations == (202,) and s.anions == (242,)
    assert s.n_groups == 13 and s.SRNY.shape == (7, 13)
    assert np.all(s.parA[:, -2:] == 0) and np.all(s.parA[-2:, :] == 0)      # ions do not interact in the SR part


def test_ordering_example_0003_after_completion():
    s = _system("0003")
    assert s.cations == (203, 205) and s.anions == (247, 250, 262)
    assert list(s.cation_z) == [1.0, 1.0] and list(s.anion_z) == [-1.0, -1.0, -2.0]


def test_unsupported_systems_are_rejected_loudly():
    case = read_input_file(REF / "inputs" / "input_0003.txt")
    with pytest.raises(NotImplementedError):
        build_sr_system(case.components)                                    # needs SetSystem completion


def test_peg_system_is_supported_and_flagged():
    """subgroup 154 (CH2OCH2[PEG]) is ported (see test_ext_coverage.py's cc07/cc08 for Fortran validation);
    this just checks the is_peg_system flag and that build succeeds."""
    peg = [Component(1, "PEG", ((16, 1),)), Component(2, "x_PEG", ((154, 4), (1, 2)))]
    s = build_sr_system(peg)
    assert s.is_peg_system is True
    not_peg = [Component(1, "Water", ((16, 1),))]
    assert build_sr_system(not_peg).is_peg_system is False

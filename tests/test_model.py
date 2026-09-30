"""Full activity-coefficient pipeline (mass/mole fractions -> ln gamma) against the instrumented Fortran model."""
import numpy as np
import pytest

from aiomfac_py import ActivityModel, Component, activity_coefficients, read_input_file
from aiomfac_py.composition import mole_frac_to_mass_frac
from aiomfac_py.numerics import safe_exp, soft_bounds_external
from reference_utils import REF, parse_dump, parse_output_table


def _scaled(a, b):
    return np.max(np.abs(np.asarray(a) - np.asarray(b)) / np.maximum(1.0, np.abs(b)))


def test_example_0001_every_stage_matches_fortran():
    case = read_input_file(REF / "inputs" / "input_0001.txt")
    pts = parse_dump(REF / "debug_terms" / "debug_terms_0001.txt")
    model = ActivityModel(case.components)
    assert len(pts) == case.npoints == 12
    for k, p in enumerate(pts):
        r = model.evaluate(case.fractions[k], case.T_K[k], case.basis)
        nn, nc, na = p["dims"]
        stages = {
            "mass fractions": (r.wtf, p["wtf"]), "X": (r.x, p["X"]), "XN": (r.xn, p["XN"][:nn]),
            "cation molalities": (r.smc[:nc], p["SMC"]), "anion molalities": (r.sma[:na], p["SMA"]),
            "LR": (r.ln_gamma_lr, np.concatenate([p["gnlrln"], p["gclrln"], p["galrln"]])),
            "MR": (r.ln_gamma_mr, np.concatenate([p["gnmrln"], p["gcmrln"], p["gamrln"]])),
            "SR": (r.ln_gamma_sr, np.concatenate([p["gnsrln"], p["gcsrln"], p["gasrln"]])),
            "ln gamma neutrals": (r.ln_gamma[:nn], p["lnactcoeff_n"]),
            "ln gamma cations": (r.ln_gamma[nn:nn + nc], p["lnactcoeff_c"]),
            "ln gamma anions": (r.ln_gamma[nn + nc:], p["lnactcoeff_a"]),
            "gamma neutrals": (r.gamma_neutral, p["actcoeff_n"]),
            "activities": (r.activity, p["activity"]),
        }
        for name, (a, b) in stages.items():
            assert _scaled(a, b) < 1e-12, f"point {k + 1}: {name}"


def test_example_0001_matches_printed_fortran_output():
    """independent of the dump: activity coefficients printed by the unmodified Fortran output routine (6 digits)"""
    case = read_input_file(REF / "inputs" / "input_0001.txt")
    out = parse_output_table(REF / "outputs" / "AIOMFAC_output_0001.txt")
    model = ActivityModel(case.components)
    for k in range(case.npoints):
        r = model.evaluate(case.fractions[k], case.T_K[k], case.basis)
        for cp in range(1, 6):
            if r.gamma_neutral[cp - 1] > 0:
                assert out[cp][k, 5] == pytest.approx(r.gamma_neutral[cp - 1], rel=5e-6)


def test_lr_mr_example_0003_with_completed_system():
    """carbonate system: water and all ions must match; CO2(aq) is overwritten by GammaCO2() in the Fortran code."""
    case = read_input_file(REF / "inputs" / "input_0003.txt")
    comps = list(case.components) + [Component(6, "H+OH- (added by SetSystem)", ((205, 1), (247, 1)))]
    model = ActivityModel(comps, assume_complete=True)
    p = parse_dump(REF / "debug_terms" / "debug_terms_0003.txt")[0]
    nn, nc, na = p["dims"]
    ngi = model.mixture.ngi
    smc = np.zeros(ngi); smc[:nc] = p["SMC"]
    sma = np.zeros(ngi); sma[:na] = p["SMA"]
    lr, mr, _ = model.lr_mr_sr(p["T_K"][0], smc, sma, p["XN"][:nn], p["X"])
    for a, b in ((lr.ln_gamma_neutral[0], p["gnlrln"][0]), (mr.ln_gamma_neutral[0], p["gnmrln"][0]),
                 (lr.ln_gamma_cation[:nc], p["gclrln"]), (lr.ln_gamma_anion[:na], p["galrln"]),
                 (mr.ln_gamma_cation[:nc], p["gcmrln"]), (mr.ln_gamma_anion[:na], p["gamrln"])):
        np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-14)
    # Fortran GammaCO2 sets: gnlrln(CO2) = 0 and gnmrln(CO2) = salting-out sum; the plain LR/MR values differ
    assert p["gnlrln"][1] == 0.0 and abs(lr.ln_gamma_neutral[1]) > 1e-3


def test_combined_bisulfate_and_bicarbonate_system_solves():
    """a system with both HSO4-/SO4-- AND HCO3-/CO3-- goes through the joint 6-unknown dissociation equilibrium
    (solve_carb_sulf) rather than being refused; see test_carbonate.py for Fortran-reference validation."""
    comps = [Component(1, "Water", ((16, 1),)), Component(2, "NaHSO4", ((202, 1), (248, 1))),
             Component(3, "NaHCO3", ((202, 1), (250, 1)))]
    r = activity_coefficients(comps, [0.9, 0.05, 0.05], 298.15, basis="mass")
    assert np.all(np.isfinite(r.ln_gamma))
    assert r.ionic_strength > 0.0


def test_pure_water_limit():
    """no solute: SR/MR of water vanish, LR of water is 0 for zero ionic strength -> gamma = 1"""
    case = read_input_file(REF / "inputs" / "input_0001.txt")
    model = ActivityModel(case.components[:1] + case.components[5:])          # water + NaCl
    r = model.evaluate([1.0, 0.0], 298.15, "mass")
    assert r.gamma_neutral[0] == pytest.approx(1.0, abs=1e-15) and r.ionic_strength == 0.0


def test_mole_to_mass_fraction_conversion_is_exactly_normalised():
    w = mole_frac_to_mass_frac([0.9, 0.06, 0.04], [0.018, 0.074, 0.102])
    assert w.sum() == pytest.approx(1.0, abs=1e-16) and np.all(w > 0)
    assert w[0] > 0.9 * 0.018 / (0.9 * 0.018 + 0.06 * 0.074 + 0.04 * 0.102) - 1e-12


def test_safe_exp_is_identity_inside_bounds_and_finite_outside():
    assert safe_exp(3.0, 300.0) == pytest.approx(np.exp(3.0), rel=0, abs=0)
    assert np.isfinite(safe_exp(1.0e6, 300.0)) and safe_exp(-1.0e6, 300.0) > 0.0
    x = np.array([-2.0, 0.5, 1.0e9])
    y = soft_bounds_external(x, -300.0, 300.0)
    assert y[0] == x[0] and y[1] == x[1] and y[2] < 320.0

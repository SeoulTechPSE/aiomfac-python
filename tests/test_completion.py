"""Auto-completion of bisulfate/bicarbonate systems (completion.py), checked structurally against Fortran.

The dissociation equilibrium itself is not ported (see completion.py docstring), so only the resulting species
set / dimensions (nneutral, Ncation, Nanion from the Fortran DIMS line) are compared, not activity coefficients.
"""
from aiomfac_py import Component, read_input_file, build_sr_system
from aiomfac_py.completion import complete_components
from reference_utils import REF, parse_dump

D = REF / "completion"


def test_nahso4_matches_fortran_dims():
    """NaHSO4 alone (HSO4- present, H+/SO4-- absent) -> Fortran adds an H2SO4-like component (2 H+, 1 SO4--)."""
    case = read_input_file(D / "inputs" / "input_c001.txt")
    r = complete_components(list(case.components))
    assert r.triggered and r.bisulfsyst and not r.bicarbsyst
    assert [c.subgroups for c in r.components] == [((16, 1),), ((202, 1), (248, 1)), ((205, 2), (261, 1))]
    dims = parse_dump(D / "dumps" / "debug_terms_c001.txt.gz")[0]["dims"]
    sys_ = build_sr_system(list(r.components), assume_complete=True)
    assert (sys_.n_neutral, sys_.n_cation, sys_.n_anion) == dims == (1, 2, 2)
    assert set(sys_.cations) == {202, 205} and set(sys_.anions) == {248, 261}


def test_nahco3_matches_fortran_dims():
    """NaHCO3 alone -> Fortran adds an H+/CO3-- pair, an OH- pairing, and CO2(aq) as a new neutral component."""
    case = read_input_file(D / "inputs" / "input_c002.txt")
    r = complete_components(list(case.components))
    assert r.triggered and r.bicarbsyst and not r.bisulfsyst
    assert r.n_neutral == 2 and r.id_co2 == 2
    assert r.components[1].subgroups == ((173, 1),)                          # CO2(aq) inserted right after water
    dims = parse_dump(D / "dumps" / "debug_terms_c002.txt.gz")[0]["dims"]
    sys_ = build_sr_system(list(r.components), assume_complete=True)
    assert (sys_.n_neutral, sys_.n_cation, sys_.n_anion) == dims == (2, 2, 3)
    assert set(sys_.cations) == {202, 205} and set(sys_.anions) == {247, 250, 262}


def test_system_needing_no_completion_is_untouched():
    water = Component(1, "Water", ((16, 1),))
    nacl = Component(2, "NaCl", ((202, 1), (242, 1)))
    r = complete_components([water, nacl])
    assert not r.triggered and not r.bisulfsyst and not r.bicarbsyst and r.id_co2 == 0
    assert r.components == (water, nacl)


def test_already_complete_bisulfate_system_is_untouched():
    """H+, HSO4- and SO4-- all already present -> nothing to add."""
    comps = [Component(1, "Water", ((16, 1),)), Component(2, "NaHSO4", ((202, 1), (248, 1))),
             Component(3, "H2SO4", ((205, 2), (261, 1)))]
    r = complete_components(comps)
    assert not r.triggered and r.bisulfsyst
    assert [c.subgroups for c in r.components] == [c.subgroups for c in comps]


# --- bisulfate dissociation equilibrium (HSO4- <-> H+ + SO4--), full activity-coefficient pipeline ------------

import numpy as np
import pytest

from aiomfac_py import ActivityModel


@pytest.mark.parametrize("n,name", [("c001", "NaHSO4"), ("c003", "H2SO4 (given already paired as H+ + SO4--)")])
def test_bisulfate_dissociation_matches_fortran(n, name):
    """ActivityModel completes the system, solves the HSO4- <-> H+ + SO4-- equilibrium, and matches the
    equilibrium ion molalities and ln(gamma) that the instrumented Fortran model reports (full dissociation loop
    included, not just the structural completion checked above)."""
    case = read_input_file(D / "inputs" / f"input_{n}.txt")
    pts = parse_dump(D / "dumps" / f"debug_terms_{n}_full.txt.gz")
    model = ActivityModel(case.components)
    assert model._bisulfate is not None, name
    for k, p in enumerate(pts):
        r = model.evaluate(case.fractions[k], case.T_K[k], case.basis)
        nn, nc, na = p["dims"]
        assert (model.mixture.n_neutral, nc, na) == p["dims"]
        np.testing.assert_allclose(r.smc[:nc], p["SMC"], atol=1e-13, err_msg=f"{name} point {k + 1}: SMC")
        np.testing.assert_allclose(r.sma[:na], p["SMA"], atol=1e-13, err_msg=f"{name} point {k + 1}: SMA")
        lnG = np.concatenate([p["lnactcoeff_n"], p["lnactcoeff_c"], p["lnactcoeff_a"]])
        np.testing.assert_allclose(r.ln_gamma, lnG, atol=1e-12, err_msg=f"{name} point {k + 1}: ln gamma")

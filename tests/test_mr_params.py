import numpy as np
import pytest

from aiomfac_py import AIOMFAC_REFERENCE_COMMIT, load_mr_params, load_subgroup_params
from reference_utils import parse_mr_params_dump

NAMES = ["bTABnc", "cTABnc", "bTABna", "cTABna", "bTABAC", "cTABAC", "Cn1TABAC", "Cn2TABAC", "omega2TAB",
         "omegaTAB", "TABhighestWTF", "TABKsp", "RccTAB", "qcca1TAB"]


@pytest.mark.parametrize("name", NAMES)
def test_mr_table_bit_identical_to_fortran(name):
    """extracted table == array the compiled Fortran model holds after MRdata (incl. single-precision literals)"""
    assert np.array_equal(getattr(load_mr_params(), name), parse_mr_params_dump()[name])


def test_ion_molar_masses_bit_identical_to_fortran():
    d, sg = parse_mr_params_dump(), load_subgroup_params()
    assert np.array_equal(sg.SMWC, d["SMWC"].ravel()) and np.array_equal(sg.SMWA, d["SMWA"].ravel())


def test_provenance_and_spot_values():
    p = load_mr_params()
    assert p.source_commit == AIOMFAC_REFERENCE_COMMIT
    assert p.bTABAC.shape == (40, 40) and p.qcca1TAB.shape == (23, 23, 40) and p.bTABna.shape == (76, 25)
    assert p.bTABAC[1, 1] == pytest.approx(0.05374079546760321)      # Na+ <-> Cl-
    assert p.bTABnc[3, 0] == -7.777778e4                              # undetermined main group <-> cation entry

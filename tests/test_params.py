import numpy as np

from aiomfac_py import AIOMFAC_REFERENCE_COMMIT, load_sr_params, load_subgroup_params
from reference_utils import parse_sr_params_dump


def test_shapes_and_provenance():
    p = load_sr_params()
    assert p.R.shape == p.Q.shape == (265,)
    assert p.ARR.shape == p.BRR.shape == p.CRR.shape == (76, 76)
    assert p.source_commit == AIOMFAC_REFERENCE_COMMIT


def test_bit_identical_to_fortran_arrays():
    """extracted tables == arrays the compiled Fortran model actually uses (dumped after SRdata)"""
    p, d = load_sr_params(), parse_sr_params_dump()
    assert np.array_equal(p.R, d["SR_RR"])
    assert np.array_equal(p.Q, d["SR_QQ"])
    for n in ("ARR", "BRR", "CRR"):
        assert np.array_equal(getattr(p, n), d[n]), n


def test_index_convention():
    p = load_sr_params()
    # standard UNIFAC: a(CH2 -> C=C) = 86.02, a(C=C -> CH2) = -35.36; ARR[i, j] == Fortran ARR(i+1, j+1)
    assert p.ARR[0, 1] == 86.02 and p.ARR[1, 0] == -35.36
    assert np.all(np.diag(p.ARR) == 0.0)


def test_arrays_are_read_only():
    p = load_sr_params()
    assert not p.ARR.flags.writeable


def test_subgroup_tables_bit_identical_to_fortran_arrays():
    p, d = load_subgroup_params(), parse_sr_params_dump()
    assert np.array_equal(p.NKTAB, d["NKTAB"])
    assert np.array_equal(p.GroupMW, d["GroupMW"])
    assert np.array_equal(p.Ioncharge, d["Ioncharge"])
    assert p.source_commit == AIOMFAC_REFERENCE_COMMIT


def test_subgroup_lookups():
    p = load_subgroup_params()
    assert p.main_group(16) == 7 and p.main_group(203) == 51            # water -> main group 7; ions -> 51
    assert p.ion_charge(203) == 1 and p.ion_charge(221) == 2 and p.ion_charge(242) == -1 and p.ion_charge(262) == -2
    assert p.GroupMW[15] == 18.01528

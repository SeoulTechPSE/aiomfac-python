from pathlib import Path

import numpy as np
import pytest

from aiomfac_py import read_input_file

IN = Path(__file__).parent / "reference" / "inputs"


def test_example_0001():
    c = read_input_file(IN / "input_0001.txt")
    assert [x.name for x in c.components][:2] == ["Water", "2-Butanol"]
    assert len(c.components) == 6
    assert c.components[0].subgroups == ((16, 1),)
    assert c.components[5].subgroups == ((202, 1), (242, 1))       # NaCl = Na+ + Cl-
    assert c.basis == "mole" and c.options == {"smiles-based pure-component method": 1}
    assert c.npoints == 12 and c.T_K[0] == 302.0 and c.T_K[-1] == 280.0
    np.testing.assert_allclose(c.fractions.sum(axis=1), 1.0, rtol=0, atol=1e-15)
    # component 1 by difference: 1 - (4*0.11 + 0.01)
    assert c.fractions[0, 0] == pytest.approx(1 - (4 * 0.11 + 0.01))


def test_example_0003_comma_separated_table():
    c = read_input_file(IN / "input_0003.txt")
    assert [x.name for x in c.components] == ["Water", "CO2(aq)", "KOH", "KHCO3", "K2CO3"]
    assert c.components[4].subgroups == ((203, 2), (262, 1))
    assert c.npoints == 1 and c.T_K[0] == 298.0
    assert c.fractions[0, 2] == pytest.approx(0.018)
    assert c.fractions[0, 0] == pytest.approx(1 - 0.018 - 1e-8)


def test_rejects_non_aiomfac_file(tmp_path):
    f = tmp_path / "x.txt"; f.write_text("hello\n")
    with pytest.raises(ValueError):
        read_input_file(f)

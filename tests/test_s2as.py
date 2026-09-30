"""aiomfac_py.s2as (SMILES -> AIOMFAC subgroups), validated against the upstream S2AS tool's own example
input/output file pairs (https://github.com/andizuend/S2AS__SMILES_to_AIOMFAC).

Unlike the rest of this package, S2AS is not a Fortran port -- the upstream tool is already pure Python, using
the `epam.indigo` cheminformatics toolkit for SMARTS matching. The matching algorithm itself is vendored
essentially verbatim (see s2as/__init__.py's module docstring for the small, deliberate integration-level
differences), so this test suite checks bit-for-bit agreement of the subgroup decomposition against upstream's
own generated AIOMFAC input files, across three example cases of increasing size (7, 174, and 2823 molecules).

Skipped automatically if the optional `epam.indigo` dependency is not installed.
"""
import gzip
from pathlib import Path

import pytest

from aiomfac_py import read_input_file

s2as = pytest.importorskip("aiomfac_py.s2as", reason="requires the optional epam.indigo dependency")

D = Path(__file__).parent / "reference" / "s2as"


def _read_smiles(path):
    text = gzip.decompress(Path(path).read_bytes()).decode() if str(path).endswith(".gz") else Path(path).read_text()
    return [line.strip() for line in text.splitlines() if line.strip()]


@pytest.mark.parametrize("smiles_file,input_file,n_molecules", [
    ("smiles_0001.txt", "input_0001.txt", 7),
    ("smiles_1409.txt", "input_1409.txt", 174),
    ("smiles_0160.txt.gz", "input_0160.txt.gz", 2823),
])
def test_subgroup_decomposition_matches_upstream(smiles_file, input_file, n_molecules):
    smiles = _read_smiles(D / smiles_file)
    assert len(smiles) == n_molecules
    result = s2as.smiles_to_components(smiles)
    assert not result.removed

    ref_path = D / input_file
    if str(ref_path).endswith(".gz"):
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".txt", mode="wb", delete=False) as tmp:
            tmp.write(gzip.decompress(ref_path.read_bytes()))
            tmp_path = tmp.name
        ref = read_input_file(tmp_path)
    else:
        ref = read_input_file(ref_path)

    assert len(ref.components) == len(result.components) == n_molecules + 1     # +1 for water
    assert result.components[0].name == "Water" and result.components[0].subgroups == ((16, 1),)
    for a, b in zip(ref.components, result.components):
        assert a.subgroups == b.subgroups, f"component {a.number} ({a.name!r}): {a.subgroups} != {b.subgroups}"


def test_result_feeds_directly_into_activity_model():
    """the whole point of integrating S2AS into aiomfac_py: SMILES -> Component -> ActivityModel, no file I/O."""
    from aiomfac_py import ActivityModel

    result = s2as.smiles_to_components(["CCCCO"])       # 1-butanol
    model = ActivityModel(result.components)
    r = model.evaluate([0.9, 0.1], 298.15, "mass")
    assert r.gamma_neutral[0] > 0.0 and r.gamma_neutral[1] > 0.0
    assert r.ionic_strength == 0.0                       # no electrolytes in this system


def test_invalid_and_empty_smiles_are_recorded_and_dropped_not_raised():
    """upstream blocks on input() for an invalid SMILES and can raise an uncaught IndigoException for SMILES
    Indigo's parser rejects outright; the library API must not do either."""
    result = s2as.smiles_to_components(["CCCCO", "not_a_valid_smiles(((", ""])
    assert [c.name for c in result.components] == ["Water", "CCCCO"]
    assert len(result.removed) == 2
    removed_indices = {ind for ind, _smi, _reason in result.removed}
    assert removed_indices == {1, 2}
    assert all(reason for _ind, _smi, reason in result.removed)     # non-empty explanation for each removal


def test_water_as_component1_is_optional():
    result = s2as.smiles_to_components(["CCCCO"], water_as_component1=False)
    assert len(result.components) == 1
    assert result.components[0].number == 1
    assert result.has_water is False

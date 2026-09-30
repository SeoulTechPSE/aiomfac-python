"""S2AS: SMILES -> AIOMFAC subgroups.

Python port/integration of https://github.com/andizuend/S2AS__SMILES_to_AIOMFAC (Amaladhasan et al., 2026,
https://doi.org/10.5194/gmd-19-4601-2026) into ``aiomfac_py``. Unlike the rest of this package (a Fortran ->
Python rewrite), the original S2AS tool is *already* pure Python -- it uses the `epam.indigo` cheminformatics
toolkit for SMARTS substructure matching, not Fortran. The "port" here is therefore integration, not translation:
the matching algorithm in ``_mapping.py`` (``detAIOMFACsubgs``, from ``S2AS_mapping.py``) and the SMARTS pattern
table in ``_smarts.py`` (from ``SMARTS_query_list.py``) are vendored essentially verbatim (only the internal
import of the SMARTS table was changed to a relative import), so subgroup decomposition results are identical to
the upstream tool for the same input. ``_tools.py`` (from ``ModSmilesTools.py``) keeps the SMILES-validity and
pure-alcohol checks, plus the original file-writing helpers for parity with the upstream CLI tool.

What changed relative to upstream, and why:

* The upstream ``SMILES_to_AIOMFAC_input.py`` script blocks on an interactive ``input()`` prompt when it meets
  an invalid SMILES (so a human at a terminal can inspect and choose to continue), and its validity check itself
  can raise an uncaught ``IndigoException`` for SMILES Indigo's parser rejects outright. Neither is usable as a
  library call, so :func:`smiles_to_components` instead *records* the invalid entry (with Indigo's own
  validity-check error text, or the caught exception's message) in the returned result's ``removed`` list and
  drops it from the component list, mirroring what upstream does structurally (invalid/empty SMILES rows are
  removed before the AIOMFAC input file is written) without the blocking prompt or the crash.
* Debug image rendering (``debugRenderedStructures`` / ``showRenderedStructures`` in upstream, producing PNG/SVG
  files of matched substructures) is not exposed here; it is a debugging aid for the matching algorithm itself,
  orthogonal to using S2AS as a subgroup-decomposition step in a larger pipeline.
* Output is returned as :class:`~aiomfac_py.io.Component` objects (the same type ``read_input_file`` produces),
  not written straight to an AIOMFAC-web input-file on disk, so the result can be fed directly into
  :class:`~aiomfac_py.ActivityModel`. :func:`write_aiomfac_input_file` (a thin wrapper around
  ``_tools.writeAIOMFACfile``) is provided for parity with upstream's file-based workflow when a
  file is wanted (e.g. for AIOMFAC-web's own file-upload interface).

Requires the optional ``epam.indigo`` dependency (``pip install aiomfac_py[smiles]``); importing this subpackage
raises ``ImportError`` with a clear message if it is not installed. It is never imported by ``aiomfac_py``'s own
top-level ``__init__.py``, so the rest of the package works without it.

GPL-3.0-or-later, inherited from both AIOMFAC and S2AS (see ``LICENSE_S2AS`` in this directory).
"""
from __future__ import annotations

from dataclasses import dataclass, field

try:
    from indigo import Indigo
except ModuleNotFoundError as exc:                                        # pragma: no cover - exercised via test
    raise ImportError(
        "aiomfac_py.s2as requires the optional 'epam.indigo' dependency (pip install aiomfac_py[smiles], "
        "or directly: pip install epam.indigo). This is the same cheminformatics toolkit the upstream S2AS "
        "tool (https://github.com/andizuend/S2AS__SMILES_to_AIOMFAC) uses for SMARTS substructure matching."
    ) from exc

from . import _tools as tools
from ._mapping import detAIOMFACsubgs
from ..io import Component

__all__ = ["smiles_to_components", "write_aiomfac_input_file", "S2ASResult", "S2AS_REFERENCE_COMMIT"]

#: commit of https://github.com/andizuend/S2AS__SMILES_to_AIOMFAC this integration is validated against
S2AS_REFERENCE_COMMIT = "88a2bffde1d375f8cb86a6e47ead6c3f0dad1833"

_indigo = Indigo()


def _clean_smiles(smi: str, *, replace_radicals: bool, replace_rdb_oo: bool) -> str:
    """upstream's SMILES string-cleaning step (radical-atom and peroxide-notation normalisation)."""
    st = smi.replace("/", "").replace("\\", "")
    if replace_radicals:
        for pat in ("[O]", "[O.]", "[C]", "[C.]", "[N]", "[N.]", "[S]", "[S.]"):
            st = st.replace(pat, pat[1])
    if replace_rdb_oo:
        st = st.replace("=[O+][O-]", "OO").replace("[O-][O+]=", "OO")
    return st.strip()


@dataclass
class S2ASResult:
    """Result of :func:`smiles_to_components`."""
    components: list[Component]                       # includes the water component iff has_water
    cleaned_smiles: list[str]                          # kept, cleaned SMILES for the ORGANIC components only
    has_water: bool = True                             # whether components[0] is the water component
    removed: list[tuple[int, str, str]] = field(default_factory=list)   # (original index, SMILES, reason)
    n_exceptions: list[int] = field(default_factory=list)               # SMARTS special-case count per component
    n_unmatched_atoms: list[int] = field(default_factory=list)          # non-H atoms left unmatched, per component


def smiles_to_components(smiles: list[str], names: list[str] | None = None, *, water_as_component1: bool = True,
                         replace_radicals: bool = True, replace_rdb_oo: bool = True) -> S2ASResult:
    """Determine the AIOMFAC subgroup decomposition of each SMILES string (port of the main loop of upstream's
    ``SMILES_to_AIOMFAC_input.py``, minus file I/O and the interactive prompt on invalid input -- see the module
    docstring). Water is *not* part of ``smiles`` (as in upstream); pass ``water_as_component1=True`` (the
    default) to have the water component (subgroup 16) prepended to the returned ``components`` list, matching
    the AIOMFAC-web input-file convention used throughout this package (and matching how ``ActivityModel``
    normally expects a mixture's component list to start).

    ``names``, if given, must have the same length as ``smiles`` and supplies each component's display name
    (falls back to the cleaned SMILES string itself, as upstream does).
    """
    if names is not None and len(names) != len(smiles):
        raise ValueError("names must have the same length as smiles")

    result = S2ASResult(components=[], cleaned_smiles=[], has_water=water_as_component1)
    number = 2 if water_as_component1 else 1
    if water_as_component1:
        result.components.append(Component(1, "Water", ((16, 1),)))

    for ind, raw in enumerate(smiles):
        st = _clean_smiles(str(raw), replace_radicals=replace_radicals, replace_rdb_oo=replace_rdb_oo)
        if not st:
            result.removed.append((ind, str(raw), "empty SMILES"))
            continue

        try:
            is_valid, errortext = tools.verifySMILES(st)
        except Exception as exc:
            # upstream's verifySMILES/PureAlcoholCheck call indigo.loadMolecule() unguarded, which raises
            # IndigoException (rather than returning False) for SMILES Indigo's parser rejects outright; treat
            # that the same as a failed validity check instead of propagating the exception to the caller.
            result.removed.append((ind, st, f"invalid SMILES: {exc}"))
            continue
        if not is_valid:
            result.removed.append((ind, st, errortext or "invalid SMILES (failed Indigo valence/H check)"))
            continue

        try:
            is_pure_alc = tools.PureAlcoholCheck(st)
        except Exception as exc:
            result.removed.append((ind, st, f"invalid SMILES: {exc}"))
            continue
        try:
            ret = detAIOMFACsubgs(ind, st, is_pure_alc, False, False, "png", False, [], [], [])
        except Exception as exc:                       # pragma: no cover - defensive, mirrors upstream fragility
            result.removed.append((ind, st, f"SMARTS matching failed: {exc}"))
            continue
        if len(ret) != 3:
            # upstream's own exceptional return path (empty canonical SMILES after passing verifySMILES) -- an
            # edge case in the vendored algorithm itself, not something this integration can fix without
            # changing upstream's matching logic; treat it the same as an invalid SMILES.
            result.removed.append((ind, st, "Indigo produced an empty canonical SMILES for this input"))
            continue
        subgs, n_unmatched, n_except = ret

        name = names[ind] if names is not None else st
        subgroups = tuple((sub, qty) for sub, qty in enumerate(subgs) if sub > 0 and qty > 0)
        result.components.append(Component(number, name, subgroups))
        result.cleaned_smiles.append(st)
        result.n_exceptions.append(n_except)
        result.n_unmatched_atoms.append(n_unmatched)
        number += 1

    return result


def write_aiomfac_input_file(fname: str, outdir: str, result: S2ASResult) -> None:
    """Write an AIOMFAC-web style input file from an :class:`S2ASResult`, matching upstream's
    ``ModSmilesTools.writeAIOMFACfile`` output exactly (thin wrapper kept for parity with the original CLI
    tool's file-based workflow, e.g. for use with AIOMFAC-web's own file-upload interface). ``result`` must have
    been produced with ``water_as_component1=True`` for a directly AIOMFAC-web-compatible file; the organic
    components' cleaned SMILES are used as the row order.
    """
    organics = result.components[1:] if result.has_water else result.components
    n_ngi = 173
    cpsubs = []
    for c in organics:
        row = [0] * n_ngi
        for sub, qty in c.subgroups:
            row[sub] = qty
        cpsubs.append(row)
    tools.writeAIOMFACfile(fname, outdir, result.cleaned_smiles, cpsubs, waterAsComp01=True)

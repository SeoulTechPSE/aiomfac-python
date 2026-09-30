"""Reader for AIOMFAC-web input files (format of ``Inputfiles/input_XXXX.txt``).

Mirrors the reading logic of Mod_InputOutput.f90:
  * the composition table is read list-directed (commas and/or whitespace separate values);
  * columns ``cp02 ... cpNN`` hold components 2..N; component 1 is ``1 - sum(others)``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Component:
    number: int                                   # component number as given in the file (1-based)
    name: str
    subgroups: tuple[tuple[int, int], ...]        # ((subgroup no., quantity), ...)


@dataclass
class InputCase:
    title: str
    components: list[Component]
    basis: str                                    # "mass" or "mole": basis of the composition table
    T_K: np.ndarray                               # (npoints,)
    fractions: np.ndarray                         # (npoints, ncomp); component 1 obtained by difference
    options: dict[str, int] = field(default_factory=dict)

    @property
    def npoints(self) -> int:
        return int(self.T_K.size)


_NUM_SPLIT = re.compile(r"[,\s]+")


def _floats(line: str) -> list[float]:
    return [float(t) for t in _NUM_SPLIT.split(line.strip()) if t]


def read_input_file(path: str | Path) -> InputCase:
    lines = Path(path).read_text(errors="replace").replace("\r\n", "\n").split("\n")
    if not lines or not lines[0].lower().startswith("input file for aiomfac"):
        raise ValueError("not an AIOMFAC input file (first line check failed)")
    title = lines[0].strip()

    components: list[Component] = []
    options: dict[str, int] = {}
    section = None
    cur: dict | None = None
    basis_flags: dict[str, int] = {}
    table_rows: list[list[float]] = []
    ncols_expected = None

    def flush():
        nonlocal cur
        if cur is not None:
            components.append(Component(cur["no"], cur["name"], tuple(cur["sub"])))
            cur = None

    for raw in lines[1:]:
        s = raw.strip()
        low = s.lower()
        if low.startswith("mixture components"):
            section = "components"; continue
        if s.startswith("@@@@"):
            flush(); section = "options_hdr"; continue
        if s.startswith("++++"):
            flush(); section = "comp_hdr"; continue
        if s.startswith("----"):
            flush()
            if section == "options":
                section = None
            continue
        if section == "options_hdr":
            section = "options"; continue          # the "calculation options:" line
        if section == "comp_hdr":
            section = "basis"; continue            # the "mixture composition and temperature:" line
        if section == "components":
            if low.startswith("component no."):
                flush(); cur = {"no": int(s.split(":")[1]), "name": "", "sub": []}
            elif low.startswith("component name"):
                cur["name"] = s.split(":", 1)[1].strip().strip("'")
            elif low.startswith("subgroup no."):
                a, b = _floats(s.split(":", 1)[1])[:2]
                cur["sub"].append((int(a), int(b)))
        elif section == "options":
            if "?" in s:
                key, val = s.split("?", 1)
                options[key.strip()] = int(_floats(val)[0])
        elif section == "basis":
            if "?" in s:
                key, val = s.split("?", 1)
                basis_flags[key.strip().lower()] = int(_floats(val)[0])
            elif low.startswith("point"):
                ncols_expected = len([t for t in _NUM_SPLIT.split(s) if t])   # "point, T_K, cp02, ..."
                section = "table"
        elif section == "table":
            if s.startswith("===="):              # end-of-file marker that closes the composition table
                break
            if s:
                table_rows.append(_floats(s))

    flush()
    if not components:
        raise ValueError("no components found")
    if basis_flags.get("mass fraction", 0) == basis_flags.get("mole fraction", 0):
        raise ValueError("exactly one of 'mass fraction?' / 'mole fraction?' must be 1")
    basis = "mass" if basis_flags.get("mass fraction", 0) == 1 else "mole"

    ncp = len(components)
    if ncols_expected != ncp + 1:
        raise ValueError(f"composition header has {ncols_expected} columns, expected {ncp + 1}")
    tab = np.array(table_rows, dtype=float)
    if tab.ndim != 2 or tab.shape[1] != ncp + 1:
        raise ValueError(f"composition table has shape {tab.shape}, expected (n, {ncp + 1})")
    frac = np.empty((tab.shape[0], ncp))
    frac[:, 1:] = tab[:, 2:]
    frac[:, 0] = 1.0 - frac[:, 1:].sum(axis=1)
    return InputCase(title=title, components=components, basis=basis, T_K=tab[:, 1].copy(),
                     fractions=frac, options=options)

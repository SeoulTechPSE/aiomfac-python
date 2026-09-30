"""Model parameter tables (extracted from the Fortran source by tools/extract_params.py).

Index convention: everything is 0-based, i.e. subgroup number s is row ``s - 1`` of ``R``/``Q`` and
main group g is index ``g - 1`` of ``ARR``/``BRR``/``CRR``. ``ARR[i, j]`` equals Fortran ``ARR(i+1, j+1)``.
BRR/CRR entries of -8.888889e5 are placeholders for interaction parameters that were never fitted.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
import io

import numpy as np

N_MAIN_GROUPS = 76      # ModSystemProp: Nmaingroups
N_SUBGROUPS = 265       # ModSystemProp: topsubno
UNFITTED = -8.888889e5  # placeholder used in BRR / CRR


@dataclass(frozen=True)
class SRParams:
    R: np.ndarray        # (265,) relative van der Waals subgroup volumes
    Q: np.ndarray        # (265,) relative subgroup surface areas
    ARR: np.ndarray      # (76, 76) main-group interaction parameters a_mn
    BRR: np.ndarray      # (76, 76) temperature-dependence parameters b_mn
    CRR: np.ndarray      # (76, 76) temperature-dependence parameters c_mn
    source_commit: str
    source_sha256: str


@lru_cache(maxsize=1)
def load_sr_params() -> SRParams:
    raw = resources.files("aiomfac_py").joinpath("data/sr_params.npz").read_bytes()
    with np.load(io.BytesIO(raw)) as d:
        p = SRParams(R=d["R"], Q=d["Q"], ARR=d["ARR"], BRR=d["BRR"], CRR=d["CRR"],
                     source_commit=str(d["source_commit"]), source_sha256=str(d["source_sha256"]))
    for a in (p.R, p.Q, p.ARR, p.BRR, p.CRR):
        a.setflags(write=False)
    return p


@dataclass(frozen=True)
class SubgroupParams:
    NKTAB: np.ndarray       # (265,) main-group number of each subgroup (51 = ion, 0 = unused)
    GroupMW: np.ndarray     # (200,) molar masses of the neutral subgroups [g/mol]
    Ioncharge: np.ndarray   # (65,) relative charge of ion subgroups 201..265
    SMWC: np.ndarray        # (40,) cation molar masses [g/mol], index = ion ID - 201 (-8.8888e5 = not set)
    SMWA: np.ndarray        # (40,) anion molar masses [g/mol], index = ion ID - 241
    lambdaIN: np.ndarray    # (65,) CO2(aq) salting-out coefficients per ion, index = ion ID - 201 (GammaCO2)
    source_commit: str
    source_sha256: str

    def main_group(self, subgroup: int) -> int:
        """static main-group number of a subgroup (both 1-based, as in the Fortran source)"""
        return int(self.NKTAB[subgroup - 1])

    def ion_charge(self, subgroup: int) -> int:
        if not 201 <= subgroup <= N_SUBGROUPS:
            raise ValueError(f"subgroup {subgroup} is not an ion")
        return int(self.Ioncharge[subgroup - 201])


@lru_cache(maxsize=1)
def load_subgroup_params() -> SubgroupParams:
    raw = resources.files("aiomfac_py").joinpath("data/subgroup_params.npz").read_bytes()
    with np.load(io.BytesIO(raw)) as d:
        p = SubgroupParams(NKTAB=d["NKTAB"], GroupMW=d["GroupMW"], Ioncharge=d["Ioncharge"],
                           SMWC=d["SMWC"], SMWA=d["SMWA"], lambdaIN=d["lambdaIN"],
                           source_commit=str(d["source_commit"]), source_sha256=str(d["source_sha256"]))
    for a in (p.NKTAB, p.GroupMW, p.Ioncharge, p.SMWC, p.SMWA, p.lambdaIN):
        a.setflags(write=False)
    return p


@dataclass(frozen=True)
class MRParams:
    """Middle-range tables (Fortran MRdata). Indices are 0-based: main group g -> g-1, cation ID c -> c-201,
    anion ID a -> a-241. Values of -7.777778e4 in bTABnc/cTABnc/bTABna/cTABna mean "not determined"."""
    bTABnc: np.ndarray      # (76, 23) main group <-> cation
    cTABnc: np.ndarray
    bTABna: np.ndarray      # (76, 25) main group <-> anion
    cTABna: np.ndarray
    bTABAC: np.ndarray      # (40, 40) cation <-> anion
    cTABAC: np.ndarray
    Cn1TABAC: np.ndarray
    Cn2TABAC: np.ndarray
    omega2TAB: np.ndarray
    omegaTAB: np.ndarray
    TABhighestWTF: np.ndarray
    TABKsp: np.ndarray
    RccTAB: np.ndarray      # (23, 23) cation <-> cation
    qcca1TAB: np.ndarray    # (23, 23, 40) three-ion interactions
    source_commit: str
    source_sha256: str


@lru_cache(maxsize=1)
def load_mr_params() -> MRParams:
    raw = resources.files("aiomfac_py").joinpath("data/mr_params.npz").read_bytes()
    with np.load(io.BytesIO(raw)) as d:
        names = [f for f in MRParams.__dataclass_fields__ if f not in ("source_commit", "source_sha256")]
        arrays = {n: d[n] for n in names}
        p = MRParams(**arrays, source_commit=str(d["source_commit"]), source_sha256=str(d["source_sha256"]))
    for n in names:
        getattr(p, n).setflags(write=False)
    return p

"""Mixture/system definition for the short-range (UNIFAC) part.

Port of the parts of SetSystem/definemixtures (SubModDefSystem.f90) and SRsystm (ModSRunifac.f90) that build
the composition-independent SR quantities. Conventions follow the Fortran arrays:

* species order: neutral components (input order), then cations, then anions;
* group order (``AllSubs``): neutral subgroups in ascending subgroup number, then the ion subgroups in
  order of first appearance in the electrolyte components (component order, ascending subgroup number).

Not ported yet (raises ``NotImplementedError``): the automatic completion of H+/HSO4-/SO4--/HCO3-/CO3--/OH-
systems with CO2(aq) (SetSystem step 4) and PEG systems (subgroup 154).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .io import Component
from .params import load_sr_params, load_subgroup_params

_EPS = np.finfo(float).eps
_ARR_UNFITTED = -8.88888e5                     # SRsystm: arr_checkval
_COMPLETION_TRIGGERS = (248, 250, 261, 262)        # HSO4-, HCO3-, SO4--, CO3--
# (205 = H+ is deliberately not in this list: a plain acid such as HCl uses H+ paired with an ordinary
# anion in one component and needs no special handling; only the presence of the ions above signals a
# possible bisulfate/bicarbonate system.)


@dataclass(frozen=True)
class SRSystem:
    n_neutral: int
    n_cation: int
    n_anion: int
    solv_subs: tuple[int, ...]       # neutral subgroups present (ascending)                  (Fortran SolvSubs)
    elect_subs: tuple[int, ...]      # ion subgroups present                                   (ElectSubs)
    cations: tuple[int, ...]         # cation subgroup numbers in species order                (Ication)
    anions: tuple[int, ...]          # anion subgroup numbers in species order                 (Ianion)
    cation_z: np.ndarray
    anion_z: np.ndarray
    SRNY: np.ndarray                 # (NK, NG) number of group J in species I
    R: np.ndarray                    # (NG,)
    Q: np.ndarray                    # (NG,)
    RS: np.ndarray                   # (NK,) species volume parameters
    QS: np.ndarray                   # (NK,) species surface parameters
    XL: np.ndarray                   # (NK,) l_i of UNIFAC
    parA: np.ndarray                 # (NG, NG); ion rows/columns are zero by definition

    @property
    def n_species(self) -> int:
        return self.n_neutral + self.n_cation + self.n_anion

    @property
    def n_groups(self) -> int:
        return len(self.solv_subs) + len(self.elect_subs)

    @property
    def all_subs(self) -> tuple[int, ...]:
        return self.solv_subs + self.elect_subs


def build_sr_system(components: list[Component], *, assume_complete: bool = False) -> SRSystem:
    """Set up the SR system for a list of *completed* input components (neutrals first, then electrolytes).

    ``assume_complete=True`` skips the check that would otherwise reject systems containing H+, HSO4-, HCO3-,
    SO4-- or CO3--, for which the Fortran code silently adds further species (see module docstring); the caller
    must then pass the components exactly as the Fortran code sees them after that completion.
    """
    sp, sg = load_sr_params(), load_subgroup_params()
    used = {s for c in components for s, _ in c.subgroups}
    if 154 in used:
        raise NotImplementedError("PEG systems (subgroup 154) use special R/Q handling that is not ported yet")
    if 240 in used:
        raise ValueError("cation subgroup 240 is handled inconsistently in the Fortran source (ITABsr) - unsupported")
    if not assume_complete and used & set(_COMPLETION_TRIGGERS):
        raise NotImplementedError(
            "systems containing H+, HSO4-, HCO3-, SO4-- or CO3-- are auto-completed by the Fortran SetSystem "
            "(extra species / CO2(aq)); that step is not ported yet (pass assume_complete=True with the completed list)")

    neutrals, electrolytes = [], []
    for c in components:
        subs = [s for s, _ in c.subgroups]
        if all(s <= 200 for s in subs):
            if electrolytes:
                raise ValueError("neutral components must precede electrolyte components")
            neutrals.append(c)
        elif all(s > 200 for s in subs):
            electrolytes.append(c)
        else:
            raise ValueError(f"component {c.number} mixes neutral and ion subgroups")
    nn = len(neutrals)

    solv_subs = tuple(sorted({s for c in neutrals for s, _ in c.subgroups}))
    elect: list[int] = []
    for c in electrolytes:
        for s in sorted(s for s, _ in c.subgroups):
            if s not in elect:
                elect.append(s)
    elect_subs = tuple(elect)
    cations = tuple(s for s in elect_subs if s < 241)
    anions = tuple(s for s in elect_subs if s > 240)
    nk = nn + len(cations) + len(anions)
    all_subs = solv_subs + elect_subs
    ng = len(all_subs)
    col = {s: j for j, s in enumerate(all_subs)}

    SRNY = np.zeros((nk, ng), dtype=int)
    for i, c in enumerate(neutrals):
        for s, q in c.subgroups:
            SRNY[i, col[s]] += q
    for k, s in enumerate(cations):
        SRNY[nn + k, col[s]] = 1
    for k, s in enumerate(anions):
        SRNY[nn + len(cations) + k, col[s]] = 1

    R = np.array([sp.R[s - 1] for s in all_subs])
    Q = np.array([sp.Q[s - 1] for s in all_subs])

    ngn = len(solv_subs)
    parA = np.zeros((ng, ng))
    mg = [sg.main_group(s) for s in solv_subs]
    for j in range(ngn):
        for l in range(ngn):
            a = sp.ARR[mg[l] - 1, mg[j] - 1]
            if a < _ARR_UNFITTED:
                raise ValueError(f"interaction parameter between main groups {mg[l]} and {mg[j]} is not defined")
            if mg[l] == mg[j] and abs(a) > _EPS:
                raise ValueError(f"self-interaction of main group {mg[j]} is not zero")
            parA[l, j] = a

    RS = SRNY @ R
    QS = SRNY @ Q
    XL = 5.0 * (RS - QS) - RS + 1.0
    return SRSystem(
        n_neutral=nn, n_cation=len(cations), n_anion=len(anions),
        solv_subs=solv_subs, elect_subs=elect_subs, cations=cations, anions=anions,
        cation_z=np.array([sg.ion_charge(s) for s in cations], dtype=float),
        anion_z=np.array([sg.ion_charge(s) for s in anions], dtype=float),
        SRNY=SRNY, R=R, Q=Q, RS=RS, QS=QS, XL=XL, parA=parA)


@dataclass(frozen=True)
class Mixture:
    """Composition-independent description of a mixture: everything the SR, MR and LR parts and the composition
    conversions need (Fortran SetSystem/definemixtures/defElectrolytes/SetMolarMass/MRinteractcoeff bookkeeping).

    Independent components are the input neutral components followed by ``n_electrol`` electrolyte components:
    the input electrolytes first, then every further electroneutral cation-anion combination (zero abundance in
    the input). ``n_indcomp = n_neutral + n_electrol`` is the length of the mass-fraction vector.
    """
    sr: SRSystem
    components: tuple                  # completed input components (neutrals first, then electrolytes)
    n_neutral: int
    n_electrol: int
    n_indcomp: int
    ngi: int                           # max(number of cations, number of anions)
    itab: np.ndarray                   # (n_neutral, 200) subgroup counts of the neutral components
    itabmg: np.ndarray                 # (n_neutral, 76) main-group counts of the neutral components
    imaingroup: tuple                  # distinct neutral main groups, in order of first appearance
    maingroup_index: np.ndarray        # (NGN,) index into ``imaingroup`` for each neutral subgroup (sr.solv_subs)
    subgroup_mw: np.ndarray            # (NGN,) neutral subgroup molar masses [kg/mol]
    elect_comps: np.ndarray            # (n_electrol, 2) ion IDs (cation, anion) of each electrolyte component
    elect_nues: np.ndarray             # (n_electrol, 2) stoichiometric numbers of cation, anion
    mmass: np.ndarray                  # (n_indcomp,) molar masses [kg/mol]
    cat_index: dict                    # cation ID -> 0-based position among the cations
    an_index: dict                     # anion ID -> 0-based position among the anions

    @property
    def n_cation(self) -> int:
        return self.sr.n_cation

    @property
    def n_anion(self) -> int:
        return self.sr.n_anion


def build_mixture(components: list[Component], *, assume_complete: bool = False) -> Mixture:
    """Set up the mixture for a list of *completed* input components (see ``build_sr_system``)."""
    sg = load_subgroup_params()
    sr = build_sr_system(components, assume_complete=assume_complete)
    nn = sr.n_neutral
    comps = tuple(components)
    neutrals, electrolytes = comps[:nn], comps[nn:]

    itab = np.zeros((nn, 200), dtype=int)
    itabmg = np.zeros((nn, 76), dtype=int)
    for i, c in enumerate(neutrals):
        for sub, q in c.subgroups:
            itab[i, sub - 1] += q
            itabmg[i, sg.main_group(sub) - 1] += q

    imaingroup: list[int] = []
    for sub in sr.solv_subs:
        g = sg.main_group(sub)
        if g not in imaingroup:
            imaingroup.append(g)
    maingroup_index = np.array([imaingroup.index(sg.main_group(sub)) for sub in sr.solv_subs], dtype=int)
    subgroup_mw = sg.GroupMW[np.array(sr.solv_subs) - 1] * 1.0e-3

    cat_index = {c: k for k, c in enumerate(sr.cations)}
    an_index = {a: k for k, a in enumerate(sr.anions)}

    def nues(kk, jj):
        zc, za = sg.ion_charge(kk), abs(sg.ion_charge(jj))
        if zc == za:
            return 1, 1
        return (2, 1) if zc == 1 else (1, 2)

    # (1) input electrolyte components: first cation and first anion (in ElectSubs order) of each component
    elect: list[tuple[int, int]] = []
    for c in electrolytes:
        subs = {s for s, _ in c.subgroups}
        kk = next((ion for ion in sr.elect_subs if ion < 241 and ion in subs), 0)
        jj = next((ion for ion in sr.elect_subs if ion > 240 and ion in subs), 0)
        if kk == 0 or jj == 0:
            raise ValueError(f"electrolyte component {c.number} ({c.name}) needs one cation and one anion subgroup")
        elect.append((kk, jj))
    # (2) all other electroneutral cation-anion combinations
    for kk in sr.cations:
        for jj in sr.anions:
            if (kk, jj) not in elect:
                elect.append((kk, jj))
    n_el = len(elect)
    elect_comps = np.array(elect, dtype=int).reshape(n_el, 2)
    elect_nues = np.array([nues(kk, jj) for kk, jj in elect], dtype=int).reshape(n_el, 2)

    mmass = np.zeros(nn + n_el)
    for i in range(nn):
        m = 0.0
        for sub in sr.solv_subs:                # same accumulation order as SetMolarMass
            m = m + itab[i, sub - 1] * sg.GroupMW[sub - 1] * 1.0e-3
        mmass[i] = m
    for k, (kk, jj) in enumerate(elect):
        mc, ma = sg.SMWC[kk - 201], sg.SMWA[jj - 241]
        if mc < 0 or ma < 0:
            raise ValueError(f"molar mass of ion {kk if mc < 0 else jj} is not set in the parameter tables")
        mmass[nn + k] = (elect_nues[k, 0] * mc + elect_nues[k, 1] * ma) * 1.0e-3

    return Mixture(sr=sr, components=comps, n_neutral=nn, n_electrol=n_el, n_indcomp=nn + n_el,
                   ngi=max(sr.n_cation, sr.n_anion), itab=itab, itabmg=itabmg, imaingroup=tuple(imaingroup),
                   maingroup_index=maingroup_index, subgroup_mw=subgroup_mw, elect_comps=elect_comps,
                   elect_nues=elect_nues, mmass=mmass, cat_index=cat_index, an_index=an_index)

"""Automatic completion of bisulfate/bicarbonate systems (SetSystem steps 4-6 of SubModDefSystem.f90).

If the input contains HSO4- without both H+ and SO4-- (or vice versa), or HCO3- without both H+ and CO3--, the
Fortran model silently adds the missing species so a dissociation equilibrium can be solved between them; a
bicarbonate system additionally gets an OH- pairing (if missing) and a CO2(aq) neutral component. This module
reproduces that *structural* bookkeeping only: which species end up present, in which components, in which order.

This module handles the *structural* bookkeeping only -- which species end up present, in which components, in
which order. The dissociation equilibrium itself (Fortran HSO4_dissociation / HSO4_and_HCO3_dissociation, an
iterative root-finding problem that redistributes molalities between H+/HSO4-/SO4-- and/or H+/HCO3-/CO3-- at
each composition point) is ported separately: dissociation.py's ``solve_bisulfate`` for the bisulfate-only case,
carbonate.py's ``solve_carbonate``/``solve_carb_sulf`` for the bicarbonate-only and joint cases. ``ActivityModel``
(model.py) wires ``complete_components`` together with whichever of those three applies, so systems that trigger
this completion are evaluated end-to-end, not rejected.
"""
from __future__ import annotations

from dataclasses import dataclass

from .io import Component

_CATIONS = range(201, 241)
_ANIONS = range(241, 266)          # topsubno = 265


def _has(row: dict, subs) -> bool:
    return any(row.get(s, 0) > 0 for s in subs)


@dataclass(frozen=True)
class CompletionResult:
    components: tuple[Component, ...]   # completed components, neutrals first, renumbered 1..N
    n_neutral: int
    bisulfsyst: bool
    bicarbsyst: bool
    id_co2: int                         # 1-based component number of CO2(aq), 0 if not added/present
    triggered: bool                     # True if any species were actually added (== Fortran updbisulf/updbicarb)
    orig_index: tuple[int, ...]         # orig_index[j] = 0-based index in the ORIGINAL input list of completed
                                         # component j, or -1 if component j was newly added (e.g. CO2(aq) is NOT
                                         # -1 when it was already present in the input, only when inserted fresh)


def complete_components(components: list[Component]) -> CompletionResult:
    """Add H+/SO4--/HSO4-/HCO3-/CO3--/OH-/CO2(aq) exactly as the Fortran SetSystem would, if needed.

    ``components`` must have all neutral components (subgroups <= 200) before all electrolyte components
    (subgroups > 200), as required elsewhere in this package (``build_sr_system``).
    """
    subs_of = [dict(c.subgroups) for c in components]
    names = [c.name for c in components]
    orig_idx = list(range(len(components)))
    nn0 = sum(1 for c in components if all(s <= 200 for s, _ in c.subgroups))
    if any(all(s <= 200 for s, _ in c.subgroups) for c in components[nn0:]):
        raise ValueError("neutral components must precede electrolyte components")
    n = len(subs_of)

    def ensure(k):
        while len(subs_of) < k:
            subs_of.append({}); names.append(""); orig_idx.append(-1)

    def cation_free(i):
        return not _has(subs_of[i - 1] if i <= len(subs_of) else {}, _CATIONS)

    def anion_free(i):
        return not _has(subs_of[i - 1] if i <= len(subs_of) else {}, _ANIONS)

    def first_electrolyte():
        for i in range(1, n + 1):
            if _has(subs_of[i - 1], _CATIONS):
                return i
        raise ValueError("bisulfate/bicarbonate completion needs at least one electrolyte component")

    Hexists = any(_has(r, (205,)) for r in subs_of)
    HSO4exists = any(_has(r, (248,)) for r in subs_of)
    SO4exists = any(_has(r, (261,)) for r in subs_of)
    HCO3exists = any(_has(r, (250,)) for r in subs_of)
    CO3exists = any(_has(r, (262,)) for r in subs_of)
    bisulfsyst, bicarbsyst = HSO4exists, HCO3exists
    triggered = False

    updbisulf = (HSO4exists and (not Hexists or not SO4exists)) or \
                (not HSO4exists and ((Hexists and SO4exists) or (HCO3exists and SO4exists)))
    if updbisulf:
        triggered = True; bisulfsyst = True
        cat_free = an_free = 0
        nnp1 = first_electrolyte()
        if not Hexists and cation_free(n + 1):
            cat_free = n + 1; ensure(cat_free); subs_of[cat_free - 1][205] = 1; n += 1
        if not SO4exists:
            for i in range(nnp1, n + 2):
                if anion_free(i):
                    an_free = i; ensure(i); subs_of[i - 1][261] = 1; n = max(n, i); break
        if not HSO4exists:
            for i in range(nnp1, n + 2):
                if anion_free(i):
                    an_free = i; ensure(i); subs_of[i - 1][248] = 1; n = max(n, i); break
        if an_free == cat_free and cat_free:
            if subs_of[cat_free - 1].get(205) == 1 and subs_of[cat_free - 1].get(261, 0) > 0:
                subs_of[cat_free - 1][205] = 2
        elif an_free > cat_free and an_free:
            if subs_of[an_free - 1].get(205, 0) == 0:
                subs_of[an_free - 1][205] = 1 if subs_of[an_free - 1].get(248, 0) > 0 else 2
        elif cat_free > an_free and cat_free:
            subs_of[cat_free - 1][248] = 1

    updbicarb = (HCO3exists and (not Hexists or not CO3exists)) or \
                (not HCO3exists and ((Hexists and CO3exists) or (HSO4exists and CO3exists)))
    nnp1 = first_electrolyte() if (bicarbsyst or updbicarb) else None
    if updbicarb:
        triggered = True; bicarbsyst = True
        cat_free = an_free = 0
        if not Hexists and cation_free(n + 1):
            cat_free = n + 1; ensure(cat_free); subs_of[cat_free - 1][205] = 1; n += 1
        if not CO3exists:
            for i in range(nnp1, n + 2):
                if anion_free(i):
                    an_free = i; ensure(i); subs_of[i - 1][262] = 1; n = max(n, i); break
        if not HCO3exists:
            for i in range(nnp1, n + 2):
                if anion_free(i):
                    an_free = i; ensure(i); subs_of[i - 1][250] = 1; n = max(n, i); break
        if an_free == cat_free and cat_free:
            if subs_of[cat_free - 1].get(205) == 1 and subs_of[cat_free - 1].get(262, 0) > 0:
                subs_of[cat_free - 1][205] = 2
        elif an_free > cat_free and an_free:
            if subs_of[an_free - 1].get(205, 0) == 0:
                subs_of[an_free - 1][205] = 1 if subs_of[an_free - 1].get(250, 0) > 0 else 2
        elif cat_free > an_free and cat_free:
            subs_of[cat_free - 1][250] = 1

    id_co2 = 0
    if bicarbsyst:
        if not any(_has(r, (247,)) for r in subs_of):
            n += 1; ensure(n); subs_of[n - 1][205] = 1; subs_of[n - 1][247] = 1
            names[n - 1] = "OH- pairing (added by SetSystem)"
        if not any(subs_of[i].get(173, 0) > 0 for i in range(0, nnp1)):
            subs_of.insert(nnp1 - 1, {173: 1}); names.insert(nnp1 - 1, "CO2(aq)"); orig_idx.insert(nnp1 - 1, -1)
            n += 1; id_co2 = nnp1
        else:
            id_co2 = 1 + next(i for i in range(0, nnp1) if subs_of[i].get(173, 0) > 0)

    for i in range(len(subs_of)):
        if not names[i]:
            names[i] = "electrolyte (added by SetSystem)" if any(s > 200 for s in subs_of[i]) else "added component"
    n_neutral = sum(1 for r in subs_of if all(s <= 200 for s in r))
    comps = tuple(Component(i + 1, names[i], tuple(sorted(subs_of[i].items()))) for i in range(len(subs_of)))
    return CompletionResult(comps, n_neutral, bisulfsyst, bicarbsyst, id_co2, triggered, tuple(orig_idx))

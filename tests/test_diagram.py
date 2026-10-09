"""Phase-boundary tracing (aiomfac_py.diagram): the deliquescence RH found by bisection on PhaseEquilibrium solves must
equal the single-salt saturation of the SLE layer (sle.binary_saturation), and the bookkeeping (states, intervals,
boundary kinds, phase maps) must be consistent."""
import warnings

import numpy as np
import pytest

pytest.importorskip("scipy")

from aiomfac_py.diagram import Boundary, PhaseState, phase_map, pie_composition, trace  # noqa: E402
from aiomfac_py.phase_equilibrium import PhaseEquilibrium  # noqa: E402
from aiomfac_py.sle import binary_saturation  # noqa: E402

T0 = 298.15


@pytest.mark.parametrize("key,feed", [("ammonium_sulfate", {"NH4+": 2.0, "SO4--": 1.0}),
                                      ("halite", {"Na+": 1.0, "Cl-": 1.0}),
                                      ("ammonium_nitrate", {"NH4+": 1.0, "NO3-": 1.0})])
def test_traced_drh_equals_binary_saturation(key, feed):
    pe = PhaseEquilibrium([], list(feed), T_K=T0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t = trace(pe, feed, n=16, tol=1e-4)
    deliq = [b for b in t.boundaries if b.kind == "deliquescence"]
    assert len(deliq) == 1 and deliq[0].resolved
    b = deliq[0]
    assert b.rh_hi - b.rh_lo <= 1e-4
    assert abs(b.rh - binary_saturation(key, T0)["aw"]) < 2e-4
    assert b.below == PhaseState(0, (key,)) and b.above == PhaseState(1, ())
    # intervals cover the requested range (also when the lowest solve failed) and agree with the boundaries
    iv = t.intervals()
    assert iv[0][0] == pytest.approx(t.rh_range[0]) and iv[-1][1] == pytest.approx(t.rh_range[1])
    assert all(a <= b for a, b, _ in iv)
    assert iv[-1][2] == PhaseState(1, ())


def test_metastable_mode_has_no_solids():
    pe = PhaseEquilibrium([], ["Na+", "Cl-"], T_K=T0)
    t = trace(pe, {"Na+": 1.0, "Cl-": 1.0}, mode="metastable", rh_min=0.5, rh_max=0.95, n=6)
    assert all(s == PhaseState(1, ()) for s in t.states)
    assert t.boundaries == []


def test_boundary_kinds():
    L, AS, AN = PhaseState(1, ()), PhaseState(0, ("ammonium_sulfate",)), PhaseState(1, ("ammonium_nitrate",))
    assert Boundary(0.8, 0.8, 0.8, AS, L).kind == "deliquescence"
    assert Boundary(0.5, 0.5, 0.5, L, AN).kind == "precipitation"
    assert Boundary(0.5, 0.5, 0.5, PhaseState(2, ()), L).kind == "phase separation"
    assert Boundary(0.1, 0.1, 0.1, PhaseState(0, ("MgCl2_4H2O",)), PhaseState(0, ("bischofite",))).kind == \
        "solid transition"
    assert L.label == "L" and PhaseState(2, ("halite",)).label == "2L + halite"


def test_phase_map_mixed_salt_eutonic():
    """NaCl + KCl: the mutual deliquescence RH (both salts dissolve together) is below both single-salt DRH."""
    pe = PhaseEquilibrium([], ["Na+", "K+", "Cl-"], T_K=T0)
    feed = lambda x: {"Na+": x, "K+": 1.0 - x, "Cl-": 1.0}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pm = phase_map(pe, feed, [0.3, 0.6], x_label="x(NaCl)", n=16, rh_min=0.6, rh_max=0.95)
    for t in pm.traces:
        first = t.boundaries[0]
        assert first.below.solids == ("halite", "sylvite") and first.rh < 0.756
    # the mutual DRH does not depend on the mixing ratio
    assert abs(pm.traces[0].boundaries[0].rh - pm.traces[1].boundaries[0].rh) < 2e-4
    recs = pm.to_records()
    assert {r["kind"] for r in recs} <= {"deliquescence", "precipitation", "phase separation", "solid transition",
                                         "other"}
    lines = pm.boundary_lines()
    assert all(len(xs) == len(rh) for xs, rh in lines.values())


def test_pie_composition_groups():
    pe = PhaseEquilibrium([], ["Na+", "Cl-"], T_K=T0)
    res = pe.solve({"Na+": 1.0, "Cl-": 1.0}, 0.9)
    pie = pie_composition(res)
    assert pie[0][0] == "L1" and set(pie[0][1]) == {"water", "ions"}
    assert np.isclose(pie[0][1]["ions"], 2.0)


def test_ammonium_sulfate_nitrate_double_salts():
    """(NH4)2SO4 + NH4NO3: the dry assemblage follows the stoichiometry of the double salts (linear program over
    the candidate solids), and the mutual deliquescence RH of (NH4)2SO4 + (NH4)2SO4.2NH4NO3 is an invariant point
    (independent of the mixing ratio)."""
    pe = PhaseEquilibrium([], ["NH4+", "SO4--", "NO3-"], T_K=T0)
    feed = lambda x: {"NH4+": 2 * x + (1 - x), "SO4--": x, "NO3-": 1 - x}
    expect = {0.1: {"ammonium_nitrate", "AS_3AN"}, 0.3: {"AS_3AN", "AS_2AN"}, 0.6: {"ammonium_sulfate", "AS_2AN"},
              0.97: {"ammonium_sulfate", "AS_2AN"}}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for x, solids in expect.items():
            r = pe.solve(feed(x), 0.4)
            assert r.status == "dry" and set(r.solids) == solids, (x, r.solids)
            assert max(r.si.values()) <= 1e-9
        mdrh = []
        for x in (0.6, 0.97):
            t = trace(pe, feed(x), rh_min=0.55, rh_max=0.66, n=8, tol=1e-4)
            mdrh.append(t.boundaries[0].rh)
            assert t.boundaries[0].below.solids == ("AS_2AN", "ammonium_sulfate")
    assert abs(mdrh[0] - mdrh[1]) < 3e-4

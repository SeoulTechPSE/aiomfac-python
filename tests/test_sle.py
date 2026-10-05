"""Tests of the solid-liquid equilibrium solver (aiomfac_py.sle) and the solid-phase database (aiomfac_py.solids).

Reference values: DRH of the pure salts at 298.15 K from the literature (Tang & Munkelwitz 1993/1994; Wexler &
Seinfeld 1991 type compilations; rounded) and the handbook solubilities in ``tools/solubility_data.py``.  There is no
Fortran reference for the SLE layer; the active-set solver is checked against an independent generic optimizer
(SLSQP) minimizing the same Gibbs function, and the KKT conditions are verified directly.
"""
import math

import numpy as np
import pytest

pytest.importorskip("scipy")
from scipy.optimize import minimize  # noqa: E402

from aiomfac_py.solids import ION_REGISTRY, SOLIDS  # noqa: E402
from aiomfac_py.sle import (AqueousIons, SLESolver, binary_saturation, feed_from_salts,  # noqa: E402
                            implied_ln_s_crit)

T0 = 298.15


# ---------------------------------------------------------------------------------------------------------
# database
# ---------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("key", sorted(SOLIDS))
def test_solid_is_electroneutral_and_k_finite(key):
    s = SOLIDS[key]
    assert s.charge_balance == 0
    for T in (273.15, 298.15, 323.15):
        assert math.isfinite(s.ln_k(T))
        assert math.isfinite(s.ln_k(T, "anchored"))


def test_fitted_matches_anchor_at_298():
    for key, s in SOLIDS.items():
        if s.kfit is not None and s.anchor_m is not None:
            assert abs(s.ln_k(T0, "fitted") - s.ln_k(T0, "anchored")) < 0.03, key


# ---------------------------------------------------------------------------------------------------------
# single-salt saturation: DRH and solubility
# ---------------------------------------------------------------------------------------------------------
DRH_298 = {"halite": 75.3, "sylvite": 84.2, "sal_ammoniac": 77.2, "nitratine": 74.3, "niter": 93.8,
           "ammonium_nitrate": 61.8, "ammonium_sulfate": 79.9, "arcanite": 97.6}


@pytest.mark.parametrize("key,drh", sorted(DRH_298.items()))
def test_binary_drh_298(key, drh):
    r = binary_saturation(key, T0)
    assert abs(100 * r["aw"] - drh) < 1.6, (key, 100 * r["aw"], drh)


@pytest.mark.parametrize("key,T_C,g100", [("halite", 40, 36.42), ("sylvite", 20, 34.24), ("sal_ammoniac", 40, 46.0),
                                          ("ammonium_sulfate", 40, 81.2), ("ammonium_nitrate", 40, 283.0),
                                          ("arcanite", 60, 18.2)])
def test_fitted_solubility_vs_handbook(key, T_C, g100):
    sg = {"halite": 58.443, "sylvite": 74.551, "sal_ammoniac": 53.491, "ammonium_sulfate": 132.14,
          "ammonium_nitrate": 80.043, "arcanite": 174.26}[key]
    r = binary_saturation(key, 273.15 + T_C)
    m_exp = g100 / sg * 10.0
    assert abs(r["molality"] / m_exp - 1.0) < 0.04


def test_na2so4_hydrate_transition():
    below = (binary_saturation("mirabilite", 303.15)["molality"], binary_saturation("thenardite", 303.15)["molality"])
    above = (binary_saturation("mirabilite", 307.15), binary_saturation("thenardite", 307.15)["molality"])
    assert below[0] < below[1]                 # mirabilite less soluble -> stable below 305.5 K
    assert not (above[0]["molality"] < above[1])   # above: mirabilite no longer less soluble (or no saturation at all)


# ---------------------------------------------------------------------------------------------------------
# mixtures
# ---------------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def nacl_kcl():
    return SLESolver(["Na+", "K+", "Cl-"])


def test_nacl_kcl_phase_sequence(nacl_kcl):
    feed = feed_from_salts({"NaCl": 1.0, "KCl": 1.0})
    assert nacl_kcl.solve(feed, T0, 0.82).status == "aqueous"
    r = nacl_kcl.solve(feed, T0, 0.76)
    assert r.status == "solid+aqueous" and set(r.solids) == {"sylvite"}
    assert r.solids["sylvite"] == pytest.approx(0.402, abs=0.01)
    r = nacl_kcl.solve(feed, T0, 0.70)
    assert r.status == "dry" and r.solids["halite"] == pytest.approx(1.0) and r.solids["sylvite"] == pytest.approx(1.0)
    mdrh = nacl_kcl.deliquescence_rh(feed, T0)
    assert 0.715 < mdrh < 0.74


def _kkt_checks(sol, feed, T, rh, res):
    # mass balance
    for ion, n0 in res.feed.items():
        n = res.aq_ions.get(ion, 0.0)
        for k, ns in res.solids.items():
            n += SOLIDS[k].ions.get(ion, 0) * ns
        assert n == pytest.approx(n0, rel=1e-9, abs=1e-12)
    if res.status == "dry":
        assert res.water_kg == 0.0
        return
    # dual feasibility and complementarity
    for k, si in res.si.items():
        assert si <= 1e-6, (k, si)
        if k in res.solids:
            assert abs(si) < 1e-6, (k, si)
    # water activity of the aqueous phase equals RH
    aq = AqueousIons(res.ions)
    m = np.array([res.molality[i] for i in res.ions])
    assert math.exp(aq.ln_gamma_aw(m, T)[1]) == pytest.approx(rh, abs=1e-9)


@pytest.mark.parametrize("rh", [0.97, 0.80, 0.72, 0.66, 0.64, 0.55])
def test_kkt_conditions_nacl_ammonium_sulfate(rh):
    sol = SLESolver(["Na+", "NH4+", "Cl-", "SO4--"])
    feed = feed_from_salts({"NaCl": 1.0, "(NH4)2SO4": 1.0})
    res = sol.solve(feed, T0, rh)
    _kkt_checks(sol, feed, T0, rh, res)


def _brute_gibbs(sol, feed, T, rh):
    ln_rh = math.log(rh)
    bfull = sol._vec(feed)
    present = bfull > 0
    ion_idx = np.flatnonzero(present)
    sol_idx = [j for j in range(len(sol.solids)) if np.all(sol.V[~present, j] == 0)]
    scale = bfull.max()
    b = bfull[ion_idx] / scale
    V = sol.V[np.ix_(ion_idx, sol_idx)]
    c = np.array([sol.solids[j].ln_k_eff(T, ln_rh, sol.mode) for j in sol_idx])
    aq = sol._aq_for(ion_idx)

    def phi(u):
        n = b - V @ u
        if np.any(n <= 1e-12):
            return 1e6
        r = aq.solve_water(n, T, ln_rh)
        return 1e6 if r is None else float(np.sum(n * r[0]) + c @ u)

    best = np.inf
    rng = np.random.default_rng(1)
    for k in range(4):
        u0 = np.zeros(V.shape[1]) if k == 0 else rng.random(V.shape[1]) * 0.1
        r = minimize(phi, u0, method="SLSQP", bounds=[(0, None)] * V.shape[1],
                     constraints=[{"type": "ineq", "fun": lambda u: (b - V @ u) - 1e-9}],
                     options={"ftol": 1e-12, "maxiter": 300})
        best = min(best, r.fun)
    return best * scale


@pytest.mark.parametrize("feed,rh", [({"NaCl": 1.0, "(NH4)2SO4": 1.0}, 0.65), ({"NaCl": 1.0, "KCl": 1.0}, 0.76),
                                     ({"KCl": 1.0, "NH4Cl": 0.5, "NaCl": 0.3}, 0.78)])
def test_active_set_not_worse_than_generic_optimizer(feed, rh):
    ions = ["Na+", "K+", "NH4+", "Cl-", "SO4--"]
    sol = SLESolver(ions)
    f = feed_from_salts(feed)
    r = sol.solve(f, T0, rh)
    assert r.gibbs <= _brute_gibbs(sol, f, T0, rh) + 1e-6


def test_rh_scan_monotone_water_and_solids():
    sol = SLESolver(["Na+", "NH4+", "Cl-", "SO4--"])
    feed = feed_from_salts({"NaCl": 1.0, "(NH4)2SO4": 1.0})
    scan = sol.scan_rh(feed, T0, [0.55, 0.62, 0.65, 0.68, 0.72, 0.80, 0.9])
    w = [r.water_kg for r in scan]
    assert all(w[i] <= w[i + 1] + 1e-12 for i in range(len(w) - 1))          # water grows with RH
    tot = [sum(r.solids.values()) for r in scan]
    assert tot[0] > 0 and tot[-1] == 0.0


def test_efflorescence_threshold_consistency():
    sol = SLESolver(["Na+", "Cl-"])
    s_crit = implied_ln_s_crit("halite", T0, 0.45)
    out = sol.efflorescence_rh(feed_from_salts({"NaCl": 1.0}), T0, ln_s_crit=s_crit)
    assert out["salt"] == "halite" and out["rh"] == pytest.approx(0.45, abs=2e-3)
    out0 = sol.efflorescence_rh(feed_from_salts({"NaCl": 1.0}), T0, ln_s_crit=0.0)
    assert out0["rh"] == pytest.approx(binary_saturation("halite", T0)["aw"], abs=2e-3)


def test_feed_must_be_electroneutral():
    sol = SLESolver(["Na+", "Cl-"])
    with pytest.raises(ValueError):
        sol.solve({"Na+": 1.0, "Cl-": 0.5}, T0, 0.8)

"""Volatile gases (NH3, HNO3, HCl) in the SLE solver: constants, open (fixed partial pressure) and closed systems.

Reference values: the gas constants are those of Clegg, Brimblecombe & Wexler (1998, J. Phys. Chem. A 102:2137 and 2155);
Kp(NH4NO3) = 4.36e-17 atm^2 at 298.15 K is the value they derive from the same constants (our solid K is the
SOLUBILITY-fitted one, so a ~5 % difference is expected)."""
import math

import numpy as np
import pytest

pytest.importorskip("scipy")

from aiomfac_py.gases import GASES  # noqa: E402
from aiomfac_py.sle import SLESolver  # noqa: E402
from aiomfac_py.solids import SOLIDS  # noqa: E402

T0 = 298.15
IONS = ["Na+", "NH4+", "H+", "NO3-", "Cl-", "SO4--"]


@pytest.fixture(scope="module")
def sol():
    return SLESolver(IONS)


def _totals(res):
    tot = dict(res.aq_ions)
    for k, n in res.solids.items():
        for ion, nu in SOLIDS[k].ions.items():
            tot[ion] = tot.get(ion, 0.0) + nu * n
    for g, n in res.gas.items():
        for ion, nu in GASES[g].ions.items():
            tot[ion] = tot.get(ion, 0.0) + nu * n
    return tot


def test_gas_constants_at_298():
    assert math.exp(GASES["NH3"].ln_k(T0)) == pytest.approx(1.066e11, rel=1e-3)
    assert math.exp(GASES["HNO3"].ln_k(T0)) == pytest.approx(853.1 * (1000 / 18.01528) ** 2, rel=2e-4)
    assert math.exp(GASES["HCl"].ln_k(T0)) == pytest.approx(662.1 * (1000 / 18.01528) ** 2, rel=1e-9)


def test_kp_ammonium_nitrate_consistent_with_clegg():
    kp = math.exp(SOLIDS["ammonium_nitrate"].ln_k(T0) - GASES["NH3"].ln_k(T0) - GASES["HNO3"].ln_k(T0))
    assert kp == pytest.approx(4.36e-17, rel=0.10)
    # temperature dependence: Clegg et al. give dH(Kp) = 184.2 kJ/mol -> ln Kp slope between 298 and 308 K
    kp35 = math.exp(SOLIDS["ammonium_nitrate"].ln_k(308.15) - GASES["NH3"].ln_k(308.15) - GASES["HNO3"].ln_k(308.15))
    dh = 8.314462618 * math.log(kp35 / kp) / (1.0 / 298.15 - 1.0 / 308.15)
    assert dh == pytest.approx(184.2e3, rel=0.06)


def test_open_equilibrium_pressures_recovered_from_activities(sol):
    p = {"NH3": 1.0e-9, "HNO3": 1.0e-9}
    r = sol.solve({"NH4+": 2.0, "SO4--": 1.0}, T0, 0.8, p_gas=p)
    assert r.status == "aqueous"
    la = r.ln_a
    assert math.exp(la["NH4+"] - la["H+"] - GASES["NH3"].ln_k(T0)) == pytest.approx(p["NH3"], rel=1e-7)
    assert math.exp(la["H+"] + la["NO3-"] - GASES["HNO3"].ln_k(T0)) == pytest.approx(p["HNO3"], rel=1e-7)
    aq = sol._aq_for([sol.ions.index(i) for i in r.ions])
    m = np.array([r.molality[i] for i in aq.ions])
    assert aq.ln_gamma_aw(m, T0)[1] == pytest.approx(math.log(0.8), abs=1e-7)
    # mass balance with the reservoir exchange (gas amounts are net transfers)
    tot = _totals(r)
    for ion, b in {"NH4+": 2.0, "SO4--": 1.0, "H+": 0.0, "NO3-": 0.0}.items():
        assert tot.get(ion, 0.0) == pytest.approx(b, abs=1e-8)


def test_open_reservoir_supersaturated_reports_failure(sol):
    r = sol.solve({"Na+": 1.0, "Cl-": 1.0}, T0, 0.3, p_gas={"NH3": 1e-8, "HNO3": 1e-8})
    assert r.status == "failed" and "supersaturated" in r.message


def test_closed_dry_ammonium_nitrate_dissociation_pressure(sol):
    """Solid NH4NO3 in equilibrium with its vapour: p(NH3) p(HNO3) = Kp(T) exactly."""
    r = sol.solve_closed({"NH4+": 1e-5, "NO3-": 1e-5}, {"HNO3": 0.0, "NH3": 0.0}, T0, 0.3, n_air=1e-3)
    assert r.status == "dry" and "ammonium_nitrate" in r.solids
    kp = math.exp(SOLIDS["ammonium_nitrate"].ln_k(T0) - GASES["NH3"].ln_k(T0) - GASES["HNO3"].ln_k(T0))
    assert r.p_gas["NH3"] * r.p_gas["HNO3"] == pytest.approx(kp, rel=1e-6)
    tot = _totals(r)
    assert tot["NH4+"] == pytest.approx(1e-5, rel=1e-9) and tot["NO3-"] == pytest.approx(1e-5, rel=1e-9)


def test_closed_complete_evaporation(sol):
    r = sol.solve_closed({"NH4+": 1e-8, "NO3-": 1e-8}, {"HNO3": 0.0, "NH3": 0.0}, T0, 0.3, n_air=41.0)
    assert r.status == "dry" and not r.solids
    assert r.gas["NH3"] == pytest.approx(1e-8, rel=1e-6) and r.gas["HNO3"] == pytest.approx(1e-8, rel=1e-6)
    assert r.p_gas["NH3"] == pytest.approx(1e-8 / (41.0 + 2e-8), rel=1e-6)


def test_closed_wet_ammonium_sulfate_with_hno3_and_nh3(sol):
    feed, gt = {"NH4+": 2e-6, "SO4--": 1e-6}, {"HNO3": 2e-6, "NH3": 3e-6}
    r = sol.solve_closed(feed, gt, T0, 0.8, n_air=41.0)
    assert r.status == "aqueous"
    tot = _totals(r)
    for ion, b in {"NH4+": 5e-6, "SO4--": 1e-6, "H+": -1e-6, "NO3-": 2e-6}.items():
        assert tot[ion] == pytest.approx(b, rel=1e-8, abs=1e-14)
    # ideal-gas closure and equilibrium: p_j = P n_j / (n_air + sum n)
    ng = sum(r.gas.values())
    for g, n in r.gas.items():
        assert r.p_gas[g] == pytest.approx(n / (41.0 + ng), rel=1e-9)
    la = r.ln_a
    assert math.exp(la["H+"] + la["NO3-"] - GASES["HNO3"].ln_k(T0)) == pytest.approx(r.p_gas["HNO3"], rel=1e-5)
    assert math.exp(la["NH4+"] - la["H+"] - GASES["NH3"].ln_k(T0)) == pytest.approx(r.p_gas["NH3"], rel=1e-5)
    # excess NH3 (base) must stay in the gas phase: acid equivalents are 2e-6 + 2e-6 + ... < NH4 total
    assert r.gas["NH3"] > 0.9e-6


def test_chloride_depletion_by_nitric_acid(sol):
    """NaCl + HNO3(g) -> NaNO3 + HCl(g): chloride leaves the aerosol, nitrate enters it."""
    r = sol.solve_closed({"Na+": 1e-5, "Cl-": 1e-5}, {"HNO3": 2e-5, "HCl": 0.0}, T0, 0.8, n_air=41.0)
    assert r.status == "aqueous"
    assert r.aq_ions["NO3-"] > 0.0 and r.gas["HCl"] > 0.5e-5
    tot = _totals(r)
    assert tot["Cl-"] == pytest.approx(1e-5, rel=1e-8) and tot["Na+"] == pytest.approx(1e-5, rel=1e-8)
    assert tot["NO3-"] == pytest.approx(2e-5, rel=1e-8)

"""Carbonate / CO2 support of the SLE solver (stoichiometric CO3-- and proton excess H+, internal HCO3-/OH- speciation)."""
import math

import numpy as np
import pytest

pytest.importorskip("scipy")

from aiomfac_py.gases import GASES  # noqa: E402
from aiomfac_py.sle import SLESolver  # noqa: E402
from aiomfac_py.solids import SOLIDS  # noqa: E402

T0 = 298.15


@pytest.fixture(scope="module")
def nac():
    return SLESolver(["Na+", "CO3--", "H+"])


@pytest.fixture(scope="module")
def nacl():
    return SLESolver(["Na+", "Cl-", "CO3--", "H+"])


def test_co2_gas_constant_consistent_with_speciation():
    from aiomfac_py.carbonate import ln_k1_hco3_at_t, ln_k2_hco3_at_t
    g = GASES["CO2"]
    assert g.h == -1 and g.ions == {"CO3--": 1, "H+": 2}
    # K_H(CO2) ~ 10^-1.47 mol/kg/atm at 25 C
    lnkh = g.ln_k(T0) - ln_k1_hco3_at_t(T0) - ln_k2_hco3_at_t(T0)
    assert math.log10(math.exp(lnkh)) == pytest.approx(-1.47, abs=0.02)


def test_sodium_carbonate_dry_sequence(nac):
    seq = {}
    for rh in (0.8, 0.5, 0.1):
        r = nac.solve({"Na+": 2e-6, "CO3--": 1e-6}, T0, rh)
        assert r.status == "dry"
        seq[rh] = list(r.solids)
    assert seq == {0.8: ["natron"], 0.5: ["thermonatrite"], 0.1: ["sodium_carbonate"]}


def test_sodium_carbonate_wet_and_mass_balance(nac):
    r = nac.solve({"Na+": 2e-6, "CO3--": 1e-6}, T0, 0.95)
    assert r.status == "aqueous" and not r.solids
    assert r.aq_ions["Na+"] == pytest.approx(2e-6, rel=1e-9)
    assert r.aq_ions["CO3--"] == pytest.approx(1e-6, rel=1e-9)


def test_nahcolite_single_salt(nac):
    r = nac.solve({"Na+": 1e-6, "CO3--": 1e-6, "H+": 1e-6}, T0, 0.9)
    assert r.status == "dry" and list(r.solids) == ["nahcolite"]


def test_open_co2_uptake_and_pressure(nacl):
    feed = {"Na+": 1.1e-5, "Cl-": 1e-5, "H+": -1e-6}
    ups = []
    for p in (1e-4, 4.2e-4, 1e-3):
        r = nacl.solve(feed, T0, 0.8, p_gas={"CO2": p})
        assert r.status == "aqueous"
        assert r.gas["CO2"] < 0
        assert r.p_gas["CO2"] == pytest.approx(p, rel=1e-12)
        # carbon in the aerosol = uptake, charge balance with the proton excess
        assert r.aq_ions["CO3--"] == pytest.approx(-r.gas["CO2"], rel=1e-9)
        ups.append(-r.gas["CO2"])
    assert ups[0] < ups[1] < ups[2]


def test_closed_co2_matches_open_for_large_reservoir(nacl):
    feed = {"Na+": 1.1e-5, "Cl-": 1e-5, "H+": -1e-6}
    n_air, y = 41.0, 4.2e-4
    gt = y * n_air / (1 - y)
    c = nacl.solve_closed(feed, {"CO2": gt}, T0, 0.8, n_air=n_air)
    o = nacl.solve(feed, T0, 0.8, p_gas={"CO2": y})
    assert c.status == "aqueous"
    assert c.gas["CO2"] + c.aq_ions["CO3--"] == pytest.approx(gt, rel=1e-9)
    assert c.p_gas["CO2"] == pytest.approx(y, rel=1e-4)
    assert c.aq_ions["CO3--"] == pytest.approx(-o.gas["CO2"], rel=1e-3)


def test_hydroxide_solids_use_kw():
    s = SOLIDS["portlandite"]
    assert s.n_oh == 2 and s.h_eff == s.h + 2

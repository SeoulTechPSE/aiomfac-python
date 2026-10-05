"""Tests of the combined liquid-liquid-solid equilibrium solver (aiomfac_py.phase_equilibrium).

There is no Fortran reference for phase equilibria (AIOMFAC provides activities only).  The solver is checked
  * against ActivityModel.evaluate (same activities in the ion basis) and the Gibbs-Duhem relation;
  * in its inorganic limit against SLESolver (an independent active-set implementation);
  * in its solid-free limit against the AIOMFAC Gibbs function used by aiomfac_py.lle (the split it finds must lower
    that function, with equal water activity in both phases);
  * by the equilibrium conditions it reports for itself (a_w = RH in every liquid, equal neutral and ion-combination
    potentials across liquids, SI <= 0 with SI = 0 for every solid present, electroneutrality, mass balance).
"""
import math

import numpy as np
import pytest

pytest.importorskip("scipy")

from aiomfac_py import ActivityModel, Component  # noqa: E402
from aiomfac_py.lle import AiomfacGFE  # noqa: E402
from aiomfac_py.phase_equilibrium import LiquidModel, PhaseEquilibrium  # noqa: E402
from aiomfac_py.sle import SLESolver, feed_from_salts, implied_ln_s_crit  # noqa: E402

PINIC = Component(2, "pinic_acid", ((1, 2), (2, 2), (3, 2), (4, 1), (137, 2)))   # S2AS decomposition of pinic acid
WATER = Component(1, "Water", ((16, 1),))
AS = Component(3, "AS", ((204, 2), (261, 1)))


def _ansan_feed(with_an: bool):
    oir, f_an = 1.15, (0.222 if with_an else 0.0)
    m_org, m_as, m_an = oir / 184.19, (1 - f_an) / 132.14, f_an / 80.04
    feed = {"pinic_acid": m_org, "NH4+": 2 * m_as + m_an, "SO4--": m_as}
    if with_an:
        feed["NO3-"] = m_an
    return feed


def _assert_equilibrium(r, tol_si=1e-4):
    c = r.checks
    assert r.status == "converged", r.message
    assert c["max_abs_ln_aw_minus_ln_rh"] < 1e-5
    assert c["max_neutral_mu_spread"] < 1e-4
    assert c["max_ion_mu_residual"] < 1e-4
    assert c["max_si"] < tol_si
    assert c["max_abs_si_present_solids"] < tol_si
    assert c["max_charge_residual"] < 1e-10
    assert c["max_mass_balance_residual"] < 1e-10
    assert r.tpd_min > -1e-7


# ---------------------------------------------------------------------------------------------------------
# activities in the ion basis
# ---------------------------------------------------------------------------------------------------------
def test_liquid_model_matches_activity_model_and_gibbs_duhem():
    T = 298.15
    lm = LiquidModel([PINIC], ["NH4+", "SO4--"])
    n = np.array([5.0, 0.3, 0.2, 0.1])
    la = lm.ln_a(n, T)
    ev = ActivityModel([WATER, Component(2, "org", PINIC.subgroups), AS]).evaluate(np.array([5.0, 0.3, 0.1]) / 5.4,
                                                                                  T, basis="mole")
    assert np.allclose(la[:2], np.log(ev.activity[:2]), atol=1e-12)
    assert 2 * la[2] + la[3] == pytest.approx(math.log(ev.activity[2]), abs=1e-12)
    H = lm.hessian(n, T)
    assert np.max(np.abs(H @ n)) < 1e-7 * np.max(np.abs(H))          # sum_i n_i d ln a_i / d n_j = 0


# ---------------------------------------------------------------------------------------------------------
# inorganic limit: identical to SLESolver
# ---------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("ions,salts,rh", [
    (["NH4+", "SO4--"], {"(NH4)2SO4": 1.0}, 0.85),
    (["Na+", "NH4+", "Cl-", "SO4--"], {"NaCl": 1.0, "(NH4)2SO4": 1.0}, 0.70),
    (["NH4+", "SO4--", "NO3-"], {"(NH4)2SO4": 1.0, "NH4NO3": 1.0}, 0.62),     # solid AS + aqueous
])
def test_inorganic_limit_matches_sle_solver(ions, salts, rh):
    T = 298.15
    feed = feed_from_salts(salts)
    rs = SLESolver(ions).solve(feed, T, rh)
    rp = PhaseEquilibrium([], ions, T_K=T).solve(feed, rh, solids="all")
    _assert_equilibrium(rp)
    assert rp.n_liquids == 1
    L = rp.liquids[0]
    w_kg = L.amounts[0] * 0.01801528
    assert w_kg == pytest.approx(rs.water_kg, rel=1e-5)
    for ion in ions:
        assert L.amounts[L.names.index(ion)] / w_kg == pytest.approx(rs.molality[ion], rel=1e-5)
    assert set(rp.solids) == set(rs.solids)
    for k, v in rs.solids.items():
        assert rp.solids[k] == pytest.approx(v, rel=1e-4)


# ---------------------------------------------------------------------------------------------------------
# solid-free limit: liquid-liquid split of pinic acid + ammonium sulfate
# ---------------------------------------------------------------------------------------------------------
def test_llps_matches_reference_split_and_lowers_lle_gibbs_function():
    T = 300.0
    pe = PhaseEquilibrium([PINIC], ["NH4+", "SO4--"], T_K=T)
    r = pe.solve(_ansan_feed(False), 0.4169, solids="none")
    _assert_equilibrium(r)
    assert r.n_liquids == 2
    # salt-formula basis (water, org, AS) of the two phases; the organic-rich one matches the split found with
    # aiomfac_py.lle at the same water activity (0.3763, 0.5818, 0.0419)
    X = [np.array([L.amounts[0], L.amounts[1], L.amounts[3]]) for L in r.liquids]
    xs = [x / x.sum() for x in X]
    assert np.allclose(xs[0], [0.3763, 0.5818, 0.0419], atol=2e-4)
    assert np.allclose(xs[1], [0.7008, 0.0, 0.2992], atol=2e-4)
    # the split also lowers the Gibbs function of aiomfac_py.lle at the same overall composition
    tot = sum(X); b = tot / tot.sum()
    gfe = AiomfacGFE([WATER, Component(2, "org", PINIC.subgroups), AS], T)
    g_split = sum(x.sum() / tot.sum() * gfe.g(x / x.sum()) for x in X)
    assert g_split < gfe.g(b) - 1e-3


def test_llps_persists_at_low_rh_where_single_phase_is_unstable():
    pe = PhaseEquilibrium([PINIC], ["NH4+", "SO4--"], T_K=300.0)
    feed = _ansan_feed(False)
    single = pe.solve(feed, 0.30, solids="none", max_liquids=1)
    best = pe.solve(feed, 0.30, solids="none")
    assert single.tpd_min < -0.1                         # the one-phase state fails the stability test
    _assert_equilibrium(best)
    assert best.n_liquids == 2 and best.gibbs < single.gibbs - 1e-3


# ---------------------------------------------------------------------------------------------------------
# organics + solids
# ---------------------------------------------------------------------------------------------------------
def test_solid_ammonium_sulfate_with_organic_liquid():
    pe = PhaseEquilibrium([PINIC], ["NH4+", "SO4--"], T_K=300.0)
    r = pe.solve(_ansan_feed(False), 0.30, solids="all")
    _assert_equilibrium(r)
    assert "ammonium_sulfate" in r.solids
    assert r.si["ammonium_sulfate"] == pytest.approx(0.0, abs=1e-4)
    L = r.liquids[0]
    dissolved = L.amounts[L.names.index("SO4--")] / _ansan_feed(False)["SO4--"]
    assert 0.0 < dissolved < 0.01


def test_equilibrium_sequence_with_nitrate():
    pe = PhaseEquilibrium([PINIC], ["NH4+", "SO4--", "NO3-"], T_K=300.0)
    feed = _ansan_feed(True)
    hi = pe.solve(feed, 0.80, solids="all")
    mid = pe.solve(feed, 0.60, solids="all")
    lo = pe.solve(feed, 0.30, solids="all")
    for r in (hi, mid, lo):
        _assert_equilibrium(r)
    assert hi.n_liquids == 2 and not hi.solids                      # LLPS, no solid
    assert mid.n_liquids == 2 and set(mid.solids) == {"ammonium_sulfate"}
    assert set(lo.solids) == {"ammonium_sulfate", "ammonium_nitrate"}


def test_drying_path_crystallizes_only_after_critical_supersaturation():
    T = 300.0
    pe = PhaseEquilibrium([PINIC], ["NH4+", "SO4--", "NO3-"], T_K=T)
    lsc = {"ammonium_sulfate": implied_ln_s_crit("ammonium_sulfate", T, 0.35), "ammonium_nitrate": 99.0}
    path = pe.drying_path(_ansan_feed(True), [0.60, 0.40, 0.30, 0.20], ln_s_crit=lsc)
    by_rh = {round(r.rh, 2): r for r in path}
    assert not by_rh[0.60].solids and not by_rh[0.40].solids        # supersaturated but metastable
    assert by_rh[0.60].si["ammonium_sulfate"] > 0
    assert "ammonium_sulfate" in by_rh[0.20].solids
    for r in path:
        _assert_equilibrium(r)


def test_rejects_unsupported_ions_and_charged_feed():
    with pytest.raises(ValueError):                         # carbonate needs H+ as the proton excess
        PhaseEquilibrium([PINIC], ["NH4+", "CO3--"], T_K=298.15)
    with pytest.raises(ValueError):                         # HCO3- is speciated internally, not an input
        PhaseEquilibrium([PINIC], ["Na+", "H+", "HCO3-"], T_K=298.15)
    with pytest.raises(ValueError):
        PhaseEquilibrium([PINIC], ["H+", "HSO4-"], T_K=298.15)
    pe = PhaseEquilibrium([PINIC], ["NH4+", "SO4--"], T_K=298.15)
    with pytest.raises(ValueError):
        pe.solve({"pinic_acid": 0.1, "NH4+": 0.1, "SO4--": 0.1}, 0.5)


def test_binary_binodal_close_to_saturation_matches_common_tangent():
    """Water + pinonaldehyde (289 K): the common tangent of a_w and a_org gives a_w* = 0.99825 with coexisting
    x_org = 0.0019 and 0.675.  Below a_w* the organic-rich liquid is the stable one, above it the dilute one; the
    stability test must find the organic-rich liquid even when the start is the dilute (metastable) branch."""
    pinonaldehyde = Component(2, "pinonaldehyde", ((1, 2), (2, 1), (3, 2), (4, 1), (18, 1), (20, 1)))   # S2AS
    pe = PhaseEquilibrium([pinonaldehyde], [], T_K=289.0)
    below = pe.solve({"pinonaldehyde": 1.0}, 0.995, solids="none")
    above = pe.solve({"pinonaldehyde": 1.0}, 0.9990, solids="none")
    _assert_equilibrium(below)
    _assert_equilibrium(above)
    assert below.liquids[0].mole_fractions[1] > 0.5
    assert above.liquids[0].mole_fractions[1] < 0.01


# ---------------------------------------------------------------------------------------------------------
# acid sulfate: stoichiometric H+ and SO4--, bisulfate speciated in every liquid
# ---------------------------------------------------------------------------------------------------------
def test_acid_activities_match_activity_model_with_bisulfate_speciation():
    T = 298.15
    lm = LiquidModel([PINIC], ["NH4+", "H+", "SO4--"])
    n = np.array([5.0, 0.3, 0.2, 0.1, 0.15])                      # water, pinic, NH4+, H+ (total), SO4-- (total)
    la = lm.ln_a(n, T)
    m = ActivityModel([WATER, Component(2, "org", PINIC.subgroups), AS, Component(4, "H2SO4", ((205, 2), (261, 1)))])
    x = np.array([5.0, 0.3, 0.1, 0.05]); x /= x.sum()
    ev = m.evaluate(x, T, basis="mole")
    assert np.allclose(la[:2], np.log(ev.activity[:2]), atol=1e-12)
    assert 2 * la[2] + la[4] == pytest.approx(math.log(ev.activity[2]), abs=1e-10)
    assert 2 * la[3] + la[4] == pytest.approx(math.log(ev.activity[3]), abs=1e-10)
    H = lm.hessian(n, T)
    assert np.max(np.abs(H @ n)) < 1e-7 * np.max(np.abs(H))


@pytest.mark.parametrize("feed,rh,status,solids", [
    ({"NH4+": 1.0, "H+": 1.0, "SO4--": 1.0}, 0.45, "converged", {"letovicite"}),     # letovicite + aqueous
    ({"NH4+": 3.0, "H+": 1.0, "SO4--": 2.0}, 0.60, "dry", {"letovicite"}),           # below the DRH: no liquid
])
def test_acid_inorganic_limit_matches_sle_solver(feed, rh, status, solids):
    T = 298.15
    ions = ["NH4+", "H+", "SO4--"]
    rs = SLESolver(ions).solve(feed, T, rh)
    rp = PhaseEquilibrium([], ions, T_K=T).solve(feed, rh, solids="all")
    assert rp.status == status and set(rp.solids) == solids == set(rs.solids)
    for k, v in rs.solids.items():
        assert rp.solids[k] == pytest.approx(v, rel=1e-4)
    if status == "converged":
        _assert_equilibrium(rp)
        L = rp.liquids[0]
        w_kg = L.amounts[0] * 0.01801528
        assert w_kg == pytest.approx(rs.water_kg, rel=1e-5)
        for ion in ions:
            assert L.amounts[L.names.index(ion)] / w_kg == pytest.approx(rs.molality[ion], rel=1e-5)


# ---------------------------------------------------------------------------------------------------------
# gas phase (NH3, HNO3, HCl, CO2) and carbonate system: inorganic limit equals SLESolver
# ---------------------------------------------------------------------------------------------------------
_GAS_IONS = ["Na+", "NH4+", "H+", "NO3-", "Cl-", "SO4--"]


def _cmp_with_sle(rs, rp, rel=1e-5):
    assert rp.status == "converged", rp.message
    L = rp.liquids[0]
    w_kg = L.amounts[0] * 0.01801528
    assert w_kg == pytest.approx(rs.water_kg, rel=rel)
    for ion, m in rs.molality.items():
        if m > 1e-12:
            assert L.amounts[L.names.index(ion)] / w_kg == pytest.approx(m, rel=rel)
    for g, v in rs.gas.items():
        assert rp.gas[g] == pytest.approx(v, rel=rel)
        assert rp.p_gas[g] == pytest.approx(rs.p_gas[g], rel=rel)


def test_open_gas_exchange_matches_sle_solver():
    p = {"NH3": 1.0e-9, "HNO3": 1.0e-9}
    rs = SLESolver(_GAS_IONS).solve({"NH4+": 2.0, "SO4--": 1.0}, 298.15, 0.8, p_gas=p)
    rp = PhaseEquilibrium([], _GAS_IONS, T_K=298.15).solve({"NH4+": 2.0, "SO4--": 1.0}, 0.8, p_gas=p)
    _cmp_with_sle(rs, rp)


@pytest.mark.parametrize("feed,gt", [
    ({"NH4+": 2e-6, "SO4--": 1e-6}, {"HNO3": 2e-6, "NH3": 3e-6}),
    ({"Na+": 1e-5, "Cl-": 1e-5}, {"HNO3": 2e-5, "HCl": 0.0}),          # chloride depletion: HCl evaporates
])
def test_closed_gas_phase_matches_sle_solver(feed, gt):
    rs = SLESolver(_GAS_IONS).solve_closed(feed, gt, 298.15, 0.8, n_air=41.0)
    rp = PhaseEquilibrium([], _GAS_IONS, T_K=298.15).solve(feed, 0.8, gas_total=gt, n_air=41.0)
    _cmp_with_sle(rs, rp)


def test_carbonate_with_co2_matches_sle_solver():
    ions = ["Na+", "Cl-", "CO3--", "H+"]
    feed = {"Na+": 1.1e-5, "Cl-": 1e-5, "H+": -1e-6}                  # negative proton excess (base)
    sle, pe = SLESolver(ions), PhaseEquilibrium([], ions, T_K=298.15)
    _cmp_with_sle(sle.solve(feed, 298.15, 0.8, p_gas={"CO2": 4.2e-4}), pe.solve(feed, 0.8, p_gas={"CO2": 4.2e-4}))
    n_air, y = 41.0, 4.2e-4
    gtc = {"CO2": y * n_air / (1 - y)}
    _cmp_with_sle(sle.solve_closed(feed, gtc, 298.15, 0.8, n_air=n_air), pe.solve(feed, 0.8, gas_total=gtc, n_air=n_air))


def test_organic_with_hcl_evaporation_is_in_equilibrium():
    """Pinic acid + NaCl + H2SO4 open to 1e-9 atm of HCl: practically all chloride leaves the particle."""
    pe = PhaseEquilibrium([PINIC], ["Na+", "H+", "Cl-", "SO4--"], T_K=298.15)
    m = 1.0 / 58.44
    feed = {"pinic_acid": 3.0 / 184.19, "Na+": m, "Cl-": m, "H+": 2 * m, "SO4--": m}
    r = pe.solve(feed, 0.8, solids="none", p_gas={"HCl": 1e-9})
    _assert_equilibrium(r)
    assert r.gas["HCl"] > 0.99 * m                                   # practically all chloride evaporated
    assert r.checks["max_abs_gas_residual"] < 1e-4


def test_near_complete_hcl_evaporation_converges():
    """DLT + NaCl + H2SO4 (r = 0.75) open to 1e-9 atm HCl at RH 0.2: chloride evaporates to a trace that must still
    satisfy the gas and ion-potential conditions (relative-step convergence test of the inner iteration)."""
    from aiomfac_py.s2as import smiles_to_components
    try:
        dlt = smiles_to_components(["CCOC(=O)C(O)C(O)C(=O)OCC"], names=["DLT"]).components[1]
    except ImportError:
        pytest.skip("S2AS (epam.indigo) not installed")
    pe = PhaseEquilibrium([dlt], ["Na+", "H+", "Cl-", "SO4--"], T_K=298.15)
    m, r = 1 / 58.44, 0.75
    feed = {"DLT": 3 / 206.19, "Na+": m, "Cl-": m, "H+": 2 * r * m, "SO4--": r * m}
    res = pe.solve(feed, 0.2, solids="none", p_gas={"HCl": 1e-9})
    _assert_equilibrium(res)
    assert res.gas["HCl"] > 0.999 * m

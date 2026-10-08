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


def _dlt_nacl_acid(r):
    from aiomfac_py.s2as import smiles_to_components
    try:
        dlt = smiles_to_components(["CCOC(=O)C(O)C(O)C(=O)OCC"], names=["DLT"]).components[1]
    except ImportError:
        pytest.skip("S2AS (epam.indigo) not installed")
    m = 1 / 58.44
    pe = PhaseEquilibrium([dlt], ["Na+", "H+", "Cl-", "SO4--"], T_K=298.15)
    return pe, {"DLT": 3 / 206.19, "Na+": m, "Cl-": m, "H+": 2 * r * m, "SO4--": r * m}


def test_trace_entries_are_removed_from_single_liquids():
    """DLT + NaCl + H2SO4 (r = 1.5) open to HCl at RH 0.1: DLT and the last chloride are traces (< 1e-10 of the feed)
    in the salt-rich liquid; removing them from that liquid lets the potential conditions converge."""
    pe, feed = _dlt_nacl_acid(1.5)
    pe.inner_method = "newton"                                     # trace removal belongs to the primal Newton solver
    res = pe.solve(feed, 0.1, solids="none", p_gas={"HCl": 1e-9})
    _assert_equilibrium(res)
    assert res.checks["n_absent_entries"] >= 1
    assert res.checks["max_removed_trace"] < 1e-9


def test_new_liquid_close_to_its_appearance_is_found_with_a_smaller_seed():
    """DLT + NaCl + H2SO4 (r = 1.5) open to HCl at RH 0.5: a third liquid (Na-sulfate-rich) has just appeared
    (TPD of the two-liquid state -1.6e-3).  With the barrier inner solver and a fixed seed of 0.5 the inner solve fell
    back to two liquids and a smaller seed was needed; the line-search seed and the fixed seeds must both find three."""
    for method in ("linesearch", "fixed"):
        pe, feed = _dlt_nacl_acid(1.5)
        pe.seed_method = method
        res = pe.solve(feed, 0.5, solids="none", p_gas={"HCl": 1e-9})
        _assert_equilibrium(res)
        assert res.n_liquids == 3


# ---------------------------------------------------------------------------------------------------------
# inner-iteration cost reductions: split Hessian and the warm-started bisulfate speciation
# ---------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("ions,n", [
    (["NH4+", "SO4--"], [5.0, 0.3, 0.2, 0.1]),
    (["NH4+", "SO4--"], [5.0, 1e-9, 0.2, 0.1]),                                  # trace organic
    (["Na+", "H+", "Cl-", "SO4--"], [3.0, 0.4, 0.2, 0.3, 1e-10, 0.25]),          # acid, trace chloride
])
def test_split_hessian_matches_central_differences(ions, n):
    """exact ideal Jacobian + forward-difference excess part = the central-difference Hessian (to the FD error)."""
    lm = LiquidModel([PINIC], ions)
    n = np.array(n, dtype=float)
    Hc, Hs = lm.hessian(n, 298.15), lm.hessian_split(n, 298.15)
    d = np.sqrt(np.abs(np.diag(Hc)))
    assert np.max(np.abs(Hs - Hc) / np.outer(d, d)) < 1e-4
    assert np.max(np.abs(Hs @ n)) < 1e-5 * np.max(np.abs(Hs))                       # Gibbs-Duhem


def test_fast_bisulfate_speciation_matches_bracketing_solver():
    """the warm-started fixed-point/secant speciation of LiquidModel equals dissociation.solve_bisulfate (1e-10)."""
    fast = LiquidModel([PINIC], ["Na+", "NH4+", "H+", "Cl-", "SO4--"])
    slow = LiquidModel([PINIC], ["Na+", "NH4+", "H+", "Cl-", "SO4--"])
    slow._speciate_hso4 = lambda *a: None                                        # forces the fallback solver
    rng = np.random.default_rng(1)
    n_cases = 0
    while n_cases < 40:
        na, nh4, h = rng.uniform(0, 1, 3)
        h *= 10 ** rng.uniform(-6, 0.5)
        cl = rng.uniform(0, 1) * (na + nh4)
        so4 = (na + nh4 + h - cl) / 2
        if so4 <= 0:
            continue
        n = np.array([rng.uniform(0.5, 10), rng.uniform(1e-6, 3), na, nh4, h, cl, so4])
        assert np.max(np.abs(fast.ln_a(n, 298.15) - slow.ln_a(n, 298.15))) < 1e-10
        n_cases += 1


def test_hessian_schemes_give_the_same_equilibrium():
    """the split (default) and the original central-difference Hessian converge to the same state."""
    feed = _ansan_feed(False)
    out = []
    for scheme in ("central", "split"):
        pe = PhaseEquilibrium([PINIC], ["NH4+", "SO4--"], T_K=298.15)
        pe.hess_scheme = scheme
        r = pe.solve(feed, 0.30, solids="none")
        _assert_equilibrium(r)
        out.append(r)
    assert out[0].n_liquids == out[1].n_liquids == 2
    assert out[1].gibbs == pytest.approx(out[0].gibbs, rel=1e-9)
    for La, Lb in zip(out[0].liquids, out[1].liquids):
        np.testing.assert_allclose(Lb.amounts, La.amounts, rtol=1e-6, atol=1e-12)


@pytest.mark.parametrize("ions", [["NH4+", "H+", "SO4--"], ["Na+", "Cl-", "CO3--", "H+"]])
def test_ad_activities_and_jacobian_match_numpy(ions):
    """The JAX transcription (aiomfac_py.ad_activity) reproduces ExplicitLiquidModel.ln_a to round-off and its
    Jacobian matches 4th-order central differences (NH4+/H+ includes the Qcca and Rcc terms, the carbonate system
    the CO2(aq) salting-out term)."""
    pytest.importorskip("jax")
    from aiomfac_py.ad_activity import build_jacobian
    from aiomfac_py.phase_equilibrium import ExplicitLiquidModel
    T = 298.15
    lm = ExplicitLiquidModel([PINIC], ions)
    jac, ln_a = build_jacobian(lm, T)
    rng = np.random.default_rng(3)
    for _ in range(3):
        n = np.exp(rng.uniform(np.log(1e-3), 0.0, lm.N)); n[0] = rng.uniform(1, 10)
        q = lm.z @ n
        k = int(np.flatnonzero(lm.z < 0 if q > 0 else lm.z > 0)[0]); n[k] += abs(q / lm.z[k])
        assert np.max(np.abs(np.asarray(ln_a(n)) - (lm.ln_a(n, T) - lm._c))) < 1e-12
        J = np.asarray(jac(n))
        F = np.zeros_like(J)
        for j in range(lm.N):
            h = 1e-3 * n[j]
            e = np.zeros(lm.N); e[j] = h
            F[:, j] = (-lm.ln_a(n + 2 * e, T) + 8 * lm.ln_a(n + e, T) - 8 * lm.ln_a(n - e, T)
                       + lm.ln_a(n - 2 * e, T)) / (12 * h)
        assert np.max(np.abs(J - F)) < 1e-9 * np.max(np.abs(J))


@pytest.mark.parametrize("case", ["dlt_3liq", "org_carbonate"])
def test_ad_hessian_gives_the_same_equilibrium(case):
    """hess_scheme "ad" (exact Jacobian by automatic differentiation) converges to the state of the default scheme."""
    pytest.importorskip("jax")
    if case == "dlt_3liq":
        pe0, feed = _dlt_nacl_acid(1.5)
        make = lambda: PhaseEquilibrium(pe0._organics, ["Na+", "H+", "Cl-", "SO4--"], T_K=298.15)
        args, kw = (feed, 0.5), dict(solids="none", p_gas={"HCl": 1e-9})
    else:
        make = lambda: PhaseEquilibrium([PINIC], ["Na+", "Cl-", "CO3--", "H+"], T_K=298.15)
        feed = {"pinic_acid": 3 / 184.19, "Na+": 1 / 58.44 + 2e-3, "Cl-": 1 / 58.44, "H+": -2e-3}
        args, kw = (feed, 0.5), dict(solids="none", p_gas={"CO2": 4.2e-4})
    out = {}
    for scheme in ("split", "ad"):
        pe = make()
        pe.hess_scheme = scheme
        out[scheme] = pe.solve(*args, **kw)
        _assert_equilibrium(out[scheme])
    assert pe.lm.n_jac > 0
    assert out["ad"].n_liquids == out["split"].n_liquids
    assert out["ad"].gibbs == pytest.approx(out["split"].gibbs, abs=1e-9)


def _rand_cases():
    dlt_pe, dlt_feed = _dlt_nacl_acid(1.5)
    org = dlt_pe._organics
    m_org, m_as, m_an = 1.15 / 184.19, 0.778 / 132.14, 0.222 / 80.04
    return {
        "pinic_as": (lambda: PhaseEquilibrium([PINIC], ["NH4+", "SO4--"], T_K=298.15),
                     (_ansan_feed(False), 0.30), dict(solids="none")),
        "pinic_as_an_solid": (lambda: PhaseEquilibrium([PINIC], ["NH4+", "SO4--", "NO3-"], T_K=298.15),
                              ({"pinic_acid": m_org, "NH4+": 2 * m_as + m_an, "SO4--": m_as, "NO3-": m_an}, 0.6), {}),
        "dlt_3liq": (lambda: PhaseEquilibrium(org, ["Na+", "H+", "Cl-", "SO4--"], T_K=298.15),
                     (dlt_feed, 0.5), dict(solids="none", p_gas={"HCl": 1e-9})),
        "org_carbonate": (lambda: PhaseEquilibrium([PINIC], ["Na+", "Cl-", "CO3--", "H+"], T_K=298.15),
                          ({"pinic_acid": 3 / 184.19, "Na+": 1 / 58.44 + 2e-3, "Cl-": 1 / 58.44, "H+": -2e-3}, 0.5),
                          dict(solids="none", p_gas={"CO2": 4.2e-4})),
    }


@pytest.mark.parametrize("case", ["pinic_as", "pinic_as_an_solid", "dlt_3liq", "org_carbonate"])
def test_rand_inner_solver_matches_newton(case):
    """inner_method "rand" (logarithmic amounts, no barrier, no trace removal) gives the phases, solids and F of the
    primal Newton solver; F may be lower by the trace entries that the primal solver removes (2e-11 for three liquids)."""
    make, args, kw = _rand_cases()[case]
    out = {}
    for method in ("newton", "rand"):
        pe = make()
        pe.inner_method = method
        out[method] = pe.solve(*args, **kw)
        _assert_equilibrium(out[method])
    a, b = out["newton"], out["rand"]
    assert b.n_liquids == a.n_liquids
    assert set(b.solids) == set(a.solids)
    assert b.gibbs == pytest.approx(a.gibbs, abs=1e-10)
    assert b.gibbs <= a.gibbs + 1e-12
    assert b.checks["n_absent_entries"] == 0


def test_rand_resolves_traces_without_removal():
    """DLT + NaCl + H2SO4 (r = 1.5) open to HCl at RH 0.1, the case that needs trace removal in the primal solver
    (test_trace_entries_are_removed_from_single_liquids): with logarithmic amounts every species stays in every liquid,
    DLT far below 1e-30 of the salt-rich liquid, and all equilibrium conditions are met."""
    pe, feed = _dlt_nacl_acid(1.5)
    pe.inner_method = "rand"
    res = pe.solve(feed, 0.1, solids="none", p_gas={"HCl": 1e-9})
    _assert_equilibrium(res)
    assert res.checks["n_absent_entries"] == 0
    salt = min(res.liquids, key=lambda L: L.mole_fractions[1])
    assert 0.0 < salt.mole_fractions[1] < 1e-30


def test_rand_binary_below_llps_onset_finds_the_organic_rich_liquid():
    """Water + pinonaldehyde (S2AS) at 289 K, a_w 0.0005 below the LLPS onset (a_w* = 0.99825, paper_2): the stable
    state is one organic-rich liquid (x_org 0.68).  Accepting uphill steps on the reduced-gradient criterion emptied
    the freshly seeded organic-rich liquid; RAND (like Newton) must find it."""
    from aiomfac_py.s2as import smiles_to_components
    try:
        c = smiles_to_components(["CC(=O)C1CC(C=O)C1(C)C"], names=["pinonaldehyde"]).components[1]
    except ImportError:
        pytest.skip("S2AS (epam.indigo) not installed")
    out = {}
    for method in ("newton", "rand"):
        pe = PhaseEquilibrium([c], [], T_K=289.0)
        pe.inner_method = method
        out[method] = pe.solve({"pinonaldehyde": 1.0}, 0.99825 - 0.0005, solids="none")
        _assert_equilibrium(out[method])
    assert out["rand"].liquids[0].mole_fractions[1] == pytest.approx(0.6755, abs=1e-3)
    assert out["rand"].gibbs == pytest.approx(out["newton"].gibbs, abs=1e-10)


def test_rand_small_draining_liquid_is_merged_and_reseeded():
    """Three-liquid DLT + NaCl + H2SO4 state with the fixed seed 0.5: the seeded liquid drains slowly in a non-convex
    region; the RAND solver merges it, and the smaller seed then finds the third liquid."""
    pe, feed = _dlt_nacl_acid(1.5)
    pe.inner_method = "rand"
    pe.seed_method = "fixed"
    res = pe.solve(feed, 0.5, solids="none", p_gas={"HCl": 1e-9})
    _assert_equilibrium(res)
    assert res.n_liquids == 3


def test_organic_carbonate_two_liquids_with_co2():
    """pinic acid + NaCl + base open to 420 ppm CO2 at RH 0.5: two liquids, carbonate and proton excess are traces in
    the organic-rich liquid (their Hessian columns need steps relative to their own amounts)."""
    pe = PhaseEquilibrium([PINIC], ["Na+", "Cl-", "CO3--", "H+"], T_K=298.15)
    feed = {"pinic_acid": 3 / 184.19, "Na+": 1 / 58.44 + 2e-3, "Cl-": 1 / 58.44, "H+": -2e-3}
    r = pe.solve(feed, 0.5, solids="none", p_gas={"CO2": 4.2e-4})
    _assert_equilibrium(r)
    assert r.n_liquids == 2
    assert r.checks["max_abs_gas_residual"] < 1e-4
    assert r.gas["CO2"] < 0                                                       # CO2 taken up by the basic liquid


@pytest.mark.parametrize("case", ["pinic_as", "dlt_hcl"])
def test_stability_test_methods_give_the_same_equilibrium(case):
    """Successive-substitution and barrier-Newton stability tests lead to the same equilibrium (same number of
    liquids, F equal within the trace entries removed along either path)."""
    if case == "pinic_as":
        make = lambda: PhaseEquilibrium([PINIC], ["NH4+", "SO4--"], T_K=300.0)
        args = (_ansan_feed(False), 0.30)
        kw = dict(solids="none")
    else:
        pe0, feed = _dlt_nacl_acid(0.75)
        make = lambda: PhaseEquilibrium(pe0._organics, ["Na+", "H+", "Cl-", "SO4--"], T_K=298.15)
        args = (feed, 0.2)
        kw = dict(solids="none", p_gas={"HCl": 1e-9})
    out = {}
    for method in ("ss", "newton"):
        pe = make()
        pe.tpd_method = method
        out[method] = pe.solve(*args, **kw)
        _assert_equilibrium(out[method])
    assert out["ss"].n_liquids == out["newton"].n_liquids
    assert out["ss"].gibbs == pytest.approx(out["newton"].gibbs, abs=1e-7)


def test_successive_substitution_finds_the_unstable_direction():
    """From the one-liquid state of pinic acid + AS at RH 0.30 (unstable, TPD < -0.1 with the Newton method), the
    substitution reaches a composition with the same negative TPD within 1e-6."""
    pe = PhaseEquilibrium([PINIC], ["NH4+", "SO4--"], T_K=300.0)
    single = pe.solve(_ansan_feed(False), 0.30, solids="none", max_liquids=1)
    L = single.liquids[0]
    mu = pe._reference_potentials(single.liquids, math.log(0.30))
    best = {}
    for method in ("ss", "newton"):
        pe.tpd_method = method
        ts = []
        for w0 in pe._trial_points(L.amounts / L.total, 0.30, single.liquids):
            w, t = pe._tpd_minimize(mu, w0, refs=[L.mole_fractions])
            if np.all(np.isfinite(w)) and np.isfinite(t):
                ts.append(t)
        best[method] = min(ts)
    assert best["ss"] < -0.1
    assert best["ss"] == pytest.approx(best["newton"], abs=1e-6)


@pytest.mark.parametrize("rh,solids", [(0.6, "all"), (0.3, "all"), (0.8, "all")])
def test_active_set_solids_match_barrier_solids(rh, solids):
    """The active-set treatment of solids (default) gives the same phases, solids and F as the barrier treatment."""
    out = {}
    for method in ("newton", "barrier"):
        pe = PhaseEquilibrium([PINIC], ["NH4+", "SO4--", "NO3-"], T_K=300.0)
        pe.inner_method = method
        out[method] = pe.solve(_ansan_feed(True), rh, solids=solids)
        _assert_equilibrium(out[method])
    a, b = out["newton"], out["barrier"]
    assert a.n_liquids == b.n_liquids and set(a.solids) == set(b.solids)
    for k in a.solids:
        assert a.solids[k] == pytest.approx(b.solids[k], rel=1e-6)
    assert a.gibbs == pytest.approx(b.gibbs, abs=1e-8)
    assert a.checks["max_abs_si_present_solids"] < 1e-8


def _org_carbonate_case(speciation):
    pe = PhaseEquilibrium([PINIC], ["Na+", "Cl-", "CO3--", "H+"], T_K=298.15, speciation=speciation)
    feed = {"pinic_acid": 3 / 184.19, "Na+": 1 / 58.44 + 2e-3, "Cl-": 1 / 58.44, "H+": -2e-3}
    return pe.solve(feed, 0.5, solids="none", p_gas={"CO2": 4.2e-4})


def _dlt_hcl_case(speciation):
    pe0, feed = _dlt_nacl_acid(0.75)
    pe = PhaseEquilibrium(pe0._organics, ["Na+", "H+", "Cl-", "SO4--"], T_K=298.15, speciation=speciation)
    return pe.solve(feed, 0.2, solids="none", p_gas={"HCl": 1e-9})


@pytest.mark.parametrize("make", [_dlt_hcl_case, _org_carbonate_case])
def test_explicit_and_internal_speciation_give_the_same_equilibrium(make):
    """HSO4- / carbonate species as explicit variables (default) or speciated inside every activity evaluation: the same
    phases and component amounts (rel 1e-5) and F within the removed traces."""
    a, b = make("explicit"), make("internal")
    _assert_equilibrium(a)
    _assert_equilibrium(b)
    assert a.n_liquids == b.n_liquids
    for La, Lb in zip(a.liquids, b.liquids):
        assert La.names == Lb.names
        big = La.amounts > 1e-6 * La.amounts.sum()
        assert np.allclose(La.amounts[big], Lb.amounts[big], rtol=1e-5)
    assert a.gibbs == pytest.approx(b.gibbs, abs=5e-8)


def test_explicit_species_are_at_reaction_equilibrium():
    """The species potentials of the explicit model satisfy the speciation equilibria in every liquid, and the species
    amounts add up to the component amounts."""
    r = _org_carbonate_case("explicit")
    for L in r.liquids:
        mu = dict(zip(L.species_names, L.species_ln_a))
        assert mu["HCO3-"] - mu["H+"] - mu["CO3--"] == pytest.approx(0.0, abs=1e-6)
        assert mu["CO2(aq)"] - 2 * mu["H+"] - mu["CO3--"] == pytest.approx(0.0, abs=1e-6)
        assert mu["OH-"] + mu["H+"] == pytest.approx(0.0, abs=1e-6)
        comp = dict(zip(L.names, L.amounts))
        sp = dict(zip(L.species_names, L.species_amounts))
        assert comp["CO3--"] == pytest.approx(sp["CO3--"] + sp["HCO3-"] + sp["CO2(aq)"], rel=1e-12)
        assert comp["H+"] == pytest.approx(sp["H+"] - sp["OH-"] + sp["HCO3-"] + 2 * sp["CO2(aq)"], rel=1e-9, abs=1e-18)
    r = _dlt_hcl_case("explicit")
    for L in r.liquids:
        mu = dict(zip(L.species_names, L.species_ln_a))
        if np.isfinite(mu["HSO4-"]):
            assert mu["HSO4-"] - mu["H+"] - mu["SO4--"] == pytest.approx(0.0, abs=1e-6)


def test_warm_started_rh_scan_matches_cold_solves():
    """rh_scan with warm starts (previous liquids, solids and gas as the start) gives the same states as solving every
    RH from scratch, through a change of the number of liquids."""
    pe0, feed = _dlt_nacl_acid(3.0)
    grid = [0.98, 0.95, 0.6, 0.3]
    res = {}
    for warm in (True, False):
        pe = PhaseEquilibrium(pe0._organics, ["Na+", "H+", "Cl-", "SO4--"], T_K=298.15)
        res[warm] = pe.rh_scan(feed, grid, warm=warm, solids="none")
    assert [r.n_liquids for r in res[True]] == [r.n_liquids for r in res[False]]
    assert len({r.n_liquids for r in res[True]}) > 1
    for a, b in zip(res[True], res[False]):
        _assert_equilibrium(a)
        assert a.gibbs == pytest.approx(b.gibbs, abs=1e-10)


def test_incompatible_warm_start_is_ignored():
    """A previous result of another feed is not used as the start (the solve proceeds from scratch)."""
    pe = PhaseEquilibrium([PINIC], ["NH4+", "SO4--"], T_K=300.0)
    other = pe.solve({"pinic_acid": 0.01, "NH4+": 0.02, "SO4--": 0.01}, 0.4, solids="none")
    a = pe.solve(_ansan_feed(False), 0.30, solids="none", init=other)
    b = pe.solve(_ansan_feed(False), 0.30, solids="none")
    _assert_equilibrium(a)
    assert a.gibbs == pytest.approx(b.gibbs, abs=1e-12)


def test_warm_drying_path_past_a_disappearing_salt_liquid():
    """Pinic acid + AS + AN (Seoul composition, 290 K) on the drying path: the small salt liquid of RH 0.20 disappears
    at lower RH.  Warm-started from it, the solves at 0.15-0.05 failed (a salt liquid without water); small liquids are
    now merged into the largest one at the start, with a cold restart as the fallback."""
    from aiomfac_py.sle import implied_ln_s_crit
    T, oir, f_an = 290.0, 2.78, 0.33124116480218            # paper_1 notebook 06, Seoul site mean
    m_org, m_as, m_an = oir / 186.207, (1 - f_an) / 132.14, f_an / 80.04
    feed = {"pinic_acid": m_org, "NH4+": 2 * m_as + m_an, "SO4--": m_as, "NO3-": m_an}
    lsc = {"ammonium_sulfate": implied_ln_s_crit("ammonium_sulfate", T, 0.35), "ammonium_nitrate": 99.0}
    grid = [0.80, 0.70, 0.60, 0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10, 0.05]
    paths = {w: PhaseEquilibrium([PINIC], ["NH4+", "SO4--", "NO3-"], T_K=T).drying_path(feed, grid, ln_s_crit=lsc,
                                                                                        warm=w) for w in (True, False)}
    for a, b in zip(paths[True], paths[False]):
        _assert_equilibrium(a)
        assert a.n_liquids == b.n_liquids and set(a.solids) == set(b.solids)
        assert a.gibbs == pytest.approx(b.gibbs, abs=1e-10)



def test_pure_water_liquid_is_dropped():
    """A liquid that holds only water cannot satisfy a_w = RH < 1 and only raises F; the outer loop drops it.  With a
    strongly non-ideal (Margules, A = 35, Gibbs-Duhem consistent) binary, the water-rich trial liquid holds about
    e^-35 of organic, which the trace removal takes out, leaving pure water (previously reported as two liquids,
    not converged; first seen with surrogate activity models)."""
    from aiomfac_py.phase_equilibrium import ExplicitLiquidModel
    oil = Component(2, "oil", ((1, 2), (2, 6)))

    class Margules(ExplicitLiquidModel):
        def ln_a(self, n, T):
            self.n_eval += 1
            n = np.asarray(n, dtype=float)
            x = n / n.sum()
            with np.errstate(divide="ignore"):
                return np.log(x) + 35.0 * x[::-1] ** 2 + self._c

    pe = PhaseEquilibrium([oil], [], T_K=298.15)
    pe.lm = Margules([oil], [])
    pe.hess_scheme = "central"
    r = pe.solve({"oil": 1.0}, 0.99999, solids="none")
    assert r.status == "converged", r.message
    assert r.n_liquids == 1 and r.liquids[0].mole_fractions[1] > 0.99

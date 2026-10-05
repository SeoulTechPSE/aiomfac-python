"""Acid-sulfate extension of the SLE solver: H+ / SO4-- as stoichiometric ions with internal HSO4- speciation,
NH4HSO4 and letovicite as solids.

Reference values: the bisulfate speciation is compared with the (Fortran-validated) ``dissociation.solve_bisulfate``;
the acid-solid K are the thermodynamic values of Clegg et al. (1998) (J. Phys. Chem. A 102:2137/2155; 2155 for Na),
converted from mole-fraction/free-ion to molal; the DRH implied by them (``tools/calibrate_acid_solids.py``) is an
independent check against the literature (NH4HSO4 ~40 %, letovicite ~69.5 %).
"""
import math

import numpy as np
import pytest

pytest.importorskip("scipy")

from aiomfac_py.dissociation import solve_bisulfate  # noqa: E402
from aiomfac_py.sle import AqueousIons, SLESolver, feed_from_salts  # noqa: E402
from aiomfac_py.solids import SOLIDS  # noqa: E402

T0 = 298.15


@pytest.mark.parametrize("m", [(1.0, 1.0, 1.0), (2.0, 0.1, 1.05), (3.0, 1.0, 2.0), (30.0, 10.0, 20.0),
                               (0.01, 0.01, 0.01)])
def test_bisulfate_speciation_matches_dissociation_module(m):
    aq = AqueousIons(["NH4+", "H+", "SO4--"])
    m = np.array(m, dtype=float)
    smc = np.zeros(aq._ngi); sma = np.zeros(aq._ngi)
    for (is_cat, idx), v in zip(aq._pos, m):
        (smc if is_cat else sma)[idx] = v
    ref = solve_bisulfate(aq.model, T0, aq._xn, smc.copy(), sma.copy(), aq._ih, aq._ihs, aq._iso)
    aq.ln_gamma_aw(m, T0)
    assert aq._x_last == pytest.approx(ref.m_hso4, rel=1e-7)


def test_hso4_alias_in_ions_and_feed():
    sol = SLESolver(["NH4+", "HSO4-", "SO4--"])
    assert sol.ions == ["NH4+", "H+", "SO4--"]
    r1 = sol.solve({"NH4+": 1.0, "HSO4-": 1.0}, T0, 0.6)
    r2 = sol.solve(feed_from_salts({"NH4HSO4": 1.0}), T0, 0.6)
    assert r1.water_kg == pytest.approx(r2.water_kg, rel=1e-9)
    with pytest.raises(ValueError):
        AqueousIons(["NH4+", "HSO4-", "SO4--"])


@pytest.mark.parametrize("key", ["ammonium_bisulfate", "letovicite"])
def test_acid_solid_database(key):
    s = SOLIDS[key]
    assert s.charge_balance == 0 and math.isfinite(s.ln_k(T0)) and s.quality == "B"


@pytest.mark.parametrize("rh", [0.45, 0.60, 0.75])
def test_acid_feed_kkt_and_balances(rh):
    sol = SLESolver(["NH4+", "H+", "SO4--"])
    feed = feed_from_salts({"(NH4)2SO4": 1.0, "H2SO4": 0.5})
    r = sol.solve(feed, T0, rh)
    assert r.status in ("aqueous", "solid+aqueous")
    n_tot = dict(r.aq_ions)
    for key, n in r.solids.items():
        for ion, nu in SOLIDS[key].ions.items():
            n_tot[ion] = n_tot.get(ion, 0.0) + nu * n
    for ion, b in feed.items():
        assert n_tot[ion] == pytest.approx(b, rel=1e-7, abs=1e-9)
    assert all(v <= 1e-7 for v in r.si.values())                  # dual feasibility
    for key in r.solids:
        assert r.si[key] == pytest.approx(0.0, abs=1e-6)          # complementarity
    # water activity of the speciated aqueous phase equals RH
    aq = AqueousIons(["NH4+", "H+", "SO4--"])
    m = np.array([r.molality[i] for i in aq.ions])
    assert aq.ln_gamma_aw(m, T0)[1] == pytest.approx(math.log(rh), abs=1e-7)


def test_pure_acid_salt_deliquescence_near_literature():
    """Clegg-K-implied DRH (independent of any DRH data) lies within ~3 RH points of the literature values."""
    sol = SLESolver(["NH4+", "H+", "SO4--"])
    feed = feed_from_salts({"NH4HSO4": 1.0})
    assert sol.solve(feed, T0, 0.35).status == "dry"          # implied 37.9 %, literature ~40 %
    assert sol.solve(feed, T0, 0.43).status != "dry"
    let = feed_from_salts({"(NH4)3H(SO4)2": 1.0})
    assert sol.solve(let, T0, 0.67).solids.get("letovicite", 0.0) == pytest.approx(1.0, abs=1e-9)
    assert sol.solve(let, T0, 0.73).solids.get("letovicite", 0.0) == pytest.approx(0.0, abs=1e-9)


def test_clegg_mole_fraction_to_molal_conversion():
    """ln K_m = ln xK + n_ions ln(1000/M_w) for dissolution into free ions"""
    L = math.log(1000.0 / 18.01528)
    assert SOLIDS["ammonium_bisulfate"].ln_k(T0) == pytest.approx(-11.408 + 3 * L, abs=1e-3)
    assert SOLIDS["letovicite"].ln_k(T0) == pytest.approx(-26.007 + 6 * L, abs=1e-3)


def test_sodium_acid_solids_present_and_ordered():
    for k in ("sodium_bisulfate", "sodium_bisulfate_hydrate", "trisodium_hydrogen_sulfate", "NaH3_SO4_2_hydrate"):
        s = SOLIDS[k]
        assert s.charge_balance == 0 and math.isfinite(s.ln_k(T0))
    sol = SLESolver(["Na+", "H+", "SO4--"])
    r = sol.solve({"Na+": 1.0, "H+": 1.0, "SO4--": 1.0}, T0, 0.30)     # NaHSO4 composition, dry
    assert r.status == "dry" and r.solids
    r = sol.solve({"Na+": 1.0, "H+": 1.0, "SO4--": 1.0}, T0, 0.90)
    assert r.status == "aqueous"


def test_acid_ammonium_sulfate_mixture_phase_sequence():
    """More acid at a fixed low RH moves the crystalline assemblage AS+LET -> AHS+LET -> AHS -> acid-rich liquid."""
    sol = SLESolver(["NH4+", "H+", "SO4--"])
    seq = []
    for f_h in (0.2, 0.6, 1.0, 1.8):
        r = sol.solve({"SO4--": 1.0, "H+": f_h, "NH4+": 2.0 - f_h}, T0, 0.30)
        seq.append((r.status, tuple(sorted(r.solids))))
    assert seq[0] == ("dry", ("ammonium_sulfate", "letovicite"))
    assert seq[1] == ("dry", ("ammonium_bisulfate", "letovicite"))
    assert seq[2] == ("dry", ("ammonium_bisulfate",))
    assert seq[3] == ("aqueous", ())


def test_pure_sulfuric_acid_solution_has_no_solids():
    sol = SLESolver(["NH4+", "H+", "SO4--"])
    r = sol.solve({"SO4--": 1.0, "H+": 2.0}, T0, 0.60)
    assert r.status == "aqueous" and not r.solids
    wt = 98.079 / (98.079 + 1000.0 * r.water_kg)          # mass fraction of H2SO4 at a_w = 0.6 (about 38-40 %)
    assert 0.35 < wt < 0.43


def test_neutral_systems_unchanged_by_acid_support():
    sol = SLESolver(["Na+", "NH4+", "Cl-", "SO4--"])
    r = sol.solve(feed_from_salts({"NaCl": 1.0, "(NH4)2SO4": 1.0}), T0, 0.70)
    assert r.status == "aqueous" and r.water_kg == pytest.approx(0.2197, abs=2e-4)

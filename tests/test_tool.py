"""Tests of the main tool (aiomfac_py.tool): every calculation mode gives the result of the solver it dispatches to,
the case input (mass and salt feeds, gases, solver options) is converted correctly, invalid cases are rejected with
a clear message, and the command line writes the requested formats."""
import json
import math
import sys

import numpy as np
import pytest

pytest.importorskip("scipy")

from aiomfac_py import ActivityModel, Component  # noqa: E402
from aiomfac_py import tool  # noqa: E402
from aiomfac_py.tool import Case, CaseError, run  # noqa: E402

PINIC_SG = [[1, 2], [2, 2], [3, 2], [4, 1], [137, 2]]
PINIC = Component(2, "pinic_acid", tuple(tuple(t) for t in PINIC_SG))


def _case(mode, **kw):
    d = {"calculation": {"mode": mode}, "system": {"T_K": 298.15, "organics": [{"name": "pinic_acid",
                                                                                 "subgroups": PINIC_SG}]}}
    for k, v in kw.items():
        if k in ("rh",):
            d["calculation"][k] = v
        elif k == "T_K":
            d["system"]["T_K"] = v
        elif k == "no_organics":
            d["system"].pop("organics")
        else:
            d[k] = v
    return d


FEED_AS = {"pinic_acid": 0.006243, "NH4+": 0.015136, "SO4--": 0.007568}


@pytest.mark.skipif(sys.version_info < (3, 11), reason="tomllib")
@pytest.mark.parametrize("name", sorted(tool.TEMPLATES))
def test_templates_parse(name):
    import tomllib
    case = Case.from_dict(tomllib.loads(tool.TEMPLATES[name]))
    assert case.mode in tool.MODES and case.engine in tool.MODES[case.mode]


def test_equilibrium_matches_phase_equilibrium():
    from aiomfac_py.phase_equilibrium import PhaseEquilibrium
    feed = {"pinic_acid": 1.15 / 184.19, "NH4+": 2 * 0.778 / 132.14 + 0.222 / 80.04, "SO4--": 0.778 / 132.14,
            "NO3-": 0.222 / 80.04}
    res = run(_case("equilibrium", rh=[0.6], feed=feed, T_K=300.0, solver={"path": "independent"}))
    ref = PhaseEquilibrium([PINIC], ["NH4+", "SO4--", "NO3-"], T_K=300.0).solve(feed, 0.6)
    p = res[0]
    assert p.status == "converged" and p.n_liquids == ref.n_liquids == 2
    assert set(p.solids) == set(ref.solids) == {"ammonium_sulfate"}
    assert p.gibbs == pytest.approx(ref.gibbs, abs=1e-12)
    assert p.liquids[0]["water_activity"] == pytest.approx(0.6, abs=1e-8)


def test_mass_and_salt_feeds_are_converted_to_mol():
    c1 = Case.from_dict(_case("equilibrium", rh=0.6, feed_mass_g={"pinic_acid": 1.15},
                              salts_mass_g={"(NH4)2SO4": 0.778, "NH4NO3": 0.222}))
    m_pin = tool.organic_molar_mass_g(PINIC)
    assert m_pin == pytest.approx(186.2, abs=0.5)                       # C9H14O4
    assert c1.feed["pinic_acid"] == pytest.approx(1.15 / m_pin)
    m_as, m_an = tool.salt_molar_mass_g("(NH4)2SO4"), tool.salt_molar_mass_g("NH4NO3")
    assert m_as == pytest.approx(132.14, abs=0.02) and m_an == pytest.approx(80.04, abs=0.02)
    assert c1.feed["NH4+"] == pytest.approx(2 * 0.778 / m_as + 0.222 / m_an)
    assert c1.feed["SO4--"] == pytest.approx(0.778 / m_as)
    assert c1.ions == ["NH4+", "SO4--", "NO3-"]


def test_bisulfate_bicarbonate_hydroxide_feeds_become_stoichiometric_ions():
    c = Case.from_dict(_case("sle", rh=0.8, feed={"NH4+": 1.0, "HSO4-": 1.0}, no_organics=True))
    assert c.feed == {"NH4+": 1.0, "H+": 1.0, "SO4--": 1.0} and c.ions == ["NH4+", "H+", "SO4--"]
    c = Case.from_dict(_case("sle", rh=0.8, feed={"Na+": 1.0, "HCO3-": 0.5, "OH-": 0.5}, no_organics=True))
    assert c.feed == {"Na+": 1.0, "H+": 0.0, "CO3--": 0.5} and set(c.ions) == {"Na+", "H+", "CO3--"}


def test_lle_engines():
    from aiomfac_py.lle import solve_pep
    r = run(_case("lle", rh=0.3, feed=FEED_AS))[0]
    assert r.status == "converged" and r.n_liquids == 2 and not r.solids
    d = _case("lle", feed={"Water": 0.6, "pinic_acid": 0.3}, salts={"(NH4)2SO4": 0.1}, solver={"engine": "pep"})
    p = run(d)[0]
    AS = Component(3, "(NH4)2SO4", ((204, 2), (261, 1)))
    ref = solve_pep([Component(1, "Water", ((16, 1),)), PINIC, AS], np.array([0.6, 0.3, 0.1]), 298.15)
    assert p.status == "converged" and p.n_liquids == 2
    xs = sorted([L["mole_fractions"]["pinic_acid"] for L in p.liquids])
    assert xs == pytest.approx(sorted(ref.x[:, 1][ref.y > 1e-3]), abs=1e-10)
    assert p.liquids[0]["water_activity"] == pytest.approx(p.liquids[1]["water_activity"], abs=1e-5)


def test_sle_engines_agree():
    base = _case("sle", rh=[0.7, 0.5], salts={"NaCl": 1.0, "(NH4)2SO4": 1.0}, no_organics=True)
    a = run(base)
    b = run({**base, "solver": {"engine": "pe"}})
    # aqueous at 0.7: same water content
    assert a[0].n_liquids == b[0].n_liquids == 1
    assert a[0].liquids[0]["amounts"]["Water"] == pytest.approx(b[0].liquids[0]["amounts"]["Water"], rel=1e-5)
    # dry at 0.5 (SLESolver) = all salt in solids (PhaseEquilibrium leaves a vanishing liquid)
    assert a[1].status == "dry" and a[1].n_liquids == 0
    assert set(a[1].solids) == set(b[1].solids)
    for k in a[1].solids:
        assert b[1].solids[k] == pytest.approx(a[1].solids[k], rel=1e-5)


def test_sle_metastable_aqueous_solution():
    from aiomfac_py.sle import SLESolver
    base = _case("sle", rh=[0.5], salts={"(NH4)2SO4": 1.0}, no_organics=True, solver={"solids": "none"})
    p = run(base)[0]
    ref = SLESolver(["NH4+", "SO4--"]).aqueous_state({"NH4+": 2.0, "SO4--": 1.0}, 298.15, 0.5)
    assert p.status == "converged" and not p.solids and p.n_liquids == 1
    assert p.si["ammonium_sulfate"] == pytest.approx(ref.si["ammonium_sulfate"], rel=1e-12) and p.si["ammonium_sulfate"] > 0
    q = run({**base, "solver": {"solids": "none", "engine": "pe"}})[0]
    assert q.liquids[0]["amounts"]["Water"] == pytest.approx(p.liquids[0]["amounts"]["Water"], rel=1e-5)


def test_examples_are_the_templates():
    from pathlib import Path
    d = Path(__file__).resolve().parents[1] / "examples" / "cases"
    for name, text in tool.TEMPLATES.items():
        assert (d / f"{name}.toml").read_text() == text, name


def test_open_gas_equilibrium_matches_direct_solve():
    from aiomfac_py.s2as import smiles_to_components
    try:
        dlt = smiles_to_components(["CCOC(=O)C(O)C(O)C(=O)OCC"], names=["DLT"]).components[1]
    except ImportError:
        pytest.skip("S2AS (epam.indigo) not installed")
    from aiomfac_py.phase_equilibrium import PhaseEquilibrium
    m, r = 1 / 58.44, 0.75
    feed = {"DLT": 3 / 206.19, "Na+": m, "Cl-": m, "H+": 2 * r * m, "SO4--": r * m}
    d = {"calculation": {"mode": "lle", "rh": 0.2},
         "system": {"T_K": 298.15, "organics": [{"name": "DLT", "subgroups": [list(t) for t in dlt.subgroups]}]},
         "feed": feed, "gas": {"mode": "open", "partial_pressure_atm": {"HCl": 1e-9}}}
    p = run(d)[0]
    ref = PhaseEquilibrium([dlt], ["Na+", "H+", "Cl-", "SO4--"], T_K=298.15).solve(feed, 0.2, solids="none",
                                                                                   p_gas={"HCl": 1e-9})
    assert p.status == "converged" and p.n_liquids == ref.n_liquids
    assert p.gibbs == pytest.approx(ref.gibbs, abs=1e-12)
    assert p.gas["HCl"] == pytest.approx(ref.gas["HCl"], rel=1e-9)


def test_solver_options_reach_phase_equilibrium():
    c = Case.from_dict(_case("lle", rh=0.3, feed=FEED_AS, solver={"inner_method": "rand", "tpd_method": "newton",
                                                                  "speciation": "explicit", "max_liquids": 2}))
    pe = c.phase_equilibrium()
    assert pe.inner_method == "rand" and pe.tpd_method == "newton"
    a = run(c)[0]
    b = run(_case("lle", rh=0.3, feed=FEED_AS))[0]
    assert a.n_liquids == b.n_liquids and a.gibbs == pytest.approx(b.gibbs, abs=1e-10)


def test_activity_mode_matches_activity_model():
    p = run(_case("activity", feed={"Water": 5.0, "pinic_acid": 0.3}, salts={"(NH4)2SO4": 0.1}))[0]
    ev = ActivityModel([Component(1, "Water", ((16, 1),)), PINIC, Component(3, "AS", ((204, 2), (261, 1)))]).evaluate(
        np.array([5.0, 0.3, 0.1]) / 5.4, 298.15, basis="mole")
    L = p.liquids[0]
    assert L["activity"]["Water"] == pytest.approx(ev.activity[0], rel=1e-12)
    assert L["activity"]["pinic_acid"] == pytest.approx(ev.activity[1], rel=1e-12)


def test_gas_particle_matches_gp_partition():
    from aiomfac_py.gp_partition import VolatileSpecies, gp_partition
    d = {"calculation": {"mode": "gas_particle", "rh": [0.6]}, "system": {"T_K": 298.15},
         "gas_particle": {"salt": "(NH4)2SO4", "n_salt_mol": 1e-8, "V_gas_m3": 1.0,
                          "species": [{"name": "glycerol", "subgroups": [[150, 2], [151, 1], [153, 3]],
                                       "M_kg_mol": 0.092094, "p0_Pa": 2.284e-2, "n_total_mol": 3e-8}]},
         "solver": {"gp_tol": 1e-4}}
    p = run(d)[0]
    gly = Component(2, "glycerol", ((150, 2), (151, 1), (153, 3)))
    ref = gp_partition(Component(3, "(NH4)2SO4", ((204, 2), (261, 1))),
                       [VolatileSpecies(gly, 0.092094, 2.284e-2, 3e-8)], 1e-8, 298.15, 0.6, 1.0, tol=1e-4)
    assert p.values["glycerol_particle_mol"] == pytest.approx(ref.n_org_PM[0], rel=1e-12)
    assert p.values["water_activity"] == pytest.approx(0.6, abs=0.02)


def test_drying_path_matches_phase_equilibrium():
    from aiomfac_py.phase_equilibrium import PhaseEquilibrium
    from aiomfac_py.sle import implied_ln_s_crit
    feed = {"pinic_acid": 1.15 / 184.19, "NH4+": 2 * 0.778 / 132.14 + 0.222 / 80.04, "SO4--": 0.778 / 132.14,
            "NO3-": 0.222 / 80.04}
    grid = [0.8, 0.5, 0.3, 0.2]
    res = run(_case("drying_path", rh=grid, feed=feed, T_K=300.0,
                    drying={"erh": {"ammonium_sulfate": 0.35}, "never": ["ammonium_nitrate"]}))
    lsc = {"ammonium_sulfate": implied_ln_s_crit("ammonium_sulfate", 300.0, 0.35), "ammonium_nitrate": 99.0}
    ref = PhaseEquilibrium([PINIC], ["NH4+", "SO4--", "NO3-"], T_K=300.0).drying_path(feed, grid, ln_s_crit=lsc)
    for p, r in zip(res, ref):
        assert p.n_liquids == r.n_liquids and set(p.solids) == set(r.solids)
        assert p.gibbs == pytest.approx(r.gibbs, abs=1e-12)


def test_deliquescence_and_efflorescence():
    d = _case("deliquescence", salts={"(NH4)2SO4": 1.0}, no_organics=True)
    p = run(d)[0]
    assert p.values["drh"] == pytest.approx(0.80, abs=0.01)
    e = run(_case("efflorescence", salts={"(NH4)2SO4": 1.0}, no_organics=True,
                  efflorescence={"erh": {"ammonium_sulfate": 0.35}}))[0]
    assert e.values["salt"] == "ammonium_sulfate" and e.values["erh"] == pytest.approx(0.35, abs=1e-3)


def test_stability_mode_flags_unstable_one_liquid_states():
    res = run(_case("stability", rh=[0.995, 0.95], feed=FEED_AS))
    assert [p.values["stable"] for p in res] == [True, False]
    assert all(p.status == "converged" and p.n_liquids == 1 for p in res)
    assert res[1].tpd_min < -0.1


@pytest.mark.parametrize("bad,msg", [
    ({"calculation": {"mode": "magic"}}, "unknown mode"),
    ({"solver": {"engine": "sle"}}, "inorganic systems only"),
    ({"solver": {"engine": "gp"}}, "cannot solve mode"),
    ({"solver": {"inner": "rand"}}, "unknown [solver] option"),
    ({"calculation": {"mode": "equilibrium"}}, "needs [calculation] rh"),
    ({"feed": {"pinic_acid": 1.0, "Xx+": 1.0}}, "neither an organic"),
])
def test_invalid_cases_are_rejected(bad, msg):
    d = _case("equilibrium", rh=0.5, feed=dict(FEED_AS))
    for k, v in bad.items():
        if k == "calculation":
            d["calculation"] = v
        else:
            d[k] = v
    with pytest.raises(CaseError, match=msg.replace("[", r"\[").replace("]", r"\]")):
        Case.from_dict(d)


def test_outputs_and_command_line(tmp_path, capsys):
    d = _case("lle", rh=[0.4, 0.3], feed=FEED_AS)
    case_file = tmp_path / "case.json"
    case_file.write_text(json.dumps(d))
    out = tmp_path / "out.json"
    assert tool.main(["run", str(case_file), "-o", str(out)]) == 0
    data = json.loads(out.read_text())
    assert data["mode"] == "lle" and len(data["points"]) == 2
    assert data["points"][1]["n_liquids"] == 2
    csv_out = tmp_path / "out.csv"
    assert tool.main(["run", str(case_file), "-o", str(csv_out)]) == 0
    lines = csv_out.read_text().splitlines()
    assert lines[0].startswith("mode,engine,rh,status") and len(lines) == 3
    assert tool.main(["run", str(case_file)]) == 0
    assert "liquid 1" in capsys.readouterr().out
    assert tool.main(["template", "sle"]) == 0
    assert "mode = \"sle\"" in capsys.readouterr().out
    assert tool.main(["list", "modes"]) == 0
    assert "gas_particle" in capsys.readouterr().out
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"calculation": {"mode": "x"}, "system": {"T_K": 298.15}}))
    assert tool.main(["run", str(bad)]) == 1


def test_json_output_has_no_nan():
    p = tool.PointResult("x", "y", None, "converged", values={"a": float("nan"), "b": np.float64(1.5),
                                                              "c": np.array([1.0, math.inf])})
    d = p.to_dict()
    assert d["values"] == {"a": None, "b": 1.5, "c": [1.0, None]}
    json.dumps(d, allow_nan=False)

"""Cost benchmark of PhaseEquilibrium: wall time and number of AIOMFAC activity evaluations (LiquidModel.ln_a calls,
children included) for representative cases.  Usage: python benchmarks/pe_bench.py [case ...]"""
import sys
import time

from aiomfac_py import Component
from aiomfac_py import phase_equilibrium as pe_mod
from aiomfac_py.phase_equilibrium import PhaseEquilibrium

PINIC = Component(2, "pinic_acid", ((1, 2), (2, 2), (3, 2), (4, 1), (137, 2)))
COUNT = {"n": 0}
_orig = pe_mod.LiquidModel.ln_a


def _counted(self, n, T):
    COUNT["n"] += 1
    return _orig(self, n, T)


pe_mod.LiquidModel.ln_a = _counted
if hasattr(pe_mod, "ExplicitLiquidModel"):
    _orig_e = pe_mod.ExplicitLiquidModel.ln_a

    def _counted_e(self, n, T):
        COUNT["n"] += 1
        return _orig_e(self, n, T)

    pe_mod.ExplicitLiquidModel.ln_a = _counted_e


def _dlt():
    from aiomfac_py.s2as import smiles_to_components
    return smiles_to_components(["CCOC(=O)C(O)C(O)C(=O)OCC"], names=["DLT"]).components[1]


def case_llps_as():
    """pinic acid + AS, RH 0.30, metastable (two liquids)"""
    m_org, m_as = 1.15 / 184.19, 1.0 / 132.14
    return PhaseEquilibrium([PINIC], ["NH4+", "SO4--"], T_K=298.15).solve(
        {"pinic_acid": m_org, "NH4+": 2 * m_as, "SO4--": m_as}, 0.30, solids="none")


def case_llps_an_solid():
    """pinic acid + AS + AN, RH 0.6 (two liquids + AS(s))"""
    m_org, m_as, m_an = 1.15 / 184.19, 0.778 / 132.14, 0.222 / 80.04
    return PhaseEquilibrium([PINIC], ["NH4+", "SO4--", "NO3-"], T_K=298.15).solve(
        {"pinic_acid": m_org, "NH4+": 2 * m_as + m_an, "SO4--": m_as, "NO3-": m_an}, 0.6)


def case_dlt_hcl():
    """DLT + NaCl + H2SO4 (r 0.75), open to 1e-9 atm HCl, RH 0.2"""
    m, r = 1 / 58.44, 0.75
    return PhaseEquilibrium([_dlt()], ["Na+", "H+", "Cl-", "SO4--"], T_K=298.15).solve(
        {"DLT": 3 / 206.19, "Na+": m, "Cl-": m, "H+": 2 * r * m, "SO4--": r * m}, 0.2, solids="none",
        p_gas={"HCl": 1e-9})


def case_dlt_3liq():
    """DLT + NaCl + H2SO4 (r 1.5), open to 1e-9 atm HCl, RH 0.5 (three liquids after a smaller seed; as in
    test_new_liquid_close_to_its_appearance_is_found_with_a_smaller_seed)"""
    m, r = 1 / 58.44, 1.5
    return PhaseEquilibrium([_dlt()], ["Na+", "H+", "Cl-", "SO4--"], T_K=298.15).solve(
        {"DLT": 3 / 206.19, "Na+": m, "Cl-": m, "H+": 2 * r * m, "SO4--": r * m}, 0.5, solids="none",
        p_gas={"HCl": 1e-9})


def case_carbonate_closed():
    """NaCl + base, closed CO2 (inorganic carbonate)"""
    n_air, y = 41.0, 4.2e-4
    return PhaseEquilibrium([], ["Na+", "Cl-", "CO3--", "H+"], T_K=298.15).solve(
        {"Na+": 1.1e-5, "Cl-": 1e-5, "H+": -1e-6}, 0.8, gas_total={"CO2": y * n_air / (1 - y)}, n_air=n_air)


def case_org_carbonate():
    """pinic acid + NaCl + base, open to 420 ppm CO2, RH 0.5 (two liquids; carbonate traces in the organic liquid)"""
    return PhaseEquilibrium([PINIC], ["Na+", "Cl-", "CO3--", "H+"], T_K=298.15).solve(
        {"pinic_acid": 3 / 184.19, "Na+": 1 / 58.44 + 2e-3, "Cl-": 1 / 58.44, "H+": -2e-3}, 0.5, solids="none",
        p_gas={"CO2": 4.2e-4})


# PE_HESS=ad|split|central selects the Hessian scheme of every PhaseEquilibrium created by the cases
import os  # noqa: E402
if os.environ.get("PE_HESS"):
    _init = PhaseEquilibrium.__init__

    def _init_h(self, *a, **k):
        _init(self, *a, **k)
        self.hess_scheme = os.environ["PE_HESS"]

    PhaseEquilibrium.__init__ = _init_h
if hasattr(pe_mod.ExplicitLiquidModel, "hessian_ad"):
    _orig_j = pe_mod.ExplicitLiquidModel.hessian_ad

    def _counted_j(self, *a, **k):
        COUNT["jac"] = COUNT.get("jac", 0) + 1
        return _orig_j(self, *a, **k)

    pe_mod.ExplicitLiquidModel.hessian_ad = _counted_j

_tpd = PhaseEquilibrium._tpd_minimize


def _tpd_counted(self, *a, **k):
    n0 = COUNT["n"]
    try:
        return _tpd(self, *a, **k)
    finally:
        COUNT["tpd"] = COUNT.get("tpd", 0) + COUNT["n"] - n0


PhaseEquilibrium._tpd_minimize = _tpd_counted

CASES = {k[5:]: v for k, v in list(globals().items()) if k.startswith("case_")}

if __name__ == "__main__":
    for name in sys.argv[1:] or list(CASES):
        COUNT["n"] = 0; COUNT["tpd"] = 0; COUNT["jac"] = 0
        t0 = time.perf_counter()
        r = CASES[name]()
        dt = time.perf_counter() - t0
        print(f"{name:18s} {r.status:14s} liq={r.n_liquids} solids={sorted(r.solids)} F={r.gibbs:.12e} "
              f"tpd={r.tpd_min:.2e} evals={COUNT['n']:7d} (stability test {COUNT['tpd']:6d}) jac={COUNT['jac']:5d} time={dt:7.1f}s", flush=True)

"""Liquid models defined by a Gibbs-energy function (aiomfac_py.gibbs_model), checked against closed forms:
a regular solution of water + one organic, and an ideal (molal) electrolyte solution."""
import math

import numpy as np
import pytest

jax = pytest.importorskip("jax")
pytest.importorskip("scipy")

import jax.numpy as jnp  # noqa: E402

from aiomfac_py import Component  # noqa: E402
from aiomfac_py.gibbs_model import GibbsFunction, GibbsLiquidModel, ideal_mixing, xlogy_safe  # noqa: E402
from aiomfac_py.phase_equilibrium import PhaseEquilibrium  # noqa: E402

ORG = Component(2, "org", ((1, 2), (2, 4), (3, 1), (14, 1)))
MW = 0.01801528
T = 298.15


def regular(A=3.0):
    """g = sum n ln x + A n_w n_o / n_T  ->  ln gamma_w = A x_o^2, ln gamma_o = A x_w^2."""
    return GibbsFunction(["Water", "org"], lambda n, k: ideal_mixing(n) + k["A"] * n[0] * n[1] / jnp.sum(n),
                         consts=lambda T_: {"A": A}, label="regular")


def ideal_electrolyte(names):
    """ideal molal solution: ln a_w = ln x_w, ln a_i = ln m_i + ln x_w (mole-fraction ideal mixing of water and ions)."""
    ions = np.array([nm != "Water" for nm in names], dtype=float)
    return GibbsFunction(names, lambda n, k: ideal_mixing(n) - math.log(MW) * jnp.sum(n * ions), label="ideal")


def test_activities_hessian_and_hvp_of_a_regular_solution():
    lm = GibbsLiquidModel([ORG], [], regular(2.0))
    n = np.array([0.7, 0.3])
    x = n / n.sum()
    la = lm.ln_a(n, T)
    np.testing.assert_allclose(la, np.log(x) + 2.0 * x[::-1] ** 2, rtol=0, atol=1e-13)
    H = lm.hessian_ad(n, T)
    h = 1e-6
    fd = np.column_stack([(lm.ln_a(n + h * e, T) - lm.ln_a(n - h * e, T)) / (2 * h) for e in np.eye(2)])
    np.testing.assert_allclose(H, fd, rtol=1e-7, atol=1e-8)
    np.testing.assert_allclose(H @ n, 0.0, atol=1e-12)                       # Gibbs-Duhem (homogeneity)
    v = np.array([0.3, -1.1])
    np.testing.assert_allclose(lm.hvp(n, T, v), H @ v, rtol=1e-12, atol=1e-14)
    assert lm.gibbs_value(n, T) == pytest.approx(float(n @ la), rel=1e-13)   # degree-1 homogeneity: g = n . mu


def test_absent_species_have_minus_infinite_potential_and_finite_others():
    lm = GibbsLiquidModel([ORG], [], regular(2.0))
    la = lm.ln_a(np.array([1.0, 0.0]), T)
    assert la[1] == -math.inf and la[0] == pytest.approx(0.0, abs=1e-14)
    assert np.all(np.isfinite(lm.hessian_ad(np.array([1.0, 0.0]), T, np.array([True, False]))))
    assert float(xlogy_safe(jnp.array(0.0), jnp.array(0.0))) == 0.0


def test_compiled_functions_are_shared_between_models():
    gf = regular(2.0)
    a, b = GibbsLiquidModel([ORG], [], gf), GibbsLiquidModel([ORG], [], gf)
    a.ln_a(np.array([1.0, 1.0]), T)
    b.ln_a(np.array([1.0, 2.0]), 300.0)                                      # other T: same compiled function
    assert len(gf._compiled) == 1 and len(gf._kcache) == 2


def test_regular_solution_binodal_in_the_phase_equilibrium_solver():
    """symmetric regular solution, A = 3: binodal x_o' solves ln((1-x)/x) = A(1-2x); two liquids only at
    a_w* = (1 - x') exp(A x'^2); below a_w* the organic-rich branch, above it the water-rich one."""
    from scipy.optimize import brentq
    A = 3.0
    x1 = brentq(lambda x: math.log((1 - x) / x) - A * (1 - 2 * x), 1e-6, 0.4)
    aw_star = (1 - x1) * math.exp(A * x1 ** 2)
    lm = GibbsLiquidModel([ORG], [], regular(A))
    pe = PhaseEquilibrium([ORG], [], T, liquid_model=lm)
    assert pe.hess_scheme == "ad" and pe._use_ad()
    lo = pe.solve({"org": 1.0}, aw_star - 2e-4, solids="none")
    hi = pe.solve({"org": 1.0}, aw_star + 2e-4, solids="none")
    assert lo.status == hi.status == "converged"
    assert lo.n_liquids == hi.n_liquids == 1
    assert lo.liquids[0].mole_fractions[1] > 1 - x1 - 0.02                 # organic-rich branch
    assert hi.liquids[0].mole_fractions[1] < x1 + 0.02                     # water-rich branch
    for r, rh in ((lo, aw_star - 2e-4), (hi, aw_star + 2e-4)):
        xo = r.liquids[0].mole_fractions[1]
        assert math.log(1 - xo) + A * xo ** 2 == pytest.approx(math.log(rh), abs=1e-7)
    assert lm.n_jac > 0                                                    # exact Hessians were used


def test_ideal_electrolyte_water_content_and_child_problem():
    """ideal molal solution of NaCl: a_w = x_w (ion basis) = RH gives n_w = RH/(1-RH) * (n_Na + n_Cl); the model's
    third ion is absent from the feed, so the solver works on a child problem with the restricted model."""
    names = ["Water", "Na+", "Cl-", "NO3-"]
    lm = GibbsLiquidModel([], ["Na+", "Cl-", "NO3-"], ideal_electrolyte(names))
    pe = PhaseEquilibrium([], ["Na+", "Cl-", "NO3-"], T, liquid_model=lm, solid_keys=[])
    for rh in (0.9, 0.6):
        r = pe.solve({"Na+": 0.5, "Cl-": 0.5}, rh, solids="none")
        assert r.status == "converged", r.message
        nw = r.liquids[0].amounts[0]
        assert nw == pytest.approx(rh / (1 - rh) * 1.0, rel=1e-8)
    child = next(iter(pe._children.values()))
    assert isinstance(child.lm, GibbsLiquidModel) and child.lm.gibbs is lm.gibbs


def test_liquid_model_must_match_the_system():
    lm = GibbsLiquidModel([ORG], [], regular())
    with pytest.raises(ValueError):
        PhaseEquilibrium([], ["Na+", "Cl-"], T, liquid_model=lm)
    with pytest.raises(ValueError):
        GibbsLiquidModel([], ["Na+", "Cl-"], regular())                      # Na+, Cl- not variables of g


def test_compiled_functions_shared_between_gibbs_functions_with_a_common_cache():
    """two 'molecules' with the same g and different constants share one compiled function."""
    shared = {}
    g = lambda n, k: ideal_mixing(n) + k["A"] * n[0] * n[1] / jnp.sum(n)
    ga = GibbsFunction(["Water", "org"], g, consts=lambda T_: {"A": 1.0}, compiled_cache=shared)
    gb = GibbsFunction(["Water", "org"], g, consts=lambda T_: {"A": 2.0}, compiled_cache=shared)
    n = np.array([0.6, 0.4])
    la = GibbsLiquidModel([ORG], [], ga).ln_a(n, T)
    lb = GibbsLiquidModel([ORG], [], gb).ln_a(n, T)
    assert len(shared) == 1
    assert lb[0] - la[0] == pytest.approx(0.4 ** 2, rel=1e-12)                  # A x_o^2 with A = 2 vs 1

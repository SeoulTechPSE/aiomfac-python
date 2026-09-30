"""Validation of the primal-dual interior-point / active-set LLE engine (lle.py) against an independently
solvable toy system -- NOT against AIOMFAC/Fortran (no such LLE reference exists; see lle.py's module
docstring). The toy system isolates the interior-point/active-set machinery itself from the AIOMFAC activity
model, so a correct result here means the *algorithm* (Newton system, barrier continuation, active-set pruning,
phase-simplex initialization) is implemented correctly, independent of AIOMFAC.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.optimize import brentq

from aiomfac_py.lle import solve_pep_gfe


class RegularSolutionGFE:
    """Symmetric binary regular solution: g(x) = x1 ln x1 + x2 ln x2 + Omega * x1 * x2 (RT units).

    Exactly degree-1 homogeneous by construction (x1, x2 need not sum to 1; the model is evaluated on the
    normalized ratio), analytic gradient/Hessian used directly (no finite differences -- this is a clean,
    independent check of the solver, not of AiomfacGFE's finite-difference machinery).
    """

    n_species = 2

    def __init__(self, omega: float):
        self.omega = float(omega)

    def g(self, x):
        x = np.asarray(x, dtype=float)
        xhat = x / x.sum()
        x1, x2 = xhat
        return float(x.sum() * (x1 * math.log(x1) + x2 * math.log(x2) + self.omega * x1 * x2))

    def grad(self, x):
        x = np.asarray(x, dtype=float)
        xhat = x / x.sum()
        x1, x2 = xhat
        # d/dx_i of the homogeneous-degree-1 extension = ln(x_i_hat) + 1 + Omega * x_j_hat^2 - Omega*x1*x2
        # (standard regular-solution chemical potential; verified against finite differences below too)
        mu1 = math.log(x1) + self.omega * x2 * x2
        mu2 = math.log(x2) + self.omega * x1 * x1
        return np.array([mu1, mu2])

    def hess(self, x):
        x = np.asarray(x, dtype=float)
        xhat = x / x.sum()
        x1, x2 = xhat
        tot = x.sum()
        # Hessian of the degree-1-homogeneous extension, satisfies H @ x = 0 (Gibbs-Duhem) by construction
        H = np.zeros((2, 2))
        H[0, 0] = (1.0 / x1 - 2.0 * self.omega * x2) / tot
        H[1, 1] = (1.0 / x2 - 2.0 * self.omega * x1) / tot
        # Gibbs-Duhem: H @ [x1,x2]*tot = 0 => x1*H00 + x2*H01 = 0 => H01 = -x1*H00/x2
        H[0, 1] = H[1, 0] = -x1 * H[0, 0] / x2
        return H


def _analytic_binodal(omega: float) -> float:
    """alpha in (0, 0.5) solving ln(alpha/(1-alpha)) = Omega*(2*alpha - 1) -- the symmetric regular-solution
    binodal condition (equal component-1 activity in both phases), derived independently of lle.py."""
    def eqn(a):
        return math.log(a / (1.0 - a)) - omega * (2.0 * a - 1.0)
    return brentq(eqn, 1.0e-6, 0.5 - 1.0e-9)


class TestRegularSolutionToy:
    def test_finite_difference_gradient_matches_analytic(self):
        gfe = RegularSolutionGFE(omega=3.0)
        x = np.array([0.3, 0.7])
        h = 1.0e-6
        for i in range(2):
            xp, xm = x.copy(), x.copy()
            xp[i] += h; xm[i] -= h
            fd = (gfe.g(xp) - gfe.g(xm)) / (2 * h)
            assert fd == pytest.approx(gfe.grad(x)[i], rel=1e-4)

    def test_phase_splits_below_and_above_critical_omega(self):
        # Omega = 2 is the critical point of the symmetric regular solution; below it, the mixture is fully
        # miscible everywhere (1 phase); above it, the equimolar feed splits into two phases.
        for omega, expect_split in [(1.0, False), (1.5, False), (3.0, True), (4.0, True)]:
            gfe = RegularSolutionGFE(omega=omega)
            result = solve_pep_gfe(gfe, np.array([0.5, 0.5]), tol=1e-11)
            assert result.converged
            n_phases = len(result.y)
            if expect_split:
                assert n_phases == 2, f"omega={omega}: expected phase split, got {n_phases} phase(s)"
            else:
                assert n_phases == 1, f"omega={omega}: expected single phase, got {n_phases} phase(s)"

    @pytest.mark.parametrize("omega", [2.5, 3.0, 4.0, 6.0])
    def test_binodal_matches_independent_analytic_solution(self, omega):
        alpha = _analytic_binodal(omega)
        gfe = RegularSolutionGFE(omega=omega)
        result = solve_pep_gfe(gfe, np.array([0.5, 0.5]), tol=1e-12, max_outer=300)
        assert result.converged
        assert len(result.y) == 2
        x1_values = sorted(result.x[:, 0])
        assert x1_values[0] == pytest.approx(alpha, abs=5e-3)
        assert x1_values[1] == pytest.approx(1.0 - alpha, abs=5e-3)
        # equal molar amounts by symmetry of the equimolar feed
        assert sorted(result.y) == pytest.approx([0.5, 0.5], abs=5e-3)

    def test_mass_balance_is_conserved(self):
        gfe = RegularSolutionGFE(omega=3.5)
        b = np.array([0.35, 0.65])
        result = solve_pep_gfe(gfe, b, tol=1e-12)
        recombined = (result.y[:, None] * result.x).sum(axis=0)
        assert recombined == pytest.approx(b, abs=1e-4)

    def test_off_center_feed_gives_unequal_phase_amounts(self):
        gfe = RegularSolutionGFE(omega=4.0)
        b = np.array([0.2, 0.8])
        result = solve_pep_gfe(gfe, b, tol=1e-12)
        assert result.converged
        assert len(result.y) == 2
        # the phase amounts must satisfy the lever rule for the converged phase compositions
        x1 = result.x[:, 0]
        y_from_lever = (b[0] - x1[1]) / (x1[0] - x1[1])
        assert sorted(result.y) == pytest.approx(sorted([y_from_lever, 1 - y_from_lever]), abs=5e-3)

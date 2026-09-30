"""Validation of the stability-determinant machinery (spinodal.py) against the same independently-solvable
toy system used in test_lle.py -- the symmetric binary regular solution, whose spinodal has a known closed
form, entirely independent of AIOMFAC (see test_lle.py's module docstring for why a toy system is used here
rather than AIOMFAC/Fortran, for which no independent LLE/stability reference exists).

For g(x) = x1 ln x1 + x2 ln x2 + Omega x1 x2 (RT units), the binary spinodal condition (d^2 g / dx1^2 = 0 at
fixed x1 + x2 = 1) is 1 / (x1 x2) = 2 * Omega; at the symmetric point x1 = x2 = 0.5 this is Omega = 2, exactly
the same critical Omega used in test_lle.py's ``test_phase_splits_below_and_above_critical_omega``. This gives
an exact, independent check of ``spinodal.stability_determinant`` (whose reduced Hessian, for a 2-species
system with one dependent mole fraction eliminated, is just this scalar second derivative).
"""
from __future__ import annotations

import numpy as np
import pytest

from aiomfac_py.spinodal import stability_determinant
from test_lle import RegularSolutionGFE


class TestRegularSolutionSpinodal:
    @pytest.mark.parametrize("omega,expect_sign", [(1.0, 1), (1.5, 1), (2.5, -1), (4.0, -1)])
    def test_sign_matches_known_stability(self, omega, expect_sign):
        gfe = RegularSolutionGFE(omega=omega)
        L = stability_determinant(gfe, np.array([0.5, 0.5]), dep_index=1)
        assert np.sign(L) == expect_sign, f"omega={omega}: expected sign {expect_sign}, got L={L}"

    def test_near_zero_at_critical_omega(self):
        gfe = RegularSolutionGFE(omega=2.0)
        L = stability_determinant(gfe, np.array([0.5, 0.5]), dep_index=1)
        assert abs(L) < 0.05, f"expected L close to 0 at the critical point, got {L}"

    def test_matches_analytic_second_derivative_away_from_center(self):
        # away from x1=0.5 the spinodal condition is 1/(x1*x2) = 2*Omega -- check the *value*, not just the
        # sign, at a generic interior point.
        omega = 3.0
        gfe = RegularSolutionGFE(omega=omega)
        x1 = 0.3
        L = stability_determinant(gfe, np.array([x1, 1.0 - x1]), dep_index=1)
        analytic = 1.0 / (x1 * (1.0 - x1)) - 2.0 * omega
        assert L == pytest.approx(analytic, rel=1.0e-2)

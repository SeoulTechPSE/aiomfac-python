"""Smoke tests for the RH-dependent gas/particle partitioning module (gp_partition.py).

No independent reference exists for this (same situation as lle.py/spinodal.py -- see their module
docstrings), so these tests check basic physical sanity and mass-balance bookkeeping rather than absolute
numbers: total moles are conserved, everything partitions towards the particle phase as RH increases for a
compound whose particle-phase solvation improves with water content, and the achieved water activity tracks
the requested RH.
"""
from __future__ import annotations

import numpy as np
import pytest

from aiomfac_py.io import Component
from aiomfac_py.gp_partition import VolatileSpecies, gp_partition


@pytest.fixture
def glycerol_case():
    glycerol = Component(2, "glycerol", ((150, 2), (151, 1), (153, 3)))
    nh4so4 = Component(3, "(NH4)2SO4", ((204, 2), (261, 1)))
    species = VolatileSpecies(component=glycerol, M=92.094e-3, p0_Pa=2.284e-2, n_total=3.0e-8)
    return nh4so4, species


class TestGPPartitionSmoke:
    def test_water_activity_tracks_target_rh(self, glycerol_case):
        salt, species = glycerol_case
        for RH in (0.3, 0.6, 0.9):
            res = gp_partition(salt, [species], n_salt=1.0e-8, T_K=298.15, RH=RH, V_gas_m3=1.0,
                                max_iter=60, tol=1.0e-4)
            assert res.aw == pytest.approx(RH, abs=0.02)

    def test_mass_balance_conserved(self, glycerol_case):
        salt, species = glycerol_case
        res = gp_partition(salt, [species], n_salt=1.0e-8, T_K=298.15, RH=0.7, V_gas_m3=1.0,
                            max_iter=60, tol=1.0e-4)
        total = res.n_org_PM[0] + res.n_org_gas[0]
        assert total == pytest.approx(species.n_total, rel=1.0e-6)

    def test_more_condenses_at_higher_rh(self, glycerol_case):
        # glycerol is hydrophilic: more of it should stay condensed as the particle takes up more water.
        salt, species = glycerol_case
        r_low = gp_partition(salt, [species], n_salt=1.0e-8, T_K=298.15, RH=0.3, V_gas_m3=1.0,
                              max_iter=60, tol=1.0e-4)
        r_high = gp_partition(salt, [species], n_salt=1.0e-8, T_K=298.15, RH=0.95, V_gas_m3=1.0,
                               max_iter=60, tol=1.0e-4)
        assert r_high.n_org_PM[0] > r_low.n_org_PM[0]

    def test_activities_are_between_zero_and_a_few(self, glycerol_case):
        salt, species = glycerol_case
        res = gp_partition(salt, [species], n_salt=1.0e-8, T_K=298.15, RH=0.8, V_gas_m3=1.0,
                            max_iter=60, tol=1.0e-4)
        assert 0.0 <= res.activities_org[0] < 10.0
        assert 0.0 <= res.aw <= 1.0


class TestGPPartitionJointLMRegression:
    """Regression test for the joint Levenberg-Marquardt solver (`method="lm"`, the default since this was
    added): the exact 4-organic system from `03_zuend2010_lle.ipynb`'s Fig. 8-10, where plain successive
    substitution (`method="successive_substitution"`) is confirmed (see gp_partition.py's module docstring)
    to never converge -- glycerol and 1,6-hexanediol chaotically alternate between near-zero and a material
    particle-phase amount, iteration to iteration, at every RH tested. `method="lm"` must actually converge
    here, tightly, which plain successive substitution cannot."""

    @pytest.fixture
    def four_organic_case(self):
        hexanediol16 = Component(2, "1,6-hexanediol", ((142, 4), (150, 2), (153, 2)))
        glycerol = Component(3, "glycerol", ((150, 2), (151, 1), (153, 3)))
        decanetriol = Component(4, "1,2,10-decanetriol", ((142, 7), (150, 2), (151, 1), (153, 3)))
        octanetetrol = Component(5, "1,2,5,8-octanetetrol", ((142, 4), (150, 2), (151, 2), (153, 4)))
        nh4so4 = Component(6, "(NH4)2SO4", ((204, 2), (261, 1)))
        organics = [
            VolatileSpecies(hexanediol16, 1.18172e-1, 5.695e-2, 3.0e-8),
            VolatileSpecies(glycerol, 9.20940e-2, 2.284e-2, 3.0e-8),
            VolatileSpecies(decanetriol, 1.90276e-1, 1.826e-4, 3.0e-8),
            VolatileSpecies(octanetetrol, 1.78224e-1, 6.725e-5, 3.0e-8),
        ]
        return nh4so4, organics

    @pytest.mark.parametrize("RH", [0.20, 0.40, 0.60, 0.80, 0.90, 0.95, 0.98, 0.99])
    def test_lm_converges_where_successive_substitution_does_not(self, four_organic_case, RH):
        salt, organics = four_organic_case
        res_lm = gp_partition(salt, organics, n_salt=1.0e-8, T_K=298.15, RH=RH, V_gas_m3=1.0,
                               max_iter=60, tol=1.0e-6, method="lm", check_lle=False)
        assert res_lm.converged
        assert res_lm.aw == pytest.approx(RH, abs=1.0e-4)

        res_ss = gp_partition(salt, organics, n_salt=1.0e-8, T_K=298.15, RH=RH, V_gas_m3=1.0,
                               max_iter=60, tol=1.0e-6, method="successive_substitution", check_lle=False)
        assert not res_ss.converged

    def test_lm_mass_balance_conserved_for_all_organics(self, four_organic_case):
        salt, organics = four_organic_case
        res = gp_partition(salt, organics, n_salt=1.0e-8, T_K=298.15, RH=0.6, V_gas_m3=1.0,
                            max_iter=60, tol=1.0e-6, method="lm", check_lle=False)
        totals = res.n_org_PM + res.n_org_gas
        expected = np.array([o.n_total for o in organics])
        assert totals == pytest.approx(expected, rel=1.0e-6)

    def test_lm_does_not_stall_at_loose_tol_near_rh_0_99(self, four_organic_case):
        """Regression test for a second, distinct failure mode found after the fix above: with a *loose*
        caller `tol` (as `03_zuend2010_lle.ipynb`'s RH sweep uses, `tol=1e-4`), passing that same loose value
        straight through to `scipy.optimize.least_squares`'s internal `xtol`/`ftol`/`gtol` let TRF declare
        victory at a bound-constrained pseudo-stationary point (one organic's particle fraction pinned near
        its r_j=1 bound at this near-saturation RH) after only ~9 function evaluations, with `aw` still
        ~0.0086 off the RH=0.99 target -- a genuine internal-tolerance stall, confirmed to be independent of
        `max_iter` (raising it alone did not help; only decoupling the internal tolerances from `tol` did).
        `_solve_joint_lm` now always uses tight internal tolerances regardless of the caller's `tol`."""
        salt, organics = four_organic_case
        res = gp_partition(salt, organics, n_salt=1.0e-8, T_K=298.15, RH=0.99, V_gas_m3=1.0,
                            max_iter=50, tol=1.0e-4, method="lm", check_lle=False)
        assert res.converged
        assert res.aw == pytest.approx(0.99, abs=1.0e-4)


class TestGPPartitionPseudoTransient:
    """Tests for `method="pseudo_transient"` (`_solve_pseudo_transient`): the RH-continuation fallback
    inspired by Amundson, Caboussat, He, Landry & Seinfeld (2007, C. R. Acad. Sci. 344, 519-522) -- see
    `_solve_pseudo_transient`'s docstring for how the two relate. Not required by any currently known
    convergence failure (`method="lm"` already resolves them directly -- see
    `TestGPPartitionJointLMRegression`), but it should be at least as robust on the same hard case, since each
    of its stages is itself just an `_solve_joint_lm` solve from a warm-started, closer initial guess."""

    @pytest.fixture
    def four_organic_case(self):
        hexanediol16 = Component(2, "1,6-hexanediol", ((142, 4), (150, 2), (153, 2)))
        glycerol = Component(3, "glycerol", ((150, 2), (151, 1), (153, 3)))
        decanetriol = Component(4, "1,2,10-decanetriol", ((142, 7), (150, 2), (151, 1), (153, 3)))
        octanetetrol = Component(5, "1,2,5,8-octanetetrol", ((142, 4), (150, 2), (151, 2), (153, 4)))
        nh4so4 = Component(6, "(NH4)2SO4", ((204, 2), (261, 1)))
        organics = [
            VolatileSpecies(hexanediol16, 1.18172e-1, 5.695e-2, 3.0e-8),
            VolatileSpecies(glycerol, 9.20940e-2, 2.284e-2, 3.0e-8),
            VolatileSpecies(decanetriol, 1.90276e-1, 1.826e-4, 3.0e-8),
            VolatileSpecies(octanetetrol, 1.78224e-1, 6.725e-5, 3.0e-8),
        ]
        return nh4so4, organics

    @pytest.mark.parametrize("RH", [0.20, 0.40, 0.60, 0.80, 0.90, 0.95, 0.98, 0.99])
    def test_pseudo_transient_converges_on_the_notebook_case(self, four_organic_case, RH):
        salt, organics = four_organic_case
        res = gp_partition(salt, organics, n_salt=1.0e-8, T_K=298.15, RH=RH, V_gas_m3=1.0,
                            max_iter=50, tol=1.0e-4, method="pseudo_transient", check_lle=False)
        assert res.converged
        assert res.aw == pytest.approx(RH, abs=1.0e-4)

    def test_pseudo_transient_mass_balance_conserved(self, four_organic_case):
        salt, organics = four_organic_case
        res = gp_partition(salt, organics, n_salt=1.0e-8, T_K=298.15, RH=0.6, V_gas_m3=1.0,
                            max_iter=60, tol=1.0e-6, method="pseudo_transient", check_lle=False)
        totals = res.n_org_PM + res.n_org_gas
        expected = np.array([o.n_total for o in organics])
        assert totals == pytest.approx(expected, rel=1.0e-6)

    def test_pseudo_transient_single_stage_matches_direct_lm(self, four_organic_case):
        """n_stages=1 (or RH already equal to the starting guess's own water activity) should fall back to
        a single `_solve_joint_lm` solve -- i.e. behave the same as `method="lm"` -- rather than erroring."""
        salt, organics = four_organic_case
        res = gp_partition(salt, organics, n_salt=1.0e-8, T_K=298.15, RH=0.6, V_gas_m3=1.0,
                            max_iter=60, tol=1.0e-6, method="pseudo_transient", n_stages=1, check_lle=False)
        assert res.converged
        assert res.aw == pytest.approx(0.6, abs=1.0e-6)

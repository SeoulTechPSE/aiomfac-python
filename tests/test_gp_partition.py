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

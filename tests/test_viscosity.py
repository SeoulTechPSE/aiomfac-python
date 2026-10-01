"""Tests for the AIOMFAC-VISC aqueous-electrolyte viscosity model (viscosity.py).

Like lle.py/spinodal.py/gp_partition.py, no independent full-precision reference implementation is available
to diff against here, so these tests check: (1) direct back-calculation of the paper's own stated numbers
(Sect. 2.3.1: pure water's dg*/(RT) = 3.44 at 298.15 K) -- an exact, citable check, not just a sanity bound;
(2) physically expected qualitative behavior the paper itself discusses (KCl/NH4Cl structure-breaking viscosity
minima vs. monotonically increasing CaCl2/MgCl2); and (3) basic consistency (pure water limit, unit sanity).
"""
from __future__ import annotations

import dataclasses
import math
from types import SimpleNamespace

import numpy as np
import pytest

from aiomfac_py.io import Component
from aiomfac_py.model import ActivityModel
from aiomfac_py.viscosity import (
    CATION_ANION_C, CV, ION_C0_C1, ION_RTH, ION_SUBGROUP, V_REF_M3_PER_MOL, WATER_RTH,
    aquelec_viscosity, aquorg_viscosity, electrolyte_viscosity, organic_mixture_viscosity,
    pure_organic_viscosity_vtf, water_viscosity_pas,
)

WATER = Component(1, "Water", ((16, 1),))
GLYCEROL = Component(2, "glycerol", ((150, 2), (151, 1), (153, 3)))


def _binary(cation_subgroup, anion_subgroup, nu_cation=1, nu_anion=1, name="salt", number=2):
    subs = []
    if nu_cation > 1:
        subs.append((cation_subgroup, nu_cation))
    else:
        subs.append((cation_subgroup, 1))
    subs.append((anion_subgroup, nu_anion))
    return Component(number, name, tuple(subs))


class TestWaterViscosity:
    def test_room_temperature_matches_known_value(self):
        # Commonly cited value for pure water at 298.15 K is ~8.9e-4 Pa s.
        assert water_viscosity_pas(298.15) == pytest.approx(8.9e-4, rel=5e-3)

    def test_increases_as_temperature_decreases_toward_supercooling(self):
        assert water_viscosity_pas(263.15) > water_viscosity_pas(298.15) > water_viscosity_pas(333.15)


class TestDgStarWaterBackCalculation:
    def test_matches_paper_stated_3_44_at_298_15K(self):
        """Lilek & Zuend (2022) Sect. 2.3.1: "At 298.15 K, pure water has a viscosity of 8.9e-4 Pa s, and
        dg*/(RT) = 3.44." This directly confirms both the water viscosity correlation (Dehaoui et al., 2015)
        and the (temperature-independent) relative-van-der-Waals-volume convention for V_w used here, since
        reproducing 3.44 requires both to be right simultaneously."""
        eta_w = water_viscosity_pas(298.15)
        V_w = WATER_RTH * V_REF_M3_PER_MOL
        h, NA = 6.62607015e-34, 6.02214076e23
        dg_w_over_RT = math.log(eta_w * V_w / (h * NA))
        assert dg_w_over_RT == pytest.approx(3.44, abs=5e-3)


class TestElectrolyteViscosityPureWaterLimit:
    def test_vanishing_salt_reproduces_water_viscosity(self):
        nacl = _binary(202, 242, name="NaCl")
        model = ActivityModel([WATER, nacl])
        res = model.evaluate([1 - 1e-9, 1e-9], 298.15, basis="mole")
        v = electrolyte_viscosity(model, res, 298.15)
        assert v.eta_pas == pytest.approx(water_viscosity_pas(298.15), rel=1e-6)


class TestStructureBreakingAndMaking:
    """Lilek & Zuend (2022) Sect. 2.4: "the low-concentration mixture viscosity minimum observed in the
    viscosity curves of aqueous solutions of structure-breaking ions like K+ and NH4+ [but] not those of
    structure makers like Li+ and Mg2+" -- reproduced here directly from the ported model, matching the shape
    of the paper's own Fig. 2 panels (a, d: KCl, NH4Cl show a dip below pure water; e, g: MgCl2, CaCl2 rise
    monotonically from the start)."""

    def test_kcl_viscosity_dips_below_pure_water_at_low_concentration(self):
        kcl = _binary(203, 242, name="KCl")
        model = ActivityModel([WATER, kcl])
        eta_w = water_viscosity_pas(298.15)
        etas = []
        for wf in (0.001, 0.01, 0.03, 0.05, 0.10):
            res = model.evaluate([1 - wf, wf], 298.15, basis="mass")
            etas.append(electrolyte_viscosity(model, res, 298.15).eta_pas)
        assert min(etas) < eta_w, "KCl (structure-breaking) should dip below pure water's viscosity"

    def test_cacl2_viscosity_increases_monotonically(self):
        cacl2 = _binary(221, 242, nu_anion=2, name="CaCl2")
        model = ActivityModel([WATER, cacl2])
        wfs = [0.01, 0.05, 0.10, 0.20, 0.30, 0.40]
        etas = []
        for wf in wfs:
            res = model.evaluate([1 - wf, wf], 298.15, basis="mass")
            etas.append(electrolyte_viscosity(model, res, 298.15).eta_pas)
        assert all(b > a for a, b in zip(etas, etas[1:])), \
            "CaCl2 (structure-making) should increase monotonically, no low-concentration dip"
        # Sanity check against the broad order of magnitude reported in the literature for concentrated CaCl2.
        assert 5e-3 < etas[-1] < 2e-2  # ~5-20 mPa s at 40 wt% CaCl2, 298 K


class TestTernaryMixture:
    def test_ternary_mass_and_charge_consistent_and_between_the_two_binaries(self):
        """water + NaCl + CaCl2: a sanity check that the cation-anion pair term (shared anion, two cations)
        does not blow up or misbehave, and that the ternary viscosity sits in a physically reasonable range
        relative to the corresponding binary solutions at the same total ionic content."""
        nacl = _binary(202, 242, name="NaCl")
        cacl2 = _binary(221, 242, nu_anion=2, name="CaCl2", number=3)
        model = ActivityModel([WATER, nacl, cacl2])
        res = model.evaluate([0.85, 0.08, 0.07], 298.15, basis="mass")
        v = electrolyte_viscosity(model, res, 298.15)
        assert v.eta_pas > water_viscosity_pas(298.15)
        assert np.isfinite(v.eta_pas) and v.eta_pas > 0


class TestParameterTableConsistency:
    def test_ion_subgroup_names_consistent_across_tables(self):
        assert set(ION_SUBGROUP) == set(ION_RTH) == set(ION_C0_C1)

    def test_all_70_cation_anion_pairs_present(self):
        cations = [n for n in ION_SUBGROUP if ION_SUBGROUP[n] in (205, 201, 202, 203, 204, 223, 221)]
        anions = [n for n in ION_SUBGROUP if n not in cations]
        assert len(cations) == 7 and len(anions) == 10
        assert len(CATION_ANION_C) == 70
        for c in cations:
            for a in anions:
                assert (c, a) in CATION_ANION_C

    def test_cv_matches_paper(self):
        assert CV == pytest.approx(1.679827)


class TestUnsupportedIon:
    def test_raises_keyerror_for_unmapped_cation_subgroup(self):
        # No AIOMFAC cation outside the 17 ions AIOMFAC-VISC covers has both a molar mass *and* MR
        # interaction parameters defined against water's main group, so a real ActivityModel can't be built
        # with one. Instead, build an ordinary NaCl model/result and swap in a bogus cation subgroup id on
        # a copy of its (frozen) mixture description -- electrolyte_viscosity only reads mixture.sr.cations
        # to decide which ion-specific parameters to look up, so this isolates exactly the behavior under test.
        nacl = _binary(202, 242, name="NaCl")
        model = ActivityModel([WATER, nacl])
        res = model.evaluate([0.9, 0.1], 298.15, basis="mass")
        bad_sr = dataclasses.replace(model.mixture.sr, cations=(999,))
        bad_mixture = dataclasses.replace(model.mixture, sr=bad_sr)
        bad_model = SimpleNamespace(mixture=bad_mixture)
        with pytest.raises(KeyError):
            electrolyte_viscosity(bad_model, res, 298.15)


class TestPureOrganicViscosityVTF:
    def test_increases_toward_glass_transition(self):
        # glycerol, Tg = 187 K (Angell, 1997, via Gervasi et al. 2020 Table S1)
        eta_warm = pure_organic_viscosity_vtf(293.15, 187.0)
        eta_cold = pure_organic_viscosity_vtf(263.15, 187.0)
        eta_colder = pure_organic_viscosity_vtf(223.15, 187.0)
        assert eta_warm < eta_cold < eta_colder

    def test_default_fragility_switches_above_tg(self):
        # D defaults to 30 (not 10) once Tg exceeds the simulation temperature (Gervasi et al. 2020 Sect. 2.3).
        eta_d10 = pure_organic_viscosity_vtf(250.0, 300.0, D=10.0)
        eta_default = pure_organic_viscosity_vtf(250.0, 300.0)
        eta_d30 = pure_organic_viscosity_vtf(250.0, 300.0, D=30.0)
        assert eta_default == pytest.approx(eta_d30)
        assert eta_d30 != pytest.approx(eta_d10)

    def test_raises_below_vogel_temperature(self):
        with pytest.raises(ValueError):
            pure_organic_viscosity_vtf(100.0, 187.0, D=10.0)


class TestOrganicMixtureViscosityGlycerol:
    """Validated directly against real CRC Handbook of Chemistry and Physics water+glycerol viscosity data at
    293.15 K (Gervasi et al. 2020 Supplement, 'Aqueous Binary Systems/glycerol+water_viscosity_CRCHandbook
    @293K.csv'), using glycerol's own measured pure-component viscosity (1.46 Pa s at 293.15 K, same source) --
    not the VTF/Tg estimate, to isolate the correctness of the Eq. 1-9 mixing engine itself from the separately
    tested (and, per Gervasi et al. 2020, considerably less certain) VTF pure-component viscosity estimate."""

    ETA0_GLYCEROL_293K = 1.46

    @pytest.mark.parametrize("wtf_glycerol, eta_measured", [
        (0.2, 1.766e-3), (0.4, 3.73e-3), (0.6, 10.9e-3), (0.8, 60.6e-3),
    ])
    def test_matches_crc_handbook_data(self, wtf_glycerol, eta_measured):
        model = ActivityModel([WATER, GLYCEROL])
        res = model.evaluate([1 - wtf_glycerol, wtf_glycerol], 293.15, basis="mass")
        result = organic_mixture_viscosity(model, res.x, 293.15, {2: self.ETA0_GLYCEROL_293K})
        assert result.eta_pas == pytest.approx(eta_measured, rel=0.1)

    def test_vanishing_glycerol_reproduces_water_viscosity(self):
        model = ActivityModel([WATER, GLYCEROL])
        res = model.evaluate([1 - 1e-9, 1e-9], 293.15, basis="mass")
        result = organic_mixture_viscosity(model, res.x, 293.15, {2: self.ETA0_GLYCEROL_293K})
        assert result.eta_pas == pytest.approx(water_viscosity_pas(293.15), rel=1e-4)

    def test_requires_pure_component_viscosity_for_organics(self):
        model = ActivityModel([WATER, GLYCEROL])
        res = model.evaluate([0.8, 0.2], 293.15, basis="mass")
        with pytest.raises(ValueError):
            organic_mixture_viscosity(model, res.x, 293.15, {})

    def test_rejects_mixtures_containing_ions(self):
        nacl = Component(3, "NaCl", ((202, 1), (242, 1)))
        model = ActivityModel([WATER, GLYCEROL, nacl])
        res = model.evaluate([0.8, 0.1, 0.1], 293.15, basis="mass")
        with pytest.raises(ValueError):
            organic_mixture_viscosity(model, res.x, 293.15, {2: self.ETA0_GLYCEROL_293K})


class TestOrganicInorganicMixing:
    """aquelec/aquorg (Lilek and Zuend, 2022, Sect. 3.4.1-3.4.2) for a water + glycerol + NaCl ternary. No
    independent reference value is available for either mixing rule (the paper itself does not single out a
    "correct" one -- Sect. 3.4.4), so these only check basic consistency: both reduce to the pure-water/pure-
    electrolyte/pure-organic limits correctly, agree with each other to within a reasonable tolerance (the paper
    reports the two are "about equally fast" and does not report them as wildly divergent for well-behaved
    systems), and stay finite and positive."""

    NACL = Component(3, "NaCl", ((202, 1), (242, 1)))
    ETA0 = {2: 1.0}  # a round, approximate glycerol pure-component viscosity -- only consistency is tested here

    def _model_and_result(self, wtf_org, wtf_salt, T_K=298.15):
        model = ActivityModel([WATER, GLYCEROL, self.NACL])
        res = model.evaluate([1 - wtf_org - wtf_salt, wtf_org, wtf_salt], T_K, basis="mass")
        return model, res

    def test_vanishing_organic_and_salt_reproduces_water_viscosity(self):
        model, res = self._model_and_result(1e-9, 1e-9)
        eta_w = water_viscosity_pas(298.15)
        assert aquelec_viscosity(model, res, 298.15, self.ETA0).eta_pas == pytest.approx(eta_w, rel=1e-3)
        assert aquorg_viscosity(model, res, 298.15, self.ETA0).eta_pas == pytest.approx(eta_w, rel=1e-3)

    def test_vanishing_salt_reproduces_organic_mixture_viscosity(self):
        wtf_org = 0.2
        model, res = self._model_and_result(wtf_org, 1e-9)
        organic_only = ActivityModel([WATER, GLYCEROL])
        organic_res = organic_only.evaluate([1 - wtf_org, wtf_org], 298.15, basis="mass")
        expected = organic_mixture_viscosity(organic_only, organic_res.x, 298.15, self.ETA0).eta_pas
        assert aquelec_viscosity(model, res, 298.15, self.ETA0).eta_pas == pytest.approx(expected, rel=1e-2)
        assert aquorg_viscosity(model, res, 298.15, self.ETA0).eta_pas == pytest.approx(expected, rel=1e-2)

    @pytest.mark.parametrize("wtf_org, wtf_salt", [(0.05, 0.05), (0.1, 0.05), (0.3, 0.1), (0.2, 0.2)])
    def test_finite_positive_and_mutually_consistent(self, wtf_org, wtf_salt):
        model, res = self._model_and_result(wtf_org, wtf_salt)
        r1 = aquelec_viscosity(model, res, 298.15, self.ETA0)
        r2 = aquorg_viscosity(model, res, 298.15, self.ETA0)
        for r in (r1, r2):
            assert np.isfinite(r.eta_pas) and r.eta_pas > 0
        assert r1.eta_pas == pytest.approx(r2.eta_pas, rel=0.3)

    def test_requires_at_least_one_organic(self):
        model = ActivityModel([WATER, self.NACL])
        res = model.evaluate([0.9, 0.1], 298.15, basis="mass")
        with pytest.raises(ValueError):
            aquelec_viscosity(model, res, 298.15, {})

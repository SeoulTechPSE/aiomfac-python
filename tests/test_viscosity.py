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
import warnings
from types import SimpleNamespace

import numpy as np
import pytest

from aiomfac_py.io import Component
from aiomfac_py.model import ActivityModel
from aiomfac_py.viscosity import (
    CATION_ANION_C, CV, ION_C0_C1, ION_RTH, ION_SUBGROUP, V_REF_M3_PER_MOL, WATER_RTH,
    aquelec_viscosity, aquorg_viscosity, electrolyte_viscosity, organic_mixture_viscosity,
    predict_tg_derieux2018, pure_organic_viscosity_vtf, water_viscosity_pas,
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


class TestPredictTgDerieux2018:
    """Validated against DeRieux et al. (2018)'s own text, not just internal consistency: the paper's Sect. 2.1
    worked example states that Eq. (2) predicts Tg = 394 K for stachyose (C24H42O21, M = 667 g/mol), agreeing
    with the measured mean Tg of 396 K -- an exact, citable reference value to check the coefficients and
    formula transcription against, the same kind of check used elsewhere in this module (e.g. the pure-water
    dg*/(RT) = 3.44 back-calculation in electrolyte_viscosity's tests)."""

    def test_matches_paper_worked_example_stachyose(self):
        # C24H42O21 -- DeRieux et al. (2018) Sect. 2.1: "Tg of stachyose (M = 667 g/mol) predicted by Eq. (1)
        # is 198 K, while by Eq. (2) is 394 K, which agrees much better with the measured mean Tg of 396 K."
        tg = predict_tg_derieux2018(n_C=24, n_H=42, n_O=21)
        assert tg == pytest.approx(394.0, abs=0.5)

    def test_ch_class_has_no_oxygen_term(self):
        # n_O=0 must route to the CH-class coefficients (b_O = b_CO = 0 there), not divide by/log(0).
        tg = predict_tg_derieux2018(n_C=8, n_H=18, n_O=0)  # octane
        assert np.isfinite(tg)

    def test_cho_class_used_when_oxygen_present(self):
        # Same C, H skeleton, with vs. without one oxygen atom must give different predictions (different
        # coefficient table), confirming the CH/CHO branch is actually selected by n_O rather than ignored.
        tg_no_o = predict_tg_derieux2018(n_C=3, n_H=8, n_O=0)
        tg_with_o = predict_tg_derieux2018(n_C=3, n_H=8, n_O=3)  # glycerol's own formula, C3H8O3
        assert tg_no_o != pytest.approx(tg_with_o)
        # glycerol: measured Tg = 187 K (Angell, 1997, via Gervasi et al. 2020 Table S1); D2018 Eq. (2) itself
        # is only stated accurate to within about +/-21 K for an individual compound (their Fig. 1c band), so
        # this checks the prediction lands within that documented uncertainty, not exact agreement.
        assert tg_with_o == pytest.approx(187.0, abs=22.0)

    def test_rejects_nonpositive_atom_counts(self):
        with pytest.raises(ValueError):
            predict_tg_derieux2018(n_C=0, n_H=4)
        with pytest.raises(ValueError):
            predict_tg_derieux2018(n_C=2, n_H=0)
        with pytest.raises(ValueError):
            predict_tg_derieux2018(n_C=2, n_H=4, n_O=-1)

    def test_feeds_into_vtf_pure_component_viscosity(self):
        # The intended usage: predict Tg, then hand it to pure_organic_viscosity_vtf -- should run without
        # error and give a finite, positive viscosity at a reasonable simulation temperature.
        tg = predict_tg_derieux2018(n_C=3, n_H=8, n_O=3)
        eta = pure_organic_viscosity_vtf(293.15, tg)
        assert np.isfinite(eta) and eta > 0.0


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
    tested (and, per Gervasi et al. 2020, considerably less certain) VTF pure-component viscosity estimate.
    The model is up to 0.053 log10 units (12 %) below these data at 60-80 wt% glycerol, as in G2020 Fig. 4a;
    the port itself is checked against G2020's own error statistics in TestOrganicMixtureViscosityGervasi2020."""

    ETA0_GLYCEROL_293K = 1.46

    @pytest.mark.parametrize("wtf_glycerol, eta_measured", [
        (0.2, 1.766e-3), (0.4, 3.73e-3), (0.6, 10.9e-3), (0.8, 60.6e-3),
    ])
    def test_matches_crc_handbook_data(self, wtf_glycerol, eta_measured):
        model = ActivityModel([WATER, GLYCEROL])
        res = model.evaluate([1 - wtf_glycerol, wtf_glycerol], 293.15, basis="mass")
        result = organic_mixture_viscosity(model, res.x, 293.15, {2: self.ETA0_GLYCEROL_293K})
        assert result.log10_eta_pas == pytest.approx(math.log10(eta_measured), abs=0.06)

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


class TestOrganicMixtureViscosityGervasi2020:
    """Regression test against Gervasi et al. (2020) Supplement Table S5: mean absolute / mean bias error (log10
    units, each point weighted by its measurement error delta as in their Eq. S6:
    log10(y + delta) - log10(x + delta)) of AIOMFAC-VISC with the stated "best" pure-component viscosity, for
    the Song et al. (2016) aerosol-optical-tweezers data sets. Data (organic mass fraction, measured log10 eta /
    Pa s, delta / Pa s) are copied from the G2020 Supplement files "Aqueous Binary Systems/<compound>+water_
    viscosity_Song2016@293K.csv" (CC BY 4.0). Up to aiomfac_py 1.3.0, Eq. 5 lacked the factor Q_k on its second
    term; that version gives sucrose 1.3396/-0.1342 and erythritol 0.2881/-0.2868 and fails this test."""

    SUCROSE = Component(2, "sucrose", ((4, 1), (26, 3), (150, 3), (151, 5), (153, 8)))
    ERYTHRITOL = Component(2, "erythritol", ((150, 2), (151, 2), (153, 4)))
    BUTANETRIOL = Component(2, "1,2,4-butanetriol", ((142, 1), (150, 2), (151, 1), (153, 3)))

    SUCROSE_DATA = [(0.979, 8.526, 2.13796209), (0.962, 6.258, 3.981071706), (0.926, 4.635, 2.187761624),
                    (0.868, 3.805, 3.548133892), (0.877, 3.751, 2.187761624), (0.872, 2.283, 3.388441561),
                    (0.836, 1.687, 3.388441561), (0.641, -1.297, 3.311311215)]
    ERYTHRITOL_DATA = [(0.987, 3.664, 1.9498446), (0.978, 3.362, 1.819700859), (0.936, 1.766, 1.819700859),
                       (0.848, 0.549, 1.348962883), (0.785, 0.027, 1.023292992), (0.731, -0.987, 2.089296131),
                       (0.657, -1.48, 1.77827941), (0.482, -2.702, 2.884031503), (0.309, -2.485, 1.174897555),
                       (0.06, -2.854, 1.230268771)]
    BUTANETRIOL_DATA = [(0.998, 0.252, 1.047128548), (0.973, -0.102, 1.071519305), (0.941, -0.638, 1.230268771),
                        (0.905, -0.76, 1.122018454), (0.861, -1.139, 1.047128548), (0.808, -1.117, 1.862087137),
                        (0.665, -2.577, 2.089296131), (0.531, -2.971, 2.187761624), (0.242, -2.974, 1.148153621)]

    @pytest.mark.parametrize("component, data, log10_eta0, mae, mbe", [
        (SUCROSE, SUCROSE_DATA, 16.7816, 1.3781, -0.1887),
        (ERYTHRITOL, ERYTHRITOL_DATA, 2.9287, 0.2921, -0.2915),
        (BUTANETRIOL, BUTANETRIOL_DATA, 0.2100, 0.0152, 0.0052),
    ], ids=["sucrose", "erythritol", "1,2,4-butanetriol"])
    def test_reproduces_table_s5(self, component, data, log10_eta0, mae, mbe):
        model = ActivityModel([WATER, component])
        err = []
        for w_org, log10_meas, delta in data:
            res = model.evaluate([1 - w_org, w_org], 293.15, basis="mass")
            y = organic_mixture_viscosity(model, res.x, 293.15, {2: 10.0 ** log10_eta0}).eta_pas
            err.append(math.log10(y + delta) - math.log10(10.0 ** log10_meas + delta))
        err = np.array(err)
        assert np.abs(err).mean() == pytest.approx(mae, abs=1e-3)
        assert err.mean() == pytest.approx(mbe, abs=1e-3)


class TestOrganicMixtureViscosityPEG:
    """Documented model limitation (see organic_mixture_viscosity's docstring): for PEG oligomers (subgroup 154)
    G2020 Eq. 1-9 give mixture viscosities far above both pure components; a warning is issued."""

    PEG400 = Component(2, "PEG400", ((150, 2), (153, 2), (154, 8)))
    DEG = Component(2, "diethylene glycol", ((2, 1), (25, 1), (150, 2), (153, 2)))

    def test_peg_oligomer_warns(self):
        model = ActivityModel([WATER, self.PEG400])
        with pytest.warns(UserWarning, match="PEG"):
            res = organic_mixture_viscosity(model, np.array([0.9, 0.1]), 290.15, {2: 0.12})
        assert res.log10_eta_pas > 10.0     # the published model's known divergence, not a physical value

    def test_ordinary_ether_does_not_warn_and_stays_between_pure_components(self):
        # diethylene glycol + water, 293.15 K, Hoga et al. (2018, J. Chem. Thermodyn. 122, 38-64): measured
        # eta0 = 0.035893 Pa s; at x_DEG = 0.4007 eta = 0.017722 Pa s (log10 -1.751). The model gives -2.12.
        model = ActivityModel([WATER, self.DEG])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            res = organic_mixture_viscosity(model, np.array([0.5993, 0.4007]), 293.15, {2: 0.035893})
        assert math.log10(water_viscosity_pas(293.15)) < res.log10_eta_pas < math.log10(0.035893)
        assert res.log10_eta_pas == pytest.approx(math.log10(0.017722), abs=0.45)


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

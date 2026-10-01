"""Activity-coefficient entry point.

Fortran counterpart: ModCalcActCoeff.f90 (AIOMFAC_calc, Gammas) plus the composition conversion of
AIOMFAC_inout / ModCompScaleConversion. Only the activity-coefficient core is in scope for the first stage;
the viscosity module (AIOMFAC-VISC) is deliberately excluded.

Species order everywhere is the one used by the Fortran arrays ``X`` / ``lnGaSR``:
neutral components (input order), then cations, then anions.

Automatically completes systems the Fortran SetSystem would complete (H+/HSO4-/SO4--/HCO3-/CO3--/OH-/CO2(aq)) and
solves the resulting bisulfate-only, bicarbonate-only, or joint bisulfate+bicarbonate dissociation equilibrium
(see dissociation.py / carbonate.py).

The AIOMFAC-VISC viscosity extension (Lilek and Zuend, 2022) is a separate module, ``viscosity.py`` -- not
excluded from this port, just layered on top of this one (it calls ``ActivityModel.evaluate`` for the ion
molal activities/activity coefficients it needs; see that module's docstring for scope).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .carbonate import gamma_co2_mr, solve_carb_sulf, solve_carbonate
from .completion import complete_components
from .composition import input_to_mass_frac, mass_frac_to_ion_molalities
from .dissociation import solve_bisulfate
from .io import Component
from .params import load_subgroup_params
from .lr import lr_terms
from .mr import build_mr_state, mr_terms
from .numerics import DTINY, LNHUGE, LOGVAL_THRESHOLD, safe_exp
from .sr import sr_terms
from .system import Mixture, build_mixture


@dataclass
class ActivityTerms:
    """Result at one composition point; the fields mirror the terms dumped by the instrumented Fortran."""
    T_K: float
    wtf: np.ndarray             # mass fractions of all independent components (neutrals, then electrolytes)
    x: np.ndarray               # species mole fractions X (dissociated basis): neutrals, cations, anions
    xn: np.ndarray              # electrolyte-free mole fractions of the neutral components
    smc: np.ndarray             # cation molalities [mol/kg solvent mixture]
    sma: np.ndarray             # anion molalities
    ln_gamma_lr: np.ndarray     # long-range term        per species (gnlrln | gclrln | galrln)
    ln_gamma_mr: np.ndarray     # middle-range term      per species (gnmrln | gcmrln | gamrln)
    ln_gamma_sr: np.ndarray     # short-range term       per species (gnsrln | gcsrln | gasrln)
    tmolal: float               # mole-fraction -> molality conversion term applied to the ions
    ln_gamma: np.ndarray        # total ln(gamma): neutrals on mole-fraction basis, ions on molality basis
    gamma_neutral: np.ndarray   # activity coefficients of the neutrals (0 where the component is absent)
    activity: np.ndarray        # (n_indcomp,) neutral activities, then electrolyte ion-activity products
    ionic_strength: float       # molality basis


class ActivityModel:
    """Activity-coefficient model for one mixture definition (parameters/system set up once)."""

    def __init__(self, components: Sequence[Component], *, assume_complete: bool = False):
        self._bisulfate = None
        self._carbonate = None
        self._carb_sulf = None
        comps = list(components)
        self._orig_index = tuple(range(len(comps)))    # maps evaluate()'s composition entries to comps positions
        auto_bisulfate = auto_carbonate = auto_carb_sulf = False
        if not assume_complete:
            r = complete_components(comps)
            if r.bicarbsyst and r.bisulfsyst:
                comps = list(r.components)
                auto_carb_sulf = True
                self._orig_index = self._invert_orig_index(r.orig_index)
            elif r.bicarbsyst:
                comps = list(r.components)
                auto_carbonate = True
                self._orig_index = self._invert_orig_index(r.orig_index)
            elif r.bisulfsyst:
                comps = list(r.components)          # adds H+/SO4--/HSO4- if the input did not have all three
                auto_bisulfate = True
                self._orig_index = self._invert_orig_index(r.orig_index)
            assume_complete = True                    # comps is now complete either way
        self.mixture: Mixture = build_mixture(comps, assume_complete=assume_complete)
        sr = self.mixture.sr
        # ionic charges, zero-padded to NGI like the Fortran arrays cationZ / anionZ
        ngi = self.mixture.ngi
        self.cation_z = np.zeros(ngi); self.cation_z[:sr.n_cation] = sr.cation_z
        self.anion_z = np.zeros(ngi); self.anion_z[:sr.n_anion] = sr.anion_z
        self._mr = build_mr_state(self.mixture)
        if auto_bisulfate and 205 in sr.cations and 248 in sr.anions and 261 in sr.anions:
            self._bisulfate = (self.mixture.cat_index[205], self.mixture.an_index[248], self.mixture.an_index[261])
        if auto_carbonate:
            idx_co2 = next((i for i in range(self.mixture.n_neutral)
                            if self.mixture.itab[i, 173 - 1] > 0), None)
            if idx_co2 is not None and 247 in sr.anions and 250 in sr.anions and 262 in sr.anions:
                self._carbonate = (self.mixture.cat_index[205], self.mixture.an_index[247],
                                   self.mixture.an_index[250], self.mixture.an_index[262], idx_co2)
        if auto_carb_sulf:
            idx_co2 = next((i for i in range(self.mixture.n_neutral)
                            if self.mixture.itab[i, 173 - 1] > 0), None)
            if (idx_co2 is not None and 247 in sr.anions and 250 in sr.anions and 262 in sr.anions
                    and 248 in sr.anions and 261 in sr.anions):
                idx_ca = self.mixture.cat_index.get(221)
                self._carb_sulf = (self.mixture.cat_index[205], self.mixture.an_index[247],
                                   self.mixture.an_index[250], self.mixture.an_index[262],
                                   self.mixture.an_index[248], self.mixture.an_index[261], idx_co2, idx_ca)

    @staticmethod
    def _invert_orig_index(orig_index):
        """orig_index[completed_pos] = original_pos (or -1) -> inverse: original_pos -> completed_pos."""
        n = sum(1 for v in orig_index if v >= 0)
        inv = [-1] * n
        for completed_pos, orig_pos in enumerate(orig_index):
            if orig_pos >= 0:
                inv[orig_pos] = completed_pos
        return tuple(inv)

    def _x_from_molalities(self, xn, smc, sma):
        """species mole fractions X (Gammas()): neutrals via XN/mean_mw, then ion molalities, all /sum_molal."""
        m = self.mixture
        nc, na = m.sr.n_cation, m.sr.n_anion
        mean_mw = float(np.sum(m.mmass[:m.n_neutral] * xn))
        m_neutral = xn / mean_mw
        sum_molal = float(np.sum(m_neutral)) + float(np.sum(sma[:na])) + float(np.sum(smc[:nc]))
        return np.concatenate([m_neutral, smc[:nc], sma[:na]]) / sum_molal

    # ------------------------------------------------------------------------------------------------------
    def lr_mr_sr(self, T_K: float, smc, sma, xn, x):
        """The three range contributions for given molalities and mole fractions (used by tests too)."""
        m = self.mixture
        sr = m.sr
        nn, nc, na = m.n_neutral, sr.n_cation, sr.n_anion
        mean_mw = float(np.sum(m.mmass[:nn] * xn[:nn]))
        sum_ion_m = float(np.sum(sma[:na]) + np.sum(smc[:nc]))
        lr = lr_terms(m.mmass[:nn], self.cation_z, self.anion_z, smc, sma, T_K)
        mr = mr_terms(m, self._mr, self.cation_z, self.anion_z, smc, sma, xn, lr.ionic_strength, sum_ion_m, mean_mw)
        s = sr_terms(sr, T_K, x, xn)
        return lr, mr, s

    # ------------------------------------------------------------------------------------------------------
    def evaluate(self, composition, T_K: float, basis: str = "mass") -> ActivityTerms:
        """ln(activity coefficients) at one composition point.

        ``composition``: fractions of the input components in input order (component 1 = 1 - sum(others)),
        ``basis``: "mass" or "mole" (basis of ``composition``; electrolytes count as undissociated components).
        """
        if basis not in ("mass", "mole"):
            raise ValueError("basis must be 'mass' or 'mole'")
        m = self.mixture
        sr = m.sr
        nn, nc, na = m.n_neutral, sr.n_cation, sr.n_anion
        composition = np.asarray(composition, dtype=float)
        if composition.size > len(self._orig_index):
            raise ValueError("more composition entries than original input components")
        inputconc = np.zeros(m.n_indcomp)
        for orig_i, val in enumerate(composition):
            pos = self._orig_index[orig_i]
            if pos < 0:
                raise ValueError(f"input component {orig_i + 1} was removed/merged during completion")
            inputconc[pos] = val
        wtf = input_to_mass_frac(inputconc, m.mmass, nn, mole_basis=(basis == "mole"))

        smc, sma = mass_frac_to_ion_molalities(wtf, m)
        sum_ion_m = float(np.sum(sma[:na]) + np.sum(smc[:nc]))

        # ---- Gammas(): mole fractions on the dissociated basis
        wbym = wtf[:nn] / m.mmass[:nn]
        xn = wbym / float(np.sum(wbym))

        if self._bisulfate is not None:
            idx_h, idx_hso4, idx_so4 = self._bisulfate
            r = solve_bisulfate(self, T_K, xn, smc, sma, idx_h, idx_hso4, idx_so4)
            smc[idx_h], sma[idx_hso4], sma[idx_so4] = r.m_h, r.m_hso4, r.m_so4
            sum_ion_m = float(np.sum(sma[:na])) + float(np.sum(smc[:nc]))

        co2_idx = None
        if self._carb_sulf is not None:
            idx_h, idx_oh, idx_hco3, idx_carb, idx_hso4, idx_so4, co2_idx, idx_ca = self._carb_sulf
            jr = solve_carb_sulf(self, T_K, wtf, smc, sma, idx_h, idx_oh, idx_hco3, idx_carb, idx_hso4, idx_so4,
                                 co2_idx, idx_ca)
            sg = load_subgroup_params()
            ngi = self.mixture.ngi
            smw_c = np.zeros(ngi); smw_c[:nc] = [sg.SMWC[c - 201] for c in sr.cations]
            smw_a = np.zeros(ngi); smw_a[:na] = [sg.SMWA[a - 241] for a in sr.anions]
            smw_c *= 1.0e-3; smw_a *= 1.0e-3
            n_cat, n_an = smc.copy(), sma.copy()
            n_cat[idx_h] = jr.n_h
            if idx_ca is not None and jr.n_ca is not None:
                n_cat[idx_ca] = jr.n_ca
            n_an[idx_hco3], n_an[idx_carb], n_an[idx_oh] = jr.n_hco3, jr.n_carb, jr.n_oh
            n_an[idx_hso4], n_an[idx_so4] = jr.n_hso4, jr.n_sulf
            solvmass = float(np.sum(jr.mol_neutral * m.mmass[:nn]))
            totmass = solvmass + float(np.sum(n_cat * smw_c)) + float(np.sum(n_an * smw_a))
            wtf = wtf.copy(); wtf[:nn] = jr.mol_neutral * m.mmass[:nn] / totmass
            xn = jr.mol_neutral / float(np.sum(jr.mol_neutral))
            smc, sma = n_cat / solvmass, n_an / solvmass
            sum_ion_m = float(np.sum(sma[:na])) + float(np.sum(smc[:nc]))
        elif self._carbonate is not None:
            idx_h, idx_oh, idx_hco3, idx_carb, co2_idx = self._carbonate
            cr = solve_carbonate(self, T_K, wtf, smc, sma, idx_h, idx_oh, idx_hco3, idx_carb, co2_idx)
            sg = load_subgroup_params()
            ngi = self.mixture.ngi
            smw_c = np.zeros(ngi); smw_c[:nc] = [sg.SMWC[c - 201] for c in sr.cations]
            smw_a = np.zeros(ngi); smw_a[:na] = [sg.SMWA[a - 241] for a in sr.anions]
            smw_c *= 1.0e-3; smw_a *= 1.0e-3
            # non-reacting ions keep their molar amount (computed on the ORIGINAL 1 kg-solvent basis); the
            # reacting ones (H+, OH-, HCO3-, CO3--) get the solver's molar amounts instead:
            n_cat, n_an = smc.copy(), sma.copy()
            n_cat[idx_h] = cr.n_h
            n_an[idx_hco3], n_an[idx_carb], n_an[idx_oh] = cr.n_hco3, cr.n_carb, cr.n_oh
            solvmass = float(np.sum(cr.mol_neutral * m.mmass[:nn]))
            totmass = solvmass + float(np.sum(n_cat * smw_c)) + float(np.sum(n_an * smw_a))
            wtf = wtf.copy(); wtf[:nn] = cr.mol_neutral * m.mmass[:nn] / totmass
            xn = cr.mol_neutral / float(np.sum(cr.mol_neutral))
            smc, sma = n_cat / solvmass, n_an / solvmass
            sum_ion_m = float(np.sum(sma[:na])) + float(np.sum(smc[:nc]))

        x = self._x_from_molalities(xn, smc, sma)

        lr, mr, s = self.lr_mr_sr(T_K, smc, sma, xn, x)
        lnlr = np.concatenate([lr.ln_gamma_neutral, lr.ln_gamma_cation[:nc], lr.ln_gamma_anion[:na]])
        lnmr = np.concatenate([mr.ln_gamma_neutral, mr.ln_gamma_cation[:nc], mr.ln_gamma_anion[:na]])
        lnsr = s.ln_gamma_sr

        if co2_idx is not None:                          # GammaCO2(): overrides CO2(aq)'s own ln(gamma) terms
            lnmr[co2_idx] = gamma_co2_mr(m, smc, sma)
            lnsr[co2_idx] = 0.0
            lnlr[co2_idx] = 0.0

        # ---- AIOMFAC_calc(): assemble the activity coefficients
        ln_n = np.full(nn, -9999.9); gam_n = np.zeros(nn)
        for i in range(nn):
            if wtf[i] > DTINY:
                ln_n[i] = lnmr[i] + lnsr[i] + lnlr[i]
                gam_n[i] = safe_exp(ln_n[i], LOGVAL_THRESHOLD)
        ln_c = np.full(nc, -9999.9); gam_c = np.zeros(nc)
        for i in range(nc):
            if smc[i] > DTINY:
                ln_c[i] = lnmr[nn + i] + lnsr[nn + i] + lnlr[nn + i] - mr.tmolal
                gam_c[i] = safe_exp(ln_c[i], LOGVAL_THRESHOLD)
        ln_a = np.full(na, -9999.9); gam_a = np.zeros(na)
        for i in range(na):
            if sma[i] > DTINY:
                ln_a[i] = lnmr[nn + nc + i] + lnsr[nn + nc + i] + lnlr[nn + nc + i] - mr.tmolal
                gam_a[i] = safe_exp(ln_a[i], LOGVAL_THRESHOLD)

        activity = np.zeros(m.n_indcomp)
        activity[:nn] = gam_n * x[:nn]
        for k in range(m.n_electrol):                       # molal ion activity products of the electrolytes
            i = m.cat_index[int(m.elect_comps[k, 0])]
            j = m.an_index[int(m.elect_comps[k, 1])]
            if smc[i] > 0.0 and sma[j] > 0.0 and gam_c[i] > 0.0 and gam_a[j] > 0.0:
                nuc, nua = float(m.elect_nues[k, 0]), float(m.elect_nues[k, 1])
                t1 = nuc * ln_c[i] + nua * ln_a[j] + nuc * math.log(smc[i]) + nua * math.log(sma[j])
                activity[nn + k] = safe_exp(t1, LNHUGE)

        return ActivityTerms(T_K=float(T_K), wtf=wtf, x=x, xn=xn, smc=smc, sma=sma, ln_gamma_lr=lnlr,
                             ln_gamma_mr=lnmr, ln_gamma_sr=lnsr, tmolal=mr.tmolal,
                             ln_gamma=np.concatenate([ln_n, ln_c, ln_a]), gamma_neutral=gam_n, activity=activity,
                             ionic_strength=lr.ionic_strength)


def activity_coefficients(components: Sequence[Component], composition, T_K: float,
                          basis: str = "mass", *, assume_complete: bool = False) -> ActivityTerms:
    """Convenience wrapper: set up the mixture and evaluate one point (use ``ActivityModel`` to reuse the setup)."""
    return ActivityModel(components, assume_complete=assume_complete).evaluate(composition, T_K, basis)

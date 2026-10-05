"""Volatile inorganic gases (NH3, HNO3, HCl) for the SLE solver: Henry-type equilibrium constants on the molal scale.

Every gas is described by the reaction  "gas  <=>  aqueous ions"  and by a constant K(T) such that

    prod_i (m_i gamma_i)^nu_i  =  K(T) * p_gas                (p in atm; infinite-dilution standard states)

``Gas.ions`` lists the aqueous (stoichiometric) ions that DISAPPEAR when one mole of the gas FORMS from the solution,
exactly like the ``ions`` of a solid (a negative entry means the ion is produced):

    HNO3(g) <=> H+ + NO3-        ions = {H+: 1, NO3-: 1}
    HCl(g)  <=> H+ + Cl-         ions = {H+: 1, Cl-: 1}
    CO2(g) + H2O <=> CO3-- + 2 H+   ions = {CO3--: 1, H+: 2}, h = -1   (carbonate systems; H+ = proton excess)
    NH3(g) + H+ <=> NH4+         ions = {NH4+: 1, H+: -1}        (forming NH3(g) turns NH4+ into H+)

Sources (data quality in parentheses)
-------------------------------------
NH3   Clegg, Brimblecombe & Wexler (1998) J. Phys. Chem. A 102:2137, eqs 11-12 (A):
          ln xK'H = 25.393 + 10373.6 (1/T - 1/Tr) + 4.131 (Tr/T - (1 + ln(Tr/T))),   Tr = 298.15 K,   xK'H in atm^-1
      NOTE: the extracted text of eq 12 reads (1/Tr - 1/T); that contradicts the paper's own dH = -86.25 kJ/mol
      (10373.6 = -dH/R: the dissolution is exothermic, K'H must FALL with T) and, with it, the paper's
      dH(Kp(NH4NO3)) = 184.2 kJ/mol (which this module reproduces to 1 % with the sign used here).  Verify the
      Sign confirmed by the project owner (co-author of the AIOMFAC-related work), 2026-10-05.
      (ratio form x_NH4/(x_H p) = m_NH4/(m_H p): identical on the mole-fraction and molal scales.)
HNO3  same paper, eq 15 (A):  ln xKH = 385.972199 - 3020.3522/T - 71.001998 ln T + 0.131442311 T - 0.420928363e-4 T^2,
      xKH = 853.1 atm^-1 at 298.15 K; molal scale: K = xKH (1000/M_w)^2 = 2.63e6 mol^2 kg^-2 atm^-1.
HCl   Clegg & Brimblecombe (1998 paper 3, J. Phys. Chem. A 102:2155): xKH = 662.1 atm^-1 at 298.15 K (A at 298 K).
      Its T-dependence is not given in the papers at hand: the enthalpy and heat capacity of HCl(g) -> H+ + Cl-
      are taken from NBS/CODATA values (dH = -74.85 kJ/mol, dCp = -165.5 J/mol/K; quality C for T != 298.15 K).

Consistency check (used in the tests): Kp(NH4NO3) = K_s(NH4NO3) / (K'H K_H) = 4.36e-17 atm^2 at 298.15 K (Clegg et al.).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

R_GAS = 8.314462618
T_REF = 298.15
_L_W = math.log(1000.0 / 18.01528)          # ln(1000/M_w): mole fraction -> molality, per ion


@dataclass(frozen=True)
class Gas:
    key: str
    formula: str
    ions: dict
    quality: str
    source: str
    h: int = 0          # water consumed on formation of the aqueous ions: CO2(g) + H2O -> CO3-- + 2 H+  has h = -1

    def ln_k(self, T: float) -> float:
        return _LN_K[self.key](T)


def _ln_khp_nh3(T: float) -> float:
    return 25.393 + 10373.6 * (1.0 / T - 1.0 / T_REF) + 4.131 * (T_REF / T - (1.0 + math.log(T_REF / T)))


def _ln_kh_hno3(T: float) -> float:
    lnx = (385.972199 - 3020.3522 / T - 71.001998 * math.log(T) + 0.131442311 * T - 0.420928363e-4 * T * T)
    return lnx + 2.0 * _L_W


def _ln_kh_hcl(T: float) -> float:
    dH, dCp = -74.85e3, -165.5
    return (math.log(662.1) + 2.0 * _L_W - dH / R_GAS * (1.0 / T - 1.0 / T_REF)
            + dCp / R_GAS * (T_REF / T - 1.0 + math.log(T / T_REF)))


def _ln_k_co2(T: float) -> float:
    """CO2(g) + H2O <=> CO3-- + 2 H+ :  K = K_H(CO2) K1 K2  (a_CO3 a_H^2 / a_w = K p).  K_H from PHREEQC pitzer.dat
    (analytical expression, mol kg^-1 atm^-1; Plummer & Busenberg 1982); K1, K2 are the constants used by the AIOMFAC
    carbonate speciation (:mod:`aiomfac_py.carbonate`) so the aqueous speciation and the gas are consistent."""
    from .carbonate import ln_k1_hco3_at_t, ln_k2_hco3_at_t
    log10_kh = (10.5624 - 2.3547e-2 * T - 3972.8 / T + 5.8746e5 / (T * T) + 1.9194e-5 * T * T)
    return log10_kh * math.log(10.0) + ln_k1_hco3_at_t(T) + ln_k2_hco3_at_t(T)


_LN_K = {"NH3": _ln_khp_nh3, "HNO3": _ln_kh_hno3, "HCl": _ln_kh_hcl, "CO2": _ln_k_co2}

GASES: dict[str, Gas] = {
    "NH3": Gas("NH3", "NH3", {"NH4+": 1, "H+": -1}, "A", "Clegg et al. 1998 eq 11-12"),
    "HNO3": Gas("HNO3", "HNO3", {"H+": 1, "NO3-": 1}, "A", "Clegg et al. 1998 eq 15"),
    "HCl": Gas("HCl", "HCl", {"H+": 1, "Cl-": 1}, "B", "Clegg et al. 1998 (paper 3) 298 K; T-dependence NBS"),
    "CO2": Gas("CO2", "CO2", {"CO3--": 1, "H+": 2}, "A", "PHREEQC pitzer.dat K_H(T); K1,K2 of the AIOMFAC carbonate model", h=-1),
}

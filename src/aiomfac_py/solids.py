"""Solid-phase (salt / hydrate) thermodynamic data for solid-liquid equilibrium (SLE) calculations.

Companion of :mod:`aiomfac_py.sle`. Every solid is described by its dissolution reaction

    solid  <=>  sum_i nu_i ion_i  +  h H2O                              (h = number of hydrate waters)

and by its thermodynamic solubility product on the molal scale, *including the water activity for hydrates*:

    K(T) = prod_i (m_i * gamma_i)^nu_i * a_w^h                         (infinite-dilution standard states,
                                                                         the same ones AIOMFAC uses for ions)

The fixed-RH solver needs ``ln K`` and the stoichiometry only (the solid itself has constant chemical potential,
paper eq. (5) of Amundson et al. 2006, JOTA 128, 469).

Temperature dependence ``ln K(T)``
----------------------------------
Two kinds of sources are used (see ``Solid.kspec``):

``"phreeqc"``  six-term analytical expression  log10 K = A1 + A2 T + A3/T + A4 log10 T + A5/T^2 + A6 T^2, taken
               from the PHREEQC ``pitzer.dat`` database (USGS; Plummer et al. 1988, Harvie et al. 1984, Pabalan &
               Pitzer 1987, Linke & Seidell tables; the "ref. 3" entries of that file). Available for the
               Na/K/Mg/Ca - Cl/SO4 phases only.
``"vanthoff"`` ln K(T) = ln K0 - dH/R (1/T - 1/T0) + dCp/R (T0/T - 1 + ln(T/T0)), with the enthalpy and heat
               capacity of dissolution computed from NBS/CODATA-type formation enthalpies and ion/solid heat
               capacities (needed for the NH4+ and NO3- salts, which PHREEQC's databases do not contain).

Anchoring to AIOMFAC
--------------------
AIOMFAC's own activity coefficients at saturation are not exactly those of the Pitzer fits the databases rely on
(differences of 0.0-0.3 in log10 K were found for the salts tested; see ``tools/calibrate_solids.py``).  To make the
model reproduce the *experimental* solubility at 298.15 K, ``ln K(T)`` is shifted by a constant

    offset = ln[ IAP_AIOMFAC(m_sat,exp, 298.15 K) * a_w^h ] - ln K_ref(298.15 K)

(``Solid.anchor_m`` is the experimental saturation molality, ``Solid.offset`` the resulting shift).  The T-dependence
then comes from the thermodynamic reference only.  Solids for which no binary saturation molality is available
(double salts) are not anchored (offset 0).  Use ``Solid.ln_k(T, anchored=False)`` for the pure database value.

Data quality flags (``Solid.quality``)
    A  database expression (PHREEQC) + well-known solubility;  anchor checked against Pitzer K to <0.2 log10
    B  van't Hoff from tabulated enthalpies/heat capacities + well-known solubility
    C  estimated / incomplete data; use with care (see ``Solid.note``)

IMPORTANT: the numerical values collected here were assembled from the sources named per entry; all entries marked
quality B/C and every ``anchor_m`` should be re-checked against the primary literature before any publication use.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

R_GAS = 8.314462618      # J / (mol K)
T_REF = 298.15

# ion key -> (AIOMFAC subgroup number, charge)
ION_REGISTRY: dict[str, tuple[int, int]] = {
    "Li+": (201, 1), "Na+": (202, 1), "K+": (203, 1), "NH4+": (204, 1), "H+": (205, 1),
    "Ca++": (221, 2), "Mg++": (223, 2),
    "F-": (241, -1), "Cl-": (242, -1), "Br-": (243, -1), "I-": (244, -1), "NO3-": (245, -1),
    "HSO4-": (248, -1), "SO4--": (261, -2), "CO3--": (262, -2), "HCO3-": (250, -1), "OH-": (247, -1),
}


@dataclass(frozen=True)
class KSpec:
    kind: str                         # "phreeqc" | "vanthoff"
    coeffs: tuple[float, ...] = ()    # phreeqc: A1..A6
    ln_k0: float = 0.0                # vanthoff: ln K at T_REF (reference value, before anchoring)
    dH: float = 0.0                   # J/mol (dissolution, per formula unit incl. hydrate water)
    dCp: float = 0.0                  # J/(mol K)
    T_min: float = 253.15
    T_max: float = 373.15

    def ln_k(self, T: float) -> float:
        if self.kind == "phreeqc":
            a = list(self.coeffs) + [0.0] * (6 - len(self.coeffs))
            return math.log(10.0) * (a[0] + a[1] * T + a[2] / T + a[3] * math.log10(T) + a[4] / (T * T)
                                     + a[5] * T * T)
        if self.kind == "vanthoff":
            return (self.ln_k0 - self.dH / R_GAS * (1.0 / T - 1.0 / T_REF)
                    + self.dCp / R_GAS * (T_REF / T - 1.0 + math.log(T / T_REF)))
        raise ValueError(self.kind)


@dataclass(frozen=True)
class KFit:
    """Model-consistent ln K(T): K0, dH_eff, dCp_eff fitted so that AIOMFAC reproduces the experimental binary
    solubility curve (tools/fit_solids.py).  Outside [T_lo, T_hi] ln K is extrapolated linearly in 1/T with the
    enthalpy at the window edge (no heat-capacity blow-up)."""
    ln_k0: float
    dH: float
    dCp: float
    T_lo: float
    T_hi: float

    def _core(self, T: float) -> float:
        return (self.ln_k0 - self.dH / R_GAS * (1.0 / T - 1.0 / T_REF)
                + self.dCp / R_GAS * (T_REF / T - 1.0 + math.log(T / T_REF)))

    def ln_k(self, T: float) -> float:
        if self.T_lo <= T <= self.T_hi:
            return self._core(T)
        Tb = self.T_lo if T < self.T_lo else self.T_hi
        dh_b = self.dH + self.dCp * (Tb - T_REF)
        return self._core(Tb) - dh_b / R_GAS * (1.0 / T - 1.0 / Tb)


@dataclass(frozen=True)
class Solid:
    key: str
    formula: str
    ions: dict                        # ion key -> stoichiometric coefficient nu_i
    h: int                            # hydrate water number
    kspec: KSpec
    anchor_m: float | None = None     # experimental saturation molality of the formula unit at 298.15 K (binary)
    offset: float = 0.0               # ln-unit anchor shift (see module docstring)
    quality: str = "B"
    source: str = ""
    note: str = ""
    kfit: KFit | None = None          # AIOMFAC-consistent fitted K(T) (default when available)
    n_oh: int = 0                     # hydroxide solids: OH- is carried as -H+ ("proton excess"); K is converted with Kw

    @property
    def h_eff(self) -> int:
        """water exponent in the saturation condition: hydrate water plus the water of the OH- -> -H+ substitution"""
        return self.h + self.n_oh

    def ln_k_ref(self, T: float) -> float:
        return self.kspec.ln_k(T)

    def ln_k(self, T: float, mode: str | None = None) -> float:
        """ln K for the dissolution written with the solid's own ``ions`` (see ``_ln_k_core``); for hydroxides
        (``n_oh``) OH- is replaced by -H+ using a_H a_OH = Kw a_w, so  ln K -> ln K - n_oh ln Kw(T)."""
        val = self._ln_k_core(T, mode)
        if self.n_oh:
            from .carbonate import ln_kw_at_t
            val -= self.n_oh * ln_kw_at_t(T)
        return val

    def _ln_k_core(self, T: float, mode: str | None = None) -> float:
        """ln of the molal solubility product incl. a_w^h (so that  sum nu ln a_i + h ln a_w = ln K  at saturation).

        ``mode``:
          ``"fitted"``   K(T) fitted to experimental solubility curves *with AIOMFAC* (default when a fit exists;
                         absorbs AIOMFAC's limited temperature dependence of gamma at saturation)
          ``"anchored"`` literature T-dependence (database / van't Hoff), level anchored to AIOMFAC at 298.15 K
          ``"thermo"``   pure literature value (no AIOMFAC anchoring; meaningful only for PHREEQC-type entries)
        """
        if mode is None:
            mode = "fitted" if self.kfit is not None else "anchored"
        if mode == "fitted":
            if self.kfit is None:
                mode = "anchored"
            else:
                return self.kfit.ln_k(T)
        if mode == "anchored":
            return self.kspec.ln_k(T) + self.offset
        if mode == "thermo":
            return self.kspec.ln_k(T)
        raise ValueError(f"unknown mode {mode!r}")

    def ln_k_eff(self, T: float, ln_rh: float, mode: str | None = None) -> float:
        """ln K with the hydrate water taken out at fixed water activity:  sum nu ln a_i = ln K - h ln(a_w)."""
        return self.ln_k(T, mode) - self.h_eff * ln_rh

    @property
    def charge_balance(self) -> int:
        return sum(nu * ION_REGISTRY[i][1] for i, nu in self.ions.items())

    def in_range(self, T: float) -> bool:
        return self.kspec.T_min <= T <= self.kspec.T_max


def _ph(*c, **kw):
    return KSpec("phreeqc", coeffs=tuple(c), **kw)


def _vh(ln_k0, dH, dCp, **kw):
    return KSpec("vanthoff", ln_k0=ln_k0, dH=dH, dCp=dCp, **kw)


# --------------------------------------------------------------------------------------------------------
# The database.  ``offset`` values are produced by ``tools/calibrate_solids.py`` (AIOMFAC, 298.15 K).
# --------------------------------------------------------------------------------------------------------
_PH_REF = "PHREEQC pitzer.dat (USGS), analytical expression 'ref. 3' (Linke & Seidell-based fits)"

_DB: list[Solid] = [
    # ---- Na / K / NH4 chlorides, nitrates --------------------------------------------------------------
    Solid("halite", "NaCl", {"Na+": 1, "Cl-": 1}, 0,
          _ph(159.605, 8.4294e-2, -3975.6, -66.857, 0, -4.9364e-5), anchor_m=6.146, quality="A", source=_PH_REF),
    Solid("sylvite", "KCl", {"K+": 1, "Cl-": 1}, 0,
          _ph(-50.571, 9.8815e-2, 1.3135e4, 0, -1.3754e6, -7.393e-5), anchor_m=4.803, quality="A", source=_PH_REF),
    Solid("sal_ammoniac", "NH4Cl", {"NH4+": 1, "Cl-": 1}, 0,
          _vh(0.0, 14.78e3, -140.6), anchor_m=7.39, quality="B",
          source="dH: heat of solution at infinite dilution 14.8 kJ/mol (NBS: dfH NH4Cl(s) -314.43, NH4+ -132.51, "
                 "Cl- -167.16); dCp from Cp(NH4+) 79.9, Cp(Cl-) -136.4, Cp(s) 84.1 J/mol/K"),
    Solid("nitratine", "NaNO3", {"Na+": 1, "NO3-": 1}, 0,
          _vh(0.0, 20.4e3, -133.1), anchor_m=10.8, quality="B",
          source="dH 20.4 kJ/mol (NBS: dfH NaNO3(s) -467.85, Na+ -240.12, NO3- -207.36); dCp from Cp(Na+) 46.4, "
                 "Cp(NO3-) -86.6, Cp(s) 92.9"),
    Solid("niter", "KNO3", {"K+": 1, "NO3-": 1}, 0,
          _vh(0.0, 34.9e3, -161.2), anchor_m=3.75, quality="B",
          source="dH 34.9 kJ/mol (NBS: dfH KNO3(s) -494.63, K+ -252.38, NO3- -207.36); dCp from Cp(K+) 21.8, "
                 "Cp(NO3-) -86.6, Cp(s) 96.4"),
    Solid("ammonium_nitrate", "NH4NO3", {"NH4+": 1, "NO3-": 1}, 0,
          _vh(0.0, 25.7e3, -146.0), anchor_m=26.0, quality="B",
          source="dH 25.7 kJ/mol (NBS: dfH NH4NO3(s) -365.56); dCp from Cp(NH4+) 79.9, Cp(NO3-) -86.6, Cp(s) 139.3",
          note="solid-solid phase transitions (IV/III at 305.4 K, ...) are ignored"),
    # ---- sulfates ---------------------------------------------------------------------------------------
    Solid("thenardite", "Na2SO4", {"Na+": 2, "SO4--": 1}, 0,
          _ph(57.185, 8.6024e-2, 0, -30.8341, 0, -7.6905e-5), anchor_m=None, quality="A", source=_PH_REF,
          note="anhydrous; stable above 305.5 K (32.4 C) only; below that mirabilite is the stable phase"),
    Solid("mirabilite", "Na2SO4.10H2O", {"Na+": 2, "SO4--": 1}, 10,
          _ph(-301.9326, -0.16232, 0, 141.078), anchor_m=1.97, quality="A", source=_PH_REF),
    Solid("arcanite", "K2SO4", {"K+": 2, "SO4--": 1}, 0,
          _ph(674.142, 0.30423, -18037, -280.236, 0, -1.44055e-4), anchor_m=0.691, quality="A", source=_PH_REF),
    Solid("ammonium_sulfate", "(NH4)2SO4", {"NH4+": 2, "SO4--": 1}, 0,
          _vh(0.0, 6.6e3, -320.7), anchor_m=5.77, quality="B",
          source="dH 6.6 kJ/mol (NBS: dfH (NH4)2SO4(s) -1180.85, NH4+ -132.51, SO4-- -909.27); dCp from Cp(NH4+) "
                 "79.9, Cp(SO4--) -293, Cp(s) 187.5"),
    # ---- Mg ---------------------------------------------------------------------------------------------
    Solid("bischofite", "MgCl2.6H2O", {"Mg++": 1, "Cl-": 2}, 6,
          _ph(7.526, -1.114e-2, 115.7), anchor_m=5.8101, quality="A", source=_PH_REF),
    Solid("MgCl2_4H2O", "MgCl2.4H2O", {"Mg++": 1, "Cl-": 2}, 4,
          _ph(12.98, -2.013e-2), anchor_m=None, quality="C", source=_PH_REF,
          note="only stable at elevated T / very low aw"),
    Solid("MgCl2_2H2O", "MgCl2.2H2O", {"Mg++": 1, "Cl-": 2}, 2,
          _ph(-10.273, 0, 7.403e3), anchor_m=None, quality="C", source=_PH_REF,
          note="only stable at elevated T / very low aw"),
    Solid("epsomite", "MgSO4.7H2O", {"Mg++": 1, "SO4--": 1}, 7,
          _ph(4.479, -6.99e-3, -1.265e3), anchor_m=3.0, quality="A", source=_PH_REF),
    Solid("hexahydrite", "MgSO4.6H2O", {"Mg++": 1, "SO4--": 1}, 6,
          _ph(-0.733, -2.80e-3, -8.57e-3), anchor_m=None, quality="B", source=_PH_REF),
    Solid("kieserite", "MgSO4.H2O", {"Mg++": 1, "SO4--": 1}, 1,
          _ph(47.24, -0.12077, -5.356e3, 0, 0, 7.272e-5), anchor_m=None, quality="B", source=_PH_REF),
    Solid("Mg_nitrate_6H2O", "Mg(NO3)2.6H2O", {"Mg++": 1, "NO3-": 2}, 6,
          _vh(0.0, 16.8e3, -200.0), anchor_m=None, quality="C",
          source="dH from NBS dfH Mg(NO3)2.6H2O(s) -2613.3, Mg++ -466.85, NO3- -207.36, H2O -285.83",
          note="dCp is an ESTIMATE; no anchor (solubility not verified)"),
    # ---- Ca ---------------------------------------------------------------------------------------------
    Solid("gypsum", "CaSO4.2H2O", {"Ca++": 1, "SO4--": 1}, 2,
          _ph(82.381, 0, -3804.5, -29.9952), anchor_m=None, quality="A", source=_PH_REF),
    Solid("anhydrite", "CaSO4", {"Ca++": 1, "SO4--": 1}, 0,
          _ph(5.009, -2.21e-2, -796.4), anchor_m=None, quality="A", source=_PH_REF),
    Solid("antarcticite", "CaCl2.6H2O", {"Ca++": 1, "Cl-": 2}, 6,
          _vh(0.0, 15.8e3, -250.0), anchor_m=7.4, quality="C",
          source="dH from NBS dfH CaCl2.6H2O(s) -2607.9, Ca++ -542.83, Cl- -167.16",
          note="dCp is an ESTIMATE; K0 anchored to AIOMFAC at 7.4 m (verify); converts to the 4-hydrate above ~302 K"),
    Solid("Ca_nitrate_4H2O", "Ca(NO3)2.4H2O", {"Ca++": 1, "NO3-": 2}, 4,
          _vh(0.0, 31.4e3, -200.0), anchor_m=8.675, quality="C",
          source="dH from NBS dfH Ca(NO3)2.4H2O(s) -2132.3, Ca++ -542.83, NO3- -207.36; saturation 8.675 m "
                 "(Apelblat & Korin, quoted in AIOMFAC ModMRpart.f90)",
          note="dCp is an ESTIMATE"),
    # ---- acid sulfates (H+ and SO4-- are stoichiometric ions; HSO4- is speciated by the aqueous-phase model) ----
    # ln K(molal) = ln xK(Clegg) + n_ions ln(1000/18.01528): the mole-fraction constants of the E-AIM papers refer to the
    # dissolution into the FREE ions (NH4+, H+, SO4-- ...; HSO4- <-> H+ + SO4-- is a separate equilibrium).
    Solid("ammonium_bisulfate", "NH4HSO4", {"NH4+": 1, "H+": 1, "SO4--": 1}, 0,
          _vh(0.6416, -15.17e3, -242.3, T_min=273.15, T_max=323.15), anchor_m=None, quality="B",
          source="Clegg, Brimblecombe & Wexler (1998) J. Phys. Chem. A 102, 2137, Table 2: ln(xK_s)=-11.408 at 298.15 K, "
                 "dH=-15.17 kJ/mol, dCp=-242.3 J/mol/K (free ions NH4+ + H+ + SO4--)",
          note="K from the E-AIM thermodynamic model; AIOMFAC-consistent without any anchor (DRH check in "
               "tools/calibrate_acid_solids.py); AIOMFAC's own gamma(T) error is not absorbed"),
    Solid("letovicite", "(NH4)3H(SO4)2", {"NH4+": 3, "H+": 1, "SO4--": 2}, 0,
          _vh(-1.9078, -5.32e3, -630.4, T_min=273.15, T_max=323.15), anchor_m=None, quality="B",
          source="Clegg, Brimblecombe & Wexler (1998) J. Phys. Chem. A 102, 2137, Table 2: ln(xK_s)=-26.007 at 298.15 K, "
                 "dH=-5.32 kJ/mol, dCp=-630.4 J/mol/K (free ions)",
          note="the large negative dCp makes the extrapolation outside ~0-50 C unreliable"),
    Solid("sodium_bisulfate_hydrate", "NaHSO4.H2O", {"Na+": 1, "H+": 1, "SO4--": 1}, 1,
          _vh(-0.2404, 0.0, 0.0, T_min=288.15, T_max=308.15), anchor_m=None, quality="B",
          source="Clegg, Brimblecombe & Wexler (1998) J. Phys. Chem. A 102, 2155, Table 3: ln(xK_s)=-12.29 at 298.15 K",
          note="298.15 K only: no enthalpy given in the paper, dH = dCp = 0 assumed (K constant)"),
    Solid("sodium_bisulfate", "NaHSO4", {"Na+": 1, "H+": 1, "SO4--": 1}, 0,
          _vh(1.3996, 0.0, 0.0, T_min=288.15, T_max=308.15), anchor_m=None, quality="B",
          source="Clegg et al. (1998) J. Phys. Chem. A 102, 2155, Table 3: ln(xK_s)=-10.65 at 298.15 K",
          note="298.15 K only (dH = dCp = 0 assumed)"),
    Solid("trisodium_hydrogen_sulfate", "Na3H(SO4)2", {"Na+": 3, "H+": 1, "SO4--": 2}, 0,
          _vh(-1.6708, 0.0, 0.0, T_min=288.15, T_max=308.15), anchor_m=None, quality="B",
          source="Clegg et al. (1998) J. Phys. Chem. A 102, 2155, Table 3: ln(xK_s)=-25.77 at 298.15 K",
          note="298.15 K only (dH = dCp = 0 assumed)"),
    Solid("NaH3_SO4_2_hydrate", "NaH3(SO4)2.H2O", {"Na+": 1, "H+": 3, "SO4--": 2}, 1,
          _vh(11.0792, 0.0, 0.0, T_min=288.15, T_max=308.15), anchor_m=None, quality="C",
          source="Clegg et al. (1998) J. Phys. Chem. A 102, 2155, Table 3: ln(xK_s)=-13.02 at 298.15 K (the paper calls "
                 "this value tentative)",
          note="298.15 K only; strongly acidic solutions (> 21 mol/kg H2SO4 equivalent)"),
    # ---- carbonates (PHREEQC pitzer.dat 298 K constants; stoichiometric ions CO3--/H+; not anchored) -------
    Solid("natron", "Na2CO3.10H2O", {"Na+": 2, "CO3--": 1}, 10,
          _vh(-0.825 * math.log(10.0), 65.3e3, 0.0, T_min=273.15, T_max=305.15), quality="B",
          source="log K = -0.825 (PHREEQC pitzer.dat); dH = +65.3 kJ/mol from NBS formation enthalpies (Na2CO3.10H2O -4081, "
                 "Na+ -240.12, CO3-- -677.14, H2O -285.83), dCp = 0",
          note="melts/transforms near 305 K (to thermonatrite/heptahydrate, not included)"),
    Solid("nahcolite", "NaHCO3", {"Na+": 1, "CO3--": 1, "H+": 1}, 0,
          _vh(-10.742 * math.log(10.0), 33.5e3, 0.0, T_min=273.15, T_max=323.15), quality="B",
          source="log K = -10.742 (PHREEQC pitzer.dat) for NaHCO3 = Na+ + CO3-- + H+; dH +33.5 kJ/mol (NBS: NaHCO3 -950.8), dCp = 0"),
    Solid("trona", "Na3H(CO3)2.2H2O", {"Na+": 3, "CO3--": 2, "H+": 1}, 2,
          _vh(-11.384 * math.log(10.0), 0.0, 0.0, T_min=288.15, T_max=308.15), quality="C",
          source="log K = -11.384 (PHREEQC pitzer.dat)", note="298.15 K only (dH = 0 assumed)"),
    Solid("kalicinite", "KHCO3", {"K+": 1, "CO3--": 1, "H+": 1}, 0,
          _vh(-9.94 * math.log(10.0), 0.0, 0.0, T_min=288.15, T_max=308.15), quality="C",
          source="log K = -9.94 (PHREEQC pitzer.dat, Harvie et al. 1984)", note="298.15 K only"),
    Solid("calcite", "CaCO3", {"Ca++": 1, "CO3--": 1}, 0,
          _ph(-237.04, -0.1077, 0, 102.25, 6.79e5), quality="A",
          source="PHREEQC pitzer.dat analytical expression (Plummer & Busenberg 1982, Ellis 1959)"),
    Solid("aragonite", "CaCO3", {"Ca++": 1, "CO3--": 1}, 0,
          _ph(-171.8607, -0.077993, 2903.293, 71.595), quality="A", source="PHREEQC pitzer.dat analytical expression"),
    Solid("magnesite", "MgCO3", {"Mg++": 1, "CO3--": 1}, 0,
          _vh(-7.834 * math.log(10.0), -6.169e3, 0.0, T_min=273.15, T_max=323.15), quality="B",
          source="log K -7.834, delta_h -6.169 kJ/mol (PHREEQC pitzer.dat)"),
    Solid("nesquehonite", "MgCO3.3H2O", {"Mg++": 1, "CO3--": 1}, 3,
          _vh(-5.167 * math.log(10.0), 0.0, 0.0, T_min=273.15, T_max=323.15), quality="C",
          source="log K -5.167 (PHREEQC pitzer.dat)", note="298.15 K only"),
    Solid("dolomite", "CaMg(CO3)2", {"Ca++": 1, "Mg++": 1, "CO3--": 2}, 0,
          _vh(-17.083 * math.log(10.0), -39.48e3, 0.0, T_min=273.15, T_max=323.15), quality="C",
          source="log K -17.083, delta_h -9.436 kcal (PHREEQC pitzer.dat)", note="ordered dolomite; kinetically hindered"),
    Solid("gaylussite", "Na2Ca(CO3)2.5H2O", {"Na+": 2, "Ca++": 1, "CO3--": 2}, 5,
          _vh(-9.421 * math.log(10.0), 0.0, 0.0, T_min=273.15, T_max=313.15), quality="C",
          source="log K -9.421 (PHREEQC pitzer.dat)", note="298.15 K only"),
    Solid("pirssonite", "Na2Ca(CO3)2.2H2O", {"Na+": 2, "Ca++": 1, "CO3--": 2}, 2,
          _vh(-9.234 * math.log(10.0), 0.0, 0.0, T_min=273.15, T_max=313.15), quality="C",
          source="log K -9.234 (PHREEQC pitzer.dat)", note="298.15 K only"),
    Solid("burkeite", "Na6CO3(SO4)2", {"Na+": 6, "CO3--": 1, "SO4--": 2}, 0,
          _vh(-0.772 * math.log(10.0), 0.0, 0.0, T_min=273.15, T_max=323.15), quality="C",
          source="log K -0.772 (PHREEQC pitzer.dat)", note="298.15 K only"),
    Solid("thermonatrite", "Na2CO3.H2O", {"Na+": 2, "CO3--": 1}, 1,
          _vh(1.392, -12.1e3, 0.0, T_min=273.15, T_max=323.15), quality="C",
          source="NBS formation Gibbs energies / enthalpies (Na2CO3.H2O -1285.3 / -1431.1 kJ/mol; Na+ -261.9 / -240.1; "
                 "CO3-- -527.8 / -677.1; H2O -237.1 / -285.8) recalled from tables, not re-verified"),
    Solid("sodium_carbonate", "Na2CO3", {"Na+": 2, "CO3--": 1}, 0,
          _vh(2.913, -26.7e3, 0.0, T_min=273.15, T_max=323.15), quality="C",
          source="NBS (Na2CO3 -1044.4 / -1130.7 kJ/mol) recalled from tables, not re-verified"),
    Solid("sodium_hydroxide", "NaOH", {"Na+": 1, "H+": -1}, 0,
          _vh(16.0, -44.3e3, 0.0, T_min=273.15, T_max=323.15), n_oh=1, quality="C",
          source="NBS (NaOH -379.5 / -425.8 kJ/mol; OH- -157.2 / -230.0) recalled, not re-verified; OH- carried as -H+"),
    Solid("portlandite", "Ca(OH)2", {"Ca++": 1, "H+": -2}, 0,
          _vh(-5.190 * math.log(10.0), 0.0, 0.0, T_min=273.15, T_max=323.15), n_oh=2, quality="C",
          source="log K -5.190 (PHREEQC pitzer.dat) for Ca(OH)2 = Ca++ + 2 OH-; OH- carried as -H+",
          note="298.15 K only (real dH ~ -16 kJ/mol)"),
    Solid("brucite", "Mg(OH)2", {"Mg++": 1, "H+": -2}, 0,
          _vh(-10.88 * math.log(10.0), 20.29e3, 0.0, T_min=273.15, T_max=323.15), n_oh=2, quality="B",
          source="log K -10.88, delta_h 4.85 kcal (PHREEQC pitzer.dat); OH- carried as -H+"),
    # ---- double salts (not anchored) ---------------------------------------------------------------------
    Solid("glauberite", "Na2Ca(SO4)2", {"Na+": 2, "Ca++": 1, "SO4--": 2}, 0,
          _ph(218.142, 0, -9285, -77.735), quality="B", source=_PH_REF),
    Solid("syngenite", "K2Ca(SO4)2.H2O", {"K+": 2, "Ca++": 1, "SO4--": 2}, 1,
          _vh(math.log(10.0) * -6.43, -32.65e3 * 1.0, 0.0), quality="B",
          source="PHREEQC pitzer.dat: log K -6.43, dH -32.65 (kJ/mol, as printed in the database)"),
]

# AIOMFAC-anchor offsets (ln units, 298.15 K) -- produced by tools/calibrate_solids.py
ANCHOR_OFFSETS: dict[str, float] = {
    "halite": -0.0377,
    "sylvite": -0.0036,
    "sal_ammoniac": 2.8610,
    "nitratine": 2.4820,
    "niter": -0.2485,
    "ammonium_nitrate": 2.4700,
    "mirabilite": 0.0324,
    "arcanite": 0.1833,
    "ammonium_sulfate": 0.0258,
    "bischofite": -0.7091,
    "epsomite": -0.1763,
    "antarcticite": 9.0691,
    "Ca_nitrate_4H2O": 4.5486,
}

# solids of the same electrolyte (same ions) without an own binary solubility anchor borrow the offset of a sibling:
# AIOMFAC's activity-coefficient error at saturation is a property of the aqueous ion mixture, not of the solid
OFFSET_FROM: dict[str, str] = {
    "thenardite": "mirabilite",
    "hexahydrite": "epsomite", "kieserite": "epsomite",
    "MgCl2_4H2O": "bischofite", "MgCl2_2H2O": "bischofite",
}

# AIOMFAC-consistent fits of ln K(T) = (ln_k0 at 298.15 K, dH_eff [J/mol], dCp_eff [J/mol/K], T_lo, T_hi),
# produced by tools/fit_solids.py from the experimental solubility curves in tools/solubility_data.py
KFIT: dict[str, tuple] = {
    "halite": (3.60630, 5329.6, 50.7, 273.15, 333.15),
    "sylvite": (2.07261, 17110.2, -42.9, 273.15, 333.15),
    "sal_ammoniac": (2.86298, 19921.9, 13.4, 273.15, 333.15),
    "nitratine": (2.48529, 14481.7, 0.8, 273.15, 333.15),
    "niter": (-0.24856, 26615.7, -365.3, 273.15, 333.15),
    "ammonium_nitrate": (2.47170, 16549.6, -109.0, 273.15, 333.15),
    "ammonium_sulfate": (0.03758, 18646.8, 73.4, 273.15, 333.15),
    "arcanite": (-4.14530, 32105.4, -224.2, 273.15, 333.15),
    "thenardite": (-0.84029, 6179.5, 0.0, 313.15, 373.15),
    "bischofite": (9.86432, 20986.8, 148.9, 273.15, 333.15),
    "Ca_nitrate_4H2O": (4.54323, 38585.0, 0.0, 273.15, 313.15),
    "mirabilite": (-2.82982, 83531.6, 244.4, 273.15, 303.15),
    "epsomite": (-4.43263, 42017.8, 0.0, 273.15, 313.15),
}

SOLIDS: dict[str, Solid] = {}


def _register():
    from dataclasses import replace
    for s in _DB:
        # for van't Hoff entries ln_k0 is the *reference* value: anchoring supplies the absolute level
        kf = KFIT.get(s.key)
        SOLIDS[s.key] = replace(s, offset=ANCHOR_OFFSETS.get(s.key, ANCHOR_OFFSETS.get(OFFSET_FROM.get(s.key), 0.0)),
                                kfit=KFit(*kf) if kf else None)


_register()


def solids_for_ions(ions) -> list[Solid]:
    """All database solids whose ions are a subset of ``ions`` (ion keys)."""
    ions = set(ions)
    return [s for s in SOLIDS.values() if set(s.ions) <= ions]

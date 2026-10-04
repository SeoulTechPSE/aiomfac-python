"""Experimental binary solubilities (g anhydrous salt / 100 g water) used to anchor/fit/validate aiomfac_py.solids.

Provenance
  * 'web': values read from the BOC Sciences "Water Solubility Table of Inorganic Compounds" (CRC-Handbook-type
    data), https://www.bocsci.com/support-documents/water-solubility-table-at-temperatures.html, retrieved 2026-10-05.
  * 'recall': values from the author's memory of the CRC Handbook / Linke & Seidell; UNVERIFIED -- re-check.
Temperatures in degC.
"""
EXP_G100 = {
    # key: (provenance, {T_C: g per 100 g water})
    "halite": ("web", {0: 35.6, 20: 35.8, 40: 36.42, 60: 37.05, 80: 38.05, 100: 39.2}),
    "sylvite": ("web", {0: 28.15, 20: 34.24, 40: 40.3, 60: 45.6, 80: 51.0, 100: 56.2}),
    "sal_ammoniac": ("web", {0: 29.7, 20: 37.56, 40: 46.0, 60: 55.3, 80: 65.6, 100: 77.3}),
    "nitratine": ("web", {0: 70.7, 20: 88.3, 40: 104.9, 60: 124.7, 80: 148.0, 100: 176.0}),
    "niter": ("web", {0: 13.25, 20: 31.66, 40: 63.9, 60: 109.9, 80: 169.0, 100: 245.2}),
    "ammonium_nitrate": ("web", {0: 118.5, 20: 187.7, 40: 283.0, 60: 415.0, 80: 610.0, 100: 1000.0}),
    "ammonium_sulfate": ("web", {0: 70.4, 20: 75.44, 40: 81.2, 60: 87.4, 80: 94.1, 100: 102.0}),
    "arcanite": ("web", {0: 7.33, 20: 11.11, 40: 14.79, 60: 18.2, 80: 21.29, 100: 24.1}),
    "thenardite": ("web", {40: 48.1, 60: 45.26, 80: 43.09, 100: 42.3}),
    "bischofite": ("web", {0: 52.8, 20: 54.57, 40: 57.5, 60: 60.7, 80: 65.87, 100: 72.7}),
    "Ca_nitrate_4H2O": ("web", {0: 101.0, 20: 129.39, 40: 196.0}),
    "mirabilite": ("recall", {0: 4.9, 10: 9.1, 20: 19.5, 30: 40.8}),
    "epsomite": ("recall", {0: 22.0, 20: 33.7, 40: 44.5}),
}
# temperature windows (degC) used for the K(T) fit (atmospherically relevant range; AIOMFAC is least reliable
# at high T)
FIT_WINDOW = {"default": (0, 60), "thenardite": (40, 100), "mirabilite": (0, 30), "Ca_nitrate_4H2O": (0, 40)}

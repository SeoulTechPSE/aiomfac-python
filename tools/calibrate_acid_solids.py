"""Consistency check of the acid-sulfate solids of ``aiomfac_py.solids`` against deliquescence RH of the pure salts.

The solubility products of these solids are the E-AIM thermodynamic values of Clegg, Brimblecombe & Wexler (1998)
(J. Phys. Chem. A 102, 2137 and 2155), converted to the molal scale of AIOMFAC:
    ln K_m = ln xK + n_ions ln(1000/18.01528)
(no AIOMFAC anchoring).  This script reports the water activity at which AIOMFAC puts the pure, congruently dissolving
salt at saturation, i.e. the DRH implied by these K values, to compare with the DRH quoted in the literature
(NH4HSO4 about 40 %, letovicite about 69.5 % at 298 K: Tang & Munkelwitz 1994, recalled -- verify).

Run:  python tools/calibrate_acid_solids.py
"""
from __future__ import annotations
import math
import numpy as np
from scipy.optimize import brentq
from aiomfac_py.sle import AqueousIons
from aiomfac_py.solids import SOLIDS

LIT = {"ammonium_bisulfate": 0.40, "letovicite": 0.695}      # recalled literature DRH at 298.15 K


def implied_drh(key: str, T: float = 298.15) -> float:
    s = SOLIDS[key]
    names = list(s.ions)
    aq = AqueousIons(names)
    n = np.array([s.ions[i] for i in names], dtype=float)
    lnk = s.ln_k(T)

    def f(ln_rh):
        r = aq.solve_water(n, T, ln_rh)
        if r is None:
            return -1.0e3
        return float(sum(s.ions[i] * r[0][k] for k, i in enumerate(names))) + s.h * ln_rh - lnk

    lo, hi = math.log(0.20), math.log(0.995)
    return math.exp(brentq(f, lo, hi, xtol=1e-10))


def main():
    for key in ("ammonium_bisulfate", "letovicite", "sodium_bisulfate_hydrate", "sodium_bisulfate", "trisodium_hydrogen_sulfate"):
        try:
            d = implied_drh(key)
            ref = LIT.get(key)
            print(f"{key:28s} ln K(298.15)={SOLIDS[key].ln_k(298.15):8.4f}   implied pure-salt DRH = {100 * d:5.1f} %"
                  + (f"   (literature, recalled: {100 * ref:.1f} %)" if ref else ""))
        except ValueError:
            print(f"{key:28s} no sign change in the RH bracket (pure salt not saturated at any 20-99.5 % RH)")


if __name__ == "__main__":
    main()

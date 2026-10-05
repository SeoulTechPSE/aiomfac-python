"""AIOMFAC-consistent solubility products of acid-sulfate solids, anchored to their 298.15 K deliquescence RH.

For acid salts (NH4HSO4, letovicite) no verified binary saturation molality is at hand (see docs/SLE_design_ko.md),
so ln K is fixed by requiring that the congruently dissolving pure salt deliquesces at its literature DRH:

    ln K = sum_i nu_i ln a_i(free ions)    evaluated for the stoichiometric salt solution at a_w = DRH

(HSO4- <-> H+ + SO4-- speciated by AqueousIons).  Run:  python tools/calibrate_acid_solids.py
"""
from __future__ import annotations
import math
import numpy as np
from aiomfac_py.sle import AqueousIons, MW_WATER

# key -> (ions, nu, DRH at 298.15 K, source)
ACID = {
    "ammonium_bisulfate": ({"NH4+": 1, "H+": 1, "SO4--": 1}, 0.40, "Tang & Munkelwitz 1994 (recalled; verify)"),
    "letovicite": ({"NH4+": 3, "H+": 1, "SO4--": 2}, 0.695, "Tang & Munkelwitz 1994 (recalled; verify)"),
}


def ln_k_at_drh(ions: dict, drh: float, T: float = 298.15) -> tuple[float, float]:
    names = list(ions)
    aq = AqueousIons(names)
    n = np.array([ions[i] for i in names], dtype=float)
    ln_a, w, lnM = aq.solve_water(n, T, math.log(drh))
    lnk = float(sum(ions[i] * ln_a[k] for k, i in enumerate(names)))
    return lnk, float(n.sum() / w / 1.0)


def main():
    for k, (ions, drh, src) in ACID.items():
        lnk, mtot = ln_k_at_drh(ions, drh)
        print(f'{k:20s} DRH={drh:.3f}  total ion molality at saturation={mtot:7.3f}  ln K(298.15)={lnk:9.4f}   [{src}]')


if __name__ == "__main__":
    main()

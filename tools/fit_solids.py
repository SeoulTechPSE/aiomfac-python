"""Fit model-consistent ln K(T) (K0, dH_eff, dCp_eff) so AIOMFAC reproduces the experimental binary solubility curves.

Run: python tools/fit_solids.py   -> prints the KFIT block to paste into aiomfac_py/solids.py
"""
from __future__ import annotations
import math
import numpy as np
from aiomfac_py import solids as S
from calibrate_solids import binary_model
from solubility_data import EXP_G100, FIT_WINDOW

T0 = 298.15
R = S.R_GAS


def ln_k_needed(s, m, T):
    mod, mw = binary_model(s)
    mass = 1.0 + m * mw / 1000.0
    r = mod.evaluate([1.0 / mass, m * mw / 1000.0 / mass], T, "mass")
    return math.log(r.activity[1]) + s.h * math.log(r.activity[0])


def main():
    print("KFIT = {")
    for k, (prov, d) in EXP_G100.items():
        s = S.SOLIDS[k]
        mod, mw = binary_model(s)
        lo, hi = FIT_WINDOW.get(k, FIT_WINDOW["default"])
        pts = [(273.15 + t, g / mw * 10.0) for t, g in d.items() if lo <= t <= hi]
        if s.anchor_m is not None and not (273.15 + lo > T0 or 273.15 + hi < T0):
            pts.append((T0, s.anchor_m))
        T = np.array([p[0] for p in pts]); y = np.array([ln_k_needed(s, p[1], p[0]) for p in pts])
        x1 = 1.0 / T - 1.0 / T0
        x2 = T0 / T - 1.0 + np.log(T / T0)
        if len(pts) >= 5:
            A = np.column_stack([np.ones_like(T), -x1 / 1.0, x2]); npar = 3
        else:
            A = np.column_stack([np.ones_like(T), -x1]); npar = 2
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        res = y - A @ coef
        a, dH, dCp = coef[0], coef[1] * R, (coef[2] * R if npar == 3 else 0.0)
        Tlo, Thi = float(T.min()), float(T.max())
        print(f'    "{k}": ({a:.5f}, {dH:.1f}, {dCp:.1f}, {Tlo:.2f}, {Thi:.2f}),  # n={len(pts)} rms(lnK)={np.sqrt(np.mean(res**2)):.4f}')
    print("}")


if __name__ == "__main__":
    main()

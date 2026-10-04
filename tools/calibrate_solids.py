"""Compute the AIOMFAC anchor offsets of ``aiomfac_py.solids`` (see that module's docstring).

For every solid with a binary saturation molality ``anchor_m`` at 298.15 K, AIOMFAC is evaluated at that composition
and  offset = ln(IAP_ions * a_w^h) - ln K_ref(298.15).  Run:  python tools/calibrate_solids.py
"""
from __future__ import annotations
import math
import numpy as np
from aiomfac_py import ActivityModel, Component
from aiomfac_py import solids as S
from aiomfac_py.params import load_subgroup_params

W = Component(1, "Water", ((16, 1),))


def binary_model(solid):
    cats = {i: n for i, n in solid.ions.items() if S.ION_REGISTRY[i][1] > 0}
    ans = {i: n for i, n in solid.ions.items() if S.ION_REGISTRY[i][1] < 0}
    if len(cats) != 1 or len(ans) != 1:
        return None
    (c, nc), (a, na) = next(iter(cats.items())), next(iter(ans.items()))
    sub = ((S.ION_REGISTRY[c][0], nc), (S.ION_REGISTRY[a][0], na))
    sg = load_subgroup_params()
    mw = nc * sg.SMWC[S.ION_REGISTRY[c][0] - 201] + na * sg.SMWA[S.ION_REGISTRY[a][0] - 241]
    return ActivityModel([W, Component(2, solid.formula, sub)]), mw


def saturation_k(solid, T=298.15, m=None):
    mod, mw = binary_model(solid)
    m = solid.anchor_m if m is None else m
    mass = 1.0 + m * mw / 1000.0
    r = mod.evaluate([1.0 / mass, m * mw / 1000.0 / mass], T, "mass")
    aw, iap = r.activity[0], r.activity[1]
    return math.log(iap) + solid.h * math.log(aw), aw, iap


def main():
    out = {}
    for k, s in S.SOLIDS.items():
        if s.anchor_m is None or binary_model(s) is None:
            continue
        ln_k_model, aw, iap = saturation_k(s)
        ref0 = S.Solid.ln_k_ref(s, 298.15)
        off = ln_k_model - ref0
        out[k] = off
        flag = ""
        if s.kspec.kind == "phreeqc":
            flag = f"  (database log10K={ref0/math.log(10):7.3f}; AIOMFAC-at-saturation log10K={ln_k_model/math.log(10):7.3f}; diff={off/math.log(10):+.3f} log10)"
        print(f"{k:18s} m={s.anchor_m:7.4f} aw={aw:.4f} lnK_model={ln_k_model:8.3f} offset={off:+8.3f}{flag}")
    print("\nANCHOR_OFFSETS = {")
    for k, v in out.items():
        print(f'    "{k}": {v:.4f},')
    print("}")


if __name__ == "__main__":
    main()

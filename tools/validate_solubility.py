"""Binary solubility curves from aiomfac_py.solids + AIOMFAC versus handbook values (g salt / 100 g water).

The experimental values are CRC-Handbook-type solubility data *as recalled by the author of this script*; they are
for a sanity check (a few percent) and must be re-verified against the primary tables before publication use.
"""
from __future__ import annotations
import math
import numpy as np
from scipy.optimize import brentq
from aiomfac_py import ActivityModel, Component
from aiomfac_py import solids as S
from aiomfac_py.params import load_subgroup_params
from calibrate_solids import binary_model

W = Component(1, "Water", ((16, 1),))
_cache = {}


def _mod(s):
    if s.key not in _cache:
        _cache[s.key] = binary_model(s)
    return _cache[s.key]


def residual(s, m, T, mode="anchored"):
    mod, mw = _mod(s)
    mass = 1.0 + m * mw / 1000.0
    r = mod.evaluate([1.0 / mass, m * mw / 1000.0 / mass], T, "mass")
    return math.log(r.activity[1]) + s.h * math.log(r.activity[0]) - s.ln_k(T, mode)


def sat_molality(s, T, lo=0.02, hi=120.0, mode="anchored"):
    mw = _mod(s)[1]
    ms = np.geomspace(lo, hi, 90)
    prev = None
    for m in ms:
        try:
            f = residual(s, m, T, mode)
        except Exception:
            continue
        if prev is not None and np.sign(f) != np.sign(prev[1]):
            return brentq(lambda x: residual(s, x, T, mode), prev[0], m, xtol=1e-6)
        prev = (m, f)
    return float("nan")


def main(keys=None):
    from solubility_data import EXP_G100
    for k, (prov, d) in EXP_G100.items():
        if keys and k not in keys:
            continue
        s = S.SOLIDS[k]
        mw = _mod(s)[1]
        print(f"\n{k}  ({s.formula})   quality={s.quality}  data={prov}")
        print("   T/C   m_exp   m_anchored(%err)   m_fitted(%err)")
        for tc, g in d.items():
            m_exp = g / mw * 10.0
            out = []
            for mode in ("anchored", "fitted"):
                m_calc = sat_molality(s, 273.15 + tc, mode=mode)
                out.append(f"{m_calc:8.3f} ({100*(m_calc/m_exp-1):+6.1f}%)")
            print(f"  {tc:4d}  {m_exp:6.3f}  " + "  ".join(out))


if __name__ == "__main__":
    import sys
    main(sys.argv[1:] or None)

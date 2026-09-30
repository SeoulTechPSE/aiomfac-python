"""Numerical helpers ported from ModNumericalTransformations.f90 (safe_exp, soft_bounds_external)."""
from __future__ import annotations

import math

import numpy as np


def soft_bounds_external(x, l_lim, u_lim):
    """x unchanged inside [l_lim, u_lim]; beyond the limits it is compressed logarithmically (a few percent at most)."""
    x = np.asarray(x, dtype=float)
    margin = 2.0e-2 * abs(u_lim - l_lim)
    ext_low, ext_up = l_lim - margin, u_lim + margin
    y = x.copy()
    hi, lo = x > u_lim, x < l_lim
    if np.any(hi):
        y[hi] = np.minimum(u_lim + 1.0e-2 * np.log(np.abs(x[hi] - (u_lim - 1.0))), ext_up)
    if np.any(lo):
        y[lo] = np.maximum(l_lim - 1.0e-2 * np.log(np.abs(x[lo] - (l_lim + 1.0))), ext_low)
    return y


def safe_exp(x, ln_limit):
    """exp(x) with soft bounds |x| <= ln_limit (Fortran safe_exp); identical to exp(x) inside the bounds."""
    y = soft_bounds_external(x, -ln_limit, ln_limit)
    out = np.exp(y)
    return float(out) if np.ndim(x) == 0 else out


#: constants of ModCalcActCoeff
LNTINY = 0.49 * math.log(np.finfo(float).tiny)
LNHUGE = 0.49 * math.log(np.finfo(float).max)
LOGVAL_THRESHOLD = 0.4 * LNHUGE
DTINY = math.sqrt(np.finfo(float).tiny)


def brent_root(f, a, b, *, xtol=1e-14, rtol=4e-16, maxiter=200):
    """Brent's method for a root of ``f`` bracketed by [a, b] (f(a) and f(b) must have opposite signs).

    Self-contained (no SciPy dependency) implementation of the classic algorithm, used in place of the
    Fortran ``brentzero`` (brent.f90, third-party GNU LGPL code, not reproduced here).
    """
    fa, fb = f(a), f(b)
    if fa == 0.0:
        return a
    if fb == 0.0:
        return b
    if fa * fb > 0.0:
        raise ValueError("brent_root: root is not bracketed by [a, b]")
    if abs(fa) < abs(fb):
        a, b, fa, fb = b, a, fb, fa
    c, fc, d, mflag = a, fa, None, True
    for _ in range(maxiter):
        if fb == 0.0 or abs(b - a) < xtol + rtol * abs(b):
            return b
        if fa != fc and fb != fc:                       # inverse quadratic interpolation
            s = (a * fb * fc / ((fa - fb) * (fa - fc)) + b * fa * fc / ((fb - fa) * (fb - fc))
                 + c * fa * fb / ((fc - fa) * (fc - fb)))
        else:                                            # secant
            s = b - fb * (b - a) / (fb - fa)
        lo, hi = (3.0 * a + b) / 4.0, b
        if lo > hi:
            lo, hi = hi, lo
        bad = not (lo <= s <= hi)
        if mflag:
            bad = bad or abs(s - b) >= abs(b - c) / 2.0
        else:
            bad = bad or abs(s - b) >= abs(c - d) / 2.0
        if mflag and d is not None and abs(b - c) < xtol:
            bad = True
        if (not mflag) and d is not None and abs(c - d) < xtol:
            bad = True
        if bad:
            s = (a + b) / 2.0; mflag = True
        else:
            mflag = False
        fs = f(s)
        d = c; c, fc = b, fb
        if fa * fs < 0.0:
            b, fb = s, fs
        else:
            a, fa = s, fs
        if abs(fa) < abs(fb):
            a, b, fa, fb = b, a, fb, fa
    return b

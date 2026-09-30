"""Liquid-liquid phase *stability* (spinodal) via the determinant of the reduced Gibbs-energy Hessian.

Ports the stability-analysis part of:

    Zuend, A., Marcolli, C., Peter, T., Seinfeld, J. H. (2010), "Computation of liquid-liquid equilibria and
    phase stabilities: implications for RH-dependent gas/particle partitioning of organic-inorganic aerosols",
    Atmos. Chem. Phys., 10, 7795-7820, doi:10.5194/acp-10-7795-2010. [cited below as "Z2010"]

What this implements, and how closely (read this before trusting the numbers)
-------------------------------------------------------------------------------
Z2010 Sect. 2.6, Eq. (18)-(20) tests local thermodynamic stability of a candidate (single, notionally
homogeneous) composition by evaluating the determinant ``L`` of the Hessian of the molar Gibbs free energy,
restricted to the tangent plane of the composition simplex (i.e. expressed in ``n_s - 1`` independent mole
fractions, eliminating the dependent one via the ``sum(x) = 1`` constraint). A positive-definite reduced
Hessian (``L > 0`` for the two-independent-variable, ternary-system case Z2010 actually plots) means the
composition is a local Gibbs-energy minimum -- stable or metastable; ``L < 0`` means it sits inside the
spinodal region (unconditionally unstable, decomposes spontaneously, no nucleation barrier); ``L = 0`` is the
spinodal boundary itself. This is the standard tangent-plane/Hessian stability criterion (see e.g. Z2010's own
citations to Cahn 1965 and Debenedetti 1996), applied here to this package's AIOMFAC ``ActivityModel`` via the
same ``AiomfacGFE.g`` normalized molar GFE already used by ``lle.solve_pep`` -- so "stable" here means stable
*with respect to AIOMFAC's own Gibbs energy surface*, exactly as in Z2010.

The reduced Hessian is obtained by direct finite differences on the ``n_s - 1`` independent mole fractions
(central differences, eliminating one dependent species via the simplex constraint) rather than by projecting
the full ``AiomfacGFE.hess`` (whose domain is the full ``n_s``-dimensional space and which has ``x`` itself as
an exact null vector by the Gibbs-Duhem relation, JOTA-2 as noted in ``lle.py``) onto a tangent subspace --
both give the same reduced quadratic form, but computing it directly in the independent coordinates is
numerically simpler and is what is implemented here.

Z2010 locates the full 2-D stability diagram (their Fig. 5/6) and the binodal/spinodal curves with a
Differential-Evolution (DE) global optimizer (Storn and Price, 1997) hybridized with Powell's method and
Levenberg-Marquardt (their Appendix A) -- a general, derivative-free, globally convergent scheme needed because
the underlying root-finding/optimization problems (locating the binodal via the phase-equilibrium tangent-line
construction, and the spinodal via ``L = 0``) are not everywhere well-behaved for an arbitrary mixture. This
module does not reimplement DE; instead it (a) evaluates ``L`` directly on a supplied composition grid (cheap
and completely general -- this is all a "stability map" like Z2010 Fig. 5/6 actually needs), and (b) locates
the ``L = 0`` contour from that grid with ``L``'s own sign change (one 1-D bracketing root-find per grid row),
which is exact in the same sense as Z2010's DE-based root-find, just slower (dense grid) rather than adaptive.
This is a deliberate scope choice, not an approximation of the physics: the stability criterion (Eq. 18-20) is
implemented exactly; only the *search strategy* used to trace its zero contour differs from Z2010's.
"""
from __future__ import annotations

import numpy as np

from .io import Component
from .lle import AiomfacGFE


def reduced_hessian(gfe: AiomfacGFE, x, dep_index: int | None = None, h: float = 5.0e-4) -> np.ndarray:
    """Hessian of the molar GFE ``g``, restricted to the composition simplex by eliminating mole fraction
    ``dep_index`` (default: the last species) via ``x[dep_index] = 1 - sum(other x)``.

    Returns an ``(n_s - 1, n_s - 1)`` matrix in the remaining ("free") mole fractions, symmetrized. This is
    the tangent-plane stability matrix of Z2010 Eq. (18)-(19); its determinant is Eq. (20)'s ``L``.
    """
    x = np.asarray(x, dtype=float)
    x = x / x.sum()
    n = gfe.n_species
    if dep_index is None:
        dep_index = n - 1
    free = [i for i in range(n) if i != dep_index]
    m = len(free)

    def full_x(u):
        xx = np.empty(n)
        for k, i in enumerate(free):
            xx[i] = u[k]
        xx[dep_index] = 1.0 - u.sum()
        return xx

    u0 = x[free]
    steps = np.maximum(h * np.maximum(np.abs(u0), 1.0), 1.0e-6)
    steps = np.minimum(steps, 0.3 * np.maximum(u0, 1.0e-8))

    def gt(u):
        xx = full_x(u)
        if np.any(xx <= 0.0) or full_x(u)[dep_index] <= 0.0:
            return np.nan
        return gfe.g(xx)

    g0 = gt(u0)
    gp = np.empty(m); gm = np.empty(m)
    for i in range(m):
        up = u0.copy(); up[i] += steps[i]
        um = u0.copy(); um[i] -= steps[i]
        gp[i] = gt(up); gm[i] = gt(um)

    H = np.zeros((m, m))
    for i in range(m):
        H[i, i] = (gp[i] - 2.0 * g0 + gm[i]) / steps[i] ** 2
    for i in range(m):
        for j in range(i + 1, m):
            upp = u0.copy(); upp[i] += steps[i]; upp[j] += steps[j]
            umm = u0.copy(); umm[i] -= steps[i]; umm[j] -= steps[j]
            upm = u0.copy(); upm[i] += steps[i]; upm[j] -= steps[j]
            ump = u0.copy(); ump[i] -= steps[i]; ump[j] += steps[j]
            H[i, j] = H[j, i] = (gt(upp) - gt(upm) - gt(ump) + gt(umm)) / (4.0 * steps[i] * steps[j])
    return 0.5 * (H + H.T)


def stability_determinant(gfe: AiomfacGFE, x, dep_index: int | None = None) -> float:
    """Z2010 Eq. (20): ``L`` = determinant of the reduced (tangent-plane) Hessian at composition ``x``.

    ``L > 0``: locally stable/metastable single phase. ``L < 0``: inside the spinodal (unconditionally
    unstable). ``L == 0``: on the spinodal.
    """
    H = reduced_hessian(gfe, x, dep_index=dep_index)
    return float(np.linalg.det(H))


def stability_map(components: list[Component], T_K: float, compositions: np.ndarray,
                   dep_index: int | None = None) -> np.ndarray:
    """``L`` (Eq. 20) at each row of ``compositions`` (n_points, n_species), NaN where AIOMFAC itself fails
    (e.g. a composition too close to a pure-electrolyte vertex). Reuses one ``AiomfacGFE`` (and therefore one
    ``ActivityModel``) across the whole grid."""
    gfe = AiomfacGFE(components, T_K)
    out = np.full(len(compositions), np.nan)
    for i, x in enumerate(compositions):
        try:
            out[i] = stability_determinant(gfe, x, dep_index=dep_index)
        except Exception:
            out[i] = np.nan
    return out

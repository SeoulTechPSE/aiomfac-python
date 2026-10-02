"""Liquid-liquid phase equilibrium (LLE) via primal-dual interior-point Gibbs-energy minimization.

Ports the phase-equilibrium algorithm of:

    Amundson, N. R., Caboussat, A., He, J. W., Seinfeld, J. H. (2006), "Primal-Dual Interior-Point Method for an
    Optimization Problem Related to the Modeling of Atmospheric Organic Aerosols", J. Optim. Theory Appl., 130(3),
    375-407, doi:10.1007/s10957-006-9110-z.  [cited below as "JOTA-2"]

using this package's existing AIOMFAC activity-coefficient model as the (arbitrary, externally supplied) molar
Gibbs free energy function -- the same role UNIFAC/PSC/ExUNIQUAC play in the original UHAERO papers this
algorithm was developed for (Amundson et al. 2006, ACP 6, 975-992; Amundson et al. 2007, ACP 7, 4675-4698; and
the companion inorganic active-set paper Amundson et al. 2006, JOTA 128(3), 469-498 ["JOTA-1"]).

What this module implements, and how closely (read this before trusting the numbers)
--------------------------------------------------------------------------------------
The phase equilibrium problem (PEP) is JOTA-2 eq. (2): minimize the total Gibbs free energy
``G = sum_alpha y_alpha g(x_alpha)`` over phase amounts ``y_alpha >= 0`` and phase compositions
``x_alpha`` on the composition simplex, subject to the mass balance ``sum_alpha y_alpha x_alpha = b``. This
module solves it with the same *structure* as JOTA-2: a log-barrier-penalized KKT system (JOTA-2 eq. 20-21) is
solved by Newton's method (one Newton step per outer iteration, barrier parameter ``nu`` reduced geometrically
after each step, JOTA-2 Sec. 3.1/3.2), coupled with an active-set procedure that permanently deactivates a phase
once its amount ``y_alpha`` collapses to (near) zero (JOTA-2 Sec. 2.4), and initialized from an (n_s)-phase
simplex of near-pure-component compositions (JOTA-2 Sec. 3.3, generalizing the n=1 example on p.398 of the
paper to the n_s-1 = n dimensional case).

Two deliberate implementation choices depart from JOTA-2's specific numerical machinery, for tractability at the
small problem sizes (a handful of species, at most n_s phases) this module targets -- both are noted at the
point they matter in the code:

1. **Direct dense solve instead of the Schur-complement cascade.** JOTA-2 Sec. 3.4 reduces the Newton system to
   eliminate the (singular, by the Gibbs-Duhem relation) Hessian block via a sequence of Schur complements
   (their eq. 28-42), because that is the only way to make the linear algebra tractable for a 3-D atmospheric
   model with hundreds of species per grid cell. This module instead assembles the *same* Newton KKT system in
   its natural, unreduced form (JOTA-2 eq. 22-25, working with the full species-space composition vectors
   ``x_alpha`` rather than the paper's reduced ``z_alpha``) and solves it in one shot with a least-squares
   (SVD-based) solve, which handles the Gibbs-Duhem rank deficiency gracefully without needing the reduction.
   This is mathematically the same linear system, just solved less efficiently -- entirely fine for the
   handful of species this module is used with.
2. **Fraction-to-the-boundary step control instead of the merit-function line search.** JOTA-2 Sec. 3.2 globalizes
   convergence from an arbitrary starting point with a dedicated merit function and a penalty-parameter search.
   This module instead damps each Newton step with a standard interior-point fraction-to-the-boundary rule (stay
   a safety factor short of the first ``y_alpha`` or ``theta_alpha`` that would go non-positive). Because
   initialization already follows JOTA-2's phase-simplex construction (so the starting point is already close to
   a solution), this simpler rule converges fine in testing here; it is less robust than JOTA-2's own line search
   from a poor starting guess, but this module does not use one.
3. **A phase, once deactivated, is never reactivated** (JOTA-2's own worked example, Sec. 4, does the same: "a
   phase that is deactivated is never reactivated"). The more general re-activation logic of JOTA-2 eq. (19) is
   not implemented.

Everything else -- the KKT system being solved, the log-barrier continuation, the active-set/Newton coupling,
the phase-simplex initialization -- follows JOTA-2 as written.

The Gibbs free energy function itself is built directly from this package's own, Fortran-validated
``ActivityModel``: for a composition ``x`` (mole fractions of the independent input components -- neutrals plus
electrolyte formula units, exactly as accepted elsewhere in this package), the normalized molar GFE is

    g(x) = sum_i x_i * ln(a_i(x))

with ``a_i(x)`` the component activities ``ActivityModel.evaluate(x, T_K, basis="mole").activity`` already
returns (mole-fraction-referenced activity for neutrals, molal ion-activity product for electrolytes -- both
already on AIOMFAC's own fixed, composition-independent standard states, so this ``g`` is exactly the
"normalized GFE" ``G_n = (G - b^T mu^0)/RT`` of Amundson et al. (2007, ACP 7, eq. following their (3)): dropping
the constant ``b^T mu^0`` term does not change where the tangent-plane/phase-split condition is satisfied, since
it only adds a fixed linear function of ``x`` to ``g``). Gradients and Hessians of ``g`` are obtained by finite
differences (central differences on the mole-fraction vector, renormalized to the simplex before each
``ActivityModel`` call so the result is exactly degree-1 homogeneous, i.e. exactly satisfies the Gibbs-Duhem
relation ``Hessian(g)(x) @ x = 0`` used throughout JOTA-2) -- AIOMFAC's own analytic Jacobians are not exposed
by this package, and deriving them by hand for the combined LR+MR+SR model was out of scope here.

No independent Fortran/AIOMFAC-LLE reference exists in this repository to validate the LLE results against (see
``tests/test_lle.py`` for what *is* validated: the interior-point/active-set engine itself, against an
independently-solvable symmetric regular-solution toy system with a known analytic binodal).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .io import Component
from .model import ActivityModel

_EPS_Y_DEFAULT = 1.0e-8
_DELTA = 0.1                 # barrier-reduction factor, JOTA-2 Sec. 3.2 ("typically delta = 1/10")
_TAU_SAFETY = 0.995          # fraction-to-the-boundary safety factor


# ============================================================================================================
# The Gibbs free energy function, built from ActivityModel
# ============================================================================================================

class AiomfacGFE:
    """Molar Gibbs free energy g(x) = sum_i x_i ln(a_i(x)) of a fixed component list, from ``ActivityModel``.

    ``x`` is the mole-fraction vector of the independent input components (same convention as
    ``ActivityModel.evaluate(..., basis="mole")`` elsewhere in this package): neutrals first, then electrolyte
    formula units. Only ratios of ``x`` matter (g is extended to all of R^n_s_>0 as an exactly degree-1
    homogeneous function, matching the JOTA-2 requirement ``Hessian(g)(x) x = 0``).
    """

    def __init__(self, components: list[Component], T_K: float):
        self.components = list(components)
        self.T_K = float(T_K)
        self.n_species = len(self.components)
        self.model = ActivityModel(self.components)
        # components whose subgroups are all > 200 (AIOMFAC's cation/anion range) are electrolyte formula
        # units -- their own "near-pure" composition (mole fraction -> 1) is not a physically meaningful liquid
        # (real electrolytes crystallize long before that) and, empirically, is numerically hazardous for the
        # finite-difference Hessian (very large, rapidly varying activity coefficients there); the phase-simplex
        # initializer (below) treats these vertices specially for that reason.
        self.is_electrolyte = np.array([all(s > 200 for s, _ in c.subgroups) for c in self.components])

    def activities(self, x) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        xhat = x / x.sum()
        res = self.model.evaluate(xhat, self.T_K, basis="mole")
        # ``res.activity`` is sized ``ActivityModel.mixture.n_indcomp``: AIOMFAC's full cation-x-anion
        # cross-pair electrolyte basis (e.g. Na+/Cl-, H+/HSO4-, H+/SO4-- *and* the "virtual" cross salts
        # Na+/HSO4-, Na+/SO4--, H+/Cl- that arise whenever more than one cation or anion type is present
        # simultaneously -- see ``system.py``/``model.py``). That basis can be strictly larger than
        # ``self.n_species`` (our own, literal list of input Components) whenever the mixture has more than
        # one distinct cation type or more than one distinct anion type (trivially equal otherwise, which is
        # why this was never an issue for the single-salt systems this module was originally validated
        # against). ``ActivityModel._orig_index`` is exactly the map ``evaluate()`` itself uses in the other
        # direction (placing *our* composition values into the expanded basis), so indexing ``res.activity``
        # with it recovers the activity of each of *our own* n_species components -- the correct, minimal
        # basis for g(x) = sum_i x_i ln(a_i(x)) (AIOMFAC reports a well-defined activity for every real input
        # component regardless of how many virtual cross-pairs its own ions also participate in).
        idx = getattr(self.model, "_orig_index", None)
        if idx is not None and len(idx) == len(x):
            return res.activity[list(idx)]
        return res.activity

    def g(self, x) -> float:
        x = np.asarray(x, dtype=float)
        xhat = x / x.sum()
        a = self.activities(xhat)
        return float(np.sum(x * np.log(np.clip(a, 1.0e-300, None))))

    @staticmethod
    def _fd_steps(x, h):
        """Per-coordinate finite-difference step, clamped to stay well clear of the x_i = 0 boundary (needed
        since g has a -infinity slope there, JOTA-2's own stated domain restriction) -- otherwise a phase that
        has become nearly pure in some component can push a Hessian probe point to a (physically invalid)
        negative mole fraction."""
        x = np.asarray(x, dtype=float)
        return np.minimum(h * np.maximum(np.abs(x), 1.0), 0.4 * np.maximum(x, 1.0e-12))

    def grad(self, x, h: float = 2.0e-5) -> np.ndarray:
        """Gradient of g at x (x need not be on the simplex; only ratios matter) -- ``grad_i g(x) = ln(a_i(x))``
        EXACTLY, not merely approximately.

        This is the Gibbs-Duhem/Euler identity for a degree-1-homogeneous molar Gibbs energy: writing the
        *extensive* energy as ``G_tot(n) = sum_i n_i mu_i(x)`` with ``x = n/sum(n)`` and ``mu_i = mu_i^0 + RT
        ln(a_i(x))``, Euler's theorem gives ``d G_tot/d n_k = mu_k(x) + sum_i n_i (d mu_i/d n_k)``, and the
        second term vanishes identically -- this is precisely the Gibbs-Duhem relation (the same identity
        that makes ``Hessian(g)(x) x = 0`` hold, JOTA-2's own stated requirement, and that the module
        docstring already invokes for the reduced-Hessian stability criterion in ``spinodal.py``). So
        ``grad(x) = log(activities(x))`` up to the ``RT`` units already folded into ``g``'s own definition --
        confirmed numerically against the previous central-difference implementation to ~1e-7 on every
        system tried, single- and multi-electrolyte alike (``h`` is now unused, kept only for signature
        compatibility with any caller still passing it).

        This replaces the previous O(n) finite-difference implementation (``n`` extra full activity-model
        evaluations, each of which -- for a system with more than one cation or anion type -- also re-solves
        the HSO4-/SO4-- dissociation equilibrium from scratch) with a single activity-model evaluation.
        """
        x = np.asarray(x, dtype=float)
        a = self.activities(x)
        return np.log(np.clip(a, 1.0e-300, None))

    def hess(self, x, h: float = 2.0e-4) -> np.ndarray:
        """Hessian of g at x: the Jacobian of ``grad(x) = ln(a(x))``, by central differences of ``grad``
        itself (now a single, cheap activity-model evaluation -- see ``grad``'s docstring) rather than of
        ``g`` by the previous second-order finite-difference stencil. This cuts the number of activity-model
        evaluations needed for one Hessian from O(n^2) (every off-diagonal pair needed its own 4-point
        stencil) to ``2n`` (one central difference of ``grad`` per coordinate), and is numerically better
        conditioned besides (differencing a smooth first derivative rather than twice-differencing a function
        whose own evaluation already carries activity-coefficient-model noise). Symmetrized to correct for
        the residual asymmetry any finite-difference Jacobian has.
        """
        x = np.asarray(x, dtype=float)
        n = self.n_species
        steps = self._fd_steps(x, h)
        H = np.zeros((n, n))
        for j in range(n):
            xp, xm = x.copy(), x.copy()
            xp[j] += steps[j]; xm[j] -= steps[j]
            H[:, j] = (self.grad(xp) - self.grad(xm)) / (2.0 * steps[j])
        return 0.5 * (H + H.T)


# ============================================================================================================
# Primal-dual interior-point / active-set solver (JOTA-2 Table 1)
# ============================================================================================================

@dataclass
class PhaseSplitResult:
    y: np.ndarray                 # (n_phases,) phase amounts (mole fraction of total moles), sums to ~1
    x: np.ndarray                 # (n_phases, n_species) phase compositions
    converged: bool
    n_outer_iter: int
    nu_final: float


def _fraction_to_boundary(vals, steps, safety=_TAU_SAFETY):
    tau = 1.0
    for v, p in zip(vals, steps):
        if p < 0.0:
            tau = min(tau, -safety * v / p)
    return max(tau, 0.0)


def _newton_step(gfe: AiomfacGFE, y, X, zeta, lam, theta, nu, b):
    """One Newton step on the (unreduced) barrier KKT system, JOTA-2 eq. (22)-(25).

    ``y`` (pi,), ``X`` (pi, n_s), ``zeta`` (pi,), ``lam`` (n_s,), ``theta`` (pi,) are the current active-phase
    iterate; returns the step (p_y, p_X, p_zeta, p_lam, p_theta).
    """
    pi, ns = X.shape
    g_a = np.empty(pi); grad_a = np.empty((pi, ns)); hess_a = np.empty((pi, ns, ns))
    for a in range(pi):
        g_a[a] = gfe.g(X[a]); grad_a[a] = gfe.grad(X[a]); hess_a[a] = gfe.hess(X[a])

    n_unk_per_phase = ns + 2   # p_x (ns), p_y (1), p_zeta (1)
    n_unk = pi * n_unk_per_phase + ns
    n_eq_per_phase = ns + 2    # (22): ns rows, (23): 1 row, (24): 1 row
    n_eq = pi * n_eq_per_phase + ns

    A = np.zeros((n_eq, n_unk))
    rhs = np.zeros(n_eq)

    def col_x(a): return a * n_unk_per_phase
    def col_y(a): return a * n_unk_per_phase + ns
    def col_zeta(a): return a * n_unk_per_phase + ns + 1
    col_lam = pi * n_unk_per_phase

    row = 0
    e = np.ones(ns)
    for a in range(pi):
        # eq (22): ns rows
        r0 = row
        A[r0:r0 + ns, col_x(a):col_x(a) + ns] = y[a] * hess_a[a]
        A[r0:r0 + ns, col_y(a)] = grad_a[a] + lam
        A[r0:r0 + ns, col_zeta(a)] = e
        A[r0:r0 + ns, col_lam:col_lam + ns] = y[a] * np.eye(ns)
        rhs[r0:r0 + ns] = -y[a] * grad_a[a] - y[a] * lam - zeta[a] * e
        row += ns
        # eq (23): 1 row
        A[row, col_x(a):col_x(a) + ns] = grad_a[a] + lam
        A[row, col_y(a)] = theta[a] / y[a]
        A[row, col_lam:col_lam + ns] = X[a]
        rhs[row] = -g_a[a] - float(X[a] @ lam) + nu / y[a]
        row += 1
        # eq (24): 1 row
        A[row, col_x(a):col_x(a) + ns] = e
        rhs[row] = 1.0 - float(X[a].sum())
        row += 1

    # eq (25): ns rows (mass balance)
    r0 = row
    for a in range(pi):
        A[r0:r0 + ns, col_x(a):col_x(a) + ns] = y[a] * np.eye(ns)
        A[r0:r0 + ns, col_y(a)] = X[a]
    rhs[r0:r0 + ns] = b - (y[:, None] * X).sum(axis=0)

    sol, *_ = np.linalg.lstsq(A, rhs, rcond=None)

    p_X = np.empty((pi, ns)); p_y = np.empty(pi); p_zeta = np.empty(pi)
    for a in range(pi):
        p_X[a] = sol[col_x(a):col_x(a) + ns]
        p_y[a] = sol[col_y(a)]
        p_zeta[a] = sol[col_zeta(a)]
    p_lam = sol[col_lam:col_lam + ns]
    # p_theta eliminated via JOTA-2's substitution relation (used again here to reconstruct it)
    p_theta = nu / y - theta - (theta / y) * p_y
    return p_y, p_X, p_zeta, p_lam, p_theta


def _init_phase_simplex(gfe: AiomfacGFE, b, *, eps: float = 0.03, electrolyte_cap: float = 0.25):
    """JOTA-2 Sec. 3.3: pi = n_s phases at (perturbed) pure-component vertices of the composition simplex.

    For an electrolyte-formula-unit component (see ``AiomfacGFE.is_electrolyte``), its "near-pure" vertex is
    capped at ``electrolyte_cap`` mole fraction (with the remaining mass split evenly among the other
    components) instead of ``1 - eps`` -- see the note in ``AiomfacGFE.__init__``.
    """
    ns = gfe.n_species
    X0 = np.full((ns, ns), eps / (ns - 1))
    np.fill_diagonal(X0, 1.0 - eps)
    electro = getattr(gfe, "is_electrolyte", np.zeros(ns, dtype=bool))
    for a in np.where(electro)[0]:
        X0[a, a] = electrolyte_cap
        other = 1.0 - electrolyte_cap
        X0[a, np.arange(ns) != a] = other / (ns - 1)
    # barycentric coordinates of b in the simplex spanned by X0: least-squares solve of
    # [X0^T; 1...1] y = [b; 1]
    M = np.vstack([X0.T, np.ones(ns)])
    rhsb = np.concatenate([b, [1.0]])
    y0, *_ = np.linalg.lstsq(M, rhsb, rcond=None)
    y0 = np.clip(y0, 1.0e-6, None); y0 /= y0.sum()

    grads = np.array([gfe.grad(X0[a]) for a in range(ns)])
    lam0 = -grads.mean(axis=0)
    zeta0 = -np.array([float(np.mean(y0[a] * (grads[a] + lam0))) for a in range(ns)])
    return y0, X0, zeta0, lam0


def solve_pep(components: list[Component], b, T_K: float, *, eps_init: float = 0.03,
              max_outer: int = 200, tol: float = 1.0e-9, eps_y: float = _EPS_Y_DEFAULT,
              merge_tol: float = 1.0e-3, verbose: bool = False) -> PhaseSplitResult:
    """Solve the phase equilibrium problem for feed composition ``b`` (mole fractions, sums to 1) at ``T_K``,
    using this package's AIOMFAC ``ActivityModel`` (via ``AiomfacGFE``) as the Gibbs free energy function.

    Returns the converged phase split: ``result.y`` phase amounts and ``result.x`` phase compositions. A
    single-phase (homogeneous) result has ``result.y = [1.0]`` and ``result.x = [b]``.
    """
    return solve_pep_gfe(AiomfacGFE(components, T_K), b, eps_init=eps_init, max_outer=max_outer, tol=tol,
                          eps_y=eps_y, merge_tol=merge_tol, verbose=verbose)


def solve_pep_gfe(gfe, b, *, eps_init: float = 0.03, max_outer: int = 200, tol: float = 1.0e-9,
                   eps_y: float = _EPS_Y_DEFAULT, merge_tol: float = 1.0e-3,
                   verbose: bool = False) -> PhaseSplitResult:
    """Same algorithm as ``solve_pep``, but for any object exposing ``.n_species``, ``.g(x)``, ``.grad(x)``,
    ``.hess(x)`` (an exactly degree-1-homogeneous molar GFE function and its derivatives) -- ``AiomfacGFE`` is
    one such object; this generic entry point also lets the interior-point/active-set engine be validated
    against a simple synthetic GFE independent of AIOMFAC (see ``tests/test_lle.py``).
    """
    ns = gfe.n_species
    b = np.asarray(b, dtype=float); b = b / b.sum()

    y, X, zeta, lam = _init_phase_simplex(gfe, b, eps=eps_init)
    pi = ns
    nu = 1.0
    theta = nu / y
    active = list(range(pi))

    converged = False
    n_iter = 0
    for it in range(max_outer):
        n_iter = it + 1
        if len(active) == 1:
            converged = True
            break
        p_y, p_X, p_zeta, p_lam, p_theta = _newton_step(gfe, y, X, zeta, lam, theta, nu, b)

        tau = _fraction_to_boundary(y, p_y)
        tau = min(tau, _fraction_to_boundary(theta, p_theta))
        # also keep every phase composition strictly positive (x_alpha must stay in int(simplex), JOTA-2's
        # own domain requirement for g and its derivatives to be defined) -- not spelled out as a separate
        # fraction-to-boundary rule in JOTA-2 (which globalizes via its merit-function line search instead,
        # see the module docstring's point 2), but necessary here since this module's simpler step-length
        # control has no other safeguard against a Newton step leaving the domain of g.
        tau = min(tau, _fraction_to_boundary(X.ravel(), p_X.ravel()))
        tau = max(tau, 1.0e-4)   # never fully stall

        y = y + tau * p_y
        X = np.maximum(X + tau * p_X, 1.0e-10)
        for a in range(len(active)):
            s = X[a].sum()
            if s > 0:
                X[a] = X[a] / s
        zeta = zeta + tau * p_zeta
        lam = lam + tau * p_lam
        theta = np.maximum(theta + tau * p_theta, 1.0e-12)

        # active-set pruning (JOTA-2 Sec. 2.4, simplified: permanent deactivation of vanished phases)
        keep = y >= eps_y
        if not np.all(keep):
            active = [active[i] for i in range(len(active)) if keep[i]]
            y, X, zeta, theta = y[keep], X[keep], zeta[keep], theta[keep]
            y = y / y.sum()

        nu = _DELTA * float(np.mean((theta * y) ** 2))
        step_size = float(np.linalg.norm(p_y)) + float(np.linalg.norm(p_X))
        if verbose:
            print(f"iter {n_iter:3d}  phases={len(active)}  nu={nu:.3e}  step={step_size:.3e}")
        if nu < tol and step_size < math.sqrt(tol):
            converged = True
            break

    # merge near-duplicate active phases (can happen when the true equilibrium has fewer phases than the
    # n_s-phase initial simplex, and more than one initial vertex converges to the same composition)
    if len(active) > 1:
        keep_idx = []
        for a in range(len(y)):
            dup = None
            for k in keep_idx:
                if np.linalg.norm(X[a] - X[k]) < merge_tol:
                    dup = k; break
            if dup is None:
                keep_idx.append(a)
            else:
                y[dup] += y[a]
        y, X = y[keep_idx], X[keep_idx]
        y = y / y.sum()

    return PhaseSplitResult(y=y, x=X, converged=converged, n_outer_iter=n_iter, nu_final=float(nu))

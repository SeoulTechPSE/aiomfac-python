"""RH-dependent gas/particle partitioning, coupled to liquid-liquid phase equilibrium.

Ports the gas/particle partitioning part of:

    Zuend, A., Marcolli, C., Peter, T., Seinfeld, J. H. (2010), "Computation of liquid-liquid equilibria and
    phase stabilities: implications for RH-dependent gas/particle partitioning of organic-inorganic aerosols",
    Atmos. Chem. Phys., 10, 7795-7820, doi:10.5194/acp-10-7795-2010. [cited below as "Z2010"]

What this implements, and how closely (read this before trusting the numbers)
-------------------------------------------------------------------------------
Z2010 Sect. 3 poses gas/particle partitioning as a fixed-point problem: each semivolatile species ``j`` splits
between the gas phase and the particulate matter (PM) phase(s) so that its gas-phase partial pressure matches
the PM-phase equilibrium vapour pressure given by modified Raoult's law, ``p_j = a_j^PM * p_j^0`` (their Eq. 21
combines this with the total-moles balance into the "partitioning coefficient" ``K_j^PM``, Eq. 22; ``C_j^*`` is
the corresponding effective saturation concentration, Eq. 23-24). Crucially, at a converged liquid-liquid
equilibrium ``a_j`` is -- by construction of ``lle.solve_pep`` -- equal in every coexisting PM phase (that is
exactly what LLE means), so ``a_j^PM`` in the Raoult's-law expression is unambiguous even when the PM splits
into two phases; this module relies on that.

Z2010 solves the resulting nonlinear system (water content set by the RH boundary condition simultaneously
with every organic's gas/particle split, Sect. 3.2 steps 1-8) with a Differential-Evolution/Powell/
Levenberg-Marquardt combination (their Appendix A) chosen for robustness on an arbitrary mixture. This module
offers two solvers, selected via ``gp_partition(..., method=...)``:

* ``method="lm"`` (the **default**): a joint Levenberg-Marquardt/trust-region solve
  (`scipy.optimize.least_squares`, ``method="trf"``) over *all* unknowns at once -- ``log(n_water)`` and every
  organic's particle-phase fraction ``r_j`` -- with residuals built directly from the same two closure
  conditions Z2010's own objective function uses (their ``F_obj = sum_j dn_j^2 + da_w^2``): each organic's
  mass-balance closure, plus the water-activity closure ``a_w - RH``. This is much closer in spirit to Z2010's
  own local-refinement step (their Powell/LM stage) than plain fixed-point iteration is, because it uses the
  full Jacobian coupling every species simultaneously rather than updating one species at a time -- see
  "Known convergence failure mode" below for why that distinction matters in practice, not just in principle.
* ``method="successive_substitution"``: the original plain **successive substitution** (a damped fixed-point
  iteration: solve for the particle water content that matches the target RH at the current organic split via
  1-D bracketing (`scipy.optimize.brentq`), then update every organic's gas/particle split from the resulting
  activities, damp, repeat) -- the standard, much simpler iterative scheme used throughout the
  absorptive-partitioning literature the paper itself builds on (Pankow, 1994; Odum et al., 1996). Kept
  available for comparison/regression testing against ``"lm"``, since it is simpler to reason about when it
  *does* converge, but it is demonstrably not reliable on its own for the paper's own case study (below) and
  should not be relied on for an arbitrary mixture. In both cases, ``GPResult.converged`` should be checked,
  not assumed.

Known convergence failure mode of plain successive substitution (confirmed, fixed by ``method="lm"``)
----------------------------------------------------------------------------------------------------------
For a mixture of organics with substantially different volatilities, plain successive substitution
(``method="successive_substitution"``) can fail to settle the two (or more) *most volatile* species at all:
rather than smoothly approaching some intermediate gas/particle split, their particle-phase amount chaotically
alternates, iteration to iteration, between "fully evaporated" (at the numerical floor) and "mostly condensed"
(a material fraction of their own total), because right at that knife-edge, each organic's own activity
coefficient is extremely sensitive to the other volatile organic's current (also-chattering) amount -- a
genuine fixed-point instability of updating one species at a time, not a tolerance or floor artifact.
Confirmed directly for `03_zuend2010_lle.ipynb`'s own Fig. 8-10 system (water + glycerol + 1,6-hexanediol +
1,2,10-decanetriol + 1,2,5,8-octanetetrol + (NH4)2SO4): with ``method="successive_substitution"``, glycerol
and 1,6-hexanediol (the two most volatile of the four organics) never settle at *any* of the notebook's 8 RH
points even after hundreds of iterations and with much smaller relaxation, while 1,2,10-decanetriol and
1,2,5,8-octanetetrol (the two least volatile) converge cleanly within a handful of iterations every time --
``GPResult.converged`` reads ``False`` at every RH point for that system under this method.

``method="lm"`` (the default) actually resolves this, because the joint Jacobian sees the coupling between
the two volatile species directly instead of chasing it one variable at a time: re-running the identical
system with ``method="lm"`` converges cleanly (``GPResult.converged is True``, ``aw`` matching the target RH
to the requested tolerance) at every one of the same 8 RH points, with smooth, monotonic particle-phase
trajectories for all four organics, including the two that chattered under successive substitution -- see
``tests/test_gp_partition.py::TestGPPartitionJointLMRegression`` for the regression test that locks this in,
and `03_zuend2010_lle.ipynb`'s Fig. 8-10 section for the worked example.

Units: SI throughout (kg/mol for molar masses, Pa for pressures, m^3 for gas volume, mol for amounts,
kg/m^3 for mass concentrations) -- converted to the paper's usual micrograms/m^3 only for display.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import brentq, least_squares

from .io import Component
from .lle import AiomfacGFE, solve_pep, solve_pep_gfe
from .model import ActivityModel

R_GAS = 8.314462618  # J / (mol K)


@dataclass
class VolatileSpecies:
    """One semivolatile organic compound tracked for gas/particle partitioning."""
    component: Component
    M: float          # molar mass, kg/mol
    p0_Pa: float       # pure-compound saturation vapour pressure, Pa
    n_total: float     # total (gas + particle) moles, fixed by the case definition


@dataclass
class GPResult:
    RH: float
    converged: bool             # should normally be True with the default method="lm"; False indicates a
                                 # genuine, not cosmetic, non-convergence -- see the module docstring's "Known
                                 # convergence failure mode" for a confirmed case (under the older
                                 # method="successive_substitution") of a volatile organic's PM amount never
                                 # settling, which method="lm" resolves
    n_iter: int
    n_water_PM: float
    n_org_PM: np.ndarray        # (n_org,) particle-phase moles of each VolatileSpecies
    n_org_gas: np.ndarray       # (n_org,) gas-phase moles
    activities_org: np.ndarray  # (n_org,) organic activities at equilibrium (equal across PM phases)
    aw: float                   # achieved particle water activity (should match RH to `tol`)
    phases: object               # lle.PhaseSplitResult of the final particle composition


def _aw_forced_1phase(model: ActivityModel, T_K: float, n_water: float, n_org_PM: np.ndarray, n_salt: float):
    """Fast water activity at a composition, forcing a single (notionally homogeneous) phase -- a plain
    forward ``ActivityModel.evaluate`` call, no Newton/active-set iteration. Used for the bulk of the
    successive-substitution iteration (Z2010 Sect. 3.2's own "forced 1-phase" fast path); the caller is
    responsible for a final LLE-aware refinement where phase separation actually matters (see
    ``gp_partition``)."""
    n = np.concatenate([[n_water], n_org_PM, [n_salt]])
    x = n / n.sum()
    res = model.evaluate(x, T_K, basis="mole")
    return float(res.activity[0]), res.activity


def _aw_lle(gfe: AiomfacGFE, n_water: float, n_org_PM: np.ndarray, n_salt: float, tol: float = 1.0e-6,
            max_outer: int = 80):
    """Water activity resolved through the full LLE solver (true multi-phase equilibrium value, identical in
    every coexisting phase) -- slower, used only for the final refinement in ``gp_partition``."""
    n = np.concatenate([[n_water], n_org_PM, [n_salt]])
    b = n / n.sum()
    res = solve_pep_gfe(gfe, b, tol=tol, max_outer=max_outer)
    a = gfe.activities(res.x[0])
    return float(a[0]), a, res


def _raoult_gas_moles(a_org, organics: list[VolatileSpecies], T_K: float, V_gas_m3: float) -> np.ndarray:
    """Modified Raoult's law (Z2010 Eq. 21): gas-phase equilibrium moles of each organic given its PM-phase
    (mole-fraction) activity."""
    return np.array([(a_org[j] * organics[j].p0_Pa) * V_gas_m3 / (R_GAS * T_K) for j in range(len(organics))])


def _successive_substitution(aw_func, organics: list[VolatileSpecies], n_salt: float, T_K: float, RH: float,
                              V_gas_m3: float, n_org_PM0: np.ndarray, n_water0: float, total_scale: float,
                              n_water_bracket, *, max_iter: int, relax: float, tol: float,
                              brentq_kw: dict):
    """One successive-substitution run (shared by the fast forced-1-phase pass and the LLE-aware refinement
    pass): find the particle water content matching ``RH`` by 1-D bracketing, apply Raoult's law to update
    every organic's gas/particle split, damp, repeat. ``aw_func(n_water, n_org_PM) -> (aw, extra)``."""
    n_org = len(organics)
    n_total_arr = np.array([o.n_total for o in organics])
    n_org_PM = n_org_PM0.copy()
    n_water = n_water0
    n_org_PM_prev2, n_water_prev2 = None, None  # state two iterations back, for period-2 oscillation check
    extra = None
    converged = False
    it = 0
    lo, hi = n_water_bracket
    # A species driven to near-total evaporation (its particle-phase amount a tiny fraction of its own
    # n_total) has an activity that is numerically delicate right near zero mole fraction: plain successive
    # substitution can make its absolute PM amount chatter indefinitely between e.g. 1e-23 and 1e-9 mol
    # without ever settling, even though *physically* it has simply (and stably) evaporated -- the chatter is
    # well below anything that matters for the gas/particle split of the other, non-negligible species. Such
    # a species is excluded from the step/tolerance test below (its value is still tracked and returned, just
    # not allowed to block convergence); the threshold is relative to each species' own total, not the
    # mixture's total_scale, since "negligible" is a per-species notion here.
    negligible_frac = 1.0e-4
    for it in range(1, max_iter + 1):
        def f_aw(nw):
            aw, _ = aw_func(nw, n_org_PM)
            return aw - RH
        flo, fhi = f_aw(lo), f_aw(hi)
        if flo > 0:      # even the driest bracket is too wet -> essentially no water (organics-only system)
            n_water_new = lo
        elif fhi < 0:
            n_water_new = hi
        else:
            n_water_new = brentq(f_aw, lo, hi, **brentq_kw)

        aw, extra = aw_func(n_water_new, n_org_PM)
        a_org = extra[1:1 + n_org] if hasattr(extra, "__len__") else extra

        n_org_gas_eq = _raoult_gas_moles(a_org, organics, T_K, V_gas_m3)
        n_org_PM_new = np.clip(np.array([o.n_total for o in organics]) - n_org_gas_eq, 1.0e-16 * total_scale, None)

        material = (n_org_PM_new / n_total_arr) >= negligible_frac
        d1 = np.abs(n_org_PM_new - n_org_PM)
        step = (np.max(d1[material]) if material.any() else 0.0) + abs(n_water_new - n_water)
        # Also check the step relative to the state *two* iterations back: even among the "material" species,
        # plain successive substitution can settle into a tiny-amplitude period-2 oscillation that never
        # satisfies the single-step `tol` test above even though the solution has, for all practical purposes,
        # stopped moving. Requiring *either* criterion avoids reporting `converged=False` forever on a
        # genuinely-settled oscillation.
        if n_org_PM_prev2 is None:
            step2 = np.inf
        else:
            d2 = np.abs(n_org_PM_new - n_org_PM_prev2)
            step2 = (np.max(d2[material]) if material.any() else 0.0) + abs(n_water_new - n_water_prev2)
        n_org_PM_prev2, n_water_prev2 = n_org_PM.copy(), n_water
        n_org_PM = n_org_PM + relax * (n_org_PM_new - n_org_PM)
        n_water = n_water_new
        if step / total_scale < tol or step2 / total_scale < tol:
            converged = True
            break
    return n_water, n_org_PM, converged, it


def _solve_joint_lm(aw_func, organics: list[VolatileSpecies], T_K: float, RH: float, V_gas_m3: float,
                     n_org_PM0: np.ndarray, n_water0: float, total_scale: float, n_water_bracket, *,
                     max_iter: int, tol: float):
    """Joint Levenberg-Marquardt/trust-region solve of the full coupled system, mirroring Z2010 Sect. 3.2's
    own numerical approach (DE + Levenberg-Marquardt minimizing ``F_obj = sum_j(Delta n_j)^2 + (Delta a_w)^2``,
    Appendix A) far more closely than ``_successive_substitution`` does. Unknowns are the particle-phase
    fraction ``r_j = n_j^PM / n_j^t`` of every organic (bounded to [0, 1]) and ``log(n_water)`` (log-scaled to
    keep it positive over many orders of magnitude); residuals are the per-organic mass-balance closure
    ``(n_j^t - n_j^PM - n_j^g) / total_scale`` (Z2010's Delta n_j, Eq. in Sect. 3.2 step 8) and the water
    activity closure ``a_w - RH`` (Delta a_w) -- i.e. driving every unknown to self-consistency *simultaneously*
    via a damped Gauss-Newton step on the full Jacobian, rather than updating one block (organics) and then the
    other (water) in alternation as plain successive substitution does. This is what actually fixes the
    "Known convergence failure mode" documented in the module docstring: a volatile organic chattering between
    near-zero and a material particle-phase amount is precisely a case where the *coupling* between that
    organic's own equation and the others' matters, which a joint Newton-type step accounts for directly
    (through the Jacobian) and a one-variable-at-a-time fixed point cannot.
    """
    n_org = len(organics)
    n_total_arr = np.array([o.n_total for o in organics])
    lo_w, hi_w = n_water_bracket
    log_lo, log_hi = np.log(lo_w), np.log(hi_w)

    def residuals(u):
        log_nw = u[0]
        r = u[1:]
        n_water = np.exp(log_nw)
        n_org_PM = r * n_total_arr
        aw, extra = aw_func(n_water, n_org_PM)
        a_org = extra[1:1 + n_org] if hasattr(extra, "__len__") else extra
        n_org_gas = _raoult_gas_moles(a_org, organics, T_K, V_gas_m3)
        res_n = (n_total_arr - n_org_PM - n_org_gas) / total_scale
        res_aw = np.array([aw - RH])
        return np.concatenate([res_n, res_aw])

    r0 = np.clip(n_org_PM0 / n_total_arr, 0.0, 1.0)
    u0 = np.concatenate([[np.log(np.clip(n_water0, lo_w, hi_w))], r0])
    lb = np.concatenate([[log_lo], np.zeros(n_org)])
    ub = np.concatenate([[log_hi], np.ones(n_org)])

    # Deliberately *not* `xtol=tol, ftol=tol, gtol=tol`: TRF's own internal termination tests (especially
    # `gtol`, the projected-gradient test at the active bounds) can fire at a bound-constrained pseudo-
    # stationary point -- e.g. a step reflected off `r_j=1` where the *projected* gradient looks flat even
    # though the true residual is nowhere near zero -- well before the mass-balance/water-activity residual
    # is actually small. Confirmed directly: at a high-RH point (one organic's PM fraction pinned near its
    # r_j=1 bound), passing a loose caller `tol` (e.g. 1e-4, as the RH-sweep notebook uses) straight through to
    # `gtol` made TRF stop after only ~9 function evaluations at `aw` still ~0.01 off the target RH, regardless
    # of how large `max_nfev` was budgeted -- i.e. a genuine internal-tolerance stall, not a budget shortfall.
    # Using fixed, tight internal tolerances here (independent of the caller's `tol`) lets TRF keep stepping
    # until the geometry actually resolves; the caller's `tol` still governs the one test that matters --
    # whether the *returned* residual vector is actually small (below) -- so a loose `tol` only relaxes what
    # counts as "good enough", never how hard the optimizer tries to get there.
    internal_tol = min(tol, 1.0e-10)
    result = least_squares(residuals, u0, bounds=(lb, ub), method="trf",
                            xtol=internal_tol, ftol=internal_tol, gtol=internal_tol,
                            max_nfev=max_iter * (n_org + 2))

    n_water = float(np.exp(result.x[0]))
    n_org_PM = np.clip(result.x[1:], 0.0, 1.0) * n_total_arr
    # "Converged" here means the residual vector (mass-balance closures + water-activity closure) is small in
    # the same normalized sense `_successive_substitution`'s `tol` uses, not just that the optimizer's own
    # internal termination test fired (which, per the comment above, can trigger on a stalled step size even
    # far from a root).
    converged = bool(np.max(np.abs(result.fun)) < tol)
    return n_water, n_org_PM, converged, result.nfev


def gp_partition(salt: Component, organics: list[VolatileSpecies], n_salt: float, T_K: float, RH: float,
                  V_gas_m3: float, *, max_iter: int = 60, relax: float = 0.5, tol: float = 1.0e-6,
                  n_water_bracket: tuple[float, float] | None = None,
                  check_lle: bool = True, method: str = "lm") -> GPResult:
    """Solve the RH-dependent gas/particle split of ``organics`` (plus water) over a fixed, non-volatile
    ``salt`` amount ``n_salt``, at temperature ``T_K`` and target relative humidity ``RH`` (0-1).

    ``method``:

    * ``"lm"`` (the default): a joint Levenberg-Marquardt/trust-region solve of the full coupled system
      (``_solve_joint_lm``) -- much closer to Z2010 Sect. 3.2's own DE + Levenberg-Marquardt approach, and the
      fix for the "Known convergence failure mode" the module docstring documents (a volatile organic
      chattering between near-zero and a material particle-phase amount under plain successive substitution).
      ``relax`` is ignored in this mode (there is no fixed-point damping step to relax).
    * ``"successive_substitution"``: the original damped fixed-point iteration (bracket the particle water
      content matching ``RH`` via 1-D root-finding, update every organic's split via Raoult's law, damp,
      repeat). Kept for comparison/regression testing against the ``"lm"`` default, and because it is cheaper
      per-iteration for a system that is not near the failure mode (most single- or few-organic cases).

    Water is treated as a semi-infinite gas-phase reservoir fixed at the target RH (Z2010 Sect. 3's own
    boundary condition).

    For speed, both methods use a *forced single phase* throughout (a plain forward AIOMFAC evaluation, no
    Newton/active-set LLE solve at every iteration) -- each ``lle.solve_pep`` call on a handful of species
    costs whole seconds, not milliseconds (the phase-simplex initializer alone starts one Newton iterate per
    species), so re-solving it at every trial composition would make an RH sweep impractically slow. Instead,
    once the fast forced-1-phase iteration has converged, a *single* ``lle.solve_pep`` call is made at that
    final composition (when ``check_lle`` is true, the default) purely as a **diagnostic**: it reports
    (``GPResult.phases``) whether the converged composition actually sits inside a miscibility gap, without
    feeding that back into another round of partitioning. When it does, the returned ``aw``/``activities_org``
    are still the *forced-1-phase* values, not the true multi-phase ones -- a known, deliberate accuracy/speed
    trade-off (the two differ only by how much non-ideality the true phase split adds/removes, typically a
    modest correction relative to the RH-driven trend this module is used to illustrate). Set
    ``check_lle=False`` to skip even the diagnostic call.
    """
    components = [Component(1, "water", ((16, 1),))] + [o.component for o in organics] + [salt]
    model = ActivityModel(components)
    n_org = len(organics)

    n_org_PM0 = np.array([o.n_total for o in organics], dtype=float)   # start: fully condensed
    total_scale = max(n_salt + n_org_PM0.sum(), 1.0e-300)
    n_water0 = n_org_PM0.sum() if n_org_PM0.sum() > 0 else 1.0e-6 * total_scale
    if n_water_bracket is None:
        n_water_bracket = (1.0e-6 * total_scale, 1.0e6 * total_scale)

    def aw_fast(nw, n_org_PM):
        return _aw_forced_1phase(model, T_K, nw, n_org_PM, n_salt)

    if method == "lm":
        n_water, n_org_PM, converged, it = _solve_joint_lm(
            aw_fast, organics, T_K, RH, V_gas_m3, n_org_PM0, n_water0, total_scale, n_water_bracket,
            max_iter=max_iter, tol=tol)
    elif method == "successive_substitution":
        brentq_kw = dict(xtol=1.0e-9 * total_scale, rtol=1.0e-7, maxiter=60, disp=False)
        n_water, n_org_PM, converged, it = _successive_substitution(
            aw_fast, organics, n_salt, T_K, RH, V_gas_m3, n_org_PM0, n_water0, total_scale, n_water_bracket,
            max_iter=max_iter, relax=relax, tol=tol, brentq_kw=brentq_kw)
    else:
        raise ValueError(f"method must be 'lm' or 'successive_substitution', got {method!r}")

    # final activities/water activity: always the fast forced-1-phase values (see docstring) -- a single
    # forward AIOMFAC evaluation, not another solve_pep call.
    b_final = np.concatenate([[n_water], n_org_PM, [n_salt]])
    b_final = b_final / b_final.sum()
    a_full = model.evaluate(b_final, T_K, basis="mole").activity
    a_org = a_full[1:1 + n_org]
    aw_final = float(a_full[0])

    phases = None
    if check_lle:
        gfe = AiomfacGFE(components, T_K)
        # one-shot diagnostic only (see docstring): does this converged composition actually sit inside a
        # miscibility gap? Not fed back into another round of partitioning.
        phases = solve_pep_gfe(gfe, b_final, tol=1.0e-5, max_outer=60)

    n_org_gas = np.array([max(o.n_total - n_org_PM[j], 0.0) for j, o in enumerate(organics)])

    if phases is None:
        from .lle import PhaseSplitResult
        x_row = np.concatenate([[n_water], n_org_PM, [n_salt]])
        x_row = x_row / x_row.sum()
        phases = PhaseSplitResult(y=np.array([1.0]), x=x_row[None, :], converged=True, n_outer_iter=0,
                                   nu_final=0.0)

    return GPResult(RH=RH, converged=converged, n_iter=it, n_water_PM=n_water, n_org_PM=n_org_PM,
                     n_org_gas=n_org_gas, activities_org=a_org, aw=aw_final, phases=phases)


def distinct_phases(phases, y_min: float = 1.0e-3, dx_min: float = 1.0e-3):
    """Indices of the ``phases`` (an ``lle.PhaseSplitResult``) that are real, non-negligible, mutually
    distinct phases -- discarding near-zero-weight or near-duplicate phases that are occasionally a
    numerical artifact of the interior-point solver lingering near a boundary rather than a true phase
    split (see ``lle.py``'s module docstring, and ``binodal_sweep``'s ``n_real_phases`` in
    ``02_reproduce_zuend2008.ipynb`` for the same filter used there)."""
    keep = [i for i in range(len(phases.y)) if phases.y[i] > y_min]
    if len(keep) <= 1:
        return keep if keep else [int(np.argmax(phases.y))]
    out = []
    for i in keep:
        dup = False
        for j in out:
            if np.max(np.abs(phases.x[i] - phases.x[j])) < dx_min:
                dup = True
                break
        if not dup:
            out.append(i)
    return out


def c_star_OA(species: VolatileSpecies, gamma_OA: float, T_K: float) -> float:
    """C*^OA (Z2010 Eq. 23-ish "organic-only" definition, e.g. Donahue et al. 2006): the effective saturation
    mass concentration (kg/m^3) a compound would have if the absorbing phase were the organic mixture alone,
    with organic-only activity coefficient ``gamma_OA``."""
    return gamma_OA * species.p0_Pa * species.M / (R_GAS * T_K)


def c_star_PM(species: VolatileSpecies, gamma_PM: float, x_org_in_PM_basis: float, M_PM_mean: float,
              T_K: float) -> float:
    """C_j* generalized to the whole PM phase including inorganics and water (Z2010 Eq. 24): same form as
    ``c_star_OA`` but referenced to the full PM-phase mean molar mass/activity coefficient rather than an
    organic-only sub-phase."""
    return gamma_PM * species.p0_Pa * M_PM_mean / (R_GAS * T_K)

# Combined liquid–liquid–solid equilibrium solver (`aiomfac_py.phase_equilibrium`)

Implementation: `src/aiomfac_py/phase_equilibrium.py` (`LiquidModel`, `PhaseEquilibrium`, `PhaseEquilibriumResult`),
tests: `tests/test_phase_equilibrium.py` (40 tests). Branch `feature/phase-equilibrium` (commits 3f1f230 onward).

Like `aiomfac_py.lle` and `aiomfac_py.sle`, this module is **not** part of the Fortran AIOMFAC code, which provides
activities only. It is therefore not Fortran-validated. It is checked against the other solvers of this package and
against the equilibrium conditions it reports for every result (Sect. 9–10).

---

## 1. Purpose and scope

`PhaseEquilibrium` computes the equilibrium state of an aerosol particle made of water, any number of neutral organic
compounds and inorganic ions at fixed temperature T and relative humidity RH. The state may contain:

* one or more **liquid phases**, each with water, organics and ions (liquid–liquid phase separation, LLPS);
* any subset of the **crystalline salts** of the `aiomfac_py.solids` database;
* optionally a **gas phase** for NH3, HNO3, HCl and CO2 (open system at fixed partial pressure, or closed system with a
  given amount of air).

Liquids, solids and gases are found together in **one** minimization of a transformed Gibbs energy. They are not
found by alternating an LLE solver and an SLE solver. Precipitation changes the ionic strength, and with it the
liquid–liquid split, and the two have to adjust at the same time.

| Feature | Supported | Notes |
|---|---|---|
| Neutral organics | any AIOMFAC `Component` (e.g. from `s2as.smiles_to_components`) | |
| Non-reactive ions | Li+, Na+, K+, NH4+, Mg2+, Ca2+ / F-, Cl-, Br-, I-, NO3-, SO4-- (as in `solids.ION_REGISTRY`) | |
| Acid sulfate | H+ and SO4-- as stoichiometric components | HSO4- is an explicit species of every liquid (Sect. 3.4); `speciation="internal"`: speciated inside the activity evaluation (Sect. 3.2) |
| Carbonate | CO3-- (total carbonate) and H+ (proton excess, either sign) | HCO3-, OH-, CO2(aq) are explicit species (Sect. 3.4); `speciation="internal"`: Sect. 3.3 |
| Solids | all database solids whose ions are present, or a user list | hydrates through the hydrate water h |
| Gases | NH3, HNO3, HCl, CO2 | open (`p_gas`) or closed (`gas_total`, `n_air`) |
| Liquid phases | up to `max_liquids` (default 3) | added by a tangent-plane stability test |
| Modes | equilibrium (`solids="all"`), metastable (`"none"`), explicit list, drying path with critical supersaturation | |

### Why a new solver

`lle.solve_pep` relies on multistart initial splits. At low water content (dry, salt-rich particles)
and close to saturation it can miss LLPS that exists. An example is pinic acid + ammonium sulfate at RH 0.10–0.45: the
two-liquid state found here has a lower value of `lle`'s own Gibbs function `AiomfacGFE` than the one-liquid state
returned by `lle`.

The new solver decides on the number of phases from a tangent-plane distance (TPD) test of the converged state. It
reports the result as converged only when the stationarity conditions hold (Sect. 9).

---

## 2. Formulation

### 2.1 Variables and the ion basis

The species of a liquid are water, the neutral organics and the **individual ions** (the "ion basis"). With
`n_liq` liquid phases, `S` candidate solids and `K` gases, the unknowns are

* `n_αi > 0`: the amount of species i in liquid α (the proton excess of a carbonate system may take either sign);
* `u_s ≥ 0`: the amount of solid s;
* `g_k`: the amount of gas k. It is the net release to a fixed-pressure reservoir in the open system (either sign),
  and the amount in the gas phase in the closed system (> 0).

Working with individual ions, not with salt formulas, means that every liquid can hold any electroneutral mixture of
the ions. No choice of "electrolyte components" is needed, and a solid is a fixed linear combination of ions.

### 2.2 Objective

At fixed T and RH, water exchanges with the gas phase at a_w = RH. Water is therefore an open component, and the
function minimized is the Legendre-transformed Gibbs energy (in units of RT):

```
F = Σ_α [ Σ_i n_αi ln a_αi − n_αw ln RH ]
  + Σ_s u_s ( ln K_s − h_s ln RH )
  + F_gas(g)
```

* `a_αi` are AIOMFAC activities: mole-fraction scale for water and organics (AIOMFAC's dissociated-basis mole
  fraction), molal scale for ions.
* Standard potentials of the ions are set to zero. They cancel through the mass balance, because every
  combination of ions that can be exchanged between phases, solids and gas is electroneutral.
* Each liquid's Gibbs energy is homogeneous of degree one in its amounts. Hence ∂F/∂n_αi = ln a_αi, and ln a_αw − ln RH
  for water.

**Solids.** The dissolution of solid s is `s → Σ ν_is ion_i + h_s H2O`. At saturation,
`ln K_s = Σ ν_is ln a_i + h_s ln a_w`. With zero ion standard potentials and water at ln RH, the potential of one mole
of solid is the constant `c_s = ln K_s − h_s ln RH`, so solids enter F linearly. The saturation index of solid s in
liquid α is

```
SI_s = Σ_i ν_is ln a_αi + h_s ln RH − ln K_s .
```

**Gases.** A gas is described by the reaction `gas ⇌ Σ ν_ik ion_i (+ h_k H2O)` with
`Σ ν_ik ln a_i + h_k ln a_w = ln K_k + ln p_k` (`aiomfac_py.gases`; `ions` lists the ions that disappear when one mole of
gas forms). Then:

* **open** (`p_gas`): the reservoir pressure is fixed, so one mole released costs the constant
  `c_k = ln K_k + ln p_k − h_k ln RH`. `F_gas = Σ g_k c_k` is linear, and g_k has no sign constraint;
* **closed** (`gas_total`, `n_air`, `P_atm`): the gas is an ideal mixture with `n_air` mol of inert air,

  ```
  F_gas = Σ_k g_k ( ln K_k − h_k ln RH + ln P_atm + ln y_k ) + n_air ln( n_air / N ),
  y_k = g_k / N,  N = n_air + Σ g_k,
  ```

  with Hessian `diag(1/g) − 1/N`.

### 2.3 Constraints

All constraints are linear:

```
Σ_α n_αi + Σ_s ν_is u_s + Σ_k ν_ik g_k = b_i       for every non-water species i   (mass balance)
Σ_i z_i n_αi = 0                                    for every liquid α              (electroneutrality)
```

`b` is the feed. In the closed gas mode, b includes the volatile totals. In the open mode, b is the particle feed,
and the released amounts leave the system. Water has no mass balance (it is open). The problem is scaled by
`Σ|b_i|` before solving.

### 2.4 Equilibrium conditions

At a stationary point of F under these constraints (with multipliers λ_i for the mass balances and ψ_α for the charge
of each liquid):

1. every liquid has a_w = RH;
2. every neutral organic has the same ln a in all liquids where it is present;
3. every ion satisfies `ln a_αi = λ_i + z_i ψ_α`. Single-ion potentials may differ between liquids by the phase
   potential z_i ψ_α, but every electroneutral combination of ions has the same potential in all liquids;
4. every solid has `SI_s ≤ 0`, and `SI_s = 0` when `u_s > 0` (complementarity);
5. every gas satisfies its equilibrium relation in every liquid (open mode: at the given p_k; closed mode: at
   `p_k = P_atm y_k`);
6. the set of liquids is stable: no trial liquid has a negative tangent-plane distance (Sect. 6.2).

Conditions 1–5 are first-order (KKT) conditions. Condition 6 separates the global minimum from other stationary
points. Since AIOMFAC's Gibbs energy is not convex, a stationary point with the wrong number of phases is common.

---

## 3. Liquid-phase activities (`LiquidModel`)

`LiquidModel(organics, ions)` builds an `ActivityModel` whose components are water, the organics and electrolyte
pseudo-components that make all ions known to AIOMFAC: one salt for each cation with the first anion, and one for the
first cation with each further anion. `ln_a(n, T)` maps species amounts to activities:

1. neutral mole fractions `x_n = n_neutral / Σ n_neutral`, solvent mass `Σ n_neutral M`, ion molalities
   `m_i = n_i / solvent mass`;
2. internal speciation, if the phase is an acid or carbonate system (below);
3. AIOMFAC LR + MR + SR activity coefficients at those molalities (`ActivityModel.lr_mr_sr`);
4. output: `ln a = ln γ + ln x` for the neutrals, and `ln a = ln γ + ln m` for the ions, with AIOMFAC's molal ion
   activity coefficients (including the `− tmolal` conversion term).

The test `test_liquid_model_matches_activity_model_and_gibbs_duhem` checks that the neutral activities and
`2 ln a_NH4 + ln a_SO4` equal `ActivityModel.evaluate` for the same mixture to 1e-12. It also checks that the Hessian
satisfies Gibbs–Duhem (`H n ≈ 0`).

### 3.1 Hessian

AIOMFAC provides no analytic Jacobian. The Hessian `∂ ln a_i / ∂ n_j` of a liquid is assembled from two parts
(`LiquidModel.hessian_split`, the default `hess_scheme = "split"`):

* **ideal part, exact** (`ln_ideal`, `ideal_jacobian`): `ln n_i − ln Σ n` for the neutrals and `ln n_i − ln(solvent
  mass)` for the ions. All 1/n curvature of trace species is in this part, and it is recomputed at every Newton step;
* **excess part, by forward differences** (`hessian_excess`): the remainder `ln a − ln_ideal` (activity coefficients
  plus the speciation correction of acid and carbonate systems) is a smooth function of the composition. It is
  differenced with a step of 1e-6 of the phase size for every bounded species, trace species included (N activity
  evaluations instead of 2N). Two exceptions: the **speciated components** (H+ and SO4-- of an acid system, CO3-- and
  SO4-- of a carbonate system) use a step of 1e-6 of their own amount, because their speciation depends on the ratios
  of their amounts, which may all be traces in an organic-rich liquid (a phase-size step was larger than the trace
  amounts and gave errors of 50 % in those columns); the sign-free carbonate proton excess keeps a central
  difference.

The excess part is **reused** while the liquid changes little: in the inner iteration it is refreshed when an amount
has changed by more than `hess_reuse_tol` = 0.02 of the liquid's size since it was computed (or when entries are
removed, Sect. 5.1); in the stability test, which starts far from its minima, after `tpd_hess_reuse_tol` = 0.02. The
gradient is always exact, so a reused Hessian changes the convergence rate, not the solution. In carbonate systems the
excess part is recomputed at every step: there the potentials are only approximately a gradient (Sect. 3.3), steps are
often accepted on the reduced-gradient norm, and that needs a current Jacobian. The sum is
symmetrized. `hess_scheme = "central"` restores the previous scheme (central differences of ln a at every step, step
`1e-5 · n_j`); the test `test_hessian_schemes_give_the_same_equilibrium` checks that both give the same state.

**Exact Hessian by automatic differentiation** (`hess_scheme = "ad"`, optional, needs `jax`). The module
`aiomfac_py.ad_activity` is a JAX transcription of `ExplicitLiquidModel.ln_a`: the LR, MR and SR terms, the conversion
from molalities to AIOMFAC's mole fractions and the CO2(aq) salting-out term, written with the same branches as the
NumPy code (`jnp.where` for the exponential cut-offs of the MR coefficients; the Qcca and Rcc sums as `einsum`). The
residual SR reference values, which do not depend on the composition, are precomputed with the NumPy code. Logarithms
of amounts are written as differences (ln n_i − ln(solvent mass), ...), so a species with zero amount only gives
non-finite entries in its own row, which is discarded. `jax.jacfwd` gives the full Jacobian ∂ ln a/∂n in one
forward-mode pass, which is jit-compiled once per system and temperature (cached at module level, so child problems
and repeated solves reuse it). The values used by the solver are still those of the NumPy code (the version validated
against the Fortran model); only the Hessian comes from JAX. With `"ad"` the Hessian is exact at every Newton step of
the inner problem and of the stability test, so nothing is reused. The scheme applies to the explicit liquid model
only; with `speciation="internal"` or without `jax`, `"ad"` falls back to `"split"`.

Checks (`test_ad_activities_and_jacobian_match_numpy`, and on six systems including the Qcca/Rcc terms of NH4+ + H+
and the carbonate species): ln a agrees with `ExplicitLiquidModel.ln_a` to 4e-14, and the Jacobian agrees with
fourth-order central differences to 1e-12 relative. That difference grows with smaller steps, as round-off in the
differences does; it is the error of the differences, not of the Jacobian. One Jacobian costs 0.04–0.09 ms, less than
one NumPy activity evaluation (0.14 ms), against N evaluations for the forward-difference excess part. Compilation
takes about 0.5–0.9 s per system and temperature, so `"ad"` pays off in sweeps (many solves of one system), not in a
single solve; `"split"` remains the default, which also keeps `jax` an optional dependency.

### 3.2 Acid sulfate

H+ and SO4-- are **stoichiometric** (total) components; passing HSO4- is an error. In every activity evaluation of a
phase that contains both, the bisulfate equilibrium HSO4- ⇌ H+ + SO4-- is solved (Knopf et al., 2003 constant, as in
`aiomfac_py.sle`). The free-ion molalities replace the totals before the activity coefficients are computed. The
potential of each stoichiometric component equals that of its free ion at the speciation equilibrium
(`μ_H,total = μ_H+`, `μ_SO4,total = μ_SO4--`). Because of this, the formulation of Sect. 2 needs no change: the
speciation is part of the phase's Gibbs function.

The speciation (`LiquidModel._speciate_hso4`) uses the method of `SLESolver`: at a fixed ratio
`q = K / (γ_H γ_SO4 / γ_HSO4)` the HSO4- molality is the smaller root of a quadratic; q is updated by fixed-point
iteration with secant acceleration, warm-started from the last solution (as a fraction of the largest possible HSO4-).
It needs about 2 AIOMFAC evaluations per call, against about 20 for the bracketing Brent solver of
`dissociation.solve_bisulfate`, which remains the fallback if the iteration does not converge. The terms of the last
evaluation are reused for the activities.

Tests: `test_acid_activities_match_activity_model_with_bisulfate_speciation` compares with `ActivityModel.evaluate`
including its own dissociation step (agreement 1e-10); `test_fast_bisulfate_speciation_matches_bracketing_solver`
compares the two speciation solvers on random compositions (1e-10).

### 3.3 Carbonate

A carbonate system has the components `CO3--`, carried as the total carbonate C_T = CO2(aq) + HCO3- + CO3--, and
`H+`, carried as the proton excess P = H+ − OH- + HCO3- + 2 CO2(aq) (+ HSO4-). P may be negative (a basic solution), so
H+ is a free (barrier-free) variable in this mode. `LiquidModel._ln_a_carb` generalizes `sle.AqueousIons._speciate_carb`
to phases that contain organics:

* equilibria: K1 = a_H a_HCO3 / (a_CO2 a_w), K2 = a_H a_CO3 / a_HCO3, Kw = a_H a_OH / a_w, and K_HSO4 if sulfate is
  present (constants from `aiomfac_py.carbonate` and `dissociation`);
* γ(CO2) from AIOMFAC's salting-out relation (`carbonate.gamma_co2_mr`);
* solution: a fixed-point iteration on the apparent constants. For given apparent constants, the proton balance is
  monotonic in ln m_H and is solved with Brent's method; the bracket is widened as needed. The carbonate fractions are
  evaluated with a log-sum-exp and [OH-] is capped, so the proton balance stays finite at extreme pH (the bracket can
  reach m_H = 1e-300). About 2 AIOMFAC evaluations per call (warm start). The apparent constants are
  updated from the new activity coefficients, until their ln changes by less than 1e-11. The last solution warm-starts
  the next call, and failed evaluations are never reused.

**Limitation.** AIOMFAC treats CO2(aq) with its own salting-out coefficient, outside the LR/MR/SR Gibbs function. With
organics present, the stoichiometric potentials are therefore only approximately the gradient of one Gibbs function:
Gibbs–Duhem holds to about 1 %. The inner solver accepts a step that reduces either F or the norm of the reduced
gradient (Sect. 5), and the final checks verify the equilibrium conditions directly. In the inorganic limit the
results equal `SLESolver` (Sect. 10).

---

### 3.4 Explicit speciation (`ExplicitLiquidModel`, `speciation="explicit"`, default)

The products of the speciation reactions are variables of every liquid, and their equilibria follow from the
minimization (plan item 4). The internal speciation of Sects. 3.2–3.3 instead solved an iteration inside every
activity evaluation, and inside every finite-difference perturbation.

* **Components and species.** The feed, mass balances, solids, gases and results stay in the component basis (H+ and
  SO4-- as totals; CO3-- as total carbonate and H+ as the proton excess). The species of a liquid are the components
  plus HSO4- (if H+ and SO4-- are present) and HCO3-, OH- and CO2(aq) (carbonate systems), appended after the
  components so that every component keeps its index. A matrix E maps species to components (HSO4- = H+ + SO4--,
  HCO3- = H+ + CO3--, CO2(aq) = 2 H+ + CO3-- − H2O, OH- = H2O − H+; water is open and has no balance). The mass-balance
  rows of the linear constraints are E applied to each liquid block, and electroneutrality uses the species charges.
* **Reaction constants.** `ln_a` of the explicit model returns the transformed potential μ_s = c_s + ln a_s, with
  c(HSO4-) = ln K_HSO4, c(HCO3-) = ln K2, c(CO2) = ln K1 + ln K2 + ln RH and c(OH-) = −(ln Kw + ln RH). The base species
  (H+, SO4--, CO3--) have c = 0, and water enters at its reservoir potential ln RH (`set_conditions`). Stationarity
  under the component balances then gives μ_HSO4 = μ_H + μ_SO4, μ_HCO3 = μ_H + μ_CO3, μ_CO2 = 2 μ_H + μ_CO3 and
  μ_OH = −μ_H, i.e. the same equilibrium conditions as Sects. 3.2–3.3.
* **Consequences.**
  * One AIOMFAC call per activity evaluation.
  * No noise from nested iterations in the finite-difference Hessian.
  * Exact Gibbs–Duhem for everything except CO2(aq). CO2(aq) is a molal solute with the salting-out activity
    coefficient and does not enter the other activities, so the ~1 % inconsistency of Sect. 3.3 remains.
  * No sign-free variable: the proton excess is a component total, and H+ and OH- are positive species.
* **Starting point.** The internal model speciates the starting liquid (`initial_species`), and species that come out
  zero get 1e-20 of the phase size.
* **Traces of reacting species.** Species that take part in a reaction (H+, SO4--, CO3-- and the extra species) are
  never removed as traces (Sect. 5.1). Their small amounts carry the equilibria, for example the free H+ of a liquid in
  which most acid is HSO4-, i.e. its pH. Removing that H+ broke the bisulfate equilibrium of the phase.
* **Barrier parameter.** Explicit H+ or OH- can be traces of 1e-14 of the feed and less. The final barrier stage
  therefore continues until μ ≤ 1e-10 · min x, so that the barrier shifts no potential by more than 1e-10. Stopping
  at μ = 1e-14 left the CO2(aq) equilibrium of a basic solution off by 1.7.
* **Results.** `LiquidPhase.amounts` and `ln_a` are in the component basis (the ln a of a component is that of its
  free species), as before. `species_names`, `species_amounts` and `species_ln_a` give the speciation.
* With the explicit model, carbonate systems with organics use the active-set solver and the successive-substitution
  stability test as well. The barrier solver and the Newton stability test are needed only with
  `speciation="internal"`.

Validation:
* Tests compare both formulations on DLT + NaCl + H2SO4 open to HCl and on pinic acid + NaCl + base open to CO2: the
  same phases, component amounts within 1e-5 and F within the removed traces. The reaction equilibria hold in every
  liquid (1e-6).
* The SLESolver limit tests (acid salts, carbonate with CO2) pass unchanged.
* DLT + NaCl + H2SO4 with HCl gives F = −0.08090260331, as 1c693be and 3b1dc3d. The internal formulation of later
  commits had removed a 6e-9 chloride trace there.

## 4. Problem reduction (child problems)

A species that is absent from the feed and cannot be supplied by any of the given gases would need an artificial
positive amount under the log barrier, and that amount distorts the mass balance. `solve` therefore drops such
species and solves a **child problem**: a `PhaseEquilibrium` with only the species present. Child problems are cached
in `self._children`. Gas ions count as reachable (HNO3 uptake can supply NO3- and H+), and so does the proton excess of a
carbonate system. The result's `names` lists the species actually used, and its `message` records the dropped ones.

---

## 5. Inner problem: log-barrier Newton in the null space

For a fixed number of liquids, `_barrier_solve` minimizes

```
Φ_μ(x) = F(x) − μ Σ_{bounded j} ln x_j      subject to   A x = A x0
```

where x stacks all liquid amounts, solid amounts and gas amounts, and A holds the mass balances and charge rows
(`_build_A`). Bounded entries are all liquid amounts except the carbonate proton excess, all solid amounts, and the
closed-mode gas amounts.

* **Null space.** `Z = null_space(A)` is computed once. Every step is `dx = Z d`, so the linear constraints hold exactly
  at every iterate, and electroneutrality and mass balance are satisfied to round-off (≈ 1e-15).
* **Newton system.** `H` is block-diagonal: one Hessian per liquid (Sect. 3.1), the closed-mode gas block, and the
  barrier term `μ / x_j²`. The objective, the activities and the gas terms of an accepted trial point are reused for
  the next Newton step and across barrier stages. The reduced Hessian `Zᵀ H Z` is diagonalized. If its smallest eigenvalue is not positive
  (a non-convex region of AIOMFAC), the spectrum is shifted to `1e-8 · max|λ|`. The result is a descent direction with
  the Newton step in the convex directions.
* **Step length.** A fraction-to-the-boundary rule (0.995 of the largest feasible step) is followed by backtracking.
  A step is accepted if it satisfies the Armijo condition on Φ_μ, **or** if it reduces the norm of the reduced gradient
  `‖Zᵀ ∇Φ_μ‖` by the same factor. The second criterion is needed close to convergence, where the decrease of Φ_μ falls
  below its numerical resolution (the speciation is solved iteratively), and in carbonate systems (Sect. 3.3). A guard
  that rejected steps increasing Φ_μ beyond 1e-10–1e-7 |Φ_μ| outside carbonate systems was tried. It caused line-search
  failures and spurious three-liquid attempts in DLT + NaCl + H2SO4, and it saved evaluations only in another case, so
  it was not adopted. Evaluations
  that raise floating-point errors (over/underflow in the speciation) count as rejected trial points.
* **Convergence at each μ.** Both conditions must hold:
  1. the Newton decrement `−∇Φᵀ dx < 1e-14 + 1e-10 μ len(x)`;
  2. the largest **relative** step of a bounded variable `max |dx_j| / x_j < 1e-8`.

  The second condition was added for trace species (commit 3a14629). A trace species has curvature ~1/n, so a tiny
  decrement can hide a large error in its ln a. An example is Cl- that is almost completely evaporated as HCl.
* **Barrier schedule.** μ starts at 1e-3 and is divided by 10 down to μ_min = 1e-14, with at most 60 Newton steps per
  μ. A failed line search ends the Newton iteration at that μ; it counts as converged only if the decrement is below
  1e-14.

### 5.1 Species absent from one liquid (trace removal)

With several liquids, the barrier keeps every species in every liquid. A species that is strongly excluded from one
liquid is driven down to amounts near the floating-point resolution, for example DLT in a concentrated salt liquid
(about 1e-16 of its total) or the last chloride after HCl evaporation. Its 1/n curvature makes the reduced Hessian
ill-conditioned (condition number above 1e17). Its contribution to F is far below the resolution of F, so the line
search cannot confirm a decrease and the potential conditions are left unmet.

After every barrier stage (each μ), `_truncate` therefore checks every entry of every liquid. An entry is removed when
three conditions hold:
* its amount is below `trace_tol` (1e-9 in scaled units, Σ|b| = 1);
* another liquid holds at least 10 times more of the species;
* it is not the carbonate components.

The amount is moved to that liquid, and the entry is fixed at zero and excluded from the null-space basis. Moving an
ion changes the charge of both liquids by at most 1e-9, so the linear constraints are restored by a least-change
correction weighted by the amounts. The stage is then repeated.

The removed entry stands for an equilibrium amount that is negligible for the mass balance. Its ln a is reported as
−∞, and the checks skip it. The ion-potential fit, the reference potentials of the TPD test and the SI use only the
liquids that contain the ion; an ion absent from the reference liquid is assigned `λ_i + z_i ψ_ref` from a gauge fit.
Removal is reversible. When the final barrier stage of `_newton_solve` has converged, `_reenter` estimates the
equilibrium amount of every absent entry. The potential of the species is taken from the other liquids (neutrals:
ln x = μ − ln γ; ions: ln m = λ + z ψ_α − ln γ, from the gauge fit), and its activity coefficient at infinite
dilution in that liquid. Entries whose estimate exceeds 10 trace_tol get it back, and the stage is repeated (at most
three times). A species removed while it was transiently small, for example DLT in a liquid that later takes up
organic, would otherwise stay absent. In DLT + NaCl + H2SO4 at RH 0.5 that left F too high by 1.4e-7. Before a new
liquid is seeded, every removed entry gets a trace amount back (`_reactivate`), so that all liquids again contain all
species. Removal is then repeated by the next inner solve. `checks["n_absent_entries"]` and
`checks["max_removed_trace"]` report what was removed.

### 5.2 Solids by an active set; shorter barrier schedule (`inner_method = "newton"`, default)

`_newton_solve` is the default inner solver; `inner_method = "barrier"` restores `_barrier_solve` described above.

* **Solids.** Solids enter F linearly, so they are handled by an active set, as in `SLESolver`. They are not barrier
  variables.
  * Only active solids are variables.
  * A step that would make an active solid negative is cut where the solid reaches zero, and the solid is dropped.
  * When a barrier stage has converged, the most supersaturated inactive solid (SI > `si_tol` = 1e-9, from the
    liquids that contain its ions) is added at zero amount and the stage is repeated. A solid that is driven negative
    immediately after being added is not added again in that solve.
  * At convergence every active solid has SI = 0 without a barrier residual, and every inactive one has SI ≤ si_tol.
  * The solve starts without solids. The active solids are kept when a liquid is added.
* **Liquids** keep a log-barrier, with a shorter schedule: μ = 1e-4, 1e-6, …, 1e-14 (factor `newton_mu_factor` = 0.01,
  six stages instead of twelve). On the benchmark cases this schedule needed the fewest evaluations. Starting at 1e-6
  or below, or a pure Newton iteration without a barrier (plan item 3), failed on the three-liquid case. Without the
  barrier, the iteration stalls or takes huge steps in non-convex regions, for example just after a third liquid is
  seeded or while a liquid disappears. The barrier curvature μ/x² regularizes these steps. A Levenberg–Marquardt
  regularization in the metric diag(1/x) did not fix this.
* **Traces and disappearing material.** These are checked after every step, not only at the end of a stage.
  * Trace entries are removed (Sect. 5.1).
  * An ion that falls below 1e-20 (scaled) in every liquid is removed from all liquids (`_vanish`) when an active gas
    or solid can take it up; its mass-balance row then fixes the gas or solid amount. This handles complete evaporation
    inside the iteration.
  * A liquid whose total falls below 1e-10 ends the inner solve, and the outer loop removes it or reports `"dry"`.
  * The constraint restoration after such removals is sign-safe: a correction that would flip a trace entry negative
    removes that entry instead.
* **Carbonate systems with organics** keep `_barrier_solve`, because their potentials are only approximately a
  gradient (Sect. 3.3). Inorganic carbonate systems use the new solver.

Validation against `_barrier_solve`:
* paper_1 phase-state cases (pinic acid + AS ± AN, 290–300 K, RH 0.05–0.80, equilibrium and drying path; 52 states):
  the same number of liquids, the same solids and the same F (within 1e-7) in every state.
* paper_2 acid sweeps (240 states): the same number of liquids wherever the reference converged. Non-converged states
  fell from 3 to 1 (r = 3.0, RH 0.1, with HCl).
* F can differ by the removed trace amounts (≤ 1e-9 of the feed), for example 6.7e-10 for pinic acid removed from the
  salt-rich liquid in pinic acid + AS + AN at RH 0.6.

### 5.3 Logarithmic amounts without a barrier (`inner_method = "rand"`, optional)

The trace handling of Sects. 5.1 and 7 (removal, re-entry, complete evaporation, `trace_tol`) is a consequence of the
primal formulation: amounts are the variables, steps are additive, and the 1/n curvature of a trace species makes the
Newton system ill-conditioned. Element-potential (RAND-type) methods avoid this (Smith and Missen, 1982). `_rand_solve`
is such an inner solver for the explicit liquid model (no sign-free entries); `inner_method = "rand"` selects it, and
the internal speciation model falls back to `"newton"`.

* **Scaling.** The Newton system is solved in the variables v = D⁻¹ n with D = diag(√n) for liquid (and closed-gas)
  amounts and D = I for solids and open-gas transfers. The ideal part of the Hessian, diag(1/n) plus rank-one terms,
  becomes the identity plus bounded terms, so a species of 1e-50 is as well conditioned as a major one. (With
  D = diag(n) the scaled ideal part is diag(n), and trace directions look singular to the eigenvalue shift.) The
  reduced system on the null space of A D is solved as in Sect. 5, with the same eigenvalue shift in non-convex
  regions; where no shift is needed, the non-symmetric Jacobian is used as it is (CO2(aq) makes it slightly
  non-symmetric, Sect. 3.4). The excess part is reused as in Sect. 3.1 and refreshed when the decrement stalls.
* **Update.** An increase of a liquid amount is applied additively, a decrease multiplicatively, n exp(t Δn / n). The
  map is smooth at Δn = 0, and an amount can approach zero but never cross it, so no fraction-to-the-boundary rule and
  no barrier are needed. A decrease is limited to a factor of exp(−7) per step: the linear step of a liquid whose
  composition changes a lot can predict Δn = −450 n for a minor species, and the exponential would turn this into
  1e-195, far below its equilibrium amount, from which the additive increases recover only slowly. This happened at
  five of the 240 paper_2 sweep states before the limit was introduced. Solids are additive and handled by the active
  set of Sect. 5.2.
* **Feasibility.** The multiplicative update leaves the mass and charge balances violated at second order. Before the
  objective is evaluated, they are restored by x ← x exp(Aᵀ y) on the logarithmic entries (and an additive Aᵀ y on
  open-gas transfers). This is a Newton iteration on the element potentials y with the matrix A diag(x) Aᵀ, and it
  needs no activity evaluations.
* **Acceptance and convergence.** The step is accepted on the Armijo condition. The alternative criterion of Sect. 5,
  a decrease of the norm of the reduced gradient, is used only close to convergence (decrement below 1e-10, where F
  reaches its round-off) and in carbonate systems (CO2(aq), Sect. 3.4). Elsewhere it accepted uphill steps that
  emptied a freshly seeded liquid: in water + pinonaldehyde at a_w 0.0005 below the LLPS onset (paper_2), the
  organic-rich liquid was emptied again at every retry, and the result was not converged.
* **Range of AIOMFAC.** Without a barrier, the water of a liquid can be driven towards zero. AIOMFAC then returns
  arbitrarily negative values far outside its range: an anhydrous NH4NO3 "liquid" with F = −6e7 was accepted as a
  decrease in pinic acid + AS + AN at RH 0.35. Trial points at which a liquid has a total ion molality above 1e4 mol/kg
  are rejected; metastable salt liquids reach about 200 mol/kg.
* **Convergence.** The iteration ends when the decrement is below 1e-15 and the largest relative
  change of a liquid amount is below 1e-8. Because relative steps of trace entries carry the round-off of the major
  entries (an absolute 1e-16), a decrement below 1e-24 also ends it.
* **Draining liquids.** In a non-convex region the Newton step can be dominated by a nearly singular direction, and a
  small liquid then drains only by short steps (seen with the fixed seed 0.5 at the three-liquid DLT state). After
  three consecutive steps shorter than 0.01, a liquid holding less than 1e-3 of the liquid total is merged into the
  liquid of the most similar composition. The outer loop removes it, and the stability test seeds it again if it is
  needed (the same treatment as the small liquids of a warm start, Sect. 9).

Every species stays in every liquid: in the DLT + NaCl + H2SO4 state of
`test_trace_entries_are_removed_from_single_liquids`, DLT is resolved at a mole fraction of 1e-54 in the salt-rich
liquid. Results agree with `"newton"`:
* the paper_1 solid states (104 equilibrium and drying-path comparisons): same phases and solids, F within 2.3e-10;
* warm and cold RH scans and drying paths: agreement within 2e-15, with the same phases;
* paper_1 notebook 06: all results within 1e-6;
* the paper_2 recalculation: the acid sweeps (240 states, all converged; HCl loss within 1e-7), the binary and
  surrogate LLPS checks and the Tong and Ye (2023) comparison give the same output as `"newton"` and the archived
  run. In the spinodal notebook, the binodal and spinodal RH are the same; only the Gibbs–Duhem residuals and the
  smallest eigenvalues of the finite-difference Hessians (internal speciation) differ, by as much as between two
  `"newton"` runs;
* a stress test of 344 cold-started states, compared one by one: pinic acid + AS(+AN) at four compositions, RH
  0.8–0.05, without and with solids; DLT + AS/NaCl with r = 0, 0.5 and 2, RH 0.99–0.05, without and with solids. It
  gave no difference in phases, solids, status or F (relative 1e-8), and every state converged with both solvers.

F can be lower by the trace entries that the primal solver removes (2e-11 for the three-liquid DLT state). `"newton"`
remains the default until the RAND solver has been used more widely.

### 5.4 Starting point

* **Liquid.** One liquid holds all non-water material; bounded entries are floored at 1e-7 (scaled units) and then
  re-neutralized. In a carbonate system the proton excess absorbs the charge; otherwise the anions are scaled.
* **Water.** `_water_guess` bisects on ln n_w until `ln a_w = ln RH` for that composition, falling back to
  `RH/(1−RH)` per mole of solute.
* **Solids.** With the default active-set solver no solid is active at the start. With `inner_method = "barrier"`,
  each solid starts at 1e-6.
* **Gases.** `_initial_gas` solves two linear programs. They give gas amounts that keep every bounded liquid species
  strictly positive without a floor, which matters for species absent from the feed, such as NO3- supplied only by
  HNO3 uptake:
  1. maximize the margin t with `b_i − (V u0)_i − (G g)_i ≥ t`;
  2. among the g with margin `min(t*, 1e-6 Σ|b|)`, take the one closest in L1 to the reference state. The reference
     state is zero net transfer in the open mode, and nearly all of each volatile in the gas phase in the closed mode.

---

## 6. Outer loop: number of liquid phases

After each inner solve:

### 6.1 Phase bookkeeping

* A liquid whose bounded amount falls below 1e-8 (scaled) is removed. The last liquid is never removed; if it
  becomes smaller than 1e-6, the state is reported as `"dry"` (Sect. 8).
* Two liquids whose normalized compositions agree within 1e-4 are merged, and the inner problem is solved again.

### 6.2 Stability test

The reference potentials μ_i are taken from the converged liquids (`_reference_potentials`):

* water: ln RH;
* each neutral: ln a in the liquid that holds most of it. A species that the barrier has squeezed into a trace has a
  depressed ln a and must not be the reference;
* all ions: the liquid with the largest ion content. All ion potentials then refer to one consistent electrical gauge.

The tangent-plane distance of a trial composition w (electroneutral, bounded entries summing to 1) is

```
TPD(w) = Σ_i w_i ( ln a_i(w) − μ_i ) .
```

`_tpd_minimize` finds a stationary point of TPD from each start by **successive substitution** (`_tpd_ss`,
`tpd_method = "ss"`; Michelsen, 1982, generalized to the ion basis). With `ln a = ln_ideal + r` (Sect. 3.1), the
unnormalized amounts W are updated from r at the current composition:

* neutrals: `W_i = exp(μ_i − r_i)`;
* ions: `W_i = M exp(μ_i − r_i + z_i ψ) / T`, with M = Σ_neutral W_j M_j (the solvent mass), T = Σ W in closed form
  (a quadratic), and ψ from electroneutrality (a monotonic one-dimensional equation).

At a fixed point `ln a_i − μ_i = −ln T + z_i ψ` for every species. w = W/T is then a stationary point of TPD under
Σ w = 1 and electroneutrality, with **TPD = −ln T**; the liquid set is unstable if T > 1. Each iteration costs one
activity evaluation and needs no Hessian.

In electrolyte mixtures the plain substitution can oscillate with period two. An example is DLT + NaCl + H2SO4, where
Na+ and H+ exchange roles at every iteration. r is therefore relaxed (`r ← r + β (r(w) − r)`). β is halved whenever the
change of ln W does not decrease (down to 0.05) and is raised again after contracting steps. Relaxing r, not W, keeps
every iterate electroneutral.

The iteration stops in three ways:
* when ln W changes by less than 1e-10;
* at the trivial solution: an iterate within 1e-3 of an existing liquid with TPD > −tol_tpd;
* for a clearly negative TPD (< −1e-2), once ln W changes by less than 1e-4 or after 60 iterations, since only a seed
  is needed.

**Descent guard.** In strongly non-ideal electrolyte mixtures a TPD minimum can be a repelling fixed point of the
substitution: an eigenvalue of its Jacobian is above one, which relaxation cannot cure. The iterates then drift uphill to
the trivial solution. An example is DLT + NaCl + H2SO4 (r = 0.1, RH 0.4): from every start the substitution returned the
one-liquid composition, whereas the Newton method finds a salt-rich composition with TPD −0.41. A sweep then reported
one liquid where there are two. The TPD of every iterate is therefore monitored. Once it exceeds the lowest value
reached (including the start) by more than max(1e-8, 1e-3 |lowest|), the substitution stops, and the Newton method
continues from the lowest iterate.

**Cost reductions.**
* *Early stop.* Once a start gives TPD < −`tpd_early_stop` (1e-3), the remaining starts are skipped. A clearly
  unstable liquid set needs one new liquid, not the most negative candidate, and the next outer iteration tests
  again (Michelsen's practice). The final test of a stable state runs every start, and so does a test at
  `max_liquids`, so the reported `tpd_min` is the smallest value found, as used in binodal searches.
  `tpd_early_stop = None` disables the rule.
* *Exits of the Newton method.* It stops when it comes within 1e-3 of a current liquid with TPD > −tol_tpd (the
  trivial solution, as in the substitution) or within 1e-3 of a minimum already found in the same test.
* *Effect.* The early stop and the Newton exits removed 60–96 % of the stability-test evaluations of the benchmark
  cases. Profiling had shown that most of them were Newton runs reaching a minimum already found from another start
  (11 of 12 starts in the three-liquid case) or the trivial solution (100–200 evaluations each).
  * The number of liquids is unchanged in the paper_2 acid sweeps (240 states) and in the 52 paper_1 phase-state
    cases.
  * paper_1 notebook 06 is unchanged (viscosities within 1e-8).

* *Excess Hessian of the Newton method* reused while the trial composition changes by less than
  `tpd_hess_reuse_tol` = 0.02 (was 0.002): 6 % fewer evaluations over the benchmark cases, same results.
* *Tried and rejected: trust region (plan item 7).* A Moré–Sorensen trust region in the metric diag(1/w) replaced the
  eigenvalue shift and line search of the Newton method. It saved 6 % of the evaluations of the acid scans. In the
  internal-speciation formulation, however, it missed the salt-rich liquid of DLT + NaCl + H2SO4 (r = 0.75, RH 0.2):
  from the start where the shifted Newton step jumps into that basin (TPD −1.17), the restricted steps stay with the
  nearest minimum, the trivial solution. In the stability test the Newton method also has to find distant minima, and
  the large shifted steps help with that. Non-convex steps were frequent (20–42 % of the Newton steps of the stability
  test), so this is a real effect. The inner iteration keeps its shift as well: there the barrier curvature
  regularizes the step, and near convergence a ratio test on F would be dominated by round-off.
* *Tried and rejected.* A backtracking line search along the substitution step, used before handing over to Newton,
  removed another 7 %. It also let the substitution reach the trivial solution where the Newton method finds the
  Na2SO4-rich third liquid of DLT + NaCl + H2SO4 (4 of the 240 acid-sweep states then had two liquids instead of
  three). Stopping the substitution at the first uphill step is part of what makes the test reliable.

After 100 iterations without convergence, the barrier Newton method (`_tpd_newton`: 9 μ stages from 1e-4,
eigenvalue shift for non-convex regions) finishes from the last iterate. Carbonate systems, whose proton excess is
sign-free, use the Newton method only. `tpd_method = "newton"` restores the original method; the tests
`test_stability_test_methods_give_the_same_equilibrium` and `test_successive_substitution_finds_the_unstable_direction`
compare the two.

The starts (`_trial_points`) are fixed
mole-fraction patterns. They do not depend on RH, so close to saturation an "organic-rich" trial stays organic-rich:

* each organic at x = 0.3, 0.7 and 0.95 in water, with traces of the ions;
* all organics in feed proportion at x = 0.5 and 0.9 (if there are two or more organics);
* the feed ions at water-to-ion ratios 2 and 10 (concentrated and moderate salt solutions);
* single-salt solutions of every cation–anion pair in the feed, at the same two water-to-ion ratios. Without them, a
  liquid enriched in one salt was reached only by the random perturbations below. An example is the Na2SO4-rich third
  liquid of DLT + NaCl + H2SO4 (r = 0.75–3, RH 0.3–0.4). Earlier versions missed it at some states and reported two
  liquids with a higher F;
* nearly pure water;
* copies of the current liquids, perturbed with a fixed random seed.

The patterns are built in the component basis and made electroneutral (by scaling the anions, or through the proton
excess). With explicit speciation they are then speciated with the internal model (Sect. 3.4). A minimum counts only if it is
finite, positive, and at least 1e-3 away (max-norm) from every existing liquid.

### 6.3 Adding a phase

If the most negative TPD is below `−tol_tpd` (1e-7), a new liquid is seeded at the minimizer w. Its non-water part is
moved out of the liquid that can supply the most of it:

* the transfer amount θ is a fraction f of the largest amount that keeps the donor positive. In the first attempt f
  is found by a **one-dimensional minimization** (`seed_method = "linesearch"`, default). The Gibbs energy of the donor
  and of the new liquid is minimized along the transfer direction by a bounded Brent search on ln f ∈ [ln 1e-4,
  ln 0.999], with about 15 evaluations of two liquids. For small f the change is f·TPD·(size) < 0, so a decrease always
  exists. If none is found, f = `seed_fractions[0]` = 0.5 is used; `seed_method = "fixed"` uses the fixed fractions
  throughout (the previous rule). Only species that are **major**
  in w limit θ (above 1e-4 of its largest entry). Trace entries are capped at half of the donor's amount instead
  (commit 3a14629). Otherwise a trace in w, or in the donor, would make θ, and the new phase, vanishingly small;
* the transfer is made electroneutral through the proton excess, or through the major counter-ion the donor holds most;
* the new phase receives water in proportion to w. If the corrected transfer would leave a negative amount, a
  strictly proportional transfer is used instead.

### 6.4 Termination

The loop stops when:

* no trial has TPD < −1e-7 (stable);
* `max_liquids` or `max_outer` is reached; or
* the same (number of liquids, TPD) pair has repeated `len(seed_fractions)` times.

A repeat means that the added liquid vanished or merged back. This happens close to the RH where the new liquid
appears: a large seed starts the inner solve far from the new state, and the Newton iteration returns to the old one.
The phase is then seeded again with the smaller fractions 0.2 and 0.05 (`seed_fractions`); with the line-search seed
these retries are rarely needed. An example is DLT + NaCl +
H2SO4 (r = 1.5) at RH 0.5. The two-liquid state has TPD −1.6e-3. The 0.5 seed falls back to it, while a 0.2 seed
converges to three liquids with F lower by 5.3e-4. The default `max_outer` is 8, to leave room for these retries.
With the active-set inner solver (Sect. 5.2) the fixed 0.5 seed already reaches three liquids in this case. On the
paper_2 acid sweeps (240 states), the line-search seed made the last non-converged state converge (r = 3.0, RH 0.1,
three liquids), with the same number of liquids everywhere and the same run time.

In the last two cases a remaining negative TPD makes the status `not_converged` (Sect. 8).

---

## 7. Trace species and complete evaporation (open gas mode)

In an open system a volatile can leave the particle almost completely. For example, HCl from acidified NaCl at
p = 1e-9 atm. The barrier then keeps a trace whose ln a must still satisfy the gas relation to 1e-4. Besides the
relative-step test (Sect. 5), `_complete_evaporation` handles the limit exactly. It applies when the gas is releasing
the ion and the amount of that ion left in liquids and solids is below 1e-9 of the particle's ion content. Then:

1. the remainder is assigned to the gas;
2. the feed is reduced by the total release, and that ion and gas are removed;
3. the problem is solved again, and the release is added back to `result.gas`.

The removed gas is undersaturated in the final state, meaning the particle's equilibrium pressure is below the
reservoir pressure. The check `<gas>_fully_released` records this.

---

## 8. Result and status

`PhaseEquilibriumResult` holds:

* `status`;
* `liquids`, a list of `LiquidPhase` with `amounts`, `ln_a`, `names`, `mole_fractions` and `water_activity`, sorted by
  organic fraction (organic-rich first);
* `solids` {key: mol}, and `si` {key: SI} evaluated in the first liquid;
* `gibbs` (F);
* `checks`;
* `tpd_min` and `n_outer`;
* `message` and `names`;
* `gas` {key: mol} and `p_gas` {key: atm}.

The status is decided by the **equilibrium conditions of the final state**, not by how the iteration ended:

| check | limit | condition |
|---|---|---|
| `max_abs_ln_aw_minus_ln_rh` | 1e-5 | a_w = RH in every liquid |
| `max_neutral_mu_spread` | 1e-4 | equal ln a of each neutral in all liquids where x > 1e-8 |
| `max_ion_mu_residual` | 1e-4 | least-squares residual of `ln a_αi = λ_i + z_i ψ_α` |
| `max_si` | 1e-4 | no enabled solid supersaturated |
| `max_abs_si_present_solids` | 1e-4 | SI = 0 for every solid present |
| `max_abs_gas_residual` | 1e-4 | gas equilibrium relation |
| `tpd_min` | > −1e-7 | stability |
| `max_charge_residual`, `max_mass_balance_residual` | reported | exact by construction (≈ 1e-15) |
| `n_absent_entries`, `max_removed_trace` | reported | entries removed from single liquids (Sect. 5.1) |
| `n_line_search_failures` | reported | inner line searches that failed, including those before a trace removal |

* `"converged"`: all limits hold.
* `"not_converged"`: at least one fails, and `message` names it. A one-liquid solve (`max_liquids=1`) of an unstable
  state is reported as not converged with "stability test still negative", which is intended.
* `"dry"`: no liquid remains, because all non-water material is crystalline (inorganic systems below their
  deliquescence RH); `liquids` is then empty.
* `max_si_all_candidates` also reports supersaturation of solids that were disabled (`solids="none"`, metastable).

If the inner iteration stopped on a failed line search, the `message` says so even when the checks pass.

---

## 9. Modes and API

```python
from aiomfac_py.s2as import smiles_to_components
from aiomfac_py.phase_equilibrium import PhaseEquilibrium

dlt = smiles_to_components(["CCOC(=O)C(O)C(O)C(=O)OCC"], names=["DLT"]).components[1]
pe = PhaseEquilibrium([dlt], ["Na+", "H+", "Cl-", "SO4--"], T_K=298.15)

m = 1 / 58.44                                    # 1 g NaCl
feed = {"DLT": 3 / 206.19, "Na+": m, "Cl-": m, "H+": 2 * 0.3 * m, "SO4--": 0.3 * m}

res = pe.solve(feed, 0.6, solids="none", p_gas={"HCl": 1e-9})   # metastable liquids, open to HCl
print(res.status, res.n_liquids, res.gas)
print(res.summary())
```

* `solve(feed, rh, *, solids="all" | "none" | [keys], p_gas=None, gas_total=None, n_air=None, P_atm=1.0,
  max_liquids=3, max_outer=8, verbose=False)`.
  * `feed` maps organic names and ion keys to mol (water excluded) and must be electroneutral.
  * Give either `p_gas` or `gas_total` (with `n_air`), not both.
* `solve(..., init=previous_result)` starts from a previous result of the same system and feed (warm start, plan
  item 9).
  * The start is its liquids (species amounts, with removed traces back at 1e-20 of the phase size), its active solids
    and its gas amounts. The water of every liquid is first adjusted to a_w ≈ RH by bisection.
  * The start must carry the feed: its component totals must equal the feed within 1e-8.
  * The stability test runs as usual, so the result does not depend on the start. An incompatible `init` (other
    components, other feed or T) is ignored.
  * Not used with the barrier inner solver, which needs every solid strictly positive.
  * Liquids smaller than 1 % of the largest are merged into it at the start: a small liquid can disappear at the new
    RH, and its water cannot always be adjusted.
  * A warm-started solve that does not converge is repeated from scratch, and the cold result is used if it converges.
    Without these two rules, the drying path of pinic acid + AS + AN (Seoul composition, 290 K) failed at RH 0.15–0.05:
    the small salt liquid of RH 0.20 lost its water.
* `rh_scan(feed, rh_grid, *, warm=True, **kw)` solves along `rh_grid` in the given order, each solve warm-started from
  the previous result.
* `drying_path(feed, rh_grid, *, ln_s_crit=0.0, warm=True, **gas_kw)` follows decreasing RH, with warm starts. A solid becomes a candidate once its
  SI in the metastable liquid exceeds `ln_s_crit` (a float, or a dict by solid key), and it stays a candidate at all
  lower RH. `ln_s_crit = 0` reproduces the equilibrium path; a large value gives the fully metastable path.
* Warm starts give the same states (F within 1e-14) at about half the cost:

  | scan | from scratch | warm start |
  |---|---|---|
  | DLT + NaCl + H2SO4 (r = 0.75, open HCl), 16 RH from 0.98 to 0.10 | 11485 evaluations, 3.0 s | 8410, 2.1 s |
  | DLT + NaCl + H2SO4 (r = 3.0), 16 RH (1 → 2 → 3 liquids) | 17850, 3.9 s | 15647, 3.3 s |
  | pinic acid + AS + AN, drying path, 13 RH from 0.80 to 0.05 | 7237, 1.6 s | 5586, 1.2 s |

  These numbers are with the early stop of the stability test (Sect. 6.2). Before it, the warm start halved the cost
  (for example 22751 → 9317 evaluations in the first scan). The final stability test, which runs every start at every
  RH, takes most of the remaining cost.
* `si_of(ln_a, ln_rh)` gives the saturation indices for one liquid's activities.
* Constructor options: `k_mode` (K(T) mode of `aiomfac_py.solids`), `solid_keys` (restrict the candidate solids) and
  `speciation` (`"explicit"`, default, or `"internal"`; Sect. 3.4).

---

## 10. Validation

There is no Fortran reference for phase equilibria. The tests in `tests/test_phase_equilibrium.py` check the solver
in its limits against independent implementations, and every result against its own equilibrium conditions
(`_assert_equilibrium`: all limits of Sect. 8, plus charge and mass balance < 1e-10).

| Test | What it checks |
|---|---|
| `test_liquid_model_matches_activity_model_and_gibbs_duhem` | ion-basis activities equal `ActivityModel.evaluate` (1e-12); Gibbs–Duhem of the FD Hessian |
| `test_inorganic_limit_matches_sle_solver` (3 cases) | without organics: water mass and molalities equal `SLESolver` (rel 1e-5), and the same solids (rel 1e-4); includes solid AS + aqueous NH4NO3 |
| `test_llps_matches_reference_split_and_lowers_lle_gibbs_function` | pinic acid + AS at a_w 0.4169: the split equals the `aiomfac_py.lle` split (2e-4) and lowers `AiomfacGFE` |
| `test_llps_persists_at_low_rh_where_single_phase_is_unstable` | at RH 0.30 the one-liquid state has TPD < −0.1, and the two-liquid state has a lower F |
| `test_solid_ammonium_sulfate_with_organic_liquid` | AS(s) coexists with an organic liquid; SI = 0 |
| `test_equilibrium_sequence_with_nitrate` | RH 0.8: LLPS without solid; 0.6: LLPS + AS(s); 0.3: AS(s) + NH4NO3(s) |
| `test_drying_path_crystallizes_only_after_critical_supersaturation` | supersaturated but metastable liquids above the critical SI |
| `test_rejects_unsupported_ions_and_charged_feed` | input validation |
| `test_binary_binodal_close_to_saturation_matches_common_tangent` | water + pinonaldehyde, 289 K: the stable branch switches at the common-tangent a_w* = 0.99825 |
| `test_acid_activities_match_activity_model_with_bisulfate_speciation` | acid activities equal `ActivityModel` with bisulfate dissociation (1e-10) |
| `test_acid_inorganic_limit_matches_sle_solver` (2 cases) | letovicite + aqueous solution, and the dry state, equal `SLESolver` |
| `test_open_gas_exchange_matches_sle_solver` | NH3/HNO3 open exchange equals `SLESolver` |
| `test_closed_gas_phase_matches_sle_solver` (2 cases) | closed gas, including chloride depletion by HNO3 |
| `test_carbonate_with_co2_matches_sle_solver` | carbonate + CO2, open and closed, with a negative proton excess |
| `test_organic_with_hcl_evaporation_is_in_equilibrium` | pinic acid + NaCl + H2SO4: > 99 % of chloride leaves as HCl; gas residual < 1e-4 |
| `test_near_complete_hcl_evaporation_converges` | DLT + NaCl + H2SO4 at RH 0.2: near-complete evaporation passes all checks |
| `test_trace_entries_are_removed_from_single_liquids` | DLT + NaCl + H2SO4 (r = 1.5), RH 0.1: trace DLT/Cl- removed from the salt liquid; all checks pass |
| `test_new_liquid_close_to_its_appearance_is_found_with_a_smaller_seed` | same system, RH 0.5: three liquids found after a smaller seed |
| `test_split_hessian_matches_central_differences` (3 cases) | split Hessian = central differences (1e-4), including trace organic and trace chloride in an acid liquid |
| `test_fast_bisulfate_speciation_matches_bracketing_solver` | warm-started speciation = `dissociation.solve_bisulfate` (1e-10) on random compositions |
| `test_hessian_schemes_give_the_same_equilibrium` | split and central Hessians converge to the same two-liquid state |
| `test_ad_activities_and_jacobian_match_numpy` (2 cases) | JAX ln a = NumPy ln a (1e-12); AD Jacobian = 4th-order central differences (1e-9 relative); NH4+ + H+ + SO4-- (Qcca, Rcc terms) and the carbonate system (CO2(aq) term) |
| `test_rand_inner_solver_matches_newton` (4 cases) | pinic acid + AS, + AN with AS(s), three-liquid DLT + NaCl + H2SO4 + HCl, pinic acid + NaCl + base + CO2: `"rand"` and `"newton"` give the same phases, solids and F (1e-10); no entry is removed |
| `test_rand_resolves_traces_without_removal` | the RH 0.1 state of the trace-removal test: all checks pass with DLT at a mole fraction below 1e-30 in the salt-rich liquid |
| `test_rand_binary_below_llps_onset_finds_the_organic_rich_liquid` | water + pinonaldehyde 0.0005 below the LLPS onset: the organic-rich liquid (x_org 0.68), as with `"newton"` (F 1e-10) |
| `test_rand_small_draining_liquid_is_merged_and_reseeded` | three-liquid DLT state with the fixed seed 0.5: the draining liquid is merged and the smaller seed finds three liquids |
| `test_ad_hessian_gives_the_same_equilibrium` (2 cases) | three-liquid DLT + NaCl + H2SO4 + HCl and pinic acid + NaCl + base + CO2: `"ad"` and `"split"` give the same phases and F (1e-9) |
| `test_organic_carbonate_two_liquids_with_co2` | pinic acid + NaCl + base open to CO2: two liquids with carbonate traces in the organic liquid |
| `test_stability_test_methods_give_the_same_equilibrium` (2 cases) | successive-substitution and Newton stability tests give the same number of liquids and F (1e-7) |
| `test_successive_substitution_finds_the_unstable_direction` | one-liquid pinic acid + AS at RH 0.30: the most negative TPD of both methods agrees (1e-6) |
| `test_active_set_solids_match_barrier_solids` (3 cases) | pinic acid + AS + AN at RH 0.8, 0.6, 0.3: active-set and barrier treatments of solids give the same phases, solids (rel 1e-6) and F (1e-8) |
| `test_explicit_and_internal_speciation_give_the_same_equilibrium` (2 cases) | DLT + NaCl + H2SO4 + HCl and pinic acid + NaCl + base + CO2: same phases, component amounts (1e-5), F (5e-8) |
| `test_explicit_species_are_at_reaction_equilibrium` | carbonate and bisulfate equilibria hold among the species potentials (1e-6); species add up to the components |
| `test_warm_started_rh_scan_matches_cold_solves` | DLT + NaCl + H2SO4 (r = 3.0), RH 0.98 → 0.3 through 1, 2 and 3 liquids: warm and cold scans agree (1e-10) |
| `test_incompatible_warm_start_is_ignored` | a result of another feed is not used as the start |
| `test_warm_drying_path_past_a_disappearing_salt_liquid` | pinic acid + AS + AN drying path (paper_1 Seoul): warm and cold paths agree through the disappearance of a salt liquid |

---

## 11. Stability limits of one liquid: binodal and spinodal

The binodal and the spinodal of a homogeneous liquid can be located along RH with the public API. On drying, a
droplet can stay homogeneous below the binodal (metastable) and separate at the latest at the spinodal, so measured
separation RH may lie between the two.

* Solve the one-liquid state: `res = pe.solve(feed, rh, solids="none", max_liquids=1)`. With a negative TPD the result is
  `not_converged` with the message "stability test still negative"; any other failed check is a real failure.
* **Binodal:** where `res.tpd_min` crosses `−pe.tol_tpd`.
* **Spinodal:** where the smallest eigenvalue of the projected Hessian changes sign, computed as follows.
  * Take `n = res.liquids[0].amounts` and `H = lm.hessian(n, T)`. `lm` is the `LiquidModel` of the (possibly reduced)
    problem whose `names` equal `res.liquids[0].names` (`pe.lm` or one of `pe._children`).
  * Form `D H D` with a positive diagonal scaling D (see below).
  * Restrict it to the subspace orthogonal to the scaled charge vector z∘D and to D⁻¹n (the homogeneity direction n is a
    null vector of H by Gibbs–Duhem).
  * The scaling changes the eigenvalues but not their signs (Sylvester's law of inertia). With `D = diag(n)` a trace
    species contributes an eigenvalue of order n_i, which can be the smallest one and look like zero. `D = diag(√n)`
    (the ideal-solution metric, in which `D H D ≈ I` for an ideal mixture) avoids this and gives O(1) eigenvalues;
    the charge row is then z∘√n and the homogeneity direction n/√n = √n.
  * `‖H n‖ / max|H_jj n_j|` measures the finite-difference error and should be well below the eigenvalue being
    resolved.

---

## 12. Limitations and open points

* **Run time.** `benchmarks/pe_bench.py` reports the wall time and the number of activity evaluations (in total and in
  the stability test) of six representative cases. Measured one after another on one machine (one CPU core each):

  | case | 1c693be | split Hessian (3b1dc3d) | + successive substitution (4be0487) | + active-set solids, 6 barrier stages | + line-search seed | + explicit speciation, single-salt trials | + early stop in the stability test | 738f416, split | `hess_scheme = "ad"` | `inner_method = "rand"` | `"rand"` + `"ad"` |
  |---|---|---|---|---|---|---|---|---|---|---|---|
  | pinic acid + AS, RH 0.30, two liquids | 9765 evals, 1.4 s | 1297, 0.3 s | 825, 0.2 s | 663, 0.2 s | 730, 0.2 s | 746, 0.2 s | 501, 0.10 s | 372 + 73 Jacobians, 0.08 s | 449, 0.10 s | 393 + 17, 0.09 s |
  | pinic acid + AS + AN, RH 0.6, two liquids + AS(s) | 11204, 1.7 s | 1570, 0.3 s | 713, 0.2 s | 438, 0.1 s | 461, 0.1 s | 755, 0.2 s | 526, 0.12 s | 392 + 80, 0.10 s | 433, 0.13 s | 362 + 34, 0.12 s |
  | DLT + NaCl + H2SO4 (r = 0.75), open HCl, RH 0.2 | 62624, 338 s | 4276, 1.5 s | 4484, 1.3 s | 4544, 1.3 s | 4049, 1.4 s | 2456, 0.6 s | 945, 0.22 s | 543 + 273, 0.19 s | 635, 0.17 s | 434 + 153, 0.16 s |
  | DLT + NaCl + H2SO4 (r = 1.5), open HCl, RH 0.5, three liquids | 42776, 136 s | 10811, 5.4 s | 8746, 4.6 s | 4038, 2.0 s | 3673, 1.8 s | 6170, 1.2 s | 2147, 0.44 s | 897 + 658, 0.29 s | 1537, 0.34 s | 694 + 399, 0.24 s |
  | NaCl + base, closed CO2 | 7662, 2.6 s | 2132, 0.8 s | 2132, 0.8 s | 1231, 0.5 s | 1231, 0.5 s | 299, 0.1 s | 531, 0.11 s | 235 + 112, 0.08 s | 188, 0.06 s | 150 + 11, 0.05 s |
  | pinic acid + NaCl + base, open CO2, RH 0.5, two liquids | 25021, 9.5 s | 17189, 6.5 s | 17189, 6.6 s | 17189, 6.6 s | 17159, 6.9 s | 6396, 1.2 s | 1744, 0.35 s | 534 + 396, 0.18 s | 448, 0.11 s | 250 + 82, 0.08 s |

  The last four columns are times after compilation (where it applies) (median of five repeated solves in one process; the first solve
  with `"ad"` adds 0.5–1.5 s for importing jax and compiling the Jacobians of the system and its child problems). With `"ad"` the activity evaluations fall by 26–69 % and the
  time by 15–49 %; F is the same as with `"split"` to 1e-12. On the paper_2 acid sweeps (240 states), the paper_1
  solid states (52) and notebook 06 of paper_1, `"ad"` gives the same phases, solids and results as `"split"`.

  The equilibrium states are the same, and with trace re-entry (Sect. 5.1) F agrees with 1c693be within 1e-11 in
  every case. The single-salt trials make the stability test more expensive in electrolyte-rich cases, but each
  evaluation is cheaper with explicit speciation, so the run time still falls. The stability test still takes most of the evaluations in electrolyte-rich cases: there the substitution often stops on
  the descent guard and the Newton method finishes (Sect. 6.2). Carbonate systems use the Newton method throughout.
  On the acid sweeps of `research/paper_2` (DLT + AS/NaCl + H2SO4, 240 states), the successive-substitution version
  gives the same number of liquids as 3b1dc3d at every state and is 3–5 times faster.
* **Phase appearance.** If none of the seed sizes reaches the new liquid, the loop stops on the repeat rule and reports
  a small negative TPD (status `not_converged`).
* **Extreme supersaturation.** With crystallization suppressed (`solids="none"`) at low RH, a liquid can be required
  that is far outside AIOMFAC's range. An example is DLT + NaCl + H2SO4 (r = 0.3) at RH 0.1, with a Na2SO4 liquid of
  about 200 mol/kg, x_w = 0.15 and SI(thenardite) = 6. Activity coefficients then change so steeply that the Newton
  iteration stalls, and the result is reported as not converged. The same point with `solids="all"` converges
  (halite + thenardite + one organic liquid).
* **Global optimality.** The TPD test uses a finite set of starts. A stable phase in an unsampled composition region
  can be missed, though less often than with the `lle` multistart, because the test runs on the converged state.
* **Carbonate with organics.** Gibbs–Duhem is consistent only to about 1 % (Sect. 3.3).
* **Complete evaporation.** Implemented for the open gas mode only. In the closed mode the gas phase holds the volatile
  and no trace problem arises at realistic n_air.
* **Temperature.** All constants have the T-dependence of `aiomfac_py.solids`, `gases`, `dissociation` and
  `carbonate`. AIOMFAC's MR interaction terms are T-independent.
* **Not included.** Kinetics (nucleation and efflorescence beyond the critical-supersaturation rule), surface tension
  and Kelvin effects, organic volatility (organics are non-volatile), and complexation.

## 13. References

* Amundson, N. R., Caboussat, A., He, J. W., Martynenko, A. V., Savarin, V. B., Seinfeld, J. H., and Yoo, K. Y.: A new
  inorganic atmospheric aerosol phase equilibrium model (UHAERO), Atmos. Chem. Phys., 6, 975–992, 2006.
* Amundson, N. R., Caboussat, A., He, J. W., Martynenko, A. V., Landry, C., Tong, C., and Seinfeld, J. H.: A new
  atmospheric aerosol phase equilibrium model (UHAERO): organic systems, Atmos. Chem. Phys., 7, 4675–4698, 2007.
* Amundson, N. R., Caboussat, A., He, J. W., and Seinfeld, J. H.: Primal-dual interior-point method for an optimization
  problem related to the modeling of atmospheric organic aerosols, J. Optim. Theory Appl., 130, 375–407,
  doi:10.1007/s10957-006-9110-z, 2006.
* Amundson, N. R., Caboussat, A., He, J. W., Seinfeld, J. H., and Yoo, K. Y.: Primal-dual active-set algorithm for
  chemical equilibrium problems related to the modeling of atmospheric inorganic aerosols, J. Optim. Theory Appl.,
  128, 469–498, 2006.
* Michelsen, M. L.: The isothermal flash problem. Part I. Stability, Fluid Phase Equilib., 9, 1–19, 1982.
* Smith, W. R. and Missen, R. W.: Chemical Reaction Equilibrium Analysis: Theory and Algorithms, Wiley, New York,
  1982.
* Zuend, A., Marcolli, C., Booth, A. M., et al.: New and extended parameterization of the thermodynamic model
  AIOMFAC, Atmos. Chem. Phys., 11, 9155–9206, 2011.
* Zuend, A., Marcolli, C., Peter, T., and Seinfeld, J. H.: Computation of liquid-liquid equilibria and phase
  stabilities: implications for RH-dependent gas/particle partitioning of organic-inorganic aerosols, Atmos. Chem.
  Phys., 10, 7795–7820, 2010.
* Knopf, D. A., Luo, B. P., Krieger, U. K., and Koop, T.: Thermodynamic dissociation constant of the bisulfate ion
  from Raman and ion interaction modeling studies of aqueous sulfuric acid at low temperatures, J. Phys. Chem. A, 107,
  4322–4332, 2003.

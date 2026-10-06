# Combined liquid–liquid–solid equilibrium solver (`aiomfac_py.phase_equilibrium`)

Implementation: `src/aiomfac_py/phase_equilibrium.py` (`LiquidModel`, `PhaseEquilibrium`, `PhaseEquilibriumResult`),
tests: `tests/test_phase_equilibrium.py` (22 tests). Branch `feature/phase-equilibrium` (commits 3f1f230 onward).

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
| Acid sulfate | H+ and SO4-- as stoichiometric components | HSO4- is speciated inside every liquid (Sect. 3.2) |
| Carbonate | CO3-- (total carbonate) and H+ (sign-free proton excess) | CO2(aq), HCO3-, CO3--, OH- are speciated inside every liquid (Sect. 3.3) |
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

AIOMFAC provides no analytic Jacobian. `LiquidModel.hessian` computes `∂ ln a_i / ∂ n_j` by central differences and
symmetrizes it:

* the step for each bounded species is `h_j = 1e-5 · n_j`. A step proportional to the amount keeps the perturbed
  state positive for trace species and resolves their 1/n curvature;
* only a sign-free species (the carbonate proton excess) uses an absolute floor, `1e-10 Σ|n|`.

The cost is 2N activity evaluations per liquid and Newton iteration. It dominates the run time.

### 3.2 Acid sulfate

H+ and SO4-- are **stoichiometric** (total) components; passing HSO4- is an error. In every activity evaluation of a
phase that contains both, the bisulfate equilibrium HSO4- ⇌ H+ + SO4-- is solved with `dissociation.solve_bisulfate`
(Knopf et al., 2003 constant, as in `aiomfac_py.sle`). The free-ion molalities replace the totals before the activity
coefficients are computed. The potential of each stoichiometric component equals that of its free ion at the
speciation equilibrium (`μ_H,total = μ_H+`, `μ_SO4,total = μ_SO4--`). Because of this, the formulation of Sect. 2
needs no change: the speciation is part of the phase's Gibbs function.

Test: `test_acid_activities_match_activity_model_with_bisulfate_speciation` compares with `ActivityModel.evaluate`
including its own dissociation step (agreement 1e-10).

### 3.3 Carbonate

A carbonate system has the components `CO3--`, carried as the total carbonate C_T = CO2(aq) + HCO3- + CO3--, and
`H+`, carried as the proton excess P = H+ − OH- + HCO3- + 2 CO2(aq) (+ HSO4-). P may be negative (a basic solution), so
H+ is a free (barrier-free) variable in this mode. `LiquidModel._ln_a_carb` generalizes `sle.AqueousIons._speciate_carb`
to phases that contain organics:

* equilibria: K1 = a_H a_HCO3 / (a_CO2 a_w), K2 = a_H a_CO3 / a_HCO3, Kw = a_H a_OH / a_w, and K_HSO4 if sulfate is
  present (constants from `aiomfac_py.carbonate` and `dissociation`);
* γ(CO2) from AIOMFAC's salting-out relation (`carbonate.gamma_co2_mr`);
* solution: a fixed-point iteration on the apparent constants. For given apparent constants, the proton balance is
  monotonic in ln m_H and is solved with Brent's method; the bracket is widened as needed. The apparent constants are
  updated from the new activity coefficients, until their ln changes by less than 1e-11. The last solution warm-starts
  the next call, and failed evaluations are never reused.

**Limitation.** AIOMFAC treats CO2(aq) with its own salting-out coefficient, outside the LR/MR/SR Gibbs function. With
organics present, the stoichiometric potentials are therefore only approximately the gradient of one Gibbs function:
Gibbs–Duhem holds to about 1 %. The inner solver accepts a step that reduces either F or the norm of the reduced
gradient (Sect. 5), and the final checks verify the equilibrium conditions directly. In the inorganic limit the
results equal `SLESolver` (Sect. 10).

---

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
* **Newton system.** `H` is block-diagonal: one finite-difference Hessian per liquid, the closed-mode gas block, and the
  barrier term `μ / x_j²`. The reduced Hessian `Zᵀ H Z` is diagonalized. If its smallest eigenvalue is not positive
  (a non-convex region of AIOMFAC), the spectrum is shifted to `1e-8 · max|λ|`. The result is a descent direction with
  the Newton step in the convex directions.
* **Step length.** A fraction-to-the-boundary rule (0.995 of the largest feasible step) is followed by backtracking.
  A step is accepted if it satisfies the Armijo condition on Φ_μ, **or** if it reduces the norm of the reduced gradient
  `‖Zᵀ ∇Φ_μ‖` by the same factor. The second criterion is needed for carbonate with organics (Sect. 3.3). Evaluations
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
Before a new liquid is seeded, every removed entry gets a trace amount back (`_reactivate`), so that all liquids again
contain all species. Removal is then repeated by the next inner solve. `checks["n_absent_entries"]` and
`checks["max_removed_trace"]` report what was removed.

### 5.2 Starting point

* **Liquid.** One liquid holds all non-water material; bounded entries are floored at 1e-7 (scaled units) and then
  re-neutralized. In a carbonate system the proton excess absorbs the charge; otherwise the anions are scaled.
* **Water.** `_water_guess` bisects on ln n_w until `ln a_w = ln RH` for that composition, falling back to
  `RH/(1−RH)` per mole of solute.
* **Solids.** Each solid starts at 1e-6.
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

`_tpd_minimize` minimizes it with the same barrier-Newton scheme in the null space of {Σ w = 1, z·w = 0}. It uses 9
μ stages from 1e-4 and an eigenvalue shift for non-convex regions. The starts (`_trial_points`) are fixed
mole-fraction patterns. They do not depend on RH, so close to saturation an "organic-rich" trial stays organic-rich:

* each organic at x = 0.3, 0.7 and 0.95 in water, with traces of the ions;
* all organics in feed proportion at x = 0.5 and 0.9 (if there are two or more organics);
* the feed ions at water-to-ion ratios 2 and 10 (concentrated and moderate salt solutions);
* nearly pure water;
* copies of the current liquids, perturbed with a fixed random seed.

Each start is made electroneutral (by scaling the anions, or through the proton excess). A minimum counts only if it is
finite, positive, and at least 1e-3 away (max-norm) from every existing liquid.

### 6.3 Adding a phase

If the most negative TPD is below `−tol_tpd` (1e-7), a new liquid is seeded at the minimizer w. Its non-water part is
moved out of the liquid that can supply the most of it:

* the transfer amount θ is a fraction (`seed_fractions[0]` = 0.5) of the largest amount that keeps the donor
  positive. Only species that are **major**
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
The phase is then seeded again with the smaller fractions 0.2 and 0.05 (`seed_fractions`). An example is DLT + NaCl +
H2SO4 (r = 1.5) at RH 0.5. The two-liquid state has TPD −1.6e-3. The 0.5 seed falls back to it, while a 0.2 seed
converges to three liquids with F lower by 5.3e-4. The default `max_outer` is 8, to leave room for these retries.

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
* `drying_path(feed, rh_grid, *, ln_s_crit=0.0, **gas_kw)` follows decreasing RH. A solid becomes a candidate once its
  SI in the metastable liquid exceeds `ln_s_crit` (a float, or a dict by solid key), and it stays a candidate at all
  lower RH. `ln_s_crit = 0` reproduces the equilibrium path; a large value gives the fully metastable path.
* `si_of(ln_a, ln_rh)` gives the saturation indices for one liquid's activities.
* Constructor options: `k_mode` (K(T) mode of `aiomfac_py.solids`) and `solid_keys` (restrict the candidate solids).

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

---

## 12. Limitations and open points

* **Run time.** About 10–25 s for a one-liquid solve and 1–1.5 min for a full solve (stability test, two liquids)
  with one organic and 4–5 ions on a laptop, dominated by finite-difference Hessians
  (2N activity evaluations per liquid and Newton step) and the TPD starts. An analytic or automatic-differentiation
  Jacobian of AIOMFAC would remove most of it.
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
* Zuend, A., Marcolli, C., Booth, A. M., et al.: New and extended parameterization of the thermodynamic model
  AIOMFAC, Atmos. Chem. Phys., 11, 9155–9206, 2011.
* Zuend, A., Marcolli, C., Peter, T., and Seinfeld, J. H.: Computation of liquid-liquid equilibria and phase
  stabilities: implications for RH-dependent gas/particle partitioning of organic-inorganic aerosols, Atmos. Chem.
  Phys., 10, 7795–7820, 2010.
* Knopf, D. A., Luo, B. P., Krieger, U. K., and Koop, T.: Thermodynamic dissociation constant of the bisulfate ion
  from Raman and ion interaction modeling studies of aqueous sulfuric acid at low temperatures, J. Phys. Chem. A, 107,
  4322–4332, 2003.

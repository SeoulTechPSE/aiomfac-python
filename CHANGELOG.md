# Changelog

## Unreleased

- **Solver robustness** (`PhaseEquilibrium`): (i) the stability test discards water-free ionic trial phases (AIOMFAC
  diverges there; `tpd_min_water`), (ii) `max_liquids` defaults to 4 (two organic and two aqueous liquids occur with
  two organics), (iii) a solve that does not converge is repeated with the `rand` and `barrier` inner methods
  (`fallback_inner`), which need no active-set exchange next to a boundary between two solids. In the UHAERO 2007
  diagrams 59 of 60 sampled failures converge; converged states are unchanged.

- **Phase diagrams** (`aiomfac_py.diagram`, extra `plot`): phase-boundary tracing in RH by bisection on
  `PhaseEquilibrium` solves (`trace`; equilibrium and metastable modes; non-converged solves are dropped and listed),
  composition–RH maps with refinement in composition where the boundary topology changes (`phase_map`), plotting with
  boundary lines and region colours (`plot_phase_map`), phase composition for pie charts (`pie_composition`). The
  traced deliquescence RH of ten single salts equals `sle.binary_saturation` to 4e-5. Also `rh_profile`,
  `particle_properties` (relative particle mass, pH), `deliquescence_point`, `label_regions`, and for neutral
  mixtures `binary_mixing_curve`, `ternary_lle` (phase diagram from the lower convex hull of the Gibbs energy of
  mixing) and `plot_ternary`. Manual `docs/phase_diagrams.md`, notebook `notebooks/07_phase_diagrams.ipynb`, and the
  figures of Amundson et al. (2006, 2007) with AIOMFAC (`tools/uhaero2006_figures.py`, `tools/uhaero2007_figures.py`).
- **Boundary-line continuation** (`diagram.trace_boundaries`, `BoundaryCurves`, `plot_boundary_curves`): every
  boundary of a phase map is followed in the (x, RH) plane by a predictor (secant) and a corrector (bisection across
  the line), vertical lines included, so the lines are smooth and meet at the junctions;
  `plot_phase_map(..., curves=)` draws them and fills the regions between them.
- **Double salts** (`aiomfac_py.solids`): (NH4)2SO4·2NH4NO3, (NH4)2SO4·3NH4NO3 and NH4HSO4·NH4NO3 as composite
  solids, ln K = sum of the simple salts' ln K (this database, fitted to AIOMFAC) + the solid-state formation reaction
  from Clegg, Brimblecombe and Wexler (1998, Table 2), which does not depend on the solution model.
- `PhaseEquilibrium`: an all-solid state is now the minimum of a linear program over the candidate solids (water
  open), which corrects dry results with a non-optimal assemblage and replaces a liquid-containing result whose
  transformed Gibbs energy is higher than the best all-solid one; saturation indices of dry results come from the
  program. Mutual deliquescence RHs are therefore invariant in the mixing ratio, as the phase rule requires.
- `PhaseEquilibrium`: the charge-balance bracket search of the tangent-plane successive substitution could loop
  forever when an activity coefficient had overflowed; it is now bounded and hands over to the Newton step.

## v1.3.0 (2026-10)

- **Batched activities** (`ExplicitLiquidModel.ln_a_batch`, optional `jax`): `ln_a` for many compositions at one
  temperature through the vectorized JAX transcription of `ad_activity` (compiled once per system, temperature and
  batch size; equal to `ln_a` to round-off, tested for organic, salt, acid-sulfate and carbonate systems). On one CPU
  thread (`tools/bench_batch.py`) it costs 0.2-1.4 microseconds per composition in batches of 1000-10 000, against
  0.1-0.3 ms per `ln_a` call (NumPy), 35-90 microseconds per call of the compiled function, and 3-4 microseconds per
  evaluation of the Fortran AIOMFAC on the same machine.
- **Numba prototype** (`tools/numba_prototype.py`): the short-range term for water + organics as one compiled loop
  kernel, 2.3 microseconds per binary evaluation (Fortran level), as an estimate of what a compiled core would gain.
- **Solid data**: Mg(NO3)2·6H2O had a placeholder ln K0 = 0 without an anchor, which kept the crystal stable up to
  RH ~ 1; it is now anchored to its 25 °C solubility (4.80 mol/kg), giving a deliquescence RH of 0.554 at 298 K
  (measured 0.529).
- **Bicarbonate systems vs. Fortran** (documentation and tests only): the differences from the Fortran code in
  bicarbonate systems without sulfate come from the Fortran `Gammas()`, which refreshes the sum of ion molalities
  only for bisulfate systems. With that line corrected, the Fortran results agree with this port to the printed
  digits; the earlier explanation (catastrophic cancellation) was wrong. A test against the corrected values is added.

## v1.2.0 (2026-10)

- **Gibbs-function liquids** (`aiomfac_py.gibbs_model`, optional `jax`): `GibbsLiquidModel` uses ln a = ∇g and the exact
  Hessian ∇²g of a JAX function g(n) = G/RT (excess-Gibbs-energy surrogates of AIOMFAC); gradient, forward-over-reverse
  Hessian and Hessian-vector product are jit-compiled once and reused (temperature as an argument, shared caches,
  optional persistent cache). `PhaseEquilibrium(..., liquid_model=...)` accepts such a model (also for child problems).

## v1.1.0 (2026-10)

New solvers on top of the validated activity-coefficient model (the AIOMFAC port itself is unchanged from v1.0.1):

- **Solid–liquid equilibrium** (`aiomfac_py.sle`, `aiomfac_py.solids`): primal–dual active-set solver with K_sp(T) and a
  hydrate database; acid sulfates (H+/HSO4-, NH4HSO4, letovicite and Na acid salts, with the thermodynamic constants of
  Clegg et al., 1998); carbonate and hydroxide solids with internal HCO3-/OH- speciation; NH3, HNO3, HCl and CO2 gas
  phases in open (fixed partial pressure) and closed (ideal gas) modes.
- **Combined liquid–liquid–solid(–gas) equilibrium** (`aiomfac_py.phase_equilibrium.PhaseEquilibrium`):
  transformed Gibbs-energy minimization with explicit speciation (HSO4-, HCO3-, OH-, CO2(aq) as species),
  tangent-plane stability tests for new liquids, solids by an active set, RH continuation with warm starts;
  Newton inner solver (split Hessian with reuse, or an exact Hessian by automatic differentiation with the optional
  `ad` extra, `hess_scheme="ad"`) and a RAND-type inner solver in logarithmic amounts (`inner_method="rand"`).
  Liquids that contain only water after trace removal are dropped.
- **Main tool** (`aiomfac-tool`, `aiomfac_py.tool`): one entry point for the SLE, LLE, combined and gas–particle
  calculations, selected by mode and engine flags in a TOML case file, with CSV/JSON output; user manual in
  `docs/user_manual.md` and example cases in `examples/cases/`.
- Documentation fixes (Gervasi et al., 2020, author list).

The LLE solver (`aiomfac_py.lle`) and the new equilibrium solvers are verified against analytical cases, against each
other and against published phase diagrams, not against the Fortran model (which has no equivalent solver).

## v1.0.1 (2026-10-04)

Metadata release (Zenodo DOI 10.5281/zenodo.23137735); code identical to v1.0.0.

## v1.0.0

First public release: pure-Python port of AIOMFAC (AIOMFAC-web v3.14), validated term by term against an instrumented
Fortran build; S2AS SMILES-to-subgroup conversion, LLE solver, AIOMFAC-VISC and the Armeli et al. (2023) ML Tg predictor.

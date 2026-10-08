# Changelog

## Unreleased

- **Batched activities** (`ExplicitLiquidModel.ln_a_batch`, optional `jax`): `ln_a` for many compositions at one
  temperature through the vectorized JAX transcription of `ad_activity` (compiled once per system, temperature and
  batch size; equal to `ln_a` to round-off, tested for organic, salt, acid-sulfate and carbonate systems). On one CPU
  thread (`tools/bench_batch.py`) it costs 0.2-1.4 microseconds per composition in batches of 1000-10 000, against
  0.1-0.3 ms per `ln_a` call (NumPy), 35-90 microseconds per call of the compiled function, and 3-4 microseconds per
  evaluation of the Fortran AIOMFAC on the same machine.

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

# aiomfac_py

Pure-Python port of the **AIOMFAC** thermodynamic group-contribution model (activity coefficients of
inorganic–organic mixtures). **Work in progress** — the activity-coefficient pipeline (LR + MR + SR and the
composition conversion) runs end-to-end, including the Fortran auto-completion and dissociation-equilibrium
machinery for bisulfate, bicarbonate, and joint bisulfate+bicarbonate systems (with Ca2+/CaSO4(s) precipitation);
see "Validation status" below for what is and isn't covered. `aiomfac_py.s2as` additionally integrates the
(already-Python) S2AS tool for going straight from a SMILES string to an AIOMFAC component.

* Reference implementation: <https://github.com/andizuend/AIOMFAC> (AIOMFAC-web v3.14, commit
  `b9cb96d0eb22dc65a5e63edafe1ed97bd07662f2`). This port is validated against that Fortran code.
* `aiomfac_py.s2as` integrates <https://github.com/andizuend/S2AS__SMILES_to_AIOMFAC> (commit
  `88a2bffde1d375f8cb86a6e47ead6c3f0dad1833`) -- SMILES -> AIOMFAC subgroups, already pure Python upstream.
* Scope of the first release: activity coefficients only. The viscosity module (AIOMFAC-VISC) is excluded.
* License: **GPL-3.0-or-later**, because this is a derivative work of the (GPL-3.0) Fortran code (and, for
  `aiomfac_py.s2as`, of the GPL-3.0 S2AS tool).
* If you use it, please cite the AIOMFAC publications listed at <https://aiomfac.lab.mcgill.ca/citation.html>
  (and, for `aiomfac_py.s2as`, Amaladhasan et al., 2026, <https://doi.org/10.5194/gmd-19-4601-2026>), and (as
  requested by the authors) let them know about your use.

## Layout

| Path | Purpose |
|---|---|
| `src/aiomfac_py/io.py` | input-file reader (`read_input_file`) — done, tested |
| `src/aiomfac_py/params.py`, `data/*.npz` | SR (UNIFAC) tables R, Q, ARR, BRR, CRR and subgroup tables NKTAB, GroupMW, Ioncharge — extracted from the Fortran source, bit-identical to the arrays the compiled model uses |
| `src/aiomfac_py/system.py` | SR system setup (species/group ordering, R/Q, interaction matrix) — port of `SetSystem`/`SRsystm` parts |
| `src/aiomfac_py/sr.py` | **short-range term** (`sr_terms`) — validated against Fortran |
| `src/aiomfac_py/lr.py`, `mr.py` | long-range (Debye–Hückel) and middle-range terms, MR parameter mapping (`build_mr_state`) — validated |
| `src/aiomfac_py/composition.py`, `numerics.py` | mass/mole fraction → mass fraction → ion molalities; `safe_exp` — validated |
| `src/aiomfac_py/completion.py` | auto-completion of bisulfate/bicarbonate systems (species set) — validated structurally |
| `src/aiomfac_py/dissociation.py` | HSO4- <-> H+ + SO4-- equilibrium (Brent's method, `numerics.brent_root`) — validated end-to-end, wired into `ActivityModel` |
| `src/aiomfac_py/carbonate.py` | bicarbonate-only equilibrium (`solve_carbonate`, CO2(aq)/HCO3-/CO3--/OH-/H+, approximate ~1e-3 to 1e-4, see below) and the joint bisulfate+bicarbonate equilibrium (`solve_carb_sulf`, machine precision, plus Ca2+/CaSO4(s) precipitation) — both `scipy.optimize.root`-based, wired into `ActivityModel` |
| `src/aiomfac_py/model.py` | `ActivityModel` / `activity_coefficients()` — end-to-end for simple systems |
| `src/aiomfac_py/s2as/` | SMILES -> AIOMFAC subgroups (optional `epam.indigo` dependency) — integration of the upstream S2AS tool, validated bit-for-bit against it |
| `tools/extract_params.py`, `tools/extract_mr_params.py` | regenerate `sr_params.npz`, `subgroup_params.npz`, `mr_params.npz` from the Fortran source (never edit tables by hand; `extract_mr_params` interprets `MRdata` statement by statement and reproduces Fortran literal kinds) |
| `fortran_patches/` | patch that makes the Fortran model dump every activity-coefficient term at full precision, plus notes on how to rebuild the references |
| `tests/reference/` | examples 0001/0003 (inputs, standard outputs, per-term dumps) |
| `tests/reference/cases/` | 9 generated validation cases, 145 points (`tools/make_cases.py` + `tools/run_fortran_cases.sh`) |
| `tests/reference/completion/` | NaHSO4, NaHCO3 and H2SO4 cases: structural completion (`DIMS`) and, for the two bisulfate cases, the full dissociation-equilibrium pipeline |
| `tests/reference/carbonate/` | NaHCO3 and KHCO3 cases used to validate `carbonate.py`'s bicarbonate-only solve |
| `tests/reference/carb_sulf/` | NaHSO4+NaHCO3 (`c005`) and Ca(IO3)2+NaHSO4+NaHCO3 (`c006`) cases used to validate `carbonate.py`'s joint `solve_carb_sulf` and the Ca2+/CaSO4(s) precipitation step |
| `tests/reference/s2as/` | upstream S2AS's own 3 example SMILES lists (7, 174, and 2823 molecules) and their generated AIOMFAC input files, used to validate `s2as/` bit-for-bit |
| `fortran_patches/SubModDissociationEquil_carb_debug.f90` | throwaway-instrumented copy used once to dump Fortran's internal dissociation constants/converged molar amounts directly, for the per-species precision analysis above (not needed to use or test the package) |

## Conventions

* 0-based arrays; subgroup `s` ↔ index `s-1`, main group `g` ↔ index `g-1`; `ARR[i, j]` = Fortran `ARR(i+1, j+1)`.
* Species order (as in Fortran `X`): neutrals (input order), cations, anions.
* Composition tables: columns `cp02…cpNN` hold components 2..N; component 1 is `1 − Σ others`.

## Development

```
pip install -e .[test]
pytest
```
`test_port_matches_fortran` is a strict `xfail` today; when the port works it will start passing,
which pytest reports as a failure until the marker is removed — that is the signal to promote it to a real test.

## 🚀 Colab Quick Start (1 Cell)

`aiomfac_py` has no compiled dependencies of its own — pure Python + numpy, with two small optional extras
(`scipy` for the dissociation-equilibrium solvers, `epam.indigo` for the SMILES tool) — so, unlike
[SeoulTechPSE/fenicsx-colab](https://github.com/SeoulTechPSE/fenicsx-colab), there's no conda/micromamba
environment and no Jupyter cell magic here: a single `pip install` does it.

Open a new Google Colab notebook and run **this single cell**:

```python
# aiomfac_py on Google Colab
# SeoulTechPSE/aiomfac-python

from pathlib import Path
import subprocess

REPO_URL = "https://github.com/SeoulTechPSE/aiomfac-python.git"
REPO_DIR = Path("/content/aiomfac-python")

if not REPO_DIR.exists():
    print("Cloning aiomfac-python...")
    subprocess.run(["git", "clone", REPO_URL, str(REPO_DIR)], check=True)
else:
    print("Repository already exists — pulling latest...")
    subprocess.run(["git", "-C", str(REPO_DIR), "pull", "--ff-only"], check=True)

INSTALL_SMILES = False  # <-- set True to also install epam.indigo (aiomfac_py.s2as)
USE_CLEAN = False       # <-- set True to force a clean reinstall

opts = ["--carbonate"]
if INSTALL_SMILES:
    opts.append("--smiles")
if USE_CLEAN:
    opts.append("--clean")

get_ipython().run_line_magic("run", f"{REPO_DIR / 'setup_aiomfac.py'} {' '.join(opts)}")
```

After this finishes, `import aiomfac_py` works directly (installed editable, so files under `src/` can also be
edited and re-imported in the same Colab session without reinstalling).

See [`notebooks/bootstrap_colab.ipynb`](notebooks/bootstrap_colab.ipynb) for the same steps as a ready-to-run
notebook, and [`notebooks/01_quickstart.ipynb`](notebooks/01_quickstart.ipynb) for four worked examples: reading
an AIOMFAC-web input file, building a mixture by hand (aqueous NaCl), the joint bisulfate+bicarbonate
dissociation equilibrium (with and without Ca2+ precipitation), and SMILES → AIOMFAC subgroups via
`aiomfac_py.s2as`.

### Manual / local install

```
pip install "aiomfac_py[carbonate] @ git+https://github.com/SeoulTechPSE/aiomfac-python.git"
# or, from a local checkout:
python setup_aiomfac.py --carbonate          # see --help for --smiles / --all / --clean / --no-editable
```

## Usage

```python
from aiomfac_py import read_input_file, ActivityModel
case = read_input_file("tests/reference/inputs/input_0001.txt")
model = ActivityModel(case.components)
res = model.evaluate(case.fractions[0], case.T_K[0], case.basis)
res.gamma_neutral, res.ln_gamma          # activity coefficients / ln(gamma) of all species (neutrals, cations, anions)
```

Going straight from SMILES (requires the optional `epam.indigo` dependency: `pip install aiomfac_py[smiles]`):

```python
from aiomfac_py import ActivityModel
from aiomfac_py.s2as import smiles_to_components

result = smiles_to_components(["CCCCO"])         # 1-butanol; water is prepended automatically
model = ActivityModel(result.components)
res = model.evaluate([0.9, 0.1], 298.15, "mass")  # 90/10 mass-fraction water/butanol
```

## Validation status (what is and is not verified)

Verified against the instrumented Fortran model (AIOMFAC-web v3.14, commit above), full pipeline
(mass/mole fractions -> molalities -> X -> LR -> MR -> SR -> ln gamma -> activities), scaled difference < 1e-12
unless noted:

* **Examples 0001 / 0003** (shipped with the Fortran repo): water + 4 organics + NaCl (12 points, 280-302 K) and a
  carbonate system (1 point; only water and the ions are checked, see below). Neutral gamma also match the values
  printed by the *unmodified* Fortran output routine (6 significant digits).
* **9 generated cases, 145 points** (`tools/make_cases.py`, references regenerated with `tools/run_fortran_cases.sh`):
  NaCl from dilute to near-saturation; CaCl2, MgCl2 (divalent cations); LiCl (high ionic strength); NaCl+KBr (adds the
  cross-combinations NaBr/KCl that are not in the input -- exercises the automatic electrolyte-component completion
  in `build_mixture`); water+ethanol+NaCl with organic and salt varied independently; 4 organics + CaCl2 over a
  275-325 K sweep; a random 6-component mixture with 4 cations / 2 anions (8 electrolyte components, exercises
  cross-ion combinations at once); and dilute/zero-abundance edge points. Together with examples 0001/0003 this
  covers ionic strength from about 1e-6 to > 8 mol/kg and temperatures 275-325 K. Every point also matches the
  *unmodified* Fortran output (6 digits), and all Fortran reference runs finished with error indicator 0 (the only
  non-zero per-point flag anywhere is warning 10, "temperature outside the recommended electrolyte range", expected
  for the two temperature sweeps).
* Parameter tables (SR, subgroup, MR) are bit-identical to the arrays the compiled Fortran model uses.

For **example 0003** specifically: LR and MR of water and of all ions agree to 1e-12 when fed the Fortran molalities
(CO2(aq) is excluded: the Fortran `GammaCO2()` overwrites its LR/MR/SR terms afterwards, not ported).

**Limits of that evidence -- please read before trusting results:**

* No NH4+/H+/HSO4- system: the `Qcca`/`Rcc` three-ion interaction terms are ported but still untested (case 0108
  has 4 cations x 2 anions = 8 pair combinations, but none of the ion triples that activate `Qcca`/`Rcc`).
* No very high ionic strength: the `omega*sqrt(I) > 300` and `sqrt(I) > 250` guard branches in `mr.py` are untested.
* All organic components in the generated cases are simple alcohols/acetone (chosen to avoid triggering the
  Fortran SMILES/Tg lookup, which needs a Python script this environment does not have installed) -- no esters,
  ethers, acids, aromatics, or amines.
* **Bisulfate systems** (HSO4- present, or H+ + SO4-- given directly, *without* any HCO3-/CO3--) are now handled
  automatically end-to-end: `ActivityModel` completes the component list with `completion.py` if needed, then
  solves the HSO4- <-> H+ + SO4-- equilibrium (`dissociation.py`, a 1-D root-find with a self-contained Brent's
  method, port of Fortran `HSO4_dissociation`/`DiffKsulfuricDissoc`). Validated against two Fortran cases (NaHSO4;
  H2SO4 given as one H+/SO4-- component) at several concentrations each: equilibrium ion molalities and total
  ln(gamma) of every species agree to atol 1e-12 to 1e-13. Coverage is narrow -- two salts, one temperature
  (298.15 K), no mixed sulfate/other-electrolyte systems, no very high or very low ionic strength -- and the
  root-finder itself is a plain bracket-and-bisect-then-Brent search, not the Fortran solver's guided initial
  guess, so behaviour at pathological edges (near-zero or near-saturation sulfate) is unverified.
* **Bicarbonate-only systems** (HCO3-/CO3--, no sulfate) are now handled automatically too: `ActivityModel`
  completes the system, then solves the 5-unknown equilibrium (CO2(aq) + H2O <-> H+ + HCO3-, HCO3- <-> H+ + CO3--,
  H2O <-> H+ + OH-, plus two mass balances) with `scipy.optimize.root(method="hybr")` -- SciPy's own MINPACK
  `hybrd` wrapper, the same algorithm Fortran uses, rather than a hand-written solver. `GammaCO2` (which overwrites
  CO2(aq)'s own ln(gamma) with a salting-out-coefficient sum) is ported and applied automatically.
  **This one is meaningfully less precise than everything else in this port, and the reason is now well
  understood** (an earlier version of this note wrongly blamed a skipped near-zero-concentration smoothing branch;
  that has been ruled out -- see below). Validated against two Fortran cases (NaHCO3, KHCO3; 2 points each) by
  instrumenting the Fortran source to dump its own converged molar amounts directly (not just the printed
  molalities), so the comparison is apples-to-apples:

  | species | relative difference vs. Fortran |
  |---|---|
  | HCO3- (the dominant carbon species, ~97% of total carbon here) | ~1e-6 |
  | CO3--, CO2(aq) (minor carbon species, ~1-2% each) | ~4e-5 |
  | OH- (trace, ~1e-8 mol/kg) | ~5e-5 |
  | H+ (trace, ~1e-10 mol/kg) | ~3e-4 |

  So the practically important quantities (water activity, the salt's own activity, which are governed by the
  *dominant* species) are accurate to ~1e-6 -- close to the rest of this port. The error is concentrated in the
  *trace* ions H+ and OH-, and the reason is a textbook catastrophic-cancellation problem, not a bug: the
  mass-balance equation that fixes H+ (`n_h_max - n_hco3 - 2*n_co2 - n_oh_max + n_oh = 0`) subtracts terms of
  order 1 from each other to obtain a remainder around 1e-8 -- roughly 8 of the ~16 significant digits of double
  precision are lost in that step. This affects *any* solver working with this formulation, including Fortran's
  own `hybrd`: instrumenting the Fortran source to print its internal constants (`nHmax`, `nCarbmax`, `nOHmax`,
  etc.) confirmed they are computed identically to this port's values (bit-for-bit, up to the last 1-2 digits), and
  seeding this port's solver with Fortran's own converged answer causes it to visibly move *away* from that answer
  to a different point where the equations (as coded, matching Fortran's source line-for-line) are satisfied to
  machine precision (residual ~1e-14) -- i.e., both solvers are finding genuine, self-consistent roots of an
  ill-conditioned system, and the small disagreement between them for the trace species is inherent to that
  conditioning rather than a fixable defect. An attempt to sidestep this for H+ specifically, by recomputing it
  from the (well-conditioned, purely multiplicative) Kw equilibrium instead of the mass balance, reproduced the
  same value -- unsurprising in hindsight, since Kw is already satisfied by the converged solution to the same
  ~1e-14 residual, so that recomputation is circular and adds no information.
  Requires the optional `scipy` dependency (`pip install aiomfac_py[carbonate]`); `ActivityModel` raises
  `ImportError` from within `carbonate.solve_carbonate` if it is used without scipy installed.
* **Bisulfate + bicarbonate systems together** (both HSO4-/SO4-- and HCO3-/CO3--) are now handled automatically
  too: `ActivityModel` solves the joint 6-unknown equilibrium (`carbonate.solve_carb_sulf`, port of Fortran
  `HSO4_and_HCO3_dissociation` with `bisulfsyst == True`) -- the same three bicarbonate equilibria as above plus
  HSO4- <-> H+ + SO4--, all coupled through the shared H+ pool. If Ca2+ is also present, the deterministic
  Ca2+ + SO4-- -> CaSO4(s) precipitation pre-step (`_precipitate_ca_sulfate`, Fortran's `idCa > 0` branch -- a
  full stoichiometric removal down to a tiny bookkeeping residual, *not* a Ksp-based solubility equilibrium; the
  original Fortran routine doesn't model one either) runs first.

  Validated against two Fortran cases instrumented the same way as the bicarbonate-only solver above: `c005`
  (Water + NaHSO4 + NaHCO3, no Ca2+, 10 points across dilute/concentrated/trace-species compositions and three
  temperatures) and `c006` (Water + Ca(IO3)2 + NaHSO4 + NaHCO3, 6 points, exercising the precipitation step; the
  divalent anion component uses AIOMFAC subgroup 246, IO3-, not Cl- -- an earlier draft of this case mislabeled it
  "CaCl2"; the underlying Ca2+/CaSO4(s) precipitation math is anion-independent, so this was a labeling error only,
  not a validation error).
  Agreement is **machine precision (~1e-14 to 1e-15) at every point where Fortran's own solver converges** --
  markedly *better* than the bicarbonate-only case, because here H+ is pinned by two independent equilibria
  (bisulfate and bicarbonate) rather than one, which sidesteps the catastrophic-cancellation conditioning issue
  documented above.

  One edge case is worth calling out explicitly: when Ca2+ is present in *large excess* of the total sulfate pool,
  Fortran's precipitation step consumes essentially all of the sulfate, leaving a residual SO4-- pool pinned at a
  ~1e-13 mol bookkeeping floor (`min(1e-5*nSulfmax, 1e3*deps)` -- for any normal-sized system this evaluates to
  the `deps`-scaled term, not the `nSulfmax`-scaled one). Solving the coupled 6-unknown system with this one
  equation living ~13 orders of magnitude below the other five is numerically degenerate -- and this was verified
  to be a limitation of Fortran's own `hybrd` solve, not an artifact of this port: instrumenting the two
  problematic input points directly showed MINPACK returning `info == 4` ("iteration is not making good
  progress") with `sum(|diffK|) ~ 38` instead of ~1e-13, i.e. Fortran's own reported output for these inputs is
  not a converged, trustworthy equilibrium either. In that regime, this port decouples the negligible sulfate
  equilibrium (fixing HSO4- ~ 0, since a pool this small dissociates essentially completely) and solves the
  remaining bicarbonate system with the already-validated `solve_carbonate`, giving a well-converged,
  mass-balanced answer of its own rather than attempting to reproduce Fortran's non-convergent one. This
  decoupling is not expected to -- and, by construction, does not -- bit-match Fortran's unreliable output for
  such inputs; `tests/test_carb_sulf.py` documents and tests this explicitly rather than silently working around
  it.
* Also not ported: PEG systems (subgroup 154), the 3-parameter temperature dependence (BRR/CRR, dataset numbers
  500-800 / 2000-2434), the solvent-mixture reference state (`solvmixrefnd`, never enabled by the Fortran web
  driver), and dataset-specific input conversions (`SpecialInputConcConversion`; the web version uses the default
  path).
* Tmolal in example 0003 differs from the dump by ~1e-8 because the Fortran value comes from a different iterate of
  its dissociation loop; expected until the dissociation equilibria are ported.
* **`aiomfac_py.s2as`** (SMILES -> AIOMFAC subgroups): the matching algorithm is vendored from upstream S2AS
  essentially verbatim (see the subpackage's module docstring for the exact, deliberate integration-level
  differences: no blocking `input()` prompt and no uncaught `IndigoException` on invalid SMILES). Validated
  bit-for-bit against upstream's own three example cases (7, 174, and 2823 molecules, `tests/reference/s2as/`,
  generated by actually running the upstream script and comparing every component's subgroup decomposition) --
  0 mismatches across all 3004 molecules combined. Requires the optional `epam.indigo` dependency
  (`pip install aiomfac_py[smiles]`); raises `ImportError` with a clear message if it is not installed, and is
  never imported by `aiomfac_py`'s own top-level package.

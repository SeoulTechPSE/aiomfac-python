# Rebuilding the Fortran reference data

1. `git clone https://github.com/andizuend/AIOMFAC && cd AIOMFAC && git checkout b9cb96d0eb22dc65a5e63edafe1ed97bd07662f2`
2. `git apply /path/to/instrumentation_v3.14.patch` (touches `FortranCode/ModSRunifac.f90` and `FortranCode/AIOMFAC_inout.f90` only)
3. Compile with the command in `FortranCode/build_command_line.txt` (gfortran >= 9; tested with 13.3).
4. Create `Inputfiles/`, `Outputfiles/` next to the executable and put the `Auxiliary/` folder there too
   (without `Auxiliary/Pure_component_smiles_table.csv` every point gets error flag 22).
5. `./AIOMFAC-web.out ./Inputfiles/input_XXXX.txt` -> `Outputfiles/debug_terms.txt` (rename per case).

`dump_sr_params.f90` is a stand-alone helper: compile it together with all model modules (instead of
`Main_IO_driver.f90`) to dump the final SR parameter arrays.

Notes
* Inputs that trigger the viscosity module's Tg lookup (`TgML_Armeli/TgML_SMILES.py`, needs a `.venv`) end with
  error flag 26 and are not usable as references yet (e.g. `Examples_Input/input_0008.txt`).
* `GammaCO2()` overwrites the CO2(aq) activity coefficient after the SR call, so for carbonate systems the SR sub-terms
  of CO2(aq) do not add up to `gnsrln`; this is expected model behaviour.

## Joint bisulfate + bicarbonate (carb_sulf) debug dumps

`SubModDissociationEquil_carb_debug.f90` also dumps `Outputfiles/debug_carb_consts.txt` (pre-solve constants,
including `nSulfmax`/`nHSO4max`/`nHSO4_init`/`nSulf_init`/`nCa_after` for the Ca2+ precipitation step) and
`Outputfiles/debug_carb_final.txt` (converged molar amounts plus `sum(abs(diffK))` and MINPACK's `info` return
code -- `info == 1` means converged, `info == 4` means MINPACK gave up; both files are appended to, so each
point's block must be told apart by order, and `HSO4_and_HCO3_dissociation` is called twice per output point by
the driver, so blocks come in pairs). `debug_terms.txt` (from the base `instrumentation_v3.14.patch`) is the more
useful reference for validating `solve_carb_sulf`/`solve_carbonate`: it is written once per output point, after
convergence, with the final SMC/SMA/ln(activity coefficient) values actually used for the AIOMFAC output.

## MR parameter dump

`dump_mr_params.f90` dumps the MR tables after `MRdata`. `RccTAB` and `qcca1TAB` are private in `ModMRpart.f90`; for
this dump compile a copy of `ModMRpart.f90` in which the two declarations are changed from `private` to `public`
(`sed -i 's/allocatable,private :: RccTAB/allocatable,public :: RccTAB/; s/allocatable,private :: qcca1TAB/allocatable,public :: qcca1TAB/'`).
The model code itself is not modified.

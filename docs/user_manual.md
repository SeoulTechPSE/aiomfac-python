# aiomfac_py equilibrium tool — user manual

This manual describes the main tool of `aiomfac_py` (module `aiomfac_py.tool`, command `aiomfac-tool`). The tool
is a single entry point to the equilibrium solvers built on the AIOMFAC activity model:

* the combined liquid–liquid–solid solver `PhaseEquilibrium` (`aiomfac_py.phase_equilibrium`);
* the inorganic solid–liquid solver `SLESolver` (`aiomfac_py.sle`);
* the UHAERO-type interior-point liquid–liquid solver `solve_pep` (`aiomfac_py.lle`);
* the gas/particle partitioning solver `gp_partition` (`aiomfac_py.gp_partition`);
* the plain AIOMFAC activity evaluation.

A calculation is described by a *case*: a TOML or JSON file, or a Python dictionary. A case names the system, the
feed, the conditions, the *calculation mode* and the solver options. The tool chooses the solver that fits the mode
(or the one you request), runs every point and returns the results in one solver-independent format. That format
can be written as a text summary, JSON or CSV.

The numerical methods themselves are documented in `docs/phase_equilibrium.md` (combined solver) and
`docs/SLE_design_ko.md` (inorganic solver), and in the module docstrings of `lle.py` and `gp_partition.py`.

---

## Contents

1. [Installation](#1-installation)
2. [Quick start](#2-quick-start)
3. [Concepts](#3-concepts)
4. [Case file reference](#4-case-file-reference)
5. [Calculation modes](#5-calculation-modes)
6. [Solvers and their options](#6-solvers-and-their-options)
7. [Results](#7-results)
8. [Python interface](#8-python-interface)
9. [Command line](#9-command-line)
10. [Troubleshooting](#10-troubleshooting)
11. [Scope, validation status and limitations](#11-scope-validation-status-and-limitations)
12. [References](#12-references)
13. [Appendix: databases](#appendix-databases)

---

## 1. Installation

```bash
pip install "aiomfac_py[sle,smiles] @ git+https://github.com/SeoulTechPSE/aiomfac-python.git"
```

The tool itself needs only `numpy` and `scipy` (extra `sle`, or `carbonate`, which installs the same package). The
other extras are optional:

| extra | package | needed for |
|---|---|---|
| `sle` / `carbonate` | `scipy` | every equilibrium mode (all solvers use `scipy`) |
| `smiles` | `epam.indigo` | organics given by SMILES (`smiles = "..."`), the `aiomfac-tool smiles` command |
| `ad` | `jax` | `[solver] hess_scheme = "ad"` (exact Hessians by automatic differentiation) |
| `test` | `pytest` | running the test suite |

Case files in TOML format need Python 3.11 or later (`tomllib`), or the `tomli` package on Python 3.9–3.10. JSON
case files work with every supported Python version.

Installing the package provides the `aiomfac-tool` command. Without installation (for example in a cloned
repository), use `python -m aiomfac_py.tool` with `src` on `PYTHONPATH`; the two forms are equivalent.

Check the installation:

```bash
aiomfac-tool list modes
```

---

## 2. Quick start

### 2.1 From the command line

Write an example case and run it:

```bash
aiomfac-tool template lle > lle.toml
aiomfac-tool run lle.toml
```

The case (`examples/cases/lle.toml`):

```toml
title = "pinic acid + ammonium sulfate at RH 0.3: liquid-liquid phase separation (no crystallization)"

[calculation]
mode = "lle"
rh = 0.3

[system]
T_K = 298.15

[[system.organics]]
name = "pinic_acid"
subgroups = [[1, 2], [2, 2], [3, 2], [4, 1], [137, 2]]

[feed]
pinic_acid = 0.006243
"NH4+" = 0.015136
"SO4--" = 0.007568

[solver]
engine = "pe"
```

The output (`examples/cases/output/lle.txt`):

```
pinic acid + ammonium sulfate at RH 0.3: liquid-liquid phase separation (no crystallization): mode=lle, engine=pe, T=298.15 K
[lle/pe] RH=0.3000: converged
  liquid 0: n=0.0108478 mol, a_w=0.30000; x: Water=0.2572, pinic_acid=0.5755, NH4+=0.1115, SO4--=0.05577
  liquid 1: n=0.030254 mol, a_w=0.30000; x: Water=0.3095, pinic_acid=0, NH4+=0.4603, SO4--=0.2302
  TPD_min=0, F=0.02475959805
```

The particle separates into an organic-rich liquid (liquid 0) and a salt-rich liquid (liquid 1). Both are in
equilibrium with water vapour at RH 0.3 (a_w = 0.30000). Ammonium sulfate is not allowed to crystallize in mode
`lle`. The tangent-plane distance `TPD_min = 0` means that no further liquid can lower the Gibbs energy.

Write JSON or CSV instead of text:

```bash
aiomfac-tool run lle.toml -o result.json
aiomfac-tool run lle.toml -o result.csv
```

### 2.2 From Python

```python
from aiomfac_py.tool import run

res = run("lle.toml")                       # or run(dict) or run(Case)
p = res[0]                                  # one PointResult per RH
print(p.status, p.n_liquids)                # converged 2
print(p.liquids[0]["mole_fractions"])       # {'Water': 0.257..., 'pinic_acid': 0.575..., ...}
res.to_json("result.json")
```

The same case as a dictionary:

```python
case = {
    "calculation": {"mode": "lle", "rh": 0.3},
    "system": {"T_K": 298.15,
               "organics": [{"name": "pinic_acid", "subgroups": [[1, 2], [2, 2], [3, 2], [4, 1], [137, 2]]}]},
    "feed": {"pinic_acid": 0.006243, "NH4+": 0.015136, "SO4--": 0.007568},
}
res = run(case)
```

---

## 3. Concepts

### 3.1 Components and species

A system consists of

* **water**, always present;
* **organics**, each defined by its AIOMFAC subgroup decomposition (or by a SMILES string, converted with S2AS);
* **ions**, given by their keys (`"Na+"`, `"NH4+"`, `"SO4--"`, …; see the appendix). Salts are not components:
  a salt in the feed is split into its ions (`[salts] NaCl = 1.0` gives 1 mol Na+ and 1 mol Cl-).

Ions are **stoichiometric totals**. Bisulfate, bicarbonate, hydroxide and dissolved CO2 are not entered separately.
The solvers determine their amounts from the reaction equilibria in every liquid:

| species | given as | solvers |
|---|---|---|
| HSO4- | H+ and SO4-- (`HSO4-` in the feed is converted to H+ + SO4--) | all |
| HCO3-, CO3--, CO2(aq) | CO3-- (total carbonate) and H+ (proton excess) | `pe`, `sle` |
| OH- | a negative proton excess: `OH-` in the feed is converted to −H+ | `pe`, `sle` (carbonate systems) |

The list of ions of a case is taken from the feed and the gases. You can also give it explicitly with
`[system] ions`. A carbonate system always carries H+ as its proton-excess component. In a carbonate system the
total amount of H+ may be negative (a basic particle).

### 3.2 Feed and units

The feed gives the amount of every **non-water** component in the particle. Amounts are in mol, and the results are
extensive: doubling the feed doubles every amount and leaves compositions and activities unchanged. The water
content is not part of the feed in the fixed-RH modes. It follows from the condition a_w = RH. The feed can be
written in four ways, and the tool adds them up:

| section | units | content |
|---|---|---|
| `[feed]` | mol | organics (by name) and ions (by key) |
| `[feed_mass_g]` | g | organics and ions; converted with the molar masses of the subgroups and ions |
| `[salts]` | mol of formula units | salts of the salt table (appendix), split into ions |
| `[salts_mass_g]` | g | salts, converted with their molar masses |

Water is given (`[feed] Water = ...`, in mol, or `[feed_mass_g] Water = ...`, in g) only in two cases: in mode
`activity`, and in mode `lle` with engine `pep`, which fixes the total composition.

Molar masses of organics come from the AIOMFAC subgroup table. For example, pinic acid with the subgroups above
has 186.166 g/mol (`aiomfac-tool smiles` prints it). Ion and salt molar masses are those of AIOMFAC
(`aiomfac-tool list ions`, `aiomfac-tool list salts`).

### 3.3 Conditions

* **Temperature** `[system] T_K` (K). AIOMFAC's middle-range interactions do not depend on temperature. The
  equilibrium constants of solids, gases and the bisulfate and carbonate reactions do (see
  `docs/phase_equilibrium.md`, Sect. 12).
* **Relative humidity** `[calculation] rh`, between 0 and 1 (exclusive). Water is *open*: the particle exchanges
  water with a gas phase of fixed RH, and every liquid has a_w = RH at equilibrium. A list of RH values gives one
  calculation per value (an RH scan).
* **Gases** `[gas]`: NH3, HNO3, HCl and CO2 can exchange between the particle and a gas phase (Sect. 4.6).
  * **Open system:** the partial pressures are fixed (an infinite reservoir).
  * **Closed system:** a finite amount of air holds the gas, and the total of each volatile species (particle plus
    gas) is conserved.

### 3.4 Phases

* **Liquids:** any number of liquid phases, up to `max_liquids` (default 3). Each liquid contains every component.
  A component can be a trace in one liquid, and it may be reported as 0 when the solver removes its trace
  (Sect. 10).
* **Solids:** pure crystalline salts and hydrates from a database of 46 solids with temperature-dependent solubility
  products (appendix). A solid can form when all of its ions are present.
* **Gas:** the volatile inorganic gases of `[gas]`. The semivolatile organics of mode `gas_particle` have their own
  description (Sect. 5.5).

### 3.5 Equilibrium, metastable states and paths

Crystallization in aerosol particles is often kinetically hindered, so the tool distinguishes three kinds of
calculations:

* **Equilibrium:** every candidate solid may form (`solids = "all"`). This is the state of lowest Gibbs energy.
* **Metastable with respect to crystallization:** no solid may form (`solids = "none"`, or mode `lle`). Liquids are
  still in equilibrium with each other and with the gas. This describes supersaturated particles.
* **Paths:** along decreasing RH (mode `drying_path`), a solid may form only after its supersaturation in the
  liquid has exceeded a critical value. After that it may stay or dissolve as equilibrium requires. This models
  efflorescence.

A list of solid keys (`solids = ["ammonium_sulfate"]`) allows only those solids.

---

## 4. Case file reference

A case file has the sections below. Keys not listed here are rejected in `[solver]`, and ignored in the other
sections. In TOML, keys that contain `+`, `-`, `(` or `)` must be quoted, for example `"NH4+" = 0.01` and
`"(NH4)2SO4" = 1.0`.

### 4.1 Top level

| key | type | default | meaning |
|---|---|---|---|
| `title` | string | `""` | printed with the results and stored in the JSON output |

### 4.2 `[calculation]`

| key | type | default | meaning |
|---|---|---|---|
| `mode` | string | `"equilibrium"` | calculation mode (Sect. 5) |
| `rh` | number or list | — | relative humidity (0–1). Required by every mode except `activity`, `deliquescence`, `efflorescence` and `lle` with engine `pep` |

### 4.3 `[system]`

| key | type | default | meaning |
|---|---|---|---|
| `T_K` | number | — (required) | temperature [K] |
| `ions` | list of strings | from feed and gases | ion keys of the system; aliases HSO4-, HCO3-, OH- are converted |
| `organics` | array of tables | `[]` | organic components, see below |

Each organic (`[[system.organics]]`) has

| key | type | meaning |
|---|---|---|
| `name` | string | unique name; used as the feed key and in the results |
| `subgroups` | list of `[subgroup, count]` | AIOMFAC subgroup decomposition (subgroup numbers of AIOMFAC-web) |
| `smiles` | string | alternative to `subgroups`: SMILES string, converted with S2AS (needs `epam.indigo`) |

`aiomfac-tool smiles "<SMILES>" <name>` prints the subgroup line for a SMILES string, which you can paste into a
case. S2AS leaves some atoms unmatched (halogens; most nitrogen groups other than nitrate and peroxide groups), and
AIOMFAC then ignores those atoms. Check the decomposition of such compounds before trusting the results.

### 4.4 Feed sections

`[feed]`, `[feed_mass_g]`, `[salts]`, `[salts_mass_g]` as described in Sect. 3.2. Keys of `[feed]` and
`[feed_mass_g]` are organic names, ion keys, or `Water`. Keys of the salt sections are salt formulas of the salt
table (`aiomfac-tool list salts`).

### 4.5 `[solver]`

| key | applies to | values (default first) | meaning |
|---|---|---|---|
| `engine` | all | `"auto"`, or an engine of the mode | solver (Sect. 6); `auto` = the first engine of the mode |
| `solids` | `pe`, `sle` | `"all"`, `"none"`, list of solid keys | candidate solids; the default of mode `stability` is `"none"`, and mode `lle` always uses `"none"` |
| `max_liquids` | `pe` | `3` | largest number of liquid phases |
| `max_outer` | `pe` | `8` | largest number of phase additions (outer iterations) |
| `path` | `pe`, `sle` | `"warm"`, `"independent"` | RH scans: start each RH from the previous result, or solve every RH from scratch |
| `speciation` | `pe` | `"explicit"`, `"internal"` | HSO4-, HCO3-, OH-, CO2(aq) as explicit species of the liquids, or speciated inside every activity evaluation |
| `inner_method` | `pe` | `"newton"`, `"rand"`, `"barrier"` | inner solver for a fixed number of liquids (Sect. 6.1) |
| `hess_scheme` | `pe` | `"split"`, `"ad"`, `"central"` | Hessian of the activities (Sect. 6.1) |
| `tpd_method` | `pe` | `"ss"`, `"newton"` | minimization in the stability test |
| `seed_method` | `pe` | `"linesearch"`, `"fixed"` | size of a newly added liquid |
| `tol_tpd` | `pe` | `1e-7` | a liquid set is stable when the smallest TPD is above `-tol_tpd` |
| `tpd_early_stop` | `pe` | `1e-3` | stop scanning stability-test starts once a TPD below `-tpd_early_stop` is found |
| `trace_tol`, `si_tol`, `hess_reuse_tol`, `tpd_hess_reuse_tol` | `pe` | see `docs/phase_equilibrium.md` | numerical tolerances; normally not changed |
| `k_mode` | `pe`, `sle` | `"fitted"`, `"anchored"`, `"thermo"` | representation of the solubility products K(T) (Sect. 6.2) |
| `rh_tol` | `deliquescence`, `efflorescence` | `2e-4` | bisection tolerance in RH |
| `eps_init`, `tol` | `pep` | `0.03`, `1e-9` | initial phase simplex and convergence tolerance of `solve_pep` |
| `gp_method` | `gp` | `"lm"`, `"pseudo_transient"`, `"successive_substitution"` | solver of the gas/particle problem |
| `gp_max_iter`, `gp_tol` | `gp` | `60`, `1e-6` | iteration limit and tolerance of `gp_partition` |
| `check_lle` | `gp` | `true` | one liquid–liquid check of the converged particle composition |

### 4.6 `[gas]`

Open system:

```toml
[gas]
mode = "open"
partial_pressure_atm = {HCl = 1.0e-9, NH3 = 5.0e-9}
```

Closed system:

```toml
[gas]
mode = "closed"
total_mol = {HNO3 = 1.0e-7, NH3 = 1.0e-7}   # mol of each gas initially in the gas phase
n_air = 41.0                                 # mol of air (about 1 m3 at 1 atm and 298 K)
P_atm = 1.0                                  # total pressure [atm]
```

Gases: `NH3`, `HNO3`, `HCl`, `CO2`. Each one exchanges through a reaction with the stoichiometric ions
(`aiomfac-tool list gases`): HNO3(g) ⇌ H+ + NO3-, HCl(g) ⇌ H+ + Cl-, NH3(g) + H+ ⇌ NH4+, and
CO2(g) + H2O ⇌ CO3-- + 2 H+ (CO2 only in carbonate systems). The ions of a gas are added to the ion list
automatically. In the results:
* `gas` is the amount in the gas phase (closed system), or the net amount released by the particle (open system;
  negative means uptake);
* `p_gas` gives the partial pressures in atm.

### 4.7 Mode-specific sections

| section | mode | keys |
|---|---|---|
| `[drying]` | `drying_path` | `ln_s_crit` (number or table by solid), or `erh` (table by solid: efflorescence RH of the binary salt); `never` (list of solids that never crystallize) |
| `[efflorescence]` | `efflorescence` | `ln_s_crit` or `erh` as in `[drying]`; `lo`, `hi` (RH bracket, default 0.05, 0.995) |
| `[deliquescence]` | `deliquescence` | `lo`, `hi` (RH bracket, default 0.05, 0.995) |
| `[gas_particle]` | `gas_particle` | see Sect. 5.5 |

### 4.8 `[output]`

| key | values | meaning |
|---|---|---|
| `file` | path | output file used by `aiomfac-tool run` when `-o` is not given |
| `format` | `"text"`, `"json"`, `"csv"` | output format; default from the file extension, else text |

---

## 5. Calculation modes

| mode | engines (default first) | needs `rh` | result per |
|---|---|---|---|
| `activity` | `aiomfac` | no | composition |
| `equilibrium` | `pe`, `sle` | yes | RH |
| `lle` | `pe`, `pep` | `pe`: yes; `pep`: no | RH (`pe`) or composition (`pep`) |
| `sle` | `sle`, `pe` | yes | RH |
| `gas_particle` | `gp` | yes | RH |
| `drying_path` | `pe` | yes (decreasing order is used) | RH |
| `deliquescence` | `sle` | no | feed |
| `efflorescence` | `sle` | no | feed |
| `stability` | `pe` | yes | RH |

Every example below is one of the files in `examples/cases`, and its output is in `examples/cases/output`.
`aiomfac-tool template <name>` prints the case.

### 5.1 `activity` — activities of a given liquid

The AIOMFAC activities of one liquid of given composition; there is no equilibrium calculation. Water must be in
the feed. Activities of neutrals are on the mole-fraction scale. AIOMFAC uses the mole fraction of the
dissociated-ion basis, in which every ion counts as a species. Activities of ions are on the molal scale (mol per kg
of water plus organics). In acid systems, H+ and SO4-- are stoichiometric totals: the bisulfate equilibrium is
solved first, and their activities are those of the free ions.

Example (`activity.toml`): 1 mol water, 0.05 mol pinic acid and 0.02 mol ammonium sulfate.

```
[activity/aiomfac]: converged (neutrals: mole-fraction scale (dissociated-ion basis); ions: molal scale)
  liquid 0: n=1.11 mol, a_w=0.93958; x: Water=0.9009, pinic_acid=0.04505, NH4+=0.03604, SO4--=0.01802
    activity: Water=0.93958, pinic_acid=0.36208, NH4+=1.1695, SO4--=0.086191
    activity coefficient: Water=1.0429, pinic_acid=8.0381, NH4+=0.79886, SO4--=0.11775
```

Each liquid dict also holds the ion molalities (`molality`) and `ln_a`. To evaluate many compositions, the
underlying classes are faster than repeated tool calls (`aiomfac_py.ActivityModel`, or
`aiomfac_py.phase_equilibrium.LiquidModel`).

### 5.2 `equilibrium` — liquid–liquid–solid equilibrium

The general problem: at fixed T and RH, find the number of liquid phases, their compositions, the solids present
and (with `[gas]`) the gas exchange, so that the Gibbs energy of the whole system is minimal. Engine `pe` solves it
for any system (organics and ions). Engine `sle` is for inorganic systems; it allows at most one liquid.

Example (`equilibrium.toml`): pinic acid (1.15 g) with ammonium sulfate (0.778 g) and ammonium nitrate (0.222 g),
300 K, RH 0.8 → 0.2.

```
[equilibrium/pe] RH=0.8000: converged
  liquid 0: n=0.0174152 mol, a_w=0.80000; x: Water=0.6171, pinic_acid=0.3542, NH4+=0.0156, SO4--=0.002561, NO3-=0.01047
  liquid 1: n=0.0919417 mol, a_w=0.80000; x: Water=0.7529, pinic_acid=9.205e-05, NH4+=0.1553, SO4--=0.06355, NO3-=0.02818
  TPD_min=0, F=-0.009035592287
[equilibrium/pe] RH=0.6000: converged
  solids [mol]: ammonium_sulfate=0.00524775
  liquid 0: n=0.0146399 mol, a_w=0.60000; x: Water=0.4631, pinic_acid=0.4219, NH4+=0.05862, SO4--=0.002279, NO3-=0.05406
  liquid 1: n=0.0124035 mol, a_w=0.60000; x: Water=0.5337, pinic_acid=5.588e-08, NH4+=0.2576, SO4--=0.04891, NO3-=0.1598
  TPD_min=0, F=0.0007136125034
[equilibrium/pe] RH=0.4000: converged
  solids [mol]: ammonium_nitrate=0.00193882, ammonium_sulfate=0.0058689
  liquid 0: n=0.0116791 mol, a_w=0.40000; x: Water=0.3233, pinic_acid=0.5289, NH4+=0.0747, SO4--=0.001613, NO3-=0.07147
  TPD_min=0, F=0.003450959949
[equilibrium/pe] RH=0.2000: converged
  solids [mol]: ammonium_nitrate=0.00208855, ammonium_sulfate=0.00587563
  liquid 0: n=0.00922369 mol, a_w=0.20000; x: Water=0.1778, pinic_acid=0.6697, NH4+=0.07689, SO4--=0.001313, NO3-=0.07426
  TPD_min=0, F=0.005203007
```

The sequence along decreasing RH:
* at 0.8, two liquids;
* at 0.6, ammonium sulfate crystallizes and two liquids remain;
* at 0.4, ammonium nitrate also crystallizes, and one organic-rich liquid with the remaining ions is left.

With `path = "warm"` (default), each RH starts from the previous result. The result does not depend on this choice:
the stability test runs at every point.

### 5.3 `lle` — liquid–liquid equilibrium without solids

The same problem as `equilibrium` with every solid suppressed: the particle is supersaturated with respect to the
salts but in equilibrium otherwise.

**Engine `pe`** (fixed RH, water open). See the quick-start example (Sect. 2.1).

**Engine `pep`** (fixed total composition, water closed) uses the primal-dual interior-point method of
`aiomfac_py.lle.solve_pep` (Amundson et al., 2006). The water amount is part of the feed, and electrolytes are
given as salts, which `pep` treats as undissociated components (binary salts only). No RH is given: a_w is a
result. Example (`lle_pep.toml`): 0.6 mol water, 0.3 mol pinic acid, 0.1 mol ammonium sulfate.

```
[lle/pep]: converged (fixed total composition (water closed))
  n_outer_iter=24
  liquid 0: n=0.413325 mol, a_w=0.54885; x: Water=0.7901, pinic_acid=8.893e-08, (NH4)2SO4=0.2099
  liquid 1: n=0.586675 mol, a_w=0.54885; x: Water=0.466, pinic_acid=0.5114, (NH4)2SO4=0.02261
```

Use `pep` for closed systems at fixed water content (for example, to reproduce phase diagrams). Use `pe` for
particles in equilibrium with ambient RH. `pep` reports phase fractions and compositions, with activities
(`activity`) from AIOMFAC. It has no gases and no solids.

### 5.4 `sle` — solid–liquid equilibrium of an inorganic system

Salts, their ions and water; no organics. Engine `sle` (`SLESolver`) is the primal-dual active-set solver of
Amundson et al. (2006). It allows at most one aqueous phase and decides explicitly when the aqueous phase
disappears (status `dry`). Engine `pe` gives the same states; with `pe` a vanishing liquid is reported as `dry`
when all solutes are crystalline.

Example (`sle.toml`): 1 mol NaCl and 1 mol (NH4)2SO4.

```
[sle/sle] RH=0.9000: converged (aqueous)
  liquid 0: n=40.7114 mol, a_w=0.90000; x: Water=0.8772, Na+=0.02456, Cl-=0.02456, NH4+=0.04913, SO4--=0.02456
[sle/sle] RH=0.8000: converged (aqueous)
  liquid 0: n=23.3576 mol, a_w=0.80000; x: Water=0.7859, Na+=0.04281, Cl-=0.04281, NH4+=0.08563, SO4--=0.04281
[sle/sle] RH=0.7000: converged (aqueous)
  liquid 0: n=17.1947 mol, a_w=0.70000; x: Water=0.7092, Na+=0.05816, Cl-=0.05816, NH4+=0.1163, SO4--=0.05816
[sle/sle] RH=0.6000: dry (aqueous phase unstable at this RH (all salts crystalline))
  solids [mol]: sal_ammoniac=1, thenardite=0.5, ammonium_sulfate=0.5
[sle/sle] RH=0.5000: dry (aqueous phase unstable at this RH (all salts crystalline))
  solids [mol]: sal_ammoniac=1, thenardite=0.5, ammonium_sulfate=0.5
```

The dry particle is not NaCl + (NH4)2SO4. It is the assemblage of lowest Gibbs energy formed from the same ions:
NH4Cl, Na2SO4 and the remaining (NH4)2SO4.

With `solids = "none"`, engine `sle` returns the supersaturated aqueous solution (`SLESolver.aqueous_state`): the
water content and the saturation index of every solid, without any solid forming.

**Gases.** Example (`sle_gas.toml`): an NH4NO3 particle (1e-6 mol) in a closed volume of 41 mol air that initially
holds 1e-7 mol each of HNO3 and NH3.

```
[sle/sle] RH=0.8000: converged (closed gas phase (ideal gas + air))
  gas [mol]: HNO3=1.69598e-07, NH3=1.70225e-07; p [atm]: HNO3=4.137e-09, NH3=4.152e-09
  liquid 0: n=6.95618e-06 mol, a_w=0.80000; x: Water=0.7325, NH4+=0.1337, NO3-=0.1338, H+=9.023e-05
[sle/sle] RH=0.5000: dry (closed dry state (solids + gas); no aqueous phase stable at this RH)
  solids [mol]: ammonium_nitrate=8.33544e-07
  gas [mol]: HNO3=2.66456e-07, NH3=2.66456e-07; p [atm]: HNO3=6.499e-09, NH3=6.499e-09
```

Closed-system SLE calculations are slower (about 8 s per point here) because the dry and wet problems are solved
in turn.

### 5.5 `gas_particle` — partitioning of semivolatile organics

The gas/particle partitioning method of Zuend et al. (2010): semivolatile organics distribute between a gas phase
of volume `V_gas_m3` and the particle (with water at a_w = RH and a non-volatile salt), so that the partial pressure
of each organic equals its activity times its pure-component vapour pressure (modified Raoult's law). The organics
are not part of `[system]`; they are defined in `[gas_particle]`:

| key | meaning |
|---|---|
| `salt` | non-volatile salt (binary salt formula of the salt table) |
| `n_salt_mol` | mol of the salt |
| `V_gas_m3` | volume of the gas phase [m³] |
| `[[gas_particle.species]]` | one table per organic: `name`, `subgroups` or `smiles`, `p0_Pa` (saturation vapour pressure [Pa]), `n_total_mol` (gas + particle), `M_kg_mol` (optional; from the subgroups if omitted) |

Example (`gas_particle.toml`): 3e-8 mol glycerol over 1e-8 mol ammonium sulfate in 1 m³.

```
[gas_particle/gp] RH=0.3000: converged (particle phases from the one-shot LLE diagnostic (lle.solve_pep))
  water_activity=0.3, n_water_PM_mol=1.34497e-08, n_iter=131, glycerol_particle_mol=3.58388e-17, glycerol_gas_mol=3e-08, glycerol_particle_fraction=1.19463e-09, glycerol_activity=0.00325607
  liquid 0: phase fraction=1; x: Water=0.5736, glycerol=6.631e-08, (NH4)2SO4=0.4264
[gas_particle/gp] RH=0.6000: converged (particle phases from the one-shot LLE diagnostic (lle.solve_pep))
  water_activity=0.6, n_water_PM_mol=4.47918e-08, n_iter=55, glycerol_particle_mol=7.54641e-12, glycerol_gas_mol=2.99925e-08, glycerol_particle_fraction=0.000251547, glycerol_activity=0.00325525
  liquid 0: phase fraction=1; x: Water=0.8174, glycerol=0.0001377, (NH4)2SO4=0.1825
[gas_particle/gp] RH=0.9000: converged (particle phases from the one-shot LLE diagnostic (lle.solve_pep))
  water_activity=0.9, n_water_PM_mol=1.92995e-07, n_iter=34, glycerol_particle_mol=5.00052e-10, glycerol_gas_mol=2.94999e-08, glycerol_particle_fraction=0.0166684, glycerol_activity=0.0032018
  liquid 0: phase fraction=1; x: Water=0.9484, glycerol=0.002457, (NH4)2SO4=0.04914
```

Glycerol stays almost entirely in the gas phase. Its particle fraction rises from 1e-9 at RH 0.3 to 1.7 % at RH
0.9, as the particle takes up water.

Notes:
* The solver iterates on a forced single particle phase. A liquid–liquid check (`check_lle = true`) is made once at
  the converged composition. It reports the phases (`liquids`, with phase fractions) but is not fed back into the
  partitioning. When it finds two phases, the reported activities are those of the single-phase composition (see
  the `gp_partition` docstring).
* `gp_method = "lm"` (default) solves the coupled system jointly. `"successive_substitution"` fails to converge for
  some multi-organic systems; the module docstring gives a case.
* The inorganic gases of `[gas]` do not apply to this mode. They apply to `equilibrium`, `lle`, `sle`,
  `drying_path` and `stability`.

### 5.6 `drying_path` — crystallization after a critical supersaturation

The RH values are visited in decreasing order. At each RH:
1. The state is solved with the solids that are already *enabled*.
2. A solid becomes enabled when its saturation index ln S in the liquid exceeds its critical value ln S_crit.
3. After that it stays enabled: it may crystallize, or dissolve again if equilibrium requires.

`ln_s_crit = 0` gives the equilibrium path; a large value suppresses the solid (metastable path).

The critical supersaturation is not a thermodynamic property. It has to come from experiment. `[drying] erh` sets
it from a measured efflorescence RH of the pure salt: ln S_crit is the saturation index of the binary aqueous salt
solution at that RH (`implied_ln_s_crit`). `never` lists solids that are never enabled.

Example (`drying_path.toml`): the system of Sect. 5.2. Ammonium sulfate is given the ln S_crit of its efflorescence
at RH 0.35, and ammonium nitrate never crystallizes.

```
[drying_path/pe] RH=0.3500: converged (enabled solids: [])
  ...
  liquid 0: n=0.0112416 mol, a_w=0.35000; x: Water=0.2907, pinic_acid=0.5495, NH4+=0.09566, SO4--=0.03149, NO3-=0.03268
  liquid 1: n=0.0332321 mol, a_w=0.35000; x: Water=0.3556, pinic_acid=0, NH4+=0.4054, SO4--=0.1665, NO3-=0.0724
[drying_path/pe] RH=0.3000: converged (enabled solids: [])
  ...
[drying_path/pe] RH=0.2500: converged (enabled solids: ['ammonium_sulfate'])
  solids [mol]: ammonium_sulfate=0.00581633
  liquid 0: n=0.011532 mol, a_w=0.25000; x: Water=0.1998, pinic_acid=0.5357, NH4+=0.1329, SO4--=0.001314, NO3-=0.1303
  liquid 1: n=0.0033952 mol, a_w=0.25000; x: Water=0.2016, pinic_acid=0, NH4+=0.4075, SO4--=0.01657, NO3-=0.3744
```

In the mixed particle, the supersaturation of ammonium sulfate reaches the critical value only between RH 0.30 and
0.25. That is lower than for the pure salt (0.35), because nitrate and the organic lower its activity. Compare
Sect. 5.2: at equilibrium, ammonium sulfate is already present at RH 0.6. `values["enabled_solids"]` lists the
enabled solids at each point.

### 5.7 `deliquescence` — deliquescence RH

For an inorganic feed, by bisection on the phase assemblage of `SLESolver`:
* `drh` is the lowest RH at which an aqueous phase exists (the mutual deliquescence RH of a mixture);
* `full_dissolution_rh` is the lowest RH above which no solid is left.

The two are equal for a single salt. Example (`deliquescence.toml`):

```
[deliquescence/sle]: converged (bisection on the phase assemblage)
  drh=0.798721, full_dissolution_rh=0.798721
```

The DRH of ammonium sulfate at 298.15 K is 0.80 (measured 0.79–0.80).

### 5.8 `efflorescence` — efflorescence RH

For an inorganic feed, the highest RH at which some salt's supersaturation in the metastable aqueous solution
reaches its critical value (`[efflorescence] ln_s_crit` or `erh`; default ln S_crit = 0.5). Returns `erh` and the
crystallizing `salt`. Example (`efflorescence.toml`), with ln S_crit implied by ERH 0.35. It reproduces 0.35 by
construction:

```
[efflorescence/sle]: converged
  erh=0.349984, salt=ammonium_sulfate
```

For mixtures this gives the RH at which the first salt is expected to crystallize, for given critical
supersaturations.

### 5.9 `stability` — is one liquid stable?

The one-liquid state is computed at each RH (`max_liquids = 1`; solids `"none"` by default), and the stability test
is applied to it. `values["stable"]` is true when the smallest tangent-plane distance is above `-tol_tpd`.
`values["tpd_min"]` is that distance. A negative TPD means the particle lowers its Gibbs energy by forming a second
liquid. Its magnitude measures how far the state is from stability, so it can be used to locate the onset of liquid–liquid phase
separation (LLPS) along RH.

Example (`stability.toml`): pinic acid + ammonium sulfate.

```
[stability/pe] RH=0.9950: converged
  stable=True, tpd_min=0
[stability/pe] RH=0.9900: converged
  stable=True, tpd_min=0
[stability/pe] RH=0.9800: converged (one-liquid state unstable: it splits into several liquids (run mode 'lle' or 'equilibrium'))
  stable=False, tpd_min=-0.00145613
[stability/pe] RH=0.9500: converged (one-liquid state unstable: ...)
  stable=False, tpd_min=-0.236079
```

LLPS begins between RH 0.99 and 0.98. The status is `converged` in both cases: it refers to the one-liquid state,
and an unstable state is an answer, not a failure.

---

## 6. Solvers and their options

| engine | solver | system | phases | conditions |
|---|---|---|---|---|
| `pe` | `PhaseEquilibrium` | water + organics + ions | up to `max_liquids` liquids, solids, gas | fixed T, RH; open or closed gases |
| `sle` | `SLESolver` | water + ions | one aqueous phase, solids, gas | fixed T, RH; open or closed gases |
| `pep` | `lle.solve_pep` | water + organics + binary salts (undissociated) | liquids | fixed T, total composition |
| `gp` | `gp_partition` | water + semivolatile organics + one salt | particle (one phase, LLE check) + gas | fixed T, RH, gas volume |
| `aiomfac` | `LiquidModel.ln_a` | water + organics + ions | one liquid (given) | fixed T, composition |

### 6.1 `pe`: the combined solver

`PhaseEquilibrium` minimizes the transformed Gibbs energy of all liquids, solids and gases at fixed RH. It works in
the ion basis, with electroneutrality of every liquid. An outer tangent-plane stability test adds liquids until no
further liquid lowers the energy. The method is described in `docs/phase_equilibrium.md`. Options:

* **`inner_method`** — solver for a fixed set of liquids:
  * `"newton"` (default): Newton iteration in the null space of the balances. Liquids carry a short logarithmic
    barrier, and solids are handled by an active set. Trace entries are removed and restored when needed.
  * `"rand"`: logarithmic amounts with element-potential restoration of the balances (RAND type). There is no
    barrier and no trace removal, and every species stays in every liquid (a trace of 1e-54 is resolved). It gave
    the same results as `"newton"` on all validation sets (paper_1 and paper_2 calculations, a 344-state stress
    test) and is up to 70 % faster in carbonate systems. Explicit speciation only.
  * `"barrier"`: the original log-barrier method for liquids and solids; kept for comparison.
* **`hess_scheme`** — Hessian of the activities:
  * `"split"` (default): exact ideal part plus a finite-difference excess part, which is reused while the
    composition changes little.
  * `"ad"`: the exact Jacobian by automatic differentiation (needs `jax`). Compilation costs 0.5–1.5 s per system
    and temperature, after which a solve is 15–49 % faster. Use it for RH scans and sweeps of one system.
  * `"central"`: central differences at every step (slowest; for testing).
* **`speciation`**:
  * `"explicit"` (default): HSO4-, HCO3-, OH- and CO2(aq) are explicit species of each liquid. Their reactions
    reach equilibrium through the minimization.
  * `"internal"`: every activity evaluation speciates the stoichiometric components (the original formulation;
    needed by `inner_method = "barrier"` with carbonate and organics).
* **`tpd_method`**: `"ss"` (successive substitution with a Newton finish, default) or `"newton"` (barrier Newton
  from every start).
* **`seed_method`**: `"linesearch"` (size of a new liquid by a one-dimensional Gibbs-energy minimization, default) or
  `"fixed"` (seeds of 0.5, 0.2, 0.05 of the largest possible transfer).
* **`max_liquids`** (3) and **`max_outer`** (8) bound the number of liquids and of phase additions.
* **`path`**:
  * `"warm"` (default): each RH of a scan starts from the previous result. Typically twice as fast, with identical
    results. A warm start that fails is repeated from scratch automatically.
  * `"independent"`: every point is solved from scratch.

Recommended settings:

| situation | settings |
|---|---|
| single calculations, any system | defaults |
| RH scans and sweeps of one system | `hess_scheme = "ad"` (with `jax`), `path = "warm"` |
| carbonate systems, systems with extreme traces | `inner_method = "rand"` |
| comparison with earlier versions | `inner_method = "barrier"`, `tpd_method = "newton"`, `seed_method = "fixed"` |

### 6.2 `sle`: the inorganic solver

`SLESolver` handles one aqueous phase and any subset of the solid database. It decides the dry state (no aqueous
phase) explicitly, by a linear program and a tangent-plane test. It supports open and closed gases, and the
bisections of modes `deliquescence` and `efflorescence`. Options: `solids` (list of keys, or `"none"` for the
metastable aqueous solution) and `k_mode`:
* `"fitted"` (default): K(T) fitted to experimental solubilities with AIOMFAC, where such a fit exists; otherwise
  `"anchored"`.
* `"anchored"`: the literature temperature dependence, with its level anchored to AIOMFAC at 298.15 K.
* `"thermo"`: pure literature values, without anchoring.

`k_mode` applies to engine `pe` as well.

### 6.3 `pep`: fixed-composition liquid–liquid solver

`lle.solve_pep` is the primal-dual interior-point method with an active set of phases (Amundson et al., 2006),
started from a simplex of near-pure phases. It solves the closed problem: total composition including water is
fixed, and there are no solids or gases. Options: `eps_init` (distance of the initial phases from the pure
components), `tol`.

### 6.4 `gp`: gas/particle partitioning

`gp_partition` couples the water content (a_w = RH) with the Raoult's-law partitioning of every organic. Options:
`gp_method`, `gp_max_iter`, `gp_tol`, `check_lle`. See Sect. 5.5.

---

## 7. Results

### 7.1 Point results

`run` returns a `RunResult`, a sequence of `PointResult` objects (one per RH, or one for the single-value modes).
Fields of a `PointResult`:

| field | type | meaning |
|---|---|---|
| `mode`, `engine` | str | the mode and the solver used |
| `rh` | float or None | RH of the point (None for single-value modes) |
| `status` | str | `converged`, `not_converged`, `dry` or `failed` (Sect. 7.2) |
| `message` | str | explanation from the solver |
| `liquids` | list of dict | one per liquid phase (Sect. 7.3) |
| `solids` | dict | mol of each solid present |
| `si` | dict | saturation index ln S of every candidate solid (≤ 0 when it may not form; 0 for solids present) |
| `gas`, `p_gas` | dict | gas amounts [mol] and partial pressures [atm] |
| `gibbs` | float | value of the minimized function (relative; comparable only within one engine and system) |
| `tpd_min` | float | smallest tangent-plane distance of the final stability test (`pe`) |
| `checks` | dict | residuals of the equilibrium conditions (`pe`; Sect. 7.4) |
| `values` | dict | mode-specific numbers (`drh`, `erh`, `salt`, `stable`, `enabled_solids`, gas/particle amounts, …) |
| `raw` | object | the solver's own result object (`PhaseEquilibriumResult`, `SLEResult`, `GPResult`, `PhaseSplitResult`) |
| `n_liquids` | int | number of liquid phases (property) |
| `ok` | bool | status is `converged` or `dry` (property) |

Liquids are listed in order of decreasing organic mole fraction (`pe`).

### 7.2 Status

| status | meaning |
|---|---|
| `converged` | all equilibrium conditions hold to their tolerances, and the stability test is non-negative |
| `dry` | no liquid phase: all solutes are crystalline (`sle` at every RH; `pe` when the last liquid vanishes) |
| `not_converged` | a condition is violated or the stability test is still negative; `message` says which |
| `failed` | the solver gave no result (for example, the RH cannot be reached within AIOMFAC's concentration range) |

`aiomfac-tool run` returns exit code 0 when every point is `converged` or `dry`, 2 when some point is not, and 1 on
an invalid case.

### 7.3 Liquid entries

Each liquid is a dict with

| key | meaning |
|---|---|
| `total_mol` | total amount of the liquid [mol] (`gp`: None, see `phase_fraction`) |
| `water_activity` | a_w of the liquid (= RH at equilibrium) |
| `amounts` | mol of each component (water, organics, ions as stoichiometric totals) |
| `mole_fractions` | mole fractions on the same basis (ions counted individually) |
| `ln_a` | ln a of each component: neutrals on the mole-fraction scale, ions on the molal scale; `null` for a component removed as a trace |
| `molality` | ion molalities [mol per kg of water + organics] |
| `activity`, `activity_coefficient` | mode `activity` (and `activity` for `pep`) |
| `phase_fraction` | `pep`, `gp`: fraction of the total amount in this phase |

At equilibrium, the potential of every component is equal across liquids: ln a of water and of the organics, and
the electroneutral combinations of the ion potentials.

### 7.4 Checks (`pe`)

| check | tolerance for `converged` | meaning |
|---|---|---|
| `max_abs_ln_aw_minus_ln_rh` | 1e-5 | water activity of every liquid vs RH |
| `max_neutral_mu_spread` | 1e-4 | spread of ln a of each organic across liquids |
| `max_ion_mu_residual` | 1e-4 | equality of the ion potentials across liquids (up to the electrostatic gauge) |
| `max_si` | 1e-4 | largest saturation index of a candidate solid (must be ≤ 0) |
| `max_abs_si_present_solids` | 1e-4 | solids present must be exactly saturated |
| `max_abs_gas_residual` | 1e-4 | gas–particle equilibrium of the volatile species |
| `max_charge_residual`, `max_mass_balance_residual` | — | electroneutrality and mass balance (round-off level) |
| `n_absent_entries`, `max_removed_trace` | — | components removed from a liquid as traces, and the largest removed amount |
| `n_line_search_failures` | — | inner iterations that ended on a failed line search |

### 7.5 JSON output

```json
{
 "aiomfac_py_version": "1.1.0",
 "title": "...",
 "mode": "lle",
 "engine": "pe",
 "T_K": 298.15,
 "case": { ... the case as read ... },
 "points": [ { "mode": "lle", "engine": "pe", "rh": 0.3, "status": "converged", "message": "",
               "liquids": [ {"total_mol": ..., "water_activity": ..., "amounts": {...}, "mole_fractions": {...},
                             "ln_a": {...}, "molality": {...}}, ... ],
               "solids": {}, "si": {...}, "gas": {}, "p_gas": {}, "gibbs": ..., "tpd_min": 0.0,
               "checks": {...}, "values": {}, "n_liquids": 2 } ]
}
```

Non-finite numbers (NaN, ±inf, e.g. ln a of a removed trace) are written as `null`. The full example is
`examples/cases/output/lle.json`.

### 7.6 CSV output

One row per point:
* the columns `mode`, `engine`, `rh`, `status`, `n_liquids`, `solids` (`key=mol;...`), `gibbs`, `tpd_min`;
* the mode's `values`;
* `gas_<gas>_mol`;
* for each liquid k: `L<k>_total_mol`, `L<k>_water_activity` and `L<k>_x_<component>`.

The columns are the union over all points; cells of absent liquids are empty. Example:
`examples/cases/output/sle.csv`.

---

## 8. Python interface

```python
from aiomfac_py.tool import Case, run, RunResult, PointResult, CaseError, TEMPLATES

case = Case.from_file("case.toml")      # or Case.from_dict(d); raises CaseError with a clear message
res = run(case)                         # RunResult
for p in res:                           # PointResult
    print(p.rh, p.status, p.n_liquids, p.solids)
res.to_json("out.json"); res.to_csv("out.csv"); print(res.summary())
rows = res.rows()                       # list of flat dicts (e.g. pandas.DataFrame(res.rows()))
```

`Case` attributes after parsing:
* `mode`, `engine`, `T_K`;
* `organics` (`Component` objects), `ions`;
* `feed` (mol, aliases converted), `water`, `salts`;
* `rh`, `gas`, `solver`, `options`.

They can be changed before `run(case)`. Changes are not re-validated, so prefer editing the dict and calling
`Case.from_dict` again.

The configured solvers are available for direct use:

```python
pe = case.phase_equilibrium()           # PhaseEquilibrium with the [solver] options applied
r = pe.solve(case.feed, 0.5, **case.gas_kwargs())
sol = case.sle_solver()                 # SLESolver with solids and k_mode
```

`TEMPLATES` maps template names to the example case texts.

An RH scan with a change of solver option, as a script:

```python
import copy
from aiomfac_py.tool import run, TEMPLATES
import tomllib

base = tomllib.loads(TEMPLATES["lle"])
base["calculation"]["rh"] = [0.9, 0.7, 0.5, 0.3]
for method in ("newton", "rand"):
    d = copy.deepcopy(base)
    d["solver"]["inner_method"] = method
    res = run(d)
    print(method, [(p.rh, p.n_liquids, round(p.gibbs, 10)) for p in res])
```

---

## 9. Command line

```
aiomfac-tool run CASE [-o OUTPUT] [-f {json,csv,text}] [-v]
aiomfac-tool template NAME
aiomfac-tool list {modes,solids,ions,gases,salts,templates}
aiomfac-tool smiles SMILES [NAME]
```

| command | action |
|---|---|
| `run` | runs a case file (`.toml` or `.json`). Output to `-o` or `[output] file`, else to standard output. The format is given by `-f`, `[output] format` or the file extension (default text). `-v` prints the solver's iteration log |
| `template` | prints an example case: `activity`, `deliquescence`, `drying_path`, `efflorescence`, `equilibrium`, `gas_particle`, `lle`, `lle_pep`, `sle`, `sle_gas`, `stability` |
| `list` | lists the modes and engines, the solid database, the ions, the gases, the salt formulas or the templates |
| `smiles` | prints the `[[system.organics]]` entry (subgroups and molar mass) for a SMILES string |

---

## 10. Troubleshooting

**`not_converged` with "stability test still negative".** The solver stopped with a liquid set that is not stable:
* `n_liquids` reached `max_liquids` (raise it), or `max_outer` was exhausted;
* or a new liquid could not be added: close to the RH where it appears, every seed may fall back to the old state.

Try `seed_method = "fixed"` or `inner_method = "rand"`, or move the RH slightly. The reported state is the best one
found; its TPD measures how far it is from stability.

**`not_converged` with "equilibrium conditions violated".** The inner iteration ended before the conditions were met.
The message names the checks (Sect. 7.4). Typical causes:
* extreme supersaturation with `solids = "none"` at low RH, where a liquid would need a salt molality far outside
  AIOMFAC's range (for example several hundred mol/kg; `docs/phase_equilibrium.md`, Sect. 12). The same point with
  solids allowed converges;
* a state very close to a phase boundary.

`inner_method = "rand"` resolves some of these.

**`failed` (engine `sle`).** The RH cannot be reached by the aqueous phase within the concentration cap (400 mol/kg).
Use `solids = "all"` (the salt crystallizes) or a higher RH.

**A component is 0 in one liquid, with `ln_a = null`.** The `newton` inner solver removes trace entries (below 1e-9
of the feed) from a liquid when another liquid holds at least ten times more of them. The removed amount is
negligible for the mass balance, and `checks["max_removed_trace"]` reports the largest one. With
`inner_method = "rand"` no entry is removed, and the trace amount is computed (for example, a mole fraction of
1e-54).

**Results differ slightly between solvers or options.** F agrees to about 1e-10. Equilibrium compositions agree to
the solver tolerances. Trace entries can differ: they are removed in one solver and resolved in another. The phase
count can differ only at a phase boundary, where both answers lie within tolerance.

**Slow calculations.**
* Closed-system SLE takes several seconds per point.
* `pe` cases with many ions and three liquids take about a second per point.
* For scans, use `path = "warm"` (default) and `hess_scheme = "ad"`.

**Organics from SMILES.** The S2AS decomposition leaves halogens and most nitrogen groups unmatched (only nitrate
and peroxide groups are covered), and AIOMFAC then ignores those atoms. Check `aiomfac-tool smiles` before using
such compounds.

**Temperature.** AIOMFAC's middle-range interaction parameters do not depend on temperature. Solubility products
are defined over limited ranges (database `in_range`); outside them, results are extrapolations.

---

## 11. Scope, validation status and limitations

* The AIOMFAC activity model of `aiomfac_py` is a port of AIOMFAC-web v3.14, validated against an instrumented
  Fortran build (251 points, scaled differences below 1e-12).
* The equilibrium solvers (`pe`, `sle`, `pep`, `gp`) are not part of AIOMFAC-web and have no Fortran reference.
  * They are validated by their own equilibrium conditions, by agreement with each other in their common ranges
    (`pe` vs `sle` for inorganic systems; `pe` vs `pep` for liquid–liquid splits), and by the tests in `tests/`.
  * `tests/test_tool.py` checks that every mode of this tool reproduces the solver it dispatches to.
* Not included:
  * kinetics (crystallization is modelled only through the critical supersaturation of `drying_path`);
  * surface tension and Kelvin effects;
  * organic volatility in the fixed-RH modes (organics are non-volatile there; use `gas_particle`);
  * complexation.
* Carbonate systems with organics are consistent with Gibbs–Duhem only to about 1 %, because of AIOMFAC's
  treatment of CO2(aq).
* The stability test of `pe` uses a finite set of starting points. A stable liquid in an unsampled composition
  region can be missed, though this is rare because the test runs on the converged state.

---

## 12. References

* Amundson, N. R., Caboussat, A., He, J. W., Seinfeld, J. H., and Yoo, K. Y.: Primal-dual active-set algorithm for
  chemical equilibrium problems related to the modeling of atmospheric inorganic aerosols, J. Optim. Theory Appl.,
  128, 469–498, 2006.
* Amundson, N. R., Caboussat, A., He, J. W., and Seinfeld, J. H.: Primal-dual interior-point method for an
  optimization problem related to the modeling of atmospheric organic aerosols, J. Optim. Theory Appl., 130,
  375–407, doi:10.1007/s10957-006-9110-z, 2006.
* Michelsen, M. L.: The isothermal flash problem. Part I. Stability, Fluid Phase Equilib., 9, 1–19, 1982.
* Smith, W. R. and Missen, R. W.: Chemical Reaction Equilibrium Analysis: Theory and Algorithms, Wiley, New York,
  1982.
* Zuend, A., Marcolli, C., Luo, B. P., and Peter, T.: A thermodynamic model of mixed organic-inorganic aerosols to
  predict activity coefficients, Atmos. Chem. Phys., 8, 4559–4593, 2008.
* Zuend, A., Marcolli, C., Peter, T., and Seinfeld, J. H.: Computation of liquid-liquid equilibria and phase
  stabilities: implications for RH-dependent gas/particle partitioning of organic-inorganic aerosols, Atmos. Chem.
  Phys., 10, 7795–7820, 2010.
* Zuend, A., Marcolli, C., Booth, A. M., et al.: New and extended parameterization of the thermodynamic model
  AIOMFAC, Atmos. Chem. Phys., 11, 9155–9206, 2011.
* Clegg, S. L., Brimblecombe, P., and Wexler, A. S.: Thermodynamic model of the system H+–NH4+–SO42-–NO3-–H2O at
  tropospheric temperatures, J. Phys. Chem. A, 102, 2137–2154, 1998.

---

## Appendix: databases

The lists are printed by `aiomfac-tool list ...`.

### Ions

| key | AIOMFAC id | charge | M [g/mol] |
|---|---|---|---|
| Li+ | 201 | +1 | 6.941 |
| Na+ | 202 | +1 | 22.990 |
| K+ | 203 | +1 | 39.098 |
| NH4+ | 204 | +1 | 18.038 |
| H+ | 205 | +1 | 1.008 |
| Ca++ | 221 | +2 | 40.078 |
| Mg++ | 223 | +2 | 24.305 |
| F- | 241 | −1 | 18.998 |
| Cl- | 242 | −1 | 35.453 |
| Br- | 243 | −1 | 79.904 |
| I- | 244 | −1 | 126.905 |
| NO3- | 245 | −1 | 62.004 |
| HSO4- | 248 | −1 | 97.071 (given as H+ + SO4--) |
| SO4-- | 261 | −2 | 96.063 |
| CO3-- | 262 | −2 | 60.008 |
| HCO3- | 250 | −1 | 61.016 (given as H+ + CO3--) |
| OH- | 247 | −1 | 17.007 (given as −H+) |

### Salts (for `[salts]`, `[salts_mass_g]`)

NaCl, KCl, NH4Cl, NaNO3, KNO3, NH4NO3, Na2SO4, K2SO4, (NH4)2SO4, MgCl2, Mg(NO3)2, MgSO4, CaCl2, Ca(NO3)2, CaSO4,
H2SO4, NH4HSO4, (NH4)3H(SO4)2, NaHSO4, KHSO4.

### Gases

| key | reaction (gas ⇌ ions) |
|---|---|
| NH3 | NH3(g) + H+ ⇌ NH4+ |
| HNO3 | HNO3(g) ⇌ H+ + NO3- |
| HCl | HCl(g) ⇌ H+ + Cl- |
| CO2 | CO2(g) + H2O ⇌ CO3-- + 2 H+ (carbonate systems) |

### Solids

| key | formula | key | formula |
|---|---|---|---|
| halite | NaCl | sylvite | KCl |
| sal_ammoniac | NH4Cl | nitratine | NaNO3 |
| niter | KNO3 | ammonium_nitrate | NH4NO3 |
| thenardite | Na2SO4 | mirabilite | Na2SO4·10H2O |
| arcanite | K2SO4 | ammonium_sulfate | (NH4)2SO4 |
| bischofite | MgCl2·6H2O | MgCl2_4H2O | MgCl2·4H2O |
| MgCl2_2H2O | MgCl2·2H2O | epsomite | MgSO4·7H2O |
| hexahydrite | MgSO4·6H2O | kieserite | MgSO4·H2O |
| Mg_nitrate_6H2O | Mg(NO3)2·6H2O | gypsum | CaSO4·2H2O |
| anhydrite | CaSO4 | antarcticite | CaCl2·6H2O |
| Ca_nitrate_4H2O | Ca(NO3)2·4H2O | ammonium_bisulfate | NH4HSO4 |
| letovicite | (NH4)3H(SO4)2 | sodium_bisulfate_hydrate | NaHSO4·H2O |
| sodium_bisulfate | NaHSO4 | trisodium_hydrogen_sulfate | Na3H(SO4)2 |
| NaH3_SO4_2_hydrate | NaH3(SO4)2·H2O | natron | Na2CO3·10H2O |
| nahcolite | NaHCO3 | trona | Na3H(CO3)2·2H2O |
| kalicinite | KHCO3 | calcite | CaCO3 |
| aragonite | CaCO3 | magnesite | MgCO3 |
| nesquehonite | MgCO3·3H2O | dolomite | CaMg(CO3)2 |
| gaylussite | Na2Ca(CO3)2·5H2O | pirssonite | Na2Ca(CO3)2·2H2O |
| burkeite | Na6CO3(SO4)2 | thermonatrite | Na2CO3·H2O |
| sodium_carbonate | Na2CO3 | sodium_hydroxide | NaOH |
| portlandite | Ca(OH)2 | brucite | Mg(OH)2 |
| glauberite | Na2Ca(SO4)2 | syngenite | K2Ca(SO4)2·H2O |

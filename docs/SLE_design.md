# aiomfac_py SLE Solver: Data Collection Plan and Design (v0.0.24+)

Based on: Amundson, Caboussat, He, Seinfeld, Yoo (2006), *Primal-Dual Active-Set Algorithm for Chemical Equilibrium Problems Related to the Modeling of Atmospheric Inorganic Aerosols*, JOTA 128(3):469–498 (hereafter **JOTA-1**).
Implementation: `src/aiomfac_py/solids.py` (solid DB), `src/aiomfac_py/sle.py` (solver), `tests/test_sle.py` (53 tests), `tools/` (calibration and validation scripts).

## 1. Scope

| Item | Supported | Notes |
|---|---|---|
| Ions | Li, Na, K, NH4, Mg, Ca / F, Cl, Br, I, NO3, SO4, CO3 | Aqueous-phase activities use AIOMFAC (LR+MR+SR) from `lle.py` as is |
| Acidic sulfate systems (H+, HSO4−) | **Supported** | H+ and SO4²⁻ are treated as total (stoichiometric) quantities, and the HSO4⁻ ⇌ H⁺+SO4²⁻ equilibrium is solved inside the aqueous activity calculation (Sec. 3.4). Solids: NH4HSO4, letovicite |
| Gas phase (NH3/HNO3/HCl) | Supported (open system with fixed p; closed system) | `gases.py`, `solve(p_gas=…)`, `solve_closed(…)` (Sec. 3.5) |
| Solids | 25 species (DB below) | Double salts: glauberite, syngenite, letovicite. NaHSO4, KHSO4, H2SO4 hydrates, etc. are not included |
| Input | Moles of salts (electroneutral; `H2SO4`, `NH4HSO4`, `HSO4-` allowed), T, RH | Output: state (dry/aqueous/solid+aqueous), solid masses, water mass, molalities, SI |

## 2. JOTA-1 ↔ Code Correspondence

| JOTA-1 | This implementation |
|---|---|
| Gibbs minimization + canonical stoichiometric form (eq. 14–19) | Neutral reaction vector V (solid j → dissociation into ions ν_ij), ion moles n_ion = b − V u |
| Mass-action law, solid saturation constraint ln k_s + a_sᵀλ ≥ 0, complementarity (eq. 23–29) | SI_j = Σν_ij ln a_i − c_j ≤ 0, u_j ≥ 0, u_j·SI_j = 0 |
| Fixed RH (water activity a_w=RH) | `AqueousIons.solve_water`: the water amount is determined by brentq so that a_w=RH (variable: total ion molality ln M) |
| Table 1 active-set + Newton, reduced KKT (eq. 53) | `SLESolver._wet`: Hessian = V_Sᵀ D V_S, D=∂ln a/∂n (a_w fixed, central differences along neutral directions) |
| ratio test (eq. 55) | Step-length limiting + detection of solid exhaustion / new saturation |
| Hydrate water (eq. 48) | Absorbed as c_j = ln K_j(T) − h_j ln RH (a constant, since RH is fixed) |
| Dry-state determination (Sec. 4.1) | LP (min c·u, V u=b) determines the solid assemblage and dual variables y → if the tangent plane distance TPD ≥ 0, there is no aqueous solution |

Differences from JOTA-1: (0) acidic systems are handled with total-quantity components plus internal equilibrium instead of the paper's "noncomponent" approach (Sec. 3.4); (i) the activity model is AIOMFAC rather than ideal solution/Pitzer (possibly nonconvex), so a multi-start TPD + BFGS refinement + RH continuation (homotopy) safety net is added; (ii) finite differences instead of an analytic Jacobian; (iii) gas phase not supported; (iv) for a single salt at RH≠DRH the extended formulation degenerates, so it is handled by the LP+TPD path.

## 3. Ksp(T) and Hydrate Data: Collection Plan and Results

### 3.1 Source hierarchy
1. **PHREEQC `pitzer.dat`** analytic expression log K = A1 + A2T + A3/T + A4 log10 T + A5/T² + A6T² (including the phase definition for each solid) → quality A
2. **van't Hoff based on NBS/CODATA ΔfH, ΔfG, Cp** (for cases absent from PHREEQC, such as NH4 and NO3 salts) → quality B
3. **Estimates** (estimated ΔCp, no anchor) → quality C
4. Experimental solubility values (CRC/literature, `tools/solubility_data.py`, source labeled web/recall) are used for validation and fitting.

### 3.2 AIOMFAC consistency calibration (K(T) modes)
- `thermo`: pure thermodynamic K(T).
- `anchored`: literature T dependence + an offset (in ln units, `ANCHOR_OFFSETS`) that matches the AIOMFAC IAP at 298 K. In saturated solutions, AIOMFAC's γ matches K to within 0–0.3 log10.
- `fitted` (default when a fit exists): fits ln K = ln k0 − ΔH/R(1/T−1/298.15)+… against experimental solubility curves under AIOMFAC (`tools/fit_solids.py`; `KFIT`). These ΔH_eff/ΔCp_eff **absorb the AIOMFAC γ(T) error along the saturation line**, so they are not guaranteed away from the saturation line. This is because the AIOMFAC MR term is T-independent.
- Outside the validity window (around 0–60 °C), 1/T-linear extrapolation is used.

### 3.3 DB (generated table)

| key | formula | h | quality | anchor m (mol/kg) | fit window (K) |
|---|---|---|---|---|---|
| halite | NaCl | 0 | A | 6.146 | 273–333 |
| sylvite | KCl | 0 | A | 4.803 | 273–333 |
| sal_ammoniac | NH4Cl | 0 | B | 7.39 | 273–333 |
| nitratine | NaNO3 | 0 | B | 10.8 | 273–333 |
| niter | KNO3 | 0 | B | 3.75 | 273–333 |
| ammonium_nitrate | NH4NO3 | 0 | B | 26.0 | 273–333 |
| thenardite | Na2SO4 | 0 | A | (mirabilite offset) | 313–373 |
| mirabilite | Na2SO4·10H2O | 10 | A | 1.97 | 273–303 |
| arcanite | K2SO4 | 0 | A | 0.691 | 273–333 |
| ammonium_sulfate | (NH4)2SO4 | 0 | B | 5.77 | 273–333 |
| bischofite | MgCl2·6H2O | 6 | A | 5.81 | 273–333 |
| MgCl2_4H2O | MgCl2·4H2O | 4 | C | – | – |
| MgCl2_2H2O | MgCl2·2H2O | 2 | C | – | – |
| epsomite | MgSO4·7H2O | 7 | A | 3.0 | 273–313 |
| hexahydrite | MgSO4·6H2O | 6 | B | – | – |
| kieserite | MgSO4·H2O | 1 | B | – | – |
| Mg_nitrate_6H2O | Mg(NO3)2·6H2O | 6 | C | – | – |
| gypsum | CaSO4·2H2O | 2 | A | – | – |
| anhydrite | CaSO4 | 0 | A | – | – |
| antarcticite | CaCl2·6H2O | 6 | C | 7.4 | – |
| Ca_nitrate_4H2O | Ca(NO3)2·4H2O | 4 | C | 8.675 | 273–313 |
| ammonium_bisulfate | NH4HSO4 | 0 | B | Clegg 1998 | 273–323 |
| letovicite | (NH4)3H(SO4)2 | 0 | B | Clegg 1998 | 273–323 |
| sodium_bisulfate_hydrate | NaHSO4·H2O | 1 | B | Clegg 1998 (298 K) | – |
| sodium_bisulfate | NaHSO4 | 0 | B | Clegg 1998 (298 K) | – |
| trisodium_hydrogen_sulfate | Na3H(SO4)2 | 0 | B | Clegg 1998 (298 K) | – |
| NaH3_SO4_2_hydrate | NaH3(SO4)2·H2O | 1 | C | Clegg 1998 (tentative) | – |
| glauberite | Na2Ca(SO4)2 | 0 | B | – | – |
| syngenite | K2Ca(SO4)2·H2O | 1 | B | – | – |

Quality: A = PHREEQC + solubility cross-check, B = van't Hoff (NBS), C = estimate. Solids without an anchor/fit use the `thermo` mode (no AIOMFAC offset applied), so use them with caution for quantitative work.

### 3.4 Acidic sulfate solids and HSO4⁻ treatment
**Formulation.** The aqueous components are set as stoichiometric totals H(total), SO4(total), NH4, …. At equilibrium the chemical potential of a total component equals ln a of the free ion (HSO4⁻ ⇌ H⁺+SO4²⁻), so the solid saturation index SI_j = Σν_i ln a_i − c_j, the mass balance, the KKT conditions, the active set, and the TPD test all hold **without any change of form**. However, `AqueousIons.ln_gamma_aw` first obtains the free H⁺ and SO4²⁻ molalities from the total molalities and then returns ln a (effective ln γ = ln γ_free + ln(m_free/m_total)).
**HSO4⁻ equilibrium.** K₂(T) uses the expression in `dissociation.py` (Knopf et al. 2003) as is. The solution is a quadratic equation with the γ ratio Γ = γ_H γ_SO4 / γ_HSO4 held fixed, plus fixed-point iteration (2–5 activity evaluations, warm-started from the previous solution); it agrees with the Fortran-validated `solve_bisulfate` to within a relative error of 1e-7.
**Solids.** NH4HSO4, letovicite (NH4)3H(SO4)2, and the Na acid salts NaHSO4, NaHSO4·H2O, Na3H(SO4)2, and NaH3(SO4)2·H2O are defined by dissociation into free ions (NH4⁺, Na⁺, H⁺, SO4²⁻).
- K: thermodynamic values from Clegg et al. (1998, J. Phys. Chem. A 102:2137 Table 2; 102:2155 Table 3). Since ln xK is on a mole-fraction, free-ion basis, it is converted to molality as ln K_m = ln xK + n_ions·ln(1000/18.01528) (the a_w^h term for hydrate water is left unchanged). Comparison with existing DB values: NaCl 3.645 vs 3.606, mirabilite −2.820 vs −2.830, NH4NO3 2.50 vs 2.47, etc., within ±0.05–0.15 (AS 0.09 vs 0.04).
- T dependence: NH4HSO4 ΔH −15.17 kJ/mol, ΔCp −242.3; letovicite ΔH −5.32, ΔCp −630.4 (paper values, 273–323 K). The Na acid salts have only 298 K values, so ΔH=0 (quality B; NaH3(SO4)2·H2O is C because the paper calls it "tentative").
- Independent DRH check (`tools/calibrate_acid_solids.py`): with this K and AIOMFAC, the pure-salt DRHs are NH4HSO4 37.9 % (literature ~40 %), letovicite 70.1 % (69.5 %), NaHSO4·H2O 57.6 %, NaHSO4 40.9 %, and Na3H(SO4)2 74.3 %. That is, even without anchoring they are within ~2–3 percentage points of the literature (the literature DRH values themselves are quoted from memory).
- Na/H/SO4 system: in a normal scan the sequence is thenardite+Na3H → Na3H+NaHSO4·H2O → NaHSO4·H2O → aqueous solution. **Limitation**: if the feed exactly matches the double-salt stoichiometry (e.g., Na 1.5 : H 0.5 : SO4 1, RH 0.7), the problem is degenerate by the phase rule, the Newton solve fails, and the homotopy fallback is very slow (>100 s). Neighboring compositions take 0–3 s. Countermeasures (time limit / small perturbation) are on the roadmap.
- Solids excluded in this phase: KHSO4, H2SO4 hydrates, double salts (NH4NO3 phase transitions, (NH4)2SO4·2NH4NO3, etc.).

**Validation.**
- Speed: about 2 ms per activity evaluation, 0.2–1 s for a wet solution, 3–9 s for the dry determination (TPD) (speed-up is on the roadmap).
- Pure H2SO4 solution: about 38 wt% sulfuric acid at a_w = 0.6 (literature about 38–40 %, from memory), about 52 wt% at a_w = 0.3.
- Phase order (RH 30 %, 298 K): with increasing acidity, AS+LET → AHS+LET → AHS → liquid (excess H2SO4). At RH 50 %, AHS exists as a liquid because it is above its DRH (40 %).
- Letovicite dissolves incongruently (in the model): above about 69.5 %, AS remains as a residual solid.
- Global-optimality sampling checks for random acid feeds are in `tools/check_sle_sampling.py`.

**Caution.** The DRHs and phase boundaries of acidic mixed systems include the error of the AIOMFAC activity model itself (relative to the Clegg model), not just Clegg's K. A quantitative comparison with literature phase diagrams (Tang 1980; Clegg et al. 1998) remains to be done.

### 3.5 ERH (efflorescence RH)
Because crystallization is kinetic, no thermodynamic ERH exists. The user supplies a critical supersaturation ln S_crit (`efflorescence_rh`), and `implied_ln_s_crit` is provided to back-calculate it from the literature ERH (e.g., NaCl 45 % → ln S ≈ 3.0).

## 4. Solver Algorithm (`SLESolver.solve`)
1. An LP yields the dry solid assemblage and the dual y.
2. Tangent plane distance: feed point + salt-pair mixing grid + BFGS refinement (top 2) → if the minimum is ≥ 0, the state is `dry`.
3. If an aqueous solution is present, active-set Newton (`_wet`): updates the active solid set and satisfies the mass balance, a_w=RH, and KKT.
4. On failure, continuation starting at RH 0.999; the final fallback is dry.

Provided: `scan_rh` (warm start), `aqueous_state` (metastable aqueous solution + SI), `deliquescence_rh`, `full_dissolution_rh`, `efflorescence_rh`.

Usage example:
```python
from aiomfac_py import SLESolver, feed_from_salts
s = SLESolver(["Na","NH4","Cl","SO4"])
r = s.solve(feed_from_salts({"NaCl":1.0,"(NH4)2SO4":1.0}), 298.15, 0.70)
print(r.status, r.solids, r.water_kg)
```

### 3.5 Gas phase (NH3, HNO3, HCl)
**Formulation.** Treat gas j as "a column like the solids": the ions that disappear from the aqueous solution when 1 mol of gas forms are the column vector G_j (HNO3: H⁺+NO3⁻, HCl: H⁺+Cl⁻, NH3: NH4⁺−H⁺, i.e., when NH3 forms, NH4⁺ is converted to H⁺). The equilibrium condition is SI_j = G_jᵀ ln a − ln K_j − ln p_j = 0, with K_j on a molality basis (`gases.py`).
- Data (Clegg et al. 1998): xK′H(NH3) = 1.066e11 atm⁻¹ (ΔH −86.25 kJ/mol, ΔCp 34.35), HNO3 xKH = 853.1 atm⁻¹ (molality basis 2.63e6, T dependence from Eq. 15 of the paper), HCl 662.1 atm⁻¹ (298 K; the paper gives no T dependence, so it is estimated from NBS ΔH −74.85 kJ/mol, ΔCp −165.5, quality C). **Note**: the (1/Tr − 1/T) in the extracted text of Eq. 12 appears to be an extraction error and is consistent with the paper's own ΔH and with the ΔH of 184.2 kJ/mol for Kp(NH4NO3) (reproduced as 182–184 by this module). → The project owner (co-author) confirmed on 2026-10-05 that (1/T − 1/Tr) is the correct form, so this matches the code.
- Consistency: Kp(NH4NO3) = K_s/(K′H K_H) = 4.22e-17 atm² (Clegg 4.36e-17, 3 %), temperature dependence ΔH ≈ 184 kJ/mol.

**Open system** (`solve(feed, T, rh, p_gas={…})`): p is fixed; the gas amount u_g has a free sign (+ means moving to the gas, − means absorption) and is always in the active set. The initial interior point is obtained by LP. The dry determination (LP+TPD) is used as is, but if the LP is unbounded (the reservoir is supersaturated with respect to the solids), no equilibrium is reported. Feeds in which all ions are volatile (e.g., pure NH4NO3) are degenerate because the total amount is not fixed.

**Closed system** (`solve_closed(feed, gas_total, T, rh, n_air, P)`): the gas amounts g_j > 0 are unknowns, ln p_j = ln(P g_j/(n_air+Σg)) (ideal gas + air). (i) The state without an aqueous solution (solid + gas) is solved by active-set Newton in dual (element potential) space (y, u_A: V_Aᵀy = c_A, b − V_A u_A − G g(y) = 0), and aqueous stability is checked with TPD; (ii) if the aqueous solution is stable, it is solved by the `_wet` active-set Newton with the gas columns included as positive variables (including the ln p term). Because Φ is convex, the solution is global.
- Validation: for solid NH4NO3 + vapor, p(NH3)p(HNO3) = Kp(T) (relative error 1e-6); complete evaporation; mass balance (1e-16) and ln p agreement for the (NH4)2SO4 + HNO3 + NH3 wet system; NaCl + HNO3 → chloride loss (HCl(g)); open-system p recovery (1e-7).
- Limitations: inert apart from ideal gas and air, no coupling to organics (CO2 is in §3.6), all data validated at 298 K (HCl T dependence unvalidated), open-system degenerate cases as above. The cost of the closed-system dry determination is dominated by TPD (1–5 s).

### 3.6 Carbonates and CO2
- **Components.** CO3⁻⁻ is the total carbon C_T, and H⁺ is the sign-free proton excess P = [H⁺] − [OH⁻] + [HCO3⁻] + 2[CO2] (+[HSO4⁻]). HCO3⁻ and OH⁻ are partitioned internally in `AqueousIons._speciate_carb` (K1, K2, and Kw are the same as the Fortran fit expressions in `carbonate.py`; fixed-γ-ratio iteration + brentq on ln h). OH⁻ solids (portlandite, brucite, etc.) are treated as −H⁺ through `n_oh`, with ln K′ = ln K − n_oh ln Kw(T).
- **Validation.** Compared with the Fortran port `solve_carbonate`: HCO3 relative difference ≲ 2e-6, H⁺ ≲ 6e-4, CO3 ≲ 7e-5.
- **Approximations.** Water consumption and the mole-fraction contribution of CO2(aq) are neglected, so Gibbs–Duhem holds only to about 1e-3. γ(CO2) is `gamma_co2_mr` (salting-out).
- **Solids (PHREEQC pitzer.dat at 298 K, etc.).** natron, nahcolite, trona, kalicinite, calcite, aragonite, magnesite, nesquehonite, dolomite, gaylussite, pirssonite, burkeite, portlandite, brucite. thermonatrite, Na2CO3, and NaOH use NBS values recalled from memory (quality C, not rechecked against the original). Na2CO3 drying order: natron → thermonatrite → anhydrous. trona is 298 K only.
- **CO2(g).** CO2 + H2O ⇌ CO3⁻⁻ + 2H⁺ (`Gas.h = −1`), ln K = ln K_H(T) (PHREEQC analytic expression) + ln K1 + ln K2. The open system recovers p exactly; in the closed system, when the reservoir ≫ the particle, Newton is ill-conditioned, so it is replaced by fixed-point iteration of the fixed-p solve (`_closed_fixed_point`).
- **Limitations.** NaOH in a system without carbonate (no anion) is unsupported; the closed-system Newton uses a fallback when the particle/reservoir ratio is very small.

## 5. Validation Results
- Tests: `tests/test_sle.py` 52 (~13 s) + `tests/test_sle_acid.py` 17 (~45 s) pass (full suite 274 passed; `tests/test_sle_gas.py` 8). Covered: DB electroneutrality and finiteness of K, fitted==anchored (298 K), single-salt DRH (298 K) within ±1.6 percentage points of the literature, fitted solubility vs handbook ≤ 4 %, Na2SO4 hydrate transition (≈305.5 K reproduced), NaCl–KCl phase order and MDRH (0.715–0.74), KKT/mass balance/a_w=RH, active-set Gibbs ≤ SLSQP, monotonicity of RH scans, ERH consistency, rejection of non-neutral inputs. Acid systems: agreement of HSO4⁻ equilibrium (1e-7), KKT, mass balance, a_w=RH, pure H2SO4 wt%, phase order.
- Random cross-validation (against SLSQP global Gibbs minimization): 0/40 (seed 42), 0/60 (seed 7) disagreements.
- Metathesis reaction ΔG: within ≈ 0.2 kJ of NBS.
- DRH(T): NH4NO3 73.7 % (273 K) → 61.2 % (298 K) → 54.0 % (313 K), NaCl nearly flat — consistent with literature trends.
- NaCl+(NH4)2SO4 RH scan: thenardite 66 %, NH4Cl+thenardite 64 %, dry at ≤62–63 %. The water-amount discontinuity at MDRH is physical.
- NaCl+KCl MDRH 72.9 % / FDRH 78.7 % have not yet been checked against the literature (benchmark target: "A database for deliquescence and efflorescence relative humidities of compounds with atmospheric relevance", 2021).

Figures: `docs/figs/sle_drh_T.png`, `sle_scan_nacl_as.png`, `sle_solubility.png`, `sle_phase_nacl_kcl.png`.

## 6. Limitations
- No gas phase. Acid systems only up to NH4HSO4·letovicite (the NaHSO4 family and H2SO4 hydrates are not included). The JOTA-1 acid examples are reproduced only qualitatively; the quantitative comparison is incomplete.
- Fits are valid for 0–60 °C; high and low temperatures (<273 K) are extrapolated.
- Quality C solids (MgCl2 low hydrates, Mg(NO3)2, antarcticite, Ca(NO3)2) have low quantitative reliability.
- The fitted ΔH_eff absorbs the AIOMFAC γ(T) error — T-extrapolation away from the saturation line is uncertain.
- Double salts (letovicite, (NH4)3H(SO4)2, NH4HSO4) and NH4NO3 solid-phase transitions are not included.
- The dry-state TPD takes up to ~10 s for multi-ion feeds (typically 0.02–0.5 s).
- `anchor_m`, "recall" solubilities (mirabilite, epsomite), quality B/C items, and NBS thermochemical values **need to be checked against the original sources before publication**.
- The single-salt RH≠DRH degeneracy is bypassed with LP+TPD (requires mixing degrees of freedom s ≤ N−2).

## 7. Roadmap
1. Acid-system extension (in progress): the NH4/Na acid solids are already reflected using the Clegg 1998 K. Remaining: KHSO4, H2SO4 hydrates, double salts, quantitative comparison with literature phase diagrams (Tang 1980, Clegg 1998) and the JOTA-1 examples, and handling of the double-salt stoichiometry degeneracy.
2. NH3/HNO3/HCl gas-phase coupling: done (Sec. 3.5). Remaining: check the HCl temperature dependence against the original, NH4NO3 phase transitions/double salts, and open-system degeneracy when all ions are volatile.
3. Analytic Jacobian (AIOMFAC derivatives) and TPD acceleration → speed and robustness (the dry determination for acid systems is slow).
4. ERH calibration: systematize ln S_crit using the 2021 DRH/ERH database.
5. Low-temperature (<273 K) data and literature values for the quality C/B solids.

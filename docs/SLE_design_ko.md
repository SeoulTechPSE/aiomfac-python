# aiomfac_py SLE 솔버: 데이터 수집 계획 및 설계 (v0.0.24+)

기반 논문: Amundson, Caboussat, He, Seinfeld, Yoo (2006), *Primal-Dual Active-Set Algorithm for Chemical Equilibrium Problems Related to the Modeling of Atmospheric Inorganic Aerosols*, JOTA 128(3):469–498 (이하 **JOTA-1**).
구현: `src/aiomfac_py/solids.py`(고체 DB), `src/aiomfac_py/sle.py`(솔버), `tests/test_sle.py`(53 tests), `tools/`(보정·검증 스크립트).

## 1. 범위

| 항목 | 지원 | 비고 |
|---|---|---|
| 이온 | Li, Na, K, NH4, Mg, Ca / F, Cl, Br, I, NO3, SO4, CO3 | 용액 활동도는 `lle.py`의 AIOMFAC(LR+MR+SR)를 그대로 사용 |
| 산성 황산염계 (H+, HSO4−) | **지원** | H+와 SO4²⁻를 총량(화학양론 이온)으로 두고 HSO4⁻ ⇌ H⁺+SO4²⁻를 수용액 활동도 계산 안에서 풂 (3.4절). 고체: NH4HSO4, letovicite |
| 기체상 (NH3/HNO3/HCl) | 지원(열린계 p 고정, 닫힌계) | `gases.py`, `solve(p_gas=…)`, `solve_closed(…)` (3.5절) |
| 고체 | 25종 (아래 DB) | 복염은 glauberite, syngenite, letovicite. NaHSO4·KHSO4·H2SO4 수화물 등은 미포함 |
| 입력 | 염 몰수(전기중성; `H2SO4`, `NH4HSO4`, `HSO4-` 허용), T, RH | 출력: 상태(dry/aqueous/solid+aqueous), 고체 질량, 물 질량, 몰랄농도, SI |

## 2. JOTA-1 ↔ 코드 대응

| JOTA-1 | 본 구현 |
|---|---|
| Gibbs 최소화 + 정준 화학양론 형태 (eq. 14–19) | 중성 반응 벡터 V (고체 j → 이온 ν_ij 해리), 이온 몰수 n_ion = b − V u |
| 질량작용 법칙, 고체 포화 제약 ln k_s + a_sᵀλ ≥ 0, 상보성 (eq. 23–29) | SI_j = Σν_ij ln a_i − c_j ≤ 0, u_j ≥ 0, u_j·SI_j = 0 |
| 고정 RH (물 활동도 a_w=RH) | `AqueousIons.solve_water`: a_w=RH가 되도록 물량을 brentq로 결정 (총 이온 몰랄농도 ln M 변수) |
| Table 1 active-set + Newton, reduced KKT (eq. 53) | `SLESolver._wet`: Hessian = V_Sᵀ D V_S, D=∂ln a/∂n (a_w 고정, 중성 방향 중심차분) |
| ratio test (eq. 55) | 단계 길이 제한 + 고체 소진/신규 포화 판정 |
| 수화물 물 (eq. 48) | c_j = ln K_j(T) − h_j ln RH로 흡수 (RH 고정이므로 상수) |
| 건조 상태 판정 (Sec. 4.1) | LP(min c·u, V u=b)로 고체 조합·쌍대변수 y 결정 → 접선평면거리 TPD ≥ 0이면 수용액 없음 |

JOTA-1과의 차이점: (0) 산성계는 논문의 "noncomponent" 방식 대신 총량 성분 + 내부 평형으로 처리 (3.4절); (i) 활동도 모델이 이상용액/Pitzer가 아니라 AIOMFAC (비볼록 가능), 따라서 TPD 다중 시작 + BFGS 정제 + RH 연속법(homotopy) 안전망 추가; (ii) 해석적 야코비안 대신 유한차분; (iii) 기체 미지원; (iv) 단일염 RH≠DRH에서 확장 정식화가 퇴화하므로 LP+TPD 경로로 처리.

## 3. Ksp(T)·수화물 데이터 수집 계획 및 결과

### 3.1 출처 위계
1. **PHREEQC `pitzer.dat`** 해석식 log K = A1 + A2T + A3/T + A4 log10 T + A5/T² + A6T² (고체별 상 정의 포함) → 품질 A
2. **NBS/CODATA ΔfH, ΔfG, Cp 기반 van't Hoff** (NH4, NO3 염 등 PHREEQC에 없는 경우) → 품질 B
3. **추정** (ΔCp 추정, 앵커 없음) → 품질 C
4. 용해도 실험값(CRC/문헌, `tools/solubility_data.py`, 출처 web/recall 표기)으로 검증·피팅.

### 3.2 AIOMFAC 일관성 보정 (K(T) 모드)
- `thermo`: 순수 열역학 K(T).
- `anchored`: 문헌 T-의존성 + 298 K에서 AIOMFAC IAP과 맞추는 오프셋(ln 단위, `ANCHOR_OFFSETS`). AIOMFAC은 포화 용액에서 γ가 K에 0~0.3 log10 이내로 일치.
- `fitted`(기본, 피팅이 있으면): 실험 용해도 곡선에 대해 AIOMFAC 하에서 ln K = ln k0 − ΔH/R(1/T−1/298.15)+… 을 피팅(`tools/fit_solids.py`; `KFIT`). 이 ΔH_eff/ΔCp_eff는 **포화선상의 AIOMFAC γ(T) 오차를 흡수**하므로 포화선 밖에서는 보증되지 않음. AIOMFAC MR 항이 T-비의존임에 기인.
- 유효 창 밖(0–60 °C 근방)은 1/T 선형 외삽.

### 3.3 DB (생성표)

| key | 화학식 | h | 품질 | 앵커 m (mol/kg) | 피팅 창 (K) |
|---|---|---|---|---|---|
| halite | NaCl | 0 | A | 6.146 | 273–333 |
| sylvite | KCl | 0 | A | 4.803 | 273–333 |
| sal_ammoniac | NH4Cl | 0 | B | 7.39 | 273–333 |
| nitratine | NaNO3 | 0 | B | 10.8 | 273–333 |
| niter | KNO3 | 0 | B | 3.75 | 273–333 |
| ammonium_nitrate | NH4NO3 | 0 | B | 26.0 | 273–333 |
| thenardite | Na2SO4 | 0 | A | (mirabilite 오프셋) | 313–373 |
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
| sodium_bisulfate_hydrate | NaHSO4·H2O | 1 | B | Clegg 1998(298 K) | – |
| sodium_bisulfate | NaHSO4 | 0 | B | Clegg 1998(298 K) | – |
| trisodium_hydrogen_sulfate | Na3H(SO4)2 | 0 | B | Clegg 1998(298 K) | – |
| NaH3_SO4_2_hydrate | NaH3(SO4)2·H2O | 1 | C | Clegg 1998(잠정) | – |
| glauberite | Na2Ca(SO4)2 | 0 | B | – | – |
| syngenite | K2Ca(SO4)2·H2O | 1 | B | – | – |

품질: A = PHREEQC + 용해도 대조, B = van't Hoff(NBS), C = 추정. 앵커/피팅 없는 고체는 `thermo` 모드(AIOMFAC 오프셋 미적용)이므로 정량 사용 시 주의.

### 3.4 산성 황산염 고체와 HSO4⁻ 처리
**정식화.** 수용액 성분을 화학양론 총량 H(총), SO4(총), NH4, …로 둡니다. 평형에서 총량 성분의 화학퍼텐셜은 자유 이온의 ln a와 같으므로(HSO4⁻ ⇌ H⁺+SO4²⁻), 고체 포화지수 SI_j = Σν_i ln a_i − c_j, 질량수지, KKT 조건, active-set, TPD 검사가 **형태 변화 없이** 그대로 성립합니다. 다만 `AqueousIons.ln_gamma_aw`가 총 몰랄농도에서 자유 H⁺, SO4²⁻ 몰랄농도를 먼저 구해 ln a를 돌려줍니다(유효 ln γ = ln γ_free + ln(m_free/m_total)).
**HSO4⁻ 평형.** K₂(T)는 `dissociation.py`(Knopf et al. 2003)의 식을 그대로 사용합니다. 풀이는 γ비 Γ = γ_H γ_SO4 / γ_HSO4를 고정한 2차방정식 + 고정점 반복(2–5회 활동도 평가, 이전 해로 warm start)이고, Fortran과 검증된 `solve_bisulfate`와 상대오차 1e-7 이내로 일치합니다.
**고체.** NH4HSO4, letovicite (NH4)3H(SO4)2와 Na 산성염 NaHSO4, NaHSO4·H2O, Na3H(SO4)2, NaH3(SO4)2·H2O를 자유 이온(NH4⁺, Na⁺, H⁺, SO4²⁻)으로의 해리로 정의했습니다.
- K: Clegg et al.(1998, J. Phys. Chem. A 102:2137 Table 2; 102:2155 Table 3)의 열역학 값. ln xK는 몰분율 기준·자유 이온 기준이므로 ln K_m = ln xK + n_ions·ln(1000/18.01528)로 몰랄 환산(수화수의 a_w^h 항은 그대로). 기존 DB 값과 대조: NaCl 3.645 vs 3.606, mirabilite −2.820 vs −2.830, NH4NO3 2.50 vs 2.47 등 ±0.05–0.15 (AS 0.09 vs 0.04).
- T 의존성: NH4HSO4 ΔH −15.17 kJ/mol, ΔCp −242.3; letovicite ΔH −5.32, ΔCp −630.4 (논문값, 273–323 K). Na 산성염은 298 K 값만 있어 ΔH=0 (품질 B, NaH3(SO4)2·H2O는 논문이 "tentative"라 C).
- DRH 독립 점검(`tools/calibrate_acid_solids.py`): 이 K와 AIOMFAC이 내는 순수 염 DRH는 NH4HSO4 37.9 %(문헌 ~40 %), letovicite 70.1 %(69.5 %), NaHSO4·H2O 57.6 %, NaHSO4 40.9 %, Na3H(SO4)2 74.3 %. 즉 앵커링 없이도 문헌과 ~2–3 %p 이내입니다(DRH 문헌값 자체는 기억 기준).
- Na/H/SO4계: 정상 스캔 시 thenardite+Na3H → Na3H+NaHSO4·H2O → NaHSO4·H2O → 수용액 순. **한계**: 공급물이 이중염 화학양론과 정확히 일치(예: Na 1.5 : H 0.5 : SO4 1, RH 0.7)하면 상률상 퇴화되어 Newton 풀이가 실패하고 homotopy 대체 경로가 매우 느립니다(>100 s). 인접 조성은 0–3 s. 대응(시간 제한/미소 섭동)은 로드맵.
- 이번 단계에서 뺀 고체: KHSO4, H2SO4 수화물, 복염(NH4NO3 상전이, (NH4)2SO4·2NH4NO3 등).
**검증.**
- 속도: 활동도 평가당 약 2 ms, 습윤 해는 0.2–1 s, 건조 판정(TPD)은 3–9 s (속도 개선은 로드맵).
- 순수 H2SO4 용액: a_w = 0.6에서 황산 약 38 wt%(문헌 약 38–40 %, 기억 기준), a_w = 0.3에서 약 52 wt%.
- 상 순서(RH 30 %, 298 K): 산성도 증가에 따라 AS+LET → AHS+LET → AHS → 액체(과잉 H2SO4). RH 50 %에서는 AHS가 DRH(40 %) 위라 액체로 존재.
- letovicite는 (모델상) 비일치 용해: 69.5 % 근처 위에서 AS가 잔류 고체로 남음.
- 무작위 acid 공급물에 대한 전역 최적성 표본검사는 `tools/check_sle_sampling.py`.
**주의.** 산성 혼합계 DRH·상 경계에는 Clegg의 K가 아니라 AIOMFAC 활동도 모델 자체 오차(Clegg 모델 대비)가 들어갑니다. 문헌 상도(Tang 1980; Clegg et al. 1998)와의 정량 비교는 남아 있습니다.

### 3.5 ERH(효력상 RH)
결정화는 동역학적이므로 열역학 ERH는 존재하지 않음. 사용자가 임계 과포화도 ln S_crit을 입력(`efflorescence_rh`), 문헌 ERH로부터 역산하는 `implied_ln_s_crit` 제공 (예: NaCl 45% → ln S ≈ 3.0).

## 4. 솔버 알고리즘 (`SLESolver.solve`)
1. LP로 건조 고체 조합과 쌍대 y 산출.
2. 접선평면거리: 공급점 + 염쌍 혼합 격자 + BFGS 정제(상위 2개) → 최소값 ≥ 0이면 `dry`.
3. 수용액 존재 시 active-set Newton(`_wet`): 활성 고체 집합 갱신, 질량수지·a_w=RH·KKT 만족.
4. 실패 시 RH 0.999에서 시작하는 연속법, 최종 폴백은 dry.
`scan_rh`(warm start), `aqueous_state`(준안정 수용액 + SI), `deliquescence_rh`, `full_dissolution_rh`, `efflorescence_rh` 제공.

사용 예:
```python
from aiomfac_py import SLESolver, feed_from_salts
s = SLESolver(["Na","NH4","Cl","SO4"])
r = s.solve(feed_from_salts({"NaCl":1.0,"(NH4)2SO4":1.0}), 298.15, 0.70)
print(r.status, r.solids, r.water_kg)
```

### 3.5 기체상 (NH3, HNO3, HCl)
**정식화.** 기체 j를 "고체와 같은 열": 기체가 1 mol 생길 때 수용액에서 사라지는 이온을 열 벡터 G_j로 둡니다 (HNO3: H⁺+NO3⁻, HCl: H⁺+Cl⁻, NH3: NH4⁺−H⁺, 즉 NH3가 생기면 NH4⁺가 H⁺로 바뀜). 평형조건은 SI_j = G_jᵀ ln a − ln K_j − ln p_j = 0이며 K_j는 몰랄 기준 (`gases.py`).
- 데이터(Clegg et al. 1998): xK′H(NH3) = 1.066e11 atm⁻¹ (ΔH −86.25 kJ/mol, ΔCp 34.35), HNO3 xKH = 853.1 atm⁻¹ (몰랄 2.63e6, 논문 식 15의 T 의존성), HCl 662.1 atm⁻¹ (298 K; T 의존성은 논문에 없어 NBS ΔH −74.85 kJ/mol, ΔCp −165.5로 추정, 품질 C). **주의**: 식 12 추출 텍스트의 (1/Tr − 1/T)는 추출 오류로 보이며 논문 자신의 ΔH와 Kp(NH4NO3)의 ΔH 184.2 kJ/mol과 일치합니다(본 모듈 182~184로 재현). → 프로젝트 소유자(공저자)가 2026-10-05에 (1/T − 1/Tr)가 정상임을 확인했으므로 코드와 일치합니다.
- 일관성: Kp(NH4NO3) = K_s/(K′H K_H) = 4.22e-17 atm² (Clegg 4.36e-17, 3 %), 온도 의존성 ΔH ≈ 184 kJ/mol.
**열린계** (`solve(feed, T, rh, p_gas={…})`): p 고정, 기체량 u_g는 부호가 자유(+는 기체로 이동, −는 흡수)이고 항상 활성 집합에 포함됩니다. 초기 내점은 LP로 구합니다. 건조 판정(LP+TPD)은 그대로 쓰되 LP가 유계가 아니면(저장고가 고체에 대해 과포화) 평형이 없다고 보고합니다. 모든 이온이 휘발성인 공급물(예: 순수 NH4NO3)은 총량이 정해지지 않아 퇴화합니다.
**닫힌계** (`solve_closed(feed, gas_total, T, rh, n_air, P)`): 기체량 g_j > 0이 미지수, ln p_j = ln(P g_j/(n_air+Σg)) (이상기체 + 공기). (i) 수용액 없는 상태(고체+기체)를 쌍대(원소 퍼텐셜) 공간에서 활성집합 Newton으로 풀고 (y, u_A: V_Aᵀy = c_A, b − V_A u_A − G g(y) = 0), TPD로 수용액 안정성을 검사; (ii) 수용액이 안정하면 `_wet` 활성집합 Newton에 기체 열을 양의 변수(ln p 항 포함)로 넣어 풉니다. Φ가 볼록하므로 전역해입니다.
- 검증: 고체 NH4NO3 + 증기에서 p(NH3)p(HNO3) = Kp(T) (상대오차 1e-6); 완전 증발; (NH4)2SO4 + HNO3 + NH3 습윤계의 질량수지(1e-16)와 ln p 일치; NaCl + HNO3 → 염화물 소실(HCl(g)); 열린계 p 복원(1e-7).
- 한계: 이상기체·공기 외 불활성, 유기물 미결합 (CO2는 §3.6), 모든 데이터는 298 K에서 검증(HCl T 의존성 미검증), 열린계 퇴화 경우 위 참조. 닫힌계 건조 판정 비용은 TPD(1~5 s)가 지배합니다.

### 3.6 탄산염과 CO2
- **성분.** CO3⁻⁻는 전체 탄소 C_T, H⁺는 부호가 자유로운 양성자 과잉 P = [H⁺] − [OH⁻] + [HCO3⁻] + 2[CO2](+[HSO4⁻])로 둡니다. HCO3⁻, OH⁻는 `AqueousIons._speciate_carb`에서 내부 분배합니다 (K1, K2, Kw는 `carbonate.py`의 Fortran 적합식과 동일, γ 고정비 반복 + ln h에 대한 brentq). OH⁻ 고체(portlandite, brucite 등)는 `n_oh`로 −H⁺ 처리하고 ln K′ = ln K − n_oh ln Kw(T).
- **검증.** Fortran 이식 `solve_carbonate`와 비교: HCO3 상대차 ≲ 2e-6, H⁺ ≲ 6e-4, CO3 ≲ 7e-5.
- **근사.** 물 소비와 CO2(aq)의 몰분율 기여는 무시하므로 Gibbs–Duhem이 약 1e-3까지만 성립합니다. γ(CO2)는 `gamma_co2_mr`(염석).
- **고체 (PHREEQC pitzer.dat 298 K 등).** natron, nahcolite, trona, kalicinite, calcite, aragonite, magnesite, nesquehonite, dolomite, gaylussite, pirssonite, burkeite, portlandite, brucite. thermonatrite, Na2CO3, NaOH는 기억한 NBS 값(품질 C, 원문 재확인 안 함). Na2CO3 건조 순서: natron → thermonatrite → 무수물. trona는 298 K 전용.
- **CO2(g).** CO2 + H2O ⇌ CO3⁻⁻ + 2H⁺ (`Gas.h = −1`), ln K = ln K_H(T)(PHREEQC 해석식) + ln K1 + ln K2. 열린계는 p 복원 정확, 닫힌계는 저장고 ≫ 입자일 때 Newton이 불량조건이라 p 고정 풀이의 고정점 반복(`_closed_fixed_point`)으로 대체합니다.
- **한계.** 탄산염 없는 계의 NaOH(음이온이 없음)는 미지원; 닫힌계 Newton은 입자/저장고 비가 매우 작으면 폴백을 사용합니다.

## 5. 검증 결과
- 테스트 `tests/test_sle.py` 52개(~13 s) + `tests/test_sle_acid.py` 17개(~45 s) 통과 (전체 스위트 274 passed; `tests/test_sle_gas.py` 8개): DB 전기중성·K 유한성, fitted==anchored(298 K), 단일염 DRH(298 K) 문헌 ±1.6 %p, 피팅 용해도 vs 핸드북 ≤ 4 %, Na2SO4 수화물 전이(≈305.5 K 재현), NaCl–KCl 상 순서·MDRH(0.715–0.74), KKT/질량수지/a_w=RH, active-set Gibbs ≤ SLSQP, RH 스캔 단조성, ERH 일관성, 비중성 입력 거부; 산성계: HSO4⁻ 속도론 일치(1e-7), KKT·질량수지·a_w=RH, 순수 H2SO4 wt%, 상 순서.
- 무작위 교차검증(SLSQP 전역 Gibbs 대조): 0/40(seed 42), 0/60(seed 7) 불일치.
- 복분해 반응 ΔG: NBS 대비 ≈ 0.2 kJ 이내.
- DRH(T): NH4NO3 73.7 %(273 K) → 61.2 %(298 K) → 54.0 %(313 K), NaCl 거의 평탄 — 문헌 경향과 일치.
- NaCl+(NH4)2SO4 RH 스캔: thenardite 66 %, NH4Cl+thenardite 64 %, ≤62–63 % 건조. MDRH에서 물량 불연속은 물리적.
- NaCl+KCl MDRH 72.9 % / FDRH 78.7 %는 문헌 검증 전(벤치마크 대상: "A database for deliquescence and efflorescence relative humidities of compounds with atmospheric relevance", 2021).

그림: `docs/figs/sle_drh_T.png`, `sle_scan_nacl_as.png`, `sle_solubility.png`, `sle_phase_nacl_kcl.png`.

## 6. 한계
- 기체상 없음. 산성계는 NH4HSO4·letovicite까지만 (NaHSO4 계열·H2SO4 수화물 미포함). JOTA-1 산성 예제는 정성 재현 수준이며 정량 비교는 미완.
- 피팅은 0–60 °C 유효, 고온·저온(<273 K)은 외삽.
- 품질 C 고체(MgCl2 저수화물, Mg(NO3)2, antarcticite, Ca(NO3)2)는 정량성 낮음.
- 피팅 ΔH_eff는 AIOMFAC γ(T) 오차를 흡수 — 포화선 밖 T-외삽 불확실.
- 복염(letovicite, (NH4)3H(SO4)2, NH4HSO4) 및 NH4NO3 고체상 전이 미포함.
- 건조 상태 TPD가 다이온 공급에서 최대 ~10 s (일반 0.02–0.5 s).
- `anchor_m`, “recall” 용해도(mirabilite, epsomite), 품질 B/C 항목, NBS 열화학 값은 **출판 전 원문 대조 필요**.
- 단일염 RH≠DRH 퇴화는 LP+TPD로 우회(혼합 자유도 s ≤ N−2 필요).

## 7. 로드맵
1. 산성계 확장(진행): NH4/Na 산성 고체는 Clegg 1998 K로 반영 완료. 남은 것: KHSO4, H2SO4 수화물, 복염, 문헌 상도(Tang 1980, Clegg 1998)·JOTA-1 예제 정량 비교, 이중염 화학양론 퇴화 대응.
2. NH3/HNO3/HCl 기체상 결합: 완료(3.5절). 남은 것: HCl 온도 의존성 원문 확인, NH4NO3 상전이·복염, 모든 이온이 휘발성일 때의 열린계 퇴화.
3. 해석적 야코비안(AIOMFAC 미분)·TPD 가속 → 속도·견고성 (산성계 건조 판정이 느림).
4. ERH 보정: 2021 DRH/ERH 데이터베이스로 ln S_crit 체계화.
5. 저온(<273 K) 데이터, 품질 C/B 고체 문헌값 확충.

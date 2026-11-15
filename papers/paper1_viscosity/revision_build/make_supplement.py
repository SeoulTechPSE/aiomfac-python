"""Write supplement.txt (Supplement of the paper_1 manuscript) with every number taken from the result files of
notebooks 08, 09 and 11. Run from paper_1/revision_build:  python make_supplement.py
Then build:  python build_docx.py supplement.txt ../aiomfac_soa_viscosity_supplement.docx ..
"""
import json

NB = "../notebooks/"
P = json.load(open(NB + "parameter_sensitivity_results.json"))
O = json.load(open(NB + "oof_tg_results.json"))
L = json.load(open(NB + "llps_song2012_results.json"))
f2 = lambda v: f"{v:.2f}"
f3 = lambda v: f"{v:.3f}"


def lit_rows(regime, n=6):
    return "\n".join(f"| {r['label']} | {f3(r['mean'])} | {f3(r['max'])} | {r['n_curves']} |"
                     for r in P["literature_summary"][regime][:n])


def soa_rows(n=3):
    rows = []
    for proxy in ("pinonic acid", "pinic acid", "2-methyltetrol", "DHOPA"):
        top = sorted([r for r in P["soa"] if r["proxy"] == proxy], key=lambda r: -r["mean_abs_dlog10"])[:n]
        rows.append(f"| {proxy} | " + "; ".join(f"{r['label']} ({f3(r['mean_abs_dlog10'])})" for r in top) + " |")
    return "\n".join(rows)


def oof_rows():
    rows = []
    for name, c in sorted(O["compounds"].items(), key=lambda kv: (kv[1]["measured"] is None, kv[0])):
        oof = sum(c["oof"]) / len(c["oof"])
        sd = (sum((v - oof) ** 2 for v in c["oof"]) / (len(c["oof"]) - 1)) ** 0.5
        meas = f"{c['measured']:.1f}" if c["measured"] else "–"
        status = "in BIMOG" if c["measured"] else "not in BIMOG"
        oof_txt = f"{oof:.1f} ± {sd:.1f}" if c["measured"] else "= published"
        cl = sum(c["oof_cluster"]) / len(c["oof_cluster"])
        cl_txt = f"{cl:.1f}" if c["measured"] else "= published"
        rows.append(f"| {name.replace('_', ' ')} | {status} | {meas} | {c['ml']:.1f} | {oof_txt} | {cl_txt} |")
    return "\n".join(rows)


def pm(v, d):
    """Signed number with a true minus sign (U+2212)."""
    return f"{v:+.{d}f}".replace("-", "\u2212")


lit = O["literature"]
mc = lit["monte_carlo"]
mcc = lit["monte_carlo_cluster"]
bcl = O["cluster_split"]["bias_by_class"]
cls = [c for c in L["classification"] if c["predicted"] is not None]
n_ok = sum(c["predicted"] == c["observed"] for c in cls)
def onset_txt(c):
    return "–" if c["onset_max"] is None else f"{c['onset_max'] * 100:.1f}"


cls_rows = "\n".join(
    f"| {c['compound']} | {'yes' if c['observed'] else 'no'} | {'yes' if c['predicted'] else 'no'} | {onset_txt(c)} |"
    for c in cls)
cmp = L["comparison"]


def dstat(sysname, key):
    d = [c["ours"] - c[key] for c in cmp if c["system"] == sysname and c["ours"] is not None and c[key] is not None]
    return sum(d) / len(d), max(abs(v) for v in d), len(d)


all_d = [c["ours"] - c["song_model"] for c in cmp if c["ours"] is not None and c["song_model"] is not None]
c5, c6, c7 = dstat("C5", "song_model"), dstat("C6", "song_model"), dstat("C7", "song_model")
m6, m7 = dstat("C6", "measured"), dstat("C7", "measured")
_c5 = {c["mfd"]: c["song_model"] - c["ours"] for c in cmp if c["system"] == "C5" and c["ours"] is not None and c["song_model"] is not None}
c5_hi = f"{min(v for m, v in _c5.items() if m >= 0.6):.2f}–{max(v for m, v in _c5.items() if m >= 0.6):.2f}"
c5_05 = _c5[0.5]

TEXT = f"""TITLE: Supplement of: Evaluating a structure-based viscosity pipeline for organic aerosol – machine-learned glass transition temperatures and AIOMFAC-VISC – against laboratory, chamber and ambient measurements
AUTHOR: Kee-Youn Yoo¹
AFFIL: ¹Department of Chemical and Biomolecular Engineering, Seoul National University of Science and Technology (SeoulTech), Seoul 01811, Republic of Korea
AFFIL: Correspondence: Kee-Youn Yoo (kyyoo@seoultech.ac.kr)

This supplement gives the sensitivity of the pipeline to the AIOMFAC interaction parameters (Sect. S1), the details of the out-of-fold T_{{g}} predictions and of the Monte Carlo propagation of T_{{g}} errors (Sect. S2), and the comparison of the phase-equilibrium solver with the calculations and measurements of Song et al. (2012) (Sect. S3). Sections, tables and figures without "S" refer to the main paper. Every number is taken from the result files of notebooks 08 (Sect. S1), 09 (Sect. S2) and 11 (Sect. S3).

# S1 Sensitivity to the AIOMFAC interaction parameters

AIOMFAC does not give uncertainties for its group-interaction parameters a_{{mn}}, so we tested the sensitivity of the pipeline to them one at a time. Every directed main-group pair m → n present in a molecule was scaled by 0.9 and 1.1 in turn (all subgroup entries belonging to that pair), and the change of the predicted log_{{10}} η was recorded; the larger of the two changes is reported. For the 53 single-compound curves, the organic mass fraction is given, so a_{{mn}} changes only the residual term of the mixing model; for the chamber-SOA proxies, the water content follows from the water activity, so a_{{mn}} also changes the composition. Main groups are labelled by the subgroups they contain in these molecules.

For the single-compound curves (Table S1), the water → hydroxyl parameter dominates: a 10 % change shifts the predicted viscosity of the concentrated curves by {f3(P['literature_summary']['concentrated'][0]['mean'])} log_{{10}} units on average and by at most {f2(P['literature_summary']['concentrated'][0]['max'])}, and that of the dilute curves by {f3(P['literature_summary']['dilute'][0]['mean'])} on average. Every other parameter changes the predictions by less than 0.01 on average. For the SOA proxies (Table S2), the parameters between water and the carboxyl, hydroxyl, alkyl and ketone groups change the predictions by 0.06–0.12 log_{{10}} units. All these changes are small compared with the model–measurement differences of the main paper (0.2–2 log_{{10}} units).

TABLE: Table S1. Mean and maximum over curves of the change of the predicted log_{{10}} η (mean over the points of a curve) for a ±10 % change of one interaction parameter, for the six most influential directed main-group pairs; dilute curves (top) and concentrated curves (bottom).
| Pair m → n | Mean | Max | Curves |
| Dilute curves | | | |
{lit_rows('dilute')}
| Concentrated curves | | | |
{lit_rows('concentrated')}

TABLE: Table S2. Chamber-SOA proxies: the three most influential directed main-group pairs and the mean change of the predicted log_{{10}} η over the RH of the data sets for a ±10 % change of the parameter.
| Proxy | Pair m → n (mean change) |
{soa_rows()}

# S2 Out-of-fold T_{{g}} predictions and Monte Carlo propagation

The SMILES-mode T_{{g}} predictor of Armeli Iapichino et al. (2023) without melting temperature is an ensemble of extremely randomized trees (100 trees, maximum depth 9, all 208 RDKit descriptors as candidate features at each split, no bootstrap). We refitted it with these hyperparameters in ten-fold cross-validation on the 368 BIMOG compounds with measured T_{{g}}, repeated with five random fold assignments (seeds 0–4, also used as the random state of the trees). Compounds with identical descriptor vectors, such as stereoisomers, would have been assigned to one fold; BIMOG contains none. Refitted on all 368 compounds, the model reproduces the published one to a mean absolute difference of 4.0 K (in-sample MAE 2.9 K versus 5.1 K for the published model, which was trained on a slightly different set). Out of fold, the MAE over BIMOG is {O['oof_mae_bimog']['mean']:.2f} ± {O['oof_mae_bimog']['sd']:.2f} K over the five repeats (bias {pm(O['oof_bias_bimog'], 2)} K; 90 % of the absolute errors are below {O['oof_abs_p90']:.1f} K), close to the nested cross-validation MAE of 15.1 K reported by Armeli Iapichino et al. (2023).

Table S3 lists the evaluated compounds. Eleven of the 16 compounds of the single-compound curves are in BIMOG; for them, the out-of-fold prediction replaces the in-sample one. The other five are not in BIMOG, so the published model is already out of sample for them.

TABLE: Table S3. Measured, in-sample and out-of-fold T_{{g}} (K) of the compounds of the single-compound curves: random split (mean ± standard deviation over five repeats) and cluster split (mean).
| Compound | Status | Measured | In-sample | Out-of-fold, random | Out-of-fold, cluster |
{oof_rows()}

In the Monte Carlo propagation, each of 300 draws assigns every compound a T_{{g}} equal to its measured value (or, for the five compounds without one, its predicted value) plus an error drawn at random from the pooled random-split out-of-fold residuals of all BIMOG compounds and repeats; all curves of one compound share the same error. The mean MAE over the curves is then computed for each draw. At the centre values (no error), it is {lit['monte_carlo_centre']['all']:.3f} for all curves, {lit['monte_carlo_centre']['dilute']:.3f} for the dilute and {lit['monte_carlo_centre']['concentrated']:.3f} for the concentrated curves. Over the draws, the medians (5–95 %) are {mc['all']['median']:.3f} ({mc['all']['p05']:.3f}–{mc['all']['p95']:.3f}), {mc['dilute']['median']:.3f} ({mc['dilute']['p05']:.3f}–{mc['dilute']['p95']:.3f}) and {mc['concentrated']['median']:.3f} ({mc['concentrated']['p05']:.3f}–{mc['concentrated']['p95']:.3f}). With the out-of-fold T_{{g}} itself, the mean MAE is {lit['oof']['all'][0]:.3f} ± {lit['oof']['all'][1]:.3f} (dilute {lit['oof']['dilute'][0]:.3f}, concentrated {lit['oof']['concentrated'][0]:.3f} ± {lit['oof']['concentrated'][1]:.3f}), against {lit['in_sample']['all']:.3f} with the in-sample T_{{g}}.

Split by molar mass, the concentrated curves respond very differently. For the 13 curves of compounds below 342 g mol^{{−1}}, the mean MAE is {lit['monte_carlo_centre']['conc_small']:.3f} at the centre and {mc['conc_small']['median']:.3f} ({mc['conc_small']['p05']:.3f}–{mc['conc_small']['p95']:.3f}) over the draws; for the seven curves of sucrose, maltose, trehalose and raffinose, it is {lit['monte_carlo_centre']['conc_large']:.3f} and {mc['conc_large']['median']:.3f} ({mc['conc_large']['p05']:.3f}–{mc['conc_large']['p95']:.3f}). The insensitivity of the large-molecule curves follows from the small combinatorial weight of η_{{0}} in the mixing model (Sect. 3.2 of the main paper).

As a stricter test of the predictor for new molecules, we also assigned whole clusters of structurally similar compounds to the folds. BIMOG was clustered with the Butina algorithm on Morgan fingerprints (radius 2, 2048 bits) with a Tanimoto distance cutoff of 0.6, giving {O['cluster_split']['n_clusters']} clusters (largest {O['cluster_split']['largest']} compounds; {O['cluster_split']['singletons']} singletons), which were distributed over the ten folds with balanced fold sizes. The out-of-fold MAE over BIMOG is then {O['cluster_split']['oof_mae_bimog']['mean']:.1f} ± {O['cluster_split']['oof_mae_bimog']['sd']:.1f} K, and for the 11 compounds of Table S3 it is {O['cluster_split']['mae_11_compounds']:.1f} K (bias {pm(O['cluster_split']['bias_11_compounds'], 1)} K). With these T_{{g}}, the mean MAE over all curves is {lit['oof_cluster']['all'][0]:.3f} ± {lit['oof_cluster']['all'][1]:.3f}, and that of the concentrated curves below 342 g mol^{{−1}} {lit['oof_cluster']['conc_small'][0]:.3f} ± {lit['oof_cluster']['conc_small'][1]:.3f} (random split: {lit['oof']['conc_small'][0]:.3f} ± {lit['oof']['conc_small'][1]:.3f}). Repeating the Monte Carlo propagation with errors drawn from the cluster-split residuals gives medians (5–95 %) of {mcc['all']['median']:.3f} ({mcc['all']['p05']:.3f}–{mcc['all']['p95']:.3f}) for all curves, {mcc['conc_small']['median']:.3f} ({mcc['conc_small']['p05']:.3f}–{mcc['conc_small']['p95']:.3f}) for the concentrated curves below 342 g mol^{{−1}} and {mcc['conc_large']['median']:.3f} ({mcc['conc_large']['p05']:.3f}–{mcc['conc_large']['p95']:.3f}) for the larger compounds (Table 1 of the main paper). The cluster-split residuals are biased: over BIMOG, the mean is {pm(O['cluster_split']['bias_bimog'], 1)} K, but the bias is concentrated in the {bcl['n_OH>=4 (sugars, polyols)']['n']} compounds with four or more hydroxyl groups (sugars and polyols; {pm(bcl['n_OH>=4 (sugars, polyols)']['bias'], 1)} K), whereas for the other {bcl['n_OH<4']['n']} compounds it is {pm(bcl['n_OH<4']['bias'], 1)} K; 10 % of the absolute residuals exceed {O['cluster_split']['abs_p90']:.0f} K.

# S3 Comparison of the phase-equilibrium solver with Song et al. (2012)

Song et al. (2012) measured the onset RH of liquid–liquid phase separation (LLPS) upon drying for particles of ammonium sulfate (AS) with equal-mass mixtures of three C5 (O:C 0.80), C6 (0.67) and C7 (0.57) dicarboxylic acids, and computed the same quantity with AIOMFAC and the global Gibbs-energy minimization of Zuend et al. (2010). We computed the onset RH, the highest RH at which two liquids form when crystallization is suppressed, with the combined solver of this study at 298.15 K for the three mixtures and for one acid of each (glutaric, 2-methylglutaric and 3,3-dimethylglutaric acid), on a 0.02 RH grid refined to 0.005. The literature values in Fig. S1 were read from Figs. 4 and 8 of Song et al. (2012) (about ±1 % RH).

The solver reproduces the calculation of Song et al. (2012) to within {sum(abs(v) for v in all_d) / len(all_d):.3f} on average: the mean difference is {pm(c6[0], 3)} for C6 (at most {c6[1]:.3f}) and {pm(c7[0], 3)} for C7 (at most {c7[1]:.3f}). For C5, the onset is {c5_hi} lower than their calculated value for mf_{{d}}(AS) ≥ 0.6 and {c5_05:.2f} lower at 0.5; at 0.4, where their coexistence curve ends near a water activity of 0.50, our solver finds no LLPS above 30 % RH. Like that calculation, the solver predicts a higher onset than measured, by {m6[0]:.2f} for C6 and {m7[0]:.2f} for C7 on average, and it predicts LLPS for C5, for which none was observed above the efflorescence RH of AS (39 %). The single acids give nearly the same onset as the mixtures (notebook 11).

!FIG figures_revised/figS1_song2012.png | Figure S1. Onset RH of liquid–liquid phase separation versus the dry mass fraction of AS, mf_{{d}}(AS), for the C5, C6 and C7 dicarboxylic-acid mixtures with AS at 298 K: this study (combined solver), the AIOMFAC calculation of Song et al. (2012, their Fig. 8) and the measured onset upon drying (their Fig. 4). Grey band: no LLPS observed above the efflorescence RH of the C5 system.

Table S4 compares predicted and observed LLPS for the 20 organic–AS systems compiled in Table 2 of Song et al. (2012), observed at 30 % < RH < 100 % and mf_{{d}}(AS) > 0.1. LLPS is predicted if two liquids form at RH ≥ 0.40, above the efflorescence RH of AS in these particles, for at least one of mf_{{d}}(AS) = 0.2, 0.5 and 0.8. Prediction and observation agree for {n_ok} of {len(cls)} systems. All 13 systems with observed LLPS (O:C 0.29–0.67) are predicted correctly, as are citric and malonic acid (O:C 1.17 and 1.33). The five disagreements are systems of O:C 0.75–1.00 for which AIOMFAC predicts LLPS that was not observed; for the C5 system, Song et al. (2012) noted the same overprediction.

TABLE: Table S4. Observed (Song et al., 2012, Table 2, and references therein) and predicted liquid–liquid phase separation of organic–AS particles, and the highest predicted onset RH (%) for mf_{{d}}(AS) = 0.2, 0.5 and 0.8.
| System | Observed LLPS | Predicted LLPS | Highest onset RH |
{cls_rows}

# References

REF: Armeli Iapichino, G., Peters, J.-H., and Koop, T.: Machine-learning-based prediction of the glass transition temperature of organic compounds using experimental data, ACS Omega, 8, 12298–12309, https://doi.org/10.1021/acsomega.2c08146, 2023.
REF: Song, M., Marcolli, C., Krieger, U. K., Zuend, A., and Peter, T.: Liquid-liquid phase separation and morphology of internally mixed dicarboxylic acids/ammonium sulfate/water particles, Atmos. Chem. Phys., 12, 2691–2712, https://doi.org/10.5194/acp-12-2691-2012, 2012.
REF: Zuend, A., Marcolli, C., Peter, T., and Seinfeld, J. H.: Computation of liquid-liquid equilibria and phase stabilities: implications for RH-dependent gas/particle partitioning of organic-inorganic aerosols, Atmos. Chem. Phys., 10, 7795–7820, https://doi.org/10.5194/acp-10-7795-2010, 2010.
"""
TEXT = TEXT.replace("(bias -", "(bias −").replace(" -> ", " → ").replace("mfd(AS)", "mf_{d}(AS)")
open("supplement.txt", "w").write(TEXT)
print("wrote supplement.txt;", TEXT.count("\n"), "lines")

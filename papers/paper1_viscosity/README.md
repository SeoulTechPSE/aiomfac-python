# paper_1 archive: code, notebooks, results and figures

Material for "Evaluating a structure-based viscosity pipeline for organic aerosol" (K.-Y. Yoo). All calculations use
`aiomfac_py` v1.3.1 (https://github.com/SeoulTechPSE/aiomfac-python), whose code is identical to commit 03a5ab2 of the
main branch; this folder is part of that release. The notebook outputs print the version string 1.3.0, which the
package carried at commit 03a5ab2 before the version number was raised for the release.

* `notebooks/aiomfac_validation_01..11*.ipynb` and `aiomfac_validation_03_ambient_pm25.ipynb`: the analyses, with
  their result files (`*.json`) and the Gervasi et al. (2020) Supplement (`gervasi_supp.zip`, CC BY 4.0).
* `revision_build/`: scripts that generate notebooks 04-11, the manuscript figures (`make_figs.py`) and the
  Supplement text (`make_supplement.py`).
* `figures_revised/`: the manuscript figures.
* `ambient_data/ambient_data.py`: loader for the ambient PM2.5 data.

Not included: the ambient PM2.5 viscosity point values (Do et al., 2026a; Ullah et al., 2026). They were provided by
the authors, who will publish them and asked that they not be deposited; they are available from the authors on
request (see `ambient_data/README.md`). Notebook 07 is therefore archived without outputs, and the measured values are
removed from `ambient_identified_samples_results.json`. To rerun notebooks 05-07, obtain the data file and set
`AMBIENT_XLSX` to its location.

Licences: the code in this folder is covered by the licence of the repository (GPL-3.0-or-later). The data files
taken from other sources keep their licences: `notebooks/gervasi_supp.zip` and the curves derived from it
(`literature_curves.json`) are from the Supplement of Gervasi et al. (2020, ACP, CC BY 4.0), and
`notebooks/bimog_unified.json` is compiled from BIMOG (Armeli Iapichino et al., 2023); cite those sources when using
them.

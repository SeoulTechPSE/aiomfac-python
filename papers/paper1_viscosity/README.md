# paper_1 archive: code, notebooks, results and figures

Material for "Evaluating a structure-based viscosity pipeline for organic aerosol" (K.-Y. Yoo). All calculations use
the code of `aiomfac_py` v1.3.2 (https://github.com/SeoulTechPSE/aiomfac-python), which is identical to v1.3.1 and
to commit 03a5ab2 of the main branch; this folder is part of release v1.3.2. The outputs of notebooks 01-11 print the
version string 1.3.0, which the package carried at commit 03a5ab2, and those of notebooks 12-14 print 1.3.1.

* `notebooks/aiomfac_validation_01..14*.ipynb` and `aiomfac_validation_03_ambient_pm25.ipynb`: the analyses, with
  their result files (`*.json`) and the Gervasi et al. (2020) Supplement (`gervasi_supp.zip`, CC BY 4.0).
* `revision_build/`: scripts that generate notebooks 04-14, the manuscript figures (`make_figs.py`, `make_fig7.py`) and the
  Supplement text (`make_supplement.py`).
* `figures_revised/`: the manuscript figures.
* `ambient_data/ambient_data.py`: loader for the ambient PM2.5 data.

Not included: the ambient PM2.5 viscosity point values (Do et al., 2026a; Ullah et al., 2026). They were provided by
the authors, who will publish them and asked that they not be deposited; they are available from the authors on
request (see `ambient_data/README.md`). Notebook 07 is therefore archived without outputs, and the measured values are
removed from `ambient_identified_samples_results.json`. To rerun notebooks 05-07 and 14, obtain the data file and set
`AMBIENT_XLSX` to its location.

Licences: the code in this folder is covered by the licence of the repository (GPL-3.0-or-later). The data files
taken from other sources keep their licences: `notebooks/gervasi_supp.zip` and the curves derived from it
(`literature_curves.json`) are from the Supplement of Gervasi et al. (2020, ACP, CC BY 4.0), and
`notebooks/bimog_unified.json` is compiled from BIMOG (Armeli Iapichino et al., 2023); cite those sources when using
them.

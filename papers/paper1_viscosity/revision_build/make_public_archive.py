"""Assemble the public archive of the paper_1 material (code, notebooks, results, figures) in ../archive_public.

The ambient PM2.5 point values of Do et al. (2026a, Fig. 2a) and Ullah et al. (2026) were provided by the authors, who
asked that neither the data file nor the point values be deposited (see ../ambient_data/README.md). This script
therefore
  * never copies ../ambient_data/private/,
  * copies notebook 07 (which tabulates the values sample by sample) with its outputs cleared,
  * removes the measured values from ambient_identified_samples_results.json (keeps model predictions, dates, RH, OIR
    and the summary statistics reported in the manuscript),
  * rounds the median measured viscosity per site in ambient_pointwise_results.json to two decimals (with an odd
    number of points the median is itself one measured value),
  * finally scans every archived text file for the measured values and stops if any is found.
Run from paper_1/revision_build:  python make_public_archive.py
"""
import json
import math
import os
import re
import shutil
import sys

import nbformat

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                                   # paper_1
OUT = os.path.join(ROOT, "archive_public")

NOTEBOOKS = ["aiomfac_validation_01_pure_compounds.ipynb", "aiomfac_validation_02_soa_proxies.ipynb",
             "aiomfac_validation_04_revision.ipynb", "aiomfac_validation_05_ambient_pointwise.ipynb",
             "aiomfac_validation_06_ambient_phase_state.ipynb",
             "aiomfac_validation_08_parameter_sensitivity.ipynb",
             "aiomfac_validation_09_oof_tg.ipynb",
             "aiomfac_validation_10_proxy_oc.ipynb",
             "aiomfac_validation_11_llps_song2012.ipynb"]
NB_CLEARED = ["aiomfac_validation_07_ambient_identified_samples.ipynb"]
NB_DATA = ["literature_curves.json", "bimog_unified.json", "gervasi_supp.zip", "validation_results.json",
           "soa_validation_results.json", "glassy_regime_diagnosis.json", "revision_results.json",
           "ambient_pointwise_results.json", "ambient_phase_state_results.json", "parameter_sensitivity_results.json", "oof_tg_results.json", "proxy_oc_results.json", "llps_song2012_results.json",
           "fig_A_predicted_vs_measured.png", "fig_B_example_curves.png", "fig_C_tg_validation.png",
           "fig_D_apinene_soa.png"]
BUILD = ["build_nb04.py", "build_nb05.py", "build_nb06.py", "build_nb07.py", "build_nb08.py", "build_nb09.py", "build_nb10.py", "build_nb11.py", "make_figs.py", "make_supplement.py",
         "make_public_archive.py", "subgroup_names.json"]
MEASURED_KEYS = {"log10_eta", "log10_lo", "log10_hi", "tau_mix_h", "censored"}


def copy(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)


def sanitize_nb07_results(src, dst):
    d = json.load(open(src))
    for key in ("points", "ansan_points"):
        d[key] = [{k: v for k, v in p.items() if k not in MEASURED_KEYS} for p in d[key]]
    d["check_sb"] = [{"site": c["site"], "si_date": c["si_date"], "data_date": c["data_date"], "rh": c["rh"],
                      "inverted_minus_provided": c["inverted"] - c["provided"]} for c in d["check_sb"]]
    d["note"] = ("Measured point values removed (provided by the authors of Do et al., 2026a, and Ullah et al., 2026, "
                 "who asked that they not be deposited). Model predictions, dates, RH, OIR and summary statistics kept.")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    json.dump(d, open(dst, "w"), indent=1)


def sanitize_nb05_results(src, dst):
    d = json.load(open(src))
    for v in d["tg_required"]:
        v["log10_eta_median"] = round(v["log10_eta_median"], 2)
    json.dump(d, open(dst, "w"), indent=1)


def measured_values():
    """All measured values (log10 Pa s, Pa s and mixing times in h) of the ambient data file."""
    sys.path.insert(0, os.path.join(ROOT, "ambient_data"))
    import ambient_data
    vals = []
    for p in ambient_data.load(levels=("sample mean", "particle", "group mean")):
        for k in ("log10_eta", "log10_lo", "log10_hi"):
            if p[k] is not None:
                vals += [p[k], 10.0 ** p[k]]
        if p.get("tau_mix_h"):
            vals.append(p["tau_mix_h"])
    return vals


NUMBER = re.compile(r"(?<![\w.])[-+]?\d+\.(\d+)(?:[eE][-+]?\d+)?")


def scan(values):
    """Numbers written with >= 4 decimals that equal a measured value rounded to the same number of digits."""
    hits = []
    for dirpath, _, files in os.walk(OUT):
        for f in files:
            if f.endswith((".png", ".zip")):
                continue
            path = os.path.join(dirpath, f)
            for m in NUMBER.finditer(open(path, encoding="utf-8", errors="ignore").read()):
                if len(m.group(1)) < 4:
                    continue
                x = float(m.group(0))
                if x == 0.0:
                    continue
                for v in values:
                    sig = len(m.group(1)) + len(str(int(abs(float(m.group(0).split("e")[0].split("E")[0])))))
                    if math.isclose(x, v, rel_tol=0.5 * 10.0 ** (1 - sig)):
                        hits.append((os.path.relpath(path, OUT), m.group(0)))
                        break
    return hits


def main():
    if os.path.exists(OUT):
        sys.exit(f"{OUT} exists; move it away first (this script does not delete anything)")
    nbdir = os.path.join(ROOT, "notebooks")
    for f in [f for f in NOTEBOOKS + NB_DATA if f != "ambient_pointwise_results.json"]:
        copy(os.path.join(nbdir, f), os.path.join(OUT, "notebooks", f))
    for f in NB_CLEARED:
        nb = nbformat.read(os.path.join(nbdir, f), as_version=4)
        for c in nb.cells:
            if c.cell_type == "code":
                c.outputs, c.execution_count = [], None
        os.makedirs(os.path.join(OUT, "notebooks"), exist_ok=True)
        nbformat.write(nb, os.path.join(OUT, "notebooks", f))
    sanitize_nb05_results(os.path.join(nbdir, "ambient_pointwise_results.json"),
                          os.path.join(OUT, "notebooks", "ambient_pointwise_results.json"))
    sanitize_nb07_results(os.path.join(nbdir, "ambient_identified_samples_results.json"),
                          os.path.join(OUT, "notebooks", "ambient_identified_samples_results.json"))
    copy(os.path.join(ROOT, "aiomfac_validation_03_ambient_pm25.ipynb"),
         os.path.join(OUT, "aiomfac_validation_03_ambient_pm25.ipynb"))
    for f in BUILD:
        copy(os.path.join(HERE, f), os.path.join(OUT, "revision_build", f))
    for f in os.listdir(os.path.join(ROOT, "figures_revised")):
        copy(os.path.join(ROOT, "figures_revised", f), os.path.join(OUT, "figures_revised", f))
    for f in ("ambient_data.py", "README.md"):
        copy(os.path.join(ROOT, "ambient_data", f), os.path.join(OUT, "ambient_data", f))
    with open(os.path.join(OUT, "README.md"), "w") as fh:
        fh.write(README)

    hits = scan(measured_values())
    if hits:
        for h in hits[:20]:
            print("MEASURED VALUE FOUND:", h)
        sys.exit("archive contains measured ambient values; fix before publishing")
    n = sum(len(fs) for _, _, fs in os.walk(OUT))
    print(f"wrote {OUT} ({n} files); no measured ambient point values found")


README = """# paper_1 archive: code, notebooks, results and figures

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
"""

if __name__ == "__main__":
    main()

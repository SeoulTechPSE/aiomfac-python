# Ambient PM2.5 viscosity data (Do et al., 2026a; Ullah et al., 2026)

The point values behind Fig. 2a of Do et al. (2026a) were provided by the authors (M. Song, Jeonbuk National University,
e-mail of 2026-10-06), including the dates and OIR of all Ansan points and the per-particle values of Seoul and
Beijing.

* `ambient_data.py`: the loader used by notebooks 05–07. It is archived with the paper.
* `private/`: the data file as received. It is **not** archived or redistributed: the authors will publish the data
  themselves and asked that neither the file nor the Fig. 2a point values be deposited (reply of 2026-10-06 to
  `../submission_prep/song_group_zenodo_permission.md`). Readers can obtain the values from the authors. Set
  `AMBIENT_XLSX` to the file's location to rerun the notebooks.

Notes confirmed by the authors (2026-10-06):
* The four Seoul/Beijing points marked "to be confirmed" are lower bounds (>= 1e8 Pa s, no flow within 3 h); their
  sample dates are not identified.
* The Seoul measurement at RH 32.8 % belongs to the sample of 10/15 (OIR 2.09). "10/12" in Ullah et al. (2026),
  Supplement Table S1, is a typographical error.
* Values are used with all decimals as provided; tables in the papers show rounded values.

Constants of the authors' viscosity–mixing-time conversion (their Eqs. S1–S4) used for Ansan, as stated by the
authors: R_matrix = 0.44 nm (ξ = 0.514; the SI states 0.4 nm) and η_w(300 K) = 0.840 mPa s (Maclean et al., 2021).
With these values, the tabulated mixing times reproduce the Ansan viscosities to 0.001 log10.

Public archive: `revision_build/make_public_archive.py` assembles `archive_public/` (code, notebooks, results, figures)
without `private/`, with notebook 07's outputs cleared and the measured values removed from its result file, and the
per-site median measured viscosity rounded to two decimals. It stops if any measured value (at four or more decimals)
is found in the archived files.

"""Machine-learning glass-transition-temperature (Tg) predictor of Armeli, Peters, and Koop (2023).

Ports (in the sense described in ``PROVENANCE.md`` -- this loads the authors' own trained model, it does not
reimplement it):

    Armeli, G., Peters, J.-H., and Koop, T. (2023), "Machine-Learning-Based Prediction of the Glass Transition
    Temperature of Organic Compounds Using Experimental Data", ACS Omega, 8, 12298-12309,
    doi:10.1021/acsomega.2c08146. [cited below as "A2023"]

This supersedes the "not implemented, no closed-form equation to port" scope note this package previously
carried for A2023 (see ``aiomfac_py.viscosity``'s module docstring and ``aiomfac_py.viscosity.
predict_tg_derieux2018``, the older, less accurate, but dependency-free alternative that module still
implements): the user supplied the trained model files themselves (from the paper's own Zenodo code deposit,
DOI 10.5281/zenodo.7650576; see ``PROVENANCE.md`` next to this file for exact hashes and license caveats), so
there is something concrete to load here after all.

What this provides
-------------------
* :func:`predict_tg_ml_fg` -- "Functional Group Mode": Tg from a hand-countable set of functional groups
  (numbers of CH3/CH2/CH/C/OH/ether-O/carbonyl-O groups, double bond equivalents, optionally N and halogen
  atom counts) plus atomic O:C ratio and molar mass -- and, optionally, melting temperature Tm, which A2023
  found improves accuracy. Needs only ``scikit-learn`` (the ``tgml`` extra); no cheminformatics toolkit.
* :func:`predict_tg_ml_smiles` -- "SMILES Mode": Tg from a SMILES string alone (plus optional Tm), which
  A2023 report as their most accurate variant (MAE ~12 K vs. ~13 K for Functional Group Mode). Needs
  ``rdkit`` as well (the ``tgml-smiles`` extra) -- see the version caveat below, which is serious enough that
  this function actively refuses to run rather than risk a silently wrong answer.

Both route to one of A2023's six trained ``ExtraTreesRegressor`` models (``data/*.pkl``: Functional Group
Mode for CHO-only vs. N/halogen-containing compounds, SMILES Mode, each with/without Tm as an input feature),
loaded with ``pickle.load`` and cached in memory after the first call.

Why this is kept separate from the rest of ``aiomfac_py``, and the dependency story
--------------------------------------------------------------------------------------
This subpackage is **never imported by aiomfac_py's own top-level ``__init__.py``**; importing it raises
``ImportError`` with a clear message if its dependencies aren't installed, exactly like ``aiomfac_py.s2as``.
It needs ``scikit-learn`` (the ``tgml`` extra) always, and ``rdkit`` as well (the ``tgml-smiles`` extra) for
:func:`predict_tg_ml_smiles` specifically -- neither is a narrow version pin, and both install straight
alongside the rest of ``aiomfac_py`` (not a separate environment). Two things worth knowing about why that's
true despite the vendored model files' own history:

* **The trained models were originally pickled with a very old scikit-learn** (0.24.1-era tree internals;
  the vendored ``vendor/requirements_console_script.txt`` pins ``scikit-learn==1.1.1``), whose binary tree
  format scikit-learn >= 1.3 cannot load at all (its node format gained a ``missing_go_to_left`` field).
  ``data/*.pkl`` as shipped here have already been migrated past that: ``tools/migrate_tgml_pickles.py``
  extracts each tree's fitted state (a plain NumPy structured array, no scikit-learn version tied to it) and
  rebuilds it under a modern scikit-learn, confirmed to reproduce the original models' predictions
  bit-for-bit (``numpy.array_equal``) across 200 random feature vectors per model -- not an approximation or
  a retrain. If you ever obtain the *original*, unmigrated pickle files again (e.g. a fresh download from the
  paper's Zenodo deposit) and want to re-run that migration, that script documents exactly how; it needs the
  old scikit-learn/numpy combination only for that one offline step, never for ordinary use of this module.
* **SMILES Mode's feature vector is 208 named RDKit molecular descriptors**, computed by looking each one up
  by name in whatever RDKit build is installed (``_RDKIT_DESCRIPTOR_NAMES`` below -- the exact list and order
  A2023's own training used, from ``rdkit-pypi==2022.3.2.1``), not by positional order in
  ``Descriptors.descList``. This matters because newer RDKit releases add descriptors (confirmed: 217 in a
  current build, not 208), which would silently shift every later entry if read positionally; reading by name
  avoids that failure mode entirely, and was confirmed (by direct comparison against the pinned old RDKit
  version) to reproduce identical values for every one of the 208 descriptors on simple test molecules.
  **Caveat**: RDKit occasionally refines an individual descriptor's own definition across releases -- found
  directly here for ``NumHAcceptors`` on one more complex test molecule (a purine-like structure), where a
  current RDKit build returns 6 instead of the old build's 7. This is a small, isolated accuracy risk (most
  descriptors were confirmed identical; A2023's own reported ~12 K MAE already dwarfs it for any single
  descriptor), not the all-or-nothing corruption a positional-order mismatch would cause. Install
  ``rdkit-pypi==2022.3.2.1`` specifically (Python <= 3.10 only) instead of a newer ``rdkit`` if you need
  byte-for-byte reproduction of A2023's own reported numbers.
* **Loading a pickle file executes arbitrary code if the file is tampered with** (this is scikit-learn's own
  documented caveat, repeated as a ``UserWarning`` on every load here) -- only use ``data/*.pkl`` as vendored
  in this subpackage (see ``PROVENANCE.md`` for their SHA-256 hashes, of the *original* files the migration
  started from) or files you otherwise trust completely.

For a dependency-free (if somewhat less accurate) alternative that needs no extra install at all, see
``aiomfac_py.viscosity.predict_tg_derieux2018``.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources

_PICKLE_NAMES = ("fg_cho", "fg_cho_no_tm", "fg_nhal", "fg_nhal_no_tm", "sm", "sm_no_tm")

#: The 208 RDKit descriptor names, in the order A2023's own training used (rdkit.Chem.Descriptors.descList
#: from rdkit-pypi==2022.3.2.1). predict_tg_ml_smiles looks these up BY NAME in whatever RDKit is installed,
#: not positionally -- see the module docstring's "SMILES Mode" bullet for why.
_RDKIT_DESCRIPTOR_NAMES = (
    "MaxEStateIndex", "MinEStateIndex", "MaxAbsEStateIndex", "MinAbsEStateIndex", "qed", "MolWt",
    "HeavyAtomMolWt", "ExactMolWt", "NumValenceElectrons", "NumRadicalElectrons", "MaxPartialCharge",
    "MinPartialCharge", "MaxAbsPartialCharge", "MinAbsPartialCharge", "FpDensityMorgan1", "FpDensityMorgan2",
    "FpDensityMorgan3", "BCUT2D_MWHI", "BCUT2D_MWLOW", "BCUT2D_CHGHI", "BCUT2D_CHGLO", "BCUT2D_LOGPHI",
    "BCUT2D_LOGPLOW", "BCUT2D_MRHI", "BCUT2D_MRLOW", "BalabanJ", "BertzCT", "Chi0", "Chi0n", "Chi0v", "Chi1",
    "Chi1n", "Chi1v", "Chi2n", "Chi2v", "Chi3n", "Chi3v", "Chi4n", "Chi4v", "HallKierAlpha", "Ipc", "Kappa1",
    "Kappa2", "Kappa3", "LabuteASA", "PEOE_VSA1", "PEOE_VSA10", "PEOE_VSA11", "PEOE_VSA12", "PEOE_VSA13",
    "PEOE_VSA14", "PEOE_VSA2", "PEOE_VSA3", "PEOE_VSA4", "PEOE_VSA5", "PEOE_VSA6", "PEOE_VSA7", "PEOE_VSA8",
    "PEOE_VSA9", "SMR_VSA1", "SMR_VSA10", "SMR_VSA2", "SMR_VSA3", "SMR_VSA4", "SMR_VSA5", "SMR_VSA6",
    "SMR_VSA7", "SMR_VSA8", "SMR_VSA9", "SlogP_VSA1", "SlogP_VSA10", "SlogP_VSA11", "SlogP_VSA12",
    "SlogP_VSA2", "SlogP_VSA3", "SlogP_VSA4", "SlogP_VSA5", "SlogP_VSA6", "SlogP_VSA7", "SlogP_VSA8",
    "SlogP_VSA9", "TPSA", "EState_VSA1", "EState_VSA10", "EState_VSA11", "EState_VSA2", "EState_VSA3",
    "EState_VSA4", "EState_VSA5", "EState_VSA6", "EState_VSA7", "EState_VSA8", "EState_VSA9", "VSA_EState1",
    "VSA_EState10", "VSA_EState2", "VSA_EState3", "VSA_EState4", "VSA_EState5", "VSA_EState6", "VSA_EState7",
    "VSA_EState8", "VSA_EState9", "FractionCSP3", "HeavyAtomCount", "NHOHCount", "NOCount",
    "NumAliphaticCarbocycles", "NumAliphaticHeterocycles", "NumAliphaticRings", "NumAromaticCarbocycles",
    "NumAromaticHeterocycles", "NumAromaticRings", "NumHAcceptors", "NumHDonors", "NumHeteroatoms",
    "NumRotatableBonds", "NumSaturatedCarbocycles", "NumSaturatedHeterocycles", "NumSaturatedRings",
    "RingCount", "MolLogP", "MolMR", "fr_Al_COO", "fr_Al_OH", "fr_Al_OH_noTert", "fr_ArN", "fr_Ar_COO",
    "fr_Ar_N", "fr_Ar_NH", "fr_Ar_OH", "fr_COO", "fr_COO2", "fr_C_O", "fr_C_O_noCOO", "fr_C_S", "fr_HOCCN",
    "fr_Imine", "fr_NH0", "fr_NH1", "fr_NH2", "fr_N_O", "fr_Ndealkylation1", "fr_Ndealkylation2", "fr_Nhpyrrole",
    "fr_SH", "fr_aldehyde", "fr_alkyl_carbamate", "fr_alkyl_halide", "fr_allylic_oxid", "fr_amide",
    "fr_amidine", "fr_aniline", "fr_aryl_methyl", "fr_azide", "fr_azo", "fr_barbitur", "fr_benzene",
    "fr_benzodiazepine", "fr_bicyclic", "fr_diazo", "fr_dihydropyridine", "fr_epoxide", "fr_ester", "fr_ether",
    "fr_furan", "fr_guanido", "fr_halogen", "fr_hdrzine", "fr_hdrzone", "fr_imidazole", "fr_imide",
    "fr_isocyan", "fr_isothiocyan", "fr_ketone", "fr_ketone_Topliss", "fr_lactam", "fr_lactone",
    "fr_methoxy", "fr_morpholine", "fr_nitrile", "fr_nitro", "fr_nitro_arom", "fr_nitro_arom_nonortho",
    "fr_nitroso", "fr_oxazole", "fr_oxime", "fr_para_hydroxylation", "fr_phenol", "fr_phenol_noOrthoHbond",
    "fr_phos_acid", "fr_phos_ester", "fr_piperdine", "fr_piperzine", "fr_priamide", "fr_prisulfonamd",
    "fr_pyridine", "fr_quatN", "fr_sulfide", "fr_sulfonamd", "fr_sulfone", "fr_term_acetylene",
    "fr_tetrazole", "fr_thiazole", "fr_thiocyan", "fr_thiophene", "fr_unbrch_alkane", "fr_urea",
)
assert len(_RDKIT_DESCRIPTOR_NAMES) == 208


@lru_cache(maxsize=None)
def _load_model(name: str):
    if name not in _PICKLE_NAMES:
        raise ValueError(f"unknown TgML model {name!r}, expected one of {_PICKLE_NAMES}")
    try:
        import sklearn  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "aiomfac_py.tgml_armeli requires scikit-learn -- pip install 'aiomfac_py[tgml]' (and, for "
            "predict_tg_ml_smiles, 'aiomfac_py[tgml-smiles]' for rdkit as well)."
        ) from e
    raw = resources.files("aiomfac_py.tgml_armeli").joinpath(f"data/{name}.pkl").read_bytes()
    return pickle.loads(raw)


def _rd_descriptors(smiles: str):
    """The 208 named RDKit descriptors _RDKIT_DESCRIPTOR_NAMES lists, for one SMILES string, computed
    directly (no deepchem/tensorflow dependency -- see PROVENANCE.md for why this is equivalent to the
    original script) and looked up BY NAME (see the module docstring's "SMILES Mode" bullet for why)."""
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors
    except ImportError as e:
        raise ImportError(
            "predict_tg_ml_smiles requires rdkit -- pip install 'aiomfac_py[tgml-smiles]'. Functional Group "
            "Mode (predict_tg_ml_fg) does not need rdkit at all."
        ) from e

    by_name = dict(Descriptors.descList)
    missing = [n for n in _RDKIT_DESCRIPTOR_NAMES if n not in by_name]
    if missing:
        raise RuntimeError(
            f"this RDKit build is missing {len(missing)} of the 208 named descriptors SMILES Mode needs "
            f"(first few: {missing[:5]}) -- this RDKit release may be too old, or these descriptors were "
            "renamed/removed upstream. Use predict_tg_ml_fg (Functional Group Mode) or "
            "aiomfac_py.viscosity.predict_tg_derieux2018 instead, or install an RDKit version that has them."
        )

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"rdkit could not parse SMILES {smiles!r}")
    values = []
    for desc_name in _RDKIT_DESCRIPTOR_NAMES:
        func = by_name[desc_name]
        values.append(func(mol, avg=True) if desc_name == "Ipc" else func(mol))
    return values


@dataclass
class TgMLResult:
    tg_K: float            # predicted glass transition temperature, K
    model_name: str        # which of the 6 trained models produced this (data/<model_name>.pkl)


def predict_tg_ml_fg(n_ch3: float, n_ch2: float, n_ch: float, n_c: float, n_oh: float, n_ether_o: float,
                      n_carbonyl_o: float, dbe: float, o_to_c: float, molar_mass_g_mol: float, *,
                      n_N: float = 0.0, n_hal: float = 0.0, Tm_K: float | None = None) -> TgMLResult:
    """Predict Tg (K) from functional-group counts: A2023's "Functional Group Mode" (their Table 1).

    ``n_ch3``/``n_ch2``/``n_ch``/``n_c`` are the numbers of methyl/methylene/methine groups and of carbon
    atoms with no attached hydrogen; ``n_oh`` the number of hydroxyl groups; ``n_ether_o`` the number of
    ether oxygens (``-O-``); ``n_carbonyl_o`` the number of double-bonded carbonyl oxygens (``=O``); ``dbe``
    the double bond equivalent (degree of unsaturation); ``o_to_c``/``molar_mass_g_mol`` the molecule's
    atomic O:C ratio and molar mass (A2023's "indirect" features, computed from the full formula, not just
    the functional groups above). ``n_N``/``n_hal`` (numbers of nitrogen and halogen atoms) are optional and
    route to A2023's separate "NHal" model (fit on CHO compounds plus N/halogen-containing ones) when
    nonzero; by default (both zero) the CHO-only model is used.

    ``Tm_K`` (melting temperature) is optional but, per A2023 (their Table 3), improves accuracy noticeably
    (CHO-set MAE 13.1 K with Tm vs. 16.4 K without, by nested cross-validation) when it's known.

    Needs ``scikit-learn`` installed (the ``tgml`` extra: ``pip install 'aiomfac_py[tgml]'``) -- no special
    environment or version pin, see the module docstring for why. ``PROVENANCE.md`` documents exactly which
    trained model file each combination of arguments routes to.
    """
    has_n_hal = n_N != 0.0 or n_hal != 0.0
    feat = [n_ch3, n_ch2, n_ch, n_c, n_oh, n_ether_o, n_carbonyl_o, dbe]
    if has_n_hal:
        feat += [n_N, n_hal]
    feat += [o_to_c, molar_mass_g_mol]
    model_name = "fg_nhal" if has_n_hal else "fg_cho"
    if Tm_K is not None:
        feat.append(Tm_K)
    else:
        model_name += "_no_tm"
    model = _load_model(model_name)
    tg = float(model.predict([feat])[0])
    return TgMLResult(tg_K=tg, model_name=model_name)


def predict_tg_ml_smiles(smiles: str, Tm_K: float | None = None) -> TgMLResult:
    """Predict Tg (K) from a SMILES string alone: A2023's "SMILES Mode" (their Sect. 3.3), their most
    accurate variant (reported MAE ~11.7-12.9 K with Tm, ~15.1 K without, vs. ~13.0-13.1 K/16.4 K for
    Functional Group Mode).

    ``Tm_K`` (melting temperature) is optional but, as for Functional Group Mode, improves accuracy per
    A2023's own nested cross-validation.

    Needs ``rdkit`` installed as well (the ``tgml-smiles`` extra: ``pip install 'aiomfac_py[tgml-smiles]'``).
    The 208-element descriptor vector A2023's SMILES-Mode models were trained on is looked up by name in
    whatever RDKit build is installed, which works with any reasonably recent RDKit (this function raises a
    clear ``RuntimeError`` rather than silently mispredicting if a build is ever missing one of the named
    descriptors) -- see this subpackage's module docstring ("SMILES Mode" bullet) for the mechanism and for a
    known, small, isolated numeric caveat (``NumHAcceptors`` on some structures) if you need byte-exact
    reproduction of A2023's own reported numbers.
    """
    desc = _rd_descriptors(smiles)
    model_name = "sm" if Tm_K is not None else "sm_no_tm"
    feat = ([Tm_K] if Tm_K is not None else []) + desc
    model = _load_model(model_name)
    tg = float(model.predict([feat])[0])
    return TgMLResult(tg_K=tg, model_name=model_name)

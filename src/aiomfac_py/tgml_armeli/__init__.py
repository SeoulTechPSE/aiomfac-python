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

Why this is kept strictly separate from the rest of ``aiomfac_py`` -- please read before installing
------------------------------------------------------------------------------------------------------
This subpackage is **never imported by aiomfac_py's own top-level ``__init__.py``**; importing it raises
``ImportError`` with a clear message if its dependencies aren't installed, exactly like ``aiomfac_py.s2as``.
Unlike that module, though, the version constraints here are unusually tight, for reasons worth knowing
before reaching for ``pip install aiomfac_py[tgml]``:

* **The trained models were pickled with a very old scikit-learn** (0.24.1-era internals; the original
  ``vendor/requirements_console_script.txt`` pins ``scikit-learn==1.1.1``). scikit-learn's own tree node
  binary format changed in 1.3 (it gained a ``missing_go_to_left`` field for missing-value support), so
  **scikit-learn >= 1.3 cannot load these files at all** (hard ``ValueError``, not a silent issue) --
  confirmed directly here against scikit-learn 1.1.1 through 1.4.2. This module therefore requires
  ``scikit-learn>=1.1,<1.3``.
* **Those scikit-learn wheels (<1.3) were themselves compiled against NumPy's pre-2.0 C ABI.** Importing
  ``sklearn`` under NumPy >= 2 with such a wheel fails immediately (``ValueError: numpy.dtype size changed``)
  -- also confirmed directly here. So this also needs **NumPy < 2**.
* **Installing NumPy < 2 into the same environment as the rest of this project is a real regression, not a
  theoretical one**: doing so in this development environment to test the above caused one existing
  ``aiomfac_py.carbonate`` test to fail outright (a divide-by-zero that NumPy >= 2's build did not trigger).
  **Install the ``tgml``/``tgml-smiles`` extras in their own, separate virtual environment** -- never into
  the same environment you run the rest of ``aiomfac_py`` in -- and call this subpackage's functions from
  there (or via a subprocess/IPC boundary) rather than importing it alongside the rest of the package.
* **SMILES Mode's accuracy depends on reproducing an exact, old RDKit version.** A2023's own
  ``vendor/requirements_console_script.txt`` pins ``rdkit-pypi==2022.3.2.1``, which gives exactly 208
  molecular descriptors from ``rdkit.Chem.Descriptors.descList`` -- the feature-vector length the SMILES-Mode
  models were trained on. RDKit adds descriptors across releases (a current RDKit build gives 217, not 208,
  confirmed directly here); feeding a mismatched-length or wrongly-ordered descriptor vector into the model
  would not raise an error, it would just silently return a wrong Tg. ``predict_tg_ml_smiles`` therefore
  checks ``len(Descriptors.descList) == 208`` before every call and raises ``RuntimeError`` rather than
  guessing -- which in practice means SMILES Mode only works with ``rdkit-pypi==2022.3.2.1`` installed, which
  in turn only has wheels for **Python <= 3.10** (confirmed: no ``cp311`` wheel exists on PyPI). Functional
  Group Mode has no such dependency and works on any Python version the ``scikit-learn<1.3`` constraint
  allows.
* **Loading a pickle file executes arbitrary code if the file is tampered with** (this is scikit-learn's own
  documented caveat, repeated as a ``UserWarning`` on every load here) -- only use ``data/*.pkl`` as vendored
  in this subpackage (see ``PROVENANCE.md`` for their SHA-256 hashes) or files you otherwise trust completely.

In short: prefer :func:`predict_tg_ml_fg` in its own virtual environment (``scikit-learn>=1.1,<1.3,
numpy<2``); reach for :func:`predict_tg_ml_smiles` only if that environment is also Python <= 3.10 with
``rdkit-pypi==2022.3.2.1`` installed. For a dependency-free (if somewhat less accurate) alternative that
works directly alongside the rest of ``aiomfac_py``, see ``aiomfac_py.viscosity.predict_tg_derieux2018``.
"""
from __future__ import annotations

import math
import pickle
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources

#: Expected length of RDKit's Descriptors.descList for predict_tg_ml_smiles's feature vector to be valid --
#: the exact count produced by rdkit-pypi==2022.3.2.1, the version A2023's own console script pins. See the
#: module docstring's "SMILES Mode" caveat for why this is checked rather than assumed.
_EXPECTED_N_RDKIT_DESCRIPTORS = 208

_PICKLE_NAMES = ("fg_cho", "fg_cho_no_tm", "fg_nhal", "fg_nhal_no_tm", "sm", "sm_no_tm")


@lru_cache(maxsize=None)
def _load_model(name: str):
    if name not in _PICKLE_NAMES:
        raise ValueError(f"unknown TgML model {name!r}, expected one of {_PICKLE_NAMES}")
    try:
        import sklearn  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "aiomfac_py.tgml_armeli requires scikit-learn (and, for predict_tg_ml_smiles, rdkit) in a "
            "DEDICATED virtual environment -- pip install 'scikit-learn>=1.1,<1.3' 'numpy<2' there (plus "
            "'rdkit-pypi==2022.3.2.1' on Python <= 3.10 for SMILES Mode). See this subpackage's module "
            "docstring for why these old, narrow version pins are required and why they should not be "
            "installed into the same environment as the rest of aiomfac_py."
        ) from e
    raw = resources.files("aiomfac_py.tgml_armeli").joinpath(f"data/{name}.pkl").read_bytes()
    return pickle.loads(raw)


def _rd_descriptors(smiles: str):
    """The 208 RDKit descriptors (Descriptors.descList) for one SMILES string, computed directly (no
    deepchem/tensorflow dependency -- see PROVENANCE.md for why this is equivalent to the original script)."""
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors
    except ImportError as e:
        raise ImportError(
            "predict_tg_ml_smiles requires rdkit (rdkit-pypi==2022.3.2.1 specifically -- see this "
            "subpackage's module docstring) in the same dedicated virtual environment as scikit-learn<1.3. "
            "Functional Group Mode (predict_tg_ml_fg) does not need rdkit at all."
        ) from e

    n_desc = len(Descriptors.descList)
    if n_desc != _EXPECTED_N_RDKIT_DESCRIPTORS:
        raise RuntimeError(
            f"this RDKit build exposes {n_desc} Descriptors.descList entries, not the "
            f"{_EXPECTED_N_RDKIT_DESCRIPTORS} the SMILES-Mode models were trained on (rdkit-pypi==2022.3.2.1). "
            "Using a mismatched descriptor set would silently produce a wrong-but-plausible-looking Tg, so "
            "this function refuses to run instead -- install rdkit-pypi==2022.3.2.1 (Python <= 3.10 only; "
            "no newer RDKit build reproduces the same 208-descriptor feature vector), or use "
            "predict_tg_ml_fg (Functional Group Mode) or aiomfac_py.viscosity.predict_tg_derieux2018 instead."
        )

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"rdkit could not parse SMILES {smiles!r}")
    values = []
    for desc_name, func in Descriptors.descList:
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

    See the module docstring for the environment this needs (``scikit-learn>=1.1,<1.3`` in a dedicated
    virtual environment) and ``PROVENANCE.md`` for exactly which trained model file each combination of
    arguments routes to.
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

    This needs an exact, old RDKit version to reproduce the 208-element descriptor vector the SMILES-Mode
    models were trained on -- see this subpackage's module docstring ("SMILES Mode" bullet) for why, and for
    why a mismatched RDKit build makes this function raise ``RuntimeError`` rather than silently mispredict.
    """
    desc = _rd_descriptors(smiles)
    model_name = "sm" if Tm_K is not None else "sm_no_tm"
    feat = ([Tm_K] if Tm_K is not None else []) + desc
    model = _load_model(model_name)
    tg = float(model.predict([feat])[0])
    return TgMLResult(tg_K=tg, model_name=model_name)

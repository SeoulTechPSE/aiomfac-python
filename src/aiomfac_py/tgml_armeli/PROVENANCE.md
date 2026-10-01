# Provenance of `data/*.pkl`

These six files are the trained `scikit-learn` `ExtraTreesRegressor` models from:

> Armeli, G., Peters, J.-H., and Koop, T.: Machine-Learning-Based Prediction of the Glass Transition
> Temperature of Organic Compounds Using Experimental Data, *ACS Omega*, 8, 12298-12309,
> doi:10.1021/acsomega.2c08146, 2023.

Obtained by the user directly from the "TgML console script and pickle files" bundle on the paper's own
Zenodo code deposit, DOI [10.5281/zenodo.7650576](https://doi.org/10.5281/zenodo.7650576), and supplied here
unmodified (only renamed with a `.pkl` extension; bytes identical to the files as given).

SHA-256 of the original files (as received, before the rename):

```
262128336cd9d6220e7c8634f1381a1ccbd8e9cad4f69289d725859847b314e8  fg_cho
c22635763bffa42ceefc275a85de498a6e31d49908c949ddd2235e814e13acbb  fg_cho_no_tm
e619280e1db920d157d070d91925bbd4ec28b3c7eb89aefa0a19b9a71322a481  fg_nhal
af886b5a1f8d3f0d1547d40ce48e2e3525ab3aaa626898d99ba1fa9f1ed08a49  fg_nhal_no_tm
766c0f23e8da022e67f223ed3e580d45314f26d1f0d1c2096bf9c28e58884cc3  sm
400eb1cb8fcec27fc31705aa6381f39b0c59742da3c4435422012f8534fc48d6  sm_no_tm
```

## License status -- please check before redistributing

The `readme.txt` shipped alongside these files states only "For further information and contact please
visit https://tgml.chemie.uni-bielefeld.de"; it does not state a license for the pickle files themselves.
The paper's text is CC BY 4.0 (open access), which is permissive, but that does not automatically extend to
the separately-deposited trained-model binaries. This subpackage is written on the assumption that the user
obtained these files legitimately and wants them usable from their own `aiomfac_py` installation -- it is
not a statement that the files carry any particular open-source license. Check the Zenodo deposit itself
(and/or contact the authors via the TgML website above) before redistributing this subpackage's `data/`
directory beyond your own use.

## Why a vendored subpackage, not a pure-Python reimplementation

Unlike `predict_tg_derieux2018` (viscosity.py), which ports a published closed-form equation, this model has
no equation -- it is a trained ensemble of decision trees, originally serialized with Python's `pickle`
module against a now-old `scikit-learn` version (the original `requirements_console_script.txt`, also in
this directory, pins `scikit-learn==1.1.1`). There is nothing to "port" in the equation-rewriting sense: the
only way to use it is to load that fitted tree structure, which is what `__init__.py` in this subpackage
does.

## Pickle-format migration (so a modern scikit-learn/numpy can load these directly)

As originally received (see the SHA-256 hashes above), these six files could only be loaded by
`scikit-learn>=1.1,<1.3` running on `numpy<2` -- scikit-learn 1.3 changed its tree node binary format (adding
a `missing_go_to_left` field), and the pre-1.3 wheels needed to read the old format were themselves only
built against NumPy's pre-2.0 C ABI. That made these files impossible to load alongside a modern
`aiomfac_py` install without a second, separately pinned virtual environment.

`data/*.pkl` as shipped in this subpackage are no longer those original bytes: they are the result of
running `tools/migrate_tgml_pickles.py` (at the repository root) once, which extracts each tree's fitted
state (a plain NumPy structured array -- node splits, impurities, leaf values -- with no scikit-learn class
references at all) under the old scikit-learn, then rebuilds equivalent `ExtraTreeRegressor`/
`ExtraTreesRegressor` objects under a modern scikit-learn, migrating the node array to the new 8-field dtype
(`missing_go_to_left` set to 0 throughout -- these trees never supported missing values, and 0 reproduces the
old splitting behavior exactly for the finite inputs they were always used on). This is **not** a retrain or
an approximation: that script's own `compare` step confirmed the migrated models reproduce the original
models' predictions bit-for-bit (`numpy.array_equal`) across 200 random feature vectors per model, for all
six models. See that script's module docstring for the exact two-environment procedure, in case the
*original*, unmigrated files are ever needed again (e.g. a fresh download from the paper's Zenodo deposit) to
re-run or re-verify the migration -- that is the only point at which the old scikit-learn/numpy combination
is needed; ordinary use of this subpackage never needs it.

## What was simplified relative to the original console script (`TgML_minimal.py`, also in this directory)

The original script's SMILES-mode descriptor extractor wraps RDKit's 208 `Descriptors.descList` functions in
a `deepchem.feat.base_classes.MolecularFeaturizer` subclass, which pulls in `deepchem` and (transitively)
`tensorflow` as dependencies. Inspecting `MolecularFeaturizer.featurize()` (deepchem 2.5.0) shows that, for
the exact way the original script calls it (passing an already-parsed `rdkit.Chem.Mol` object, not a raw
SMILES string), it is a thin loop that simply calls the featurizer's own `_featurize(mol)` once per molecule
with no further transformation -- the method's SMILES-canonicalization/atom-renumbering branch is only taken
when a raw string is passed in, which this script's own `rd_descriptor()` never does (it always parses the
SMILES into a `Mol` itself first). Since every one of the 208 RDKit descriptors used is a whole-molecule
property (molecular weight, counts, topological indices, etc.), not an atom-indexed one, atom renumbering
would not change their values regardless. `tgml_armeli/__init__.py` therefore computes the same 208
descriptors directly from RDKit (`Chem.MolFromSmiles` + `Descriptors.descList`, with `Ipc` using `avg=True`
exactly as the original custom class does), with no `deepchem`/`tensorflow` dependency at all. This was
checked against the original class's logic by reading deepchem 2.5.0's source directly (see git history/PR
description for the exact reasoning), not merely assumed.

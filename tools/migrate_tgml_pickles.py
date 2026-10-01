"""Migrate aiomfac_py/tgml_armeli/data/*.pkl across the scikit-learn tree-serialization format change.

Background: the trained ExtraTreesRegressor models of Armeli, Peters and Koop (2023) were originally
distributed pickled with scikit-learn 1.1.1 (0.24.1-era tree internals). scikit-learn's own tree node binary
format changed in 1.3 (a ``missing_go_to_left`` field was added for missing-value support), so
scikit-learn>=1.3 cannot load the original files at all (``ValueError: node array from the pickle has an
incompatible dtype``) -- and the <1.3 wheels needed to load them were themselves only built against NumPy's
pre-2.0 C ABI. This script performs a one-time, exact migration so the *data/*.pkl files in this repository
can be loaded directly by a modern scikit-learn/numpy, with no special environment needed at import time --
run once (see "Usage" below), its output is what is committed to ``data/``.

What it does, and why it's exact rather than approximate: a scikit-learn ``Tree`` object's fitted state is
just a plain, inspectable structure -- a NumPy structured array of split nodes (``left_child``,
``right_child``, ``feature``, ``threshold``, impurity/sample-count bookkeeping) plus a ``values`` array of
leaf predictions. ``extract()`` (run under the OLD scikit-learn) pulls this out of every tree in every model
into plain dicts/NumPy arrays -- no scikit-learn class references, so the result is safe to unpickle with any
NumPy version and no scikit-learn installed at all. ``reconstruct()`` (run under a MODERN scikit-learn)
rebuilds equivalent ``ExtraTreeRegressor``/``ExtraTreesRegressor`` objects by manually constructing a
``Tree`` and calling its ``__setstate__`` with the extracted node array migrated to the new 8-field dtype
(``missing_go_to_left`` set to 0 throughout -- these trees never supported missing values, and 0 reproduces
the old splitting behavior exactly for the finite inputs they were always used on). This is not a retrain and
not an approximation: it was verified (see ``verify()``) that the migrated models reproduce the original
models' predictions bit-for-bit (``np.array_equal``) across 200 random feature vectors per model.

Usage (two separate Python environments are needed -- see aiomfac_py/tgml_armeli/__init__.py's module
docstring for why the old one is unusually narrowly pinned and should never be the environment you otherwise
work in):

    # 1. In an OLD environment (Python <= 3.10, scikit-learn>=1.1,<1.3, numpy<2) with the *original*,
    #    unmigrated data/*.pkl files from wherever you obtained them (e.g. the paper's Zenodo deposit):
    python migrate_tgml_pickles.py extract /path/to/original_pickles/ /path/to/extracted/

    # 2. Still in the OLD environment, save the original models' predictions on random test vectors (a
    #    migrated file, once rebuilt with a modern numpy, generally cannot be loaded back by the old numpy
    #    used here -- numpy's own internal module layout changed too -- so verification is split into a
    #    "predict" half that runs in each environment separately, and a "compare" half, below):
    python migrate_tgml_pickles.py predict /path/to/original_pickles/ /path/to/extracted/original_preds.pkl

    # 3. In a MODERN environment (whatever this package's own tests run under), reconstruct, predict with
    #    the migrated models, and compare -- this is the actual bit-for-bit verification:
    python migrate_tgml_pickles.py reconstruct /path/to/extracted/ /path/to/aiomfac_py/tgml_armeli/data/
    python migrate_tgml_pickles.py predict /path/to/aiomfac_py/tgml_armeli/data/ /path/to/extracted/migrated_preds.pkl
    python migrate_tgml_pickles.py compare /path/to/extracted/original_preds.pkl /path/to/extracted/migrated_preds.pkl
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

MODEL_NAMES = ("fg_cho", "fg_cho_no_tm", "fg_nhal", "fg_nhal_no_tm", "sm", "sm_no_tm")

_OLD_NODE_FIELDS = ["left_child", "right_child", "feature", "threshold", "impurity", "n_node_samples",
                    "weighted_n_node_samples"]


def extract(src_dir: Path, dst_dir: Path) -> None:
    """Run under the OLD scikit-learn. Pulls each model's fitted state into plain dicts/NumPy arrays."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    for name in MODEL_NAMES:
        with open(src_dir / name, "rb") as f:  # original files have no extension
            model = pickle.load(f)

        estimators = []
        for est in model.estimators_:
            tree_state = est.tree_.__getstate__()
            estimators.append({
                "params": est.get_params(),
                "tree_state": {
                    "max_depth": int(tree_state["max_depth"]),
                    "node_count": int(tree_state["node_count"]),
                    "nodes": tree_state["nodes"].copy(),
                    "values": tree_state["values"].copy(),
                },
            })

        out = {
            "forest_params": model.get_params(),
            "n_features_in_": int(model.n_features_in_),
            "n_outputs_": int(model.n_outputs_),
            "estimators": estimators,
        }
        with open(dst_dir / f"extracted_{name}.pkl", "wb") as f:
            pickle.dump(out, f, protocol=4)
        print(f"{name}: extracted {len(estimators)} trees, n_features_in_={out['n_features_in_']}")


def _migrate_nodes(old_nodes: np.ndarray, new_dtype: np.dtype) -> np.ndarray:
    new_nodes = np.zeros(old_nodes.shape[0], dtype=new_dtype)
    for field in _OLD_NODE_FIELDS:
        new_nodes[field] = old_nodes[field]
    new_nodes["missing_go_to_left"] = 0
    return new_nodes


def reconstruct(src_dir: Path, dst_dir: Path) -> None:
    """Run under a MODERN scikit-learn. Rebuilds loadable ExtraTreesRegressor objects."""
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
    from sklearn.tree._tree import Tree

    new_node_dtype = (DecisionTreeRegressor(max_depth=2)
                       .fit(np.random.RandomState(0).rand(20, 3), np.random.RandomState(0).rand(20))
                       .tree_.__getstate__()["nodes"].dtype)

    dst_dir.mkdir(parents=True, exist_ok=True)
    for name in MODEL_NAMES:
        with open(src_dir / f"extracted_{name}.pkl", "rb") as f:
            extracted = pickle.load(f)
        n_features_in_, n_outputs_ = extracted["n_features_in_"], extracted["n_outputs_"]

        estimators = []
        for est_data in extracted["estimators"]:
            params = dict(est_data["params"])
            if params.get("criterion") == "mse":
                params["criterion"] = "squared_error"
            for p in ("max_features", "random_state", "max_leaf_nodes", "min_impurity_split"):
                params.pop(p, None)  # only affect fit(), never called here; dropped to dodge version drift
            params.setdefault("monotonic_cst", None)

            est = ExtraTreeRegressor(**{k: v for k, v in params.items()
                                         if k in ExtraTreeRegressor().get_params()})
            est.n_features_in_, est.n_outputs_ = n_features_in_, n_outputs_

            tree_state = est_data["tree_state"]
            tree = Tree(n_features_in_, np.array([1], dtype=np.intp), n_outputs_)
            tree.__setstate__({
                "max_depth": tree_state["max_depth"],
                "node_count": tree_state["node_count"],
                "nodes": _migrate_nodes(tree_state["nodes"], new_node_dtype),
                "values": tree_state["values"],
            })
            est.tree_ = tree
            estimators.append(est)

        forest = ExtraTreesRegressor(**{k: v for k, v in extracted["forest_params"].items()
                                         if k in ExtraTreesRegressor().get_params()})
        forest.estimators_ = estimators
        forest.n_features_in_, forest.n_outputs_ = n_features_in_, n_outputs_
        forest.estimator_ = ExtraTreeRegressor()  # present on a normally-fitted forest; unused by predict()

        with open(dst_dir / f"{name}.pkl", "wb") as f:
            pickle.dump(forest, f, protocol=4)
        print(f"{name}: rebuilt {len(estimators)} trees -> {dst_dir / f'{name}.pkl'}")


_N_FEATURES_BY_NAME = {"fg_cho": 11, "fg_cho_no_tm": 10, "fg_nhal": 13, "fg_nhal_no_tm": 12,
                        "sm": 209, "sm_no_tm": 208}


def predict_on_random_vectors(model_dir: Path, out_path: Path, n_samples: int = 200, seed: int = 42) -> None:
    """Run in whichever environment ``model_dir`` is loadable in (original files need the OLD scikit-learn;
    migrated ``*.pkl`` files need a MODERN one). Saves {name: predictions} as a plain dict of NumPy arrays
    (no scikit-learn class references), loadable in the *other* environment for ``compare``."""
    rng = np.random.RandomState(seed)
    preds = {}
    for name, n_feat in _N_FEATURES_BY_NAME.items():
        candidates = [model_dir / name, model_dir / f"{name}.pkl"]
        path = next((p for p in candidates if p.exists()), None)
        if path is None:
            raise FileNotFoundError(f"neither {candidates[0]} nor {candidates[1]} exists")
        with open(path, "rb") as f:
            model = pickle.load(f)
        X = rng.rand(n_samples, n_feat) * 100.0
        preds[name] = model.predict(X)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(preds, f, protocol=4)
    print(f"saved predictions for {len(preds)} models to {out_path}")


def compare(preds_a_path: Path, preds_b_path: Path) -> None:
    """Run in either environment (both files are plain NumPy arrays, no scikit-learn needed to load them).
    Confirms two predict_on_random_vectors() outputs -- one from the original models, one from the migrated
    ones, using the same seed/n_samples -- are bit-for-bit identical."""
    with open(preds_a_path, "rb") as f:
        preds_a = pickle.load(f)
    with open(preds_b_path, "rb") as f:
        preds_b = pickle.load(f)
    all_ok = True
    for name in _N_FEATURES_BY_NAME:
        exact = bool(np.array_equal(preds_a[name], preds_b[name]))
        all_ok &= exact
        print(f"{name}: exact match = {exact}")
    if not all_ok:
        raise SystemExit("verification FAILED for at least one model -- do not ship these migrated files")
    print("All models verified bit-for-bit identical.")


def main():
    valid = ("extract", "reconstruct", "predict", "compare")
    if len(sys.argv) < 4 or sys.argv[1] not in valid:
        print(__doc__)
        raise SystemExit(1)
    cmd = sys.argv[1]
    if cmd == "extract":
        extract(Path(sys.argv[2]), Path(sys.argv[3]))
    elif cmd == "reconstruct":
        reconstruct(Path(sys.argv[2]), Path(sys.argv[3]))
    elif cmd == "predict":
        predict_on_random_vectors(Path(sys.argv[2]), Path(sys.argv[3]))
    else:
        compare(Path(sys.argv[2]), Path(sys.argv[3]))


if __name__ == "__main__":
    main()

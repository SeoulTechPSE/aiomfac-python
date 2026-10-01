"""aiomfac_py.tgml_armeli (Armeli, Peters and Koop, 2023 ML Tg predictor).

Two independent kinds of check here, split because they need different environments:

1. ``TestFeatureVectorConstruction`` -- the feature-vector assembly and model-name routing logic in
   ``predict_tg_ml_fg``/``predict_tg_ml_smiles`` (order of entries, CHO vs. NHal routing, with/without Tm
   routing), checked with ``_load_model`` monkeypatched to a fake model that just records what it was asked
   to predict on. This needs no optional dependency at all and always runs.
2. ``TestRealPredictions`` -- actually loading the vendored (migrated) pickles and predicting, which needs
   ``scikit-learn`` installed (the ``tgml`` extra) and, for SMILES Mode, ``rdkit`` as well (the
   ``tgml-smiles`` extra) -- both install normally alongside the rest of aiomfac_py, see
   ``aiomfac_py/tgml_armeli/__init__.py``'s module docstring. Skipped automatically unless scikit-learn is
   actually importable here. The reference Tg values below were obtained by running this exact test class
   during development (Python 3.10, scikit-learn 1.2.2, numpy 1.22.4, rdkit-pypi 2022.3.2.1, before the
   pickle-format migration) and reconfirmed bit-for-bit identical after the migration via
   ``tools/migrate_tgml_pickles.py``'s own verification step.
"""
from __future__ import annotations

import pytest

import aiomfac_py.tgml_armeli as tgml


def _sklearn_is_installed() -> bool:
    try:
        import sklearn  # noqa: F401
    except ImportError:
        return False
    return True


_HAS_SKLEARN = _sklearn_is_installed()


class _FakeModel:
    """Stands in for a loaded sklearn estimator: records the X it was asked to predict on, returns a fixed
    value, so the feature-vector-construction logic can be checked without scikit-learn installed."""

    def __init__(self, tg_value=123.0):
        self.tg_value = tg_value
        self.calls = []

    def predict(self, X):
        self.calls.append(X)
        return [self.tg_value]


class TestFeatureVectorConstruction:
    """No optional dependency needed -- _load_model is monkeypatched out."""

    def _patch_loader(self, monkeypatch):
        fake = _FakeModel()
        calls = {}

        def fake_load_model(name):
            calls["name"] = name
            return fake

        monkeypatch.setattr(tgml, "_load_model", fake_load_model)
        return fake, calls

    def test_fg_cho_no_tm_feature_order(self, monkeypatch):
        fake, calls = self._patch_loader(monkeypatch)
        tgml.predict_tg_ml_fg(1, 2, 3, 4, 5, 6, 7, 8, o_to_c=0.5, molar_mass_g_mol=100.0)
        assert calls["name"] == "fg_cho_no_tm"
        # [CH3, CH2, CH, C, OH, -O-, =O, DBE, O:C, M] -- no N/Hal (both default 0), no Tm
        assert fake.calls[-1] == [[1, 2, 3, 4, 5, 6, 7, 8, 0.5, 100.0]]

    def test_fg_cho_with_tm_feature_order(self, monkeypatch):
        fake, calls = self._patch_loader(monkeypatch)
        tgml.predict_tg_ml_fg(1, 2, 3, 4, 5, 6, 7, 8, o_to_c=0.5, molar_mass_g_mol=100.0, Tm_K=300.0)
        assert calls["name"] == "fg_cho"
        # [CH3, CH2, CH, C, OH, -O-, =O, DBE, O:C, M, Tm]
        assert fake.calls[-1] == [[1, 2, 3, 4, 5, 6, 7, 8, 0.5, 100.0, 300.0]]

    def test_fg_nhal_routes_when_n_or_hal_nonzero(self, monkeypatch):
        fake, calls = self._patch_loader(monkeypatch)
        tgml.predict_tg_ml_fg(0, 0, 5, 1, 0, 0, 0, 4, o_to_c=1.0, molar_mass_g_mol=93.0,
                               n_N=0.0, n_hal=1.0, Tm_K=200.0)
        assert calls["name"] == "fg_nhal"
        # [CH3, CH2, CH, C, OH, -O-, =O, DBE, N, Hal, O:C, M, Tm]
        assert fake.calls[-1] == [[0, 0, 5, 1, 0, 0, 0, 4, 0.0, 1.0, 1.0, 93.0, 200.0]]

    def test_fg_nhal_no_tm_routes_and_omits_tm(self, monkeypatch):
        fake, calls = self._patch_loader(monkeypatch)
        tgml.predict_tg_ml_fg(0, 0, 5, 1, 0, 0, 0, 4, o_to_c=1.0, molar_mass_g_mol=93.0, n_N=1.0)
        assert calls["name"] == "fg_nhal_no_tm"
        assert fake.calls[-1] == [[0, 0, 5, 1, 0, 0, 0, 4, 1.0, 0.0, 1.0, 93.0]]

    def test_result_carries_model_name_and_predicted_value(self, monkeypatch):
        fake, _ = self._patch_loader(monkeypatch)
        fake.tg_value = 187.0
        result = tgml.predict_tg_ml_fg(1, 1, 0, 0, 1, 0, 0, 0, o_to_c=0.5, molar_mass_g_mol=46.07)
        assert result.tg_K == pytest.approx(187.0)
        assert result.model_name == "fg_cho_no_tm"

    def test_smiles_mode_routes_on_tm(self, monkeypatch):
        fake, calls = self._patch_loader(monkeypatch)
        monkeypatch.setattr(tgml, "_rd_descriptors", lambda smiles: [0.0] * 208)
        tgml.predict_tg_ml_smiles("C1=CC=CC=C1", Tm_K=279.0)
        assert calls["name"] == "sm"
        assert fake.calls[-1] == [[279.0] + [0.0] * 208]

        tgml.predict_tg_ml_smiles("C1=CC=CC=C1")
        assert calls["name"] == "sm_no_tm"
        assert fake.calls[-1] == [[0.0] * 208]

    def test_rd_descriptors_guards_missing_named_descriptor(self, monkeypatch):
        # Simulate an RDKit build that is missing one of the 208 named descriptors SMILES Mode looks up by
        # name -- this must refuse with a clear error, not silently mispredict on a short feature vector.
        pytest.importorskip("rdkit", reason="requires rdkit to exercise the real descriptor-presence guard")
        from rdkit.Chem import Descriptors
        monkeypatch.setattr(Descriptors, "descList", Descriptors.descList[:5])  # pretend only 5 exist
        with pytest.raises(RuntimeError, match="missing .* of the 208 named descriptors"):
            tgml._rd_descriptors("C1=CC=CC=C1")


@pytest.mark.skipif(not _HAS_SKLEARN, reason="requires the optional tgml extra (see module docstring)")
class TestRealPredictions:
    """Needs scikit-learn installed (and, for SMILES Mode, rdkit as well) -- see this module's docstring.
    Skipped automatically otherwise (the skipif above, not an in-body importorskip, so that skipping this
    class doesn't also skip collection of the rest of this file)."""

    def test_fg_cho_no_tm_ethanol_is_physically_reasonable(self):
        # ethanol: CH3=1, CH2=1, CH=0, C=0, OH=1, -O-=0, =O=0, DBE=0, O:C=0.5, M=46.07 g/mol.
        # Measured ethanol Tg is ~97 K (Angell, 1997); A2023 itself reports ~13 K MAE for this mode.
        result = tgml.predict_tg_ml_fg(1, 1, 0, 0, 1, 0, 0, 0, o_to_c=0.5, molar_mass_g_mol=46.07)
        assert result.model_name == "fg_cho_no_tm"
        assert 60.0 < result.tg_K < 140.0

    def test_fg_nhal_paper_example_matches_reference_run(self):
        # TgML_minimal.py's own worked example (vendor/TgML_minimal.py): feat = [0, 0, 5, 1, 0, 0, 0, 4, 1,
        # 0, 0.0, 93, 200] for model 'fg_nhal'. Reference value from a dedicated-environment run during
        # development (Python 3.10, scikit-learn 1.2.2, numpy 1.22.4); re-run this test in the same kind of
        # environment to reconfirm after any dependency or pickle-file change.
        result = tgml.predict_tg_ml_fg(0, 0, 5, 1, 0, 0, 0, 4, o_to_c=0.0, molar_mass_g_mol=93.0,
                                        n_N=1.0, n_hal=0.0, Tm_K=200.0)
        assert result.model_name == "fg_nhal"
        assert result.tg_K == pytest.approx(129.892, abs=0.5)

    def test_fg_cho_vs_fg_nhal_give_different_models(self):
        kwargs = dict(n_ch3=0, n_ch2=0, n_ch=5, n_c=1, n_oh=0, n_ether_o=0, n_carbonyl_o=0, dbe=4,
                      o_to_c=1.0, molar_mass_g_mol=93.0, Tm_K=200.0)
        cho = tgml.predict_tg_ml_fg(**kwargs)
        nhal = tgml.predict_tg_ml_fg(**kwargs, n_N=1.0)
        assert cho.model_name == "fg_cho"
        assert nhal.model_name == "fg_nhal"

    def test_smiles_mode_benzene_matches_reference_run(self):
        pytest.importorskip("rdkit", reason="SMILES Mode also requires rdkit (see module docstring)")
        # Reference values from a dedicated-environment run during development (scikit-learn 1.2.2, numpy
        # 1.22.4, rdkit-pypi 2022.3.2.1); the by-name descriptor lookup means this is expected to still match
        # (within tolerance) on any reasonably current RDKit build, not just that exact pinned version.
        result = tgml.predict_tg_ml_smiles("C1=CC=CC=C1", Tm_K=279.0)
        assert result.model_name == "sm"
        assert result.tg_K == pytest.approx(118.577, abs=0.5)

        result_no_tm = tgml.predict_tg_ml_smiles("C1=CC=CC=C1")
        assert result_no_tm.model_name == "sm_no_tm"
        assert result_no_tm.tg_K == pytest.approx(118.157, abs=0.5)

    def test_smiles_mode_invalid_smiles_raises(self):
        pytest.importorskip("rdkit", reason="SMILES Mode also requires rdkit (see module docstring)")
        with pytest.raises(ValueError):
            tgml.predict_tg_ml_smiles("not a smiles string(((")


class TestImportGuardWithoutDeps:
    """Only meaningful (and only runs) when scikit-learn is genuinely not installed -- checks that the
    error message is actually helpful rather than a bare ModuleNotFoundError traceback."""

    def test_predict_tg_ml_fg_raises_clear_import_error(self):
        if _HAS_SKLEARN:
            pytest.skip("scikit-learn is installed in this environment; see TestRealPredictions instead")
        with pytest.raises(ImportError, match="pip install"):
            tgml.predict_tg_ml_fg(1, 1, 0, 0, 1, 0, 0, 0, o_to_c=0.5, molar_mass_g_mol=46.07)

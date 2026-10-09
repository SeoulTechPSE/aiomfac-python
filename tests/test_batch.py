"""ExplicitLiquidModel.ln_a_batch (vectorized JAX transcription) against the validated NumPy ln_a, point by point."""
import numpy as np
import pytest

jax = pytest.importorskip("jax")

from aiomfac_py import Component  # noqa: E402
from aiomfac_py.phase_equilibrium import ExplicitLiquidModel  # noqa: E402

ORG = Component(2, "pinonic-like", ((1, 2), (2, 1), (3, 1), (137, 1), (21, 1)))


@pytest.mark.parametrize("organics,ions", [
    ([], ["Na+", "Cl-"]),
    ([ORG], []),
    ([ORG], ["Na+", "Cl-"]),
    ([], ["NH4+", "H+", "SO4--"]),
    ([], ["Na+", "H+", "CO3--"]),
    ([], ["Mg++", "Li+", "Br-", "Cl-"]),
])
def test_batch_matches_pointwise(organics, ions):
    lm = ExplicitLiquidModel(organics, ions)
    rng = np.random.default_rng(0)
    for T in (273.15, 298.15):
        n = np.concatenate([np.full((40, 1), 55.5), rng.uniform(0.01, 3.0, (40, lm.N - 1))], 1)
        ref = np.stack([lm.ln_a(x, T) for x in n])
        got = lm.ln_a_batch(n, T)
        assert got.shape == ref.shape
        np.testing.assert_allclose(got, ref, rtol=0, atol=1e-11)


def test_batch_single_row():
    lm = ExplicitLiquidModel([], ["Na+", "Cl-"])
    n = np.array([55.5, 1.0, 1.0])
    np.testing.assert_allclose(lm.ln_a_batch(n, 298.15)[0], lm.ln_a(n, 298.15), rtol=0, atol=1e-12)

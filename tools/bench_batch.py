"""Per-composition cost of ExplicitLiquidModel.ln_a (NumPy), one call of the jit-compiled JAX transcription, and the
vectorized batch (ln_a_batch), one CPU thread.  usage: python tools/bench_batch.py"""
import os
import time

os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
import numpy as np
import jax.numpy as jnp

from aiomfac_py import Component
from aiomfac_py.ad_activity import build_jacobian
from aiomfac_py.phase_equilibrium import ExplicitLiquidModel

ORG = Component(2, "pinonic-like", ((1, 2), (2, 1), (3, 1), (137, 1), (21, 1)))


def tm(f, k):
    for _ in range(10):
        f()
    t = time.perf_counter()
    for _ in range(k):
        f()
    return (time.perf_counter() - t) / k * 1e6


if __name__ == "__main__":
    T = 298.15
    for label, orgs, ions in (("water + organic", [ORG], []), ("NaCl", [], ["Na+", "Cl-"]),
                              ("organic + NaCl", [ORG], ["Na+", "Cl-"]), ("NH4+ / H+ / SO4--", [], ["NH4+", "H+", "SO4--"])):
        lm = ExplicitLiquidModel(orgs, ions)
        n = np.concatenate([[55.5], np.full(lm.N - 1, 1.0)])
        mu = build_jacobian(lm, T)[1]
        row = {"numpy_us": tm(lambda: lm.ln_a(n, T), 500),
               "jax_one_us": tm(lambda: mu(jnp.asarray(n)).block_until_ready(), 500)}
        for B in (100, 1000, 10000):
            nb = n * np.linspace(0.8, 1.2, B)[:, None]
            lm.ln_a_batch(nb, T)
            row[f"batch{B}_us_per_point"] = tm(lambda: lm.ln_a_batch(nb, T), max(3, 20000 // B)) / B
        print(f"{label:20s}", {k: round(v, 3) for k, v in row.items()}, flush=True)

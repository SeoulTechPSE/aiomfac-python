"""Liquid phases defined by a Gibbs-energy function, for the phase-equilibrium solver.

:class:`GibbsLiquidModel` replaces the AIOMFAC activities of :class:`aiomfac_py.phase_equilibrium.ExplicitLiquidModel`
by the derivatives of one scalar function, the (transformed) Gibbs energy of the liquid,

    g(n) = G/RT of a liquid with species amounts n  (ideal mixing + excess part, without the reaction constants),
    ln a_i = dg/dn_i,   d ln a_i/dn_j = d2g/dn_i dn_j,

so that the Gibbs-Duhem equation holds exactly and the Hessian is symmetric.  It is meant for excess-Gibbs-energy
surrogates of AIOMFAC (neural networks) or any other model written as a JAX function.

All derivatives are obtained by automatic differentiation, compiled once and reused:

* ``ln_a``: reverse mode, ``jax.jit(jax.grad(g))``;
* ``hessian_ad``: forward-over-reverse, ``jax.jit(jax.jacfwd(jax.grad(g)))``, i.e. N Hessian-vector products
  evaluated together in one compiled call (the solver needs the whole reduced Hessian: it is diagonalized to handle
  non-convex regions);
* ``hvp``: one Hessian-vector product, forward-over-reverse ``jax.jvp(jax.grad(g), (n,), (v,))``, for matrix-free
  use (e.g. Krylov solvers or curvature checks along a direction in systems with many species).

The compiled functions are cached by (Gibbs function, species of the liquid), so the solver's child problems (fewer
species), repeated solves and new :class:`PhaseEquilibrium` instances of the same system reuse them.  Temperature
enters only through the constants returned by ``consts(T)`` (computed with NumPy outside the compiled code and passed
as arguments), so a change of temperature does not recompile.

Optional dependency: ``jax`` (``pip install jax``).  Importing this module enables 64-bit floats in JAX.

Example (regular solution of water and one organic, g = sum n ln x + A n_w n_o / n_T)::

    import jax.numpy as jnp
    from aiomfac_py.gibbs_model import GibbsFunction, GibbsLiquidModel, ideal_mixing
    def g(n, k):                      # n over the species names below
        return ideal_mixing(n) + k["A"] * n[0] * n[1] / jnp.sum(n)
    gf = GibbsFunction(["Water", "org"], g, consts=lambda T: {"A": 2.5})
    lm = GibbsLiquidModel([org_component], [], gf)
    pe = PhaseEquilibrium([org_component], [], 298.15, liquid_model=lm)
"""
from __future__ import annotations

import math
from typing import Callable, Sequence

import numpy as np

try:
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    AVAILABLE = True
except ImportError:                                           # pragma: no cover - optional dependency
    jax = jnp = None
    AVAILABLE = False

from .io import Component
from .phase_equilibrium import ExplicitLiquidModel

__all__ = ["AVAILABLE", "GibbsFunction", "GibbsLiquidModel", "enable_persistent_cache", "ideal_mixing", "xlogy_safe"]


def enable_persistent_cache(path: str, min_compile_time_s: float = 0.0):
    """Keep JAX's compiled functions on disk (``path``), so that new processes reuse them instead of compiling again
    (the first solve of a system otherwise pays one compilation, ~1-3 s for a neural-network Gibbs function)."""
    jax.config.update("jax_compilation_cache_dir", str(path))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", float(min_compile_time_s))
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)


def xlogy_safe(x, y):
    """x ln y with 0 ln 0 = 0 and finite derivatives at x = 0 (double-where), for amounts that may be zero."""
    ok = x > 0.0
    return jnp.where(ok, x * jnp.log(jnp.where(ok, y, 1.0)), 0.0)


def ideal_mixing(n):
    """sum_i n_i ln(n_i / n_T): ideal mixing on the mole-fraction scale of all species (its gradient is ln x_i)."""
    return jnp.sum(xlogy_safe(n, n / jnp.sum(n)))


class GibbsFunction:
    """A Gibbs-energy function over a fixed list of species names.

    ``g(n, k)``: JAX-traceable scalar G/RT of a liquid, ``n`` the amounts of all ``names`` (absent species have amount
    0 and must not produce NaN, see :func:`xlogy_safe`), ``k`` the pytree returned by ``consts(T)``.  ``g`` must be
    homogeneous of degree one in ``n``.  Names follow :class:`ExplicitLiquidModel`: "Water", the organic component
    names, ion keys of :data:`aiomfac_py.solids.ION_REGISTRY` (HSO4- as an explicit species of acid systems)."""

    def __init__(self, names: Sequence[str], g: Callable, consts: Callable[[float], dict] | None = None,
                 label: str = "gibbs", compiled_cache: dict | None = None):
        """``compiled_cache``: a dict shared by Gibbs functions that differ only in their constants (the same ``g``,
        with e.g. molecule-specific inputs passed through ``consts``), so that they share the compiled functions:
        a family of molecules evaluated with one network is then compiled once."""
        if not AVAILABLE:
            raise ImportError("jax is not installed (pip install jax)")
        self.names = list(names)
        self.g = g
        self.consts = consts or (lambda T: {})
        self.label = label
        self._compiled: dict = {} if compiled_cache is None else compiled_cache
        self._kcache: dict = {}
        self._klast = None

    def constants(self, T: float):
        if self._klast is not None and self._klast[0] == T:          # the solver works at one temperature
            return self._klast[1]
        key = round(float(T), 9)
        if key not in self._kcache:
            self._kcache[key] = jax.tree_util.tree_map(lambda v: jnp.asarray(v, dtype=jnp.float64), self.consts(float(T)))
        self._klast = (T, self._kcache[key])
        return self._klast[1]

    def compiled(self, idx: tuple):
        """(value_and_grad, grad, Hessian, HVP) of g restricted to the species ``idx`` (indices into ``names``), jit-
        compiled once per species set."""
        if idx not in self._compiled:
            full, ix = len(self.names), jnp.asarray(idx, dtype=jnp.int32)

            def g_sub(n, k):
                return self.g(jnp.zeros(full, dtype=n.dtype).at[ix].set(n), k)

            grad = jax.grad(g_sub)
            self._compiled[idx] = (
                jax.jit(jax.value_and_grad(g_sub)),
                jax.jit(grad),
                jax.jit(jax.jacfwd(grad)),                                     # forward-over-reverse Hessian
                jax.jit(lambda n, k, v: jax.jvp(lambda m: grad(m, k), (n,), (v,))[1]),   # forward-over-reverse HVP
            )
        return self._compiled[idx]


class GibbsLiquidModel(ExplicitLiquidModel):
    """Liquid model with ``ln a = grad g`` of a :class:`GibbsFunction` (see the module docstring).

    The species, components, balances and reaction constants are those of :class:`ExplicitLiquidModel` for the same
    organics and ions; every species of the liquid must be one of ``gibbs.names``.  Starting points of the solver
    (initial speciation, water guesses) still use the AIOMFAC model of the base class; they do not enter the
    equilibrium conditions."""

    has_exact_hessian = True

    def __init__(self, organics: Sequence[Component], ions: Sequence[str], gibbs: GibbsFunction):
        super().__init__(organics, ions)
        missing = [s for s in self.names if s not in gibbs.names]
        if missing:
            raise ValueError(f"species {missing} are not variables of the Gibbs function {gibbs.label!r}")
        self.gibbs = gibbs
        self._idx = tuple(gibbs.names.index(s) for s in self.names)
        self._f = gibbs.compiled(self._idx)                       # compiled once, held by the model
        self.n_hvp = 0

    def restrict(self, organics: Sequence[Component], ions: Sequence[str]) -> "GibbsLiquidModel":
        """The same Gibbs function for a subsystem (the solver's child problems); compiled functions are shared."""
        return GibbsLiquidModel(organics, ions, self.gibbs)

    # ---------------------------------------------------------------------------------------------------
    def _fns(self):
        return self._f

    def gibbs_value(self, n: np.ndarray, T: float) -> float:
        """g(n) (without the reaction constants)."""
        return float(self._f[0](np.asarray(n, dtype=float), self.gibbs.constants(T))[0])

    def ln_a(self, n: np.ndarray, T: float) -> np.ndarray:
        """c_s + dg/dn_s of every species; -inf for absent species (amount 0), as for the AIOMFAC model."""
        self.n_eval += 1
        if self._T != T:
            self.set_conditions(T, self._ln_rh)
        n = np.asarray(n, dtype=float)
        out = np.asarray(self._f[1](n, self.gibbs.constants(T))) + self._c
        if (n <= 0.0).any():
            out[n <= 0.0] = -math.inf
        return out

    def hessian_ad(self, n: np.ndarray, T: float, active: np.ndarray | None = None) -> np.ndarray:
        """Exact d ln a_i/d n_j = d2g/dn_i dn_j (forward-over-reverse; symmetric to round-off, symmetrized); rows and
        columns of absent species are zero."""
        self.n_jac += 1
        H = np.array(self._f[2](np.asarray(n, dtype=float), self.gibbs.constants(T)))
        if active is not None and not np.all(active):
            off = ~np.asarray(active, dtype=bool)
            H[off, :] = 0.0
            H[:, off] = 0.0
        H += H.T
        H *= 0.5
        return H

    def hvp(self, n: np.ndarray, T: float, v: np.ndarray) -> np.ndarray:
        """Hessian-vector product (d2g/dn2) v without forming the Hessian (forward-over-reverse)."""
        self.n_hvp += 1
        return np.asarray(self._f[3](np.asarray(n, dtype=float), self.gibbs.constants(T), np.asarray(v, dtype=float)))

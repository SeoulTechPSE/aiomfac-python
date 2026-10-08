"""Main tool: one entry point for the equilibrium solvers of aiomfac_py.

A *case* (a TOML or JSON file, or a Python dict) describes the system (temperature, organics, ions), the feed, the
conditions (RH, gases) and the *calculation mode*; the solver used for that mode, and its options, are chosen in the
``[solver]`` section.  :func:`run` returns a :class:`RunResult` with one :class:`PointResult` per RH (or one for the
modes that return a single number), which can be written as JSON, CSV or a text summary.

Calculation modes (``[calculation] mode``) and the engines that can solve them (``[solver] engine``; the first one is
the default):

==================  =====================  =========================================================================
mode                engines                problem
==================  =====================  =========================================================================
``activity``        ``aiomfac``            activities of a given liquid composition (no equilibrium)
``equilibrium``     ``pe``, ``sle``        liquid-liquid-solid equilibrium at fixed T, RH (water open)
``lle``             ``pe``, ``pep``        liquid-liquid equilibrium without solids (``pep``: fixed total composition)
``sle``             ``sle``, ``pe``        solid-liquid equilibrium of an inorganic aqueous system
``gas_particle``    ``gp``                 gas/particle partitioning of semivolatile organics (Zuend et al., 2010)
``drying_path``     ``pe``                 decreasing RH with crystallization after a critical supersaturation
``deliquescence``   ``sle``                deliquescence and full-dissolution RH of an inorganic feed
``efflorescence``   ``sle``                efflorescence RH of an inorganic feed for a critical supersaturation
``stability``       ``pe``                 stability (tangent-plane distance) of the one-liquid state
==================  =====================  =========================================================================

Volatile inorganic gases (NH3, HNO3, HCl, CO2) are given in the ``[gas]`` section and apply to ``equilibrium``,
``lle``, ``sle``, ``drying_path`` and ``stability``: an open system at fixed partial pressures, or a closed system with
a finite amount of air.

Command line::

    aiomfac-tool run case.toml [-o result.json] [--format json|csv|text]
    aiomfac-tool template equilibrium > case.toml
    aiomfac-tool list solids|ions|gases|salts|modes
    aiomfac-tool smiles "CC(=O)O" acetic_acid

See ``docs/user_manual.md`` for the full description of the case format.
"""
from __future__ import annotations

import argparse
import csv
import io as _io
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from . import __version__
from .io import Component
from .params import load_subgroup_params

MW_WATER_G = 18.01528

MODES = {
    "activity": ("aiomfac",),
    "equilibrium": ("pe", "sle"),
    "lle": ("pe", "pep"),
    "sle": ("sle", "pe"),
    "gas_particle": ("gp",),
    "drying_path": ("pe",),
    "deliquescence": ("sle",),
    "efflorescence": ("sle",),
    "stability": ("pe",),
}

MODE_HELP = {
    "activity": "activities of a given liquid composition (no equilibrium)",
    "equilibrium": "liquid-liquid-solid equilibrium at fixed T and RH",
    "lle": "liquid-liquid equilibrium without solids (metastable w.r.t. crystallization)",
    "sle": "solid-liquid equilibrium of an inorganic aqueous system",
    "gas_particle": "gas/particle partitioning of semivolatile organics",
    "drying_path": "decreasing-RH path with crystallization after a critical supersaturation",
    "deliquescence": "deliquescence and full-dissolution RH of an inorganic feed",
    "efflorescence": "efflorescence RH of an inorganic feed for a critical supersaturation",
    "stability": "stability (tangent-plane distance) of the one-liquid state",
}

# options of PhaseEquilibrium that can be set from [solver] (attribute name -> allowed values or type)
_PE_ATTRS = {
    "inner_method": ("newton", "rand", "barrier"),
    "hess_scheme": ("split", "ad", "central"),
    "tpd_method": ("ss", "newton"),
    "seed_method": ("linesearch", "fixed"),
    "tpd_early_stop": float,
    "trace_tol": float,
    "tol_tpd": float,
    "si_tol": float,
    "hess_reuse_tol": float,
    "tpd_hess_reuse_tol": float,
}
_SOLVER_KEYS = set(_PE_ATTRS) | {"engine", "solids", "max_liquids", "max_outer", "speciation", "k_mode", "warm",
                                 "path", "gp_method", "gp_max_iter", "gp_tol", "check_lle", "tol", "eps_init",
                                 "bracket", "rh_tol"}


_ALIASES = {"HSO4-": {"H+": 1, "SO4--": 1}, "HCO3-": {"H+": 1, "CO3--": 1}, "OH-": {"H+": -1}}


class CaseError(ValueError):
    """An invalid case description."""


# ============================================================================================================
# Reading a case
# ============================================================================================================

def load_case_file(path: str | Path) -> dict:
    """Read a case file: ``.toml`` (Python >= 3.11, or with the ``tomli`` package) or ``.json``."""
    path = Path(path)
    text = path.read_text()
    if path.suffix.lower() == ".json":
        return json.loads(text)
    try:
        import tomllib                                       # Python >= 3.11
    except ImportError:                                      # pragma: no cover
        try:
            import tomli as tomllib
        except ImportError as e:
            raise CaseError("reading TOML needs Python >= 3.11 or the 'tomli' package; use a .json case file") from e
    return tomllib.loads(text)


def _organic(spec: dict, number: int) -> Component:
    if "name" not in spec:
        raise CaseError(f"organic #{number - 1} needs a 'name'")
    name = str(spec["name"])
    if "subgroups" in spec:
        sg = tuple((int(s), int(c)) for s, c in spec["subgroups"])
        return Component(number, name, sg)
    if "smiles" in spec:
        from .s2as import smiles_to_components
        res = smiles_to_components([str(spec["smiles"])], names=[name])
        c = res.components[1]
        return Component(number, name, tuple(c.subgroups))
    raise CaseError(f"organic {name!r} needs 'subgroups' or 'smiles'")


def organic_molar_mass_g(c: Component) -> float:
    """Molar mass [g/mol] of an organic component from its AIOMFAC subgroups."""
    gmw = load_subgroup_params().GroupMW
    return float(sum(n * gmw[s - 1] for s, n in c.subgroups))


def ion_molar_mass_g(ion: str) -> float:
    from .solids import ION_REGISTRY
    sg = load_subgroup_params()
    sid, z = ION_REGISTRY[ion]
    return float(sg.SMWC[sid - 201] if z > 0 else sg.SMWA[sid - 241])


def salt_ions(formula: str) -> dict:
    from .sle import SALT_IONS
    if formula not in SALT_IONS:
        raise CaseError(f"unknown salt {formula!r}; known salts: {', '.join(SALT_IONS)}")
    return dict(SALT_IONS[formula])


def salt_molar_mass_g(formula: str) -> float:
    return sum(nu * ion_molar_mass_g(i) for i, nu in salt_ions(formula).items())


def _as_list(v):
    if v is None:
        return None
    if isinstance(v, (list, tuple)):
        return [float(x) for x in v]
    return [float(v)]


@dataclass
class Case:
    """A parsed case.  Build it with :meth:`from_dict` or :meth:`from_file`."""
    mode: str
    engine: str
    T_K: float
    organics: list
    ions: list
    feed: dict                                  # mol of organics (by name) and ions (keys of ION_REGISTRY)
    water: float | None = None                  # mol of water (activity mode, fixed-composition LLE)
    salts: dict = field(default_factory=dict)   # mol of salt formula units as given (fixed-composition LLE)
    rh: list | None = None
    gas: dict = field(default_factory=dict)
    solver: dict = field(default_factory=dict)
    options: dict = field(default_factory=dict) # mode-specific sections: [drying], [gas_particle], [efflorescence]
    output: dict = field(default_factory=dict)
    title: str = ""
    spec: dict = field(default_factory=dict)

    # --------------------------------------------------------------------------------------------------
    @classmethod
    def from_file(cls, path: str | Path) -> "Case":
        return cls.from_dict(load_case_file(path))

    @classmethod
    def from_dict(cls, d: dict) -> "Case":
        from .solids import ION_REGISTRY
        d = dict(d)
        calc = dict(d.get("calculation", {}))
        system = dict(d.get("system", {}))
        mode = str(calc.get("mode", d.get("mode", "equilibrium")))
        if mode not in MODES:
            raise CaseError(f"unknown mode {mode!r}; modes: {', '.join(MODES)}")
        solver = dict(d.get("solver", {}))
        unknown = set(solver) - _SOLVER_KEYS
        if unknown:
            raise CaseError(f"unknown [solver] option(s): {', '.join(sorted(unknown))}")
        engine = str(solver.get("engine", "auto"))
        T = system.get("T_K", d.get("T_K"))
        if T is None:
            raise CaseError("[system] T_K is required")
        T = float(T)
        organics = [_organic(o, k + 2) for k, o in enumerate(system.get("organics", []))]
        org_names = [c.name for c in organics]
        if len(set(org_names)) != len(org_names):
            raise CaseError("organic names must be unique")
        # feed: mol, mol of salts, g, g of salts
        feed: dict = {}

        def add(k, v):
            if v:
                feed[k] = feed.get(k, 0.0) + float(v)

        water = None
        salts_mol: dict = {}
        for k, v in dict(d.get("feed", {})).items():
            if k.lower() == "water":
                water = float(v)
            else:
                add(k, v)
        for k, v in dict(d.get("feed_mass_g", {})).items():
            if k.lower() == "water":
                water = (water or 0.0) + float(v) / MW_WATER_G
            elif k in org_names:
                add(k, float(v) / organic_molar_mass_g(organics[org_names.index(k)]))
            elif k in ION_REGISTRY:
                add(k, float(v) / ion_molar_mass_g(k))
            else:
                raise CaseError(f"[feed_mass_g]: {k!r} is neither an organic of [system] nor an ion")
        for k, v in dict(d.get("salts", {})).items():
            salts_mol[k] = salts_mol.get(k, 0.0) + float(v)
        for k, v in dict(d.get("salts_mass_g", {})).items():
            salts_mol[k] = salts_mol.get(k, 0.0) + float(v) / salt_molar_mass_g(k)
        for k, n in salts_mol.items():
            for ion, nu in salt_ions(k).items():
                add(ion, nu * n)
        # bisulfate, bicarbonate and hydroxide are carried as stoichiometric ions (speciated inside the solvers):
        # HSO4- = H+ + SO4--, HCO3- = H+ + CO3--, OH- = -H+ (the proton excess of a carbonate system)
        for alias, parts in _ALIASES.items():
            if alias in feed:
                nb = feed.pop(alias)
                for ion, nu in parts.items():
                    feed[ion] = feed.get(ion, 0.0) + nu * nb
        for k in feed:
            if k not in org_names and k not in ION_REGISTRY:
                raise CaseError(f"feed entry {k!r} is neither an organic of [system] nor an ion "
                                f"(ions: {', '.join(ION_REGISTRY)})")
        # gases
        gas = dict(d.get("gas", {}))
        if gas:
            gmode = gas.get("mode", "open")
            if gmode not in ("open", "closed"):
                raise CaseError("[gas] mode must be 'open' or 'closed'")
            from .gases import GASES
            keys = list(gas.get("partial_pressure_atm", {})) if gmode == "open" else list(gas.get("total_mol", {}))
            for g in keys:
                if g not in GASES:
                    raise CaseError(f"unknown gas {g!r}; gases: {', '.join(GASES)}")
            if gmode == "closed" and "n_air" not in gas:
                raise CaseError("[gas] mode = 'closed' needs n_air (mol of air)")
        # ions: given, or every ion of the feed and of the gases (H+ with CO3--)
        ions = system.get("ions")
        if ions is None:
            ions = [k for k in feed if k in ION_REGISTRY]
            if gas:
                from .gases import GASES
                for g in (gas.get("partial_pressure_atm", {}) or gas.get("total_mol", {})):
                    for i in GASES[g].ions:
                        if i not in ions:
                            ions.append(i)
            if "CO3--" in ions and "H+" not in ions:
                ions.append("H+")
        ions_in = [str(i) for i in ions]
        ions = []
        for i in ions_in:
            for j in (_ALIASES[i] if i in _ALIASES else {i: 1}):
                if j not in ions:
                    ions.append(j)
        if "CO3--" in ions and "H+" not in ions:
            ions.append("H+")
        for i in ions:
            if i not in ION_REGISTRY:
                raise CaseError(f"unknown ion {i!r}")
        # engine
        allowed = MODES[mode]
        if engine == "auto":
            engine = allowed[0]
        if engine not in allowed:
            raise CaseError(f"engine {engine!r} cannot solve mode {mode!r}; engines for it: {', '.join(allowed)}")
        rh = _as_list(calc.get("rh", d.get("rh")))
        options = {k: dict(d[k]) for k in ("drying", "gas_particle", "efflorescence", "deliquescence") if k in d}
        case = cls(mode=mode, engine=engine, T_K=T, organics=organics, ions=ions, feed=feed, water=water,
                   salts=salts_mol, rh=rh, gas=gas, solver=solver, options=options, output=dict(d.get("output", {})),
                   title=str(d.get("title", "")), spec=d)
        case._validate()
        return case

    def _validate(self):
        m, e = self.mode, self.engine
        needs_rh = m in ("equilibrium", "sle", "gas_particle", "drying_path", "stability") or (m == "lle" and e == "pe")
        if needs_rh:
            if not self.rh:
                raise CaseError(f"mode {m!r} needs [calculation] rh (a number or a list)")
            for r in self.rh:
                if not 0.0 < r < 1.0:
                    raise CaseError("rh values must be in (0, 1)")
        if m in ("sle", "deliquescence", "efflorescence") or e == "sle":
            if self.organics:
                raise CaseError(f"mode {m!r} with engine 'sle' is for inorganic systems only (no organics)")
        if m == "activity" and not self.water:
            raise CaseError("mode 'activity' needs the amount of water: [feed] Water = <mol>")
        if m == "lle" and e == "pep":
            if not self.water:
                raise CaseError("engine 'pep' needs [feed] Water (fixed total composition)")
            if any(k in self.feed for k in self.ions) and not self.salts:
                raise CaseError("engine 'pep' takes electrolytes as [salts] (undissociated components)")
        if m == "gas_particle" and "gas_particle" not in self.options:
            raise CaseError("mode 'gas_particle' needs a [gas_particle] section")
        if m == "drying_path" and "drying" not in self.options:
            raise CaseError("mode 'drying_path' needs a [drying] section (ln_s_crit or erh)")
        if self.mode not in ("activity", "gas_particle") and not self.feed:
            raise CaseError("the feed is empty")

    # --------------------------------------------------------------------------------------------------
    def phase_equilibrium(self):
        """The configured :class:`aiomfac_py.phase_equilibrium.PhaseEquilibrium` of this case."""
        from .phase_equilibrium import PhaseEquilibrium
        s = self.solver
        pe = PhaseEquilibrium(self.organics, self.ions, self.T_K, k_mode=s.get("k_mode"),
                              speciation=s.get("speciation", "explicit"))
        for k, kind in _PE_ATTRS.items():
            if k in s:
                v = s[k]
                if isinstance(kind, tuple):
                    if v not in kind:
                        raise CaseError(f"[solver] {k} must be one of {kind}")
                else:
                    v = None if v is None else kind(v)
                setattr(pe, k, v)
        return pe

    def sle_solver(self):
        from .sle import SLESolver
        solids = self.solver.get("solids", "all")
        keys = list(solids) if isinstance(solids, (list, tuple)) else None
        if self.solver.get("k_mode") not in (None, "fitted", "anchored", "thermo"):
            raise CaseError("[solver] k_mode must be 'fitted', 'anchored' or 'thermo'")
        return SLESolver(self.ions, keys, mode=self.solver.get("k_mode"))

    def gas_kwargs(self) -> dict:
        g = self.gas
        if not g:
            return {}
        if g.get("mode", "open") == "open":
            return {"p_gas": {k: float(v) for k, v in g.get("partial_pressure_atm", {}).items()}}
        return {"gas_total": {k: float(v) for k, v in g.get("total_mol", {}).items()}, "n_air": float(g["n_air"]),
                "P_atm": float(g.get("P_atm", 1.0))}


# ============================================================================================================
# Results
# ============================================================================================================

@dataclass
class PointResult:
    """Result of one calculation point (one RH, or one value of a single-number mode), in a solver-independent form.

    ``liquids``: one dict per liquid phase with ``total_mol``, ``water_activity``, ``amounts`` [mol],
    ``mole_fractions``, ``ln_a`` (neutrals on the mole-fraction scale, ions on the molal scale) and ``molality``
    [mol/kg of water + organics] of the ions.  ``values``: mode-specific numbers (e.g. ``drh``, ``erh``, ``stable``).
    ``raw``: the solver's own result object (not serialized)."""
    mode: str
    engine: str
    rh: float | None
    status: str
    message: str = ""
    liquids: list = field(default_factory=list)
    solids: dict = field(default_factory=dict)
    si: dict = field(default_factory=dict)
    gas: dict = field(default_factory=dict)
    p_gas: dict = field(default_factory=dict)
    gibbs: float | None = None
    tpd_min: float | None = None
    checks: dict = field(default_factory=dict)
    values: dict = field(default_factory=dict)
    raw: Any = None

    @property
    def n_liquids(self) -> int:
        return len(self.liquids)

    @property
    def ok(self) -> bool:
        return self.status in ("converged", "dry")

    def to_dict(self) -> dict:
        d = {k: getattr(self, k) for k in ("mode", "engine", "rh", "status", "message", "liquids", "solids", "si",
                                            "gas", "p_gas", "gibbs", "tpd_min", "checks", "values")}
        d["n_liquids"] = self.n_liquids
        return _jsonable(d)

    def summary(self) -> str:
        head = f"[{self.mode}/{self.engine}]" + (f" RH={self.rh:.4f}" if self.rh is not None else "")
        lines = [f"{head}: {self.status}" + (f" ({self.message})" if self.message else "")]
        if self.values:
            lines.append("  " + ", ".join(f"{k}={_fmt(v)}" for k, v in self.values.items()))
        if self.mode not in ("deliquescence", "efflorescence"):
            if self.solids:
                lines.append("  solids [mol]: " + ", ".join(f"{k}={v:.6g}" for k, v in self.solids.items()))
            if self.gas:
                lines.append("  gas [mol]: " + ", ".join(f"{k}={v:.6g}" for k, v in self.gas.items())
                             + ("; p [atm]: " + ", ".join(f"{k}={v:.4g}" for k, v in self.p_gas.items())
                                if self.p_gas else ""))
            for k, L in enumerate(self.liquids):
                xs = ", ".join(f"{n}={v:.4g}" for n, v in L["mole_fractions"].items())
                aw = L.get("water_activity")
                size = (f"n={L['total_mol']:.6g} mol" if L.get("total_mol") is not None
                        else f"phase fraction={L.get('phase_fraction', float('nan')):.4g}")
                lines.append(f"  liquid {k}: {size}" + (f", a_w={aw:.5f}" if aw is not None else "") + f"; x: {xs}")
                if "activity_coefficient" in L:
                    lines.append("    activity: " + ", ".join(f"{n}={v:.5g}" for n, v in L["activity"].items()))
                    lines.append("    activity coefficient: " + ", ".join(
                        f"{n}={v:.5g}" for n, v in L["activity_coefficient"].items() if v is not None))
            if self.tpd_min is not None:
                lines.append(f"  TPD_min={self.tpd_min:.3g}" + (f", F={self.gibbs:.10g}" if self.gibbs is not None else ""))
        return "\n".join(lines)


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return [_jsonable(v) for v in o.tolist()]
    if isinstance(o, (np.floating, float)):
        v = float(o)
        return v if math.isfinite(v) else None
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


@dataclass
class RunResult:
    case: Case
    points: list

    def __iter__(self):
        return iter(self.points)

    def __len__(self):
        return len(self.points)

    def __getitem__(self, i):
        return self.points[i]

    def to_dict(self) -> dict:
        return {"aiomfac_py_version": __version__, "title": self.case.title, "mode": self.case.mode,
                "engine": self.case.engine, "T_K": self.case.T_K, "case": _jsonable(self.case.spec),
                "points": [p.to_dict() for p in self.points]}

    def to_json(self, path: str | Path | None = None, indent: int = 1) -> str:
        s = json.dumps(self.to_dict(), indent=indent)
        if path is not None:
            Path(path).write_text(s)
        return s

    def rows(self) -> list:
        """One flat dict per point (the CSV rows)."""
        rows = []
        for p in self.points:
            r = {"mode": p.mode, "engine": p.engine, "rh": p.rh, "status": p.status, "n_liquids": p.n_liquids,
                 "solids": ";".join(f"{k}={v:.6g}" for k, v in p.solids.items()), "gibbs": p.gibbs,
                 "tpd_min": p.tpd_min}
            for k, v in p.values.items():
                r[k] = v if not isinstance(v, (list, dict)) else json.dumps(_jsonable(v))
            for k, v in p.gas.items():
                r[f"gas_{k}_mol"] = v
            for k, L in enumerate(p.liquids):
                r[f"L{k}_total_mol"] = L["total_mol"]
                r[f"L{k}_water_activity"] = L.get("water_activity")
                for n, x in L["mole_fractions"].items():
                    r[f"L{k}_x_{n}"] = x
            rows.append(_jsonable(r))
        return rows

    def to_csv(self, path: str | Path | None = None) -> str:
        rows = self.rows()
        cols: list = []
        for r in rows:
            for k in r:
                if k not in cols:
                    cols.append(k)
        buf = _io.StringIO()
        w = csv.DictWriter(buf, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow(r)
        s = buf.getvalue()
        if path is not None:
            Path(path).write_text(s)
        return s

    def summary(self) -> str:
        c = self.case
        head = f"{c.title + ': ' if c.title else ''}mode={c.mode}, engine={c.engine}, T={c.T_K} K"
        return "\n".join([head] + [p.summary() for p in self.points])


# ------------------------------------------------------------------------------------------------------------
# conversion of the solvers' results
# ------------------------------------------------------------------------------------------------------------

def _liquid_dict(names, amounts, ln_a, n_neutral, mm_neutral_kg) -> dict:
    amounts = np.asarray(amounts, dtype=float)
    tot = float(np.sum(amounts))
    solv = float(np.dot(amounts[:n_neutral], mm_neutral_kg))
    d = {"total_mol": tot,
         "water_activity": float(math.exp(ln_a[0])) if ln_a is not None and np.isfinite(ln_a[0]) else None,
         "amounts": {n: float(v) for n, v in zip(names, amounts)},
         "mole_fractions": {n: float(v / tot) for n, v in zip(names, amounts)} if tot > 0 else {},
         "ln_a": {n: float(v) for n, v in zip(names, ln_a)} if ln_a is not None else {},
         "molality": {n: float(v / solv) for n, v in zip(names[n_neutral:], amounts[n_neutral:])} if solv > 0 else {}}
    return d


def _from_pe(r, mode, engine, pe, values=None) -> PointResult:
    nn = 1 + len(pe._organics)
    mm = np.asarray(pe.lm._mm if hasattr(pe.lm, "_mm") else np.zeros(nn), dtype=float)[:nn]
    liquids = [_liquid_dict(L.names, L.amounts, L.ln_a, nn, mm) for L in r.liquids]
    status = r.status
    return PointResult(mode, engine, r.rh, status, r.message, liquids, dict(r.solids), dict(r.si), dict(r.gas),
                       dict(r.p_gas), float(r.gibbs), float(r.tpd_min), dict(r.checks), dict(values or {}), r)


def _from_sle(r, mode, engine, values=None) -> PointResult:
    from .sle import MW_WATER
    liquids = []
    if r.water_kg > 0.0:
        names = ["Water"] + list(r.aq_ions)
        amounts = [r.water_kg / MW_WATER] + [r.aq_ions[i] for i in r.aq_ions]
        ln_a = [math.log(r.rh)] + [r.ln_a.get(i, float("nan")) for i in r.aq_ions]
        liquids = [_liquid_dict(names, amounts, ln_a, 1, np.array([MW_WATER]))]
    status = {"dry": "dry", "failed": "failed"}.get(r.status, "converged")
    return PointResult(mode, engine, r.rh, status, r.message or r.status, liquids, dict(r.solids), dict(r.si),
                       dict(r.gas), dict(r.p_gas), float(r.gibbs), None, {}, dict(values or {}), r)


# ============================================================================================================
# Running a case
# ============================================================================================================

def run(case: "Case | dict | str | Path", verbose: bool = False) -> RunResult:
    """Run a case (a :class:`Case`, a dict in the case format, or the path of a case file)."""
    if isinstance(case, (str, Path)):
        case = Case.from_file(case)
    elif isinstance(case, dict):
        case = Case.from_dict(case)
    fn = {"activity": _run_activity, "equilibrium": _run_equilibrium, "lle": _run_lle, "sle": _run_sle,
          "gas_particle": _run_gas_particle, "drying_path": _run_drying, "deliquescence": _run_deliquescence,
          "efflorescence": _run_efflorescence, "stability": _run_stability}[case.mode]
    return RunResult(case, fn(case, verbose))


def _pe_scan(case: Case, verbose: bool, *, solids, max_liquids=None, values_fn=None) -> list:
    pe = case.phase_equilibrium()
    s = case.solver
    kw = dict(solids=solids, max_liquids=int(max_liquids if max_liquids is not None else s.get("max_liquids", 3)),
              max_outer=int(s.get("max_outer", 8)), verbose=verbose, **case.gas_kwargs())
    warm = s.get("path", "warm") == "warm"
    out, prev = [], None
    for rh in case.rh:
        r = pe.solve(case.feed, rh, init=prev if warm else None, **kw)
        prev = r if r.liquids else None
        out.append(_from_pe(r, case.mode, case.engine, pe, values_fn(r) if values_fn else None))
    return out


def _solids_option(case: Case, default="all"):
    s = case.solver.get("solids", default)
    if isinstance(s, (list, tuple)):
        return list(s)
    if s not in ("all", "none"):
        raise CaseError("[solver] solids must be 'all', 'none' or a list of solid keys")
    return s


def _sle_scan(case: Case) -> list:
    sol = case.sle_solver()
    warm = case.solver.get("path", "warm") == "warm"
    gk = case.gas_kwargs()
    metastable = case.solver.get("solids", "all") == "none"
    if metastable and gk:
        raise CaseError("engine 'sle' with solids = 'none' (metastable aqueous solution) does not take gases; "
                        "use engine 'pe'")
    out = []
    for rh in case.rh:
        if metastable:                                     # supersaturated aqueous solution, every solid suppressed
            r = sol.aqueous_state(case.feed, case.T_K, rh)
        elif "gas_total" in gk:
            r = sol.solve_closed(case.feed, gk["gas_total"], case.T_K, rh, n_air=gk["n_air"], P_atm=gk["P_atm"])
        else:
            r = sol.solve(case.feed, case.T_K, rh, p_gas=gk.get("p_gas"), warm=warm and bool(out))
        out.append(_from_sle(r, case.mode, case.engine))
    return out


def _run_equilibrium(case: Case, verbose: bool) -> list:
    if case.engine == "sle":
        return _sle_scan(case)
    return _pe_scan(case, verbose, solids=_solids_option(case, "all"))


def _run_sle(case: Case, verbose: bool) -> list:
    if case.engine == "pe":
        return _pe_scan(case, verbose, solids=_solids_option(case, "all"))
    return _sle_scan(case)


def _run_lle(case: Case, verbose: bool) -> list:
    if case.engine == "pe":
        if case.solver.get("solids", "none") != "none":
            raise CaseError("mode 'lle' excludes solids; use mode 'equilibrium' for solids")
        return _pe_scan(case, verbose, solids="none")
    # engine "pep": fixed total composition (water closed), lle.solve_pep
    from .lle import solve_pep
    from .solids import ION_REGISTRY
    water = Component(1, "Water", ((16, 1),))
    comps = [water] + [Component(k + 2, c.name, tuple(c.subgroups)) for k, c in enumerate(case.organics)]
    names = ["Water"] + [c.name for c in case.organics]
    amounts = [case.water] + [case.feed.get(c.name, 0.0) for c in case.organics]
    for formula, n in case.salts.items():
        ions = salt_ions(formula)
        if len(ions) != 2:
            raise CaseError(f"engine 'pep' needs binary salts, got {formula!r}")
        (ci, cn), (ai, an) = sorted(ions.items(), key=lambda kv: -ION_REGISTRY[kv[0]][1])
        comps.append(Component(len(comps) + 1, formula, ((ION_REGISTRY[ci][0], cn), (ION_REGISTRY[ai][0], an))))
        names.append(formula)
        amounts.append(n)
    b = np.array(amounts, dtype=float)
    b = b / b.sum()
    s = case.solver
    r = solve_pep(comps, b, case.T_K, eps_init=float(s.get("eps_init", 0.03)), tol=float(s.get("tol", 1e-9)),
                  verbose=verbose)
    from .model import ActivityModel
    model = ActivityModel(comps)
    from .gp_partition import distinct_phases
    liquids = []
    for i in distinct_phases(r):
        y, x = r.y[i], r.x[i]
        a = model.evaluate(np.asarray(x, dtype=float), case.T_K, basis="mole").activity
        liquids.append({"phase_fraction": float(y), "total_mol": float(y * sum(amounts)),
                        "water_activity": float(a[0]),
                        "mole_fractions": {n: float(v) for n, v in zip(names, x)},
                        "activity": {n: float(v) for n, v in zip(names, a)}})
    status = "converged" if r.converged else "not_converged"
    return [PointResult(case.mode, case.engine, None, status, "fixed total composition (water closed)", liquids,
                        values={"n_outer_iter": int(r.n_outer_iter)}, raw=r)]


def _run_stability(case: Case, verbose: bool) -> list:
    pe_tol = float(case.solver.get("tol_tpd", 1e-7))

    def values(r):
        return {"stable": bool(r.tpd_min > -pe_tol), "tpd_min": float(r.tpd_min)}
    out = _pe_scan(case, verbose, solids=_solids_option(case, "none"), max_liquids=1, values_fn=values)
    for p in out:
        # a negative TPD is the answer here, not a failure: the one-liquid state itself must be converged
        if p.status == "not_converged" and "violated" not in p.message and "inner" not in p.message:
            p.status = "converged"
            p.message = "one-liquid state " + ("stable" if p.values["stable"] else "unstable: it splits into "
                                                "several liquids (run mode 'lle' or 'equilibrium')")
    return out


def _ln_s_crit(case: Case, section: dict):
    from .sle import implied_ln_s_crit
    if "ln_s_crit" in section and "erh" in section:
        raise CaseError("give either ln_s_crit or erh, not both")
    if "ln_s_crit" in section:
        v = section["ln_s_crit"]
        return {k: float(x) for k, x in v.items()} if isinstance(v, dict) else float(v)
    if "erh" in section:
        v = section["erh"]
        if not isinstance(v, dict):
            raise CaseError("erh must be a table {solid_key = efflorescence RH}")
        return {k: implied_ln_s_crit(k, case.T_K, float(x), case.solver.get("k_mode")) for k, x in v.items()}
    raise CaseError("the critical supersaturation needs ln_s_crit (number or table) or erh (table)")


def _run_drying(case: Case, verbose: bool) -> list:
    pe = case.phase_equilibrium()
    sec = case.options["drying"]
    lsc = _ln_s_crit(case, sec)
    if isinstance(lsc, dict):
        lsc = dict(lsc)
        for k in sec.get("never", []):                      # solids that are never allowed to crystallize
            lsc[k] = 99.0
    warm = case.solver.get("path", "warm") == "warm"
    path = pe.drying_path(case.feed, case.rh, ln_s_crit=lsc, verbose=verbose, warm=warm, **case.gas_kwargs())
    return [_from_pe(r, case.mode, case.engine, pe, {"enabled_solids": r.message.replace("enabled solids: ", "")})
            for r in path]


def _run_deliquescence(case: Case, verbose: bool) -> list:
    sol = case.sle_solver()
    sec = case.options.get("deliquescence", {})
    lo, hi = float(sec.get("lo", 0.05)), float(sec.get("hi", 0.995))
    tol = float(case.solver.get("rh_tol", 2e-4))
    drh = sol.deliquescence_rh(case.feed, case.T_K, lo, hi, tol)
    full = sol.full_dissolution_rh(case.feed, case.T_K, lo, hi, tol)
    return [PointResult(case.mode, case.engine, None, "converged", "bisection on the phase assemblage",
                        values={"drh": drh, "full_dissolution_rh": full})]


def _run_efflorescence(case: Case, verbose: bool) -> list:
    sol = case.sle_solver()
    sec = case.options.get("efflorescence", {})
    lsc = _ln_s_crit(case, sec) if sec else 0.5
    lo, hi = float(sec.get("lo", 0.05)), float(sec.get("hi", 0.995))
    r = sol.efflorescence_rh(case.feed, case.T_K, lsc, lo, hi, float(case.solver.get("rh_tol", 2e-4)))
    status = "converged" if r["salt"] is not None else "not_converged"
    msg = "" if r["salt"] is not None else "no salt reaches its critical supersaturation in the RH range"
    return [PointResult(case.mode, case.engine, None, status, msg, values={"erh": r["rh"], "salt": r["salt"]})]


def _run_activity(case: Case, verbose: bool) -> list:
    from .phase_equilibrium import LiquidModel
    lm = LiquidModel(case.organics, case.ions)
    n = np.array([case.water] + [case.feed.get(c.name, 0.0) for c in case.organics]
                 + [case.feed.get(i, 0.0) for i in case.ions], dtype=float)
    la = lm.ln_a(n, case.T_K)
    nn = lm.n_neutral
    L = _liquid_dict(lm.names, n, la, nn, lm._mm)
    # activity coefficients on AIOMFAC's scales: neutrals a / x with x the mole fraction of the dissociated-ion basis
    # (AIOMFAC's X), ions a / m with m the molality per kg of water + organics
    solv = float(np.dot(n[:nn], lm._mm))
    smc = np.zeros(lm._ngi)
    sma = np.zeros(lm._ngi)
    for (is_cat, idx), v in zip(lm._pos, n[nn:]):
        (smc if is_cat else sma)[idx] = v / solv
    x_sp = lm.model._x_from_molalities(n[:nn] / n[:nn].sum(), smc, sma)
    gam = {}
    for k, name in enumerate(lm.names):
        if k < nn:
            gam[name] = float(math.exp(la[k]) / x_sp[k])
        else:
            m = n[k] / solv
            gam[name] = float(math.exp(la[k]) / m) if m > 0 else None
    L["activity"] = {nm: float(math.exp(v)) for nm, v in zip(lm.names, la)}
    L["activity_coefficient"] = gam
    return [PointResult(case.mode, case.engine, None, "converged",
                        "neutrals: mole-fraction scale (dissociated-ion basis); ions: molal scale", [L])]


def _run_gas_particle(case: Case, verbose: bool) -> list:
    from .gp_partition import VolatileSpecies, distinct_phases, gp_partition
    from .solids import ION_REGISTRY
    sec = case.options["gas_particle"]
    salt = sec.get("salt")
    if salt is None:
        raise CaseError("[gas_particle] needs salt (a formula of 'aiomfac-tool list salts')")
    ions = salt_ions(salt)
    if len(ions) != 2:
        raise CaseError("[gas_particle] salt must be a binary salt")
    (ci, cn), (ai, an) = sorted(ions.items(), key=lambda kv: -ION_REGISTRY[kv[0]][1])
    species_specs = sec.get("species", [])
    if not species_specs:
        raise CaseError("[gas_particle] needs at least one [[gas_particle.species]]")
    species = []
    for k, sp in enumerate(species_specs):
        c = _organic(sp, k + 2)
        M = float(sp["M_kg_mol"]) if "M_kg_mol" in sp else organic_molar_mass_g(c) * 1e-3
        for req in ("p0_Pa", "n_total_mol"):
            if req not in sp:
                raise CaseError(f"species {c.name!r} needs {req}")
        species.append(VolatileSpecies(c, M, float(sp["p0_Pa"]), float(sp["n_total_mol"])))
    salt_c = Component(len(species) + 2, salt, ((ION_REGISTRY[ci][0], cn), (ION_REGISTRY[ai][0], an)))
    s = case.solver
    out = []
    for rh in case.rh:
        r = gp_partition(salt_c, species, float(sec.get("n_salt_mol", 0.0)), case.T_K, rh, float(sec["V_gas_m3"]),
                         method=str(s.get("gp_method", "lm")), max_iter=int(s.get("gp_max_iter", 60)),
                         tol=float(s.get("gp_tol", 1e-6)), check_lle=bool(s.get("check_lle", True)))
        names = ["Water"] + [sp.component.name for sp in species] + [salt]
        keep = distinct_phases(r.phases)                     # real, non-negligible, mutually distinct phases
        liquids = [{"phase_fraction": float(r.phases.y[i]), "total_mol": None, "water_activity": None,
                    "mole_fractions": {n: float(v) for n, v in zip(names, r.phases.x[i])}} for i in keep]
        values = {"water_activity": float(r.aw), "n_water_PM_mol": float(r.n_water_PM), "n_iter": int(r.n_iter)}
        for j, sp in enumerate(species):
            nm = sp.component.name
            values[f"{nm}_particle_mol"] = float(r.n_org_PM[j])
            values[f"{nm}_gas_mol"] = float(r.n_org_gas[j])
            values[f"{nm}_particle_fraction"] = float(r.n_org_PM[j] / sp.n_total) if sp.n_total > 0 else None
            values[f"{nm}_activity"] = float(r.activities_org[j])
        out.append(PointResult(case.mode, case.engine, rh, "converged" if r.converged else "not_converged",
                               "particle phases from the one-shot LLE diagnostic (lle.solve_pep)", liquids,
                               values=values, raw=r))
    return out


# ============================================================================================================
# Templates (also used by the manual and the tests)
# ============================================================================================================

TEMPLATES = {
    "equilibrium": '''title = "pinic acid + ammonium sulfate + ammonium nitrate, liquid-liquid-solid equilibrium"

[calculation]
mode = "equilibrium"
rh = [0.8, 0.6, 0.4, 0.2]          # one point per RH; warm-started along the list

[system]
T_K = 300.0

[[system.organics]]
name = "pinic_acid"
subgroups = [[1, 2], [2, 2], [3, 2], [4, 1], [137, 2]]   # (subgroup, count); or smiles = "..."

[feed_mass_g]                      # or [feed] in mol, [salts] in mol of formula units
pinic_acid = 1.15

[salts_mass_g]
"(NH4)2SO4" = 0.778
NH4NO3 = 0.222

[solver]
engine = "pe"                      # PhaseEquilibrium (combined liquid-liquid-solid solver)
solids = "all"                     # "all", "none" or a list of solid keys
max_liquids = 3
inner_method = "newton"            # "newton" (default), "rand", "barrier"
hess_scheme = "split"              # "split" (default), "ad" (needs jax), "central"
''',
    "lle": '''title = "pinic acid + ammonium sulfate at RH 0.3: liquid-liquid phase separation (no crystallization)"

[calculation]
mode = "lle"
rh = 0.3

[system]
T_K = 298.15

[[system.organics]]
name = "pinic_acid"
subgroups = [[1, 2], [2, 2], [3, 2], [4, 1], [137, 2]]

[feed]
pinic_acid = 0.006243
"NH4+" = 0.015136
"SO4--" = 0.007568

[solver]
engine = "pe"
''',
    "lle_pep": '''title = "water + pinic acid + ammonium sulfate at fixed total composition (UHAERO-type interior point)"

[calculation]
mode = "lle"

[system]
T_K = 298.15

[[system.organics]]
name = "pinic_acid"
subgroups = [[1, 2], [2, 2], [3, 2], [4, 1], [137, 2]]

[feed]                             # mol; water is closed (fixed amount)
Water = 0.6
pinic_acid = 0.3

[salts]
"(NH4)2SO4" = 0.1

[solver]
engine = "pep"
''',
    "sle": '''title = "NaCl + (NH4)2SO4: solid-liquid equilibrium along RH"

[calculation]
mode = "sle"
rh = [0.9, 0.8, 0.7, 0.6, 0.5]

[system]
T_K = 298.15

[salts]
NaCl = 1.0
"(NH4)2SO4" = 1.0

[solver]
engine = "sle"                     # SLESolver (inorganic active-set solver); "pe" also works
''',
    "sle_gas": '''title = "NH4NO3 particle in a closed volume of air with HNO3 and NH3"

[calculation]
mode = "sle"
rh = [0.8, 0.5]

[system]
T_K = 298.15
ions = ["NH4+", "NO3-", "H+"]

[salts]
NH4NO3 = 1.0e-6

[gas]
mode = "closed"                    # "open": [gas] partial_pressure_atm = {HNO3 = 1e-9, ...}
total_mol = {HNO3 = 1.0e-7, NH3 = 1.0e-7}
n_air = 41.0                       # mol of air (about 1 m3 at 1 atm, 298 K)
P_atm = 1.0
''',
    "gas_particle": '''title = "glycerol partitioning over ammonium sulfate seed (Zuend et al., 2010, method)"

[calculation]
mode = "gas_particle"
rh = [0.3, 0.6, 0.9]

[system]
T_K = 298.15

[gas_particle]
salt = "(NH4)2SO4"
n_salt_mol = 1.0e-8
V_gas_m3 = 1.0

[[gas_particle.species]]
name = "glycerol"
subgroups = [[150, 2], [151, 1], [153, 3]]
M_kg_mol = 0.092094                # optional; computed from the subgroups if omitted
p0_Pa = 2.284e-2
n_total_mol = 3.0e-8

[solver]
engine = "gp"
gp_method = "lm"                   # "lm" (default), "pseudo_transient", "successive_substitution"
gp_tol = 1.0e-4
''',
    "drying_path": '''title = "Ansan-type drying path: AS crystallizes after its critical supersaturation"

[calculation]
mode = "drying_path"
rh = [0.8, 0.6, 0.45, 0.35, 0.3, 0.25, 0.2, 0.1]

[system]
T_K = 300.0

[[system.organics]]
name = "pinic_acid"
subgroups = [[1, 2], [2, 2], [3, 2], [4, 1], [137, 2]]

[feed_mass_g]
pinic_acid = 1.15

[salts_mass_g]
"(NH4)2SO4" = 0.778
NH4NO3 = 0.222

[drying]
erh = {ammonium_sulfate = 0.35}    # ln S_crit implied by the efflorescence RH of the binary salt
never = ["ammonium_nitrate"]       # solids that never crystallize on this path
''',
    "deliquescence": '''title = "deliquescence of ammonium sulfate"

[calculation]
mode = "deliquescence"

[system]
T_K = 298.15

[salts]
"(NH4)2SO4" = 1.0
''',
    "efflorescence": '''title = "efflorescence of ammonium sulfate for ln S_crit implied by ERH 0.35"

[calculation]
mode = "efflorescence"

[system]
T_K = 298.15

[salts]
"(NH4)2SO4" = 1.0

[efflorescence]
erh = {ammonium_sulfate = 0.35}
''',
    "stability": '''title = "stability of the one-liquid state of pinic acid + ammonium sulfate"

[calculation]
mode = "stability"
rh = [0.995, 0.99, 0.98, 0.95, 0.9, 0.6]

[system]
T_K = 298.15

[[system.organics]]
name = "pinic_acid"
subgroups = [[1, 2], [2, 2], [3, 2], [4, 1], [137, 2]]

[feed]
pinic_acid = 0.006243
"NH4+" = 0.015136
"SO4--" = 0.007568
''',
    "activity": '''title = "activities of water + pinic acid + ammonium sulfate"

[calculation]
mode = "activity"

[system]
T_K = 298.15

[[system.organics]]
name = "pinic_acid"
subgroups = [[1, 2], [2, 2], [3, 2], [4, 1], [137, 2]]

[feed]                             # mol
Water = 1.0
pinic_acid = 0.05

[salts]
"(NH4)2SO4" = 0.02
''',
}


# ============================================================================================================
# Command line
# ============================================================================================================

def _cmd_list(what: str) -> str:
    if what == "modes":
        return "\n".join(f"{m:14s} engines: {', '.join(MODES[m]):10s} {MODE_HELP[m]}" for m in MODES)
    if what == "solids":
        from .solids import SOLIDS
        return "\n".join(f"{k:22s} {s.formula:28s} ions={s.ions} h={s.h}" for k, s in SOLIDS.items())
    if what == "ions":
        from .solids import ION_REGISTRY
        return "\n".join(f"{k:8s} AIOMFAC id {sid}, charge {z:+d}, M = {ion_molar_mass_g(k):.4f} g/mol"
                         for k, (sid, z) in ION_REGISTRY.items())
    if what == "gases":
        from .gases import GASES
        return "\n".join(f"{k:6s} reaction ions={g.ions} (gas forms from these; quality {g.quality})"
                         for k, g in GASES.items())
    if what == "salts":
        from .sle import SALT_IONS
        return "\n".join(f"{k:16s} {v}  M = {salt_molar_mass_g(k):.3f} g/mol" for k, v in SALT_IONS.items())
    if what == "templates":
        return "\n".join(TEMPLATES)
    raise CaseError(f"unknown list {what!r}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="aiomfac-tool", description="aiomfac_py main tool: equilibrium calculations "
                                "from a case file (see docs/user_manual.md)")
    sub = p.add_subparsers(dest="cmd", required=True)
    pr = sub.add_parser("run", help="run a case file (.toml or .json)")
    pr.add_argument("case")
    pr.add_argument("-o", "--output", help="output file (default: [output] file of the case, or stdout)")
    pr.add_argument("-f", "--format", choices=("json", "csv", "text"), help="output format (default: from the "
                    "output file's extension, else text)")
    pr.add_argument("-v", "--verbose", action="store_true")
    pt = sub.add_parser("template", help="print an example case file")
    pt.add_argument("name", choices=sorted(TEMPLATES))
    pl = sub.add_parser("list", help="list modes, solids, ions, gases, salts or templates")
    pl.add_argument("what", choices=("modes", "solids", "ions", "gases", "salts", "templates"))
    ps = sub.add_parser("smiles", help="AIOMFAC subgroups of a SMILES string (needs epam.indigo)")
    ps.add_argument("smiles")
    ps.add_argument("name", nargs="?", default="organic")
    a = p.parse_args(argv)
    try:
        if a.cmd == "run":
            case = Case.from_file(a.case)
            res = run(case, verbose=a.verbose)
            outfile = a.output or case.output.get("file")
            fmt = a.format or case.output.get("format")
            if fmt is None:
                fmt = {".json": "json", ".csv": "csv"}.get(Path(outfile).suffix.lower(), "text") if outfile else "text"
            text = {"json": res.to_json, "csv": res.to_csv, "text": res.summary}[fmt]()
            if outfile:
                Path(outfile).write_text(text if text.endswith("\n") else text + "\n")
                print(f"wrote {outfile} ({len(res)} point(s))", file=sys.stderr)
            else:
                print(text)
            return 0 if all(pt_.status in ("converged", "dry") for pt_ in res) else 2
        if a.cmd == "template":
            print(TEMPLATES[a.name], end="")
            return 0
        if a.cmd == "list":
            print(_cmd_list(a.what))
            return 0
        if a.cmd == "smiles":
            c = _organic({"name": a.name, "smiles": a.smiles}, 2)
            print(f'[[system.organics]]\nname = "{c.name}"\nsubgroups = {[list(t) for t in c.subgroups]}\n'
                  f"# M = {organic_molar_mass_g(c):.4f} g/mol")
            return 0
    except CaseError as e:
        print(f"aiomfac-tool: error: {e}", file=sys.stderr)
        return 1
    return 1


if __name__ == "__main__":
    sys.exit(main())

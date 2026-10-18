"""Composition--RH phase diagrams from a configuration file: map, boundary lines, topology check, plot.

    aiomfac-diagram run diagram.json [--steps map,lines,plot] [--workers N] [--force map,lines]
    aiomfac-diagram example > diagram.json

The configuration is an aiomfac-tool case file (TOML with Python >= 3.11 or the ``tomli`` package, or JSON; see
docs/user_manual.md) without ``[calculation]``: ``[system]`` and ``[solver]`` as there, the feeds at the two ends of
the composition axis in ``[feed_x0]`` and ``[feed_x1]`` (each with the feed tables of a case file: ``feed`` in mol,
``feed_mass_g``, ``salts``, ``salts_mass_g``; the feed is linear in between), and the options of the steps::

    name = "AS-AN with pinic acid"
    output = "out"                         # directory, relative to this file
    workers = 2                            # processes for the map and the lines

    [system]
    T_K = 298.15
    [[system.organics]]
    name = "pinic_acid"
    smiles = "CC1(C)C(CC1CC(=O)O)C(=O)O"   # or subgroups = [[1, 2], ...]

    [solver]
    inner_method = "rand"
    solids = "all"

    [feed_x0.salts]
    "(NH4)2SO4" = 1.0
    [feed_x1.salts]
    NH4NO3 = 2.0
    [feed_x0.feed]
    pinic_acid = 0.3
    [feed_x1.feed]
    pinic_acid = 0.3

    [axis]
    label = "x"
    x = {start = 0.0, stop = 1.0, num = 11}   # or a list
    [map]                                  # keywords of aiomfac_py.diagram.phase_map / trace
    n = 24
    rh_min = 0.02
    rh_max = 0.98
    [lines]                                # keywords of aiomfac_py.diagram.trace_until_clean
    solve_timeout = 20
    rounds = 2
    [plot]
    labels = {ammonium_sulfate = "A", ammonium_nitrate = "D"}

* ``[system] options``: further attributes of the solver (e.g. ``fallback_inner = ["newton"]``).
* Instead of the system and feed tables, ``[system] factory = "module:function"`` (with ``args``) returns the
  PhaseEquilibrium, and ``[feed] factory = "module:function"`` (with ``args``) is the feed as a function of x; the
  directory of the file and the directories ``python_path`` (relative to it) are on the import path
  (tools/configs/uhaero2007_*.json).
* ``[plot]``: ``labels`` (solid key -> letter), ``validity_limit`` (ionic strength drawn as a red dashed line,
  default 100 mol/kg; omit with ``validity_limit = false``), ``show_failed`` (default true), ``title``, ``dpi``,
  ``figsize``, ``min_cells``.

Every step writes its result to the output directory (``map.pkl``, ``lines_cache.pkl``, ``lines.pkl``,
``report.json``, ``diagram.png``) and is skipped when its result exists and is newer than its input, so an
interrupted run continues where it stopped (the solves of the lines are saved every 50 solves); ``--force``
repeats the named steps.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import pickle
import sys
import time

import numpy as np

EXAMPLE = {
    "name": "NaCl-KCl",
    "output": "out_nacl_kcl",
    "system": {"T_K": 298.15},
    "feed_x0": {"salts": {"KCl": 1.0}},
    "feed_x1": {"salts": {"NaCl": 1.0}},
    "axis": {"label": "x(NaCl)", "x": {"start": 0.05, "stop": 0.95, "num": 5}},
    "map": {"n": 16, "rh_min": 0.6, "rh_max": 0.95},
    "lines": {"solve_timeout": 20, "rounds": 2},
    "plot": {"labels": {"halite": "H", "sylvite": "S"}},
    "workers": 1,
}


def load_config(path: str) -> dict:
    """the configuration (TOML or JSON, read as aiomfac-tool case files) with ``_dir``, the directory of the file"""
    from .tool import load_case_file
    cfg = dict(load_case_file(path))
    cfg["_dir"] = os.path.dirname(os.path.abspath(path))
    return cfg


def _import(spec: str, base: str, extra=()):
    for d in [base] + [os.path.join(base, e) for e in extra]:
        d = os.path.abspath(d)
        if d not in sys.path:
            sys.path.insert(0, d)
    mod, _, fn = spec.partition(":")
    return getattr(importlib.import_module(mod), fn)


_FEED_KEYS = ("feed", "feed_mass_g", "salts", "salts_mass_g")


def _endpoint(cfg: dict, which: str) -> dict:
    """the feed sections (aiomfac-tool format) of the end ``which`` ("x0" or "x1") of the composition axis"""
    sec = cfg.get(f"feed_{which}")
    if sec is None:                                          # "feed": {"x0": {...}, "x1": {...}} in mol
        sec = {"feed": cfg.get("feed", {}).get(which, {})}
    if not any(k in sec for k in _FEED_KEYS):                # a plain table of mol
        sec = {"feed": sec}
    return {k: v for k, v in sec.items() if k in _FEED_KEYS}


def build_system(cfg: dict):
    """(PhaseEquilibrium, feed_of_x, solve keywords) of the configuration.  [system] and [solver] are those of
    aiomfac-tool case files; the feeds at the two ends of the axis are [feed_x0] and [feed_x1] (each with the feed
    sections of a case file: feed, feed_mass_g, salts, salts_mass_g), linear in between."""
    from .tool import Case
    system, solver = dict(cfg.get("system", {})), dict(cfg.get("solver", {}))
    base, extra = cfg.get("_dir", "."), cfg.get("python_path", [])
    options = dict(system.pop("options", {}))
    if "factory" in system:
        pe = _import(system["factory"], base, extra)(**system.get("args", {}))
        ends = None
    else:
        system.setdefault("T_K", 298.15)
        dummy = {"calculation": {"mode": "equilibrium", "rh": 0.5}}
        ends = [Case.from_dict({**dummy, "system": system, "solver": solver, **_endpoint(cfg, w)}) for w in ("x0", "x1")]
        ions = list(dict.fromkeys(ends[0].ions + ends[1].ions))
        pe = Case.from_dict({**dummy, "system": {**system, "ions": ions}, "solver": solver,
                             **_endpoint(cfg, "x0")}).phase_equilibrium()
    for k, v in options.items():
        setattr(pe, k, tuple(v) if isinstance(v, list) else v)
    feedc = cfg.get("feed", {})
    if isinstance(feedc, dict) and "factory" in feedc:
        import functools
        feed_of_x = functools.partial(_import(feedc["factory"], base, extra), **feedc.get("args", {}))
    else:
        if ends is None:
            from .tool import Case as _C
            dummy = {"calculation": {"mode": "equilibrium", "rh": 0.5}, "system": {"T_K": 298.15, "ions": []}}
            ends = [_C.from_dict({**dummy, **_endpoint(cfg, w)}) for w in ("x0", "x1")]
        x0, x1 = ends[0].feed, ends[1].feed
        keys = list(dict.fromkeys(list(x0) + list(x1)))

        def feed_of_x(x, x0=x0, x1=x1, keys=keys):
            f = {k: (1.0 - x) * float(x0.get(k, 0.0)) + x * float(x1.get(k, 0.0)) for k in keys}
            return {k: v for k, v in f.items() if v > 1e-14}
    solids = solver.get("solids", system.get("solids"))
    solve_kw = {"solids": solids} if solids is not None else {}
    if "max_liquids" in solver:
        solve_kw["max_liquids"] = int(solver["max_liquids"])
    return pe, feed_of_x, solve_kw


def _x_values(axis: dict) -> np.ndarray:
    x = axis["x"]
    if isinstance(x, dict):
        return np.linspace(float(x["start"]), float(x["stop"]), int(x["num"]))
    return np.asarray(x, dtype=float)


def _save(obj, path):
    with open(path + ".tmp", "wb") as fh:
        pickle.dump(obj, fh)
    os.replace(path + ".tmp", path)


def _fresh(out: str, inp: str | None) -> bool:
    """``out`` exists and is newer than ``inp``"""
    return os.path.exists(out) and (inp is None or not os.path.exists(inp) or os.path.getmtime(out) >= os.path.getmtime(inp))


def run(cfg: dict, steps=("map", "lines", "plot"), workers: int | None = None, force=(), verbose: bool = True) -> dict:
    """Run the steps of the configuration; returns the report (also written to report.json)."""
    from .diagram import check_topology, phase_map, trace_until_clean
    out = os.path.join(cfg.get("_dir", "."), cfg.get("output", "out"))
    os.makedirs(out, exist_ok=True)
    f_map, f_cache, f_lines = (os.path.join(out, n) for n in ("map.pkl", "lines_cache.pkl", "lines.pkl"))
    f_report, f_png = os.path.join(out, "report.json"), os.path.join(out, "diagram.png")
    workers = int(workers if workers is not None else cfg.get("workers", 1))
    pe, feed_of_x, solve_kw = build_system(cfg)
    report = json.load(open(f_report)) if os.path.exists(f_report) else {}
    report["name"] = cfg.get("name", "")

    if "map" in steps:
        if "map" in force or not os.path.exists(f_map):
            t0 = time.time()
            kw = dict(cfg.get("map", {}))
            kw.setdefault("x_label", cfg.get("axis", {}).get("label", "x"))
            pm = phase_map(pe, feed_of_x, _x_values(cfg["axis"]), verbose=verbose, workers=workers, **solve_kw, **kw)
            _save(pm, f_map)
            report["map"] = {"seconds": round(time.time() - t0, 1), "x": [float(v) for v in pm.x],
                             "solves": int(sum(t.n_solves for t in pm.traces)),
                             "states": sorted({s.label for s in pm.states()})}
            if verbose:
                print(f"map: {len(pm.x)} compositions, {report['map']['solves']} solves, "
                      f"{report['map']['seconds']:.0f} s", flush=True)
        elif verbose:
            print("map: up to date", flush=True)
    pm = pickle.load(open(f_map, "rb")) if os.path.exists(f_map) else None

    if "lines" in steps and pm is not None:
        if "lines" in force or not _fresh(f_lines, f_map):
            if "lines" in force and "map" in force and os.path.exists(f_cache):
                os.remove(f_cache)                           # a new map: the solves of the old lines do not apply
            cache = pickle.load(open(f_cache, "rb")) if _fresh(f_cache, f_map) else {}
            t0 = time.time()
            kw = dict(cfg.get("lines", {}))
            try:
                bc = trace_until_clean(pe, feed_of_x, pm, cache=cache, checkpoint=lambda c: _save(c, f_cache),
                                       checkpoint_every=50, verbose=verbose, workers=workers, **solve_kw, **kw)
            finally:
                _save(cache, f_cache)
            _save(bc, f_lines)
            rep = check_topology(bc)
            report["lines"] = {"seconds": round(time.time() - t0, 1), "lines": len(bc.curves),
                               "solves": int(bc.n_solves), "failed": int(len(bc.failed)), "clean": rep.clean,
                               "dangling": len(rep.dangling), "adjoining": len(rep.adjoining),
                               "failed_far": len(rep.failed_far), "reports": list(bc.reports)}
            if verbose:
                print(f"lines: {len(bc.curves)} lines, {len(bc.failed)} failed solves, "
                      f"{report['lines']['seconds']:.0f} s\n{rep.summary()}", flush=True)
        elif verbose:
            print("lines: up to date", flush=True)
    bc = pickle.load(open(f_lines, "rb")) if os.path.exists(f_lines) else None

    if "plot" in steps and pm is not None:
        plot(cfg, pm, bc, f_png, cache_file=f_cache)
        report["plot"] = os.path.basename(f_png)
        if verbose:
            print(f"plot: {f_png}", flush=True)
    with open(f_report, "w") as fh:
        json.dump(report, fh, indent=1)
    return report


def plot(cfg: dict, pm, bc, path: str, cache_file: str | None = None):
    """the phase diagram (regions labelled, traced lines, failed solves, validity limit) as an image"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .diagram import label_regions, plot_phase_map, plot_validity
    pc = cfg.get("plot", {})
    cache = pickle.load(open(cache_file, "rb")) if cache_file and os.path.exists(cache_file) else None
    fig, ax = plt.subplots(figsize=tuple(pc.get("figsize", (8, 6))))
    plot_phase_map(pm, ax=ax, legend=False, colors={s: "#ffffff" for s in pm.states()}, curves=bc,
                   show_failed=bool(pc.get("show_failed", True)))
    lim = pc.get("validity_limit", 100.0)
    if lim not in (None, False):
        try:
            plot_validity(ax, pm, cache=cache, limit=float(lim))
        except Exception:                                    # e.g. no ions: nothing to draw
            pass
    label_regions(ax, pm, pc.get("labels", {}), min_cells=int(pc.get("min_cells", 200)), fontsize=7, curves=bc,
                  cache=cache)
    ax.set_ylabel("RH")
    ax.set_title(pc.get("title", cfg.get("name", "")), fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=int(pc.get("dpi", 130)))
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="aiomfac-diagram", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run the steps of a configuration file")
    r.add_argument("config")
    r.add_argument("--steps", default="map,lines,plot", help="comma-separated: map, lines, plot")
    r.add_argument("--workers", type=int, default=None, help="processes (default: the configuration's, else 1)")
    r.add_argument("--force", default="", help="comma-separated steps to repeat even if up to date")
    r.add_argument("--quiet", action="store_true")
    sub.add_parser("example", help="print an example configuration (NaCl-KCl)")
    a = ap.parse_args(argv)
    if a.cmd == "example":
        print(json.dumps(EXAMPLE, indent=1))
        return 0
    cfg = load_config(a.config)
    rep = run(cfg, steps=tuple(s for s in a.steps.split(",") if s), workers=a.workers,
              force=tuple(s for s in a.force.split(",") if s), verbose=not a.quiet)
    return 0 if rep.get("lines", {}).get("clean", True) else 1


if __name__ == "__main__":
    sys.exit(main())

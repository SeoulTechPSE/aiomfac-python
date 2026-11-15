"""Loader for the ambient PM2.5 viscosity data of Fig. 2a of Do et al. (2026a, ES&T; Ansan, 300 K) and Ullah et al.
(2026, ACP; Seoul and Beijing, 290 K), as provided by the authors (M. Song, Jeonbuk National University, 2026-10-06).

The data file itself is not part of the archive (it is not redistributed without the authors' permission). Place it at
``ambient_data/private/Fig2a_viscosity_data_Do2026.xlsx`` or point the environment variable ``AMBIENT_XLSX`` to it.

Workbook sheets
---------------
* ``Ansan_300K``: date (YY.MM.DD), OIR, RH (%), viscosity mean / lower / upper limit (Pa s), tau_mix_H2O (h), note.
  Each resolved value is the mean of three particles of one sample at one RH; rows with the note ">= 1E8 Pa s" are
  lower bounds (no flow within 3 h).
* ``Seoul_Beijing_290K``: site, sample date, OIR, average RH (%), number of particles, mean / lower / upper (Pa s);
  the means over the particles of one sample at one RH (the measurements of Ullah et al., 2026, Supplement Table S1).
* ``SB_per_particle``: site, sample date, particle number, RH (%), mean / lower / upper (Pa s), note. Fig. 2a shows
  Seoul and Beijing per particle. Four lower bounds (>= 1E8 Pa s) have no confirmed date; their RH was read from the
  figure.

Every record returned by :func:`load` has the keys
``site, T_K, date (MM/DD), oir (None if unknown), rh_pct, log10_eta, log10_lo, log10_hi, censored, level, tau_mix_h``
with ``level`` = "sample mean" (Ansan), "particle" (Seoul/Beijing) or "group mean" (Seoul/Beijing sheet 2).
"""
from __future__ import annotations

import math
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT = os.path.join(_HERE, "private", "Fig2a_viscosity_data_Do2026.xlsx")
T_SITE = {"Ansan": 300.0, "Seoul": 290.0, "Beijing": 290.0}


def _path():
    p = os.environ.get("AMBIENT_XLSX", DEFAULT)
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"ambient viscosity data not found at {p}. The file is provided by the authors of Do et al. (2026a) and "
            "Ullah et al. (2026) on request; set AMBIENT_XLSX to its location.")
    return p


def _lg(v):
    return None if v is None else math.log10(float(v))


def _mmdd(date: str) -> str:
    """'24.08.03' or '2023-09-21' -> '08/03' / '09/21'."""
    d = str(date)
    if "." in d:
        _, m, dd = d.split(".")
    else:
        _, m, dd = d.split("-")
    return f"{int(m):02d}/{int(dd):02d}"


def load(levels=("sample mean", "particle")):
    """Points of Fig. 2a (Ansan sample means, Seoul/Beijing particles) and, on request, the Seoul/Beijing group means."""
    import openpyxl
    wb = openpyxl.load_workbook(_path(), data_only=True, read_only=True)
    out = []
    if "sample mean" in levels:
        for r in list(wb["Ansan_300K"].iter_rows(min_row=2, values_only=True)):
            if not r[0]:
                continue
            cens = r[4] is None
            out.append(dict(site="Ansan", T_K=300.0, date=_mmdd(r[0]), oir=float(r[1]), rh_pct=float(r[2]),
                            log10_eta=_lg(r[3]), log10_lo=_lg(r[4]), log10_hi=_lg(r[5]), censored=cens,
                            level="sample mean", tau_mix_h=None if r[6] is None else float(r[6])))
    oir_sb = {}
    group = []
    for r in list(wb["Seoul_Beijing_290K"].iter_rows(min_row=2, values_only=True)):
        if not r[0]:
            continue
        oir_sb[(r[0], _mmdd(r[1]))] = float(r[2])
        group.append(dict(site=r[0], T_K=290.0, date=_mmdd(r[1]), oir=float(r[2]), rh_pct=float(r[3]),
                          log10_eta=_lg(r[5]), log10_lo=_lg(r[6]), log10_hi=_lg(r[7]), censored=False,
                          level="group mean", n_particles=int(r[4]), tau_mix_h=None))
    if "particle" in levels:
        for r in list(wb["SB_per_particle"].iter_rows(min_row=2, values_only=True)):
            if not r[0]:
                continue
            cens = r[5] is None
            known = not str(r[1]).startswith("(")
            date = _mmdd(r[1]) if known else None
            out.append(dict(site=r[0], T_K=290.0, date=date, oir=oir_sb.get((r[0], date)) if known else None,
                            rh_pct=float(r[3]), log10_eta=_lg(r[4]), log10_lo=_lg(r[5]), log10_hi=_lg(r[6]),
                            censored=cens, level="particle", particle=r[2], tau_mix_h=None))
    if "group mean" in levels:
        out += group
    return out


if __name__ == "__main__":
    pts = load()
    for s in T_SITE:
        sel = [p for p in pts if p["site"] == s]
        print(s, "resolved", sum(not p["censored"] for p in sel), "lower bounds", sum(p["censored"] for p in sel),
              "RH", min(p["rh_pct"] for p in sel if not p["censored"]), "-", max(p["rh_pct"] for p in sel if not p["censored"]))
    print("group means", len(load(levels=("group mean",))))

"""Helpers to read the reference data produced by the instrumented Fortran build."""
import gzip
import re
from pathlib import Path

import numpy as np

REF = Path(__file__).parent / "reference"


def _read_text(path):
    path = Path(path)
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as fh:
            return fh.read()
    return path.read_text()


def parse_dump(path):
    """debug_terms_XXXX.txt[.gz] -> list of dicts (one per composition point)."""
    pts, cur = [], None
    for line in _read_text(path).splitlines():
        p = line.split()
        if not p:
            continue
        if p[0] == "POINT":
            cur = {}; pts.append(cur)
        elif p[0] == "END":
            cur = None
        elif p[0] == "DIMS":
            cur["dims"] = tuple(map(int, p[-3:]))
        elif p[0] == "FLAGS":
            cur["flags"] = tuple(t == "T" for t in p[-3:])   # bisulfsyst, bicarbsyst, solvmixrefnd
        else:
            cur[p[0]] = np.array([float(v) for v in p[1:]])
    return pts


def parse_output_table(path):
    """AIOMFAC_output_XXXX.txt -> {component no.: array of table rows (T, RH, w, x, m, a_coeff_x, a_x, flag)}"""
    txt = Path(path).read_text()
    res = {}
    for m in re.finditer(r"Mixture's component # : (\d+)\n(.*?)\n(?=\s*\n\s*Mixture's component|\Z)", txt, re.S):
        rows = []
        for line in m.group(2).splitlines():
            t = line.split()
            if len(t) >= 8 and re.fullmatch(r"\d{3}", t[0]):
                rows.append([float(x) for x in t[1:]])
        res[int(m.group(1))] = np.array(rows)
    return res


def parse_sr_params_dump():
    """sr_params_dump.txt.gz -> dict with SR_RR, SR_QQ and ARR/BRR/CRR as (76, 76) arrays."""
    cols = {"ARR": {}, "BRR": {}, "CRR": {}}
    out = {}
    with gzip.open(REF / "sr_params_dump.txt.gz", "rt") as fh:
        for line in fh:
            p = line.split()
            if p[0] in ("SR_RR", "SR_QQ", "GroupMW"):
                out[p[0]] = np.array(list(map(float, p[1:])))
            if p[0] in ("NKTAB", "Ioncharge"):
                out[p[0]] = np.array(list(map(int, p[1:])))
            for n in cols:
                if p[0].startswith(n + "col"):
                    cols[n][int(p[0][len(n) + 3:])] = np.array(list(map(float, p[1:])))
    for n in cols:
        out[n] = np.stack([cols[n][j] for j in range(1, len(cols[n]) + 1)], axis=1)
    return out


def parse_mr_params_dump():
    """mr_params_dump.txt.gz -> {array name: ndarray} (Fortran column-major order restored)."""
    out = {}
    with gzip.open(REF / "mr_params_dump.txt.gz", "rt") as fh:
        lines = fh.read().split("\n")
    for i, line in enumerate(lines):
        if line.startswith("ARRAY"):
            p = line.split()
            shape = tuple(int(x) for x in p[2:])
            out[p[1]] = np.array([float(v) for v in lines[i + 1].split()]).reshape(shape, order="F")
    return out

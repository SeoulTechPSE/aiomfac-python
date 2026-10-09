"""Prototype: AIOMFAC activity coefficients of water + organic mixtures (short-range UNIFAC part; no ions, so the
long- and middle-range terms vanish) written as one Numba-compiled loop kernel, to estimate what a Numba rewrite of
the core terms would gain per evaluation against the NumPy code (ActivityModel.evaluate) and the Fortran model.
usage: python tools/numba_prototype.py   (needs numba)"""
import json
import math
import os
import time

import numpy as np
from numba import njit

from aiomfac_py import ActivityModel, Component
from aiomfac_py.s2as import smiles_to_components


@njit(cache=True)
def ln_gamma_neutral(w, mmass, SRNY, Q, RS, QS, parA, T, ref_only=False, ref_in=np.zeros(0)):
    nk, ng = SRNY.shape
    # mass fractions -> mole fractions
    x = np.empty(nk)
    s = 0.0
    for i in range(nk):
        x[i] = w[i] / mmass[i]; s += x[i]
    for i in range(nk):
        x[i] /= s
    psi = np.empty((ng, ng))
    for a in range(ng):
        for b in range(ng):
            psi[a, b] = math.exp(-parA[a, b] / T)
    out = np.empty(nk)
    # combinatorial
    QSS = 0.0; RSS = 0.0
    for i in range(nk):
        QSS += QS[i] * x[i]; RSS += RS[i] * x[i]
    for i in range(nk):
        ViV = RS[i] / RSS
        VQR = ViV * QSS / QS[i]
        out[i] = math.log(ViV) + 1.0 - ViV - 5.0 * QS[i] * (math.log(VQR) + 1.0 - VQR)
    # residual of the mixture and of each pure component (reference)
    gam = np.empty(ng); TH = np.empty(ng); S4 = np.empty(ng); xr = np.empty(nk)
    have_ref = ref_in.shape[0] == nk
    if have_ref:
        for i in range(nk):
            out[i] -= ref_in[i]
    refv = np.zeros(nk)
    for r in range(nk + 1):                     # r = nk: mixture; r < nk: pure component r
        if r < nk and have_ref:
            continue
        if r == nk and ref_only:
            break
        if r < nk:
            nsub = 0.0
            for g in range(ng):
                nsub += SRNY[r, g]
            if nsub <= 1.0:
                continue
            for i in range(nk):
                xr[i] = 1.0 if i == r else 0.0
        else:
            for i in range(nk):
                xr[i] = x[i]
        tot = 0.0
        for g in range(ng):
            v = 0.0
            for i in range(nk):
                v += SRNY[i, g] * xr[i]
            TH[g] = v; tot += v
        qs = 0.0
        for g in range(ng):
            TH[g] = Q[g] * TH[g] / tot; qs += TH[g]
        for g in range(ng):
            TH[g] /= qs
        for g in range(ng):
            v = 0.0
            for k in range(ng):
                v += psi[k, g] * TH[k]
            S4[g] = v
        for g in range(ng):
            v = 0.0
            for k in range(ng):
                v += psi[g, k] * TH[k] / S4[k]
            gam[g] = Q[g] * (1.0 - math.log(S4[g]) - v)
        if r < nk:
            v = 0.0
            for g in range(ng):
                v += SRNY[r, g] * gam[g]
            out[r] -= v
            refv[r] = v
        else:
            for i in range(nk):
                v = 0.0
                for g in range(ng):
                    v += SRNY[i, g] * gam[g]
                out[i] += v
    return refv if ref_only else out


def timeit(f, n):
    for _ in range(20):
        f()
    t = time.perf_counter()
    for _ in range(n):
        f()
    return (time.perf_counter() - t) / n * 1e6


if __name__ == "__main__":
    WATER = Component(1, "Water", ((16, 1),))
    pool = json.load(open(os.environ.get("S1_POOL", "../aiomfac-surrogates/excess_gibbs/results/s1_pool.json")))["pool"]
    mcm = [o["subgroups"] for o in pool if o["src"] == "mcm"][:400]
    rng = np.random.RandomState(1)
    cases = [("binary (pinonaldehyde)", smiles_to_components(["CC(=O)C1CC(CC(=O)O)C1(C)C"]).components)]
    for n_org in (10, 40):
        picks = [mcm[i] for i in rng.choice(len(mcm), n_org, replace=False)]
        cases.append((f"water + {n_org} organics", [WATER] + [Component(k + 2, f"o{k}", tuple(map(tuple, p)))
                                                              for k, p in enumerate(picks)]))
    T = 298.15
    for label, comps in cases:
        m = ActivityModel(comps)
        sr = m.mixture.sr
        nn = m.mixture.n_neutral
        w = np.full(nn, 1.0 / nn)
        args = (w, np.asarray(m.mixture.mmass[:nn], float), np.ascontiguousarray(sr.SRNY, dtype=float),
                np.asarray(sr.Q, float), np.asarray(sr.RS, float), np.asarray(sr.QS, float),
                np.ascontiguousarray(sr.parA, dtype=float), T)
        ref = m.evaluate(list(w), T, basis="mass").ln_gamma[:nn]
        got = ln_gamma_neutral(*args)
        t_np = timeit(lambda: m.evaluate(list(w), T, basis="mass"), 300)
        t_nb = timeit(lambda: ln_gamma_neutral(*args), 20000)
        # pure-component residual references depend on T only: computed once per temperature, as Fortran caches them
        refT = ln_gamma_neutral(*args, True)
        got2 = ln_gamma_neutral(*args, False, refT)
        t_nb2 = timeit(lambda: ln_gamma_neutral(*args, False, refT), 20000)
        print(f"{label:24s} groups {sr.SRNY.shape[1]:3d}  NumPy {t_np:8.1f} us  Numba {t_nb:7.2f} us  "
              f"Numba, T-cached references {t_nb2:7.2f} us  max|diff| {max(np.max(np.abs(got - ref)), np.max(np.abs(got2 - ref))):.1e}",
              flush=True)

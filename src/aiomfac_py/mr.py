"""Middle-range term (main group <-> ion and ion <-> ion interactions).

Fortran counterpart: ModMRpart.f90 -- MRinteractcoeff (mapping the tabulated parameters onto the mixture) and the
MR part of LR_MR_activity. Stateless port; the solvent-mixture reference state (``solvmixrefnd``, never enabled by
the Fortran web driver) is not ported.

Array conventions: ion arrays are zero-padded to NGI = max(#cations, #anions) exactly like the Fortran arrays;
neutral-main-group arrays have NGN rows (rows beyond the number of distinct main groups are zero).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .params import load_mr_params
from .system import Mixture

_EPS = np.finfo(float).eps
_UNDEFINED = -1000.0


@dataclass(frozen=True)
class MRState:
    BAC: np.ndarray; CAC: np.ndarray; omega: np.ndarray; omega2: np.ndarray
    Cnac1: np.ndarray; Cnac2: np.ndarray                 # (NGI, NGI): cation I <-> anion J
    bnc: np.ndarray; cnc: np.ndarray; omegaNC: np.ndarray     # (NGN, NGI): main group <-> cation
    bna: np.ndarray; cna: np.ndarray; omegaNA: np.ndarray     # (NGN, NGI): main group <-> anion
    Qcca: np.ndarray                                     # (NGI, NGI, NGI)
    Rcc: np.ndarray                                      # (NGI, NGI)
    QccaInteract: bool
    RccInteract: bool


def build_mr_state(mixture: Mixture) -> MRState:
    p = load_mr_params()
    sr = mixture.sr
    ngi, ngn = mixture.ngi, len(sr.solv_subs)
    ncat, nan_ = sr.n_cation, sr.n_anion
    bnc = np.zeros((ngn, ngi)); cnc = np.zeros((ngn, ngi)); bna = np.zeros((ngn, ngi)); cna = np.zeros((ngn, ngi))
    omega_nc = np.full((ngn, ngi), 1.2); omega_na = np.full((ngn, ngi), 1.2)

    for i, mg in enumerate(mixture.imaingroup):
        for j, ion in enumerate(sr.cations):
            nc = ion - 200
            if nc > p.bTABnc.shape[1]:
                raise ValueError(f"no main group <-> cation MR parameters for ion {ion}")
            bnc[i, j], cnc[i, j] = p.bTABnc[mg - 1, nc - 1], p.cTABnc[mg - 1, nc - 1]
            if sr.is_peg_system and mg in (52, 68) and nc == 4:          # CHn[OH,PEG] <-> NH4+
                bnc[i, j], cnc[i, j] = 3.0576783e-01, 4.0411133e-01      # ModMRpart.f90 fitparam(273)/(274)
            if bnc[i, j] < _UNDEFINED or cnc[i, j] < _UNDEFINED:
                raise ValueError(f"main group {mg} <-> cation {ion} interaction is not defined (Fortran errorflagmix 1)")
        for j, ion in enumerate(sr.anions):
            na = ion - 240
            if na > p.bTABna.shape[1]:
                raise ValueError(f"no main group <-> anion MR parameters for ion {ion}")
            bna[i, j], cna[i, j] = p.bTABna[mg - 1, na - 1], p.cTABna[mg - 1, na - 1]
            if sr.is_peg_system and mg in (52, 68) and na == 21:         # CHn[OH,PEG] <-> SO4--
                bna[i, j], cna[i, j] = -2.1822703e-01, -1.2595744e-01    # ModMRpart.f90 fitparam(275)/(276)
            if bna[i, j] < _UNDEFINED or cna[i, j] < _UNDEFINED:
                raise ValueError(f"main group {mg} <-> anion {ion} interaction is not defined (Fortran errorflagmix 2)")

    bac = np.zeros((ngi, ngi)); cac = np.zeros((ngi, ngi)); cn1 = np.zeros((ngi, ngi)); cn2 = np.zeros((ngi, ngi))
    omega = np.full((ngi, ngi), 0.8); omega2 = np.full((ngi, ngi), 0.6)
    for i, c in enumerate(sr.cations):
        for j, a in enumerate(sr.anions):
            nc, na = c - 200, a - 240
            bac[i, j], cac[i, j] = p.bTABAC[nc - 1, na - 1], p.cTABAC[nc - 1, na - 1]
            cn1[i, j], cn2[i, j] = p.Cn1TABAC[nc - 1, na - 1], p.Cn2TABAC[nc - 1, na - 1]
            omega2[i, j], omega[i, j] = p.omega2TAB[nc - 1, na - 1], p.omegaTAB[nc - 1, na - 1]
            if abs(p.bTABAC[nc - 1, na - 1]) < 1.0e-10 and not (nc == 5 and na == 7):     # H+ <-> OH- is exempt
                raise ValueError(f"cation {c} <-> anion {a} interaction parameters are not defined (errorflagmix 9)")

    qcca = np.zeros((ngi, ngi, ngi)); rcc = np.zeros((ngi, ngi))
    for i, c in enumerate(sr.cations):
        for j, c2 in enumerate(sr.cations):
            nc, nc2 = c - 200, c2 - 200
            if nc <= p.RccTAB.shape[0] and nc2 <= p.RccTAB.shape[0]:
                rcc[i, j] = p.RccTAB[nc - 1, nc2 - 1]
                for k, a in enumerate(sr.anions):
                    qcca[i, j, k] = p.qcca1TAB[nc - 1, nc2 - 1, a - 241]
    rcc_int = ncat > 1 and 204 in sr.cations and 205 in sr.cations
    qcca_int = rcc_int and 248 in sr.anions
    return MRState(bac, cac, omega, omega2, cn1, cn2, bnc, cnc, omega_nc, bna, cna, omega_na, qcca, rcc,
                   bool(qcca_int), bool(rcc_int))


@dataclass
class MRTerms:
    ln_gamma_neutral: np.ndarray     # (n_neutral,)  gnmrln
    ln_gamma_cation: np.ndarray      # (NGI,)        gcmrln
    ln_gamma_anion: np.ndarray       # (NGI,)        gamrln
    tmolal: float                    # basis conversion term used for the ions


def mr_terms(mixture: Mixture, st: MRState, cation_z, anion_z, smc, sma, xn, si: float,
             sum_ion_molalities: float, mean_solvent_mw: float) -> MRTerms:
    sr = mixture.sr
    nn, ngi, ngn = mixture.n_neutral, mixture.ngi, len(sr.solv_subs)
    mwater = 0.01801528
    if not (mixture.n_electrol > 0 and si > _EPS):
        return MRTerms(np.zeros(nn), np.zeros(ngi), np.zeros(ngi), 0.0)

    # salt-free subgroup / main-group mole fractions and main-group molar masses
    sub_x = np.array([np.sum(mixture.itab[:nn, s - 1] * xn[:nn]) for s in sr.solv_subs])
    tot = float(sub_x.sum())
    if tot > 0.0:
        sub_x = sub_x / tot
    mg_x = np.zeros(ngn); mg_mk = np.zeros(ngn)
    for j in range(ngn):
        mg_x[mixture.maingroup_index[j]] += sub_x[j]
    for j in range(ngn):
        i = mixture.maingroup_index[j]
        if mg_x[i] > 0.0:
            mg_mk[i] += mixture.subgroup_mw[j] * sub_x[j] / mg_x[i]

    SI = si
    SI2 = math.sqrt(SI)
    Z = float(np.sum(sma * np.abs(anion_z) + smc * cation_z))
    zc2 = cation_z ** 2
    za = np.abs(anion_z); za2 = za ** 2
    lrw1 = -0.5 / SI2

    oexp1 = st.omega * SI2
    Bca = np.where(oexp1 > 300.0, st.BAC, st.BAC + st.CAC * np.exp(-np.minimum(oexp1, 700.0)))
    oexp2 = st.omega2 * SI2
    Cnca = np.where(oexp2 > 300.0, st.Cnac1, st.Cnac1 + st.Cnac2 * np.exp(-np.minimum(oexp2, 700.0)))
    BSca = lrw1 * st.omega * (Bca - st.BAC)
    CSca = lrw1 * st.omega2 * (Cnca - st.Cnac1)
    if SI2 > 250.0:
        Bkc, Bka = st.bnc.copy(), st.bna.copy()
    else:
        Bkc = st.bnc + st.cnc * np.exp(-st.omegaNC * SI2)
        Bka = st.bna + st.cna * np.exp(-st.omegaNA * SI2)
    BSkc = lrw1 * st.omegaNC * (Bkc - st.bnc)
    BSka = lrw1 * st.omegaNA * (Bka - st.bna)

    avg_mw = float(np.sum(mg_mk * mg_x))

    # neutral part
    SumBm = (Bka * sma[None, :] + Bkc * smc[None, :]).sum(axis=1)
    Sumkion1 = float(np.sum((Bka + SI * BSka) * mg_x[:, None] * sma[None, :]
                            + (Bkc + SI * BSkc) * mg_x[:, None] * smc[None, :]))
    M = Bca + SI * BSca + 2.0 * Z * Cnca + Z * SI * CSca
    SumCA1 = float(np.sum(M * smc[:, None] * sma[None, :]))
    SumQ1 = 0.0
    if st.QccaInteract:
        for i in range(ngi):
            for j in range(i, ngi):
                SumQ1 += float(np.sum(2.0 * st.Qcca[i, j, :] * smc[i] * smc[j] * sma))
    SumRc = 0.0
    if st.RccInteract:
        for i in range(ngi):
            SumRc += float(np.sum(st.Rcc[i, i:] * smc[i] * smc[i:]))
    gk = SumBm - mg_mk / avg_mw * Sumkion1 - mg_mk * SumCA1 - mg_mk * SumQ1 - mg_mk * SumRc
    n_mg = len(mixture.imaingroup)
    gnmr = mixture.itabmg[:nn][:, np.array(mixture.imaingroup) - 1] @ gk[:n_mg]

    # ions
    SumBcx = (Bkc * mg_x[:, None]).sum(axis=0)
    SumBax = (Bka * mg_x[:, None]).sum(axis=0)
    SumCBca = ((Bca + Z * Cnca) * sma[None, :]).sum(axis=1)
    SumABca = ((Bca + Z * Cnca) * smc[:, None]).sum(axis=0)
    pair = smc[:, None] * sma[None, :]
    SumBsCA = float(np.sum(BSca * pair)); SumCnca = float(np.sum(Cnca * pair)); SumCsca = float(np.sum(CSca * pair))
    Sumkion2 = float(np.sum(BSka * mg_x[:, None] * sma[None, :] + BSkc * mg_x[:, None] * smc[None, :]))

    gcmr = (SumBcx / avg_mw + zc2 / (2.0 * avg_mw) * Sumkion2 + SumCBca + 0.5 * zc2 * SumBsCA
            + cation_z * SumCnca + 0.5 * zc2 * Z * SumCsca)
    gamr = (SumBax / avg_mw + za2 / (2.0 * avg_mw) * Sumkion2 + SumABca + 0.5 * za2 * SumBsCA
            + za * SumCnca + 0.5 * za2 * Z * SumCsca)
    if st.QccaInteract:
        for i in range(ngi):
            gcmr[i] += float(np.sum(smc[None, :, None] * sma[None, None, :] * st.Qcca[i][None, :, :]))
            s = 0.0
            for j in range(ngi):
                s += float(np.sum(smc[j] * smc[j:] * st.Qcca[j, j:, i]))
            gamr[i] += s
    if st.RccInteract:
        for i in range(ngi):
            gcmr[i] += float(np.sum(st.Rcc[i, :] * smc))

    tmolal = math.log(mwater / mean_solvent_mw + mwater * sum_ion_molalities)
    return MRTerms(gnmr, gcmr, gamr, tmolal)

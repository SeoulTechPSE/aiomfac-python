"""Automatic differentiation of the AIOMFAC activities of :class:`aiomfac_py.phase_equilibrium.ExplicitLiquidModel`.

A JAX transcription of ``ExplicitLiquidModel.ln_a`` (LR, MR and SR terms of :mod:`aiomfac_py.lr`, :mod:`aiomfac_py.mr`,
:mod:`aiomfac_py.sr`, the molality-to-mole-fraction conversion of ``ActivityModel._x_from_molalities`` and the CO2(aq)
salting-out term), used for the exact Jacobian d(c + ln a)/dn in the phase-equilibrium Newton iterations.  The values
themselves are still computed with the NumPy code, which is the version validated against the Fortran model; the tests
check that this transcription reproduces it to round-off.

Optional dependency: ``jax`` (``pip install jax``).  Importing this module enables 64-bit floats in JAX
(``jax_enable_x64``), which the activity calculation needs.
"""
from __future__ import annotations

import math

import numpy as np

try:
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    AVAILABLE = True
except ImportError:                                           # pragma: no cover - optional dependency
    jax = jnp = None
    AVAILABLE = False

from .lr import debye_huckel_parameters
from .params import load_subgroup_params
from .sr import _residual_reference, psi_t

_MWATER = 0.01801528


def build_jacobian(lm, T: float):
    """Return the jit-compiled ``jac(n) -> d ln a/dn`` (N x N) of the explicit liquid model ``lm`` at temperature
    ``T`` and the function ``ln_a(n)`` itself (``lm.ln_a`` without the reaction constants c, which do not depend on
    n).  Logarithms of amounts are written as differences (ln n_i - ln solvent mass, ...), so that a species with zero
    amount only gives non-finite entries in its own row."""
    if not AVAILABLE:
        raise ImportError("jax is not installed")
    model = lm.model
    m = model.mixture
    sr = m.sr
    nn, ngi = lm.n_neutral, max(m.ngi, 1)
    nc_, na_ = sr.n_cation, sr.n_anion
    N = lm.N
    mm = jnp.asarray(lm._mm)                                  # neutral molar masses [kg/mol]
    mm_all = jnp.asarray(np.asarray(m.mmass[:nn], dtype=float))
    # species -> padded ion arrays
    cat_sp = np.array([nn + k for k, (c, _) in enumerate(lm._pos) if c], dtype=int)
    cat_ix = np.array([i for (c, i) in lm._pos if c], dtype=int)
    an_sp = np.array([nn + k for k, (c, _) in enumerate(lm._pos) if not c], dtype=int)
    an_ix = np.array([i for (c, i) in lm._pos if not c], dtype=int)
    kco2 = lm._kco2
    cz = jnp.asarray(np.asarray(model.cation_z, dtype=float))
    az = jnp.asarray(np.asarray(model.anion_z, dtype=float))
    A_dh, b_dh = debye_huckel_parameters(T)
    # ---- MR constants
    st = model._mr
    has_mr = m.n_electrol > 0
    solv_subs = np.array(sr.solv_subs, dtype=int)
    ngn = len(solv_subs)
    itab_s = jnp.asarray(np.asarray(m.itab[:nn][:, solv_subs - 1], dtype=float))      # (nn, ngn)
    P = np.zeros((ngn, ngn))
    for j in range(ngn):
        P[m.maingroup_index[j], j] = 1.0
    P = jnp.asarray(P)
    sub_mw = jnp.asarray(np.asarray(m.subgroup_mw, dtype=float))
    n_mg = len(m.imaingroup)
    itabmg = jnp.asarray(np.asarray(m.itabmg[:nn][:, np.array(m.imaingroup) - 1], dtype=float))
    BAC, CAC = jnp.asarray(st.BAC), jnp.asarray(st.CAC)
    om, om2 = jnp.asarray(st.omega), jnp.asarray(st.omega2)
    Cn1, Cn2 = jnp.asarray(st.Cnac1), jnp.asarray(st.Cnac2)
    bnc, cnc, omNC = jnp.asarray(st.bnc), jnp.asarray(st.cnc), jnp.asarray(st.omegaNC)
    bna, cna, omNA = jnp.asarray(st.bna), jnp.asarray(st.cna), jnp.asarray(st.omegaNA)
    Qcca, Rcc = jnp.asarray(st.Qcca), jnp.asarray(st.Rcc)
    U = jnp.asarray(np.triu(np.ones((ngi, ngi))))           # i <= j
    # ---- SR constants
    psi = psi_t(sr, T)
    ref = _residual_reference(sr, psi, np.ones(nn) / nn, False)
    psi_j = jnp.asarray(psi)
    SRNY, Qg = jnp.asarray(sr.SRNY), jnp.asarray(sr.Q)
    RS, QS = jnp.asarray(sr.RS), jnp.asarray(sr.QS)
    cinf = np.zeros(sr.n_species)
    if sr.n_species > nn:
        ion = slice(nn, None)
        B = 5.0 * sr.QS[ion] * np.log(sr.QS[ion] / 1.40 * 0.92 / sr.RS[ion]) + sr.XL[ion] - sr.RS[ion] / 0.92 * (-2.32)
        cinf[ion] = np.log(sr.RS[ion] / 0.92) + B
    ref_plus_cinf = jnp.asarray(ref + cinf)
    # ---- CO2(aq) salting-out
    if kco2 is not None:
        sg = load_subgroup_params()
        lam_c = jnp.asarray(sg.lambdaIN[np.array(sr.cations) - 201])
        lam_a = jnp.asarray(sg.lambdaIN[np.array(sr.anions) - 201])

    def mu(n):
        nw = n[:nn]
        solv = jnp.dot(nw, mm)
        xn = nw / jnp.sum(nw)
        smc = jnp.zeros(ngi).at[cat_ix].set(n[cat_sp] / solv) if len(cat_ix) else jnp.zeros(ngi)
        sma = jnp.zeros(ngi).at[an_ix].set(n[an_sp] / solv) if len(an_ix) else jnp.zeros(ngi)
        # mole fractions of the AIOMFAC species
        mean_mw = jnp.sum(mm_all * xn)
        m_neutral = xn / mean_mw
        sum_ion = jnp.sum(sma[:na_]) + jnp.sum(smc[:nc_])
        sum_molal = jnp.sum(m_neutral) + sum_ion
        x = jnp.concatenate([m_neutral, smc[:nc_], sma[:na_]]) / sum_molal
        ln_x_n = jnp.log(nw) - jnp.log(jnp.sum(nw)) - jnp.log(mean_mw) - jnp.log(sum_molal)
        ln_solv = jnp.log(solv)
        # LR
        za, zc2 = jnp.abs(az), cz ** 2
        za2 = za ** 2
        SI = 0.5 * jnp.sum(sma * za2 + smc * zc2)
        SI2 = jnp.sqrt(SI)
        bb = 1.0 + b_dh * SI2
        lr_n = mm_all * (2.0 * A_dh / b_dh ** 3 * (bb - 1.0 / bb - 2.0 * jnp.log(bb)))
        w_lr = A_dh * SI2 / bb
        lr_c, lr_a = -zc2 * w_lr, -za2 * w_lr
        # MR
        if has_mr:
            sub_x = itab_s.T @ xn
            sub_x = sub_x / jnp.sum(sub_x)
            mg_x = P @ sub_x
            mg_mk = jnp.where(mg_x > 0.0, (P @ (sub_mw * sub_x)) / jnp.where(mg_x > 0.0, mg_x, 1.0), 0.0)
            Z = jnp.sum(sma * za + smc * cz)
            lrw1 = -0.5 / SI2
            oexp1 = om * SI2
            Bca = jnp.where(oexp1 > 300.0, BAC, BAC + CAC * jnp.exp(-jnp.minimum(oexp1, 700.0)))
            oexp2 = om2 * SI2
            Cnca = jnp.where(oexp2 > 300.0, Cn1, Cn1 + Cn2 * jnp.exp(-jnp.minimum(oexp2, 700.0)))
            BSca = lrw1 * om * (Bca - BAC)
            CSca = lrw1 * om2 * (Cnca - Cn1)
            big = SI2 > 250.0
            Bkc = jnp.where(big, bnc, bnc + cnc * jnp.exp(-omNC * SI2))
            Bka = jnp.where(big, bna, bna + cna * jnp.exp(-omNA * SI2))
            BSkc = lrw1 * omNC * (Bkc - bnc)
            BSka = lrw1 * omNA * (Bka - bna)
            avg_mw = jnp.sum(mg_mk * mg_x)
            SumBm = (Bka * sma[None, :] + Bkc * smc[None, :]).sum(axis=1)
            Sumkion1 = jnp.sum((Bka + SI * BSka) * mg_x[:, None] * sma[None, :]
                               + (Bkc + SI * BSkc) * mg_x[:, None] * smc[None, :])
            M = Bca + SI * BSca + 2.0 * Z * Cnca + Z * SI * CSca
            SumCA1 = jnp.sum(M * smc[:, None] * sma[None, :])
            SumQ1 = 2.0 * jnp.einsum("ij,ijk,i,j,k->", U, Qcca, smc, smc, sma) if st.QccaInteract else 0.0
            SumRc = jnp.einsum("ij,ij,i,j->", U, Rcc, smc, smc) if st.RccInteract else 0.0
            gk = SumBm - mg_mk / avg_mw * Sumkion1 - mg_mk * SumCA1 - mg_mk * SumQ1 - mg_mk * SumRc
            mr_n = itabmg @ gk[:n_mg]
            SumBcx = (Bkc * mg_x[:, None]).sum(axis=0)
            SumBax = (Bka * mg_x[:, None]).sum(axis=0)
            SumCBca = ((Bca + Z * Cnca) * sma[None, :]).sum(axis=1)
            SumABca = ((Bca + Z * Cnca) * smc[:, None]).sum(axis=0)
            pair = smc[:, None] * sma[None, :]
            SumBsCA, SumCn, SumCs = jnp.sum(BSca * pair), jnp.sum(Cnca * pair), jnp.sum(CSca * pair)
            Sumkion2 = jnp.sum(BSka * mg_x[:, None] * sma[None, :] + BSkc * mg_x[:, None] * smc[None, :])
            mr_c = (SumBcx / avg_mw + zc2 / (2.0 * avg_mw) * Sumkion2 + SumCBca + 0.5 * zc2 * SumBsCA
                    + cz * SumCn + 0.5 * zc2 * Z * SumCs)
            mr_a = (SumBax / avg_mw + za2 / (2.0 * avg_mw) * Sumkion2 + SumABca + 0.5 * za2 * SumBsCA
                    + za * SumCn + 0.5 * za2 * Z * SumCs)
            if st.QccaInteract:
                mr_c = mr_c + jnp.einsum("j,k,ijk->i", smc, sma, Qcca)
                mr_a = mr_a + jnp.einsum("jl,j,l,jli->i", U, smc, smc, Qcca)
            if st.RccInteract:
                mr_c = mr_c + Rcc @ smc
            tmolal = jnp.log(_MWATER / mean_mw + _MWATER * sum_ion)
        else:
            mr_n, mr_c, mr_a, tmolal = jnp.zeros(nn), jnp.zeros(ngi), jnp.zeros(ngi), 0.0
        # SR
        S1 = SRNY.T @ x
        XG = S1 / S1.sum()
        TH = Qg * XG / jnp.dot(Qg, XG)
        S4 = psi_j.T @ TH
        lnR = SRNY @ (Qg * (1.0 - jnp.log(S4) - psi_j @ (TH / S4)))
        QSS, RSS = jnp.dot(QS, x), jnp.dot(RS, x)
        ViV = RS / RSS
        VQR = ViV * QSS / QS
        lnC = jnp.log(ViV) + 1.0 - ViV - 5.0 * QS * (jnp.log(VQR) + 1.0 - VQR)
        srt = lnC + lnR - ref_plus_cinf
        out = jnp.zeros(N)
        out = out.at[:nn].set(lr_n + mr_n + srt[:nn] + ln_x_n)
        if len(cat_ix):
            g_c = mr_c[cat_ix] + srt[nn + cat_ix] + lr_c[cat_ix] - tmolal
            out = out.at[cat_sp].set(g_c + jnp.log(n[cat_sp]) - ln_solv)
        if len(an_ix):
            g_a = mr_a[an_ix] + srt[nn + nc_ + an_ix] + lr_a[an_ix] - tmolal
            out = out.at[an_sp].set(g_a + jnp.log(n[an_sp]) - ln_solv)
        if kco2 is not None:
            lg = 2.0 * jnp.sum(smc[:nc_] * lam_c) + 2.0 * jnp.sum(sma[:na_] * lam_a)
            out = out.at[kco2].set(lg + jnp.log(n[kco2]) - ln_solv)
        return out

    jac = jax.jit(jax.jacfwd(mu, argnums=0))
    return jac, jax.jit(mu)

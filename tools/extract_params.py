"""Extract the short-range (UNIFAC) parameter tables from the AIOMFAC Fortran source.

The tables are parsed from ModSRparam.f90 (not typed by hand) and written to
src/aiomfac_py/data/sr_params.npz together with provenance metadata.

usage: python tools/extract_params.py <path/to/AIOMFAC/FortranCode> [--out PATH]
"""
import argparse, hashlib, re, subprocess
from pathlib import Path
import numpy as np

NMAIN, TOPSUB = 76, 265   # ModSystemProp: Nmaingroups, topsubno (checked below)


def _strip(text):
    """drop Fortran comments and continuation ampersands; keep everything else"""
    out = []
    for line in text.replace('\r\n', '\n').split('\n'):
        line = line.split('!')[0].replace('&', ' ')
        out.append(line)
    return '\n'.join(out)


def _block(src, name, occurrence):
    """numbers inside the `name = ... real([ ... ], kind=wp)` statement (occurrence-th match, 0-based)"""
    pat = re.compile(r'\b' + name + r'\s*=\s*(?:reshape\(\s*)?real\(\s*\[(.*?)\]\s*,\s*kind\s*=\s*wp\s*\)', re.S | re.I)
    matches = list(pat.finditer(src))
    if len(matches) <= occurrence:
        raise RuntimeError(f'{name}: occurrence {occurrence} not found ({len(matches)} matches)')
    body = matches[occurrence].group(1)
    vals = re.findall(r'[-+]?\d*\.?\d+(?:[dDeE][-+]?\d+)?', body)
    return np.array([float(v.replace('D', 'e').replace('d', 'e')) for v in vals])


def extract(fortran_dir):
    fdir = Path(fortran_dir)
    sp = (fdir / 'ModSystemProp.f90').read_text(errors='replace')
    assert int(re.search(r'Nmaingroups\s*=\s*(\d+)', sp).group(1)) == NMAIN
    assert int(re.search(r'topsubno\s*=\s*(\d+)', sp).group(1)) == TOPSUB
    raw = (fdir / 'ModSRparam.f90').read_text(errors='replace')
    src = _strip(raw)
    # only the `else` (use_latest_param = .false.) branch is active in this version; the `if` branch is empty.
    assert re.search(r'use_latest_param\s*=\s*\.false\.', src)
    R = _block(src, 'SR_RR', 0)
    Q = _block(src, 'SR_QQ', 0)
    tabs = {n: _block(src, n, 0) for n in ('ARR', 'BRR', 'CRR')}
    assert R.size == TOPSUB and Q.size == TOPSUB, (R.size, Q.size)
    for n, v in tabs.items():
        assert v.size == NMAIN * NMAIN, (n, v.size)
    # Fortran reshape() is column-major: ARR(i,j) = flat[i + NMAIN*j]  (0-based here)
    mats = {n: v.reshape((NMAIN, NMAIN), order='F') for n, v in tabs.items()}
    return R, Q, mats, hashlib.sha256(raw.encode()).hexdigest()


def _array_after(src, name):
    """numbers of `name = [ ... ]` (handles `(v, i = a,b)` implied-do repeats; b may be `topsubno`)"""
    m = re.search(r'\b' + name + r'\s*=\s*(?:real\(\s*)?\[', src, re.I)
    if not m:
        raise RuntimeError(f'{name}: array constructor not found')
    depth, i = 1, m.end()
    while depth:
        depth += {'[': 1, ']': -1}.get(src[i], 0)
        i += 1
    body = src[m.end():i - 1]

    def rep(mo):
        v, a, b = mo.group(1), int(mo.group(2)), mo.group(3)
        b = TOPSUB if b.lower() == 'topsubno' else int(b)
        return ' '.join([v] * (b - a + 1))
    body = re.sub(r'\(\s*([-+]?[\d.]+(?:[dDeE][-+]?\d+)?)(?:_wp)?\s*,\s*\w+\s*=\s*(\d+)\s*,\s*(\w+)\s*\)', rep, body)
    body = body.replace(',', ' ')
    vals = re.findall(r'[-+]?\d*\.?\d+(?:[dDeE][-+]?\d+)?', body)
    return np.array([float(v.replace('D', 'e').replace('d', 'e')) for v in vals])


def extract_lambda_in(fortran_dir):
    """CO2(aq) salting-out coefficients lambdaIN(201:topsubno), used by GammaCO2 (ModMRpart.f90)."""
    raw = (Path(fortran_dir) / 'ModMRpart.f90').read_text(errors='replace')
    src = _strip(raw)
    vals = _array_after(src, 'lambdaIN')
    assert vals.size == TOPSUB - 200, vals.size
    return vals, hashlib.sha256(raw.encode()).hexdigest()


def extract_subgroups(fortran_dir):
    raw = (Path(fortran_dir) / 'ModSubgroupProp.f90').read_text(errors='replace')
    src = _strip(raw)
    nk = _array_after(src, 'NKTAB').astype(int)
    mw = _array_after(src, 'GroupMW')
    ch = _array_after(src, 'Ioncharge').astype(int)
    smwc = _array_after(src, 'SMWC')
    smwa = _array_after(src, 'SMWA')
    assert nk.size == TOPSUB and mw.size == 200 and ch.size == TOPSUB - 200, (nk.size, mw.size, ch.size)
    assert smwc.size == 40 and smwa.size == 40, (smwc.size, smwa.size)
    return nk, mw, ch, smwc, smwa, hashlib.sha256(raw.encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('fortran_dir')
    ap.add_argument('--out', default=str(Path(__file__).resolve().parents[1] / 'src/aiomfac_py/data/sr_params.npz'))
    a = ap.parse_args()
    R, Q, m, sha = extract(a.fortran_dir)
    try:
        commit = subprocess.check_output(['git', '-C', a.fortran_dir, 'rev-parse', 'HEAD'], text=True).strip()
    except Exception:
        commit = 'unknown'
    np.savez_compressed(a.out, R=R, Q=Q, ARR=m['ARR'], BRR=m['BRR'], CRR=m['CRR'],
                        source_commit=commit, source_file='FortranCode/ModSRparam.f90', source_sha256=sha)
    lam, sha_lam = extract_lambda_in(a.fortran_dir)
    nk, mw, ch, smwc, smwa, sha2 = extract_subgroups(a.fortran_dir)
    out2 = str(Path(a.out).with_name('subgroup_params.npz'))
    np.savez_compressed(out2, NKTAB=nk, GroupMW=mw, Ioncharge=ch, SMWC=smwc, SMWA=smwa, lambdaIN=lam, source_commit=commit,
                        source_file='FortranCode/ModSubgroupProp.f90', source_sha256=sha2)
    print('wrote', out2, '| NKTAB', nk.shape, '| GroupMW', mw.shape, '| Ioncharge', ch.shape)
    print('wrote', a.out, '| commit', commit[:7], '| R,Q:', R.shape, '| ARR/BRR/CRR:', m['ARR'].shape)


if __name__ == '__main__':
    main()

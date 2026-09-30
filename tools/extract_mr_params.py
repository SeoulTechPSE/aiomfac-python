"""Extract the middle-range (MR) parameter tables from the AIOMFAC Fortran source (subroutine MRdata).

MRdata is a straight sequence of assignments (no conditionals), so it is interpreted statement by statement:
  NAME = value_wp                       (whole-array default)
  NAME(i,j[,k]) = value_wp              (single element)
  NAME(1:Nmaingroups,NN) = real([...], kind=wp)      (column vectors)
The result is written to src/aiomfac_py/data/mr_params.npz. Every array is verified bit-for-bit against a dump of the
arrays that the compiled Fortran model holds after MRdata (see tests/test_mr_params.py); statements that cannot be
interpreted make the tool fail instead of being skipped.

usage: python tools/extract_mr_params.py <path/to/AIOMFAC/FortranCode> [--out PATH]
"""
import argparse, hashlib, re, subprocess
from pathlib import Path
import numpy as np

CONST = {'nmaingroups': 76, 'topsubno': 265}
NUM = r'[-+]?(?:\d+\.?\d*|\.\d+)(?:[eEdD][-+]?\d+)?'
NUM_RE = re.compile(NUM)


def _num(tok):
    """value of a Fortran real literal, reproducing its kind: only literals with a `_wp` suffix or a D exponent
    are double precision; anything else (e.g. `0.114470419384503`) is default-kind REAL, i.e. single precision,
    and is widened to double afterwards -- exactly what the compiled model does with such a literal."""
    t = tok.strip()
    is_double = bool(re.search(r'_wp$', t, re.I)) or bool(re.search(r'\d[dD][-+]?\d+$', t))
    x = float(re.sub(r'_wp$', '', t, flags=re.I).replace('D', 'e').replace('d', 'e'))
    return x if is_double else float(np.float32(x))


def _statements(text):
    """comment-free, continuation-joined statements of a Fortran block"""
    out, buf = [], ''
    for line in text.replace('\r\n', '\n').split('\n'):
        line = line.split('!')[0].strip()
        if not line:
            continue
        if line.startswith('&'):
            line = line[1:].strip()
        if line.endswith('&'):
            buf += line[:-1].strip() + ' '
            continue
        out.append((buf + line).strip()); buf = ''
    return out


def _eval_dim(expr):
    e = expr.strip().lower()
    for k, v in CONST.items():
        e = e.replace(k, str(v))
    e = re.sub(r'\b0+(\d)', r'\1', e)              # Fortran allows leading zeros in indices (e.g. 01)
    if not re.fullmatch(r'[\d+\-*\s]+', e):
        raise ValueError(f'cannot evaluate dimension {expr!r}')
    return int(eval(e))


def _index(tok, size):
    t = tok.strip().lower()
    if ':' in t:
        a, b = (x.strip() for x in t.split(':'))
        lo = 1 if a == '' else _eval_dim(a)
        hi = size if b == '' else _eval_dim(b)
        return slice(lo - 1, hi)
    return _eval_dim(t) - 1


def extract_mr(fortran_dir):
    raw = (Path(fortran_dir) / 'ModMRpart.f90').read_text(errors='replace')
    m = re.search(r'subroutine\s+MRdata\s*\(\s*\)(.*?)end\s+subroutine\s+MRdata', raw, re.S | re.I)
    stmts = _statements(m.group(1))
    canon, arrays, unparsed = {}, {}, []
    for s in stmts:
        low = s.lower()
        if low == 'implicit none':
            continue
        if low.startswith('allocate'):
            for name, dims in re.findall(r'(\w+)\(([^()]*)\)', s[s.index('(') + 1:s.rindex(')')]):
                shape = tuple(_eval_dim(d) for d in dims.split(','))
                canon[name.lower()] = name
                arrays[name.lower()] = np.zeros(shape)
            continue
        mm = re.fullmatch(r'(\w+)\s*(?:\(([^=]*)\))?\s*=\s*(.*)', s)
        if not mm:
            unparsed.append(s); continue
        name, idx, rhs = mm.group(1).lower(), mm.group(2), mm.group(3).strip()
        if name not in arrays:
            unparsed.append(s); continue
        arr = arrays[name]
        vec = re.fullmatch(r'real\(\s*\[(.*)\]\s*,\s*kind\s*=\s*wp\s*\)', rhs, re.S | re.I)
        if vec:
            val = np.array([_num(t) for t in re.findall(NUM + r'(?:_wp)?', vec.group(1))])
        elif re.fullmatch(NUM + r'(?:_wp)?', rhs):
            val = _num(rhs)
        elif re.fullmatch(r'(\w+)\(([^():]*)\)', rhs) and re.fullmatch(r'(\w+)\(([^():]*)\)', rhs).group(1).lower() in arrays:
            # copy of a single element of an array, evaluated in statement order (e.g. bTABAC(4,6) = bTABAC(4,5))
            r = re.fullmatch(r'(\w+)\(([^():]*)\)', rhs)
            src_arr = arrays[r.group(1).lower()]
            src_idx = tuple(_eval_dim(t) - 1 for t in r.group(2).split(','))
            if len(src_idx) != src_arr.ndim:
                unparsed.append(s); continue
            val = float(src_arr[src_idx])
        else:
            unparsed.append(s); continue
        if idx is None:
            sel = (slice(None),) * arr.ndim
        else:
            parts = idx.split(',')
            if len(parts) != arr.ndim:
                unparsed.append(s); continue
            sel = tuple(_index(p, arr.shape[k]) for k, p in enumerate(parts))
        target = arr[sel]
        if np.ndim(val) and target.shape != val.shape:
            raise ValueError(f'shape mismatch in statement {s[:80]!r}: {target.shape} vs {val.shape}')
        arr[sel] = val
    if unparsed:
        raise RuntimeError('MRdata statements that were not interpreted:\n  ' + '\n  '.join(u[:110] for u in unparsed))
    return {canon[k]: v for k, v in arrays.items()}, hashlib.sha256(raw.encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('fortran_dir')
    ap.add_argument('--out', default=str(Path(__file__).resolve().parents[1] / 'src/aiomfac_py/data/mr_params.npz'))
    a = ap.parse_args()
    arrays, sha = extract_mr(a.fortran_dir)
    try:
        commit = subprocess.check_output(['git', '-C', a.fortran_dir, 'rev-parse', 'HEAD'], text=True).strip()
    except Exception:
        commit = 'unknown'
    np.savez_compressed(a.out, source_commit=commit, source_file='FortranCode/ModMRpart.f90 (MRdata)',
                        source_sha256=sha, **arrays)
    print('wrote', a.out, '|', ', '.join(f'{k}{v.shape}' for k, v in arrays.items()))


if __name__ == '__main__':
    main()

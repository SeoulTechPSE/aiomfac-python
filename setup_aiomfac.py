"""
setup_aiomfac.py -- Colab/Jupyter installer for aiomfac_py.

Unlike FEniCSx (SeoulTechPSE/fenicsx-colab), aiomfac_py has no compiled
dependencies of its own: it is pure Python + numpy, with two optional extras
(scipy for the dissociation-equilibrium solvers, epam.indigo for the SMILES
subgroup-decomposition tool). So there is no conda/micromamba environment and
no MPI/Jupyter-magic machinery here -- this script is just a thin, repeatable
wrapper around `pip install` that Colab's "Quick Start" cell `%run`s, mirroring
fenicsx-colab's setup_fenicsx.py in spirit (repo-relative paths, idempotent
re-runs, a --clean option, a verification step) but not its complexity.

Usage (normally invoked via %run from the Quick Start cell, see
notebooks/bootstrap_colab.ipynb):

    python setup_aiomfac.py [--carbonate] [--smiles] [--all] [--clean] [--no-editable]

OPTIONS:
    --carbonate     install the 'carbonate' extra (scipy; dissociation-equilibrium
                    solvers for bisulfate/bicarbonate systems) -- installed by
                    default, pass --no-carbonate to skip it
    --smiles        install the 'smiles' extra (epam.indigo; aiomfac_py.s2as,
                    SMILES -> AIOMFAC subgroups). NOT installed by default: it
                    is a heavier, less commonly needed dependency
    --all           shorthand for --carbonate --smiles
    --clean         uninstall aiomfac_py first (useful after pulling repo
                    changes that altered pyproject.toml)
    --no-editable   install normally instead of `pip install -e .` (editable
                    installs let you edit files under src/ and re-import
                    without reinstalling, which is usually what you want when
                    iterating on the package itself inside Colab)
    --help, -h      show this help message
"""
from pathlib import Path
import subprocess
import sys

REPO_DIR = Path(__file__).resolve().parent
PYPROJECT = REPO_DIR / "pyproject.toml"


def run(cmd, cwd=None):
    print("$", " ".join(map(str, cmd)))
    subprocess.run(cmd, cwd=cwd, check=True)


def parse_args():
    carbonate = True
    smiles = False
    clean = False
    editable = True

    args = sys.argv[1:]
    if any(a in ("--help", "-h") for a in args):
        print(__doc__)
        sys.exit(0)

    for a in args:
        if a == "--carbonate":
            carbonate = True
        elif a == "--no-carbonate":
            carbonate = False
        elif a == "--smiles":
            smiles = True
        elif a == "--no-smiles":
            smiles = False
        elif a == "--all":
            carbonate = True
            smiles = True
        elif a == "--clean":
            clean = True
        elif a == "--no-editable":
            editable = False
        else:
            print(f"WARNING: unknown option: {a}  (see --help)")
    return carbonate, smiles, clean, editable


def main():
    if not PYPROJECT.exists():
        print(f"ERROR: pyproject.toml not found at {PYPROJECT}")
        print("       Run this script from inside a cloned aiomfac-python checkout.")
        sys.exit(1)

    carbonate, smiles, clean, editable = parse_args()

    extras = []
    if carbonate:
        extras.append("carbonate")
    if smiles:
        extras.append("smiles")
    extras_spec = f"[{','.join(extras)}]" if extras else ""

    print("=" * 70)
    print("aiomfac_py Colab setup")
    print("=" * 70)
    print(f"Repo dir : {REPO_DIR}")
    print(f"Extras   : {', '.join(extras) if extras else '(none -- numpy only)'}")
    print(f"Editable : {editable}")
    print(f"Clean    : {clean}")
    print("=" * 70)
    print()

    if clean:
        print("Removing any existing aiomfac_py install...")
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "aiomfac_py"])
        print()

    print("Installing aiomfac_py" + (f" with extras {extras_spec}" if extras else "") + " ...")
    target = f".{extras_spec}"
    pip_cmd = [sys.executable, "-m", "pip", "install", "-q"]
    if editable:
        pip_cmd.append("-e")
    pip_cmd.append(target)
    run(pip_cmd, cwd=REPO_DIR)
    print()

    print("Verifying installation...")
    check = subprocess.run(
        [sys.executable, "-c",
         "import aiomfac_py as a; "
         "print(f'aiomfac_py {a.__version__}  "
         "(validated against AIOMFAC-web commit {a.AIOMFAC_REFERENCE_COMMIT[:12]})')"],
        capture_output=True, text=True,
    )
    print(check.stdout.strip() or check.stderr.strip())

    if carbonate:
        c = subprocess.run([sys.executable, "-c", "import scipy; print(f'scipy {scipy.__version__} OK')"],
                           capture_output=True, text=True)
        print(("  " + c.stdout.strip()) if c.returncode == 0 else "  WARNING: scipy import failed:\n" + c.stderr)
    if smiles:
        c = subprocess.run([sys.executable, "-c", "from indigo import Indigo; print('epam.indigo OK, version', Indigo().version())"],
                           capture_output=True, text=True)
        print(("  " + c.stdout.strip()) if c.returncode == 0 else "  WARNING: epam.indigo import failed:\n" + c.stderr)

    print()
    print("=" * 70)
    print("aiomfac_py setup complete!")
    print("=" * 70)
    print()
    print("Next steps:")
    print("  from aiomfac_py import read_input_file, ActivityModel")
    if not carbonate:
        print("  (pass --carbonate above, or `pip install aiomfac_py[carbonate]`, to enable")
        print("   bisulfate/bicarbonate dissociation-equilibrium systems)")
    if not smiles:
        print("  (pass --smiles above, or `pip install aiomfac_py[smiles]`, to enable")
        print("   aiomfac_py.s2as, SMILES -> AIOMFAC subgroups)")
    print("  See notebooks/01_quickstart.ipynb for worked examples.")
    print("=" * 70)


if __name__ == "__main__":
    main()

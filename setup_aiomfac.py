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

NOTE on the 'tgml'/'tgml-smiles' extras (aiomfac_py.tgml_armeli, the machine-learning glass-transition-
temperature predictor of Armeli, Peters and Koop, 2023): this script deliberately has no flag for them,
simply because they're rarely needed and pull in scikit-learn (and, for SMILES input, rdkit). Nothing
stops installing them alongside the rest of this package -- they no longer pin an old scikit-learn/numpy
ABI (see aiomfac_py/tgml_armeli/__init__.py's module docstring) -- just run, after this script:
pip install -e ".[tgml,tgml-smiles]" (see the README for usage).
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

    # This script is normally executed with `%run` from the notebook's Quick Start cell, which runs it in the
    # *same* process as the rest of the notebook (that's the whole point of `%run` over `!python ...`). A pip
    # install -- especially an editable one -- registers its import hooks (a .pth file / __editable__ finder
    # under site-packages) that Python's `site` module only reads when a process *starts*, not while one is
    # already running. So without the fix below, `import aiomfac_py` can fail in the notebook's own cells right
    # after this script reports success, even though a fresh `python -c "import aiomfac_py"` subprocess works
    # fine (that subprocess *did* just start, so it picks the new .pth file up) -- which is exactly what made
    # the old version of this script's subprocess-based "Verifying installation" check misleadingly pass while
    # the very next notebook cell failed with ModuleNotFoundError. Adding src/ directly to sys.path sidesteps
    # relying on pip's editable-install machinery working correctly in an already-running process, and the
    # verification below now actually imports in-process so a real failure is reported here, not one cell later.
    src_dir = str(REPO_DIR / "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    print("Verifying installation (in this process, not a subprocess)...")
    for mod_name in ("aiomfac_py",):
        sys.modules.pop(mod_name, None)   # drop any earlier failed/partial import before retrying
    try:
        import aiomfac_py as _aiomfac
        print(f"aiomfac_py {_aiomfac.__version__}  "
              f"(validated against AIOMFAC-web commit {_aiomfac.AIOMFAC_REFERENCE_COMMIT[:12]})")
    except ImportError as e:
        print(f"WARNING: `import aiomfac_py` still failed after the sys.path fix: {e}")
        print(f"         sys.path now includes: {src_dir}")
        print("         Try Runtime -> Restart session in Colab, then re-run this cell.")

    if carbonate:
        try:
            import scipy
            print(f"  scipy {scipy.__version__} OK")
        except ImportError as e:
            print(f"  WARNING: scipy import failed: {e}")
    if smiles:
        try:
            from indigo import Indigo
            print("  epam.indigo OK, version", Indigo().version())
        except ImportError as e:
            print(f"  WARNING: epam.indigo import failed: {e}")

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

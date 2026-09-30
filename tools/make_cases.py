"""Generate the extra validation inputs (AIOMFAC-web input-file format) used for stage 3 of the port.

usage: python tools/make_cases.py <output_dir>

Every case avoids the species that the Fortran SetSystem completes automatically (H+, HSO4-, SO4--, HCO3-, CO3--),
because that part is not ported yet.  Subgroup IDs: 16 water; alcohols 145/146 (tail CH3/CH2), 149/150 (CH3/CH2 at OH),
141 + 151 (secondary alcohols as in the shipped example), 153 OH; 1 CH3, 18 CH3CO (acetone = 1 + 18);
cations 201 Li+, 202 Na+, 203 K+, 204 NH4+, 221 Ca2+, 223 Mg2+; anions 241 F-, 242 Cl-, 243 Br-, 244 I-, 245 NO3-.
"""
import sys
from pathlib import Path
import numpy as np

WATER = ("Water", [(16, 1)])
# Component names get a prefix on purpose: AIOMFAC-web recognises well-known names (e.g. "Methanol") and then looks up
# a SMILES-based glass-transition temperature for its viscosity module; if that lookup needs the external Python
# TgML script (not installed here) the run ends with error 26 even though the activity coefficients are fine.
ORG = {
    "x_Methanol": [(149, 1), (153, 1)],
    "x_Ethanol": [(145, 1), (150, 1), (153, 1)],
    "x_1-Propanol": [(145, 1), (146, 1), (150, 1), (153, 1)],
    "x_2-Butanol": [(141, 1), (145, 1), (146, 1), (151, 1), (153, 1)],
    "x_Acetone": [(1, 1), (18, 1)],
}
SALT = {
    "NaCl": [(202, 1), (242, 1)], "KCl": [(203, 1), (242, 1)], "LiCl": [(201, 1), (242, 1)],
    "NH4Cl": [(204, 1), (242, 1)], "NaBr": [(202, 1), (243, 1)], "KBr": [(203, 1), (243, 1)],
    "NaNO3": [(202, 1), (245, 1)], "KNO3": [(203, 1), (245, 1)], "NH4NO3": [(204, 1), (245, 1)],
    "CaCl2": [(221, 1), (242, 2)], "MgCl2": [(223, 1), (242, 2)],
}


def write_case(path, title, comps, basis, T, table):
    """comps: list of (name, [(subgroup, qty), ...]); table: (npoints, ncomp-1) fractions of components 2..N."""
    lines = ["Input file for AIOMFAC-web model", " ", "mixture components:"]
    for i, (name, subs) in enumerate(comps, 1):
        lines += ["----", f"component no.:\t{i:02d}", f"component name:\t'{name}'"]
        lines += [f"subgroup no., qty:\t{s:03d},\t{q:02d}" for s, q in subs]
    lines += ["----", "++++", "mixture composition and temperature:",
              f"mass fraction?\t{1 if basis == 'mass' else 0}", f"mole fraction?\t{1 if basis == 'mole' else 0}", "----",
              "point,\tT_K,\t" + ",\t".join(f"cp{i:02d}" for i in range(2, len(comps) + 1))]
    T = np.broadcast_to(np.asarray(T, dtype=float), (len(table),))
    for k, (t, row) in enumerate(zip(T, table), 1):
        lines.append(f"{k:02d}\t\t{t:.2f}\t" + "\t".join(f"{v:.10g}" for v in row))
    lines.append("====")
    Path(path).write_text("\n".join(lines) + "\n")


def main(out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    cases = {}
    # 0101: NaCl solution, dilute -> near saturation (mass fractions), 298.15 K
    w = np.concatenate([np.geomspace(1e-3, 0.02, 8), np.linspace(0.04, 0.26, 12)])
    cases["0101"] = ("NaCl sweep", [WATER, ("NaCl", SALT["NaCl"])], "mass", 298.15, w[:, None])
    # 0102 / 0103 / 0104: divalent cations and LiCl (high ionic strength)
    for n, salt, top in (("0102", "CaCl2", 0.40), ("0103", "MgCl2", 0.33), ("0104", "LiCl", 0.42)):
        w = np.concatenate([np.geomspace(2e-3, 0.03, 6), np.linspace(0.06, top, 12)])
        cases[n] = (f"{salt} sweep", [WATER, (salt, SALT[salt])], "mass", 298.15, w[:, None])
    # 0105: two salts with different ions -> additional electroneutral combinations (NaBr, KCl) in the system
    rng = np.random.default_rng(7)
    a, b = rng.uniform(0.002, 0.10, 14), rng.uniform(0.002, 0.10, 14)
    cases["0105"] = ("NaCl + KBr", [WATER, ("NaCl", SALT["NaCl"]), ("KBr", SALT["KBr"])], "mass", 298.15,
                     np.column_stack([a, b]))
    # 0106: water-ethanol-NaCl: ethanol and salt vary independently (mole fractions)
    xe = np.tile(np.array([0.0, 0.02, 0.06, 0.12, 0.25, 0.45]), 4)
    xs = np.repeat(np.array([0.002, 0.008, 0.02, 0.04]), 6)
    cases["0106"] = ("water + ethanol + NaCl", [WATER, ("x_Ethanol", ORG["x_Ethanol"]), ("NaCl", SALT["NaCl"])], "mole", 298.15,
                     np.column_stack([xe, xs]))
    # 0107: four organics + CaCl2, temperature sweep (mass fractions)
    T = np.linspace(275.0, 325.0, 11)
    row = [0.08, 0.06, 0.05, 0.07, 0.10]                       # methanol, 1-propanol, acetone, 2-butanol, CaCl2
    cases["0107"] = ("organics + CaCl2, T sweep",
                     [WATER, ("x_Methanol", ORG["x_Methanol"]), ("x_1-Propanol", ORG["x_1-Propanol"]), ("x_Acetone", ORG["x_Acetone"]),
                      ("x_2-Butanol", ORG["x_2-Butanol"]), ("CaCl2", SALT["CaCl2"])], "mass", T, np.tile(row, (len(T), 1)))
    # 0108: several salts with shared/unshared ions + one organic, random compositions and temperatures (mass fractions)
    rng = np.random.default_rng(11)
    n = 16
    tab = np.column_stack([rng.uniform(0.01, 0.20, n), rng.uniform(0.005, 0.06, n), rng.uniform(0.005, 0.06, n),
                           rng.uniform(0.005, 0.05, n), rng.uniform(0.005, 0.05, n)])
    cases["0108"] = ("random: 1-propanol + NaNO3 + NH4Cl + MgCl2 + KNO3",
                     [WATER, ("x_1-Propanol", ORG["x_1-Propanol"]), ("NaNO3", SALT["NaNO3"]), ("NH4Cl", SALT["NH4Cl"]),
                      ("MgCl2", SALT["MgCl2"]), ("KNO3", SALT["KNO3"])], "mass", rng.uniform(280.0, 320.0, n), tab)
    # 0109: very dilute limit, salt-free and organic-free edge points (component with zero abundance)
    w = np.array([0.0, 1e-9, 1e-7, 1e-5, 1e-4, 1e-3])
    cases["0109"] = ("dilute limit + zero-abundance components", [WATER, ("x_Methanol", ORG["x_Methanol"]), ("NaBr", SALT["NaBr"])],
                     "mass", 298.15, np.column_stack([[0.0, 0.05, 0.0, 0.2, 0.05, 0.0], w]))
    for k, (title, comps, basis, T, table) in cases.items():
        write_case(out / f"input_{k}.txt", title, comps, basis, T, np.asarray(table))
        print(k, f"{title:55s} {len(table):3d} points, {len(comps)} components, {basis} fractions")


if __name__ == "__main__":
    main(sys.argv[1])

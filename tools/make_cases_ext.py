"""Generate stage-4 validation inputs: Qcca/Rcc coverage (NH4+/H+/HSO4-), extended organic functional groups
(ester, ether, carboxylic acid, aromatic ring, amine) not exercised by tools/make_cases.py, and PEG systems
(subgroup 154).

usage: python tools/make_cases_ext.py <output_dir>

Every input includes an explicit "@@@@ calculation options / smiles-based pure-component method? 0" section
(armeliON = .false. in Fortran) so the reference Fortran run doesn't need the TgML_Armeli machine-learning
viscosity lookup (a heavy, separately-licensed dependency not installed here, and irrelevant to activity
coefficients: armeliON only affects PureCompViscosity, verified bit-identical debug_terms.txt with/without it
for an existing case). Subgroup IDs: 16 water; 1 CH3, 2 CH2 (generic alkyl); 21 CH3COO (ester); 24 CH3O (ether);
42 COOH (acid); 9 ACH, 11 ACCH3 (aromatic ring, toluene); 28 CH3NH2 (amine); 202 Na+, 242 Cl-; 204 NH4+, 248 HSO4-.
"""
import sys
from pathlib import Path
import numpy as np

WATER = ("x_Water", [(16, 1)])
NACL = ("x_NaCl", [(202, 1), (242, 1)])


def write_case(path, title, comps, basis, T, table, armeli=False):
    lines = ["Input file for AIOMFAC-web model", " ", "mixture components:"]
    for i, (name, subs) in enumerate(comps, 1):
        lines += ["----", f"component no.:\t{i:02d}", f"component name:\t'{name}'"]
        lines += [f"subgroup no., qty:\t{s:03d},\t{q:02d}" for s, q in subs]
    lines += ["----", "@@@@", "calculation options:", f"smiles-based pure-component method?\t{1 if armeli else 0}", "----"]
    lines += ["++++", "mixture composition and temperature:",
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

    # cc01: NH4HSO4 alone -- dissociates to NH4+, H+, HSO4-, SO4--, activating Qcca/Rcc (Fortran comment:
    # "so far only used for [NH4+|H+|HSO4-] containing mixtures"), dilute to concentrated (mass fractions)
    w = np.concatenate([np.geomspace(2e-3, 0.02, 6), np.linspace(0.04, 0.30, 12)])
    cases["cc01"] = ("NH4HSO4 sweep (Qcca/Rcc)", [WATER, ("x_NH4HSO4", [(204, 1), (248, 1)])], "mass", 298.15, w[:, None])

    # cc02: ester (ethyl acetate: CH3 + CH2 + CH3COO) + NaCl, water varied
    rng = np.random.default_rng(21)
    a = rng.uniform(0.01, 0.20, 10); b = rng.uniform(0.01, 0.15, 10)
    cases["cc02"] = ("ethyl acetate + NaCl",
                     [WATER, ("x_EthylAcetate", [(1, 1), (2, 1), (21, 1)]), NACL], "mass", 298.15,
                     np.column_stack([a, b]))

    # cc03: ether (dimethyl ether: CH3 + CH3O) + NaCl
    rng = np.random.default_rng(22)
    a = rng.uniform(0.01, 0.20, 10); b = rng.uniform(0.01, 0.15, 10)
    cases["cc03"] = ("dimethyl ether + NaCl",
                     [WATER, ("x_DimethylEther", [(1, 1), (24, 1)]), NACL], "mass", 298.15,
                     np.column_stack([a, b]))

    # cc04: carboxylic acid (acetic acid: CH3 + COOH) + NaCl
    rng = np.random.default_rng(23)
    a = rng.uniform(0.01, 0.20, 10); b = rng.uniform(0.01, 0.15, 10)
    cases["cc04"] = ("acetic acid + NaCl",
                     [WATER, ("x_AceticAcid", [(1, 1), (42, 1)]), NACL], "mass", 298.15,
                     np.column_stack([a, b]))

    # cc05: aromatic ring (toluene: 5x ACH + ACCH3), electrolyte-free -- MR main-group<->ion parameters for the
    # aromatic main group (AC/ACH) are not defined for any cation (errorflagmix 1), so no salt component here;
    # this exercises the SR/LR aromatic-ring pathway with water only.
    w = np.concatenate([np.geomspace(1e-3, 0.02, 6), np.linspace(0.05, 0.60, 10)])
    cases["cc05"] = ("toluene + water (electrolyte-free)", [WATER, ("x_Toluene", [(9, 5), (11, 1)])], "mass", 298.15, w[:, None])

    # cc06: amine (methylamine: CH3NH2), electrolyte-free -- same reason as cc05 (amine main group has no
    # defined MR ion-interaction parameters either).
    w = np.concatenate([np.geomspace(1e-3, 0.02, 6), np.linspace(0.05, 0.50, 10)])
    cases["cc06"] = ("methylamine + water (electrolyte-free)", [WATER, ("x_Methylamine", [(28, 1)])], "mass", 298.15, w[:, None])

    # cc07: PEG oligomer (HO-CH2CH2-(OCH2CH2)3-OH: CH2[OH] ends + 3x CH2OCH2[PEG] repeats), electrolyte-free --
    # exercises the subgroup-154 R/Q override in the SR combinatorial term (isPEGsystem). CH2[OH] (150) needs a
    # matching count of OH (153) subgroups (Fortran errorflagmix 13 otherwise).
    PEG = ("x_PEG", [(150, 2), (153, 2), (154, 3)])
    w = np.concatenate([np.geomspace(1e-3, 0.02, 6), np.linspace(0.05, 0.50, 10)])
    cases["cc07"] = ("PEG oligomer + water (electrolyte-free)", [WATER, PEG], "mass", 298.15, w[:, None])

    # cc08: same PEG oligomer + ammonium sulfate (NH4+/SO4--) -- additionally exercises the isPEGsystem MR
    # main-group<->ion override for CHn[OH,PEG] (main group 52/68) <-> NH4+/SO4--.
    rng = np.random.default_rng(24)
    a = rng.uniform(0.02, 0.20, 10); b = rng.uniform(0.01, 0.10, 10)
    cases["cc08"] = ("PEG oligomer + (NH4)2SO4",
                     [WATER, PEG, ("x_AmmoniumSulfate", [(204, 2), (261, 1)])], "mass", 298.15,
                     np.column_stack([a, b]))

    for k, (title, comps, basis, T, table) in cases.items():
        write_case(out / f"input_{k}.txt", title, comps, basis, T, np.asarray(table))
        print(k, f"{title:40s} {len(table):3d} points, {len(comps)} components, {basis} fractions")


if __name__ == "__main__":
    main(sys.argv[1])

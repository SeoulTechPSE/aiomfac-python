"""Generate notebooks/aiomfac_validation_11_llps_song2012.ipynb: external check of the phase-equilibrium solver
against Song, Marcolli, Krieger, Zuend and Peter (2012, ACP 12, 2691-2712).

A. LLPS onset RH (highest RH with two liquids, AS crystallization suppressed) versus the dry mass fraction of AS for
   the C5, C6 and C7 dicarboxylic-acid mixtures + AS and for one acid of each mixture + AS, at 298.15 K, compared with
   (i) the measured onset upon drying (their Fig. 4) and (ii) their own AIOMFAC liquid-liquid calculation with the
   global Gibbs-energy minimization of Zuend et al. (2010) (their Fig. 8). Literature values are read from the
   figures (about +/-1 % RH, +/-0.01 in water activity).
B. LLPS / no-LLPS classification of the 24 organic + AS systems of their Table 2 (observed at 30 % < RH < 100 %,
   mfd(AS) > 0.1): predicted LLPS = two liquids at RH >= 0.40 (above the AS efflorescence RH of these particles,
   e.g. 39 % for C5/AS) for at least one of mfd(AS) = 0.2, 0.5, 0.8.
Results: llps_song2012_results.json; figure ../figures_revised/figS1_song2012.png.
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md("""
# External check of the phase-equilibrium solver: Song et al. (2012)

Song et al. (2012) measured liquid-liquid phase separation (LLPS) in particles of ammonium sulfate (AS) with three
mixtures of dicarboxylic acids (C5, O:C 0.80; C6, 0.67; C7, 0.57) and compared the onset RH with AIOMFAC
calculations based on the global Gibbs-energy minimization of Zuend et al. (2010). The solver used in this study
(`aiomfac_py.phase_equilibrium`) is an independent implementation (interior-point minimization with a tangent-plane
stability test) on the same thermodynamic model, so it should reproduce their calculated onset RH, while both may
differ from the measurements.
""")

code(r'''
import json, math, time, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem.Descriptors import MolWt
import aiomfac_py
from aiomfac_py import Component
from aiomfac_py.s2as import smiles_to_components
from aiomfac_py.phase_equilibrium import PhaseEquilibrium
warnings.filterwarnings("ignore")
print("aiomfac_py", aiomfac_py.__version__, aiomfac_py.__file__)
T = 298.15
MW_AS = 132.14

def component(i, name, smi):
    r = smiles_to_components([smi], water_as_component1=True)
    return Component(i, name, r.components[1].subgroups), int(r.n_unmatched_atoms[0])

def make_pe(organics):
    comps = [component(2 + i, n, s)[0] for i, (n, s) in enumerate(organics.items())]
    return PhaseEquilibrium(comps, ["NH4+", "SO4--"], T_K=T)

def feed(organics, mfd):
    """equal masses of the organics; mfd = mass(AS) / mass(organics + AS)"""
    f = {n: (1 - mfd) / len(organics) / MolWt(Chem.MolFromSmiles(s)) for n, s in organics.items()}
    f.update({"NH4+": 2 * mfd / MW_AS, "SO4--": mfd / MW_AS})
    return f

def onset_rh(pe, fd, rh_hi=0.99, rh_lo=0.30, step=0.02, fine=0.005):
    """highest RH with two liquids (solids suppressed): coarse downward scan, then refinement"""
    rh = rh_hi
    while rh >= rh_lo - 1e-9:
        if pe.solve(fd, rh, solids="none").n_liquids >= 2:
            hi = min(rh + step, rh_hi); r = hi
            while r > rh + 1e-9:                       # refine between rh and rh + step
                if pe.solve(fd, r, solids="none").n_liquids >= 2:
                    return round(r, 3)
                r -= fine
            return round(rh, 3)
        rh = round(rh - step, 4)
    return None
''')

md("## A. Onset RH versus mfd(AS) for the C5, C6 and C7 systems")

code(r'''
SYSTEMS = {
    "C5": {"glutaric acid": "OC(=O)CCCC(=O)O", "methylsuccinic acid": "OC(=O)C(C)CC(=O)O",
           "dimethylmalonic acid": "OC(=O)C(C)(C)C(=O)O"},
    "C6": {"2-methylglutaric acid": "OC(=O)C(C)CCC(=O)O", "3-methylglutaric acid": "OC(=O)CC(C)CC(=O)O",
           "2,2-dimethylsuccinic acid": "OC(=O)C(C)(C)CC(=O)O"},
    "C7": {"3-methyladipic acid": "OC(=O)CC(C)CCC(=O)O", "3,3-dimethylglutaric acid": "OC(=O)CC(C)(C)CC(=O)O",
           "diethylmalonic acid": "OC(=O)C(CC)(CC)C(=O)O"},
    "glutaric acid": {"glutaric acid": "OC(=O)CCCC(=O)O"},
    "2-methylglutaric acid": {"2-methylglutaric acid": "OC(=O)C(C)CCC(=O)O"},
    "3,3-dimethylglutaric acid": {"3,3-dimethylglutaric acid": "OC(=O)CC(C)(C)CC(=O)O"},
}
MFD = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# Read from Song et al. (2012): measured onset upon drying (Fig. 4, % RH -> fraction) and their AIOMFAC
# coexistence curve maximum at each mfd(AS) (Fig. 8b, e, h and c, f, i; water activity).
MEASURED = {"C5": {},        # no LLPS observed above the efflorescence RH (39 %), OIR 2:1, 1:1, 1:4
            "C6": {0.2: 0.73, 0.3: 0.74, 0.5: 0.73, 0.7: 0.73, 0.8: 0.735, 0.9: 0.725},
            "C7": {0.1: 0.89, 0.2: 0.88, 0.3: 0.89, 0.5: 0.90, 0.7: 0.88, 0.8: 0.875, 0.9: 0.88}}
SONG_MODEL = {"C5": {0.4: 0.50, 0.5: 0.585, 0.6: 0.61, 0.7: 0.62, 0.8: 0.62, 0.9: 0.60},
              "C6": {0.3: 0.79, 0.4: 0.81, 0.5: 0.815, 0.6: 0.815, 0.7: 0.81, 0.8: 0.79, 0.9: 0.77},
              "C7": {0.1: 0.95, 0.2: 0.945, 0.3: 0.94, 0.4: 0.935, 0.5: 0.93, 0.6: 0.92, 0.7: 0.91, 0.8: 0.90, 0.9: 0.88}}
ONSET = {}
t0 = time.time()
for name, orgs in SYSTEMS.items():
    pe = make_pe(orgs)
    ONSET[name] = {m: onset_rh(pe, feed(orgs, m)) for m in MFD}
    print(f"{name:26s} " + " ".join(f"{m:.1f}:{'  -  ' if v is None else f'{v:.3f}'}" for m, v in ONSET[name].items())
          + f"   ({time.time() - t0:.0f} s)", flush=True)
''')

code(r'''
CMP = []
for sysname in ("C5", "C6", "C7"):
    for m in MFD:
        ours = ONSET[sysname][m]
        CMP.append(dict(system=sysname, mfd=m, ours=ours, song_model=SONG_MODEL[sysname].get(m),
                        measured=MEASURED[sysname].get(m)))
def diffs(key):
    d = [c["ours"] - c[key] for c in CMP if c["ours"] is not None and c[key] is not None]
    return d
for sysname in ("C5", "C6", "C7"):
    dm = [c["ours"] - c["song_model"] for c in CMP if c["system"] == sysname and c["ours"] and c["song_model"]]
    dx = [c["ours"] - c["measured"] for c in CMP if c["system"] == sysname and c["ours"] and c["measured"]]
    print(f"{sysname}: ours - Song AIOMFAC: mean {np.mean(dm):+.3f}, max |.| {np.max(np.abs(dm)):.3f} (n={len(dm)})"
          + (f"; ours - measured: mean {np.mean(dx):+.3f} (n={len(dx)})" if dx else "; no LLPS measured above ERH"))
d_all = diffs("song_model")
print(f"all: ours - Song AIOMFAC mean {np.mean(d_all):+.3f}, mean |.| {np.mean(np.abs(d_all)):.3f}, max |.| {np.max(np.abs(d_all)):.3f}")
''')

code(r'''
fig, axs = plt.subplots(1, 3, figsize=(10, 3.4), sharey=True)
for ax, sysname in zip(axs, ("C5", "C6", "C7")):
    o = ONSET[sysname]
    ax.plot([m for m in MFD if o[m]], [o[m] * 100 for m in MFD if o[m]], "o-", color="#2b6cb0", label="this study")
    sm = SONG_MODEL[sysname]
    ax.plot(list(sm), [v * 100 for v in sm.values()], "s--", color="#c05621", mfc="none",
            label="AIOMFAC, Song et al. (2012)")
    me = MEASURED[sysname]
    if me:
        ax.plot(list(me), [v * 100 for v in me.values()], "k^", label="measured, Song et al. (2012)")
    else:
        ax.axhspan(30, 39.2, color="0.85", lw=0)
        ax.text(0.5, 34.5, "no LLPS observed above\nthe efflorescence RH", ha="center", va="center", fontsize=8)
    ax.set_title(f"{sysname}/AS/H$_2$O", fontsize=10)
    ax.set_xlabel("mf$_d$(AS)"); ax.set_xlim(0, 1); ax.set_ylim(30, 100)
axs[0].set_ylabel("onset RH of LLPS (%)"); axs[2].legend(frameon=False, fontsize=7.5, loc="lower left")
fig.tight_layout(); fig.savefig("../figures_revised/figS1_song2012.png", dpi=300); plt.close(fig)
print("wrote ../figures_revised/figS1_song2012.png")
''')

md("## B. LLPS classification of the systems of Song et al. (2012), Table 2")

code(r'''
TABLE2 = [  # (compound, smiles or mixture key, observed LLPS)
    ("glycerol", "OCC(O)CO", False), ("1,2,4-butanetriol", "OCCC(O)CO", False),
    ("levoglucosan", "OC1C2COC(O2)C(O)C1O", False), ("pentanedioic acid", "OC(=O)CCCC(=O)O", False),
    ("citric acid", "OC(=O)CC(O)(CC(O)=O)C(O)=O", False), ("propanedioic acid", "OC(=O)CC(=O)O", False),
    ("C5 mixture", "C5", False),
    ("1,2-hexanediol", "CCCCC(O)CO", True), ("1,4-butanediol", "OCCCCO", True),
    ("1,2,6-hexanetriol", "OCCCCC(O)CO", True), ("2,4-pentanediol", "CC(O)CC(C)O", True),
    ("2,5-hexanediol", "CC(O)CCC(C)O", True), ("1,7-heptanediol", "OCCCCCCCO", True),
    ("monomethyl octanedioate", "COC(=O)CCCCCCC(=O)O", True),
    ("vanillylmandelic acid", "COc1cc(C(O)C(=O)O)ccc1O", True),
    ("2,5-dihydroxybenzoic acid", "OC(=O)c1cc(O)ccc1O", True),
    ("2,2-dimethylbutanedioic acid", "OC(=O)C(C)(C)CC(=O)O", True),
    ("PEG-400 (n = 9)", "OCC" + "OCC" * 8 + "O", True),
    ("C6 mixture", "C6", True), ("C7 mixture", "C7", True)]
CLS = []
for name, key, obs in TABLE2:
    orgs = SYSTEMS[key] if key in SYSTEMS else {name: key}
    unmatched = sum(component(2, n, s)[1] for n, s in orgs.items())
    if unmatched:
        CLS.append(dict(compound=name, observed=obs, predicted=None, unmatched=unmatched)); continue
    pe = make_pe(orgs)
    on = {m: onset_rh(pe, feed(orgs, m), rh_lo=0.40) for m in (0.2, 0.5, 0.8)}
    top = max([v for v in on.values() if v is not None], default=None)
    pred = top is not None
    CLS.append(dict(compound=name, observed=obs, predicted=pred, onset_max=top, onset=on))
    print(f"{name:30s} observed {'LLPS' if obs else 'no LLPS':8s} predicted {'LLPS' if pred else 'no LLPS':8s} "
          f"onset max {top}  {'OK' if pred == obs else 'MISMATCH'}", flush=True)
ok = [c for c in CLS if c["predicted"] is not None]
n_ok = sum(c["predicted"] == c["observed"] for c in ok)
print(f"agreement {n_ok}/{len(ok)}; not decomposed: {[c['compound'] for c in CLS if c['predicted'] is None]}")
''')

code(r'''
OUT = dict(aiomfac_py_version=aiomfac_py.__version__, T_K=T, onset=ONSET, comparison=CMP,
           measured_read_from_fig4=MEASURED, song_model_read_from_fig8=SONG_MODEL, classification=CLS)
json.dump(OUT, open("llps_song2012_results.json", "w"), indent=1, default=float)
print("wrote llps_song2012_results.json")
''')

nb["cells"] = cells
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "../notebooks/aiomfac_validation_11_llps_song2012.ipynb")
print("wrote ../notebooks/aiomfac_validation_11_llps_song2012.ipynb")

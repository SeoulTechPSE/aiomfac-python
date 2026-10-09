"""Example: X--RH phase diagram of (NH4)2SO4 + NH4NO3 particles at 298.15 K with AIOMFAC (aiomfac_py.diagram).
X is the mole fraction of (NH4)2SO4 in the dry salt.  About 3 min on one CPU thread.
usage: python tools/diagram_as_an.py [out_prefix]      (needs scipy and matplotlib: pip install aiomfac_py[plot])
Note: the solid database has no NH4NO3-(NH4)2SO4 double salts (2NH4NO3.(NH4)2SO4, 3NH4NO3.(NH4)2SO4), so the dry
region and the mutual deliquescence RH are those of a mixture of the two single salts."""
import json
import sys
import time
import warnings

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from aiomfac_py.diagram import phase_map, plot_phase_map  # noqa: E402
from aiomfac_py.phase_equilibrium import PhaseEquilibrium  # noqa: E402

warnings.filterwarnings("ignore")
out = sys.argv[1] if len(sys.argv) > 1 else "as_an"
pe = PhaseEquilibrium([], ["NH4+", "SO4--", "NO3-"], T_K=298.15)


def feed(x):                                     # 1 mol of dry salt: x (NH4)2SO4 + (1 - x) NH4NO3
    return {"NH4+": 2 * x + (1 - x), "SO4--": x, "NO3-": 1 - x}


xs = np.linspace(0.0, 1.0, 11)
xs[0], xs[-1] = 0.005, 0.995
t0 = time.time()
pm = phase_map(pe, feed, xs, x_label="x((NH$_4$)$_2$SO$_4$) in dry salt", n=32, verbose=True)
print(f"{time.time() - t0:.0f} s, {sum(t.n_solves for t in pm.traces)} solves, "
      f"{sum(len(t.failed) for t in pm.traces)} not converged (dropped)")
json.dump(pm.to_records(), open(out + "_boundaries.json", "w"), indent=1)
ax = plot_phase_map(pm)
ax.set_title("(NH$_4$)$_2$SO$_4$–NH$_4$NO$_3$–H$_2$O, 298.15 K (AIOMFAC)", fontsize=10)
ax.figure.set_size_inches(7.6, 4.8)
plt.tight_layout()
plt.savefig(out + ".png", dpi=150)

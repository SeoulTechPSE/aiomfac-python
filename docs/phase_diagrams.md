# Phase diagrams with `aiomfac_py.diagram` — user manual

`aiomfac_py.diagram` builds phase diagrams of aerosol particles on top of the equilibrium solver
`PhaseEquilibrium` (water + organics + ions, liquid–liquid–solid equilibrium at fixed T and RH) and of the AIOMFAC
activity model. It produces the kinds of diagrams of the UHAERO papers (Amundson et al., 2006, 2007):

* **composition–RH diagrams** (e.g. ammonium fraction X versus RH): regions of equal phase state, deliquescence
  and phase-separation boundaries, contour fields of water uptake or pH;
* **deliquescence curves**: water content, particle mass growth or pH along RH for one particle composition;
* **deliquescence-RH maps**: the full deliquescence RH and the first solid over a two-parameter composition space;
* **liquid–liquid diagrams of neutral mixtures**: binary Gibbs-energy curves with miscibility gaps, and ternary
  diagrams with one-, two- and three-liquid regions, tie lines and equilibrium activities.

A worked example of every function is in `notebooks/07_phase_diagrams.ipynb`. The scripts
`tools/uhaero2006_figures.py` and `tools/uhaero2007_figures.py` reproduce the figures of the two UHAERO papers with
AIOMFAC, and `tools/diagram_as_an.py` draws the (NH4)2SO4–NH4NO3–H2O diagram.

---

## Contents

1. [Installation](#1-installation)
2. [Quick start](#2-quick-start)
3. [Concepts](#3-concepts)
4. [Function reference](#4-function-reference)
5. [Recipes](#5-recipes)
6. [How it works](#6-how-it-works)
7. [Performance](#7-performance)
8. [Limitations](#8-limitations)
9. [References](#9-references)

---

## 1. Installation

The tracing functions need SciPy (as `PhaseEquilibrium` does); the plotting functions also need Matplotlib:

```bash
pip install "aiomfac_py[plot] @ git+https://github.com/SeoulTechPSE/aiomfac-python.git"
# organics given as SMILES additionally need the "smiles" extra
```

```python
from aiomfac_py.phase_equilibrium import PhaseEquilibrium
from aiomfac_py import diagram as dg
```

## 2. Quick start

Deliquescence RH of ammonium sulfate, and an X–RH diagram of (NH4)2SO4 + NH4NO3 particles:

```python
import matplotlib.pyplot as plt
from aiomfac_py.phase_equilibrium import PhaseEquilibrium
from aiomfac_py import diagram as dg

pe = PhaseEquilibrium([], ["NH4+", "SO4--"], T_K=298.15)
t = dg.trace(pe, {"NH4+": 2.0, "SO4--": 1.0})
for b in t.boundaries:
    print(f"{b.rh:.4f}  {b.kind}: {b.below.label} -> {b.above.label}")
# 0.7987  deliquescence: ammonium_sulfate -> L   (binary_saturation: a_w = 0.79875)

pe = PhaseEquilibrium([], ["NH4+", "SO4--", "NO3-"], T_K=298.15)
feed = lambda x: {"NH4+": 2 * x + (1 - x), "SO4--": x, "NO3-": 1 - x}   # x = (NH4)2SO4 fraction of the dry salt
pm = dg.phase_map(pe, feed, [0.005, 0.2, 0.4, 0.6, 0.8, 0.995], x_label="x((NH4)2SO4)")
dg.plot_phase_map(pm)
plt.show()
```

## 3. Concepts

**Feed.** A particle is given by its non-water content in mol, a dict of organic names (the `Component.name` of the
organics passed to `PhaseEquilibrium`) and ion keys (`"NH4+"`, `"H+"`, `"SO4--"`, `"NO3-"`, `"Na+"`, `"Cl-"`, …). The
feed must be electroneutral. Water is open: at each RH, the liquids take up water until a_w = RH.

**Phase state.** `PhaseState(n_liquids, solids)` — the number of liquid phases and the sorted tuple of the solids
present (keys of `aiomfac_py.solids.SOLIDS`). `PhaseState.label` gives a readable form, e.g. `"L + letovicite"`,
`"2L"`, `"ammonium_nitrate + AS_3AN"` (dry).

**Boundary.** A change of phase state between two RHs; `Boundary.below` / `above` are the states on either side,
`Boundary.rh` the midpoint of the final bracket `[rh_lo, rh_hi]`, `Boundary.kind` one of

| kind | meaning (with increasing RH) |
|---|---|
| `deliquescence` | a solid dissolves |
| `precipitation` | a solid appears |
| `phase separation` | the number of liquids changes, solids unchanged |
| `solid transition` | one solid replaces another (e.g. a hydrate change, or a peritectic in the presence of a liquid) |
| `other` | anything else |

**Modes.** `"equilibrium"`: all candidate solids of the solver's database (or the list `solids=` passed through).
`"metastable"`: no solids — supersaturated liquids, so only liquid–liquid phase separation is traced.

**Composition parameter.** A diagram is drawn over one composition parameter X, given as a function
`feed_of_x(x) -> feed`. Any parametrization works: the ammonium fraction X = NH4+/(NH4+ + H+), the salt fraction of a
dry mixture, an organic/inorganic mixing ratio, …

## 4. Function reference

### 4.1 Tracing along RH

#### `trace(pe, feed, *, rh_min=0.05, rh_max=0.98, n=32, mode="equilibrium", tol=1e-4, rh_grid=None, warm=True, **solve_kw) -> Trace`

Phase states of `feed` over RH and the RH of every change of state.

* `pe` — a `PhaseEquilibrium` for the system (organics, ions, T; optionally a surrogate `liquid_model=`).
* `n`, `rh_min`, `rh_max` or `rh_grid` — the initial RH grid. Solves go from high to low RH with warm starts.
* `tol` — every interval between two converged solves of different state is bisected until it is narrower than
  `tol`. A state found inside an interval splits it, so several changes within one grid interval are all found.
* `mode` — `"equilibrium"` or `"metastable"` (see §3).
* `**solve_kw` — passed to `PhaseEquilibrium.solve` (`solids=[...]`, `max_liquids=`, …).

A state that exists only in an RH interval narrower than the grid spacing can be missed; raise `n` if needed.

`Trace` attributes and methods:

| name | content |
|---|---|
| `rh`, `results` | every RH solved (grid and bisection points, increasing) and the `PhaseEquilibriumResult`s |
| `states` | `PhaseState` of every result |
| `boundaries` | `Boundary` list, increasing RH |
| `failed` | RHs whose solve did not converge (dropped; see §6.2) |
| `n_solves` | number of `solve` calls |
| `rh_range` | the requested (lowest, highest) RH |
| `intervals()` | `[(rh_from, rh_to, PhaseState)]` covering the requested range |
| `state_at(rh)` | state of the nearest solve |
| `signature` | tuple of `(below, above)` of all boundaries: equal signatures = same topology |

#### `rh_profile(pe, feed, rh_grid, *, mode="equilibrium", refine=None, **solve_kw) -> dict`

Properties of the equilibrium states along RH (`particle_properties` and `ionic_strength`), solved from high to low
RH with warm starts. Without `refine` the arrays follow `rh_grid`, with NaN where a solve did not converge. With
`refine` (an RH tolerance, e.g. 1e-4), each change of phase state between neighbouring grid points is located by
bisection, and the states just below and above it are added. The arrays are then sorted by RH and hold only
converged states, so a deliquescence step is drawn as a vertical jump at its RH.

#### `particle_properties(pe, feed, result) -> dict`

Properties of one equilibrium state:

* `rel_mass` — particle mass relative to the dry particle of the same feed, W_p/W_dry = 1 + W_water/W_dry
  (the "relative particle mass" of Amundson et al., 2006);
* `water_g` — liquid water in g (per the feed amounts in mol);
* `pH` — −log10 a(H+) on the **molal** scale in the liquid that holds most water (NaN without H+ or liquid). The
  UHAERO papers use the mole-fraction scale: pH_x = pH − log10(0.018015) = pH + 1.744;
* `n_liquids`, `solids`, `status`.

#### `deliquescence_point(pe, feed, *, rh_max=0.98, rh_min=0.02, step=0.03, tol=1e-4, method="equilibrium", **solve_kw)`

The full deliquescence RH (the highest RH at which a solid is present) and the solids just below it, as
`(rh, solids)`, or `(None, ())` if no solid forms down to `rh_min`.

* `method="equilibrium"` — downward scan with full equilibrium solves, then bisection.
* `method="si"` — follows the solid-free liquid and finds the RH at which the largest saturation index of the
  candidate solids reaches zero (Brent's method); `solids` is then the solid that saturates first. Same result for
  particles with one liquid phase above the deliquescence RH (inorganic systems), two to three times faster.

### 4.2 Composition–RH diagrams

#### `phase_map(pe, feed_of_x, x_values, *, x_label="X", refine_x=True, x_tol=None, verbose=False, **trace_kw) -> PhaseMap`

`trace` at every x (keywords `**trace_kw` as for `trace`). With `refine_x`, every interval of x whose two traces
have different topology (`Trace.signature`) is bisected until it is narrower than `x_tol` (default 1/100 of the x
range), so that boundaries end close to where they meet (eutonic and peritectic points).

`PhaseMap` attributes and methods: `x`, `traces`, `x_label`, `mode`; `states()` (all states that occur);
`boundary_lines()` (`{(below, above): (x, rh)}`, for analysis); `to_records()` (flat list of boundaries for CSV/JSON).

#### `trace_boundaries(pe, feed_of_x, pm, *, tol=5e-4, h0=0.02, h_min=0.002, h_max=0.05, delta_max=0.05, rh_samples=40, snap=None, solve_timeout=None, x_lines=(), junction_rounds=2, dangling_rounds=2, refine_rounds=1, refine_at=(), refine_n=13, refine_size=0.03, method="states", cache=None, checkpoint=None, checkpoint_every=200, verbose=False, **solve_kw) -> BoundaryCurves`

Traces every boundary line of a phase map continuously in the (x, RH) plane, so that the lines are smooth and meet
at the junctions (eutonic and peritectic points) instead of being interpolated between the computed compositions.
`pe` and `feed_of_x` must be those used for `pm`; the boundary points of `pm`'s traces are the starting points.
Lengths are in units of the diagram (x and RH ranges scaled to 1):

* `tol` — accuracy of each point across the line;
* `h0`, `h_min`, `h_max` — first, smallest and largest step along the line (a line end is located to within `h_min`);
* `delta_max` — widest bracket searched across the line;
* `rh_samples` — RH levels at which changes of state between neighbouring traces are checked, to find lines that
  cross no trace (vertical boundaries at fixed composition);
* `snap` — line ends closer than this (default `2.5 h_min`) are joined at a common junction point;
* `solve_timeout` — seconds after which a solve is abandoned and counted as not converged (POSIX, main thread), for
  compositions where the solver is very slow;
* `x_lines` — compositions where a vertical boundary is expected (e.g. the stoichiometric X of a salt); such lines
  are also detected automatically;
* `junction_rounds` — rounds of the search for missing lines around the line ends (0 to skip);
* `dangling_rounds` — rounds of the search around line ends that no other line reaches (circles of 3, 6 and
  12 `h_min`; 0 to skip);
* `method` — `"states"` (default; the tracing described here) or `"zpf"` (zero-phase-fraction lines, Sect. 6.5a);
* `refine_rounds`, `refine_at`, `refine_n`, `refine_size` — local refinement where the topology rules are still
  violated, and at the points `refine_at` [(x, RH)]: a `refine_n` × `refine_n` grid of half-width `refine_size`
  (unit square) is scanned, and every change of state that no line explains is traced with half the step and
  tolerance;
* `cache` — a dict that keeps the solves; pass the same dict again (same `pm`) to reuse them, e.g. after changing
  options;
* `verbose=True` prints each line as it is traced (`2` also prints every step).

`BoundaryCurves` holds `curves` (a list of `BoundaryCurve`), `x_range`, `rh_range`, `n_solves` and
`to_records()` (for JSON). A `BoundaryCurve` has `left`, `right` (the `PhaseState`s on either side; `left` is
at higher RH when the points run towards higher x), `x`, `rh` (arrays), `ends` (how each end stopped:
`"junction"`, `"edge"` or `"merged"`), `kind` and `sides(i)` ((state at lower RH, state at higher RH) at segment
`i`).

#### `plot_boundary_curves(bc, ax, *, color="#222222", lw=1.3)`

Draws the lines of a `BoundaryCurves` on an axes (`plot_phase_map(..., curves=bc)` calls it).

#### `plot_phase_map(pm, ax=None, *, colors=None, rh_points=400, x_points=300, legend=True, metastable=None, curves=None, show_failed=False)`

Draws the diagram on a matplotlib axes and returns it:

* regions coloured by phase state (`colors`: `{PhaseState: colour}`; pass white for a line drawing);
* region outlines as boundary lines — boundaries along RH and along composition alike;
* phase states that occur at a single interior composition only (e.g. the exact stoichiometry of letovicite in the
  NH4+/H+/SO4-- system) as dotted vertical lines;
* `metastable=` a second `PhaseMap` (mode `"metastable"`) whose boundaries are overlaid in grey;
* `rh_points`, `x_points` — resolution of the region grid; raise them for zoomed views;
* `curves=` the result of `trace_boundaries`: the boundaries are drawn as the traced lines and the regions are
  filled between them;
* `show_failed=True` marks the points where solves did not converge (grey dots; from the traces and, with
  `curves`, from `BoundaryCurves.failed`), i.e. where the diagram is not resolved.

Without `curves`, the regions between computed compositions are interpolated: each boundary of the nearer trace is interpolated
linearly in x towards the boundary of the same kind (same states below and above, closest in RH) of the other trace.

#### `label_regions(ax, pm, labels, *, min_cells=40, fontsize=8, rh_points=400, x_points=300, curves=None, legend=True, min_marker_cells=3, cache=None) -> [(number, label)]`

Writes a label in every region (and next to every singular line), built from `labels` = {solid key: short label},
e.g. `{"ammonium_sulfate": "A", "letovicite": "B"}` → "L+A", "A+B". Pass the same `curves` as to `plot_phase_map`,
and call it after `plot_phase_map` (the label sizes are judged from the size of the axes).

* With `curves`, the regions are the areas enclosed by the drawn lines (on a raster of half a point; gaps of about a
  point at junctions are closed). The state of an area is the most frequent state among the solves inside it: those
  of the traces and, with `cache` (the cache dict of `trace_boundaries`), the many solves along the lines.
  Without `curves`, the regions are the connected parts of each state on the region grid.
* A label is written inside a region if the region has at least `min_cells` cells and the text fits at the current
  font size. Otherwise the region gets a number (blue), and the numbers with their labels are listed to the right
  of the axes (`legend=False`: small regions are not labelled). A state keeps its number in all its small regions.
* A singular line that lies on a traced vertical boundary is not drawn dotted (it would hide the line); its state is
  numbered next to the line and listed as "label (line, x = …)".

The return value is the numbered list.

#### `ionic_strength(result) -> float`

Largest molal ionic strength I = ½ Σ m_i z_i² [mol/kg water] among the liquids of a result that contain ions
(explicit species such as HSO4- included); 0 if there is none.

#### `plot_validity(ax, pm, *, cache=None, limit=100.0, measure=ionic_strength, n=300, color="#c0392b", lw=1.4, points=None, smooth=0.006)`

Draws a dashed line where `measure` (default `ionic_strength`) equals `limit`, i.e. around the part of the diagram
where the liquids stay within the concentration range of the activity model. The measure is taken from the solves of
the traces and of `cache` (or `points` from `validity_points(pm, cache, measure)`: an array of (x, RH, value)),
interpolated linearly in log(1 + value) onto an n × n grid and smoothed. AIOMFAC states no numeric limit of
validity (Zuend et al., 2008, fit to electrodynamic-balance data of supersaturated solutions; beyond them the curves
are predictions), so `limit` is a judgement; 100 mol/kg is the default.

#### `singular_lines(pm) -> [(x, rh_from, rh_to, PhaseState)]`

The states that occur at a single interior x only (drawn as lines by `plot_phase_map`).

#### `pie_composition(result, group=None) -> [(label, {group: mol})]`

Composition of each liquid (`"L1"`, `"L2"`, …) and each solid of a result, grouped for pie charts (default groups:
water, every organic, all ions pooled). `group` maps a component name to a group name.

### 4.3 Liquid–liquid diagrams of neutral mixtures

#### `binary_mixing_curve(components, T_K, *, n=401) -> dict`

For two neutral components (`aiomfac_py.Component`, e.g. water and an organic): `x` (mole fraction of the second),
the normalized Gibbs energy of mixing `g` = (1−x) ln a1 + x ln a2, the equilibrium activities `a1`, `a2` (constant
across a miscibility gap) and `splits` = [(x', x'')] of the coexisting compositions.

#### `ternary_lle(components, T_K, *, h=0.005, long_edge=0.06) -> TernaryLLE`

Phase diagram of three neutral components from the lower convex hull of g(x) over the composition triangle (§6.6).
`h` is the grid spacing (about 25 000 AIOMFAC evaluations, 5–15 s, at the default); `long_edge` the hull-edge length
above which an edge joins two coexisting phases.

`TernaryLLE`: `names`, `T_K`, grid `P` (columns x2, x3), `g`, `facets`; `three_phase` (compositions of the three
coexisting liquids of each three-phase triangle); `tie_lines(max_lines=None)`; `equilibrium(x2, x3)` → number of
phases and equilibrium activities of an overall composition.

#### `plot_ternary(td, ax=None, *, show="phases", n_tie=60, levels=None)`

Right-triangle plot (x axis: component 3, y axis: component 2, component 1 at the origin). `show="phases"`: one-,
two- and three-phase regions with `n_tie` tie lines; `show=0, 1, 2`: contours of the equilibrium activity of that
component.

## 5. Recipes

### 5.1 Deliquescence RH of single salts

```python
pe = PhaseEquilibrium([], ["Na+", "Cl-"], T_K=298.15)
rh, solids = dg.deliquescence_point(pe, {"Na+": 1.0, "Cl-": 1.0})       # 0.7561, ('halite',)
```

The traced DRH of ten single salts agrees with `aiomfac_py.sle.binary_saturation` to 4e-5.

### 5.2 Acid sulfate–nitrate particles in the UHAERO coordinates

```python
X, Y = 0.6, 0.85                         # ammonium fraction, sulfate fraction
feed = {"NH4+": X, "H+": 1 - X, "SO4--": Y / (1 + Y), "NO3-": (1 - Y) / (1 + Y)}
pe = PhaseEquilibrium([], ["NH4+", "H+", "SO4--", "NO3-"], T_K=298.15)
t = dg.trace(pe, feed, rh_min=0.05, rh_max=0.95, n=24)
p = dg.rh_profile(pe, feed, np.arange(0.05, 0.951, 0.01))
pH_x = p["pH"] + 1.744                   # mole-fraction scale, as in the papers
```

For a full diagram use `phase_map` over X and `rh_profile` on a regular (X, RH) grid for contour fields
(`tools/uhaero2006_figures.py`).

### 5.3 Particles with organics

```python
from aiomfac_py.s2as import smiles_to_components
org = smiles_to_components(["CC1(C)C(CC1CC(=O)O)C(=O)O"]).components[1]      # pinic acid
pe = PhaseEquilibrium([org], ["NH4+", "SO4--"], T_K=298.15)
feed = {"NH4+": 2.0, "SO4--": 1.0, org.name: 0.75}
t = dg.trace(pe, feed, n=24)                               # deliquescence and liquid-liquid phase separation
tm = dg.trace(pe, feed, n=24, mode="metastable")          # phase separation of the supersaturated liquid
```

An excess-Gibbs-energy surrogate is used in the same way through
`PhaseEquilibrium(..., liquid_model=GibbsLiquidModel(...))`.

### 5.4 Region labels and line drawings

```python
ax = dg.plot_phase_map(pm, legend=False, colors={s: "white" for s in pm.states()})
dg.label_regions(ax, pm, {"ammonium_sulfate": "A", "letovicite": "B", "ammonium_bisulfate": "C"})
```

### 5.5 Smooth boundary lines

`phase_map` computes the boundaries at a set of compositions only; between them the plot interpolates, so a
boundary that bends or ends between two compositions is drawn as a polygon. `trace_boundaries` follows each line
from those points with a few solves per point and gives smooth lines that meet at the junctions:

```python
pm = dg.phase_map(pe, feed, np.linspace(0.005, 0.995, 11), x_label="x", n=32)
bc = dg.trace_boundaries(pe, feed, pm, verbose=True)
ax = dg.plot_phase_map(pm, curves=bc)
dg.label_regions(ax, pm, {"ammonium_sulfate": "A"}, curves=bc)
```

The phase map can be coarser when the lines are traced (they need one boundary point per line, plus the vertical
boundaries found between traces), e.g. 11 compositions without `refine_x`.

### 5.6 Saving results

`PhaseMap`, `BoundaryCurves` and `TernaryLLE` objects can be pickled; `PhaseMap.to_records()` gives a list of dicts for
`json.dump` or `pandas.DataFrame`.

## 6. How it works

### 6.1 Boundary tracing

`trace` solves the RH grid from high to low RH, each solve warm-started from the previous one (a warm start that does
not converge is repeated from scratch by `PhaseEquilibrium.solve`). Every pair of neighbouring converged solves with
different phase states is then bisected; the midpoint solve is warm-started from the side that has a liquid. A
boundary is resolved when its bracket is narrower than `tol`.

### 6.2 Failed solves

Solves that do not converge (status `"not_converged"`) are dropped and listed in `Trace.failed`; boundaries are
located between converged solves only. A bisection point that does not converge is retried off-centre (at 1/4 and
3/4 of the bracket) and from the other side; if all fail, the boundary is reported with `resolved=False` and its
bracket. In very concentrated acid or nitrate solutions at low RH a few percent of the solves can fail; they show as
gaps in `rh_profile` arrays.

### 6.3 All-solid states

When no liquid remains, the solid assemblage is the minimum of the linear program
min Σ_s n_s (ln K_s − h_s ln RH) over the candidate solids subject to the mass balance (water is open, so the
transformed Gibbs energy of an all-solid state is linear in the solid amounts). `PhaseEquilibrium` compares this
minimum with the result of its iteration and takes the lower one, so that mutual deliquescence RHs are invariant in
the mixing ratio, as the phase rule requires. The saturation indices of a dry result are the negative reduced costs
of the program.

### 6.4 Region grid

For plotting, the state is evaluated on a fine (x, RH) grid: for each x, the boundaries of the nearer computed trace
are used, each interpolated linearly towards the boundary of the same kind in the other neighbouring trace. Traces
holding a state that exists at that x only (a singular composition) are drawn as lines and left out of the grid.
Region outlines are the contour lines of the grid. With traced lines (`curves=`), each column of the grid is instead
split at the RH values where the lines cross it, and each part takes the state on that side of the line (vertical
lines separate columns and are not used).

### 6.5 Continuation of the boundary lines

`trace_boundaries` works in the unit square (x and RH scaled by their ranges). From a point on the boundary between
states L and R it predicts the next point one step h along the secant of the last two points and corrects it along
the normal: the states at ±δ across the line must be L and R (δ is doubled until they are, up to `delta_max`), and
the bracket is bisected to `tol`. δ is set from the previous correction (at least `2 tol`, at most h/2), so a
straight line costs about four solves per point; each solve is warm-started from the last liquid result on its side.
The step grows by 1.6 after a small correction (up to `h_max`) and is halved when the bracket holds a third state,
cannot be found, or the new point turns by more than 25°; below `h_min` the line ends there (a junction). Because
the correction is across the line, vertical boundaries (fixed composition) are followed as well as horizontal ones
(invariant RH).

The starting points are the boundary points of the traces (followed in both x directions). A line that runs into one
already traced between the same states stops, and starting points on a traced line are skipped. Then, at
`rh_samples` RH levels, the states of neighbouring traces are compared; a change that no traced line explains is
located by bisection in x and followed up and down. If the change of state is also found at the same x 0.03
above or below, the boundary is vertical (typically the stoichiometric composition of a salt, where a solid exists
alone on the line and thin states can lie next to it); it is then built from two RH scans just left and right of the
line (bisected to `tol`), and each RH interval where the two sides differ becomes a line. Compositions in `x_lines`
are treated the same way first.

Lines that cross no trace and are not vertical — typically short lines between two junctions — are found by
checking the states at 16 points on a circle of radius `3 h_min` around every line end at a junction: a change of
state between two neighbouring points that no traced line crosses is located on the circle by bisection and
followed outwards. This is repeated (`junction_rounds`) for the ends of the new lines.

At each of the `rh_samples` levels, the lines already traced that cross the interval between two traces must lead
from the state of the left trace to that of the right one (the states on the two sides of each crossing are chained);
where they do not, the line is missing in that sub-interval and is located there. A seed line is treated as vertical
only if all its points lie within `2 tol` of one x; a line that bends away from a vertical boundary is kept. Lines
between the same states that run along a longer one (at least 80 % of their length within `2 h_min`, traced twice
from two seeds) are dropped.

The tracing uses two rules of phase-diagram topology:

* **Law of adjoining phase regions** (Palatnik and Landau; Rhines). In a two-dimensional section, the phase states
  on the two sides of a boundary line differ by exactly one phase. The exceptions are degenerate lines: an invariant
  RH (horizontal line, where a third state exists only on the line) and a single composition (vertical line of a
  singular state). Where a line separates states that differ by two or more phases, either a thin region between
  them has been missed, or one side is a local minimum. A converged solve can still be a local minimum when the
  stability test misses a phase. So at such a bracket every solve is also started from the other side's state, and
  the result with the lower Gibbs energy is kept (`_Tracer.state_min`).
* **A boundary line cannot end inside the diagram.** It ends on the edge, or at a junction where other lines meet.
  After the junction rounds, every line end that no other line reaches is probed on circles of 3, 6 and 12 `h_min`
  (24 points), and a change of state that no line between those two states crosses starts a new line
  (`dangling_rounds`). Here and around junctions, a change between two probe points counts as explained only by a
  line between the same two states.
* **Local refinement.** Where these rules are still violated after the probes, or at points given in
  `refine_at`, the states are scanned on a local grid, and every unexplained change of state between neighbouring
  grid points is traced with half the step and tolerance. The lines found this way, and all short lines (3–6
  points, typically from the probes), are checked with fresh solves across them at a quarter, half and three
  quarters of their length. Each side is also started from the other side's result, and the lower Gibbs energy
  decides. A line is kept only if the two sides differ at most of these points. This removes lines made by a cached
  local minimum.
* **Closing and extending.** Two free ends of lines between the same states that are closer than 15 `h_min` are
  the two sides of the tip of a closed region (e.g. a lens) and are joined. Any other free end is extended along its
  tangent to the first line or edge within 12 `h_min`. Finally, line ends at a junction that are closer than `snap` to
each other are joined at the point nearest (least squares) to their tangent lines, and a single end is extended
along its tangent to the line it meets.

### 6.5a ZPF tracing (`method="zpf"`)

`trace_boundaries(..., method="zpf")` traces the boundaries as zero-phase-fraction (ZPF) lines (Morral; Hillert's
"mapping", as in CALPHAD programs) instead of following pairs of phase states. For each phase p, a signed function
f_p(x, RH) is evaluated from an ordinary equilibrium solve (`aiomfac_py.zpf.phase_functions`):

* a solid: its amount where it is present, its saturation index (≤ 0) where it is absent;
* the k-th liquid (liquids ordered by amount): its amount where there are at least k liquids; where there are
  k − 1, minus the smallest tangent-plane distance of a non-trivial trial phase of the stability test
  (`PhaseEquilibriumResult.tpd_nontrivial`), which goes to zero where the k-th liquid is about to form.

f_p changes sign exactly where p appears or disappears, so the zero lines of all f_p are the boundary lines. Each
is followed by continuation: a predictor along the tangent, and a corrector along the normal that finds the zero of
f_p by regula falsi. Because f_p is smooth on both sides, the zero is interpolated rather than bisected. When the
bracket holds two zeros (near an acute corner), the one nearer to the predicted point is taken. A ZPF line is
defined across junctions, so it continues through them: where it turns sharply (a liquidus meeting an invariant
line, say), the continuation is found on small circles around the last point. So a line ends only at the edge or
on itself (a closed line). The states on both sides are recorded at every point, and a line is split into boundary
curves where they change; that is where the junctions are. Each curve is the boundary of one phase, so the law of
adjoining phase regions holds by construction. Lines that coincide (an invariant RH, a single composition) are
kept once per pair of states. As in the state tracing, line ends at junctions are joined, tips of closed regions
are closed, and free ends are extended to the line or edge they meet.

Seeds: the phases that differ across each boundary point of the traces, and phases present on one trace and
absent on the neighbouring one at the same RH with no ZPF line of that phase between them. Where the states on
two neighbouring traces differ by two or more phases at the same RH, several boundaries lie between them, and a phase
present only between the traces (a thin strip at a nearly stoichiometric composition) has the same sign of f_p on
both traces; such a segment is halved until its ends differ by at most one phase or it is shorter than
`seed_subdivide` (default 0.004 of the axis ranges; `None` switches this off), and each piece is seeded. A solid that
occurs on no trace at all is included through its saturation index. Finally the lines are checked against the
solves (region consistency): all solves in a region enclosed by the lines must have the same phase state, so two
neighbouring converged solves (the `consistency_k` = 8 nearest, closer than `consistency_radius` = 0.01) whose states
differ while no line of a phase that differs between them crosses or passes near the segment joining them point to a
missing line. Both states are first checked against each other (re-solved from the other one, the lower Gibbs energy
kept, so a local minimum does not count), then the segment is seeded as above. Pairs within `consistency_radius` of
a failed solve are left out (the states there are not resolved, and new solves tend to run into the time limit). At
most `consistency_max` = 300 pairs per round, `consistency_rounds` = 2 rounds and `consistency_time` = 1800 s in all
(`consistency_rounds=0` switches the check off). In the UHAERO 2007 Y = 0.85 diagrams this
reduces the boundary pieces missing against the state tracing from 1, 3, 0, 4 (seeds between traces only) to
1, 2, 0, 3, for a few hundred extra solves per diagram. Pairs within 5 `h_min` of a junction (where the states beside
a line change) are left out as well: several regions meet there within a few `h_min`. The cache is shared with the state tracing.

### 6.6 Convex hull for liquid–liquid equilibrium

For a mixture of neutral components at fixed T and pressure, the equilibrium state of an overall composition z is
given by the lower convex hull of the molar Gibbs energy of mixing g(x): the hull point above z is a combination of
hull vertices, which are the coexisting phases, and the hull plane gives the equilibrium chemical potentials
(μ_i/RT = ln a_i at the vertex of pure i). `ternary_lle` evaluates g on a triangular grid (refined towards the edges
and corners, down to mole fractions of 1e-12) and computes the hull with Qhull. Hull triangles are classified by
their edge lengths. This is a global method: it cannot converge to a local minimum or miss a phase, but the phase
compositions are resolved only to the grid spacing.

## 7. Performance

On one CPU thread (cloud VM, Xeon 2.1 GHz), one `PhaseEquilibrium.solve` takes about 0.05–0.15 s for single and
mixed salts, 0.3–0.6 s for acid sulfate–nitrate systems and 1–3 s with two organics. A trace needs 40–90 solves; a
phase map of 11–21 compositions with refinement 500–3000 solves (a few minutes for salts, 30–60 min for acid
sulfate–nitrate systems with contour fields). Independent traces can be run in parallel processes.
`ternary_lle` takes 5–15 s.

`trace_boundaries` needs about four solves per point on a straight line and 10–20 per line end. Examples (same
machine, one process per diagram): (NH4)2SO4–NH4NO3, 14 lines, 1400 solves, 2–3.5 min (the phase map: 2.5 min);
the UHAERO 2006 diagrams (NH4+/H+/SO4--/NO3-), 14–44 lines, 2600–6000 solves, 35–100 min each (comparable to their
phase maps); the UHAERO 2007 diagrams with two organics, 3600–4800 solves, about 1.5 h each. Most of the extra cost
is in compositions where the solver is slow or does not converge (very acidic, nearly dry particles); there
`solve_timeout` keeps it bounded, and `show_failed=True` shows where the diagram is not resolved. With `cache`, a
second call with other options only solves the new points (a minute instead of hours).

## 8. Limitations

* **Activity model.** Results inherit AIOMFAC's accuracy. Two cases seen in the UHAERO comparisons:
  AIOMFAC underestimates the salting-out of (NH4)2SO4 by NH4NO3 (the computed invariant solutions of the
  (NH4)2SO4–NH4NO3–H2O system hold about 2 mol/kg more (NH4)2SO4 than measured), and in very acidic, nearly water-free
  NH4+/H+/SO4-- mixtures (X < 0.35, RH < 25 %) AIOMFAC predicts a second, almost molten liquid (about 190 mol of ions
  per kg of water) that is more stable than solid NH4HSO4. Such states lie far outside the range AIOMFAC was fitted
  to.
* **Solid data.** Diagrams expose the weaknesses of the solubility products. The quality flag of each solid is in
  `aiomfac_py.solids` (`Solid.quality`); double salts are composite (ln K = sum of the simple salts' ln K + the
  solid-state formation reaction of Clegg et al., 1998).
* **Fixed RH only.** Water is open; diagrams at fixed water content (the (f_H+, f_NH4+) coordinates of Amundson et
  al., 2006, Fig. 14) are not supported. Efflorescence is not traced (use `PhaseEquilibrium.drying_path` with
  critical supersaturations).
* **Range of the activity model.** At low RH the liquids become very concentrated (ionic strength of hundreds to
  thousands of mol/kg in acidic, nearly water-free liquids, or organic-rich liquids with dissolved salt and little
  water). AIOMFAC was not fitted there, and the phase states there (e.g. four liquids) are extrapolations;
  `plot_validity` marks this part of a diagram.
* **Grid resolution.** States narrower than the RH grid spacing or the x refinement tolerance can be missed;
  ternary phase compositions are resolved to `h`. `trace_boundaries` finds the lines that cross a trace, that
  separate neighbouring traces at one of the `rh_samples` levels, or that leave a junction; a short line that does
  none of these is missed, and no line can be traced where the solver does not converge. Where a line is missing,
  the regions keep the interpolated states.

## 9. References

* Amundson, N. R., Caboussat, A., He, J. W., Martynenko, A. V., Savarin, V. B., Seinfeld, J. H., and Yoo, K. Y.:
  A new inorganic atmospheric aerosol phase equilibrium model (UHAERO), Atmos. Chem. Phys., 6, 975–992, 2006.
* Amundson, N. R., Caboussat, A., He, J. W., Martynenko, A. V., Landry, C., Tong, C., and Seinfeld, J. H.: A new
  atmospheric aerosol phase equilibrium model (UHAERO): organic systems, Atmos. Chem. Phys., 7, 4675–4698, 2007.
* Clegg, S. L., Brimblecombe, P., and Wexler, A. S.: Thermodynamic model of the system H+–NH4+–SO42−–NO3−–H2O at
  tropospheric temperatures, J. Phys. Chem. A, 102, 2137–2154, 1998.
* Potukuchi, S. and Wexler, A. S.: Identifying solid-aqueous phase transitions in atmospheric aerosols — I.
  Neutral-acidity solutions, Atmos. Environ., 29, 1663–1676, 1995.
* Zuend, A., Marcolli, C., Luo, B. P., and Peter, T.: A thermodynamic model of mixed organic-inorganic aerosols to
  predict activity coefficients, Atmos. Chem. Phys., 8, 4559–4593, 2008.

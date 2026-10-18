"""aiomfac_py.pipeline: phase diagram from a configuration file (map, lines, topology check, plot)."""
import json
import os

import pytest

pytest.importorskip("scipy")
pytest.importorskip("matplotlib")


def test_pipeline_nacl_kcl_runs_and_resumes(tmp_path, capsys):
    from aiomfac_py.pipeline import EXAMPLE, load_config, main
    cfg = dict(EXAMPLE, output="out")
    cfg["axis"] = {"label": "x(NaCl)", "x": [0.1, 0.5, 0.9]}
    cfg["map"] = {"n": 12, "rh_min": 0.6, "rh_max": 0.95, "refine_x": False}
    path = tmp_path / "nacl.json"
    path.write_text(json.dumps(cfg))
    assert main(["run", str(path), "--quiet"]) == 0
    out = tmp_path / "out"
    for f in ("map.pkl", "lines.pkl", "lines_cache.pkl", "report.json", "diagram.png"):
        assert (out / f).exists(), f
    rep = json.loads((out / "report.json").read_text())
    assert rep["lines"]["clean"] and rep["lines"]["lines"] == 5 and rep["lines"]["failed"] == 0
    t = os.path.getmtime(out / "lines.pkl")
    assert main(["run", str(path)]) == 0                 # up to date: map and lines are not recomputed
    assert "map: up to date" in capsys.readouterr().out
    assert os.path.getmtime(out / "lines.pkl") == t
    assert load_config(str(path))["_dir"] == str(tmp_path)


def test_pipeline_feed_interpolation_and_subgroups():
    from aiomfac_py.pipeline import build_system
    cfg = {"system": {"ions": ["NH4+", "SO4--"], "T_K": 298.15,
                      "organics": [{"name": "glycerol", "subgroups": [[2, 2], [3, 1], [14, 3]]}],
                      "options": {"inner_method": "rand", "fallback_inner": ["newton"]}},
           "feed": {"x0": {"NH4+": 2.0, "SO4--": 1.0}, "x1": {"glycerol": 1.0}}}
    pe, feed_of_x, kw = build_system(cfg)
    assert pe.inner_method == "rand" and pe.fallback_inner == ("newton",)
    assert feed_of_x(0.25) == {"NH4+": 1.5, "SO4--": 0.75, "glycerol": 0.25}
    assert feed_of_x(0.0) == {"NH4+": 2.0, "SO4--": 1.0} and kw == {}

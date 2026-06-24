"""Progress-visualization tests. The CSV logging is pure stdlib (fully unit-tested here); the plot
needs matplotlib (importorskip smoke that it writes a PNG). This is what turns the per-iteration eval
numbers into a shareable learning curve.
"""
import json
import os

import pytest

from training.progress_log import (progress_row, append_progress, read_progress,
                                    append_milestone, read_milestones)


def test_append_read_milestones_roundtrip(tmp_path):
    """Named milestones (latent-BC, direct+recovery, ...) -> a CSV for the 'accuracy over the project'
    chart. Header written once; names + metrics round-trip."""
    p = str(tmp_path / "milestones.csv")
    append_milestone(p, "latent-BC", {"route_completion": 0.02, "crash_rate": 0.0, "off_road_rate": 0.0})
    append_milestone(p, "direct+recovery", {"route_completion": 0.39, "crash_rate": 0.0, "off_road_rate": 0.30})
    rows = read_milestones(p)
    assert len(rows) == 2
    assert rows[0]["milestone"] == "latent-BC"
    assert rows[1]["route_completion"] == pytest.approx(0.39)
    assert rows[1]["off_road_rate"] == pytest.approx(0.30)
    with open(p) as f:
        assert sum(1 for ln in f if ln.startswith("milestone")) == 1   # header once


def _summary(route=0.23, success=0.0, crash=0.4, off=0.6, ret=12.3, n=5):
    return {"route_completion": route, "success_rate": success, "crash_rate": crash,
            "off_road_rate": off, "mean_return": ret, "n": n}


def test_progress_row_extracts_fixed_keys():
    row = progress_row(2, _summary(route=0.23))
    assert row["iter"] == 2
    assert row["route_completion"] == pytest.approx(0.23)
    assert set(row) == {"iter", "route_completion", "success_rate", "crash_rate",
                        "off_road_rate", "mean_return"}


def test_append_writes_header_once_and_read_roundtrips(tmp_path):
    p = str(tmp_path / "progress.csv")
    append_progress(p, progress_row(0, _summary(route=0.03)))
    append_progress(p, progress_row(1, _summary(route=0.23, crash=0.8)))
    rows = read_progress(p)
    assert len(rows) == 2
    assert rows[0]["iter"] == 0 and rows[1]["iter"] == 1
    assert rows[1]["route_completion"] == pytest.approx(0.23)
    assert rows[1]["crash_rate"] == pytest.approx(0.8)
    # header appears exactly once (a second append must not re-write it)
    with open(p) as f:
        header_lines = [ln for ln in f.read().splitlines() if ln.startswith("iter")]
    assert len(header_lines) == 1


def test_plot_progress_writes_png(tmp_path):
    pytest.importorskip("matplotlib")
    from scripts.plot_progress import main as plot_main
    csv_path = str(tmp_path / "progress.csv")
    for it, r in enumerate([0.03, 0.10, 0.23]):
        append_progress(csv_path, progress_row(it, _summary(route=r)))
    bl_path = str(tmp_path / "baselines.json")
    with open(bl_path, "w") as f:
        json.dump({"random_route": 0.02, "idm_route": 0.99}, f)
    out = str(tmp_path / "progress.png")
    plot_main(progress_csv=csv_path, baselines=bl_path, out=out)
    assert os.path.exists(out) and os.path.getsize(out) > 0


def test_plot_milestones_writes_png(tmp_path):
    pytest.importorskip("matplotlib")
    from scripts.plot_milestones import main as plot_main
    mc = str(tmp_path / "milestones.csv")
    for name, route, off in [("RANDOM", 0.02, 0.0), ("direct-BC", 0.24, 0.5),
                             ("direct+recovery", 0.39, 0.30), ("IDM", 0.99, 0.0)]:
        append_milestone(mc, name, {"route_completion": route, "crash_rate": 0.0, "off_road_rate": off})
    out = str(tmp_path / "milestones.png")
    plot_main(milestones=mc, out=out)
    assert os.path.exists(out) and os.path.getsize(out) > 0

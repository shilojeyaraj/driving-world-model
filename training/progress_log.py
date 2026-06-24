"""Dependency-free progress logging: append per-iteration eval metrics to a CSV (stdlib `csv`) so a
learning curve can be plotted later (scripts/plot_progress.py). Generic over any eval summary with
the standard driving keys (eval.closed_loop.summarize_driving), so DAgger -- and run_metadrive later
-- can both feed it. No third-party dependency, so it's always safe to call during training.
"""
import csv
import os

COLUMNS = ["iter", "route_completion", "success_rate", "crash_rate", "off_road_rate", "mean_return"]


def progress_row(it, summary):
    """Assemble one CSV row (the fixed COLUMNS) from an eval summary. PURE."""
    return {"iter": int(it),
            "route_completion": float(summary.get("route_completion", 0.0)),
            "success_rate": float(summary.get("success_rate", 0.0)),
            "crash_rate": float(summary.get("crash_rate", 0.0)),
            "off_road_rate": float(summary.get("off_road_rate", 0.0)),
            "mean_return": float(summary.get("mean_return", 0.0))}


def append_progress(path, row):
    """Append `row` to the CSV at `path`, writing the header exactly once (when the file is new or
    empty). Creates the parent directory. Only the fixed COLUMNS are written, in order."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fresh = (not os.path.exists(path)) or os.path.getsize(path) == 0
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if fresh:
            w.writeheader()
        w.writerow({k: row[k] for k in COLUMNS})


def read_progress(path):
    """Read the CSV back to a list of dicts (iter -> int, the rest -> float). PURE round-trip."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return [{"iter": int(float(r["iter"])),
             **{k: float(r[k]) for k in COLUMNS if k != "iter"}} for r in rows]


# --- named MILESTONES (for the "accuracy over the project" chart: latent-BC, direct, +recovery, ...) ---
MILESTONE_COLUMNS = ["milestone", "route_completion", "crash_rate", "off_road_rate"]


def append_milestone(path, name, summary):
    """Append a named milestone (a label + its held-out driving metrics from summarize_driving) to a
    CSV, header written once. Lets the learning curve span the WHOLE project, not just one run."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fresh = (not os.path.exists(path)) or os.path.getsize(path) == 0
    row = {"milestone": str(name),
           "route_completion": float(summary.get("route_completion", 0.0)),
           "crash_rate": float(summary.get("crash_rate", 0.0)),
           "off_road_rate": float(summary.get("off_road_rate", 0.0))}
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MILESTONE_COLUMNS)
        if fresh:
            w.writeheader()
        w.writerow(row)


def read_milestones(path):
    """Read the milestone CSV back (name str, metrics float). PURE round-trip."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return [{"milestone": r["milestone"],
             **{k: float(r[k]) for k in MILESTONE_COLUMNS if k != "milestone"}} for r in rows]

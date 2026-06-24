"""Plot driving accuracy across the project's MILESTONES (route completion % per approach), so the
whole arc is one image: random -> latent-BC (idled) -> direct-BC -> +recovery -> +scale -> IDM.
Reads runs/milestones.csv (append_milestone). Complements scripts/plot_progress.py (per-iteration
curve); this is the cross-approach bar chart. Needs matplotlib (pip install matplotlib).

Usage:  python -m scripts.plot_milestones
        python -m scripts.plot_milestones runs/milestones.csv out.png
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.progress_log import read_milestones


def _color(name):
    n = name.lower()
    if "random" in n:
        return "gray"
    if "idm" in n:
        return "green"
    return "#1f77b4"


def main(milestones="runs/milestones.csv", out="runs/milestones.png"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        raise SystemExit("matplotlib not installed -- run:  pip install matplotlib")
    if not os.path.exists(milestones):
        raise SystemExit(f"no milestones at {milestones} -- record some with append_milestone first.")

    rows = read_milestones(milestones)
    names = [r["milestone"] for r in rows]
    route = [r["route_completion"] * 100 for r in rows]

    fig, ax = plt.subplots(figsize=(max(7, 1.3 * len(rows)), 5))
    bars = ax.bar(range(len(rows)), route, color=[_color(n) for n in names])
    for i, (b, r) in enumerate(zip(bars, rows)):
        label = f"{route[i]:.0f}%"
        if r["crash_rate"] or r["off_road_rate"]:
            label += f"\ncrash {r['crash_rate']:.0%}\noff {r['off_road_rate']:.0%}"
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1, label, ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("route completion (%)")
    ax.set_title("Driving accuracy across milestones (held-out maps)")
    ax.set_ylim(0, 105)
    ax.grid(True, axis="y", alpha=0.3)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    mc = a[0] if len(a) > 0 else "runs/milestones.csv"
    ou = a[1] if len(a) > 1 else "runs/milestones.png"
    main(milestones=mc, out=ou)

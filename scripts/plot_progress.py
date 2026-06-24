"""Plot the learning curve: route completion % vs DAgger iteration, with dashed Random (floor) and
IDM-expert (ceiling) reference lines. Reads runs/dagger/progress.csv (written each iteration during
training) + runs/dagger/baselines.json. The one-image "is the car learning?" chart to show people.

Usage:  python -m scripts.plot_progress
        python -m scripts.plot_progress runs/dagger/progress.csv runs/dagger/progress.png
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.progress_log import read_progress


def main(progress_csv="runs/dagger/progress.csv", baselines="runs/dagger/baselines.json",
         out="runs/dagger/progress.png"):
    try:
        import matplotlib
        matplotlib.use("Agg")                       # headless: write a file, no display needed
        import matplotlib.pyplot as plt
    except ImportError:
        raise SystemExit("matplotlib not installed -- run:  pip install matplotlib")
    if not os.path.exists(progress_csv):
        raise SystemExit(f"no progress log at {progress_csv} -- run training (python -m scripts.dagger) first.")

    rows = read_progress(progress_csv)
    iters = [r["iter"] for r in rows]
    route = [r["route_completion"] * 100 for r in rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(iters, route, "-o", color="#1f77b4", linewidth=2, label="DAgger policy")
    if os.path.exists(baselines):
        with open(baselines) as f:
            bl = json.load(f)
        if bl.get("random_route") is not None:
            ax.axhline(bl["random_route"] * 100, ls="--", color="gray",
                       label=f"Random ({bl['random_route'] * 100:.0f}%)")
        if bl.get("idm_route") is not None:
            ax.axhline(bl["idm_route"] * 100, ls="--", color="green",
                       label=f"IDM expert ({bl['idm_route'] * 100:.0f}%)")
    ax.set_xlabel("DAgger iteration")
    ax.set_ylabel("route completion (%)")
    ax.set_title("Driving skill over training (held-out maps)")
    ax.set_ylim(0, 100)
    if iters:
        ax.set_xticks(iters)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    pc = a[0] if len(a) > 0 else "runs/dagger/progress.csv"
    ou = a[1] if len(a) > 1 else "runs/dagger/progress.png"
    main(progress_csv=pc, out=ou)

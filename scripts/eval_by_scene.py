"""Per-scene failure breakdown (roadmap follow-up): evaluate a direct policy on each road GEOMETRY
separately -- straight / curve / intersection / roundabout -- to see WHERE it goes off-road, not just
that it does. Reveals whether to target recovery data at, say, curves vs intersections.

Saves a grouped bar chart (route + off-road per scene) to runs/direct_bc/by_scene.png.

Usage:  python -m scripts.eval_by_scene runs/direct_bc/policy.pt
        python -m scripts.eval_by_scene runs/direct_bc/policy.pt 10 SCXO   # episodes, scene letters
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from scripts.dagger import build_cfg
from scripts.ablate_direct_bc import _DirectAct
from scripts.eval_driving import _run_episodes, _fmt
from eval.closed_loop import summarize_driving
from envs.metadrive_env import train_eval_seed_split
from training.direct_bc import load_direct

SCENE_NAMES = {"S": "straight", "C": "curve", "X": "intersection", "O": "roundabout",
               "T": "T-junction", "r": "ramp", "R": "ramp"}


def eval_scene(policy, scene, episodes=10, num_scenarios=50):
    """Run the policy on held-out maps of ONE block type (scene) and summarize. Live."""
    cfg = build_cfg(num_scenarios=num_scenarios, road_map=scene)
    _, (eval_start, eval_num) = train_eval_seed_split(int(cfg.metadrive_num_scenarios),
                                                      int(cfg.metadrive_eval_scenarios))
    cfg.metadrive_start_seed, cfg.metadrive_num_scenarios = eval_start, eval_num
    cfg.metadrive_render, cfg.metadrive_endless = False, False
    return summarize_driving(_run_episodes(cfg, episodes, _DirectAct(policy)))


def plot_scene_breakdown(results, out):
    """Grouped bar chart: route % and off-road % per scene. `results` = [(scene, summary), ...]."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = [f"{s}\n{SCENE_NAMES.get(s, s)}" for s, _ in results]
    route = [r["route_completion"] * 100 for _, r in results]
    offroad = [r["off_road_rate"] * 100 for _, r in results]
    x = np.arange(len(results)); w = 0.38
    fig, ax = plt.subplots(figsize=(max(7, 1.6 * len(results)), 5))
    ax.bar(x - w / 2, route, w, label="route completion %", color="#1f77b4")
    ax.bar(x + w / 2, offroad, w, label="off-road rate %", color="#d62728")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("%"); ax.set_ylim(0, 105)
    ax.set_title("Direct policy by road geometry (held-out maps)")
    ax.legend(); ax.grid(True, axis="y", alpha=0.3)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)


def main(ckpt="runs/direct_bc/policy.pt", episodes=10, scenes="SCXO", out="runs/direct_bc/by_scene.png"):
    if not os.path.exists(ckpt):
        raise SystemExit(f"no policy at {ckpt} -- train one first (scripts.train_direct_policy)")
    policy = load_direct(ckpt)
    print(f"per-scene breakdown: {ckpt}  ({episodes} eps/scene)", flush=True)
    results = []
    for scene in scenes:
        s = eval_scene(policy, scene, episodes)
        results.append((scene, s))
        print(f"  {scene} {SCENE_NAMES.get(scene, scene):12s}: {_fmt(s)}", flush=True)
    try:
        plot_scene_breakdown(results, out)
        print(f"saved {out}", flush=True)
    except ImportError:
        print("(install matplotlib for the chart: pip install matplotlib)", flush=True)
    worst = max(results, key=lambda sr: sr[1]["off_road_rate"])
    print(f"worst geometry: {worst[0]} ({SCENE_NAMES.get(worst[0], worst[0])}) "
          f"off-road {worst[1]['off_road_rate']:.0%} -> target recovery data there", flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    ck = a[0] if len(a) > 0 and a[0] not in ("-", "none") else "runs/direct_bc/policy.pt"
    ep = int(a[1]) if len(a) > 1 and a[1].isdigit() else 10
    sc = a[2] if len(a) > 2 else "SCXO"
    main(ckpt=ck, episodes=ep, scenes=sc)

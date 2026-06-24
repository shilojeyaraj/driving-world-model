"""Real MetaDrive run (state mode) with the ITERATED DREAMER LOOP: seed random, then repeat
collect-with-policy -> train world model -> train policy in imagination. This grounds the world
model in policy-visited states (the fix for the single-shot model-exploitation failure in
experiments/010). Headless on CPU, but slow (MetaDrive stepping) -- run it ALONE.
See docs/METADRIVE.md.

Trains across a POOL of procedurally-generated maps (domain randomization, default 100) so the
policy learns to drive rather than memorize one road; eval_driving then grades it on a disjoint
held-out map pool. All training knobs are CLI flags (see --help) -- no `python -c` needed.

Usage:  python -m scripts.run_metadrive                                  # defaults: 100 maps, 4 iters
        python -m scripts.run_metadrive --num-scenarios 500             # bigger map pool
        python -m scripts.run_metadrive --iters 12 --wm-steps 1500 --behavior-steps 1000 --collect 2000 --seed-steps 3000
        python -m scripts.run_metadrive --entropy 0.03                  # more exploration (fight collapse)
        python -m scripts.run_metadrive --map SSSS                      # fix the scene (highway), traffic varies
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from config import get_config
from envs.base import make_env
from training.dreamer_loop import dreamer_train
from eval.closed_loop import closed_loop_eval
from utils import save_checkpoint

DEFAULT_ENTROPY = 1e-2     # actor entropy bonus. Raised from the old 1e-3 (which let the policy
                           # collapse to saturated full-left/full-throttle): more exploration pressure
                           # keeps it trying varied actions instead of locking into one bad habit.


def build_cfg(num_scenarios=100, road_map=None, traffic_density=0.1, entropy_coef=DEFAULT_ENTROPY):
    """Build the training cfg. Extracted from main() so the map-pool + entropy wiring is unit-testable
    WITHOUT running the sim. road_map: None -> 3 random blocks (varied geometry per seed); a digit
    string -> that many random blocks; a letter string (e.g. "SSSS") -> a fixed scene.
    Trains on seeds [0, num_scenarios); eval_driving derives the DISJOINT held-out range from this."""
    if isinstance(road_map, str) and road_map.isdigit():
        road_map = int(road_map)
    if road_map is None:
        road_map = 3
    return get_config(env="metadrive", obs_type="state", state_dim=259, action_dim=2,
                      deter_dim=128, stoch_dim=32, hidden_dim=128, seq_len=10, imagine_horizon=15,
                      gamma=0.99, lambda_=0.95, entropy_coef=entropy_coef, actor_lr=3e-4, critic_lr=3e-4,
                      lr=3e-4, batch_size=16, max_episode_steps=200,
                      metadrive_map=road_map, metadrive_traffic_density=traffic_density,
                      metadrive_num_scenarios=int(num_scenarios), metadrive_start_seed=0)


def parse_args(argv):
    """CLI for the training knobs, so they're tunable without editing code or `python -c`."""
    p = argparse.ArgumentParser(prog="run_metadrive", description="Dreamer loop on MetaDrive (state).")
    p.add_argument("--map", dest="road_map", default=None,
                   help='scene: int N (N random blocks) or block letters e.g. "SSSS"; default 3 blocks')
    p.add_argument("--num-scenarios", dest="num_scenarios", type=int, default=100,
                   help="size of the TRAIN map pool (domain randomization)")
    p.add_argument("--iters", type=int, default=4, help="collect->train-WM->train-actor iterations")
    p.add_argument("--seed-steps", dest="seed_steps", type=int, default=1500, help="initial random steps")
    p.add_argument("--collect", dest="collect_per_iter", type=int, default=1000, help="policy steps per iter")
    p.add_argument("--wm-steps", dest="wm_steps", type=int, default=400, help="world-model grad steps/iter")
    p.add_argument("--behavior-steps", dest="behavior_steps", type=int, default=400,
                   help="actor-critic grad steps/iter (in imagination)")
    p.add_argument("--entropy", dest="entropy_coef", type=float, default=DEFAULT_ENTROPY,
                   help="actor entropy bonus; raise to fight action-collapse")
    p.add_argument("--out", default="runs/metadrive/ckpt.pt", help="checkpoint path")
    return p.parse_args(argv)


def main(iters=4, seed_steps=1500, collect_per_iter=1000, wm_steps=400, behavior_steps=400,
         out="runs/metadrive/ckpt.pt", road_map=None, traffic_density=0.1, num_scenarios=100,
         entropy_coef=DEFAULT_ENTROPY):
    cfg = build_cfg(num_scenarios, road_map, traffic_density, entropy_coef)
    print(f"training on {num_scenarios} maps (seeds 0-{int(num_scenarios) - 1}), "
          f"map={cfg.metadrive_map} blocks, entropy={entropy_coef}, iters={iters} "
          f"(wm {wm_steps}/iter, actor {behavior_steps}/iter)", flush=True)
    torch.manual_seed(0); np.random.seed(0)
    env = make_env(cfg)                                  # one sim instance, reused throughout

    wm, actor, critic, buf = dreamer_train(
        cfg, env=env, iters=iters, seed_steps=seed_steps, collect_per_iter=collect_per_iter,
        wm_steps=wm_steps, behavior_steps=behavior_steps, explore_std=0.3)

    out_m = closed_loop_eval(actor, wm, env, episodes=5, max_steps=cfg.max_episode_steps)
    save_checkpoint(out, wm, actor, critic, cfg)
    print("MetaDrive (Dreamer loop) closed-loop:", {k: round(v, 3) for k, v in out_m.items()}, flush=True)
    print(f"saved {out}", flush=True)
    env.close()


if __name__ == "__main__":
    a = parse_args(sys.argv[1:])
    main(iters=a.iters, seed_steps=a.seed_steps, collect_per_iter=a.collect_per_iter,
         wm_steps=a.wm_steps, behavior_steps=a.behavior_steps, out=a.out, road_map=a.road_map,
         num_scenarios=a.num_scenarios, entropy_coef=a.entropy_coef)

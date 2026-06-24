"""Real MetaDrive run (state mode) with the ITERATED DREAMER LOOP: seed random, then repeat
collect-with-policy -> train world model -> train policy in imagination. This grounds the world
model in policy-visited states (the fix for the single-shot model-exploitation failure in
experiments/010). Headless on CPU, but slow (MetaDrive stepping) -- run it ALONE.
See docs/METADRIVE.md.

Trains across a POOL of procedurally-generated maps (domain randomization, default 100) so the
policy learns to drive rather than memorize one road; eval_driving then grades it on a disjoint
held-out map pool. Pass `-` to keep the default random-block map, or a map arg to fix the scene.

Usage:  python -m scripts.run_metadrive            # 100 random maps (seeds 0-99), 3-block geometry
        python -m scripts.run_metadrive - 500      # bigger pool: 500 maps
        python -m scripts.run_metadrive SSSS 100   # fix the scene (highway), 100 seeds (traffic varies)
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from config import get_config
from envs.base import make_env
from training.dreamer_loop import dreamer_train
from eval.closed_loop import closed_loop_eval
from utils import save_checkpoint


def main(iters=4, seed_steps=1500, collect_per_iter=1000, wm_steps=400, behavior_steps=400,
         out="runs/metadrive/ckpt.pt", road_map=None, traffic_density=0.1, num_scenarios=100):
    if isinstance(road_map, str) and road_map.isdigit():
        road_map = int(road_map)
    if road_map is None:
        road_map = 3                # 3 random blocks -> different road geometry per map seed
    # TRAIN on a pool of `num_scenarios` maps starting at seed 0 (domain randomization). eval_driving
    # derives the DISJOINT held-out eval range from this via train_eval_seed_split, so it grades the
    # policy on maps it never trained on.
    cfg = get_config(env="metadrive", obs_type="state", state_dim=259, action_dim=2,
                     deter_dim=128, stoch_dim=32, hidden_dim=128, seq_len=10, imagine_horizon=15,
                     gamma=0.99, lambda_=0.95, entropy_coef=1e-3, actor_lr=3e-4, critic_lr=3e-4,
                     lr=3e-4, batch_size=16, max_episode_steps=200,
                     metadrive_map=road_map, metadrive_traffic_density=traffic_density,
                     metadrive_num_scenarios=int(num_scenarios), metadrive_start_seed=0)
    print(f"training on {num_scenarios} maps (seeds 0-{int(num_scenarios) - 1}), "
          f"map={road_map} random blocks", flush=True)
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
    rm = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] not in ("-", "none") else None
    ns = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 100
    main(road_map=rm, num_scenarios=ns)

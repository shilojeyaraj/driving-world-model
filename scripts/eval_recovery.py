"""Targeted RECOVERY metric (roadmap B): the off-road failure isolated. Each episode starts by
FORCING the car off-center (a fixed steering push for the first K steps), then hands control to the
policy and measures whether it gets back without ending off-road. This probes the exact skill
recovery data is meant to add -- far more informative than aggregate off-road %, which is dominated
by wherever the policy happens to wander.

Run it on the recovery policy vs the clean one to see the difference:
    python -m scripts.eval_recovery runs/direct_bc/policy_recovery.pt
    python -m scripts.eval_recovery runs/direct_bc/policy.pt

Usage:  python -m scripts.eval_recovery [policy.pt] [episodes] [perturb_steps]
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from scripts.dagger import build_cfg
from scripts.eval_driving import _metadrive
from eval.closed_loop import eval_seeds, summarize_recovery
from envs.metadrive_env import adapt_obs, train_eval_seed_split
from training.direct_bc import load_direct


def eval_recovery(cfg, policy, episodes=10, perturb_steps=10, perturb_steer=0.5, perturb_throttle=0.3):
    """Run perturbed-start episodes on FIXED held-out seeds. First `perturb_steps` push the car
    off-center; then the policy drives. recovered = the episode didn't end out-of-road."""
    env = _metadrive(cfg)
    max_steps = cfg.max_episode_steps
    seeds = eval_seeds(int(cfg.metadrive_start_seed), int(cfg.metadrive_num_scenarios), episodes)
    policy.eval()
    recs = []
    try:
        for i in range(episodes):
            obs = adapt_obs(env.reset(seed=seeds[i])[0], "state")
            info = {}
            for t in range(max_steps):
                if t < perturb_steps:
                    a = np.array([perturb_steer, perturb_throttle], np.float32)   # forced drift off-center
                else:
                    with torch.no_grad():
                        a = policy(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy()
                raw, r, terminated, truncated, info = env.step(a)
                obs = adapt_obs(raw, "state")
                if terminated or truncated:
                    break
            recs.append({"recovered": not bool(info.get("out_of_road")),
                         "route_completion": float(info.get("route_completion", 0.0))})
    finally:
        env.close()
    return recs


def main(ckpt="runs/direct_bc/policy_recovery.pt", episodes=10, perturb_steps=10, num_scenarios=50):
    if not os.path.exists(ckpt):
        raise SystemExit(f"no policy at {ckpt} -- train one first (scripts.recovery_bc / scripts.watch_direct_bc)")
    cfg = build_cfg(num_scenarios=num_scenarios)
    policy = load_direct(ckpt)
    _, (eval_start, eval_num) = train_eval_seed_split(int(cfg.metadrive_num_scenarios),
                                                      int(cfg.metadrive_eval_scenarios))
    cfg.metadrive_start_seed, cfg.metadrive_num_scenarios = eval_start, eval_num
    cfg.metadrive_render, cfg.metadrive_endless = False, False
    print(f"recovery probe: {ckpt}  (held-out seeds {eval_start}-{eval_start + eval_num - 1}, "
          f"forced off-center for {perturb_steps} steps)", flush=True)
    s = summarize_recovery(eval_recovery(cfg, policy, episodes, perturb_steps=perturb_steps))
    print(f"  recovery_rate {s['recovery_rate']:.0%}  mean_route {s['mean_route']:.0%}  (n={s['n']})", flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    ck = a[0] if len(a) > 0 and a[0] not in ("-", "none") else "runs/direct_bc/policy_recovery.pt"
    ep = int(a[1]) if len(a) > 1 and a[1].isdigit() else 10
    ps = int(a[2]) if len(a) > 2 and a[2].isdigit() else 10
    main(ckpt=ck, episodes=ep, perturb_steps=ps)

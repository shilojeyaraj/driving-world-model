"""Watch the DIRECT obs->action BC policy drive in MetaDrive's 3-D window (no world model in the
loop). This is the policy that scored route ~22% in the ablation -- the best learned driver so far.

First run trains it (collect IDM -> direct BC) and saves to runs/direct_bc/policy.pt; later runs load
that instantly. Needs a display (your laptop). Drives with REAL terminations, so you'll see it commit
to driving and then drift/crash (off-road ~90%) -- the recovery-data problem we'd fix next.

Usage:  python -m scripts.watch_direct_bc                 # train (first time) or load, then watch
        python -m scripts.watch_direct_bc - SSSS          # watch on a highway scene
        python -m scripts.watch_direct_bc runs/direct_bc/policy.pt   # explicit checkpoint
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch


def main(ckpt="runs/direct_bc/policy.pt", road_map=None, steps=2000, num_scenarios=50,
         collect_steps=4000, direct_steps=4000, traffic_density=0.1):
    from scripts.dagger import build_cfg
    from training.direct_bc import DirectPolicy, train_direct_bc, save_direct, load_direct
    if isinstance(road_map, str) and road_map.isdigit():
        road_map = int(road_map)
    cfg = build_cfg(num_scenarios=num_scenarios, road_map=road_map, traffic_density=traffic_density)

    if os.path.exists(ckpt):
        policy = load_direct(ckpt)
        print(f"loaded direct-BC policy <- {ckpt}", flush=True)
    else:
        from training.train_reference import collect_idm
        print(f"no policy at {ckpt} -- training direct-BC ({collect_steps} IDM steps, then "
              f"{direct_steps} BC steps); this is a one-time cost...", flush=True)
        buf = collect_idm(cfg, collect_steps); buf._flush()
        obs = np.concatenate([ep["obs"] for ep in buf._episodes]).astype(np.float32)
        act = np.concatenate([ep["action"] for ep in buf._episodes]).astype(np.float32)
        policy = DirectPolicy(cfg.state_dim, cfg.action_dim)
        loss = train_direct_bc(policy, obs, act, direct_steps, device=cfg.device)
        save_direct(ckpt, policy, cfg.state_dim, cfg.action_dim)
        print(f"trained direct-BC (loss {loss:.4f}) -> saved {ckpt}", flush=True)

    # --- render the 3-D window and let the direct policy drive ---
    cfg.metadrive_render = True
    from metadrive.envs import MetaDriveEnv
    from envs.metadrive_env import adapt_obs, metadrive_config, disable_shadows
    env = MetaDriveEnv(metadrive_config(cfg))
    obs = adapt_obs(env.reset()[0], "state")
    if getattr(cfg, "metadrive_low_graphics", True):
        disable_shadows(env)                          # GPU win on a weak card; no-op if unavailable
    policy.eval()
    print("watching direct-BC drive -- close the window or Ctrl+C to stop.", flush=True)
    try:
        for _ in range(steps):
            with torch.no_grad():
                a = policy(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy()
            raw, r, terminated, truncated, info = env.step(a)
            env.render()
            obs = adapt_obs(env.reset()[0], "state") if (terminated or truncated) else adapt_obs(raw, "state")
    except KeyboardInterrupt:
        print("\nstopped.", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    a = [x for x in sys.argv[1:]]
    ck = a[0] if len(a) > 0 and a[0] not in ("-", "none") else "runs/direct_bc/policy.pt"
    rm = a[1] if len(a) > 1 else None
    main(ckpt=ck, road_map=rm)

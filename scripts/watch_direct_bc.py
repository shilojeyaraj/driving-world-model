"""Watch the DIRECT obs->action BC policy drive in MetaDrive's 3-D window (no world model).

Usage:  python -m scripts.watch_direct_bc runs/direct_bc/policy_boosted.pt O   # roundabout
        python -m scripts.watch_direct_bc runs/direct_bc/policy_boosted.pt O --small  # smoother recording
        python -m scripts.watch_direct_bc - SSSS                                # IDM on highway
        python -m scripts.watch_direct_bc runs/direct_bc/policy.pt             # default policy

For recording (Win+Alt+R): pass --small to drop to 640x480 and strip traffic -- the lighter
render gives a steadier framerate. Low-graphics mode (shadows/skybox off) is always on.
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch


def main(ckpt="runs/direct_bc/policy.pt", road_map=None, steps=2000, num_scenarios=50,
         collect_steps=4000, direct_steps=4000, traffic_density=0.1, window_size=(800, 600)):
    from scripts.dagger import build_cfg
    from training.direct_bc import DirectPolicy, train_direct_bc, save_direct, load_direct
    if isinstance(road_map, str) and road_map.isdigit():
        road_map = int(road_map)
    cfg = build_cfg(num_scenarios=num_scenarios, road_map=road_map, traffic_density=traffic_density)
    cfg.metadrive_window_size = window_size

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
    print(f"window {window_size[0]}x{window_size[1]}, traffic {traffic_density}, map={road_map or 'random'}", flush=True)
    env = MetaDriveEnv(metadrive_config(cfg))
    obs = adapt_obs(env.reset()[0], "state")
    if getattr(cfg, "metadrive_low_graphics", True):
        disable_shadows(env)
    policy.eval()
    print("watching -- close the window or Ctrl+C to stop.", flush=True)
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
    p = argparse.ArgumentParser()
    p.add_argument("ckpt", nargs="?", default="runs/direct_bc/policy.pt",
                   help="policy checkpoint (- or omit for default)")
    p.add_argument("map", nargs="?", default=None,
                   help="road map: O roundabout, SSSS highway, X intersection, etc.")
    p.add_argument("--small", action="store_true",
                   help="640x480 window + no traffic: steadier framerate for screen recording")
    p.add_argument("--window", nargs=2, type=int, default=None, metavar=("W", "H"),
                   help="explicit window size, e.g. --window 1024 768")
    p.add_argument("--traffic", type=float, default=None,
                   help="traffic density (0.0 = empty, default 0.1)")
    p.add_argument("--steps", type=int, default=2000)
    a = p.parse_args()
    ck = a.ckpt if a.ckpt not in ("-", "none") else "runs/direct_bc/policy.pt"
    win = tuple(a.window) if a.window else ((640, 480) if a.small else (800, 600))
    traffic = a.traffic if a.traffic is not None else (0.0 if a.small else 0.1)
    main(ckpt=ck, road_map=a.map, steps=a.steps, window_size=win, traffic_density=traffic)

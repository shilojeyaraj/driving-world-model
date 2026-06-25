"""Record a smooth top-down video of MetaDrive to a GIF or MP4.

Renders OFFSCREEN (no window, no GPU contention) so every frame is captured at the exact
simulation rate -- no dropped or duplicated frames, no choppiness. For a 3-D live view use
watch_direct_bc / watch_metadrive_3d; this script is for clean recordings.

policy arg:
  "idm"                       -- MetaDrive's built-in expert (default; cleanest driving)
  "forward"                   -- simple throttle-only baseline
  path/to/policy.pt           -- any DirectPolicy checkpoint (e.g. policy_boosted.pt)

Usage:  python -m scripts.record_metadrive                                  # IDM, runs/metadrive_drive.gif
        python -m scripts.record_metadrive idm O                            # IDM on roundabout
        python -m scripts.record_metadrive runs/direct_bc/policy_boosted.pt O   # OUR best policy, roundabout
        python -m scripts.record_metadrive runs/direct_bc/policy_boosted.pt O --steps 400 --out runs/roundabout.gif
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


def _frame(env, size=320):
    rgb = np.asarray(env.render(mode="topdown", window=False))      # (800,800,3) uint8
    from PIL import Image
    return np.asarray(Image.fromarray(rgb).resize((size, size)))


def main(policy="idm", steps=300, out="runs/metadrive_drive.gif", road_map=None,
         traffic_density=0.0, frame_size=320, fps=20):
    import imageio.v2 as imageio
    from metadrive.envs import MetaDriveEnv
    from config import get_config
    from envs.metadrive_env import metadrive_config, adapt_obs

    if isinstance(road_map, str) and road_map.isdigit():
        road_map = int(road_map)

    cfg_obj = get_config(env="metadrive", obs_type="state", max_episode_steps=1000,
                         metadrive_map=road_map, metadrive_traffic_density=traffic_density,
                         metadrive_num_scenarios=50, metadrive_start_seed=50)
    cfg = metadrive_config(cfg_obj)

    # --- resolve policy ---
    direct_policy = None
    use_idm = False
    if os.path.isfile(policy):
        import torch
        from training.direct_bc import load_direct
        direct_policy = load_direct(policy)
        direct_policy.eval()
        print(f"recording with direct policy: {policy}", flush=True)
    elif policy == "idm":
        try:
            from metadrive.policy.idm_policy import IDMPolicy
            cfg["agent_policy"] = IDMPolicy
            use_idm = True
            print("recording with IDM expert", flush=True)
        except Exception as e:
            print(f"IDM unavailable ({e}); falling back to forward", flush=True)
    if not use_idm and direct_policy is None:
        print("recording with forward policy", flush=True)

    env = MetaDriveEnv(cfg)
    raw_obs, _ = env.reset()
    obs = adapt_obs(raw_obs, "state") if direct_policy is not None else None

    frames = []
    for t in range(steps):
        if direct_policy is not None:
            import torch
            with torch.no_grad():
                action = direct_policy(
                    torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                ).squeeze(0).numpy()
        elif use_idm:
            action = np.zeros(2, dtype=np.float32)
        else:
            action = np.array([0.0, 0.4], dtype=np.float32)

        raw_obs, r, terminated, truncated, info = env.step(action)
        frames.append(_frame(env, size=frame_size))

        if terminated or truncated:
            raw_obs, _ = env.reset()
        obs = adapt_obs(raw_obs, "state") if direct_policy is not None else None
    env.close()

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    duration = 1.0 / fps
    imageio.mimsave(out, frames, duration=duration, loop=0)
    label = os.path.basename(policy) if os.path.isfile(policy) else policy
    print(f"saved {out}  ({len(frames)} frames @ {fps}fps, policy={label})", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("policy", nargs="?", default="idm",
                   help="'idm', 'forward', or path to a direct policy .pt")
    p.add_argument("map", nargs="?", default=None,
                   help="road map: O roundabout, SSSS highway, X intersection, etc.")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--out", default="runs/metadrive_drive.gif")
    p.add_argument("--fps", type=int, default=20, help="output frame rate (default 20)")
    p.add_argument("--size", type=int, default=320, help="frame size in pixels (default 320)")
    p.add_argument("--traffic", type=float, default=0.0)
    a = p.parse_args()
    main(policy=a.policy, steps=a.steps, out=a.out, road_map=a.map,
         traffic_density=a.traffic, frame_size=a.size, fps=a.fps)

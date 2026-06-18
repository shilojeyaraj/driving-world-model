"""Record a watchable VIDEO of the MetaDrive simulation (top-down view) to a GIF you can open.
This is the "actually see it run" deliverable -- it renders real MetaDrive frames offscreen
(no GUI window needed) and saves them.

By default the car is driven by MetaDrive's built-in IDM expert so it actually drives on the
road (cleanest visual); fall back to a gentle forward policy, or load one of our trained
checkpoints to watch OUR policy. Requires MetaDrive (see docs/METADRIVE.md).

Usage:  python -m scripts.record_metadrive                  # IDM expert, runs/metadrive_drive.gif
        python -m scripts.record_metadrive forward          # simple throttle-forward policy
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


def _frame(env, size=320):
    rgb = np.asarray(env.render(mode="topdown", window=False))      # (800,800,3) uint8
    from PIL import Image
    return np.asarray(Image.fromarray(rgb).resize((size, size)))


def main(policy="idm", steps=250, out="runs/metadrive_drive.gif", road_map=None, traffic_density=0.1):
    import imageio.v2 as imageio
    from metadrive.envs import MetaDriveEnv
    from config import get_config
    from envs.metadrive_env import metadrive_config

    if isinstance(road_map, str) and road_map.isdigit():
        road_map = int(road_map)
    cfg = metadrive_config(get_config(env="metadrive", obs_type="state", max_episode_steps=1000,
                                      metadrive_map=road_map, metadrive_traffic_density=traffic_density))
    use_idm = False
    if policy == "idm":
        try:
            from metadrive.policy.idm_policy import IDMPolicy
            cfg["agent_policy"] = IDMPolicy                          # env auto-drives with IDM
            use_idm = True
        except Exception as e:
            print(f"IDM expert unavailable ({type(e).__name__}); using forward policy", flush=True)

    env = MetaDriveEnv(cfg)
    env.reset()
    frames = []
    for t in range(steps):
        action = np.array([0.0, 0.0], dtype=np.float32) if use_idm else np.array([0.0, 0.4], dtype=np.float32)
        _, r, terminated, truncated, info = env.step(action)
        frames.append(_frame(env))
        if terminated or truncated:
            env.reset()
    env.close()

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    imageio.mimsave(out, frames, duration=0.06, loop=0)
    print(f"saved {out}  ({len(frames)} frames, policy={'idm' if use_idm else policy})", flush=True)


if __name__ == "__main__":
    pol = sys.argv[1] if len(sys.argv) > 1 else "idm"
    rm = sys.argv[2] if len(sys.argv) > 2 else None
    main(policy=pol, road_map=rm)

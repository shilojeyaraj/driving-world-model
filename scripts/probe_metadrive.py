"""Probe MetaDrive's ACTUAL observation shapes for YOUR installed version, so you can set
cfg.state_dim / cfg.image_size correctly. Observation dims differ across MetaDrive releases --
this is the #1 thing that breaks. Requires: pip install metadrive-simulator (see docs/METADRIVE.md).

Usage:  python -m scripts.probe_metadrive [state|image]
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from config import get_config
from envs.metadrive_env import MetaDriveDrivingEnv


def main(obs_type="state"):
    cfg = get_config(env="metadrive", obs_type=obs_type, max_episode_steps=100, image_size=64)
    env = MetaDriveDrivingEnv(cfg)
    try:
        o = env.reset()
        print(f"obs_type={obs_type}: adapted shape={o.shape} dtype={o.dtype} "
              f"range=[{float(o.min()):.3f}, {float(o.max()):.3f}]")
        if obs_type == "state":
            print(f"  => set  cfg.state_dim = {o.shape[0]}")
        else:
            print(f"  => adapted image is (C,H,W)={o.shape}; set cfg.image_size to its H/W")
        print("  raw observation_space:", env.observation_space)
        print("  action_space:", env.metadrive_action_space)
        o2, r, done, info = env.step(np.zeros(cfg.action_dim, dtype=np.float32))
        print(f"  step ok: reward={r:.3f} done={done}")
    finally:
        env.close()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "state")

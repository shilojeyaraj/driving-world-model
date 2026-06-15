"""Probe what MetaDrive RENDERING works on this machine (for saving a video of the sim).
Tries top-down render (lightest, most headless-friendly) and reports the frame shape, so we
know how to record the simulation. Requires MetaDrive (see docs/METADRIVE.md).

Usage:  python -m scripts.probe_metadrive_render
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


def main():
    from metadrive.envs import MetaDriveEnv
    env = MetaDriveEnv(dict(use_render=False, horizon=100))
    env.reset()
    env.step(np.array([0.0, 1.0], dtype=np.float32))

    # Approach 1: top-down render returning an RGB array (no GUI window).
    for kwargs in (dict(mode="topdown", window=False),
                   dict(mode="topdown", window=False, screen_size=(256, 256)),
                   dict(mode="top_down", window=False)):
        try:
            frame = env.render(**kwargs)
            arr = np.asarray(frame)
            print(f"OK  env.render({kwargs}) -> type={type(frame).__name__} shape={arr.shape} dtype={arr.dtype}")
        except Exception as e:
            print(f"FAIL env.render({kwargs}) -> {type(e).__name__}: {str(e)[:140]}")

    env.close()


if __name__ == "__main__":
    main()

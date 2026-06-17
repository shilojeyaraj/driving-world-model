"""DonkeyCar (DonkeyGym) wrapper -- the Unity sdsandbox sim behind the same envs/base.py contract,
for FULL RENDERED CAR GRAPHICS. Camera-image observations -> our image-mode world model.

The rest of the codebase never imports gym_donkeycar; it only sees reset()/step() returning a
(C,H,W) image. See docs/DONKEYCAR.md for install (a separate Unity binary + a py3.11 env are the
clean setup) and gotchas.

OK TO USE A LIBRARY / modify freely.
Install: pip install gym-donkeycar   (+ download the Unity sim; py>=3.12 also needs `pyasyncore`)
Docs:    https://docs.donkeycar.com/guide/deep_learning/simulator/
"""
import os

import numpy as np

from .base import DrivingEnv

_LEVELS = {0: "GeneratedRoadsEnv", 1: "WarehouseEnv", 2: "AvcSparkfunEnv", 3: "GeneratedTrackEnv"}


def donkey_image(raw, image_size):
    """Camera frame (H,W,3) uint8 0..255 -> (3, image_size, image_size) float32 in [0,1]. Resizes
    to a square multiple-of-16 so the strided-CNN encoder works (Donkey's native frame is 120x160)."""
    from PIL import Image
    img = np.asarray(raw)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    if img.ndim == 3 and img.shape[0] in (1, 3) and img.shape[-1] not in (1, 3):
        img = np.transpose(img, (1, 2, 0))                       # CHW -> HWC (just in case)
    pil = Image.fromarray(img.astype(np.uint8)).resize((image_size, image_size))
    arr = (np.asarray(pil, dtype=np.float32) / 255.0).transpose(2, 0, 1)
    return np.ascontiguousarray(arr, dtype=np.float32)


def to_donkey_action(action, throttle_max):
    """Our [steer, throttle] in [-1,1] -> DonkeyGym [steer in [-1,1], throttle in [0, throttle_max]].
    (Donkey throttle is non-negative, range [0, 5]; we keep the rest of the codebase in [-1,1] and
    map only here, so recorded actions stay in our convention.)"""
    a = np.asarray(action, dtype=np.float32)
    steer = float(np.clip(a[0], -1.0, 1.0))
    throttle = float(np.clip((a[1] + 1.0) * 0.5, 0.0, 1.0) * throttle_max)   # -1->0, +1->throttle_max
    return np.array([steer, throttle], dtype=np.float32)


class DonkeyDrivingEnv(DrivingEnv):
    def __init__(self, cfg):
        self.cfg = cfg
        self.obs_type = "image"                                  # Donkey is camera-only
        self.action_dim = cfg.action_dim
        self.image_size = cfg.image_size
        self.throttle_max = getattr(cfg, "donkey_throttle", 1.0)

        # The sim binary path is read from $DONKEY_SIM_PATH by gym_donkeycar; mirror cfg into it.
        if getattr(cfg, "donkey_exe_path", None):
            os.environ["DONKEY_SIM_PATH"] = cfg.donkey_exe_path

        import gym_donkeycar.envs.donkey_env as de               # needs `pyasyncore` shim on py>=3.12
        klass = getattr(de, _LEVELS[getattr(cfg, "donkey_level", 3)])
        self._env = klass()                                      # launches/connects to the Unity sim

    def reset(self):
        out = self._env.reset()
        raw = out[0] if isinstance(out, tuple) else out          # old-gym obs vs gymnasium (obs,info)
        return donkey_image(raw, self.image_size)

    def step(self, action):
        out = self._env.step(to_donkey_action(action, self.throttle_max))
        if len(out) == 5:                                        # gymnasium 5-tuple
            raw, reward, terminated, truncated, info = out
            done = bool(terminated or truncated)
        else:                                                    # old-gym 4-tuple (gym_donkeycar 1.0.x)
            raw, reward, done, info = out
        return donkey_image(raw, self.image_size), float(reward), bool(done), info

    def close(self):
        try:
            self._env.close()
        except Exception:
            pass

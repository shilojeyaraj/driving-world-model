"""MetaDrive wrapper. STRUCTURALLY written but VERIFY against your installed MetaDrive
version -- config keys and observation shapes differ across releases (this is the #1 thing
that breaks). The rest of the codebase doesn't care, because it only sees envs/base.py.

OK TO USE A LIBRARY / modify freely.
Install: pip install metadrive-simulator
Docs:    https://metadrive-simulator.readthedocs.io
"""
import numpy as np
from .base import DrivingEnv


class MetaDriveDrivingEnv(DrivingEnv):
    def __init__(self, cfg):
        self.cfg = cfg
        self.obs_type = cfg.obs_type
        self.action_dim = cfg.action_dim
        from metadrive.envs import MetaDriveEnv      # imported here so dummy runs without it

        md_cfg = dict(use_render=False, horizon=cfg.max_episode_steps)
        if cfg.obs_type == "image":
            # VERIFY: enabling an RGB sensor + image_observation is version-specific.
            md_cfg.update(dict(image_observation=True))
        self._env = MetaDriveEnv(md_cfg)

    def _to_obs(self, raw):
        if self.obs_type == "state":
            return np.asarray(raw, dtype=np.float32)
        img = np.asarray(raw, dtype=np.float32)
        if img.ndim == 3 and img.shape[0] not in (1, 3):
            img = np.transpose(img, (2, 0, 1))          # HWC -> CHW
        if img.max() > 1.5:
            img = img / 255.0
        return img.astype(np.float32)

    def reset(self):
        raw, _ = self._env.reset()                      # gymnasium API: (obs, info)
        return self._to_obs(raw)

    def step(self, action):
        raw, reward, terminated, truncated, info = self._env.step(np.asarray(action))
        return self._to_obs(raw), float(reward), bool(terminated or truncated), info

"""MetaDrive wrapper -- a REAL driving sim behind the same envs/base.py contract.

The rest of the codebase never imports MetaDrive; it only sees reset()/step() returning a
state vector or a (C,H,W) image. So adopting MetaDrive is localized to this file.

INSTALL + USAGE + GOTCHAS: see docs/METADRIVE.md. The #1 thing that breaks across MetaDrive
versions is the OBSERVATION SHAPE/DIM -- run `python -m scripts.probe_metadrive` to print the
actual state_dim / image shape for YOUR install, then set cfg.state_dim / cfg.image_size to match.

OK TO USE A LIBRARY / modify freely.
Install: pip install metadrive-simulator      Docs: https://metadrive-simulator.readthedocs.io
"""
import numpy as np

from .base import DrivingEnv


def adapt_obs(raw, obs_type):
    """Normalize a raw MetaDrive observation to the envs/base.py contract. Pure + dependency-free
    so it can be unit-tested without MetaDrive (this is where version differences bite):

    - state -> a flat float32 vector. (raw may be a dict for multi-modal obs.)
    - image -> a float32 (C,H,W) in [0,1]. MetaDrive may return HWC, may STACK frames as a
      trailing axis (H,W,C,stack), and may be 0..255; we take the last frame, move to CHW, scale.
    """
    if obs_type == "state":
        if isinstance(raw, dict):                       # multi-modal -> pick the vector entry
            raw = raw.get("state", next(iter(raw.values())))
        return np.asarray(raw, dtype=np.float32).reshape(-1)

    if isinstance(raw, dict):                            # image obs often nested under "image"
        raw = raw.get("image", next(iter(raw.values())))
    img = np.asarray(raw, dtype=np.float32)
    if img.ndim == 4:                                   # (H,W,C,stack) -> most recent frame
        img = img[..., -1]
    if img.ndim == 3 and img.shape[0] not in (1, 3):    # HWC -> CHW
        img = np.transpose(img, (2, 0, 1))
    if img.max() > 1.5:                                 # 0..255 -> 0..1
        img = img / 255.0
    return np.ascontiguousarray(img, dtype=np.float32)


class MetaDriveDrivingEnv(DrivingEnv):
    def __init__(self, cfg):
        self.cfg = cfg
        self.obs_type = cfg.obs_type
        self.action_dim = cfg.action_dim                # MetaDrive action = [steering, throttle], in [-1,1]
        from metadrive.envs import MetaDriveEnv         # imported here so the dummy env runs without it

        md_cfg = dict(use_render=False, horizon=cfg.max_episode_steps)
        if cfg.obs_type == "image":
            # VERSION-SPECIFIC: enabling a camera + image observation differs across releases
            # (older: image_observation=True; newer: a `sensors=` RGBCamera spec). See
            # docs/METADRIVE.md. Image mode also generally needs a GPU/offscreen renderer.
            md_cfg.update(image_observation=True)
        try:
            self._env = MetaDriveEnv(config=md_cfg)
        except TypeError:                               # some versions take the dict positionally
            self._env = MetaDriveEnv(md_cfg)
        self.observation_space = getattr(self._env, "observation_space", None)
        self.metadrive_action_space = getattr(self._env, "action_space", None)

    def reset(self):
        out = self._env.reset()
        raw = out[0] if isinstance(out, tuple) else out  # gymnasium (obs, info) vs old gym obs
        return adapt_obs(raw, self.obs_type)

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        out = self._env.step(action)
        if len(out) == 5:                                # gymnasium: obs, r, terminated, truncated, info
            raw, reward, terminated, truncated, info = out
            done = bool(terminated or truncated)
        else:                                            # old gym: obs, r, done, info
            raw, reward, done, info = out
        return adapt_obs(raw, self.obs_type), float(reward), bool(done), info

    def close(self):
        if getattr(self, "_env", None) is not None:
            self._env.close()

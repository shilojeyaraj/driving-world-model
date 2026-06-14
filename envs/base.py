"""Environment interface. Everything downstream depends ONLY on this contract, so
Dummy / MetaDrive / CARLA are interchangeable.

OK TO USE A LIBRARY.

reset()/step() return an observation that is either:
  - a 1-D float32 vector                    (obs_type == "state")
  - a (C, H, W) float32 image in [0, 1]     (obs_type == "image")
"""
from abc import ABC, abstractmethod
import numpy as np


class DrivingEnv(ABC):
    obs_type: str
    action_dim: int

    @abstractmethod
    def reset(self) -> np.ndarray: ...

    @abstractmethod
    def step(self, action: np.ndarray):
        """Returns (obs, reward, done, info)."""
        ...

    def close(self):
        """Release env resources. No-op by default; real sims (MetaDrive) override it."""
        pass


def make_env(cfg):
    if cfg.env == "dummy":
        from .dummy import DummyDrivingEnv
        return DummyDrivingEnv(cfg)
    if cfg.env == "metadrive":
        from .metadrive_env import MetaDriveDrivingEnv
        return MetaDriveDrivingEnv(cfg)
    raise ValueError(f"unknown env: {cfg.env}")

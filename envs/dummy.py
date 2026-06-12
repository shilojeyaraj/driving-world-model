"""Dependency-free toy driving env. Not realistic -- its only job is to let you smoke-test
the whole pipeline (buffer, shapes, training loop) on your laptop before installing
MetaDrive or implementing the models.

OK TO USE A LIBRARY / modify freely.
"""
import numpy as np
from .base import DrivingEnv


class DummyDrivingEnv(DrivingEnv):
    def __init__(self, cfg):
        self.cfg = cfg
        self.obs_type = cfg.obs_type
        self.action_dim = cfg.action_dim
        self._t = 0
        self._pos = 0.0

    def _obs(self):
        if self.obs_type == "state":
            v = np.zeros(self.cfg.state_dim, dtype=np.float32)
            v[0] = np.float32(self._pos)
            v[1:] = (np.random.randn(self.cfg.state_dim - 1) * 0.1).astype(np.float32)
            return v
        return np.random.rand(3, self.cfg.image_size, self.cfg.image_size).astype(np.float32)

    def reset(self):
        self._t, self._pos = 0, 0.0
        return self._obs()

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        steer, throttle = float(action[0]), float(action[1])
        self._pos += throttle * 0.1
        self._t += 1
        reward = float(throttle - abs(steer))            # toy: go forward, don't swerve
        done = self._t >= self.cfg.max_episode_steps
        return self._obs(), reward, done, {}

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
        return self._render()

    def _render(self):
        """Render pos as a Gaussian blob whose horizontal position encodes pos. A pure
        function of state (no noise), so the world model can learn dynamics from pixels:
        throttle -> pos -> the blob slides right/left, exactly the thing imagination must
        predict. pos is squashed through tanh so the column stays on-canvas as pos drifts."""
        H = W = self.cfg.image_size
        cx = (np.tanh(self._pos / 3.0) * 0.5 + 0.5) * (W - 1)   # pos -> column (monotonic, bounded)
        cy = (H - 1) / 2.0
        sigma = max(1.0, W * 0.1)
        ys, xs = np.mgrid[0:H, 0:W]
        blob = np.exp(-(((xs - cx) ** 2 + (ys - cy) ** 2) / (2.0 * sigma ** 2)))
        return np.broadcast_to(blob, (3, H, W)).astype(np.float32)

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

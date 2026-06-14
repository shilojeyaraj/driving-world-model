"""Visual DummyEnv (V1): in image mode the observation must be a RENDER of the state (pos),
not random noise -- so the world model can actually learn dynamics from pixels.
"""
import numpy as np

from config import get_config
from envs.base import make_env


def _cfg(**ov):
    d = dict(obs_type="image", env="dummy", image_size=16, max_episode_steps=50)
    d.update(ov)
    return get_config(**d)


def _center_x(frame):
    """Horizontal center-of-mass of the rendered blob (channel 0)."""
    col_mass = frame[0].sum(axis=0)                 # sum over rows -> (W,)
    return float((col_mass * np.arange(frame.shape[-1])).sum() / (col_mass.sum() + 1e-8))


def test_visual_obs_is_rendered_deterministically():
    cfg = _cfg()
    env = make_env(cfg)
    o1 = env.reset()
    assert o1.shape == (3, 16, 16)
    assert o1.min() >= 0.0 and o1.max() <= 1.0001
    # A render is a pure function of state: two resets (pos=0) give the SAME frame.
    o2 = env.reset()
    assert np.allclose(o1, o2), "image obs is not deterministic given state (still noise?)"


def test_blob_moves_right_with_positive_throttle():
    cfg = _cfg()
    env = make_env(cfg)
    o0 = env.reset()
    x0 = _center_x(o0)
    o = o0
    for _ in range(10):
        o, _, _, _ = env.step(np.array([0.0, 1.0], dtype=np.float32))   # steer 0, throttle +1
    x1 = _center_x(o)
    assert x1 > x0 + 0.5, f"blob did not move with throttle: {x0:.2f} -> {x1:.2f}"

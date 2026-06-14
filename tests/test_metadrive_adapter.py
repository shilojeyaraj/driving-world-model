"""MetaDrive obs-adapter tests. The adapter (envs.metadrive_env.adapt_obs) is the part that
breaks across MetaDrive versions (obs shape / channel order / value range / frame stacking),
so it's a PURE function we can test thoroughly WITHOUT installing MetaDrive. A live smoke test
runs only if MetaDrive is actually importable.
"""
import numpy as np
import pytest

from envs.metadrive_env import adapt_obs


def test_state_vector_is_flattened_float32():
    out = adapt_obs(np.arange(259, dtype=np.float64), "state")
    assert out.shape == (259,) and out.dtype == np.float32


def test_state_from_multimodal_dict_picks_state_entry():
    raw = {"state": np.zeros(19, dtype=np.float32), "image": np.zeros((8, 8, 3))}
    assert adapt_obs(raw, "state").shape == (19,)


def test_image_hwc_uint8_is_transposed_and_scaled():
    raw = (np.random.rand(16, 16, 3) * 255).astype(np.uint8)
    out = adapt_obs(raw, "image")
    assert out.shape == (3, 16, 16) and out.dtype == np.float32
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_image_already_chw_in_unit_range_is_unchanged_shape():
    raw = np.random.rand(3, 16, 16).astype(np.float32)
    out = adapt_obs(raw, "image")
    assert out.shape == (3, 16, 16) and out.max() <= 1.0


def test_image_stacked_frames_takes_most_recent():
    raw = np.zeros((16, 16, 3, 3), dtype=np.float32)   # (H, W, C, stack)
    raw[..., -1] = 1.0                                  # only the last frame is bright
    out = adapt_obs(raw, "image")
    assert out.shape == (3, 16, 16)
    assert np.allclose(out, 1.0), "did not select the most recent stacked frame"


def test_image_from_dict_under_image_key():
    raw = {"image": (np.random.rand(8, 8, 3) * 255).astype(np.uint8)}
    out = adapt_obs(raw, "image")
    assert out.shape == (3, 8, 8) and out.max() <= 1.0


def test_metadrive_live_smoke():
    """Runs only where MetaDrive is installed (see docs/METADRIVE.md); otherwise skipped."""
    pytest.importorskip("metadrive")
    import numpy as np
    from config import get_config
    from envs.metadrive_env import MetaDriveDrivingEnv

    cfg = get_config(env="metadrive", obs_type="state", max_episode_steps=50)
    env = MetaDriveDrivingEnv(cfg)
    try:
        o = env.reset()
        assert o.ndim == 1
        o2, r, done, info = env.step(np.zeros(cfg.action_dim, dtype=np.float32))
        assert o2.shape == o.shape and isinstance(r, float) and isinstance(done, bool)
    finally:
        env.close()

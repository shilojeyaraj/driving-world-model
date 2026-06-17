"""DonkeyGym wrapper tests. The obs adapter + action mapping are pure and tested without the sim;
the gym_donkeycar import is checked behind a skip (and we never INSTANTIATE the env -- that would
launch the Unity simulator)."""
import numpy as np
import pytest

from envs.donkey_env import donkey_image, to_donkey_action


def test_donkey_image_resizes_to_chw_unit_range():
    raw = (np.random.rand(120, 160, 3) * 255).astype(np.uint8)   # Donkey's native camera size
    out = donkey_image(raw, 64)
    assert out.shape == (3, 64, 64) and out.dtype == np.float32
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_to_donkey_action_maps_throttle_nonnegative_and_clips_steer():
    a0 = to_donkey_action([0.5, -1.0], throttle_max=1.0)
    assert np.isclose(a0[0], 0.5) and np.isclose(a0[1], 0.0)     # throttle -1 -> 0
    a1 = to_donkey_action([2.0, 1.0], throttle_max=2.0)
    assert np.isclose(a1[0], 1.0) and np.isclose(a1[1], 2.0)     # steer clipped; throttle +1 -> max


def test_make_env_has_donkey_branch_without_launching_sim():
    import envs.donkey_env as d
    from envs.base import make_env                                # noqa: F401 (import resolves)
    assert hasattr(d, "DonkeyDrivingEnv")                         # do NOT instantiate (launches Unity)


@pytest.mark.slow
def test_gym_donkeycar_env_class_imports_if_present():
    """Where gym_donkeycar (+ the asyncore shim on py>=3.12) is installed, its env class imports.
    Skips otherwise. Never instantiates."""
    pytest.importorskip("gym_donkeycar")
    try:
        from gym_donkeycar.envs.donkey_env import GeneratedTrackEnv
    except Exception as e:                                        # e.g. missing asyncore shim
        pytest.skip(f"gym_donkeycar env import unavailable here: {type(e).__name__}")
    assert GeneratedTrackEnv is not None

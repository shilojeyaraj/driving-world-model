"""MetaDrive obs-adapter tests. The adapter (envs.metadrive_env.adapt_obs) is the part that
breaks across MetaDrive versions (obs shape / channel order / value range / frame stacking),
so it's a PURE function we can test thoroughly WITHOUT installing MetaDrive. A live smoke test
runs only if MetaDrive is actually importable.
"""
import numpy as np
import pytest

from envs.metadrive_env import (adapt_obs, metadrive_config, disable_shadows, applied_action,
                                train_eval_seed_split)
from config import get_config


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


def test_metadrive_config_render_sets_window_and_low_graphics():
    """When the 3-D window is on, the perf knobs (smaller window + shadows/skybox off) are applied
    so a weak GPU can render smoothly. Pure dict-builder -- no MetaDrive needed."""
    cfg = get_config(env="metadrive", metadrive_render=True, metadrive_window_size=(640, 480),
                     metadrive_low_graphics=True, max_episode_steps=50)
    md = metadrive_config(cfg)
    assert md["use_render"] is True
    assert md["window_size"] == (640, 480)
    assert md["show_skybox"] is False
    assert md["show_logo"] is False
    # shadow_range=0 crashes MetaDrive's PSSM (assert distance>0); shadows are disabled at RUNTIME
    # via the engine, never by zeroing the range here:
    assert "shadow_range" not in md


def test_metadrive_config_headless_omits_render_perf_keys():
    """Headless (top-down / training) path must not gain a window or graphics overrides."""
    cfg = get_config(env="metadrive", metadrive_render=False, max_episode_steps=50)
    md = metadrive_config(cfg)
    assert md["use_render"] is False
    assert "window_size" not in md and "shadow_range" not in md


def test_metadrive_config_full_graphics_keeps_shadows_but_still_sizes_window():
    """Resolution (window_size) is independent of quality (low_graphics): rendering always sets the
    window, but full-graphics mode leaves shadows/skybox at MetaDrive's defaults."""
    cfg = get_config(env="metadrive", metadrive_render=True, metadrive_low_graphics=False,
                     max_episode_steps=50)
    md = metadrive_config(cfg)
    assert "window_size" in md
    assert "show_skybox" not in md and "show_logo" not in md


def test_metadrive_config_endless_disables_early_termination():
    """Endless mode lets a human keep driving through mistakes (no crash/off-road reset). MetaDrive
    ends the episode by default on out_of_road / continuous-line / collision; we switch those off."""
    cfg = get_config(env="metadrive", metadrive_endless=True, max_episode_steps=50)
    md = metadrive_config(cfg)
    assert md["out_of_road_done"] is False
    assert md["on_continuous_line_done"] is False
    assert md["crash_vehicle_done"] is False
    # ...and never truncate at the horizon either, or the car would still reset mid-drive:
    assert md["horizon"] >= 1_000_000


def test_metadrive_config_default_keeps_metadrive_terminations():
    """Default must NOT override MetaDrive's own termination defaults (so training/IDM behave normally),
    and the horizon stays at max_episode_steps."""
    cfg = get_config(env="metadrive", max_episode_steps=50)
    md = metadrive_config(cfg)
    assert "out_of_road_done" not in md and "crash_vehicle_done" not in md
    assert md["horizon"] == 50


def test_metadrive_config_manual_control_enables_keyboard_when_rendering():
    """keyboard driving = MetaDrive's manual_control (WASD). Only valid with the 3-D window, which
    captures key input, so it's gated on use_render."""
    cfg = get_config(env="metadrive", metadrive_render=True, metadrive_manual_control=True,
                     max_episode_steps=50)
    md = metadrive_config(cfg)
    assert md["manual_control"] is True
    assert md["controller"] == "keyboard"


def test_metadrive_config_manual_control_ignored_when_headless():
    """No window -> no keyboard input possible, so manual_control must not be set."""
    cfg = get_config(env="metadrive", metadrive_render=False, metadrive_manual_control=True,
                     max_episode_steps=50)
    md = metadrive_config(cfg)
    assert "manual_control" not in md


def test_applied_action_reads_back_keyboard_action_in_manual_mode():
    """In manual mode MetaDrive overrides our env-input action with the controller's; we must record
    what the vehicle ACTUALLY did (env.agent.last_current_action[-1]), not the dummy we passed."""
    class _Agent: last_current_action = [(0.0, 0.0), (0.5, -0.3)]
    class _Env: agent = _Agent()
    out = applied_action(_Env(), proposed=np.zeros(2, np.float32), manual=True)
    assert np.allclose(out, [0.5, -0.3]) and out.dtype == np.float32
    # non-manual: passthrough of the proposed action
    out2 = applied_action(_Env(), proposed=np.array([0.1, 0.2], np.float32), manual=False)
    assert np.allclose(out2, [0.1, 0.2])
    # defensive: a malformed env never raises -> falls back to proposed
    assert np.allclose(applied_action(object(), np.array([0.7, 0.0], np.float32), manual=True), [0.7, 0.0])


def test_disable_shadows_is_a_safe_noop_without_a_live_engine():
    """disable_shadows touches MetaDrive internals defensively: on anything without a live
    engine/pssm it must return False and never raise (so it can't break a run)."""
    class _NoEngine: pass
    assert disable_shadows(_NoEngine()) is False
    assert disable_shadows(None) is False


def test_metadrive_config_sets_scenario_pool_from_cfg():
    """Map randomization: the scene pool (num_scenarios) + first seed (start_seed) pass straight
    through to MetaDrive, so each reset() samples a map from [start_seed, start_seed+num_scenarios)."""
    cfg = get_config(env="metadrive", metadrive_num_scenarios=100, metadrive_start_seed=5,
                     max_episode_steps=50)
    md = metadrive_config(cfg)
    assert md["num_scenarios"] == 100
    assert md["start_seed"] == 5


def test_metadrive_config_default_is_single_fixed_map():
    """Backward compatible: a default cfg keeps the single-map pool (seed 0) we trained on before."""
    cfg = get_config(env="metadrive", max_episode_steps=50)
    md = metadrive_config(cfg)
    assert md["num_scenarios"] == 1
    assert md["start_seed"] == 0


def test_train_eval_seed_split_gives_disjoint_ranges():
    """The held-out split: train and eval seed ranges must not overlap, so eval drives maps the
    policy never trained on. PURE."""
    (train_start, train_num), (eval_start, eval_num) = train_eval_seed_split(100, 50)
    assert (train_start, train_num) == (0, 100)
    assert (eval_start, eval_num) == (100, 50)
    assert train_start + train_num == eval_start          # adjacent, no gap, no overlap


def test_train_eval_seed_split_honors_base_offset():
    """A non-zero base shifts both ranges together, still disjoint."""
    train, ev = train_eval_seed_split(10, 4, base=1000)
    assert train == (1000, 10)
    assert ev == (1010, 4)


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

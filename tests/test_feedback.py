"""Driving-feedback engine (GF4). The 3 signals + the report are testable on tiny untrained
models and synthetic traces -- no MetaDrive, no webcam, no training quality needed (we only
check shapes/ranges/wiring and event detection)."""
import numpy as np
import torch

from config import get_config
from envs.base import make_env
from models.world_model import WorldModel
from models.actor_critic import Actor, Critic
from eval.feedback import (forecast_safety, style_deviation, state_value, DrivingFeedback,
                           report_from_traces, should_forecast)


def _cfg(**ov):
    d = dict(obs_type="state", env="dummy", state_dim=35, deter_dim=16, stoch_dim=4,
             hidden_dim=16, action_dim=2, forecast_horizon=5, risk_threshold=0.5,
             max_episode_steps=20)
    d.update(ov)
    return get_config(**d)


def _models(cfg):
    feat_dim = cfg.deter_dim + cfg.stoch_dim
    return WorldModel(cfg, cfg.action_dim), Actor(cfg, feat_dim, cfg.action_dim), Critic(cfg, feat_dim)


def test_forecast_safety_keys_and_ranges():
    cfg = _cfg()
    wm, _, _ = _models(cfg)
    state = wm.rssm.initial_state(1, torch.device("cpu"))
    out = forecast_safety(wm, state, np.array([0.0, 1.0], np.float32), horizon=cfg.forecast_horizon)
    assert {"survival", "pred_return", "risk"} <= set(out)
    assert 0.0 <= out["survival"] <= 1.0
    assert isinstance(out["risk"], bool)


def test_style_deviation_shapes():
    cfg = _cfg()
    _, actor, _ = _models(cfg)
    feat = torch.randn(1, cfg.deter_dim + cfg.stoch_dim)
    out = style_deviation(actor, feat, np.array([0.5, -0.3], np.float32))
    assert {"d_steer", "d_throttle", "surprise"} <= set(out)
    assert isinstance(out["d_steer"], float) and isinstance(out["surprise"], float)


def test_state_value_is_float():
    cfg = _cfg()
    _, _, critic = _models(cfg)
    assert isinstance(state_value(critic, torch.randn(1, cfg.deter_dim + cfg.stoch_dim)), float)


def test_driving_feedback_step_and_report():
    torch.manual_seed(0)
    cfg = _cfg()
    wm, actor, critic = _models(cfg)
    fb = DrivingFeedback(wm, actor, critic, cfg)
    env = make_env(cfg)
    obs = env.reset()
    for _ in range(8):
        action = np.random.uniform(-1, 1, cfg.action_dim).astype(np.float32)
        out = fb.step(obs, action)
        assert {"survival", "value", "d_steer", "d_throttle"} <= set(out)
        obs, _, done, _ = env.step(action)
        if done:
            obs = env.reset()
    report = fb.finalize()
    assert "events" in report and report["n_steps"] == 8


def test_should_forecast_cadence():
    # every<=1 -> recompute every step (no throttling)
    assert all(should_forecast(i, 1) for i in range(5))
    # every=3 -> recompute at 0,3,6; reuse the cached forecast in between
    assert [should_forecast(i, 3) for i in range(7)] == [True, False, False, True, False, False, True]
    # defensive: every<=0 behaves like "every step", never divides by zero
    assert all(should_forecast(i, 0) for i in range(3))


def test_feedback_forecast_gating_reuses_last_forecast():
    """forecast=False skips the EXPENSIVE imagine (the laptop win) but still advances the recurrent
    state and recomputes the cheap style/value signals, so the HUD/traces stay live and aligned."""
    torch.manual_seed(0)
    cfg = _cfg()
    wm, actor, critic = _models(cfg)
    fb = DrivingFeedback(wm, actor, critic, cfg)
    obs0 = np.random.randn(cfg.state_dim).astype(np.float32)
    out0 = fb.step(obs0, np.array([0.0, 1.0], np.float32), forecast=True)
    obs1 = np.random.randn(cfg.state_dim).astype(np.float32)
    out1 = fb.step(obs1, np.array([1.0, -1.0], np.float32), forecast=False)
    # the safety forecast (A) is reused from the last real forecast...
    assert out1["survival"] == out0["survival"]
    assert out1["pred_return"] == out0["pred_return"]
    assert out1["risk"] == out0["risk"]
    # ...but style (B) and value (C) are recomputed on the advanced state + new action:
    assert out1["d_steer"] != out0["d_steer"]
    assert out1["value"] != out0["value"]
    # every step still appends a full row, so the offline report stays aligned:
    assert len(fb.traces["survival"]) == 2


def test_report_detects_injected_events():
    traces = {
        "survival": [0.9, 0.9, 0.1, 0.9],
        "pred_return": [1.0, 1.0, 1.0, 1.0],
        "d_steer": [0.0, 0.0, 0.0, 0.9],
        "d_throttle": [0.0, 0.0, 0.0, 0.0],
        "value": [1.0, 1.0, 1.0, 1.0],
        "risk": [False, False, True, False],
    }
    rep = report_from_traces(traces, _cfg())
    labels = {e["type"] for e in rep["events"]}
    assert "near_off_road" in labels      # the low-survival / risk step
    assert "oversteer" in labels          # the large steer-deviation step

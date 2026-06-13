"""Open-loop prediction eval (spec §4.7, §7).

  - contract: returns per-horizon error arrays for the action-conditioned model and the
    no-action baseline.
  - milestone gate (spec §9): with the true actions, prediction error is LOWER than with no
    actions -- proof the model learned action-conditioned DYNAMICS, not a video autoplayer.
"""
import numpy as np
import pytest
import torch

from config import get_config
from envs.base import make_env
from data.replay_buffer import SequenceReplayBuffer
from models.world_model import WorldModel
from eval.open_loop import open_loop_eval


def _cfg(**ov):
    d = dict(obs_type="state", env="dummy", state_dim=35, deter_dim=64, stoch_dim=16,
             hidden_dim=64, seq_len=20, free_bits=1.0, kl_scale=1.0, max_episode_steps=200)
    d.update(ov)
    return get_config(**d)


def _pool(cfg, steps=2000, seed=0):
    np.random.seed(seed)
    env = make_env(cfg)
    buf = SequenceReplayBuffer(cfg.buffer_capacity, cfg.seq_len)
    obs = env.reset()
    for _ in range(steps):
        a = np.random.uniform(-1, 1, cfg.action_dim).astype(np.float32)
        nxt, r, done, _ = env.step(a)
        buf.add(obs, a, r, done)
        obs = env.reset() if done else nxt
    return buf


def _batch(buf):
    return {k: torch.as_tensor(v) for k, v in buf.sample(16).items()}


def test_open_loop_returns_per_horizon_errors():
    torch.manual_seed(0)
    cfg = _cfg()
    wm = WorldModel(cfg, cfg.action_dim)
    batch = _batch(_pool(cfg, seed=0))

    H = 10
    out = open_loop_eval(wm, batch, context=5, horizon=H)

    assert len(out["model"]) == H
    assert len(out["no_action"]) == H
    assert np.all(np.asarray(out["model"]) >= 0)


@pytest.mark.slow
def test_action_conditioning_beats_no_action_baseline():
    """Spec §9 gate: after training, true-action open-loop prediction must beat the
    no-action rollout -- especially as the horizon grows, since the only thing that drives
    the future away from 'nothing happens' is the action (pos integrates throttle)."""
    torch.manual_seed(0)
    cfg = _cfg()
    wm = WorldModel(cfg, cfg.action_dim)
    opt = torch.optim.Adam(wm.parameters(), lr=3e-3)

    pool = _pool(cfg, steps=3000, seed=0)
    for _ in range(600):
        loss, _ = wm.assemble_loss(_batch(pool))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(wm.parameters(), 100.0)
        opt.step()

    torch.manual_seed(1)
    out = open_loop_eval(wm, _batch(_pool(cfg, seed=777)), context=5, horizon=10)
    model, no_action = np.asarray(out["model"]), np.asarray(out["no_action"])
    print("\nhorizon :", out["horizon"])
    print("model   :", np.round(model, 4))
    print("no_act  :", np.round(no_action, 4))
    print("repeat  :", np.round(np.asarray(out["repeat_last"]), 4))

    # Action-conditioning lowers error overall, and the gap is clearest at long horizon.
    assert model.sum() < no_action.sum(), (model.sum(), no_action.sum())
    assert model[-3:].mean() < no_action[-3:].mean(), (model[-3:], no_action[-3:])

"""Closed-loop eval (spec §4.8, §9): run the trained actor IN the env, maintaining the RSSM
posterior across steps, and measure DRIVING (episode return), not prediction.

  - contract: runs episodes, returns actor/random returns + action stats.
  - milestone gate: the dream-trained actor beats the random baseline and drives
    (throttle -> +1, steer -> 0) in the REAL env -- closing the loop.
"""
import numpy as np
import pytest
import torch

from config import get_config
from envs.base import make_env
from data.replay_buffer import SequenceReplayBuffer
from models.world_model import WorldModel
from models.actor_critic import Actor, Critic
from training.train_behavior import train_behavior_in_imagination
from eval.closed_loop import closed_loop_eval


def _cfg(**ov):
    d = dict(obs_type="state", env="dummy", state_dim=35, deter_dim=64, stoch_dim=16,
             hidden_dim=64, seq_len=16, free_bits=1.0, kl_scale=1.0, max_episode_steps=200,
             imagine_horizon=10, gamma=0.99, lambda_=0.95, entropy_coef=1e-3,
             actor_lr=3e-3, critic_lr=3e-3, batch_size=16)
    d.update(ov)
    return get_config(**d)


def _pool(cfg, steps=3000, seed=0):
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


def _train_world_model(cfg, buf, steps=400):
    wm = WorldModel(cfg, cfg.action_dim)
    opt = torch.optim.Adam(wm.parameters(), lr=3e-3)
    for _ in range(steps):
        batch = {k: torch.as_tensor(v) for k, v in buf.sample(cfg.batch_size).items()}
        loss, _ = wm.assemble_loss(batch)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(wm.parameters(), 100.0)
        opt.step()
    return wm


def test_closed_loop_returns_metrics():
    torch.manual_seed(0)
    cfg = _cfg(max_episode_steps=20)
    env = make_env(cfg)
    wm = WorldModel(cfg, cfg.action_dim)
    actor = Actor(cfg, cfg.deter_dim + cfg.stoch_dim, cfg.action_dim)

    out = closed_loop_eval(actor, wm, env, episodes=3, max_steps=20)

    for k in ("actor_return", "random_return", "actor_throttle", "actor_steer"):
        assert k in out and isinstance(out[k], float)


@pytest.mark.slow
def test_trained_actor_drives_and_beats_random():
    """Spec §9 closing gate: the actor trained PURELY in imagination drives the REAL env --
    throttle -> +1, steer -> 0 -- and its episode return beats the random baseline."""
    torch.manual_seed(0)
    cfg = _cfg(max_episode_steps=100)
    buf = _pool(cfg, seed=0)
    wm = _train_world_model(cfg, buf, steps=400)
    feat_dim = cfg.deter_dim + cfg.stoch_dim
    actor, critic = Actor(cfg, feat_dim, cfg.action_dim), Critic(cfg, feat_dim)
    train_behavior_in_imagination(cfg, wm, buf, actor, critic, steps=500, log_every=500)

    env = make_env(cfg)
    out = closed_loop_eval(actor, wm, env, episodes=5, max_steps=100)
    print("\nclosed-loop:", {k: round(v, 3) for k, v in out.items()})

    assert out["actor_return"] > out["random_return"], out
    assert out["actor_throttle"] > 0.8, out
    assert abs(out["actor_steer"]) < 0.2, out

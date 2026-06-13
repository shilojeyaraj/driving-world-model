"""Behavior-training milestone gate (spec §9): the actor, trained PURELY in imagination from
a frozen world model, should discover the env's optimal action -- throttle ~ +1, steer ~ 0
(because reward = throttle - |steer|) -- with zero environment steps during policy learning.
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


@pytest.mark.slow
def test_actor_learns_optimal_action_in_imagination():
    torch.manual_seed(0)
    cfg = _cfg()
    buf = _pool(cfg, seed=0)

    wm = _train_world_model(cfg, buf, steps=400)
    feat_dim = cfg.deter_dim + cfg.stoch_dim
    actor, critic = Actor(cfg, feat_dim, cfg.action_dim), Critic(cfg, feat_dim)
    train_behavior_in_imagination(cfg, wm, buf, actor, critic, steps=500, log_every=100)

    # Evaluate the deterministic policy on a batch of real latent states.
    obs = torch.as_tensor(buf.sample(cfg.batch_size)["obs"])
    B, T = obs.shape[:2]
    with torch.no_grad():
        emb = wm.encoder(obs.reshape(B * T, cfg.state_dim)).reshape(B, T, -1)
        feat = wm.rssm.observe(emb, torch.zeros(B, T, cfg.action_dim),
                               wm.rssm.initial_state(B, obs.device))["feat"]
        action, _ = actor(feat.reshape(B * T, -1), deterministic=True)
    steer, throttle = action[:, 0].mean().item(), action[:, 1].mean().item()
    print(f"\nlearned policy: throttle={throttle:.3f}  steer={steer:.3f}")

    assert throttle > 0.8, f"actor did not learn to accelerate: throttle={throttle:.3f}"
    assert abs(steer) < 0.2, f"actor did not learn steer~0: steer={steer:.3f}"

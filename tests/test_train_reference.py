"""Reference-stack training (GF3). The IDM data collection needs MetaDrive, but the two
learning steps -- behavior-cloning the reference Actor and policy-evaluating the Critic -- are
pure given a buffer, so they're tested here on a DummyEnv buffer with a KNOWN policy (no
MetaDrive needed)."""
import numpy as np
import torch

from config import get_config
from envs.base import make_env
from data.replay_buffer import SequenceReplayBuffer
from models.world_model import WorldModel
from training.train_reference import bc_actor, eval_critic


def _cfg(**ov):
    d = dict(obs_type="state", env="dummy", state_dim=35, deter_dim=32, stoch_dim=8,
             hidden_dim=32, action_dim=2, seq_len=10, batch_size=8, max_episode_steps=40,
             gamma=0.99, lambda_=0.95)
    d.update(ov)
    return get_config(**d)


def _constant_action_buffer(cfg, const, steps=1200, seed=0):
    """DummyEnv data driven by a FIXED action -> behavior-cloning should recover that action."""
    np.random.seed(seed)
    env = make_env(cfg)
    buf = SequenceReplayBuffer(cfg.buffer_capacity, cfg.seq_len)
    obs = env.reset()
    a = np.asarray(const, dtype=np.float32)
    for _ in range(steps):
        nxt, r, done, _ = env.step(a)
        buf.add(obs, a, r, done)
        obs = env.reset() if done else nxt
    return buf


def _trained_wm(cfg, buf, steps=150):
    wm = WorldModel(cfg, cfg.action_dim)
    opt = torch.optim.Adam(wm.parameters(), lr=3e-3)
    for _ in range(steps):
        batch = {k: torch.as_tensor(v) for k, v in buf.sample(cfg.batch_size).items()}
        loss, _ = wm.assemble_loss(batch)
        opt.zero_grad(); loss.backward(); opt.step()
    return wm


def test_bc_actor_recovers_the_demonstrated_action():
    torch.manual_seed(0)
    cfg = _cfg()
    buf = _constant_action_buffer(cfg, const=[0.3, -0.6])
    wm = _trained_wm(cfg, buf)

    actor, bc_loss = bc_actor(cfg, wm, buf, steps=300, lr=3e-3)
    assert bc_loss < 0.05, f"BC did not fit the constant action: loss={bc_loss:.4f}"

    # The cloned actor should output ~[0.3, -0.6] on real states.
    obs = torch.as_tensor(buf.sample(cfg.batch_size)["obs"])
    B, T = obs.shape[:2]
    with torch.no_grad():
        emb = wm.encoder(obs.reshape(B * T, cfg.state_dim)).reshape(B, T, -1)
        feat = wm.rssm.observe(emb, torch.zeros(B, T, cfg.action_dim),
                               wm.rssm.initial_state(B, obs.device))["feat"]
        act, _ = actor(feat.reshape(B * T, -1), deterministic=True)
    assert abs(act[:, 0].mean() - 0.3) < 0.15 and abs(act[:, 1].mean() + 0.6) < 0.15


def test_eval_critic_runs_and_outputs_finite_values():
    torch.manual_seed(0)
    cfg = _cfg()
    buf = _constant_action_buffer(cfg, const=[0.0, 1.0])
    wm = _trained_wm(cfg, buf)

    critic, critic_loss = eval_critic(cfg, wm, buf, steps=200, lr=3e-3)
    assert np.isfinite(critic_loss)
    v = critic(torch.randn(4, cfg.deter_dim + cfg.stoch_dim))
    assert v.shape == (4,) and torch.all(torch.isfinite(v))

"""World-model ELBO tests (spec §7: test_world_model.py).

  - assemble_loss returns a scalar loss + metrics, and gradients flow to all submodules.
  - overfitting one batch drops the loss sharply (the model can fit dynamics).
  - posterior-collapse guard: while reconstruction improves, the KL stays off the floor
    (the latent keeps carrying information -- it doesn't collapse to the prior).
"""
import numpy as np
import pytest
import torch

from config import get_config
from envs.base import make_env
from data.replay_buffer import SequenceReplayBuffer
from models.world_model import WorldModel


def _small_cfg(**overrides):
    defaults = dict(
        obs_type="state", env="dummy", state_dim=35,
        deter_dim=32, stoch_dim=8, hidden_dim=32,
        seq_len=15, free_bits=1.0, kl_scale=1.0,
        max_episode_steps=200,
    )
    defaults.update(overrides)
    return get_config(**defaults)


def _batch_pool(cfg, episodes_steps=2000, seed=0):
    """A filled replay buffer to draw many fresh windows from (avoids single-batch memorization)."""
    np.random.seed(seed)
    env = make_env(cfg)
    buf = SequenceReplayBuffer(cfg.buffer_capacity, cfg.seq_len)
    obs = env.reset()
    for _ in range(episodes_steps):
        a = np.random.uniform(-1, 1, cfg.action_dim).astype(np.float32)
        nxt, r, done, _ = env.step(a)
        buf.add(obs, a, r, done)
        obs = env.reset() if done else nxt
    return buf


def _sample_pool(buf, B, seq_len):
    raw = buf.sample(B)
    return {k: torch.as_tensor(v) for k, v in raw.items()}


def _batch(cfg, B, steps=1200, seed=0):
    return _sample_pool(_batch_pool(cfg, episodes_steps=steps, seed=seed), B, cfg.seq_len)


def test_assemble_loss_runs_and_backprops():
    torch.manual_seed(0)
    cfg = _small_cfg()
    wm = WorldModel(cfg, cfg.action_dim)
    batch = _batch(cfg, B=8)

    loss, metrics = wm.assemble_loss(batch)

    assert loss.ndim == 0 and torch.isfinite(loss)
    for k in ("recon", "reward", "cont", "kl", "kl_loss"):
        assert k in metrics, f"missing metric {k}"

    loss.backward()
    grads = [p.grad for p in wm.parameters() if p.requires_grad]
    assert any(g is not None and torch.any(g != 0) for g in grads), "no gradient reached params"


@pytest.mark.slow
def test_overfits_one_batch_without_posterior_collapse():
    """Milestone gate (spec §9, §10.2): overfitting a single batch must drive recon AND
    reward losses down sharply, while the KL stays off the floor (no posterior collapse).

    Reward in DummyEnv = throttle - |steer|, a function of the ACTION. So the reward head
    can only fit it if the feature it reads from actually contains that action -- this test
    is what catches an action-timing misalignment (spec §11's #1 risk)."""
    torch.manual_seed(0)
    cfg = _small_cfg()
    wm = WorldModel(cfg, cfg.action_dim)
    batch = _batch(cfg, B=8)
    opt = torch.optim.Adam(wm.parameters(), lr=3e-3)

    _, m0 = wm.assemble_loss(batch)
    init = {k: float(v) for k, v in m0.items()}

    for _ in range(500):
        loss, m = wm.assemble_loss(batch)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(wm.parameters(), 100.0)
        opt.step()
    final = {k: float(v) for k, v in m.items()}

    print("init :", {k: round(v, 4) for k, v in init.items()})
    print("final:", {k: round(v, 4) for k, v in final.items()})

    # Loss + reconstruction drop sharply (the model fits the batch).
    # NOTE: reward dropping here is NOT meaningful -- with 34 random noise dims, overfitting
    # one fixed batch lets the posterior memorize each step's reward via the noise as a
    # fingerprint. Whether reward is learned as a FUNCTION OF THE ACTION is tested by
    # test_reward_prediction_generalizes (the test that actually catches action-timing bugs).
    assert final["loss"] < init["loss"] * 0.6, (init["loss"], final["loss"])
    assert final["recon"] < init["recon"] * 0.5, (init["recon"], final["recon"])
    # posterior NOT collapsed: latent keeps carrying information.
    assert final["kl"] > 0.1, f"posterior collapsed: KL={final['kl']:.4f}"


@pytest.mark.slow
def test_reward_prediction_generalizes():
    """The real action-timing test (spec §11's #1 risk). Train on one data stream, then
    measure reward error on a DISJOINT held-out stream where memorization can't help.

    Reward = throttle - |steer| is a function of the action. The reward head reads `feat`;
    `feat_t` carries the action via the recurrence h_t = GRU([z_{t-1}, a_{t-1}], h_{t-1}).
    So the head can only generalize if the reward target is aligned to the action that
    actually lives in the feature. Misalign it and held-out reward MSE stays at the reward
    variance (~0.42) -- the head can do no better than predicting the mean reward."""
    torch.manual_seed(0)
    cfg = _small_cfg(seq_len=10)
    wm = WorldModel(cfg, cfg.action_dim)
    opt = torch.optim.Adam(wm.parameters(), lr=3e-3)

    train = _batch_pool(cfg, episodes_steps=2000, seed=0)
    for _ in range(300):
        batch = _sample_pool(train, cfg.batch_size, cfg.seq_len)
        loss, _ = wm.assemble_loss(batch)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(wm.parameters(), 100.0)
        opt.step()

    # Disjoint held-out stream (different seed -> unseen noise + actions).
    held_out = _batch(cfg, B=16, steps=1200, seed=12345)
    with torch.no_grad():
        _, m = wm.assemble_loss(held_out)
    print("held-out reward MSE:", round(float(m["reward"]), 4))

    assert float(m["reward"]) < 0.1, (
        f"reward does not generalize (held-out MSE={float(m['reward']):.4f} ~ reward "
        f"variance): the reward target is misaligned with the action in the feature"
    )

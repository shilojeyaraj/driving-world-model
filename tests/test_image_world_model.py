"""Image-mode world model (V1): the full pixel pipeline (CNN encoder -> RSSM -> transposed-CNN
decoder -> ELBO) runs and trains on rendered frames from the visual DummyEnv.
"""
import numpy as np
import pytest
import torch

from config import get_config
from envs.base import make_env
from data.replay_buffer import SequenceReplayBuffer
from models.world_model import WorldModel


def _cfg(**ov):
    d = dict(obs_type="image", env="dummy", image_size=16, deter_dim=64, stoch_dim=16,
             hidden_dim=64, seq_len=12, free_bits=1.0, kl_scale=1.0, max_episode_steps=100,
             batch_size=8)
    d.update(ov)
    return get_config(**d)


def _batch(cfg, B, steps=800, seed=0):
    np.random.seed(seed)
    env = make_env(cfg)
    buf = SequenceReplayBuffer(cfg.buffer_capacity, cfg.seq_len)
    obs = env.reset()
    for _ in range(steps):
        a = np.random.uniform(-1, 1, cfg.action_dim).astype(np.float32)
        nxt, r, done, _ = env.step(a)
        buf.add(obs, a, r, done)
        obs = env.reset() if done else nxt
    return {k: torch.as_tensor(v) for k, v in buf.sample(B).items()}


def test_image_assemble_loss_runs_and_backprops():
    torch.manual_seed(0)
    cfg = _cfg()
    wm = WorldModel(cfg, cfg.action_dim)
    batch = _batch(cfg, B=8)
    assert batch["obs"].shape[2:] == (3, 16, 16)

    loss, metrics = wm.assemble_loss(batch)
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and torch.any(p.grad != 0) for p in wm.parameters())


@pytest.mark.slow
def test_image_world_model_recon_drops():
    """Train briefly on rendered frames; pixel reconstruction must drop sharply (the blob is
    fully predictable from the latent via the posterior)."""
    torch.manual_seed(0)
    cfg = _cfg()
    wm = WorldModel(cfg, cfg.action_dim)
    opt = torch.optim.Adam(wm.parameters(), lr=3e-3)
    batch = _batch(cfg, B=8)

    init = float(wm.assemble_loss(batch)[1]["recon"])
    for _ in range(200):
        loss, m = wm.assemble_loss(batch)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(wm.parameters(), 100.0)
        opt.step()
    final = float(m["recon"])
    print(f"\nimage recon: {init:.3f} -> {final:.3f}")
    assert final < init * 0.3, f"pixel recon did not drop: {init:.3f} -> {final:.3f}"

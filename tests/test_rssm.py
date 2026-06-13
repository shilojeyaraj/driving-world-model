"""RSSM contracts (spec §7: test_rssm.py).

The RSSM is the core sequence model. These tests pin:
  - `observe` (posterior path) returns per-step prior/posterior stats + features at the
    right shapes, following the previous-action timing convention (spec §3).
  - `imagine` (prior path) returns per-step features and is reproducible given a fixed seed.
"""
import torch

from config import get_config
from models.rssm import RSSM


def _rssm(embed_dim=16, action_dim=2, **overrides):
    cfg = get_config(deter_dim=32, stoch_dim=8, hidden_dim=24, min_std=0.1, **overrides)
    return RSSM(cfg, embed_dim, action_dim), cfg


def test_observe_shapes():
    rssm, cfg = _rssm()
    B, T, E, A = 4, 6, 16, 2

    embeds = torch.randn(B, T, E)
    actions = torch.randn(B, T, A)
    state = rssm.initial_state(B, torch.device("cpu"))

    out = rssm.observe(embeds, actions, state)

    feat_dim = cfg.deter_dim + cfg.stoch_dim
    assert out["feat"].shape == (B, T, feat_dim)
    for stats in (out["prior"], out["post"]):
        assert stats["mean"].shape == (B, T, cfg.stoch_dim)
        assert stats["std"].shape == (B, T, cfg.stoch_dim)
        assert torch.all(stats["std"] > 0)            # std must be strictly positive
    h, z = out["state"]
    assert h.shape == (B, cfg.deter_dim)
    assert z.shape == (B, cfg.stoch_dim)


def test_initial_state_is_zeros():
    rssm, cfg = _rssm()
    h, z = rssm.initial_state(3, torch.device("cpu"))
    assert h.shape == (3, cfg.deter_dim) and z.shape == (3, cfg.stoch_dim)
    assert torch.count_nonzero(h) == 0 and torch.count_nonzero(z) == 0


def test_imagine_shapes():
    rssm, cfg = _rssm()
    B, H, A = 4, 7, 2
    actions = torch.randn(B, H, A)
    state = rssm.initial_state(B, torch.device("cpu"))

    out = rssm.imagine(actions, state)

    feat_dim = cfg.deter_dim + cfg.stoch_dim
    assert out["feat"].shape == (B, H, feat_dim)
    assert out["prior"]["mean"].shape == (B, H, cfg.stoch_dim)
    assert out["prior"]["std"].shape == (B, H, cfg.stoch_dim)


def test_imagine_reproducible_given_seed():
    """imagine samples z from the prior, so it's stochastic -- but with the same seed,
    same start state and same actions it must reproduce exactly (sane sampling, no hidden
    nondeterminism)."""
    rssm, cfg = _rssm()
    B, H, A = 2, 5, 2
    actions = torch.randn(B, H, A)
    state = rssm.initial_state(B, torch.device("cpu"))

    torch.manual_seed(123)
    a = rssm.imagine(actions, state)["feat"]
    torch.manual_seed(123)
    b = rssm.imagine(actions, state)["feat"]
    assert torch.equal(a, b)

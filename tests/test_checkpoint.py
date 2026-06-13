"""Checkpoint round-trip (spec §6): save {world_model, actor, critic, config} and reload
into freshly-built models that produce identical outputs."""
import torch

from config import get_config
from models.world_model import WorldModel
from models.actor_critic import Actor, Critic
from utils import save_checkpoint, load_models


def _cfg(**ov):
    d = dict(obs_type="state", state_dim=35, deter_dim=32, stoch_dim=8, hidden_dim=32,
             action_dim=2)
    d.update(ov)
    return get_config(**d)


def test_checkpoint_roundtrip(tmp_path):
    torch.manual_seed(0)
    cfg = _cfg()
    feat_dim = cfg.deter_dim + cfg.stoch_dim
    wm = WorldModel(cfg, cfg.action_dim)
    actor = Actor(cfg, feat_dim, cfg.action_dim)
    critic = Critic(cfg, feat_dim)

    path = str(tmp_path / "ckpt.pt")
    save_checkpoint(path, wm, actor, critic, cfg)

    cfg2, wm2, actor2, critic2 = load_models(path)

    # config round-trips
    assert cfg2.deter_dim == cfg.deter_dim and cfg2.stoch_dim == cfg.stoch_dim
    assert cfg2.action_dim == cfg.action_dim

    # reloaded actor produces identical output (weights match exactly)
    feat = torch.randn(4, feat_dim)
    a1, _ = actor(feat, deterministic=True)
    a2, _ = actor2(feat, deterministic=True)
    assert torch.equal(a1, a2)
    assert torch.equal(critic(feat), critic2(feat))

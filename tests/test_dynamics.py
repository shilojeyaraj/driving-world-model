"""Dynamics-ablation seam (spec §13): the deterministic recurrence is a swappable module
selected by cfg.dynamics, behind one interface. v1 ships the GRU ('rssm') and a minimal
Mamba-style selective SSM ('mamba'); both satisfy the same observe/imagine contract.
"""
import pytest
import torch

from config import get_config
from models.rssm import RSSM
from models.recurrence import make_recurrence, GRURecurrence


IMPLEMENTED = ["rssm", "mamba"]


def test_default_dynamics_builds_gru_recurrence():
    cfg = get_config(dynamics="rssm", deter_dim=32, stoch_dim=8, hidden_dim=24)
    rssm = RSSM(cfg, embed_dim=16, action_dim=2)
    assert isinstance(rssm.recurrence, GRURecurrence)
    assert rssm.recurrence.deter_dim == cfg.deter_dim


@pytest.mark.parametrize("dyn", IMPLEMENTED)
def test_recurrence_step_shapes(dyn):
    cfg = get_config(dynamics=dyn, deter_dim=32, stoch_dim=8, hidden_dim=24)
    rec = make_recurrence(cfg, action_dim=2)
    B = 5
    h = rec.initial_state(B, torch.device("cpu"))
    assert h.shape == (B, cfg.deter_dim)
    h2 = rec(h, torch.randn(B, cfg.stoch_dim), torch.randn(B, 2))
    assert h2.shape == (B, cfg.deter_dim)


@pytest.mark.parametrize("dyn", IMPLEMENTED)
def test_rssm_observe_imagine_shapes_for_each_dynamics(dyn):
    """Both recurrences satisfy the same observe/imagine contract behind the seam."""
    cfg = get_config(dynamics=dyn, deter_dim=32, stoch_dim=8, hidden_dim=24)
    rssm = RSSM(cfg, embed_dim=16, action_dim=2)
    B, T = 4, 6
    state = rssm.initial_state(B, torch.device("cpu"))

    obs_out = rssm.observe(torch.randn(B, T, 16), torch.randn(B, T, 2), state)
    feat_dim = cfg.deter_dim + cfg.stoch_dim
    assert obs_out["feat"].shape == (B, T, feat_dim)

    img_out = rssm.imagine(torch.randn(B, T, 2), state)
    assert img_out["feat"].shape == (B, T, feat_dim)


def test_transformer_dynamics_not_yet_implemented():
    cfg = get_config(dynamics="transformer", deter_dim=32, stoch_dim=8, hidden_dim=24)
    with pytest.raises(NotImplementedError):
        RSSM(cfg, embed_dim=16, action_dim=2)

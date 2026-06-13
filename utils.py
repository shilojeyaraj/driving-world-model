"""Checkpointing (spec §6). Save/load {world_model, actor, critic, config} so a trained run
can be reloaded for closed-loop eval.

OK TO USE A LIBRARY / modify freely.
"""
import os
from dataclasses import asdict

import torch

from config import get_config


def save_checkpoint(path, world_model, actor=None, critic=None, config=None):
    """Save state_dicts + the config (as a plain dict) to `path` (e.g. runs/<name>/ckpt.pt)."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    torch.save({
        "world_model": world_model.state_dict(),
        "actor": None if actor is None else actor.state_dict(),
        "critic": None if critic is None else critic.state_dict(),
        "config": None if config is None else asdict(config),
    }, path)


def load_models(path, map_location="cpu"):
    """Rebuild and load (config, world_model, actor, critic) from a checkpoint. actor/critic
    are None if they weren't saved. Models are built from the saved config so dims match."""
    # Lazy imports avoid a circular import (models import nothing from utils, but keep it clean).
    from models.world_model import WorldModel
    from models.actor_critic import Actor, Critic

    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    cfg = get_config(**ckpt["config"])
    feat_dim = cfg.deter_dim + cfg.stoch_dim

    world_model = WorldModel(cfg, cfg.action_dim)
    world_model.load_state_dict(ckpt["world_model"])

    actor = critic = None
    if ckpt.get("actor") is not None:
        actor = Actor(cfg, feat_dim, cfg.action_dim)
        actor.load_state_dict(ckpt["actor"])
    if ckpt.get("critic") is not None:
        critic = Critic(cfg, feat_dim)
        critic.load_state_dict(ckpt["critic"])

    return cfg, world_model, actor, critic

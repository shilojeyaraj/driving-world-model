"""Drive-from-pixels CONTRACT (V1, step ①): the closed-loop eval runs end-to-end in image
mode (CNN encode -> posterior -> actor -> env.step). This is the FAST guard that the pixel
control path is wired; the actual "trained policy drives from pixels" demonstration is heavy
(image training) and lives in scripts/drive_from_pixels.py + experiments/008 instead, so the
test suite stays minutes, not hours.
"""
import torch

from config import get_config
from envs.base import make_env
from models.world_model import WorldModel
from models.actor_critic import Actor
from eval.closed_loop import closed_loop_eval


def test_closed_loop_runs_in_image_mode():
    torch.manual_seed(0)
    cfg = get_config(obs_type="image", env="dummy", image_size=16, deter_dim=32, stoch_dim=8,
                     hidden_dim=32, max_episode_steps=8)
    env = make_env(cfg)
    wm = WorldModel(cfg, cfg.action_dim)
    actor = Actor(cfg, cfg.deter_dim + cfg.stoch_dim, cfg.action_dim)

    out = closed_loop_eval(actor, wm, env, episodes=2, max_steps=8)

    for k in ("actor_return", "random_return", "actor_throttle", "actor_steer"):
        assert k in out and isinstance(out[k], float)

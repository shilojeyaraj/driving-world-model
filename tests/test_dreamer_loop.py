"""Iterated Dreamer loop: collect-with-policy + the outer collect->train-WM->train-behavior
loop that grounds the world model in the states the policy actually visits (the fix for the
single-shot model-exploitation failure seen on MetaDrive, experiments/010)."""
import numpy as np
import pytest
import torch

from config import get_config
from envs.base import make_env
from data.replay_buffer import SequenceReplayBuffer
from models.world_model import WorldModel
from models.actor_critic import Actor
from training.dreamer_loop import collect_with_policy, dreamer_train
from eval.closed_loop import closed_loop_eval


def _cfg(**ov):
    d = dict(obs_type="state", env="dummy", state_dim=35, deter_dim=32, stoch_dim=8,
             hidden_dim=32, seq_len=10, action_dim=2, max_episode_steps=20)
    d.update(ov)
    return get_config(**d)


def test_collect_with_policy_fills_buffer_in_bounds():
    torch.manual_seed(0); np.random.seed(0)
    cfg = _cfg()
    env = make_env(cfg)
    wm = WorldModel(cfg, cfg.action_dim)
    actor = Actor(cfg, cfg.deter_dim + cfg.stoch_dim, cfg.action_dim)
    buf = SequenceReplayBuffer(cfg.buffer_capacity, cfg.seq_len)

    collect_with_policy(env, wm, actor, buf, steps=200, explore_std=0.3)

    assert len(buf) == 200                      # one transition added per env step
    assert buf.can_sample()
    b = buf.sample(4)
    assert b["action"].min() >= -1.0 and b["action"].max() <= 1.0   # actions stay in range
    assert b["obs"].shape == (4, cfg.seq_len, cfg.state_dim)


@pytest.mark.slow
def test_dreamer_loop_learns_to_drive_dummy():
    """The full iterated loop should still solve the known-good toy: drive (throttle->1,
    steer->0) and beat random. Validates the loop end-to-end before the (slow) MetaDrive run."""
    torch.manual_seed(0); np.random.seed(0)
    cfg = _cfg(max_episode_steps=100, deter_dim=64, stoch_dim=16, hidden_dim=64, seq_len=16,
               imagine_horizon=10, actor_lr=3e-3, critic_lr=3e-3)

    wm, actor, critic, _ = dreamer_train(cfg, iters=4, seed_steps=800, collect_per_iter=300,
                                         wm_steps=200, behavior_steps=200, explore_std=0.3)

    out = closed_loop_eval(actor, wm, make_env(cfg), episodes=5, max_steps=100)
    print("\ndreamer-loop closed-loop:", {k: round(v, 3) for k, v in out.items()})
    assert out["actor_return"] > out["random_return"], out
    assert out["actor_throttle"] > 0.7, out
    assert abs(out["actor_steer"]) < 0.3, out

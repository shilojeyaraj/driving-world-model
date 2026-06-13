"""Dynamics ablation (spec §13): train the SAME pipeline with each cfg.dynamics and compare
on the same eval harness -- open-loop prediction, held-out reward, closed-loop driving.

This is the payoff of the swappable-recurrence seam: a controlled comparison where ONLY the
deterministic recurrence changes.

Usage:  python -m scripts.ablate_dynamics
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from config import get_config
from envs.base import make_env
from data.replay_buffer import SequenceReplayBuffer
from models.world_model import WorldModel
from models.actor_critic import Actor, Critic
from training.train_behavior import train_behavior_in_imagination
from eval.open_loop import open_loop_eval
from eval.closed_loop import closed_loop_eval


def pool(cfg, steps, seed):
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


def batch(buf, B):
    return {k: torch.as_tensor(v) for k, v in buf.sample(B).items()}


def run_one(dynamics, wm_steps=500, behavior_steps=400, seed=0):
    cfg = get_config(obs_type="state", env="dummy", state_dim=35, deter_dim=64, stoch_dim=16,
                     hidden_dim=64, seq_len=16, free_bits=1.0, kl_scale=1.0, max_episode_steps=200,
                     imagine_horizon=10, gamma=0.99, lambda_=0.95, entropy_coef=1e-3,
                     actor_lr=3e-3, critic_lr=3e-3, batch_size=16, dynamics=dynamics)
    torch.manual_seed(seed)
    train_buf = pool(cfg, 3000, seed=seed)

    # --- world model ---
    wm = WorldModel(cfg, cfg.action_dim)
    opt = torch.optim.Adam(wm.parameters(), lr=3e-3)
    last = {}
    for _ in range(wm_steps):
        loss, last = wm.assemble_loss(batch(train_buf, cfg.batch_size))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(wm.parameters(), 100.0); opt.step()

    # --- open-loop + held-out reward ---
    held = pool(cfg, 2000, seed=seed + 777)
    ol = open_loop_eval(wm, batch(held, 32), context=5, horizon=10)
    with torch.no_grad():
        _, m = wm.assemble_loss(batch(held, 32))
    reward_heldout = float(m["reward"])

    # --- behavior + closed-loop ---
    feat_dim = cfg.deter_dim + cfg.stoch_dim
    actor, critic = Actor(cfg, feat_dim, cfg.action_dim), Critic(cfg, feat_dim)
    train_behavior_in_imagination(cfg, wm, train_buf, actor, critic, steps=behavior_steps, log_every=10_000)
    cl = closed_loop_eval(actor, wm, make_env(cfg), episodes=5, max_steps=100)

    return {
        "wm_recon": float(last["recon"]),
        "wm_kl": float(last["kl"]),
        "reward_heldout": reward_heldout,
        "openloop_model": float(np.sum(ol["model"])),
        "openloop_noact": float(np.sum(ol["no_action"])),
        "closed_return": cl["actor_return"],
        "throttle": cl["actor_throttle"],
        "steer": cl["actor_steer"],
    }


def main():
    cols = ["wm_recon", "wm_kl", "reward_heldout", "openloop_model", "openloop_noact",
            "closed_return", "throttle", "steer"]
    rows = {}
    for dyn in ("rssm", "mamba"):
        print(f"=== training dynamics={dyn} ===")
        rows[dyn] = run_one(dyn)
        print(f"   {dyn}: {{ {', '.join(f'{c}={rows[dyn][c]:.3f}' for c in cols)} }}")

    print("\n" + "=" * 92)
    print(f"{'metric':<16} {'rssm (GRU)':>14} {'mamba (SSM)':>14}    note")
    notes = {
        "wm_recon": "lower=better (noise floor ~0.34)",
        "wm_kl": "healthy if >0",
        "reward_heldout": "lower=better (<0.1 = learned)",
        "openloop_model": "lower=better",
        "openloop_noact": "baseline (model should be < this)",
        "closed_return": "higher=better (random ~ -50)",
        "throttle": "want ~ +1.0",
        "steer": "want ~ 0.0",
    }
    for c in cols:
        print(f"{c:<16} {rows['rssm'][c]:>14.3f} {rows['mamba'][c]:>14.3f}    {notes[c]}")


if __name__ == "__main__":
    main()

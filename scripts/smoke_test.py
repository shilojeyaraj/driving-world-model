"""Smoke test: confirms env + replay buffer + shapes work end-to-end, with NO GPU and NO
models implemented yet. Run this FIRST:   python scripts/smoke_test.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from config import get_config
from envs.base import make_env
from data.replay_buffer import SequenceReplayBuffer


def main():
    for obs_type in ("state", "image"):
        cfg = get_config(env="dummy", obs_type=obs_type, seq_len=10, max_episode_steps=40)
        env = make_env(cfg)
        buf = SequenceReplayBuffer(cfg.buffer_capacity, cfg.seq_len)
        obs = env.reset()
        for _ in range(500):
            a = np.random.uniform(-1, 1, cfg.action_dim).astype(np.float32)
            nxt, r, done, _ = env.step(a)
            buf.add(obs, a, r, done)
            obs = env.reset() if done else nxt
        b = buf.sample(4)
        print(f"[{obs_type}] obs batch {b['obs'].shape}  action {b['action'].shape}  "
              f"reward {b['reward'].shape}")
        assert b["obs"].shape[0] == 4 and b["obs"].shape[1] == cfg.seq_len
    print("smoke test passed -- plumbing works. Now implement the model stubs (see CONCEPTS.md).")


if __name__ == "__main__":
    main()

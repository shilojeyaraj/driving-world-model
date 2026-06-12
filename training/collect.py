"""Collect trajectories into the replay buffer. Runs against the dummy env with a random
policy out of the box (no GPU, no model needed) so you can verify the data path first.

OK TO USE A LIBRARY / modify freely.   Run:  python -m training.collect
"""
import numpy as np

from config import get_config
from envs.base import make_env
from data.replay_buffer import SequenceReplayBuffer


def random_policy(action_dim):
    return np.random.uniform(-1, 1, size=action_dim).astype(np.float32)


def collect(cfg, num_steps=5000):
    env = make_env(cfg)
    buf = SequenceReplayBuffer(cfg.buffer_capacity, cfg.seq_len)
    obs = env.reset()
    for _ in range(num_steps):
        action = random_policy(cfg.action_dim)     # TODO: swap in your trained actor later
        nxt, reward, done, _ = env.step(action)
        buf.add(obs, action, reward, done)
        obs = env.reset() if done else nxt
    print(f"collected {len(buf)} steps across {len(buf._episodes)} usable episodes")
    return buf


if __name__ == "__main__":
    collect(get_config(env="dummy", obs_type="state", max_episode_steps=200))

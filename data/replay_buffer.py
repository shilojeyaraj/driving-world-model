"""Sequence replay buffer for world-model training. Stores whole episodes and samples
contiguous length-L sequences (world models learn *dynamics*, so order matters).

OK TO USE A LIBRARY -- but it's short, and reading it teaches the sequence-sampling point.
See CONCEPTS.md: "Why sample contiguous sequences, not i.i.d. transitions?"
"""
import numpy as np


def _new_episode():
    return {"obs": [], "action": [], "reward": [], "done": []}


class SequenceReplayBuffer:
    def __init__(self, capacity, seq_len):
        self.capacity = capacity
        self.seq_len = seq_len
        self._episodes = []
        self._current = _new_episode()
        self._size = 0

    def add(self, obs, action, reward, done):
        self._current["obs"].append(np.asarray(obs, dtype=np.float32))
        self._current["action"].append(np.asarray(action, dtype=np.float32))
        self._current["reward"].append(np.float32(reward))
        self._current["done"].append(np.float32(done))
        self._size += 1
        if done:
            self._flush()
        while self._size > self.capacity and self._episodes:
            self._size -= len(self._episodes.pop(0)["reward"])

    def _flush(self):
        if len(self._current["reward"]) >= self.seq_len:
            self._episodes.append({k: np.asarray(v) for k, v in self._current.items()})
        self._current = _new_episode()

    def __len__(self):
        return self._size

    def can_sample(self):
        return len(self._episodes) > 0

    def sample(self, batch_size):
        """Returns a dict of arrays shaped (batch, seq_len, ...)."""
        batch = {k: [] for k in ("obs", "action", "reward", "done")}
        for _ in range(batch_size):
            ep = self._episodes[np.random.randint(len(self._episodes))]
            start = np.random.randint(0, len(ep["reward"]) - self.seq_len + 1)
            sl = slice(start, start + self.seq_len)
            for k in batch:
                batch[k].append(ep[k][sl])
        return {k: np.stack(v) for k, v in batch.items()}

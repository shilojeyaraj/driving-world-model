"""buffer_from_session: turn a flat recorded driving session (e.g. gesture-driven MetaDrive) into
a trainable SequenceReplayBuffer."""
import numpy as np

from data.replay_buffer import buffer_from_session


def test_splits_on_done_and_keeps_trailing_run():
    N, D = 60, 35
    obs = np.random.randn(N, D).astype(np.float32)
    action = np.random.uniform(-1, 1, (N, 2)).astype(np.float32)
    reward = np.random.randn(N).astype(np.float32)
    done = np.zeros(N, np.float32); done[29] = 1.0      # one boundary; trailing 30 has no done

    buf = buffer_from_session(obs, action, reward, done, capacity=100_000, seq_len=10)

    assert buf.can_sample()
    assert len(buf._episodes) == 2                       # [0:30] (done) + trailing [30:60] (flushed)
    b = buf.sample(4)
    assert b["obs"].shape == (4, 10, D) and b["action"].shape == (4, 10, 2)


def test_drops_episodes_shorter_than_seq_len():
    obs = np.random.randn(12, 4).astype(np.float32)
    action = np.zeros((12, 2), np.float32)
    reward = np.zeros(12, np.float32)
    done = np.zeros(12, np.float32); done[4] = 1.0       # episode of 5 (< seq_len) is dropped
    buf = buffer_from_session(obs, action, reward, done, seq_len=10)
    assert len(buf._episodes) == 0                       # 5-step ep dropped; trailing 7-step ep dropped

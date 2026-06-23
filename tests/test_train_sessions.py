"""Accumulating gesture sessions: train_on_gesture can train on MANY recorded drives, not just one.
concat_sessions is PURE; load_sessions is tested against real tmp .npz files (no mocks)."""
import numpy as np
import pytest

from scripts.train_on_gesture import concat_sessions, load_sessions


def _session(n, obs_dim=4, seed=0):
    rng = np.random.RandomState(seed)
    return {"obs": rng.randn(n, obs_dim).astype(np.float32),
            "action": rng.randn(n, 2).astype(np.float32),
            "reward": rng.randn(n).astype(np.float32),
            "done": np.zeros(n, np.float32)}              # no terminations within the drive


def test_concat_sessions_marks_each_file_boundary_done():
    out = concat_sessions([_session(5), _session(3, seed=1)])
    assert out["obs"].shape == (8, 4) and out["action"].shape == (8, 2)
    # each file's LAST step is forced done so episodes never bleed across separate drives:
    assert out["done"][4] == 1.0      # end of session A
    assert out["done"][7] == 1.0      # end of session B
    assert out["done"][:4].sum() == 0 # nothing spurious mid-session


def test_concat_sessions_keeps_internal_dones():
    a = _session(5); a["done"][2] = 1.0
    out = concat_sessions([a])
    assert out["done"][2] == 1.0 and out["done"][4] == 1.0   # real crash + forced boundary


def test_load_sessions_globs_directory_and_explicit_files(tmp_path):
    p1, p2 = tmp_path / "s1.npz", tmp_path / "s2.npz"
    np.savez(p1, **_session(5)); np.savez(p2, **_session(4, seed=2))
    data_dir, files_dir = load_sessions([str(tmp_path)])           # by directory
    assert data_dir["obs"].shape[0] == 9 and len(files_dir) == 2
    data_files, files = load_sessions([str(p1), str(p2)])          # by explicit files
    assert data_files["obs"].shape[0] == 9 and len(files) == 2


def test_load_sessions_errors_when_none_found(tmp_path):
    with pytest.raises(SystemExit):
        load_sessions([str(tmp_path / "does_not_exist.npz")])

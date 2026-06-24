"""DAgger tests. The live rollout (drive the learner, relabel with IDM) needs MetaDrive, so the
PURE pieces -- the dataset-aggregation boundary logic and the CLI -- are unit-tested here without
the sim, and the live crux gets an importorskip smoke. Aggregation correctness is the load-bearing
property: DAgger pours many rollouts into ONE growing buffer, and a rollout must never bleed into
the next (that would train the world model on a transition that never happened).
"""
import numpy as np
import pytest

from data.replay_buffer import SequenceReplayBuffer
from training.dagger import extend_buffer
from scripts.dagger import parse_args, build_cfg


def test_extend_buffer_keeps_rollouts_as_separate_episodes():
    """Two rollouts added to one buffer stay two episodes (the boundary is flushed, no bleed)."""
    buf = SequenceReplayBuffer(capacity=1000, seq_len=4)
    traj = ([np.zeros(3, np.float32)] * 6, [np.zeros(2, np.float32)] * 6, [0.1] * 6, [False] * 6)
    extend_buffer(buf, *traj)
    assert len(buf._episodes) == 1 and len(buf) == 6
    extend_buffer(buf, *traj)
    assert len(buf._episodes) == 2, "second rollout bled into the first instead of a new episode"
    assert len(buf) == 12


def test_extend_buffer_respects_natural_dones_within_a_rollout():
    """A crash/off-road mid-rollout (done=True) splits the episode; the trailing run is flushed too."""
    buf = SequenceReplayBuffer(capacity=1000, seq_len=2)
    dones = [False, False, True, False, False]            # one natural episode end at index 2
    extend_buffer(buf, [np.zeros(3, np.float32)] * 5, [np.zeros(2, np.float32)] * 5, [0.0] * 5, dones)
    assert len(buf._episodes) == 2                        # [0..2] and [3..4]


def test_parse_args_exposes_dagger_knobs():
    a = parse_args(["--iters", "5", "--rollout-steps", "3000", "--collect", "5000",
                    "--wm-steps", "800", "--bc-steps", "800", "--num-scenarios", "30",
                    "--eval-episodes", "0"])
    assert a.iters == 5
    assert a.rollout_steps == 3000
    assert a.collect_steps == 5000
    assert a.wm_steps == 800
    assert a.bc_steps == 800
    assert a.num_scenarios == 30
    assert a.eval_episodes == 0


def test_build_cfg_sets_state_mode_train_pool_and_real_terminations():
    cfg = build_cfg(num_scenarios=30)
    assert cfg.env == "metadrive" and cfg.obs_type == "state"
    assert cfg.metadrive_num_scenarios == 30
    assert cfg.metadrive_start_seed == 0
    assert cfg.metadrive_endless is False                 # DAgger needs REAL terminations (drift -> reset)
    assert cfg.metadrive_map == 3                         # varied geometry per seed


def test_idm_relabel_rollout_smoke():
    """Live (needs MetaDrive): drive the learner, relabel each visited state with IDM. Verifies the
    true-DAgger crux end to end -- a usable buffer with the right shapes."""
    pytest.importorskip("metadrive")
    from models.world_model import WorldModel
    from models.actor_critic import Actor
    from training.dagger import idm_relabel_rollout
    cfg = build_cfg(num_scenarios=2)
    wm = WorldModel(cfg, cfg.action_dim)
    actor = Actor(cfg, cfg.deter_dim + cfg.stoch_dim, cfg.action_dim)
    buf = idm_relabel_rollout(cfg, wm, actor, steps=40, seed=0)
    assert len(buf) == 40
    if buf.can_sample():
        batch = buf.sample(2)
        assert batch["obs"].shape[1] == cfg.seq_len
        assert batch["action"].shape[-1] == cfg.action_dim

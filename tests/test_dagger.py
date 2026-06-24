"""DAgger tests. The live rollout (drive the learner, relabel with IDM) needs MetaDrive, so the
PURE pieces -- the dataset-aggregation boundary logic and the CLI -- are unit-tested here without
the sim, and the live crux gets an importorskip smoke. Aggregation correctness is the load-bearing
property: DAgger pours many rollouts into ONE growing buffer, and a rollout must never bleed into
the next (that would train the world model on a transition that never happened).
"""
import numpy as np
import pytest

from data.replay_buffer import SequenceReplayBuffer
from training.dagger import extend_buffer, relabel_action, dagger_capacity
from scripts.dagger import parse_args, build_cfg


def test_relabel_action_sanitizes_nonfinite():
    """clip alone leaves NaN as NaN (np.clip(nan, -1, 1) == nan) -> NaN loss -> SILENT collapse,
    even harder to debug than the -198 case. IDM can emit NaN/inf in degenerate states (speed~0,
    no lead vehicle). Relabels must be FINITE and in range: NaN -> 0 (neutral), +-inf -> +-1."""
    out = relabel_action([np.nan, np.inf])
    assert np.all(np.isfinite(out)), "NaN/inf leaked into a BC target"
    assert out.tolist() == [0.0, 1.0]
    assert relabel_action([-np.inf, np.nan]).tolist() == [-1.0, 0.0]


def test_dagger_capacity_holds_all_planned_data():
    """The buffer evicts OLDEST episodes first -- the iter-0 IDM expert base. Capacity must hold all
    planned data so a long run never discards the clean demonstrations."""
    assert dagger_capacity(collect_steps=4000, iters=5, rollout_steps=2000) >= 4000 + 5 * 2000


def test_relabel_action_clips_idm_emergency_brake_to_action_range():
    """REGRESSION: at near-collision states the learner drives into, MetaDrive's IDM returns a huge
    UN-normalized emergency-brake acceleration (observed throttle ~ -198). The agent's action space
    is [-1,1] and env.step clips to it, so the stored BC target MUST be clipped -- an unreachable
    target (the tanh-bounded actor maxes at +-1) blew bc_loss up to ~12M and collapsed the policy."""
    assert relabel_action([-0.197, -198.647]).tolist() == [pytest.approx(-0.197, abs=1e-3), -1.0]
    assert relabel_action([2.0, -3.0]).tolist() == [1.0, -1.0]
    out = relabel_action([0.3, 0.8])                          # already in range -> unchanged
    assert out.tolist() == [pytest.approx(0.3, abs=1e-3), pytest.approx(0.8, abs=1e-3)]
    assert out.dtype == np.float32


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
        assert np.abs(batch["action"]).max() <= 1.0, "relabel targets must be clipped to the action range"

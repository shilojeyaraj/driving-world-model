"""Direct-policy DAgger tests. The live rollout needs MetaDrive, so pure pieces are unit-tested
without the sim, and the live crux gets a smoke test. The key correctness property: rollout data
from the policy's failure states must accumulate into the training dataset without episode bleed.
"""
import numpy as np
import pytest

from training.direct_bc import DirectPolicy
from training.dagger import relabel_action
from scripts.direct_dagger import parse_args


def test_parse_args_defaults():
    a = parse_args([])
    assert a.iters == 3
    assert a.rollout_steps == 2000
    assert a.clean_steps == 8000
    assert a.recovery_steps == 8000
    assert a.eval_episodes == 5
    assert a.boost_scene is None
    assert a.out == "runs/direct_dagger/policy.pt"


def test_parse_args_all_knobs():
    a = parse_args([
        "--iters", "5", "--rollout-steps", "3000",
        "--clean", "4000", "--recovery", "4000",
        "--bc-steps", "2000", "--num-scenarios", "20",
        "--boost-scene", "O", "--boost-steps", "2000",
        "--eval-episodes", "0", "--out", "runs/test/p.pt",
    ])
    assert a.iters == 5
    assert a.rollout_steps == 3000
    assert a.clean_steps == 4000
    assert a.recovery_steps == 4000
    assert a.direct_steps == 2000
    assert a.num_scenarios == 20
    assert a.boost_scene == "O"
    assert a.boost_steps == 2000
    assert a.eval_episodes == 0
    assert a.out == "runs/test/p.pt"


def test_direct_dagger_rollout_smoke():
    """Live (needs MetaDrive): policy drives, IDM relabels. Verifies the direct-DAgger crux:
    the returned buffer has the right shapes and all actions are finite and in [-1,1]."""
    pytest.importorskip("metadrive")
    from scripts.dagger import build_cfg
    from training.direct_dagger import direct_dagger_rollout

    cfg = build_cfg(num_scenarios=2)
    policy = DirectPolicy(cfg.state_dim, cfg.action_dim)
    buf = direct_dagger_rollout(cfg, policy, steps=40, seed=0)

    assert len(buf) == 40
    if buf.can_sample():
        batch = buf.sample(2)
        assert batch["obs"].shape[-1] == cfg.state_dim
        assert batch["action"].shape[-1] == cfg.action_dim
        assert np.all(np.isfinite(batch["action"])), "relabeled actions must be finite"
        assert np.abs(batch["action"]).max() <= 1.0, "relabeled actions must be in [-1,1]"

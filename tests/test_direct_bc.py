"""Direct obs->action BC ablation: a plain MLP cloned straight from the 259-dim state vector, with NO
world model in the loop. The diagnostic for the latent-cloning hypothesis -- if this lane-keeps where
the WM-latent actor goes off-road 100%, the under-trained WM latent is the bottleneck. Pure pieces
(the policy + its trainer) are unit-tested here; the closed-loop comparison is the live ablation script.
"""
import numpy as np
import torch

from training.direct_bc import DirectPolicy, train_direct_bc, save_direct, load_direct


def test_train_direct_policy_cli_knobs():
    from scripts.train_direct_policy import parse_args
    a = parse_args(["--clean", "16000", "--recovery", "12000", "--perturb-prob", "0.1", "--gamma", "0.25"])
    assert a.clean_steps == 16000 and a.recovery_steps == 12000
    assert a.perturb_prob == 0.1 and a.gamma == 0.25
    b = parse_args([])
    assert b.clean_steps == 8000 and b.perturb_prob == 0.08      # defaults: scaled demos + tuned perturbation


def test_flatten_buffer_concatenates_episodes():
    """Flatten a sequence buffer's episodes to (obs, action) arrays for direct BC."""
    from data.replay_buffer import SequenceReplayBuffer
    from training.direct_bc import flatten_buffer
    buf = SequenceReplayBuffer(capacity=1000, seq_len=2)
    for i in range(5):
        buf.add(np.full(3, i, np.float32), np.full(2, i, np.float32), 0.0, i == 4)  # one 5-step episode
    obs, act = flatten_buffer(buf)
    assert obs.shape == (5, 3) and act.shape == (5, 2)
    assert obs[0].tolist() == [0, 0, 0] and act[4].tolist() == [4, 4]


def test_save_load_direct_roundtrip(tmp_path):
    """A saved direct policy must load back to the SAME function (so watch/eval reuse it exactly)."""
    pol = DirectPolicy(obs_dim=10, action_dim=2, hidden=32)
    x = torch.randn(4, 10)
    before = pol(x).detach()
    p = str(tmp_path / "policy.pt")
    save_direct(p, pol, obs_dim=10, action_dim=2)
    after = load_direct(p)(x).detach()
    assert torch.allclose(before, after, atol=1e-6)


def test_direct_policy_outputs_bounded_actions():
    pol = DirectPolicy(obs_dim=259, action_dim=2)
    out = pol(torch.randn(8, 259))
    assert out.shape == (8, 2)
    assert out.abs().max().item() <= 1.0          # tanh-bounded to the action range


def test_train_direct_bc_fits_a_simple_mapping():
    """BC should drive the loss down on a learnable obs->action mapping (sanity that it trains)."""
    torch.manual_seed(0); np.random.seed(0)
    n, d = 512, 6
    obs = np.random.randn(n, d).astype(np.float32)
    # a smooth target in [-1,1]: tanh of a fixed linear map -> definitely representable
    W = np.random.randn(d, 2).astype(np.float32)
    act = np.tanh(obs @ W).astype(np.float32)
    pol = DirectPolicy(obs_dim=d, action_dim=2, hidden=64)
    first = train_direct_bc(pol, obs, act, steps=1, lr=3e-3)
    last = train_direct_bc(pol, obs, act, steps=400, lr=3e-3)
    assert last < first                            # learning happened
    assert last < 0.1                              # and it actually fit the mapping

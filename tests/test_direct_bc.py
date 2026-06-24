"""Direct obs->action BC ablation: a plain MLP cloned straight from the 259-dim state vector, with NO
world model in the loop. The diagnostic for the latent-cloning hypothesis -- if this lane-keeps where
the WM-latent actor goes off-road 100%, the under-trained WM latent is the bottleneck. Pure pieces
(the policy + its trainer) are unit-tested here; the closed-loop comparison is the live ablation script.
"""
import numpy as np
import torch

from training.direct_bc import DirectPolicy, train_direct_bc


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

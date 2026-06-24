"""Direct obs->action behavior cloning (NO world model). The ablation baseline for the latent-cloning
hypothesis: clone the expert straight from the 259-dim state vector with a plain MLP. PilotNet /
ChauffeurNet style. If this lane-keeps where the WM-latent actor goes off-road, the under-trained
world-model latent is the bottleneck (not the data or the imitation objective).

L1 loss by default (Codevilla et al. 2019: better-correlated with driving than MSE, and outlier-robust).
"""
import os

import numpy as np
import torch
import torch.nn as nn


class DirectPolicy(nn.Module):
    """Plain state->action MLP, tanh-bounded to the env action range [-1,1]. Stateless (no RSSM)."""
    def __init__(self, obs_dim, action_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, obs):
        return torch.tanh(self.net(obs))


def train_direct_bc(policy, obs, act, steps, lr=3e-4, batch_size=256, l1=True, device="cpu"):
    """Behavior-clone `policy` to map obs -> expert action on flat (N, dim) arrays. Returns the final
    minibatch loss. L1 by default (driving-correlated + robust to the rare out-of-range expert action)."""
    policy.to(device).train()
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    obs_t = torch.as_tensor(np.asarray(obs), dtype=torch.float32, device=device)
    act_t = torch.as_tensor(np.asarray(act), dtype=torch.float32, device=device)
    n = obs_t.shape[0]
    bs = min(batch_size, n)
    last = 0.0
    for _ in range(steps):
        idx = torch.randint(0, n, (bs,), device=device)
        pred = policy(obs_t[idx])
        target = act_t[idx]
        loss = (pred - target).abs().mean() if l1 else ((pred - target) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        last = float(loss.detach())
    return last


def flatten_buffer(buf):
    """Flatten a SequenceReplayBuffer's stored episodes to flat (obs, action) arrays for direct BC.
    PURE (reads buf._episodes; call buf._flush() first to include a trailing run)."""
    if not buf._episodes:
        return np.zeros((0, 0), np.float32), np.zeros((0, 0), np.float32)
    obs = np.concatenate([e["obs"] for e in buf._episodes]).astype(np.float32)
    act = np.concatenate([e["action"] for e in buf._episodes]).astype(np.float32)
    return obs, act


def save_direct(path, policy, obs_dim, action_dim):
    """Save a DirectPolicy (with its dims/width) so watch/eval can reload the exact function."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({"state_dict": policy.state_dict(), "obs_dim": int(obs_dim),
                "action_dim": int(action_dim), "hidden": policy.net[0].out_features}, path)


def load_direct(path, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    pol = DirectPolicy(ckpt["obs_dim"], ckpt["action_dim"], hidden=ckpt.get("hidden", 256))
    pol.load_state_dict(ckpt["state_dict"])
    pol.eval()
    return pol

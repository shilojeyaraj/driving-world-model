"""
=== IMPLEMENT FROM SCRATCH ===

Concept:  Actor-critic trained INSIDE the world model's imagined rollouts (model-based RL).
          Actor proposes actions on latent states; critic estimates value; optimize with
          lambda-returns over imagined trajectories -- zero environment steps.

Question: Why can you train the policy on imagined rollouts with no env interaction at all,
          and what's the failure mode if the world model is wrong?

This payoff is what separates model-based from model-free RL.
"""
import torch.nn as nn


class Actor(nn.Module):
    def __init__(self, cfg, feat_dim, action_dim):
        super().__init__()
        raise NotImplementedError

    def forward(self, feat):
        raise NotImplementedError


class Critic(nn.Module):
    def __init__(self, cfg, feat_dim):
        super().__init__()
        raise NotImplementedError

    def forward(self, feat):
        raise NotImplementedError

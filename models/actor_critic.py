"""
=== IMPLEMENT FROM SCRATCH ===

Concept:  Actor-critic trained INSIDE the world model's imagined rollouts (model-based RL).
          Actor proposes actions on latent states; critic estimates value; optimize with
          lambda-returns over imagined trajectories -- zero environment steps.

Question: Why can you train the policy on imagined rollouts with no env interaction at all,
          and what's the failure mode if the world model is wrong?

This payoff is what separates model-based from model-free RL.

----------------------------------------------------------------------------------------
WHY a Tanh-Normal, reparameterized actor:
  Actions must live in [-1,1] (the env's range), so we squash a Gaussian through tanh. We
  sample with the reparameterization trick -- action = tanh(mean + std*eps) -- so gradients
  flow from the imagined RETURN, back through the world model's dynamics + reward head, and
  into the actor. That value-gradient is only possible because the whole simulator is
  differentiable. The entropy estimate (base-Normal entropy; the tanh Jacobian correction is
  omitted -- a v1 simplification) is a mild bonus that keeps the policy from collapsing too
  early.

WHY a critic:
  Imagined rollouts are short (cfg.imagine_horizon). The critic bootstraps the value beyond
  the horizon so returns reflect long-term reward, not just the next few steps.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _mlp(in_dim, hidden, out_dim):
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.SiLU(),
        nn.Linear(hidden, hidden),
        nn.SiLU(),
        nn.Linear(hidden, out_dim),
    )


class Actor(nn.Module):
    def __init__(self, cfg, feat_dim, action_dim):
        super().__init__()
        self.cfg = cfg
        self.min_std = cfg.min_std
        self.net = _mlp(feat_dim, cfg.hidden_dim, 2 * action_dim)  # -> (mean, std_raw)

    def forward(self, feat, deterministic=False):
        mean, std_raw = self.net(feat).chunk(2, dim=-1)
        std = F.softplus(std_raw) + self.min_std
        if deterministic:
            action = torch.tanh(mean)                       # eval / closed-loop: the mode
        else:
            action = torch.tanh(mean + std * torch.randn_like(std))   # reparameterized sample
        # Entropy of the pre-tanh Normal, summed over action dims (v1 simplification).
        entropy = (0.5 + 0.5 * torch.log(2 * torch.pi * std ** 2)).sum(-1)
        return action, entropy

    def log_prob(self, feat, action):
        """Log-density of `action` under this Tanh-Normal policy, summed over action dims. Used as
        the 'surprise' style signal (negative log-prob = how unusual the action is vs the policy).

        Change of variables for the tanh squash: log p(a) = log N(atanh(a); mean, std)
        - sum log(1 - a^2). `a` is clamped off ±1 so atanh is finite."""
        mean, std_raw = self.net(feat).chunk(2, dim=-1)
        std = F.softplus(std_raw) + self.min_std
        a = torch.clamp(torch.as_tensor(action, dtype=mean.dtype, device=mean.device), -0.999, 0.999)
        if a.dim() == 1:
            a = a.expand_as(mean)
        pre = 0.5 * (torch.log1p(a) - torch.log1p(-a))            # atanh(a)
        normal = torch.distributions.Normal(mean, std)
        return (normal.log_prob(pre) - torch.log(1.0 - a ** 2 + 1e-6)).sum(-1)


class Critic(nn.Module):
    def __init__(self, cfg, feat_dim):
        super().__init__()
        self.cfg = cfg
        self.net = _mlp(feat_dim, cfg.hidden_dim, 1)

    def forward(self, feat):
        return self.net(feat).squeeze(-1)

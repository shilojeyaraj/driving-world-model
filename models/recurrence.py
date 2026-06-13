"""
=== THE ABLATION SEAM (spec §13) ===

The RSSM's deterministic recurrence h_t = f(h_{t-1}, z_{t-1}, a_{t-1}) is isolated here so it
can be swapped (GRU vs SSM/Mamba vs ...) WITHOUT touching the prior/posterior heads or the
observe/imagine loops. A controlled swap of just this block, measured on the same eval harness
(open-loop error, sample efficiency, closed-loop return), is the project's unique angle.

Interface (Markov, fixed-size state):
    initial_state(batch, device) -> h0          # the recurrent carry (also the feature)
    forward(h, z, action)        -> h_next

NOTE on Transformer: a Transformer is NON-Markov (it attends over history), so it doesn't fit
this fixed-size-state interface directly. It needs a windowed-state variant (carry the last K
inputs); that's a later sub-step. GRU and a diagonal-SSM / Mamba-style block both fit as-is.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GRURecurrence(nn.Module):
    """h_t = GRU( proj([z_{t-1}, a_{t-1}]), h_{t-1} ). The v1 default."""

    def __init__(self, cfg, action_dim):
        super().__init__()
        self.deter_dim = cfg.deter_dim
        self.input_proj = nn.Sequential(nn.Linear(cfg.stoch_dim + action_dim, cfg.hidden_dim), nn.SiLU())
        self.gru = nn.GRUCell(cfg.hidden_dim, cfg.deter_dim)

    def initial_state(self, batch_size, device):
        return torch.zeros(batch_size, self.deter_dim, device=device)

    def forward(self, h, z, action):
        x = self.input_proj(torch.cat([z, action], dim=-1))
        return self.gru(x, h)


class SSMRecurrence(nn.Module):
    """Minimal Mamba-style SELECTIVE diagonal state-space recurrence.

    A GRU mixes the whole state through a gated nonlinearity every step. A state-space model
    instead runs a *linear* recurrence per channel: s_t = ā ⊙ s_{t-1} + Δ ⊙ (B·x). What makes
    it "selective" (Mamba's idea) is that the step size Δ is INPUT-DEPENDENT: per channel, the
    input decides how much to forget vs. absorb this step. With negative real `a` the discrete
    decay ā = exp(Δ·a) lives in (0,1), so the recurrence is stable by construction.

    We expose the state `s` directly as the deterministic feature `h` (the prior/posterior MLP
    heads supply the nonlinearity), so it satisfies the same Markov interface as the GRU:
    state == feature == an R^{deter_dim} vector.
    """

    def __init__(self, cfg, action_dim):
        super().__init__()
        D = cfg.deter_dim
        self.deter_dim = D
        in_dim = cfg.stoch_dim + action_dim
        self.A_log = nn.Parameter(torch.zeros(D))      # a = −softplus(A_log) < 0  (stable decay)
        self.x_proj = nn.Linear(in_dim, D)             # B·x : maps the input into state space
        self.dt_proj = nn.Linear(in_dim, D)            # Δ   : per-channel, input-dependent step

    def initial_state(self, batch_size, device):
        return torch.zeros(batch_size, self.deter_dim, device=device)

    def forward(self, h, z, action):
        u = torch.cat([z, action], dim=-1)
        a = -F.softplus(self.A_log)                    # (D,)   negative real
        dt = F.softplus(self.dt_proj(u))               # (B, D) > 0   selective step
        a_bar = torch.exp(dt * a)                      # (B, D) in (0,1)  discretized decay
        bx = dt * self.x_proj(u)                       # (B, D)  discretized input
        return a_bar * h + bx                          # s_t = ā ⊙ s_{t-1} + Δ ⊙ (B·x)


def make_recurrence(cfg, action_dim):
    """Select the deterministic recurrence by cfg.dynamics."""
    if cfg.dynamics == "rssm":
        return GRURecurrence(cfg, action_dim)
    if cfg.dynamics == "mamba":
        return SSMRecurrence(cfg, action_dim)
    if cfg.dynamics == "transformer":
        raise NotImplementedError(
            "cfg.dynamics='transformer' is non-Markov (attends over history) and needs a "
            "windowed-state recurrence variant -- a later sub-step. Use 'rssm' or 'mamba'."
        )
    raise ValueError(f"unknown dynamics: {cfg.dynamics!r}")

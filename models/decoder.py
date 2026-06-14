"""
=== IMPLEMENT FROM SCRATCH ===

Concept:  Likelihood / reconstruction heads. From the feature (h_t, z_t) predict:
            - the observation (image or state vector)  -> reconstruction term of the ELBO
            - the reward                               -> so the policy can plan for return
            - the continue flag (1 - done)             -> so imagination knows when to stop
          Optional advanced: a DiT (diffusion transformer) image head instead of a plain CNN
          decoder. NOTE: diffusion belongs HERE (it's a generator), not in the encoder.

Question: Why predict reward and continue, not just the observation? What breaks in
          imagination if you drop the continue head?

----------------------------------------------------------------------------------------
WHY three heads (not just obs):
  The world model has to be a *self-contained simulator* the policy can dream inside. To
  compute returns in imagination there is no env to ask, so the model must predict the
  reward itself (reward head) and predict WHEN an episode ends (continue head, cont = 1-done)
  so returns stop discounting past termination. The obs head gives the reconstruction signal
  that forces the latent to actually encode the observation. Drop the continue head and
  imagined rollouts run past 'done' forever, inflating returns with reward that can't happen.

  All three read the SAME feature `feat = [h; z]`. We model obs and reward as Gaussians with
  unit variance (=> the NLL reduces to MSE) and continue as a Bernoulli (=> BCE on a logit).
  The reward/continue heads emit one scalar per item; we squeeze the trailing dim so shapes
  are (N,), matching the (B, T) reward/done targets from the buffer.
"""
import torch.nn as nn


class _ImageObsHead(nn.Module):
    """Transposed-CNN mirror of the encoder: feat -> (3,H,W). A linear lifts feat to a small
    256-channel spatial grid, then 4 stride-2 transposed convs double the size each step back
    up to image_size. No final activation (Gaussian mean -> MSE), matching the state head."""

    def __init__(self, cfg, feat_dim):
        super().__init__()
        self.s0 = cfg.image_size // 16
        assert self.s0 >= 1, "image_size must be a multiple of 16 (>=16)"
        self.fc = nn.Linear(feat_dim, 256 * self.s0 * self.s0)
        chs, in_c, layers = [128, 64, 32], 256, []
        for c in chs:
            layers += [nn.ConvTranspose2d(in_c, c, kernel_size=4, stride=2, padding=1), nn.SiLU()]
            in_c = c
        layers += [nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1)]
        self.deconv = nn.Sequential(*layers)

    def forward(self, feat):
        x = self.fc(feat).reshape(-1, 256, self.s0, self.s0)
        return self.deconv(x)


def _mlp(in_dim, hidden, out_dim):
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.SiLU(),
        nn.Linear(hidden, hidden),
        nn.SiLU(),
        nn.Linear(hidden, out_dim),
    )


class Decoder(nn.Module):
    def __init__(self, cfg, feat_dim):
        super().__init__()
        self.cfg = cfg
        h = cfg.hidden_dim

        if cfg.obs_type == "state":
            self.obs_head = _mlp(feat_dim, h, cfg.state_dim)   # Gaussian mean (unit var -> MSE)
        elif cfg.obs_type == "image":
            self.obs_head = _ImageObsHead(cfg, feat_dim)       # transposed-CNN -> (3,H,W)
        else:
            raise ValueError(f"unknown obs_type: {cfg.obs_type}")

        self.reward_head = _mlp(feat_dim, h, 1)   # Gaussian mean (unit var -> MSE)
        self.cont_head = _mlp(feat_dim, h, 1)     # Bernoulli logit (BCE-with-logits)

    def forward(self, feat):
        # feat: (N, feat_dim) -> dict of per-item predictions.
        return {
            "obs": self.obs_head(feat),
            "reward": self.reward_head(feat).squeeze(-1),       # (N, 1) -> (N,)
            "cont_logit": self.cont_head(feat).squeeze(-1),     # (N, 1) -> (N,)
        }

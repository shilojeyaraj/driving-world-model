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
            # Deferred to the GPU/image phase (spec §13): a transposed-CNN mirror of the
            # encoder. Written + validated there, with its own test.
            raise NotImplementedError(
                "image-mode Decoder is a later (GPU) phase; v1 Phase 1 is state-only"
            )
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

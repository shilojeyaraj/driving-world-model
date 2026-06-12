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
"""
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, cfg, feat_dim):
        super().__init__()
        self.cfg = cfg
        # feat_dim = cfg.deter_dim + cfg.stoch_dim
        # TODO: obs head (CNN/MLP per obs_type), reward head, continue head.
        raise NotImplementedError

    def forward(self, feat):
        # return {"obs": ..., "reward": ..., "cont": ...}
        raise NotImplementedError

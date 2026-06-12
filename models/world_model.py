"""
World model = Encoder + RSSM + Decoder, plus the LOSS that ties them together.

Encoder / RSSM / Decoder: from scratch (their own files).
assemble_loss():          === IMPLEMENT FROM SCRATCH -- this is the ELBO. ===

Concept:  The full variational objective:
            loss = recon_nll + reward_nll + cont_nll + kl_scale * KL(posterior || prior)
          Use KL balancing and free bits (cfg.free_bits) so the KL term doesn't collapse
          the posterior.

Question: What is posterior collapse, and how would you detect it from the loss curves?
          (Hint: watch whether the KL term and the reconstruction term move together.)
"""
import torch.nn as nn

from .encoder import Encoder
from .rssm import RSSM
from .decoder import Decoder


class WorldModel(nn.Module):
    def __init__(self, cfg, action_dim):
        super().__init__()
        self.cfg = cfg
        self.encoder = Encoder(cfg)
        self.rssm = RSSM(cfg, self.encoder.embed_dim, action_dim)
        self.decoder = Decoder(cfg, cfg.deter_dim + cfg.stoch_dim)

    def assemble_loss(self, batch):
        """batch: dict of (B, T, ...) tensors from the replay buffer.
        Return (scalar_loss, metrics_dict). IMPLEMENT THE ELBO HERE."""
        raise NotImplementedError

"""
=== IMPLEMENT FROM SCRATCH ===

Concept:  Representation learning. Map a high-dim observation -> a compact embedding.
          obs_type == "image": CNN vs ViT is convolutional inductive bias vs. global
          attention. obs_type == "state": a small MLP is fine.

Question: Why might a ViT need more data than a CNN to reach the same reconstruction
          quality? (What prior does convolution bake in that attention does not?)

Don't import a prebuilt encoder. Writing it yourself is the point.
"""
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed_dim = None   # SET THIS -- the size of the embedding you return
        # TODO:
        #   obs_type == "state" -> MLP over cfg.state_dim
        #   obs_type == "image" -> CNN (cfg.encoder == "cnn") or ViT (cfg.encoder == "vit")
        raise NotImplementedError("Encoder.__init__")

    def forward(self, obs):
        # obs: (batch, *obs_shape) -> return (batch, embed_dim)
        raise NotImplementedError("Encoder.forward")

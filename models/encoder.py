"""
=== IMPLEMENT FROM SCRATCH ===

Concept:  Representation learning. Map a high-dim observation -> a compact embedding.
          obs_type == "image": CNN vs ViT is convolutional inductive bias vs. global
          attention. obs_type == "state": a small MLP is fine.

Question: Why might a ViT need more data than a CNN to reach the same reconstruction
          quality? (What prior does convolution bake in that attention does not?)

Don't import a prebuilt encoder. Writing it yourself is the point.

----------------------------------------------------------------------------------------
WHY this design (state mode):
  The encoder's only job is to turn one observation into a vector `e_t` that the RSSM's
  postgerior can read. In `state` mode the obs is already a low-dim vector, so a plain MLP
  is enough -- there is no spatial structure for convolution to exploit. We map
  state_dim -> hidden -> hidden -> embed_dim with SiLU nonlinearities. embed_dim is set to
  cfg.hidden_dim so the rest of the model has one width knob to tune.

  (image mode -- a Dreamer-style strided CNN -- is deferred to the GPU phase and raises
  below until it has its own test. v1 Phase 1 is state-only.)
"""
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        if cfg.obs_type == "state":
            # embed_dim is the contract the RSSM/posterior depend on. One width knob.
            self.embed_dim = cfg.hidden_dim
            self.net = nn.Sequential(
                nn.Linear(cfg.state_dim, cfg.hidden_dim),
                nn.SiLU(),
                nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
                nn.SiLU(),
                nn.Linear(cfg.hidden_dim, self.embed_dim),
            )
        elif cfg.obs_type == "image":
            # Deferred to the GPU/image phase (spec §13). Written + validated there, with
            # its own shape test. Kept explicit rather than half-built.
            raise NotImplementedError(
                "image-mode Encoder is a later (GPU) phase; v1 Phase 1 is state-only"
            )
        else:
            raise ValueError(f"unknown obs_type: {cfg.obs_type}")

    def forward(self, obs):
        # obs: (N, *obs_shape) -> (N, embed_dim). Caller flattens (B, T) into N=B*T.
        return self.net(obs)

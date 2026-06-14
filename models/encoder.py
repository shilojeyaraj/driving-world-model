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

  image mode (V1): a Dreamer-style strided CNN. 4 stride-2 conv layers halve the spatial
  size each time (so image_size must be a multiple of 16), then a linear maps the flattened
  features to embed_dim. Validated on a tiny 16x16 render on CPU; same code scales to 64x64
  on a GPU.
"""
import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed_dim = cfg.hidden_dim

        if cfg.obs_type == "state":
            self.net = nn.Sequential(
                nn.Linear(cfg.state_dim, cfg.hidden_dim),
                nn.SiLU(),
                nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
                nn.SiLU(),
                nn.Linear(cfg.hidden_dim, self.embed_dim),
            )
        elif cfg.obs_type == "image":
            chs, in_c, layers = [32, 64, 128, 256], 3, []
            for c in chs:
                layers += [nn.Conv2d(in_c, c, kernel_size=4, stride=2, padding=1), nn.SiLU()]
                in_c = c
            self.conv = nn.Sequential(*layers)
            with torch.no_grad():                          # infer the flattened conv size
                flat = self.conv(torch.zeros(1, 3, cfg.image_size, cfg.image_size)).flatten(1).shape[1]
            self.head = nn.Linear(flat, self.embed_dim)
        else:
            raise ValueError(f"unknown obs_type: {cfg.obs_type}")

    def forward(self, obs):
        # obs: (N, *obs_shape) -> (N, embed_dim). Caller flattens (B, T) into N=B*T.
        if self.cfg.obs_type == "state":
            return self.net(obs)
        return self.head(self.conv(obs).flatten(1))

"""Shape contracts for the from-scratch model components (spec §7: test_shapes.py).

These are the first tests written test-first: each pins the I/O contract a component
must satisfy. Start with the Encoder in state mode.
"""
import torch

from config import get_config
from models.encoder import Encoder
from models.decoder import Decoder


def test_encoder_state_output_shape():
    """state-mode Encoder maps (N, state_dim) -> (N, embed_dim), and exposes embed_dim."""
    cfg = get_config(obs_type="state", state_dim=35)
    enc = Encoder(cfg)

    assert isinstance(enc.embed_dim, int) and enc.embed_dim > 0

    N = 8
    obs = torch.randn(N, cfg.state_dim)
    embed = enc(obs)

    assert embed.shape == (N, enc.embed_dim)


def test_decoder_state_output_shapes():
    """state-mode Decoder maps feat (N, feat_dim) -> obs (N, state_dim), reward (N,),
    cont_logit (N,). These three heads are the reconstruction + reward + continue terms
    of the ELBO."""
    cfg = get_config(obs_type="state", state_dim=35)
    feat_dim = cfg.deter_dim + cfg.stoch_dim
    dec = Decoder(cfg, feat_dim)

    N = 8
    feat = torch.randn(N, feat_dim)
    out = dec(feat)

    assert set(out) >= {"obs", "reward", "cont_logit"}
    assert out["obs"].shape == (N, cfg.state_dim)
    assert out["reward"].shape == (N,)
    assert out["cont_logit"].shape == (N,)

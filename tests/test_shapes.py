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


def test_encoder_image_output_shape():
    """image-mode CNN encoder maps (N,3,H,W) -> (N, embed_dim)."""
    cfg = get_config(obs_type="image", image_size=16, encoder="cnn", hidden_dim=32)
    enc = Encoder(cfg)
    assert isinstance(enc.embed_dim, int) and enc.embed_dim > 0
    e = enc(torch.rand(4, 3, 16, 16))
    assert e.shape == (4, enc.embed_dim)


def test_decoder_image_output_shapes():
    """image-mode decoder maps feat -> obs (N,3,H,W) image + reward/continue scalars."""
    cfg = get_config(obs_type="image", image_size=16, hidden_dim=32)
    feat_dim = cfg.deter_dim + cfg.stoch_dim
    dec = Decoder(cfg, feat_dim)
    out = dec(torch.randn(4, feat_dim))
    assert out["obs"].shape == (4, 3, 16, 16)
    assert out["reward"].shape == (4,)
    assert out["cont_logit"].shape == (4,)

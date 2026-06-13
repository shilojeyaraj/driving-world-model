"""Phase 1 milestone (spec §10.1): overfit-a-batch reconstruction sanity.

With only the Encoder + Decoder built (no RSSM yet), the reconstruction loop is an
autoencoder: encode an observation to `embed`, then decode `embed` straight back to the
observation. If the two modules are wired correctly and gradients flow through both, a
small net can drive the MSE on a *fixed* batch to ~0. If anything is miswired (detached
graph, wrong shapes feeding the loss, dead activations) the loss will not move.

This is the standalone check that "reconstruction works" before the RSSM exists.
"""
import pytest
import torch

from config import get_config
from envs.base import make_env
from models.encoder import Encoder
from models.decoder import Decoder


def _fixed_state_batch(cfg, n, seed=0):
    """A fixed batch of real DummyEnv state observations to overfit."""
    import numpy as np
    np.random.seed(seed)
    env = make_env(cfg)
    obs = [env.reset()]
    for _ in range(n - 1):
        a = np.random.uniform(-1, 1, cfg.action_dim).astype(np.float32)
        o, _, done, _ = env.step(a)
        obs.append(env.reset() if done else o)
    return torch.from_numpy(np.stack(obs[:n]))


@pytest.mark.slow
def test_encoder_decoder_overfits_one_batch():
    torch.manual_seed(0)
    cfg = get_config(obs_type="state", state_dim=35, max_episode_steps=200)

    enc = Encoder(cfg)
    # Autoencoder: the decoder consumes the encoder embedding directly as its "feat".
    dec = Decoder(cfg, enc.embed_dim)

    obs = _fixed_state_batch(cfg, n=16)
    params = list(enc.parameters()) + list(dec.parameters())
    opt = torch.optim.Adam(params, lr=1e-3)

    def recon_loss():
        pred = dec(enc(obs))["obs"]
        return ((pred - obs) ** 2).mean()

    initial = recon_loss().item()
    for _ in range(300):
        opt.zero_grad()
        loss = recon_loss()
        loss.backward()
        opt.step()
    final = recon_loss().item()

    # Overfitting a fixed batch must drive the loss down sharply.
    assert final < initial * 0.1, f"recon did not drop: {initial:.4f} -> {final:.4f}"
    assert final < 0.01, f"recon did not reach ~0: {final:.4f}"

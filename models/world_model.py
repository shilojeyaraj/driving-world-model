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
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import Encoder
from .rssm import RSSM
from .decoder import Decoder


def _gaussian_kl(q, p):
    """KL( q || p ) for diagonal Gaussians, PER latent dim. q=posterior, p=prior.
    Closed form: log(σ_p/σ_q) + (σ_q² + (μ_q−μ_p)²)/(2σ_p²) − 1/2.  (No sampling needed.)"""
    qm, qs = q["mean"], q["std"]
    pm, ps = p["mean"], p["std"]
    return torch.log(ps / qs) + (qs ** 2 + (qm - pm) ** 2) / (2 * ps ** 2) - 0.5


class WorldModel(nn.Module):
    def __init__(self, cfg, action_dim):
        super().__init__()
        self.cfg = cfg
        self.encoder = Encoder(cfg)
        self.rssm = RSSM(cfg, self.encoder.embed_dim, action_dim)
        self.decoder = Decoder(cfg, cfg.deter_dim + cfg.stoch_dim)

    def assemble_loss(self, batch):
        """batch: dict of (B, T, ...) tensors from the replay buffer. Returns (loss, metrics).

        This IS the ELBO. For a sequential latent model with a deterministic h_t:
            log p(o,r,c | a) >= Σ_t E_q[ log p(o_t,r_t,c_t | h_t,z_t) ] - Σ_t KL( q(z_t|h_t,e_t) || p(z_t|h_t) )
        Maximizing that bound == minimizing the loss below (recon + reward + cont are the
        negative log-likelihood terms; KL regularizes the posterior toward the prior).
        """
        cfg = self.cfg
        obs, actions = batch["obs"], batch["action"]
        reward, done = batch["reward"], batch["done"]
        B, T = obs.shape[:2]

        # 1) Encode every timestep. Encoder is per-step, so flatten (B,T) -> (B*T) and back.
        embeds = self.encoder(obs.reshape(B * T, *obs.shape[2:])).reshape(B, T, -1)

        # 2) Posterior roll-out: features + per-step prior/posterior stats.
        out = self.rssm.observe(embeds, actions, self.rssm.initial_state(B, obs.device))
        feat = out["feat"]

        # 3) Decode the features into observation / reward / continue predictions.
        dec = self.decoder(feat.reshape(B * T, feat.shape[-1]))
        obs_hat = dec["obs"].reshape(B, T, -1)
        reward_hat = dec["reward"].reshape(B, T)
        cont_logit = dec["cont_logit"].reshape(B, T)

        # 4) Negative-log-likelihood terms (Gaussian unit-var -> MSE; Bernoulli -> BCE).
        #
        # OBS aligns straight: feat_t is the posterior state at t (it consumed e_t = enc(obs_t)),
        # so feat_t predicts obs_t. No shift.
        recon = ((obs_hat - obs) ** 2).sum(-1).mean()          # sum over obs dims, mean over B,T
        #
        # REWARD / CONTINUE are TRANSITION quantities. In this env reward[t] = R(s_t, a_t),
        # a function of a_t -- but feat_t carries a_{t-1} (the recurrence consumes the previous
        # action), NOT a_t. So feat_t can only predict the reward/continue produced by a_{t-1},
        # i.e. buffer index t-1. We align by shifting one step: predictions at t=1..T-1 are
        # matched to targets at t=0..T-2. (feat_0 consumed the fake a_{-1}=0, so it has no real
        # reward target and is dropped.) This is also what makes imagination consistent: there
        # the reward for action a_i is read from feat_{i+1}, which is the state that consumed a_i.
        reward_nll = ((reward_hat[:, 1:] - reward[:, :-1]) ** 2).mean()
        cont_nll = F.binary_cross_entropy_with_logits(cont_logit[:, 1:], 1.0 - done[:, :-1])

        # 5) KL( post || prior ), summed over the latent dim -> per-step KL (B,T).
        kl_per_step = _gaussian_kl(out["post"], out["prior"]).sum(-1)
        raw_kl = kl_per_step.mean()                            # for the collapse watch
        # Free bits: don't pay for the first `free_bits` nats. Removes the gradient pressure
        # that would otherwise drive the posterior to match the prior (posterior collapse).
        kl_loss = torch.clamp(kl_per_step, min=cfg.free_bits).mean()

        loss = recon + reward_nll + cont_nll + cfg.kl_scale * kl_loss

        metrics = {
            "loss": loss.detach(),
            "recon": recon.detach(),
            "reward": reward_nll.detach(),
            "cont": cont_nll.detach(),
            "kl": raw_kl.detach(),          # watch THIS: if it -> 0 while recon stalls = collapse
            "kl_loss": kl_loss.detach(),
        }
        return loss, metrics

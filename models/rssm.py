"""
=== IMPLEMENT FROM SCRATCH (this is the core of the whole project) ===

Concept:  Recurrent State-Space Model -- a latent-variable sequence model with:
            h_t = f(h_{t-1}, z_{t-1}, a_{t-1})     deterministic recurrence (e.g. GRUCell)
            prior     p(z_t | h_t)                 predict the latent WITHOUT the observation
            posterior q(z_t | h_t, e_t)            infer the latent WITH it (e_t = encoder out)
          Training is variational (the loss is an ELBO). The KL(posterior || prior) term
          teaches the prior to predict the future -- which is what makes imagination work.

Question: Where does KL(posterior || prior) come from in the ELBO, and what role does the
          prior play at imagination time (when there is no observation to condition on)?

Derive the ELBO by hand once before coding. If you can derive it, you own it.
Use the reparameterization trick for the stochastic z_t. See CONCEPTS.md.

ABLATION HOOK: cfg.dynamics in {"rssm","transformer","mamba"} -- swap ONLY the recurrence
here, keep the prior/posterior interface identical. That controlled swap is your unique angle.
"""
import torch.nn as nn


class RSSM(nn.Module):
    def __init__(self, cfg, embed_dim, action_dim):
        super().__init__()
        self.cfg = cfg
        # TODO: GRUCell for the deterministic path; MLP heads for prior and posterior.
        raise NotImplementedError

    def initial_state(self, batch_size, device):
        # return (h_0, z_0)
        raise NotImplementedError

    def observe(self, embeds, actions, state):
        """Roll forward WITH observations (posterior path). Return per-step priors,
        posteriors, and features so the loss can compute reconstruction + KL."""
        raise NotImplementedError

    def imagine(self, actions, state):
        """Roll forward WITHOUT observations (prior path). This is the dream loop the
        policy trains inside -- no encoder, no pixels, pure latent."""
        raise NotImplementedError

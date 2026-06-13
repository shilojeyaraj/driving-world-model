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

----------------------------------------------------------------------------------------
WHY this split (deterministic h + stochastic z):
  The state is a *pair*. `h` is a deterministic GRU memory: it carries forward everything
  the model is *sure* about (a clean channel for long-range info, no sampling noise). `z` is
  a small stochastic latent that captures what's *uncertain* about the current step. Two
  heads read `h`:
    - prior p(z|h): the model's GUESS of z before seeing the observation -- this is what runs
      at imagination time (no obs available).
    - posterior q(z|h,e): the CORRECTED z after seeing the encoded observation e.
  The KL(post || prior) term in the loss drags the prior toward the posterior, i.e. teaches
  the prior to predict what the observation *would* have said. Once the prior is good, you can
  drop the observation entirely and roll forward in latent space (`imagine`). Delete the KL
  and the prior never learns to predict -> imagination is garbage.

ACTION TIMING (spec §3, the off-by-one that bites everyone):
  A batch entry means: at step t you saw obs[t], took action[t], got reward[t], done[t].
  The recurrence consumes the PREVIOUS action: h_t = GRU(h_{t-1}, [z_{t-1}, a_{t-1}]).
  So in `observe`, step t feeds a_{t-1} (a_{-1}=0 for the first step); feat_t = [h_t; z_t]
  then predicts obs_t/reward_t/cont_t. The last action a_{T-1} is unused by observe.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .recurrence import make_recurrence


def _mlp(in_dim, hidden, out_dim):
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.SiLU(),
        nn.Linear(hidden, out_dim),
    )


class RSSM(nn.Module):
    def __init__(self, cfg, embed_dim, action_dim):
        super().__init__()
        self.cfg = cfg
        self.deter_dim = cfg.deter_dim
        self.stoch_dim = cfg.stoch_dim
        self.min_std = cfg.min_std
        h = cfg.hidden_dim

        # Deterministic path: a SWAPPABLE recurrence (GRU / SSM / ...), selected by cfg.dynamics.
        # This is the ablation seam -- the prior/posterior heads below don't change when it does.
        self.recurrence = make_recurrence(cfg, action_dim)

        # Stochastic heads emit (mean, std_raw); std = softplus(std_raw) + min_std keeps it
        # strictly positive and floored (avoids a collapsing / exploding variance).
        self.prior_head = _mlp(self.deter_dim, h, 2 * self.stoch_dim)              # p(z | h)
        self.post_head = _mlp(self.deter_dim + embed_dim, h, 2 * self.stoch_dim)   # q(z | h, e)

    # --- helpers -------------------------------------------------------------------------
    def _stats(self, head_out):
        mean, std_raw = head_out.chunk(2, dim=-1)
        std = F.softplus(std_raw) + self.min_std
        return {"mean": mean, "std": std}

    def _sample(self, stats):
        # Reparameterization trick: z = mean + std * eps. Keeps gradients flowing through the
        # sampler (essential -- the policy later backprops returns THROUGH these samples).
        eps = torch.randn_like(stats["std"])
        return stats["mean"] + stats["std"] * eps

    def _recur(self, h, z, action):
        # h_t = recurrence( h_{t-1}, z_{t-1}, a )  -- GRU by default, swappable via cfg.dynamics.
        return self.recurrence(h, z, action)

    # --- single-step primitives (shared by both loops) -----------------------------------
    def obs_step(self, state, prev_action, embed):
        """Posterior step. Advances h with the PREVIOUS action, then infers z FROM the obs."""
        h_prev, z_prev = state
        h = self._recur(h_prev, z_prev, prev_action)
        prior = self._stats(self.prior_head(h))                       # guess (for the KL target)
        post = self._stats(self.post_head(torch.cat([h, embed], dim=-1)))  # corrected by obs
        z = self._sample(post)
        return (h, z), prior, post

    def img_step(self, state, action, sample=True):
        """Prior step. Advances h with `action`, draws z FROM THE PRIOR (no observation).
        This is the simulator step: it produces the next latent state with no obs in sight.

        sample=True  -> reparameterized sample (behavior training needs the stochasticity and
                        the gradient path through the sampler).
        sample=False -> the prior MEAN (deterministic). Open-loop *prediction* eval uses this:
                        injecting sampling noise into a prediction metric just measures noise."""
        h_prev, z_prev = state
        h = self._recur(h_prev, z_prev, action)
        prior = self._stats(self.prior_head(h))
        z = self._sample(prior) if sample else prior["mean"]
        return (h, z), prior

    # --- primitives shared / state ------------------------------------------------------
    def initial_state(self, batch_size, device):
        h = self.recurrence.initial_state(batch_size, device)
        z = torch.zeros(batch_size, self.stoch_dim, device=device)
        return (h, z)

    @staticmethod
    def _feat(h, z):
        return torch.cat([h, z], dim=-1)

    @staticmethod
    def _stack_stats(stats_list):
        return {
            "mean": torch.stack([s["mean"] for s in stats_list], dim=1),
            "std": torch.stack([s["std"] for s in stats_list], dim=1),
        }

    # --- public loops -------------------------------------------------------------------
    def observe(self, embeds, actions, state):
        """Roll forward WITH observations (posterior path). embeds (B,T,E), actions (B,T,A).
        Returns per-step prior/post stats, feat=[h;z] (B,T,deter+stoch), and final state."""
        B, T, _ = embeds.shape
        feats, priors, posts = [], [], []
        for t in range(T):
            # Previous-action convention: step t consumes a_{t-1}; a_{-1} = 0.
            prev_action = actions[:, t - 1] if t > 0 else torch.zeros_like(actions[:, 0])
            state, prior, post = self.obs_step(state, prev_action, embeds[:, t])
            h, z = state
            feats.append(self._feat(h, z))
            priors.append(prior)
            posts.append(post)
        return {
            "feat": torch.stack(feats, dim=1),
            "prior": self._stack_stats(priors),
            "post": self._stack_stats(posts),
            "state": state,
        }

    def imagine(self, actions, state, sample=True):
        """Roll forward WITHOUT observations (prior path). This is the dream loop the
        policy trains inside -- no encoder, no pixels, pure latent. actions (B,H,A).
        Convention: actor acts then the world steps -- action[:, i] drives the i-th step.
        sample=False rolls the prior MEAN (deterministic prediction; see img_step).
        Returns per-step feat=[h;z] (B,H,deter+stoch), prior stats, and final state."""
        H = actions.shape[1]
        feats, priors = [], []
        for i in range(H):
            state, prior = self.img_step(state, actions[:, i], sample=sample)
            h, z = state
            feats.append(self._feat(h, z))
            priors.append(prior)
        return {
            "feat": torch.stack(feats, dim=1),
            "prior": self._stack_stats(priors),
            "state": state,
        }

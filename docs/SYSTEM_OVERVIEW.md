# System overview — the ML story (interview guide)

A narrative, interview-oriented explanation of the whole project in ML terms. For the math
(ELBO derivation, λ-returns, action-timing), see `ARCHITECTURE.md`; this doc is the *story* and
the *talking points*.

---

## 0. The 60-second pitch

> I built a **Dreamer-style world model** for driving, from scratch. A world model is a learned,
> **differentiable simulator**: it compresses observations into a latent state and learns the
> latent *dynamics*, so you can "dream" future trajectories without touching the real
> environment. I then train a controller **entirely inside that dream** (model-based RL), which
> is sample-efficient because the policy learns from imagined rollouts, not env steps. I
> validated it end-to-end on a toy env (it learns the optimal policy with zero env interaction
> during policy learning), ran a **controlled ablation** of the dynamics core (GRU vs a
> Mamba-style state-space model), extended it to **pixels** (it learns to *dream video*), and
> connected it to a **real driving sim (MetaDrive)** — where I documented exactly where the
> simplified stack succeeds and where it hits known limits.

## 1. Why a world model (the motivation)

Model-free RL needs millions of real env steps. A **world model** learns the environment's
dynamics, then trains the policy on cheap **imagined** rollouts → sample efficiency, and it
gives you a *predictive* model you can introspect (forecast outcomes, render dreams). This is
the Dreamer line of work (Hafner et al.).

## 2. The architecture (and the ML role of each piece)

```
obs ─Encoder─► e_t ─┐
                    ├─ RSSM ─► (h_t, z_t) ─ Decoder ─► ô_t, r̂_t, ĉ_t        (world-model learning)
prev action ────────┘            │
                                 └─ imagine (prior only) ─► Actor–Critic       (behavior learning)
```

- **Encoder** (`models/encoder.py`) — representation learning. MLP for low-dim state; a strided
  **CNN** for pixels. Maps an observation to an embedding `e_t`.
- **RSSM** (`models/rssm.py`) — the core: a **recurrent state-space model**. State is a pair
  `(h, z)`: `h` is a deterministic GRU memory (clean long-range channel), `z` a stochastic latent
  (per-step uncertainty). Two heads read `h`: the **prior** `p(z|h)` (predict the latent *without*
  the obs) and the **posterior** `q(z|h,e)` (corrected *by* the obs). Trained variationally.
- **Decoder** (`models/decoder.py`) — likelihood heads: reconstruct the obs, and predict
  **reward** and **continue** (`1−done`), so the model is a *self-contained simulator* the policy
  can plan inside (no env to query in a dream).
- **Actor–Critic** (`models/actor_critic.py`) — a Tanh-Normal policy + value critic trained on
  **imagined** rollouts (model-based RL), zero env steps.

## 3. The core ML concepts (what you'd be quizzed on)

- **Variational inference / ELBO.** The world model is a latent-variable sequence model; training
  maximizes an evidence lower bound: `Σ_t E_q[log p(o,r,c | h,z)] − Σ_t KL(q(z|h,e) ‖ p(z|h))`.
  The likelihood terms become reconstruction + reward + continue losses; the KL regularizes.
- **Prior vs posterior, and imagination.** The KL term *teaches the prior to predict the latent
  the observation would have implied*. Once the prior is good, you drop the obs and roll forward
  in latent space (`imagine`) — that's the dream the policy trains in.
- **Reparameterization trick.** `z = μ + σ·ε` so gradients flow through the sampler — essential
  because the policy's value gradient backprops *through* the sampled latents.
- **Posterior collapse + free bits.** If the KL is driven to 0 the latent is ignored; you detect
  it by watching KL vs reconstruction together. **Free bits** (don't penalize the first N nats of
  KL) removes the gradient pressure that causes collapse.
- **Model-based RL / value gradients.** The actor maximizes the imagined **λ-return**; because the
  whole simulator is differentiable, the return backprops through the reward head and dynamics
  into the actor (a *value gradient*) — no env steps.
- **λ-returns.** `V^λ_t = r_t + γc_t[(1−λ)v_{t+1} + λV^λ_{t+1}]` interpolates one-step TD (λ=0,
  low-variance/biased) and Monte-Carlo (λ=1, unbiased/high-variance); the critic bootstraps past
  the short imagination horizon.
- **Two eval axes.** **Open-loop** (feed true actions, measure prediction) tests *dynamics*;
  **closed-loop** (let the policy act, measure outcome) tests *control*. A model can ace one and
  fail the other (distribution shift / compounding error).

## 4. The training algorithm

- **Single-shot** (`training/train_behavior.py`): collect random data → train WM → train policy
  in imagination. Enough for the toy.
- **Iterated Dreamer loop** (`training/dreamer_loop.py`): seed → repeat (collect *with the policy*
  + exploration → train WM → train policy). Collecting with the policy **grounds the model in the
  states the policy actually visits**, closing the imagination-vs-reality gap. This is the real
  algorithm.

## 5. Engineering & debugging stories (the interview gold)

These show judgment, not just knowledge:

1. **Action-timing off-by-one.** The feature `feat_t` carries the *previous* action `a_{t-1}`, but
   the env's `reward_t = R(s_t, a_t)` depends on `a_t`. Training the reward head naively made it
   predict the *mean* reward. I caught it not with an overfit test (which *hid* it via
   memorization) but with a **held-out generalization test** — reward MSE was stuck at the reward
   variance. Fix: align reward/continue to the action that's actually in the feature. *Lesson:
   overfitting proves optimization works, not that something is learned for the right reason.*

2. **A measurement bug that looked like a modeling failure.** Open-loop prediction with the true
   actions looked no better than with no actions — the model seemed to ignore them. Root cause
   (found by instrumenting, not guessing): I was rolling **stochastic samples**; the sampling
   noise on unpredictable dims swamped the signal. Rolling the **prior mean** made action-
   conditioning separate cleanly. *Lesson: a prediction metric must be deterministic.*

3. **Model exploitation on a real sim.** On MetaDrive the policy looked great *in imagination*
   (high imagined return) but drove off-road in reality — it exploited the world model's errors.
   The iterated loop helped but didn't solve it; the actor collapsed into a **saturated tanh
   corner** under an unnormalized value gradient + weak reward. *This is the canonical model-based
   failure mode, demonstrated on a real sim.*

4. **A fix that backfired (judgment).** I tried the DreamerV3 stabilizers (return normalization +
   symlog) in simplified form. `symexp` of an unbounded head **NaN'd**, and per-batch return
   scaling made the loop **collapse on the toy** — breaking my "keep the toy green" bar. I
   **reverted** rather than ship a regression, and documented that the proper fix needs *bounded
   two-hot heads + EMA normalization*. *Lesson: knowing when not to ship.*

## 6. Extensions (breadth)

- **Dynamics ablation.** The recurrence is isolated behind a one-method `Recurrence` interface
  (`models/recurrence.py`), so `cfg.dynamics` swaps **GRU** ↔ a minimal **Mamba-style selective
  SSM** and re-runs the *same* eval harness (`scripts/ablate_dynamics.py`, exp 006). Results
  (state mode, identical config, only the recurrence differs):

  | metric | rssm (GRU) | mamba (SSM) | note |
  |---|--:|--:|---|
  | reward (held-out) | 0.001 | 0.004 | both ≪ 0.1 → learned |
  | open-loop gap (no-action − model) | **0.224** | 0.190 | both use actions; GRU edges it |
  | closed-loop return | 96.8 | 94.9 | random ≈ −50 |
  | throttle / steer | 1.00 / 0.01 | 1.00 / −0.02 | both reach the optimum |

  Both learn, beat the no-action baseline, and drive optimally; GRU edges the SSM but the gap is
  **a wash on this one-step-dynamics toy** — the value is the controlled-swap machinery, which
  would actually separate on a long-horizon / partial-observability task.
- **Visual mode.** A CNN encoder + transposed-CNN decoder make the model learn from **pixels**;
  the same RSSM/ELBO/actor-critic are untouched (obs-type-agnostic core). It learns to **dream
  video** — roll the prior under chosen actions and decode frames (`scripts/dream_video.py`).
- **Real sim.** MetaDrive behind the same env contract (`envs/metadrive_env.py`); the world model
  trains on real lidar+ego state, and the sim renders to a watchable top-down video.

## 7. Results & honest limitations

- ✅ Toy: world model trains (healthy KL, no collapse); open-loop beats the no-action baseline;
  a policy trained **only in imagination** drives optimally (return ≈ 95 vs ≈ −51 random).
- ✅ Ablation, pixels (recon → ~0, crisp dream video), real-sim *learning* (recon 245→0.17).
- ⚠️ Real-sim **control**: the simplified stack does not yet drive MetaDrive well (model
  exploitation + corner-collapse). Closing that needs the full DreamerV3 stabilizers + GPU-scale
  compute. Documented faithfully in `experiments/010–012`.

The honesty is a feature: the project maps exactly where a simplified, from-scratch Dreamer
succeeds and where the production tricks (two-hot heads, EMA return norm, lots of compute) become
necessary.

## 8. Anticipated interview questions (crisp answers)

- **"What's a world model?"** A learned differentiable simulator — encode obs to a latent, learn
  latent dynamics, predict obs/reward/continue, so you can imagine futures and train a policy in
  them.
- **"Why train the policy in imagination?"** Sample efficiency: the world model is differentiable,
  so you backprop the imagined return into the actor with zero env steps. Risk: if the model is
  wrong, the actor exploits its errors — which is why closed-loop eval and iterated data
  collection exist.
- **"Where does the KL come from?"** It's the ELBO's regularizer; minimizing it trains the prior
  to predict the next latent without the observation — the thing that makes imagination work.
- **"Posterior collapse?"** KL→0, latent unused, recon stalls; detect via KL-vs-recon; guard with
  free bits.
- **"Open-loop vs closed-loop?"** Prediction (dynamics) vs control (outcomes); good open-loop +
  bad closed-loop = distribution shift / compounding error / model exploitation.
- **"GRU vs Mamba here?"** Same Markov interface; on a one-step-dynamics toy it's a wash — the
  point is the controlled ablation harness; SSMs would matter on long-horizon/partial-obs tasks.
- **"What broke and how did you find it?"** The action-timing and sample-vs-mean stories in §5 —
  found by generalization tests and instrumentation, not guessing.

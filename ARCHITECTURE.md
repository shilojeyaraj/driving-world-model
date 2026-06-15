# Architecture & walkthrough

This is the "you derive it, you own it" document. It explains the whole v1 world-model loop:
the math (ELBO, λ-returns), the conventions that everything depends on (action timing), the
failure modes we hit and how we caught them, and direct answers to every `CONCEPTS.md`
question. Code references are `file:concept`, not line numbers, so they don't rot.

---

## 1. The picture

```
  REAL DATA  (world-model learning)              IMAGINATION  (behavior learning, no env)
  obs ─Encoder─► e_t                              start (h,z) from real obs  (DETACHED)
        │                                                │
  a_t ──┤                                          a_i = Actor(feat_i)   (reparam sample)
        ▼                                                ▼
   RSSM.observe ─► h_t,z_t  (posterior q(z|h,e))    RSSM.img_step ─► h,z  (prior p(z|h), no obs)
        │                                                │
   Decoder ─► ô_t, r̂_t, ĉ_t                          Decoder ─► r̂_i, ĉ_i ;  Critic ─► v_i
        │                                                │
   loss = recon + reward + cont + β·KL              λ-returns ─► actor loss (−return), critic loss
```

Three swappable slots — Encoder (`models/encoder.py`), **RSSM dynamics** (`models/rssm.py`,
the ablation axis), Decoder (`models/decoder.py`) — plus the Actor-Critic
(`models/actor_critic.py`) trained inside the RSSM's imagination.

**State = (h, z).** `h` is a deterministic GRU memory (a clean, sample-free channel for
long-range info). `z` is a small stochastic latent for what's uncertain this step. Two heads
read `h`: the **prior** `p(z|h)` (guess before seeing the obs — used in imagination) and the
**posterior** `q(z|h,e)` (corrected by the encoded obs). `feat = [h; z]` is what every output
head and the policy consume.

---

## 2. The ELBO, derived by hand

This is the heart of `models/world_model.py:assemble_loss` and `models/rssm.py`.

**Setup.** Per timestep we have observation `o_t`, reward `r_t`, continue `c_t = 1−done_t`,
and action `a_t`. Write `x_t = (o_t, r_t, c_t)`. Latents are the stochastic `z_{1:T}`; the
deterministic `h_t` is *computed*, not sampled:

```
h_t = f(h_{t-1}, z_{t-1}, a_{t-1})          (GRU recurrence — deterministic)
```

**Generative model** (the "prior" / simulator):

```
z_t ~ p(z_t | h_t)                          prior  (a.k.a. transition / dynamics predictor)
x_t ~ p(o_t,r_t,c_t | h_t, z_t)             likelihood heads (decoder)
```

Because `h_t` is a deterministic function of the past, both the prior and the likelihood
factorize over time given the `h_t` sequence.

**Goal.** Maximize the data log-likelihood `log p(x_{1:T} | a_{1:T})`. Marginalizing over
`z_{1:T}` is intractable, so introduce a **variational posterior** that filters using the
encoded observation `e_t = Encoder(o_t)`:

```
q(z_{1:T} | o, a) = Π_t q(z_t | h_t, e_t)
```

**Derivation** (standard ELBO; Jensen's inequality):

```
log p(x|a) = log ∫ p(x, z | a) dz
           = log ∫ q(z|·) · [ p(x, z | a) / q(z|·) ] dz
          ≥  E_q[ log p(x, z | a) − log q(z|·) ]                         (Jensen)
           = E_q[ log p(x | z, a) ]  −  KL( q(z|·) ‖ p(z|a) )
```

Now factorize both terms over time (legal because everything is conditioned on the
deterministic `h_t`):

```
log p(x|a) ≥  Σ_t E_q[ log p(o_t, r_t, c_t | h_t, z_t) ]      ← reconstruction (fit the data)
            − Σ_t E_q[ KL( q(z_t|h_t,e_t) ‖ p(z_t|h_t) ) ]    ← regularize posterior → prior
```

**That second term is where `KL(posterior ‖ prior)` comes from.** It is the price of using an
obs-informed posterior instead of the blind prior. Minimizing it drags the prior toward the
posterior — i.e. **teaches the prior to predict the latent the observation would have implied**,
without seeing the observation. That is exactly what makes imagination (prior-only rollout)
work.

**From bound to loss.** Choose simple likelihoods (v1):
- `o_t`, `r_t`: Gaussian with unit variance ⇒ `−log p = ½‖·‖² + const` ⇒ **MSE**.
- `c_t`: Bernoulli ⇒ **BCE-with-logits**.

Negate the ELBO to get a minimization objective (per step, then mean over batch & time):

```
loss = MSE(ô, o)  +  MSE(r̂, r)  +  BCE(ĉ_logit, 1−done)  +  β · KL(q ‖ p)
```

**Free bits.** We replace the KL with `max(KL, free_bits)` (summed over latent dims, per
step). Below the floor we pay nothing, which removes the gradient pressure that would otherwise
push the posterior to match the prior and ignore the obs (**posterior collapse**). `β =
cfg.kl_scale`. (DreamerV3 uses two-sided KL balancing instead — see `spec §12`.)

Code: `models/world_model.py:assemble_loss` (the `_gaussian_kl` closed form is
`log(σ_p/σ_q) + (σ_q² + (μ_q−μ_p)²)/(2σ_p²) − ½`, summed over the latent dim).

---

## 3. Action timing (the off-by-one everything depends on)

Convention (`spec §3`): a buffer entry `t` means *you saw `o_t`, took `a_t`, then received
`r_t` and `done_t`*. The recurrence consumes the **previous** action:

```
h_t = GRU( h_{t-1}, [ z_{t-1} , a_{t-1} ] )           a_{-1} = 0 at the first step
```

So `feat_t = [h_t; z_t]` carries `a_{t-1}` — **not** `a_t`.

- **Observation** aligns directly: `feat_t` (whose posterior consumed `e_t = Enc(o_t)`)
  predicts `o_t`. No shift.
- **Reward / continue** are *transition* quantities. In this env `r_t = R(s_t, a_t)` depends
  on `a_t`, which is **not** in `feat_t`. `feat_t` can only predict the reward/continue
  produced by `a_{t-1}` — i.e. buffer index `t−1`. So we **shift by one**: predictions at
  `t = 1..T−1` are matched to targets at `t = 0..T−2` (`models/world_model.py:assemble_loss`).

We discovered this the hard way (`experiments/002`): with no shift, held-out reward MSE sat at
`Var(reward) ≈ 0.42` — the reward head could only predict the mean. After the shift it dropped
to `0.007`. The same alignment makes imagination consistent: there, the reward for action
`a_i` is read from `feat_{i+1}` (the state that consumed `a_i`) — see
`training/train_behavior.py:imagine_rollout` and `eval/closed_loop.py`.

> ⚠️ Note vs `spec §3`: the spec says "feat_t predicts reward_t," which is only correct if
> "reward_t" means the reward *arriving at* state t (the Dreamer convention). The buffer gives
> gym-convention `R(s_t, a_t)`, so it must be shifted. This is the one place the implementation
> deliberately departs from a literal reading of the spec.

---

## 4. Imagination & λ-returns (behavior learning)

`training/train_behavior.py`. The world model is **frozen** (`requires_grad_(False)`); the
policy learns with **zero env steps**.

1. **Start states.** Encode + `observe` a real batch; take every per-step `(h,z)` as an
   imagination start state, **detached** (so behavior grads never leak into the world model).
2. **Dream.** For `i = 0..H−1`: `a_i = Actor(feat_i)` (reparameterized **sample** — gradients
   flow through the sampler), then `img_step` (prior, **sampled**) → next state; read
   `r̂_i, ĉ_i` from `feat_{i+1}`; read `v_i = Critic(feat_i)`.
3. **λ-returns** (bootstrap with `v_H`):

```
V^λ_t = r_t + γ·c_t·[ (1−λ)·v_{t+1} + λ·V^λ_{t+1} ] ,      V^λ_H = v_H
```

   `λ=0` → one-step TD `r_t + γc_t v_{t+1}` (low variance, biased by the critic). `λ=1` →
   Monte-Carlo discounted sum (unbiased, high variance). `λ=0.95` interpolates.
   (`lambda_returns`, verified against a hand-worked example in `tests/test_actor_critic.py`.)

4. **Losses.** `actor_loss = −mean(V^λ_t) − entropy_coef·mean(entropy)` — a **value gradient**:
   maximizing the return backpropagates through the reward head and dynamics into the actor.
   `critic_loss = mean( (v_t − stop_grad(V^λ_t))² )`.

**Gradient routing (subtle but essential).** Start states detached; the critic reads
*detached* feats (its gradient touches only critic params); the actor's bootstrap **values are
detached** so the actor gradient flows only through `rewards → dynamics → actions`. See
`training/train_behavior.py:behavior_losses`.

### 4a. The iterated Dreamer loop — making it actually drive

`train_behavior.train_behavior` is **single-shot**: collect once (random) → train WM → train
policy. That's enough for the toy (state-independent optimum, dense reward), but it **fails on a
real sim**: a policy trained inside a model built only from random, crash-prone data **exploits
the model's errors** — confident in imagination, off-road in reality (MetaDrive: imagined return
4.4 but closed-loop −2.9, *worse than random*; `experiments/010`).

`training/dreamer_loop.py:dreamer_train` is the real algorithm — seed random, then repeat:

```
collect WITH the current policy (+ exploration noise)  → append to the buffer
train the world model on the growing buffer
train the policy in imagination on the frozen world model
```

Collecting *with the policy* grounds the world model in the states the policy actually visits,
closing the imagination-vs-reality gap. WM `requires_grad_` is toggled between the WM-train and
behavior phases (frozen during behavior so gradients flow to the actor but don't update the WM).
Verified to drive the toy end-to-end (return 87.6 vs −48 random; `tests/test_dreamer_loop.py`).

---

## 5. Two rollout modes: sample vs mean (the eval bug we fixed)

Same `img_step`, opposite `sample` flag — on purpose:

| Use | `sample` | Why |
|-----|----------|-----|
| Behavior training (imagination) | `True` (reparam) | policy must explore; gradients flow through the sampler |
| Open-loop **prediction** eval | `False` (prior mean) | a prediction metric must be deterministic |

We learned this the hard way (`experiments/003`). With *sampled* open-loop rollouts, sampling
noise on the 34 unpredictable obs dims swamped the action signal and true-action prediction
looked no better than no-action — a **measurement bug masquerading as a modeling failure**.
Switching to the prior mean made true-action clearly beat no-action (`eval/open_loop.py`).

---

## 6. Posterior collapse — what it is, how to detect it

Collapse = the posterior stops using the latent and just matches the prior: `KL → 0`, `z`
carries no information, reconstruction stalls. **Detect it from the curves:** watch KL and
recon *together*. Healthy = KL sits meaningfully above 0 while recon improves (ours rose
`0.16 → 0.70`). Collapse = KL flatlines near 0 while recon plateaus high. `assemble_loss` logs
the **raw KL** (pre-free-bits) precisely so you can watch this; free bits is the guard.

---

## 7. Known-answer sanity (why DummyEnv)

DummyEnv: `reward = throttle − |steer|`, `pos` integrates `throttle`, other 34 dims are noise.
So the truth is known:
- Recon **on `pos`** + reward should drop; recon on the 34 noise dims **plateaus at the noise
  variance** (the prior can't predict noise — expected, `spec §9`). Judge the model on
  `pos` + reward + open-loop, not raw recon.
- Optimal policy: `throttle = +1, steer = 0`. Closed-loop confirmed `throttle=1.000,
  steer=−0.009`, return `94.8` vs `−51` random (`experiments/005`).

---

## 8. Answers to every CONCEPTS.md question

**Encoder — why might a ViT need more data than a CNN for the same reconstruction?**
A CNN bakes in locality + translation-equivariance (weight sharing) — priors that are *correct*
for images, so it doesn't spend data learning them. A ViT starts with global attention and
almost no spatial prior; it must *learn* that structure from data, so it's data-hungry and only
wins at scale. (State mode has no spatial structure, hence a plain MLP.)

**RSSM — where does `KL(posterior ‖ prior)` come from, and what does the prior do at
imagination time?** It's the regularization term of the ELBO (§2): the cost of using the
obs-informed posterior `q(z|h,e)` instead of the blind prior `p(z|h)`. Minimizing it teaches
the prior to predict the next latent without the observation. At imagination time there *is* no
observation, so `imagine`/`img_step` sample `z` from that prior — only good because the KL
trained it to predict.

**Decoder — why predict reward and continue, not just the obs?** Because the world model must
be a self-contained simulator the policy can dream inside — there's no env to query. Reward lets
it compute returns; continue (`1−done`) tells imagination when episodes end so returns stop
discounting past termination. Drop the continue head and imagined rollouts run past `done`
forever, inflating returns with impossible reward. The obs head is the pressure that forces the
latent to actually encode the observation.

**World model — what is posterior collapse and how do you detect it?** See §6: `KL→0` with
stalled recon; detect by watching KL vs recon together; free bits guards it.

**Actor-Critic — why can you train the policy on imagined rollouts with zero env steps, and
what's the failure mode if the model is wrong?** Because the world model is a *differentiable
simulator*; maximizing the imagined λ-return backprops through reward + dynamics into the actor
(value gradient) — no env needed (sample efficiency). Failure mode: if the model is wrong, the
actor learns to **exploit its errors** — it optimizes a fantasy that doesn't transfer. That's
why closed-loop eval exists. Mitigations: detached start states, entropy bonus, grad clipping.

**Replay buffer — why sample contiguous sequences, not i.i.d. transitions?** Because the RSSM
learns *dynamics* — `h_t` depends on the whole history. You need contiguous `(o,a,r)` sequences
to roll the recurrence and train the prior to predict forward; i.i.d. transitions destroy the
temporal structure the model exists to capture.

**Envs — what's in a "state" obs vs an "image" obs, and why does that change compute cost?**
`state` is a low-dim vector (lidar + ego) → small MLPs → trains on CPU. `image` is `(C,H,W)`
pixels → conv encoder + transposed-conv decoder → needs a GPU. Same interface (`envs/base.py`),
very different FLOPs.

**Open-loop — why does action-conditioning matter for this metric?** Open-loop rolls the prior
with no observations — the only thing that makes the true future differ from "nothing happens"
is the action sequence (`pos` integrates throttle). If true-action prediction doesn't beat
no-action, the model didn't learn dynamics — it learned to autoplay the average future.

**Closed-loop — great open-loop but bad closed-loop, what went wrong?** Distribution shift +
compounding errors: the policy drives itself into states the data never covered (where the
model is wrong), and per-step errors compound. Or the actor exploited model errors (§4 failure
mode). Open-loop only ever tests in-distribution prediction with true actions; closed-loop
tests the policy's *own* choices and *own* states.

---

## 9. How to run

```bash
pip install -r requirements.txt
python scripts/smoke_test.py                 # plumbing sanity (no models)
pytest -m "not slow"                         # fast unit/contract tests (~9s)
pytest                                        # everything incl. milestone gates (~minutes)
python -m training.train_world_model         # train the world model
python -m training.train_behavior            # collect → train WM → train policy in imagination
```

Checkpoints: `utils.save_checkpoint(path, wm, actor, critic, cfg)` /
`utils.load_models(path)` (`spec §6`).

---

## 10. What v1 deliberately leaves out (designed-for)

Image/pixel mode on GPU (CNN/transposed-CNN paths stubbed with explicit "later phase" errors),
MetaDrive, and the Transformer/Mamba **dynamics ablation** (the `img_step`/`obs_step` interface
is the seam). DreamerV3 upgrades (categorical latents, symlog + two-hot heads, two-sided KL
balancing, return normalization) are catalogued in `spec §12`.

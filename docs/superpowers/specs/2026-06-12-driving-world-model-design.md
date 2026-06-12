# Driving World Model — Design Spec (v1: simplified Dreamer, state-mode)

Date: 2026-06-12
Owner: Shilo
Implementer: Claude (writes all code) — Shilo learns the code + architecture as it's built.

---

## 1. Goal & non-goals

**Goal.** Implement the *complete* Dreamer-style world-model loop end-to-end, runnable on a
laptop CPU in `state` observation mode against the `DummyDrivingEnv`. "Complete" means:

1. A world model (encoder → RSSM → decoder) that **trains** via an ELBO.
2. **Open-loop** prediction that beats a no-action baseline (proves it learned *dynamics*).
3. An **actor-critic** trained purely inside the model's imagined rollouts (no env steps).
4. **Closed-loop** evaluation showing the policy actually "drives" the toy env.
5. A walkthrough doc + tests so the whole thing is understandable and verifiable.

**Fidelity.** Simplified-but-correct (Gaussian latents, MSE/Gaussian + BCE heads, single-
direction KL with free-bits). Every simplification is flagged in code comments with what
real **DreamerV3** does instead and why (see §12).

**Non-goals (v1).** Real driving (MetaDrive), image/pixel runs, GPU/Kaggle, and the
Transformer/Mamba dynamics ablation. The code is *structured* so these drop in later
(image encoder/decoder are written but validated later on GPU; an RSSM recurrence seam is
left open), but they are not part of the v1 definition of done.

**Working mode.** Claude implements all code. Each component ships with "why" comments,
is narrated in chat as it's written, and is captured in `ARCHITECTURE.md`.

---

## 2. Architecture & data flow

```
  REAL DATA (world-model learning)            IMAGINATION (behavior learning, no env)
  obs ──Encoder──► e_t                        start feat (h,z) from real obs (detached)
        │                                            │
  a_t ──┤                                       a_t = Actor(feat)   (reparameterized)
        ▼                                            ▼
   RSSM.observe ──► h_t, z_t   (posterior:      RSSM.imagine ──► h_t, z_t  (prior: NO obs)
        │            uses e_t)                       │
   Decoder ─► obŝ_t, reward̂_t, cont̂_t           Decoder ─► reward̂_t, cont̂_t
        │                                            │
   ELBO = recon + reward + cont + β·KL          Critic(feat) ─► v_t ─► λ-returns
                                                     │
                                               actor loss = −λ-returns − α·entropy
                                               critic loss = (v_t − sg(λ-return))²
```

**The load-bearing idea.** `KL(posterior ‖ prior)` trains the *prior* `p(z_t | h_t)` to
predict the next latent **without** seeing the observation. Once the prior is good, the
model can roll forward with no inputs (`imagine`), and the policy can be trained in that
dream. If you delete the KL term, the prior never learns to predict and imagination is junk.

---

## 3. Conventions (fix these once; everything depends on them)

- **Tensors.** Batches are `(B, T, ...)`. `B` = batch_size, `T` = seq_len. The encoder/
  decoder operate per-timestep (we flatten to `(B*T, ...)`, apply, reshape back).
- **Obs types.** `state`: 1-D float32 vector of `cfg.state_dim`. `image`: `(C,H,W)` float32
  in `[0,1]`. The env contract (`envs/base.py`) already guarantees this.
- **Action timing (critical).** A batch entry means: *at step t you saw `obs[t]`, took
  `action[t]`, received `reward[t]` and `done[t]`.* The recurrence consumes the **previous**
  action: `h_t = GRU(h_{t-1}, [z_{t-1}, a_{t-1}])`. For the first step `a_{-1}=0` and
  `(h,z)` come from `initial_state`. Therefore in `observe`, `feat_t` predicts `obs_t`,
  `reward_t`, `cont_t` using actions only up to `a_{t-1}`; the last action `a_{T-1}` is
  unused by `observe`.
- **Imagination timing.** `imagine` is "actor acts, then world steps": from state `s_i` the
  action `a_i` produces `s_{i+1}`. We implement this with single-step primitives so the two
  loops share code (see §4.2).
- **Continue flag.** We predict `cont = 1 − done`. Discount used in returns is
  `d_t = gamma · cont_t`.

---

## 4. Component specs

### 4.1 Encoder — `models/encoder.py`
- `__init__(cfg)` sets `self.embed_dim`.
- `state`: MLP `state_dim → hidden → hidden → embed_dim` (ELU/SiLU activations), `embed_dim ≈ 256`.
- `image`: Dreamer-style CNN (4 conv layers, stride 2, channels ~[32,64,128,256]) → flatten
  → linear to `embed_dim`. **Written now, validated on GPU later.**
- `forward(obs) -> (batch, embed_dim)`. Accepts `(N, *obs_shape)`; caller flattens `(B,T)`.

### 4.2 RSSM — `models/rssm.py` (the core)
State is a tuple `(h, z)`: `h ∈ R^{deter_dim}` (deterministic, GRU hidden), `z ∈ R^{stoch_dim}`
(stochastic latent). Gaussian prior/posterior with reparameterization.

Submodules:
- input projector: `Linear([z; a]) → hidden` then into `GRUCell(hidden, deter_dim)`.
- prior head: `MLP(h) → (mean, std)` with `std = softplus(·) + min_std`.
- posterior head: `MLP([h; e]) → (mean, std)`.

Primitives (shared by both loops):
- `initial_state(batch, device) -> (h0, z0)` zeros.
- `img_step(state, action) -> (next_state, prior_stats)`: `h = GRU([z,a], h)`, `prior =
  head(h)`, `z ~ prior` (reparam). Prior path, no observation.
- `obs_step(state, prev_action, embed) -> (next_state, prior_stats, post_stats)`:
  `h = GRU([z,prev_action], h)`, `prior = prior_head(h)`, `post = post_head([h,e])`,
  `z ~ post` (reparam).

Public methods:
- `observe(embeds (B,T,E), actions (B,T,A), state) -> dict`: loop `obs_step` with the
  previous-action convention. Returns per-step `prior` (means, stds), `post` (means, stds),
  `feat = [h; z]` of shape `(B,T,deter+stoch)`, and final state.
- `imagine(actions (B,H,A), state) -> dict`: loop `img_step`. Returns per-step `feat` and
  `prior`. (Used by open-loop eval, which supplies *true* actions. Behavior training instead
  calls `img_step` in a loop with on-the-fly actor actions — see §4.6.)

**Ablation seam:** the GRU recurrence is isolated so a Transformer/Mamba block can replace it
later behind the same `img_step`/`obs_step` interface. Not implemented in v1.

### 4.3 Decoder — `models/decoder.py`
- `__init__(cfg, feat_dim)`, `feat_dim = deter_dim + stoch_dim`.
- obs head: `state` → MLP `feat → state_dim` (Gaussian mean, unit var → MSE). `image` →
  dense + transposed-CNN mirror of the encoder (validated later).
- reward head: `MLP(feat) → scalar` (Gaussian/MSE).
- continue head: `MLP(feat) → logit` (Bernoulli; BCE-with-logits).
- `forward(feat) -> {"obs", "reward", "cont_logit"}`.

### 4.4 World model + ELBO — `models/world_model.py`
`WorldModel` already wires Encoder+RSSM+Decoder. Implement `assemble_loss(batch)`:

1. Encode `obs (B,T,·)` → `embeds (B,T,E)`.
2. `out = rssm.observe(embeds, actions, initial_state)`.
3. Decode `out.feat` → `obŝ, reward̂, cont̂`.
4. Losses (all averaged over `B,T`):
   - `recon_nll  = MSE(obŝ, obs)` summed over obs dims. *(Note: DummyEnv's noise dims are
     unpredictable by construction; recon plateaus at the noise variance — expected.)*
   - `reward_nll = MSE(reward̂, reward)`.
   - `cont_nll   = BCE_with_logits(cont̂_logit, 1 − done)`.
   - `kl = KL( post ‖ prior )` per step, summed over `stoch_dim`, with **free bits**:
     `kl_loss = mean( max(kl, cfg.free_bits) )`. *(Single-direction KL; v3 uses two-sided
     balancing — noted.)*
   - `loss = recon_nll + reward_nll + cont_nll + cfg.kl_scale · kl_loss`.
5. Return `(loss, metrics)` where metrics include each term **and the raw KL** so we can
   watch for posterior collapse (KL → 0 while recon stalls).

ELBO derivation (for `ARCHITECTURE.md`): for a sequential latent model with deterministic
`h_t`, `log p(o,r,c | a) ≥ Σ_t E_q[log p(o_t,r_t,c_t | h_t,z_t)] − Σ_t KL(q(z_t|h_t,e_t) ‖
p(z_t|h_t))`. Maximizing the ELBO = minimizing the loss above.

### 4.5 Actor & Critic — `models/actor_critic.py`
- `Actor(cfg, feat_dim, action_dim)`: `MLP(feat) → (mean, std)`; action = `tanh(mean + std·ε)`
  (Tanh-Normal, reparameterized so gradients flow through the dynamics). Exposes a sample +
  an entropy estimate. Actions live in `[-1,1]^action_dim` (matches env).
- `Critic(cfg, feat_dim)`: `MLP(feat) → scalar` value.
- Optional EMA **target critic** for stability (flagged; simple version regresses to returns
  directly).

### 4.6 Behavior training — `training/train_behavior.py`
1. Sample a real batch; encode + `observe`; take per-step feats as imagination **start
   states** (detached from the world-model graph so behavior grads don't flow into it).
2. Roll `cfg.imagine_horizon` steps: at each step `a_i = Actor(feat_i)` (reparam), then
   `img_step` → next state; decode `reward̂_i`, `cont̂_i`; record `value_i = Critic(feat_i)`.
3. **λ-returns** (bootstrap with `v_H`): `V^λ_t = r_t + γ·c_t·[(1−λ)·v_{t+1} + λ·V^λ_{t+1}]`,
   `V^λ_H = v_H`.
4. `actor_loss = −mean(V^λ_t) − cfg.entropy_coef · mean(entropy)` (value-gradient objective:
   backprops returns through dynamics + reward heads).
5. `critic_loss = mean( (value_t − stop_grad(V^λ_t))² )`.
6. Separate Adam optimizers (`actor_lr`, `critic_lr`); grad clip. World-model params frozen
   here.

### 4.7 Open-loop eval — `eval/open_loop.py`
`open_loop_eval(model, batch, context=5, horizon=20)`:
1. Encode first `context` frames, `observe` → state.
2. `imagine` `horizon` steps using the **true** actions from the batch.
3. Decode predicted obs; compute error vs ground truth **per horizon step**.
4. Return `error[h]` (array over horizon) for the action-conditioned model **and** a
   no-action / repeat-last baseline, so we can show action-conditioning lowers error.

### 4.8 Closed-loop eval — `eval/closed_loop.py`
`closed_loop_eval(actor, world_model, env, episodes=10)`:
- Maintain RSSM state across env steps: each step encode `obs` → `obs_step` (posterior) →
  `feat` → `action = Actor(feat)` (mean, deterministic) → `env.step`.
- Aggregate **mean episode return** and **action stats** (mean steer/throttle).
- Baselines: random policy and the data-collection policy.

---

## 5. Config additions — `config.py` (marked "modify freely")
Add: `gamma=0.99`, `lambda_=0.95`, `entropy_coef=1e-3`, `actor_lr=8e-5`, `critic_lr=8e-5`,
`hidden_dim=256`, `min_std=0.1`. Keep existing fields.

## 6. Checkpointing
Save `{world_model, actor, critic, config}` state_dicts to `runs/<name>/ckpt.pt`; closed-loop
eval loads from there. Small helper in a `models/__init__.py` or a tiny `utils.py`.

## 7. Tests — `tests/` (pytest; add `pytest` as a dev dep)
- `test_shapes.py`: encoder/decoder/RSSM I/O shapes for both obs types.
- `test_rssm.py`: `observe` and `imagine` return correct shapes; `imagine` with fixed actions
  is deterministic given a fixed seed for the deterministic path.
- `test_world_model.py`: **overfit one batch** → loss drops sharply; **KL-not-collapsed**
  guard (KL stays above a floor while recon improves).
- `test_actor_critic.py`: λ-return computation matches a hand-worked small example; actor
  output is in `[-1,1]`.
- `test_smoke.py`: wraps the existing smoke test.

## 8. Walkthrough doc — `ARCHITECTURE.md`
Covers: the architecture diagram, the ELBO derived by hand, the action-timing convention,
the imagination/λ-returns math, posterior-collapse detection, and **direct answers to every
`CONCEPTS.md` question**. Updated as each phase lands.

## 9. Success criteria (DummyEnv has a known-optimal answer)
`reward = throttle − |steer|`; `pos` integrates `throttle`; other state dims are noise.
- ✅ World model: recon (on `pos`) + reward losses drop; KL healthy (not ~0); **open-loop
  with true actions beats the no-action baseline**.
- ✅ Policy: closed-loop actor converges to **throttle ≈ +1, steer ≈ 0**; mean return
  exceeds the random-policy baseline.

## 10. Build phases (each ends runnable + a checkpoint we discuss)
1. Encoder + Decoder (state) → overfit-a-batch reconstruction sanity.
2. RSSM + `assemble_loss` (ELBO) → world model trains; watch KL vs recon.
3. `open_loop.py` → action-conditioned prediction beats baseline.
4. Actor-Critic + `train_behavior` → policy learns in imagination.
5. `closed_loop.py` → actor drives toy env (throttle→1, steer→0, return↑).
6. `ARCHITECTURE.md` + tests + polish.

## 11. Risks & mitigations
- **Posterior collapse** (KL→0, blurry recon): free-bits + KL metric in the loss log; the
  test guards it.
- **Imagination instability** (actor exploits a wrong model): start-state detach, entropy
  bonus, grad clipping, optional target critic.
- **DummyEnv noise floods recon**: expected; we judge the world model on `pos`+reward+open-
  loop, not raw recon, and document this.
- **Action-timing off-by-one**: pinned in §3 and covered by `test_rssm.py`.

## 12. DreamerV3 differences (the inline "notes")
| Area | This v1 (simplified) | DreamerV3 |
|------|----------------------|-----------|
| Latent `z` | diagonal Gaussian + reparam | categorical (32×32) + straight-through |
| Obs/reward | MSE / Gaussian | symlog + two-hot (discretized regression) |
| KL | single-direction + free-bits | two-sided KL balancing |
| Returns | λ-returns, value-grad actor | + percentile return normalization |
| Critic | MSE regression (optional EMA target) | two-hot value + EMA target + regularizers |

## 13. Future extensions (out of scope for v1, designed-for)
- Image mode on Kaggle GPU (CNN/transposed-CNN paths already written).
- MetaDrive env (wrapper exists; verify config keys/obs shapes per install).
- Dynamics ablation: swap GRU → Transformer/Mamba behind the RSSM step interface.
- Optional DreamerV3 upgrades from the §12 table, added as labeled diffs.

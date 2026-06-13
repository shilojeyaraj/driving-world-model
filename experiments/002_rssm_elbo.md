# Experiment 002 -- RSSM + ELBO (the world model trains)

**Date:** 2026-06-12
**Component / change:** `models/rssm.py` (GRU recurrence, Gaussian prior/posterior with
reparameterization, `observe`/`imagine`) + `models/world_model.py::assemble_loss` (the ELBO
with free-bits KL). Phase 2 of the v1 spec. Built test-first.

## Hypothesis (write BEFORE running)
With a posterior that sees the observation, the world model should fit a batch easily; the
load-bearing question is whether the KL stays healthy (latent used, prior learning to predict)
and whether the **reward** -- which depends on the action -- is learned as a function of the
action rather than memorized. Recon on the 34 noise dims should plateau at the noise variance
once we stop letting the posterior memorize (fresh batches each step).

## Setup
- config diffs: deter_dim/stoch_dim/hidden_dim small for fast CPU tests (32/8/32) in tests;
  64/16/64 for the end-to-end run. free_bits=1.0, kl_scale=1.0.
- obs_type / env / device: state / dummy / cpu
- what I implemented: RSSM (input_proj -> GRUCell; prior_head p(z|h); post_head q(z|h,e);
  reparmeterized sampling; previous-action timing in `observe`), and the ELBO loss.

## Result
- Tests (10 total) green: RSSM observe/imagine shapes + reproducibility; assemble_loss
  runs/backprops; overfit-one-batch drops loss+recon with KL off the floor.
- End-to-end `train_world_model` (400 steps): loss 3.33->1.77, reward 0.93->0.076,
  cont ~0, recon ~0.75->0.6 (noise floor), **kl 0.16->0.70 (rising, healthy -- NOT collapsing)**.

## Hypothesis vs. reality -- THE BUG (action-timing off-by-one, spec risk #1)
My first `assemble_loss` matched `reward_hat[t]` to `reward[t]`. The overfit-one-batch test
PASSED (reward -> 0.06) -- but for the WRONG reason: with 34 random noise dims, the posterior
memorizes each step's reward using the noise as a fingerprint. I caught it by writing
`test_reward_prediction_generalizes`: train on one data stream, measure reward on a DISJOINT
held-out stream. Held-out reward MSE was **0.4458 ~ Var(reward)=0.42** -- i.e. the head could
only predict the mean reward.

Cause: `feat_t` carries `a_{t-1}` (recurrence consumes the previous action), but in this env
`reward[t] = R(s_t, a_t)` depends on `a_t`, which is NOT in `feat_t`. Reward/continue are
*transition* quantities and must be aligned to the action that lives in the feature. Fix:
shift one step -- match predictions at t=1..T-1 to targets at t=0..T-2 (feat_t predicts the
reward/continue produced by a_{t-1}). After the fix, held-out reward MSE = **0.0072**.

Note vs spec §3: §3 says "feat_t predicts reward_t". That is only consistent if "reward_t"
means the reward of the transition *arriving* at state t (Dreamer convention). The buffer
provides `reward_t = R(s_t, a_t)` (gym convention), so feeding it unshifted is the bug. The
shift reconciles the two and keeps observe-training consistent with imagine-behavior (where
the reward for a_i is read from feat_{i+1}).

## Failures / debugging
- Overfit test gave false confidence (memorization). Lesson: an overfit-a-batch test proves
  optimization works, NOT that a quantity is learned for the right reason. Generalization to a
  held-out stream is the test that catches action-timing/representation bugs.
- Minor: a test helper passed `seq_len` both positionally and via overrides -> TypeError; fixed.

## One-line takeaway (the interview sentence)
> The reward head can only learn reward-as-a-function-of-action if the reward target is aligned
> to the action that actually lives in the feature; an overfit-a-batch test hides this because
> high-dim noise lets the posterior memorize -- only held-out generalization exposes the
> off-by-one.

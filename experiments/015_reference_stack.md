# Experiment 015 -- GF3: reference stack (WM + BC actor + eval critic on IDM)

**Date:** 2026-06-16
**Component / change:** `training/train_reference.py` -- `collect_idm` (MetaDrive IDM expert),
`bc_actor` (behavior-clone the expert), `eval_critic` (policy-evaluate it via lambda-returns on
recorded rewards), `train_reference` (orchestrate -> save_checkpoint). Phase GF3. Built test-first.

## Hypothesis (write BEFORE running)
The feedback engine needs a GOOD, feat-queryable reference. IDM drives well but only acts on the
live sim, so we BC it into an Actor (feat -> action) and policy-evaluate a Critic on IDM
trajectories. The BC + critic steps are pure given a buffer, so they're testable on DummyEnv with
a KNOWN policy without MetaDrive; only IDM collection needs the sim.

## Setup
- `collect_idm`: MetaDrive `agent_policy=IDMPolicy`; the applied action is `info["action"]`
  (probe confirmed: `[steering, acceleration]`), recorded as the BC target.
- `bc_actor`: MSE(actor(feat, deterministic), recorded action), WM frozen.
- `eval_critic`: regress Critic to `lambda_returns` of the RECORDED rewards (reuses
  train_behavior.lambda_returns).
- Saved as the standard {world_model, actor, critic, config} ckpt -> `utils.load_models` returns
  exactly what `eval/feedback.py` consumes.

## Result
- `tests/test_train_reference.py` (DummyEnv, no MetaDrive): BC on a CONSTANT-action buffer
  recovers the action (bc_loss < 0.05; cloned actor outputs ~[0.3,-0.6]); eval_critic runs and
  produces finite values. Fast suite: 47 passed.
- (Real IDM reference training on MetaDrive is launched separately to produce runs/reference/ckpt.pt.)

## Hypothesis vs. reality
Matched: decomposing into `collect_idm` (sim-only) + pure `bc_actor`/`eval_critic` made the
learning logic testable headless, and the standard checkpoint format means zero glue between GF3
(produces the reference) and GF4 (consumes it).

## Failures / debugging
- `float(loss)` on a grad tensor warned; switched to `float(loss.detach())`.
- Confirmed via a probe that `info["action"]` (not the passed dummy action) is the applied IDM
  action when `agent_policy=IDMPolicy`.

## One-line takeaway (the interview sentence)
> The feedback reference is a behavior-cloned IDM expert (so it's queryable from the world model's
> latent) plus a critic that policy-evaluates IDM via lambda-returns on its real rewards -- saved
> in the same {wm,actor,critic} checkpoint the rest of the codebase already uses, so it drops
> straight into the feedback engine.

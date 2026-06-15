# Experiment 011 -- Iterated Dreamer loop on MetaDrive: loop is correct, MetaDrive still unsolved

**Date:** 2026-06-15
**Component / change:** `training/dreamer_loop.py` (iterated collect-with-policy -> train WM ->
train policy), wired into `scripts/run_metadrive.py`. No other model changes.

## Hypothesis (write BEFORE running)
The single-shot policy failed on MetaDrive by exploiting a model built from random data
(experiments/010). Collecting WITH the policy should ground the model in policy-visited states
and improve closed-loop driving -- at least beating random.

## Setup
- The loop is first validated on the toy: it drives DummyEnv end-to-end (return 87.6 vs -48
  random; `tests/test_dreamer_loop.py`). So the loop itself is correct.
- MetaDrive (state, state_dim=259): iters=4, seed=1500, collect_per_iter=1000, wm=400,
  behavior=400, explore_std=0.3, actor_lr=3e-4.

## Result
- World model keeps learning across iterations: recon 0.29 -> 0.13, KL healthy; buffer
  2500 -> 5500 (policy-collected data).
- **Closed-loop UNCHANGED and still degenerate: actor_return = -2.89, random = +2.49; the
  policy is steer = -1.0, throttle = +1.0 (full-left + full-throttle).** Identical to the
  single-shot result (exp 010) to 3 decimals.

## Hypothesis vs. reality -- the loop was NECESSARY but NOT SUFFICIENT
Grounding the data did not fix it. The failure is two compounding things the toy never exposed:
1. **Actor corner-collapse.** steer=-1, throttle=+1 are SATURATED tanh outputs; once the value
   gradient drives the mean to an extreme, d tanh/d mean ~ 0 and the actor is stuck in a corner
   with no gradient to escape. The identical deterministic -2.893 both runs = a robust collapse.
2. **Weak/sparse reward + persistent exploitation.** MetaDrive's per-step reward barely
   registers for the model; the actor rides a high IMAGINED return (~14) that doesn't transfer.

These are exactly the v1 simplifications flagged in spec §12 that DreamerV3 adds to prevent this:
- **return normalization** (percentile scaling) -- keeps the value gradient well-scaled so the
  actor doesn't slam into a corner;
- **symlog + two-hot reward/value heads** -- handle MetaDrive's reward scale/sparsity;
- plus entropy/exploration scheduling and MUCH more compute (GPU, many more iterations).

So: the toy is solved with the simplified stack; a real sim needs the v3 upgrades. The loop is a
prerequisite (it's in place and correct), not the whole answer.

## Failures / debugging
- Trivial: `dreamer_train` called `env.close()`; DummyEnv lacked it -> added a no-op `close()`
  to the base env (real sims override). Cost a 7-min re-run -- lesson: smoke the teardown path.
- Tooling: piping a long background run through `grep` block-buffers its output, so interim
  progress is invisible until exit -- use `grep --line-buffered` (or no filter) next time.

## One-line takeaway (the interview sentence)
> The iterated data loop is necessary (it drives the toy and grounds the model in policy-visited
> states) but not sufficient for a real sim: the Tanh-Normal actor collapses into a saturated
> corner under an unnormalized value gradient and weak reward -- which is exactly why DreamerV3
> adds return normalization and symlog/two-hot heads (spec §12), the v1 simplifications we left out.

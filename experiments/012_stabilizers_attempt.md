# Experiment 012 -- DreamerV3 stabilizers (simplified): attempted, reverted

**Date:** 2026-06-15
**Component / change:** attempted return normalization + symlog reward/value
(`models/transforms.py`, edits to `assemble_loss` and `behavior_losses`) to fix the MetaDrive
corner-collapse (experiments/011). **Reverted** -- the simplified versions were a net negative.

## Hypothesis (write BEFORE running)
Return normalization (divide returns by their spread) keeps the value gradient ~unit-scale and
should stop the Tanh actor from collapsing into a saturated corner; symlog reward/value handles
MetaDrive's reward scale. Both are the v1->v3 upgrades flagged in spec §12.

## What happened
1. **NaN.** symexp of an UNBOUNDED critic/reward head overflows to inf -> NaN once the actor
   drives the model into out-of-distribution latents. Patched by clamping symexp input to ±20.
2. **Toy regression (the dealbreaker).** With the clamp, the single-shot behavior gate still
   passed (throttle=1, steer=0.019), but the ITERATED loop **collapsed on the toy**:
   steer=-1.0 (full-left, saturated), return 0.0 -- the exact corner-collapse the stabilizers
   were meant to prevent, now on the easy toy (before stabilizers the loop drove it at
   steer=0.03). The reward-generalization tests passed in symlog space.

## Hypothesis vs. reality -- why the simplified versions fail
- **Scalar symlog regression is not bounded.** DreamerV3 predicts reward/value as a TWO-HOT
  distribution over a fixed symlog-spaced support, so the decoded value is bounded by the
  support and `symexp` never overflows. A raw scalar + `symexp` has no such bound -> NaN.
- **Per-batch return scale is noisy and interacts badly with policy-collected data.** DreamerV3
  uses an EMA of the percentile spread; a per-batch estimate over 16 sequences, combined with
  the iterated loop feeding back its own data, destabilized the steer gradient enough to let
  the saturated corner win.

So the proper fix is the FULL DreamerV3 machinery (two-hot heads + EMA return normalization +
careful entropy/tuning), not a quick scalar stand-in. That's a substantial change, out of scope
for a single step.

## Decision
Reverted to the last green state (commit 71c0982): the iterated loop drives the toy
(return 87.6 vs -48) and the WM learns on MetaDrive; MetaDrive *control* remains the documented
open problem (experiments/010, 011). Keeping the toy green (the stated constraint) takes
priority over shipping a fragile stabilizer.

## One-line takeaway (the interview sentence)
> The simplified return-norm + scalar-symlog stabilizers backfired -- symexp of an unbounded head
> NaNs, and per-batch return scaling let the iterated loop collapse the steer into a saturated
> corner even on the toy -- which is exactly why DreamerV3 uses BOUNDED two-hot heads and an EMA
> return scale; the quick stand-in isn't equivalent, so I reverted rather than ship a regression.

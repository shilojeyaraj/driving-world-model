# Experiment 000 -- plumbing smoke test (example entry)

**Date:** 2026-06-11
**Component / change:** none yet -- verifying env + buffer + shapes.

## Hypothesis
The dummy env + sequence buffer should yield (batch, seq_len, obs_dim) tensors for both
state and image obs, with no model implemented.

## Setup
- config diffs: env="dummy", seq_len=10, max_episode_steps=40
- obs_type: both "state" and "image"
- implemented: nothing -- random policy only

## Result
- state:  obs batch (4, 10, 35)
- image:  obs batch (4, 10, 3, 64, 64)

## Hypothesis vs. reality
Matched. The base interface decouples shape handling from the env, as intended.

## Failures / debugging
Brace-expansion mkdir failed once under dash; unrelated to the code path.

## One-line takeaway
> The whole data path runs on a laptop with only numpy -- I can iterate before touching a GPU.

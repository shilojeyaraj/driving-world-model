# Experiment 001 -- Encoder + Decoder (state mode), reconstruction sanity

**Date:** 2026-06-12
**Component / change:** `models/encoder.py` (state MLP) + `models/decoder.py` (obs/reward/continue heads), built test-first. Phase 1 of the v1 spec.

## Hypothesis (write BEFORE running)
A small MLP encoder (state_dim -> hidden -> hidden -> embed_dim) and an MLP decoder should
form an over-complete autoencoder (embed_dim=256 >> state_dim=35). Overfitting a *fixed*
batch should drive recon MSE to ~0 -- including the noise dims, because an autoencoder SEES
the observation (unlike the world model, which must *predict* it and can't beat the noise
floor). If the loss doesn't move, something is miswired (detached graph / dead ReLU /
shape mismatch into the loss).

## Setup
- config diffs from default: added `hidden_dim=256`, `min_std=0.1`, and behavior fields
  (`gamma`, `lambda_`, `entropy_coef`, `actor_lr`, `critic_lr`). state_dim=35.
- obs_type / env / device: state / dummy / cpu
- what I implemented: state-mode Encoder MLP; Decoder with obs head (-> state_dim),
  reward head (-> scalar), continue head (-> logit). Image branches deferred (raise) until
  the GPU phase. SiLU activations throughout.

## Result
- `tests/test_shapes.py`: encoder (N, state_dim) -> (N, embed_dim); decoder feat (N, F) ->
  obs (N, state_dim), reward (N,), cont_logit (N,). Both pass.
- `tests/test_reconstruction.py`: autoencoder overfit of a 16-sample fixed batch, 300 Adam
  steps @ lr=1e-3 -> final MSE < 0.01 and < 10% of initial. Pass.
- Full suite: 3 passed.

## Hypothesis vs. reality
Matched. The autoencoder collapses recon to ~0, confirming the wiring and gradient flow.
Key distinction internalized: an autoencoder reconstructing noise is *expected and easy*;
the world model later will plateau at the noise variance on those same dims because it must
predict them from latent dynamics without seeing the current obs. Different jobs, different
floors -- that's why §9 judges the world model on `pos` + reward + open-loop, not raw recon.

## Failures / debugging
None. TDD caught nothing because shapes were pinned before implementation; the RED runs
confirmed each test failed with `NotImplementedError` (feature missing) before GREEN.

## One-line takeaway (the interview sentence)
> An autoencoder can trivially reconstruct an observation (it sees it); a world model can't,
> because it has to *predict* the obs from latent dynamics -- so recon MSE means different
> things in the two settings, and only the predictive one tells you the dynamics are learned.

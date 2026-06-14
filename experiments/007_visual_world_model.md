# Experiment 007 -- Visual world model (V1): learning + dreaming from pixels

**Date:** 2026-06-13
**Component / change:** visual DummyEnv render (`envs/dummy.py:_render`), image Encoder (CNN)
and Decoder (transposed-CNN) in `models/`, shape-general ELBO recon in `world_model.py`, and
the dream-video renderer `scripts/dream_video.py`. Built test-first.

## Hypothesis (write BEFORE running)
If the obs is a clean RENDER of pos (a Gaussian blob whose column = tanh(pos)), then the whole
pipeline should work in pixels with NO change to the RSSM/ELBO/actor-critic -- only the encoder,
decoder, env, and the recon reduction change (obs-type-agnostic core). Pixel reconstruction
should drop sharply (the blob is fully determined by the latent via the posterior), and rolling
the PRIOR forward with true actions (the "dream") should reproduce the blob's motion.

## Setup
- Visual DummyEnv: same dynamics (pos integrates throttle; reward = throttle-|steer|), but obs
  is a (3,H,W) render of a blob at column = (tanh(pos/3)*0.5+0.5)*(W-1). No pixel noise (clean).
- Image Encoder: 4 stride-2 convs [32,64,128,256] -> flatten -> Linear(embed_dim). Decoder:
  Linear -> 256x(s0)x(s0) -> 4 stride-2 transposed convs -> (3,H,W). image_size multiple of 16.
- Recon term generalized to sum over ALL obs dims (C*H*W), so the same ELBO covers state+image.
- Tested at 16x16 (CPU); dream rendered at 16-32.

## Result
- Shape + contract tests pass (encoder/decoder image I/O; assemble_loss on an image batch
  runs + backprops). Fast suite: 26 passed.
- Image world model trains: pixel recon **28.07 -> 0.17** in 200 steps (16x16).
- Dream-video renderer: conditions on `context` real frames, rolls the prior MEAN with true
  actions, decodes frames. dream_montage.png shows truth (top) vs dream (bottom); the dreamed
  blob tracks the true positions across the horizon. per-pixel dream MSE = 0.011.

## Hypothesis vs. reality
Matched. The obs-type-agnostic design held: switching to pixels touched only env + encoder +
decoder + the recon reduction; the RSSM, KL, action-timing, and (untouched) actor-critic all
carried over. The dream is slightly dimmer/blurrier than truth -- the classic MSE-decoder
signature (it hedges toward the mean) -- but spatially correct. A real upgrade here would be a
sharper decoder (e.g. the DiT/diffusion head noted in decoder.py) and a richer scene.

## Failures / debugging
- Recon initially summed only the last dim (fine for state vectors, wrong for (3,H,W)); fixed
  to sum over all obs dims. Caught immediately by the image contract test.

## One-line takeaway (the interview sentence)
> Because the world model is obs-type-agnostic between the encoder and decoder, going from a
> state vector to pixels meant swapping only those two nets + the env; the model then learns to
> reconstruct frames (recon 28->0.17) and to DREAM -- rolling the prior forward and decoding
> latents into a video that tracks reality (pixel MSE 0.011).

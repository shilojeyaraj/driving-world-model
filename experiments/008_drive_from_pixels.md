# Experiment 008 -- Driving from pixels (V1 step ①)

**Date:** 2026-06-14
**Component / change:** none new in the model -- this validates that the EXISTING actor-critic
+ closed-loop, which are observation-agnostic, work in image mode on a frozen visual world
model. One efficiency fix: the imagination loop now skips the (image) obs decode
(`decoder(..., decode_obs=False)`) since it only needs reward/continue.

## Hypothesis (write BEFORE running)
The policy code operates on `feat=[h;z]`, never on pixels. So a policy trained purely in the
imagination of an IMAGE world model should drive the visual env just as it did in state mode --
throttle -> +1, steer -> 0 -- with zero env steps during policy learning. Distribution shift is
mild because the optimum is state-independent.

## Setup
- Image cfg: 16x16, deter=64, stoch=16, hidden=64, imagine_horizon=10, actor_lr=3e-3.
- Train image world model 400 steps, behavior 400 steps (in imagination), then closed-loop
  in the visual DummyEnv, 5 episodes x 100 steps. (`scripts/drive_from_pixels.py`.)

## Result
- **Closed-loop from pixels: actor_return=94.63, random_return=-49.79, throttle=1.000,
  steer=0.045.** The policy drives the visual env near-optimally.

## Hypothesis vs. reality
Matched. The obs-agnostic core held end to end: the only image-specific code is the CNN
encoder, transposed-CNN decoder, and the render -- the RSSM, ELBO, actor-critic, lambda-returns,
and closed-loop logic are byte-for-byte the same as state mode. The loop perceive-pixels ->
encode -> dream -> act -> drive is closed.

## Failures / debugging
- **Cost / tooling lesson (not a model bug):** the first run took 3h40m wall-clock because
  several heavy training jobs were left running CONCURRENTLY, oversubscribing the CPU. Image
  training is inherently heavy; the fix is operational -- run heavy jobs ONE AT A TIME, and keep
  image-training demonstrations as scripts/experiments, not pytest gates. The suite now has only
  a FAST image closed-loop *contract* test; this full demonstration lives in the script above.
- Efficiency: imagination was decoding the image obs head every step and discarding it; added
  `decode_obs=False` so only the reward/continue MLPs run in the dream loop.

## One-line takeaway (the interview sentence)
> Because the policy lives entirely in latent space, a controller trained only inside an image
> world model's imagination drives the real pixel env near-optimally (return 94.6 vs -50 random)
> -- the same actor-critic code as the state-vector version, unchanged.

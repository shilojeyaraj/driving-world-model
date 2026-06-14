# Experiment 009 -- MetaDrive: the world model learns from a real driving sim

**Date:** 2026-06-14
**Component / change:** hardened `envs/metadrive_env.py` (pure `adapt_obs` + defensive
reset/step), `scripts/probe_metadrive.py`, `docs/METADRIVE.md`, and a skip-or-run live test.
No model changes -- the point is that the sim-agnostic pipeline runs on a REAL simulator.

## Hypothesis (write BEFORE running)
Everything downstream depends only on the `envs/base.py` contract, so swapping the toy render
for MetaDrive should require only the wrapper + setting `cfg.state_dim` to MetaDrive's real obs
dim. The world model should then train on real lidar+ego observations with no other changes.

## Setup
- Install: `pip install metadrive-simulator` (PyPI) FAILED on this Windows/CPU box -- it pulls
  the legacy `gym`, which won't build with modern setuptools. `pip install
  git+https://github.com/metadriverse/metadrive.git` (gymnasium-based) SUCCEEDED -> MetaDrive
  0.4.3, panda3d 1.10.16, gymnasium 1.3.0.
- Probe (`scripts/probe_metadrive state`): real obs is **Box(259,)** in [0,1]; action
  **Box(2,) in [-1,1]** = [steering, throttle] -- matches our Tanh actor. Render mode: none
  (headless). So `cfg.state_dim = 259`.
- End-to-end: collect 600 steps (random policy) -> 6 usable episodes (seq_len=10); train the
  world model ~30 steps. deter=64, stoch=16, hidden=64.

## Result
- Live smoke test (reset/step) passes; runs headless on CPU in state mode.
- World model trains on real MetaDrive data: loss 245 -> 11, recon 243 -> 9.7 (259-dim state),
  reward ~0.001 (MetaDrive's reward is small/sparse), **cont 0.70 -> 0.001**, KL healthy.

## Hypothesis vs. reality
Matched. Only the wrapper + `state_dim=259` changed; the encoder/RSSM/ELBO/actor-critic are
unchanged. One qualitative difference from the toy worth noting: MetaDrive episodes really
TERMINATE (crash / off-road), so the continue head learns a meaningful signal (cont -> ~0 near
terminations), unlike the DummyEnv where done was just a timeout. Image mode is NOT verified --
it needs a GPU/offscreen renderer (Colab/Kaggle).

## Failures / debugging
- PyPI install failure (legacy gym) -> used the GitHub source install. Documented in
  docs/METADRIVE.md with all three fixes.
- The #1 cross-version trap (obs dim) is handled by the probe: state_dim is 259, not the toy's
  35; the probe prints it so config can be set before training.

## One-line takeaway (the interview sentence)
> Because the model only sees the env contract, pointing it at a real driving simulator
> (MetaDrive) meant writing one wrapper and setting state_dim=259 from a probe -- the same
> encoder/RSSM/ELBO then learns from real lidar+ego observations (recon 243->9.7), with the
> continue head now learning real episode terminations.

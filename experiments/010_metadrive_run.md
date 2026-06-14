# Experiment 010 -- Real MetaDrive run (state mode): WM learns, single-shot policy FAILS

**Date:** 2026-06-14
**Component / change:** `scripts/run_metadrive.py` -- single-shot pipeline on real MetaDrive
state obs (collect random -> train WM -> train policy in imagination -> closed-loop in the sim).
No model changes.

## Hypothesis (write BEFORE running)
The world model should learn the 259-dim MetaDrive state (recon drops, KL healthy). The policy
is the open question: trained only in imagination of a model that saw ONLY random-policy
(crash-prone) data, it may exploit model errors. I expected a modest-at-best policy.

## Setup
- env=metadrive, state mode, state_dim=259; deter=128, stoch=32, hidden=128; seq_len=10;
  imagine_horizon=15; actor_lr=critic_lr=3e-4. Collect 4000 steps; WM 1500; behavior 1500;
  closed-loop 5 episodes x 200 steps in the real sim.

## Result
- Collected 4000 steps -> 20 usable episodes.
- **World model: recon 245 -> 0.17, KL ~0.8 (healthy). Learns the real state well.** ✅
- Behavior: imagined_return -> ~4.4, but actor ENTROPY crept up (2.3 -> 10.8).
- **Closed-loop (REAL sim): actor_return = -2.89, random_return = +1.80; the learned policy is
  steer = -1.0, throttle = +1.0 -- i.e. full-left + full-throttle.** The policy is degenerate
  and WORSE than random. ❌

## Hypothesis vs. reality
Two clean lessons, both expected in hindsight:
1. **Model exploitation (the imagination failure mode).** The actor found actions the IMAGINED
   model rates highly (return 4.4) but that drive straight off-road in reality (return -2.9).
   The world model never saw consistent full-throttle/hard-steer trajectories (only random,
   crashy data), so its prediction there is fantasy and the actor exploits it. This is exactly
   what train_behavior.py and ARCHITECTURE.md §4 warn about, now demonstrated on a real sim.
2. **No usable reward signal.** MetaDrive's per-step reward was ~0 to the model (reward loss
   ~0.000), so the policy had little real reward to chase and rode the (poorly grounded) critic
   value into a corner; the tiny entropy bonus then let the action std drift up unopposed.

The toy DummyEnv hid both: its optimum is state-independent (throttle=1, steer=0) and its
reward is dense and one-step, so a single shot worked. MetaDrive is the regime where the
SINGLE-SHOT shortcut breaks -- which is precisely why Dreamer uses an ITERATED data loop.

## Failures / debugging
Not a code bug -- a methodology limit. The fix is the real-Dreamer outer loop:
- collect WITH the current policy (+ exploration noise), append to the buffer, retrain WM, repeat;
- this grounds the model in the states the policy actually visits, closing the
  imagination-vs-reality gap. Likely also: reward normalization / longer horizons / more data.

## One-line takeaway (the interview sentence)
> On a real sim the single-shot shortcut breaks exactly as theory predicts: the world model
> learns the observations (recon 245->0.17), but a policy trained only inside a model built from
> random data EXPLOITS the model's errors -- it's confident in imagination (return 4.4) yet drives
> off-road in reality (return -2.9 < random) -- which is why Dreamer alternates collect-with-policy
> and retrain instead of training once.

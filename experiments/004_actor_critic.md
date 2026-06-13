# Experiment 004 -- Actor-Critic in imagination (the policy learns with zero env steps)

**Date:** 2026-06-13
**Component / change:** `models/actor_critic.py` (Tanh-Normal actor, value critic) +
`training/train_behavior.py` (lambda-returns, imagination rollout, value-gradient training).
Phase 4 of the v1 spec. Built test-first.

## Hypothesis (write BEFORE running)
With a frozen, reward-correct world model, an actor that backprops the imagined lambda-return
through the differentiable dynamics + reward head should discover the env's optimum:
reward = throttle - |steer|, so the per-step optimum is throttle=+1, steer=0, independent of
state. The critic bootstraps value past the short horizon. Convergence should be fast since
the optimum is simple and state-independent.

## Setup
- config: deter=64, stoch=16, hidden=64; imagine_horizon=10; gamma=0.99, lambda=0.95;
  entropy_coef=1e-3. Test used actor_lr=critic_lr=3e-3 (the spec default 8e-5 is correct but
  slow for a unit-test convergence check).
- Trained world model ~400 steps, then behavior ~500 steps, all in imagination (no env steps
  during policy learning). World model frozen via requires_grad_(False).

## Result
- lambda_returns matches a hand-worked example (gamma=0.9, lam=0.5: [3.565, 4.70]).
- Behavior gate: imagined_return rose -4.6 -> ~68; deterministic policy converged to
  **throttle=1.000, steer=0.016** -- the exact optimum. Actor/critic shape + range tests pass.

## Hypothesis vs. reality
Matched cleanly. Three design points that made it work and connect to earlier phases:
1. **Reward alignment (from exp 002):** the reward for action a_i is decoded from feat_{i+1}
   (the state that consumed a_i). Without this the actor would get no usable action gradient.
2. **Sample vs mean (from exp 003):** imagination here SAMPLES (reparameterized) so gradients
   flow through the sampler and the policy explores; open-loop prediction used the mean. Same
   img_step, opposite `sample` flag -- on purpose.
3. **Gradient routing:** start states detached (no grad into the world model); critic reads
   detached feats (its grad touches only critic params); the actor's bootstrap values are
   detached so the actor gradient flows ONLY through rewards -> dynamics -> actions.

## Failures / debugging
None. Entropy went negative as the policy sharpened (small std -> negative differential
entropy) -- expected and harmless; deterministic eval uses tanh(mean) so std doesn't affect
the evaluated action.

## One-line takeaway (the interview sentence)
> The policy is trained entirely inside the model's dream: maximizing the imagined lambda-return
> backpropagates through the differentiable reward head and dynamics into the actor, so it
> discovers throttle=1, steer=0 with zero environment interaction -- the model-based payoff.

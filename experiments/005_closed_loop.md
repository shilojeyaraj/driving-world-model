# Experiment 005 -- Closed-loop driving (the dream transfers to the real env)

**Date:** 2026-06-13
**Component / change:** `eval/closed_loop.py::closed_loop_eval`. Phase 5 of the v1 spec --
the final functional phase. Built test-first.

## Hypothesis (write BEFORE running)
The actor was trained only inside the world model's imagination. Running it in the REAL env
(maintaining the RSSM posterior across steps: encode obs -> obs_step -> feat -> deterministic
action -> env.step) should produce near-optimal driving, because the learned per-step optimum
(throttle=+1, steer=0) is state-independent, so distribution shift shouldn't hurt. Episode
return should approach +1/step and far exceed the random baseline (~ -0.5/step).

## Setup
- config: deter=64, stoch=16, hidden=64; trained world model ~400 steps, behavior ~500 steps
  (actor_lr=critic_lr=3e-3), then closed-loop eval over 5 episodes x 100 steps.

## Result
- closed-loop: **actor_return=94.84, random_return=-51.04, actor_throttle=1.000,
  actor_steer=-0.009.** The actor drives the real env near-optimally and beats random by a
  wide margin. Contract test (runs, returns metric keys) also passes.

## Hypothesis vs. reality
Matched. The imagination-trained policy transfers cleanly because the optimum is simple and
state-independent here -- exactly the case where open-loop and closed-loop agree. The general
lesson stands for harder envs: closed-loop is a SEPARATE axis because the policy visits states
it drives itself into (distribution shift) and per-step errors compound; a model can predict
well open-loop yet the policy can steer into states the model gets wrong. DummyEnv just doesn't
exercise that failure mode.

## Failures / debugging
None. The action-timing convention (feed previous action + current obs to obs_step, a_{-1}=0
at reset) carried straight over from observe -- no new off-by-one.

## One-line takeaway (the interview sentence)
> A policy trained with zero environment interaction -- purely by maximizing returns inside a
> learned world model's imagination -- drives the real env near-optimally (return 94.8 vs -51
> random); closed-loop is the only test that proves the dream-trained behavior actually
> transfers, because it lets the policy choose the actions and visit its own states.

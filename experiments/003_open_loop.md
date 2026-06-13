# Experiment 003 -- Open-loop prediction (true actions beat no-action)

**Date:** 2026-06-12
**Component / change:** `eval/open_loop.py::open_loop_eval` + a `sample=False` (prior-mean)
rollout path on `RSSM.img_step`/`imagine`. Phase 3 of the v1 spec. Built test-first; the
milestone gate failed first and was root-caused with the systematic-debugging process.

## Hypothesis (write BEFORE running)
Rolling the PRIOR forward from a context state with the TRUE actions should predict the
future much better than rolling with ZERO actions, because in DummyEnv the only thing that
moves the future is the action (pos integrates throttle; everything else is noise). So
error_true(h) < error_noaction(h), with the gap widening as horizon grows.

## Setup
- config: deter=64, stoch=16, hidden=64, seq_len=20; context=5, horizon=10.
- trained the world model ~600-800 steps on fresh batches, then ran open_loop_eval.

## Result
- Gate test passes: full-obs MSE/horizon model sum=3.60 < no_action sum=3.79, model < no_action
  at EVERY step, gap widening with horizon. pos-only: model [0.0035..0.026] < no_act
  [0.008..0.065]; example trajectory tracks ground truth closely. Full suite: 12 passed.

## Hypothesis vs. reality -- TWO methodological bugs (not the model)
The gate FAILED at first: summed-over-35-dims, true-action sum (3.784) ~ no-action (3.777),
and even slightly WORSE at long horizon. The model looked like it ignored actions. Instead of
guessing, I instrumented pos specifically. Evidence:

1. **Metric drowning.** 34 of 35 obs dims are pure noise (irreducible ~0.34 added to BOTH
   rollouts). The predictable signal (pos) is 1/35 of a summed metric -> the action effect was
   invisible in the aggregate even though pos-only showed model < no_action.
2. **Sampling variance (the big one).** `imagine` SAMPLED z from the prior. Open-loop
   *prediction* must be DETERMINISTIC -- the prior MEAN. Sampling injected noise into the 34
   unpredictable dims that swamped the pos signal and made true vs no-action indistinguishable.
   Switching to the prior mean: pos MSE dropped ~3x AND the full-obs summed metric cleanly
   separated (3.60 vs 3.79). No env-specific pos hack needed.

So the model HAD learned action-conditioned dynamics all along; the eval methodology was wrong.
(Aside: persistence beats the model on pos-only because the random policy is zero-mean, so
"pos holds" is a strong baseline -- but that is NOT the spec's gate, which is vs no-action.)

## Failures / debugging
Root-caused via the 4-phase process: read the symptom precisely, ruled out an action-index bug
by hand, instrumented pos (evidence), formed ONE hypothesis (sampling noise), tested it
minimally (add sample=False), confirmed. No thrashing.

## One-line takeaway (the interview sentence)
> Open-loop prediction must roll the prior MEAN, not samples: with stochastic rollouts the
> sampling noise on unpredictable observation dims drowns the action signal, so a model that
> genuinely learned dynamics looks like it ignored the actions -- a measurement bug masquerading
> as a modeling failure.

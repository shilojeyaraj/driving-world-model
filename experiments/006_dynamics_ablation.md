# Experiment 006 -- Dynamics ablation: GRU (RSSM) vs Mamba-style SSM

**Date:** 2026-06-13
**Component / change:** `models/recurrence.py` -- isolated the deterministic recurrence behind
a `Recurrence` seam (`make_recurrence` dispatch on `cfg.dynamics`); ported the GRU
(`GRURecurrence`) and added a minimal Mamba-style selective diagonal SSM (`SSMRecurrence`).
RSSM now delegates `_recur`/`initial_state` to the recurrence. Built test-first.

## Hypothesis (write BEFORE running)
Swapping ONLY the recurrence (GRU -> selective SSM), with the same prior/posterior heads and
the same eval harness, both should learn the toy dynamics. On DummyEnv the dynamics are trivial
(pos integrates throttle; reward is one-step), so I expect **little separation** -- this toy
doesn't stress long-range memory or selectivity, where SSMs are supposed to help. The point of
this step is the controlled-swap *machinery*, not a verdict on GRU vs Mamba.

## Setup
- Same config for both, only `cfg.dynamics` differs: deter=64, stoch=16, hidden=64, seq_len=16,
  imagine_horizon=10. World model 500 steps, behavior 400 steps. `scripts/ablate_dynamics.py`.
- SSMRecurrence: s_t = exp(Δ·a) ⊙ s_{t-1} + Δ ⊙ (B·x), with a = −softplus(A_log) (stable),
  Δ = softplus(Linear([z,a])) (input-dependent "selectivity"). State == feature == R^deter.

## Result (controlled swap)
| metric          | rssm (GRU) | mamba (SSM) | note |
|-----------------|-----------:|------------:|------|
| wm_recon        | 0.352      | 0.353       | noise floor ~0.34 |
| wm_kl           | 0.769      | 0.699       | both healthy (>0) |
| reward_heldout  | 0.001      | 0.004       | both << 0.1 -> learned |
| openloop_model  | 3.564      | 3.805       | lower=better |
| openloop_noact  | 3.788      | 3.995       | baseline |
| **openloop gap**| **0.224**  | **0.190**   | model beats no-action (action-conditioning) |
| closed_return   | 96.8       | 94.9        | random ~ −50 |
| throttle/steer  | 1.00/0.01  | 1.00/−0.02  | both reach the optimum |

## Hypothesis vs. reality
Matched. Both recurrences learn dynamics, beat the no-action baseline open-loop, and drive the
real env near-optimally. GRU edges the SSM slightly here (lower open-loop error, marginally
higher closed return), but the gaps are small and **not meaningful on this toy** -- DummyEnv's
dynamics are one-step/linear, exactly the regime where the recurrence choice barely matters.
The honest conclusion: the **seam works** (swap one module, rerun the same gates), and a real
verdict needs a task with long-range dependencies (image mode, longer horizons, partial
observability) -- that's where selective SSMs are designed to win.

## Failures / debugging
None. The refactor was behavior-preserving (the GRU path's training gate, reward-generalization,
still passed at 0.001 after the move). The Transformer is deliberately *not* added here: it's
non-Markov (attends over history) and needs a windowed-state recurrence variant, a separate
sub-step.

## One-line takeaway (the interview sentence)
> I isolated the RSSM's recurrence behind a one-method interface so GRU and a selective SSM are
> a one-line config swap measured on the identical eval harness; on the toy they're a wash (as
> expected for one-step dynamics), but the controlled-swap machinery is what makes a real
> long-horizon comparison cheap.

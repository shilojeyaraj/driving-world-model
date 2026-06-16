# Experiment 014 -- GF4: the driving-feedback engine

**Date:** 2026-06-16
**Component / change:** `eval/feedback.py` -- three signals (A outcome forecast, B style
deviation, C state value) + a `DrivingFeedback` online aggregator + a pure `report_from_traces`.
Phase GF4 of the gesture-feedback spec. Built test-first.

## Hypothesis (write BEFORE running)
The 3 signals each read a different part of the model-based stack and should be computable from a
maintained RSSM posterior with no new model machinery: A = imagine-your-action + decoder
continue/reward (prior MEAN, like open-loop); B = compare your action to a reference actor's; C =
the critic's value. The fusion into labeled events is where "feedback on habits" actually lives.

## Setup
- `forecast_safety(wm, state, action, horizon)` -> survival (prod of predicted continue),
  pred_return, risk flag. Uses `imagine(sample=False)` + `decoder(decode_obs=False)`.
- `style_deviation(reference_actor, feat, action)` -> per-dim deviation from the reference.
- `state_value(critic, feat)` -> scalar.
- `DrivingFeedback.step(obs, action)` carries the RSSM posterior exactly like closed_loop.
- `report_from_traces` -> labeled events (near_off_road, oversteer L/R, harsh/late throttle) + stats.
- Tested on tiny untrained models + DummyEnv + synthetic traces (no MetaDrive/webcam).

## Result
- `tests/test_feedback.py`: 5 pass -- forecast keys/ranges (survival in [0,1]), style-deviation
  shapes, value is a float, DrivingFeedback.step+finalize over a DummyEnv rollout, and
  report_from_traces detecting injected near_off_road + oversteer events.

## Hypothesis vs. reality
Matched: the signals drop out of the existing heads with no new model code, and the engine reuses
the closed-loop posterior-carry pattern. The honest caveat carries over from experiments/010-011:
the CONTINUE-head forecast (A) is the trustworthy signal on a hard sim; reward/value (in A/C) are
soft until the world model + critic are well-trained on good (IDM) data -- which is what GF3 is for.

## Failures / debugging
None. Kept the metrics pure-ish and the report a pure function so the whole engine is testable
without MetaDrive or a camera.

## One-line takeaway (the interview sentence)
> Driving feedback falls straight out of a world model: imagine the driver's action and read the
> continue head for "is this safe?", compare to a reference policy for "is this how an expert
> drives?", and read the critic for "is this a good state?" -- then fuse the per-step signals into
> labeled events (oversteer, near-off-road) that surface the driver's habits.

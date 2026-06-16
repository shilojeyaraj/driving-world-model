# Experiment 013 -- GF1: gesture -> action controller

**Date:** 2026-06-15
**Component / change:** `control/gesture.py` (new package): a PURE `landmarks_to_action` mapping
+ a `GestureController` that reads a webcam hand via MediaPipe. First phase of the gesture-
control + driving-feedback system (spec: docs/superpowers/specs/2026-06-15-gesture-feedback-design.md).
Built test-first.

## Hypothesis (write BEFORE running)
A gesture is just another ACTION SOURCE producing `[steer, throttle] ∈ [-1,1]`, so it plugs into
the existing env contract with no changes to envs/base.py. The fragile part is the sensor->action
mapping (range, deadzone, smoothing), so that should be a pure, unit-tested function; the live
cv2/MediaPipe capture is a thin adapter on top, exercised only by a skip-able smoke test.

## Setup
- `landmarks_to_action(steer_raw, throttle_raw, prev, deadzone, smoothing)`: deadzone -> EMA
  smoothing -> clip to [-1,1]. Pure, dep-free.
- `GestureController`: lazy-imports cv2 + mediapipe; steer = hand-center x (left/right),
  throttle = hand height (higher = accelerate); delegates to the pure mapping.
- config: added webcam_id, gesture_smoothing, gesture_deadzone (+ forecast_horizon,
  risk_threshold for later phases).

## Result
- `tests/test_gesture.py`: 5 pure-mapping tests pass (range, clip, deadzone, monotonic steer,
  smoothing blends toward previous). Live smoke `importorskip`s cv2+mediapipe and skips when no
  camera (headless here) -- marked @slow because importing mediapipe is heavy.
- Fast suite: 40 passed, 10 deselected (~19s).

## Hypothesis vs. reality
Matched. The pure-mapping/adapter split (same pattern as envs/metadrive_env.py:adapt_obs) made
the tricky part fully testable without hardware, and the controller is a drop-in action source.
cv2 + mediapipe are both installed here, so the live path is runnable on a machine with a webcam.

## Failures / debugging
- The live smoke test imported mediapipe (heavy, ~tens of seconds) and slowed the fast suite to
  71s; marking it @slow restored the fast loop to ~19s. Same lesson as the MetaDrive live test:
  keep optional-heavy-dep tests behind the slow marker + importorskip.

## One-line takeaway (the interview sentence)
> A webcam gesture is just another action source: I isolate the noisy sensor->[steer,throttle]
> mapping (deadzone + EMA smoothing + clip) as a pure, unit-tested function, with the
> cv2/MediaPipe capture as a thin adapter -- so the control layer plugs into the same env contract
> as the random policy or the trained actor, with zero changes downstream.

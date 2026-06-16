# Experiment 016 -- GF2 + GF5: live gesture drive loop + feedback HUD/report

**Date:** 2026-06-16
**Component / change:** `scripts/drive_gesture.py` (drive MetaDrive from an action source +
render + record + optional live feedback HUD) and `scripts/feedback_report.py` (offline session
-> habits report). Phases GF2 + GF5. Completes the gesture-feedback system (GF1-GF5).

## Hypothesis (write BEFORE running)
The driving loop is just "action source -> env.step -> render -> record", with the gesture
controller as one swappable source; the HUD/report are a thin presentation layer over the
already-tested DrivingFeedback engine + the existing top-down recorder. Making the action source
pluggable (gesture / random / forward) should let the WHOLE pipeline run headless (no camera) so
it's verifiable here.

## Setup
- `drive_gesture(policy, ckpt, ...)`: action source = GestureController (webcam) or random/forward
  (headless); steps a raw MetaDriveEnv, renders top-down, records (obs,action,reward,done) to
  .npz; if a reference ckpt is given, runs DrivingFeedback each step and draws a HUD (safety bar,
  RISK alert, steer/throttle deviation, value).
- `feedback_report(session, ckpt)`: replays the recorded session through DrivingFeedback ->
  summary stats + labeled events, dumped to JSON.
- Headless smoke: tiny UNTRAINED reference ckpt + policy=random, 60 steps.

## Result
- Smoke ran end-to-end: 60-frame GIF with HUD + session.npz saved; report printed stats + event
  counts. A rendered frame confirms the HUD overlay draws on the top-down view (safety bar +
  text) with the car on the road.
- Numbers are meaningless by design (untrained reference -> survival ~0, everything flags risk);
  the smoke verifies the PLUMBING (render + feedback + HUD + recording + report), not quality.

## Hypothesis vs. reality
Matched. The pluggable action source made the full live system verifiable without a camera, and
the HUD/report reused DrivingFeedback (GF4) + the top-down recorder with no new model code. For
REAL feedback, run `training/train_reference.py` on actual IDM data, then
`drive_gesture(policy="gesture", ckpt="runs/reference/ckpt.pt")` on a webcam machine.

## Failures / debugging
- Replaced an ugly inline `__import__("PIL.Image")` with a clean `_resize` helper.

## One-line takeaway (the interview sentence)
> The live system is a pluggable action source (your hand, via MediaPipe) driving MetaDrive while
> the world-model feedback engine overlays a real-time HUD -- safety from the continue head, style
> deviation from a behavior-cloned expert, value from the critic -- and an offline pass turns the
> session into a labeled habits report; making the source pluggable let me verify the whole
> pipeline headless before ever touching a webcam.

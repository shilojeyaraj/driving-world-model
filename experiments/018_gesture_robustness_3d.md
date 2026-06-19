# Experiment 018 -- Gesture driving robustness: 3-D render, scene config, position+pose scheme

**Date:** 2026-06-18
**Component / change:** `scripts/drive_gesture.py`, `control/gesture.py`, `envs/metadrive_env.py`,
`config.py`, `scripts/{watch_metadrive_3d,record_metadrive,run_metadrive}.py`. Post-capstone
usability work to make the live human demo actually pleasant: a real 3-D view, choosable scenes,
and a robust hand-control scheme. No change to the world-model / feedback math.

## Hypothesis (write BEFORE running)
The capstone (experiments/017) proved the feedback signals are meaningful, leaving only the live
human ergonomics. Three gaps to close, each expected to be self-contained because everything is an
*action source* / *env config* and never touches the RSSM/ELBO core:
1. People want MetaDrive's **rendered 3-D window**, not the top-down GIF.
2. They want to pick the **scene** (highway / intersection / roundabout), not a fixed map.
3. The first discrete control (pointing + downward-swipe-reverse) is fragile: poses are mutually
   exclusive on one hand, and motion-based reverse misfires. A **position-steers + pose-throttles**
   scheme should make forward+turn simultaneous and reverse deliberate.

## Setup
- **Scene config:** added `metadrive_map` (int N or block letters S/C/X/O/T/r) +
  `metadrive_traffic_density`, threaded through every entry point via a single
  `envs/metadrive_env.py:metadrive_config(cfg)` helper.
- **3-D render:** `drive_gesture` now sets `use_render=True` for gesture modes and draws the HUD on
  the 3-D window via `env.render(text=...)` (confirmed supported in MetaDrive 0.4.3 BaseEnv.render);
  a `2d` flag keeps the headless top-down GIF path.
- **Control scheme (chosen with the user):** *position steers + pose throttles.* Pure, camera-free
  functions `hands_together` / `throttle_command` / `steer_from_position` / `position_pose_action`;
  `GestureController` now tracks **two hands** (`num_hands=2`) for the prayer/reverse sign.
  Mapping: hand x = steer, fist = forward, open palm = coast/stop, two hands together = reverse.

## Result
- Fast suite: **69 passed** (added 16 gesture tests across the two discrete schemes), incl. a test
  asserting *fist-held-right -> steer>0 AND throttle>0 at once* (the simultaneity the user asked for).
- Headless 5-step `drive_gesture(policy="random", render_3d=False)` smoke confirmed the top-down
  path still records frames + session after the refactor (`render=topdown`).
- 3-D path verified by construction (BaseEnv.render `text=` overlay) + clean imports; the live
  webcam+window run is the user's machine (no camera/display on this box).

## Hypothesis vs. reality
Matched, with one in-flight bug caught: the live window only opened for `policy=="gesture"` exactly,
so `gesture-discrete` ran headless (webcam read, no window). Root-caused to the `show` default
(`policy=="gesture"`), fixed to `policy.startswith("gesture")`. The position+pose scheme is the
robust win -- steering and throttle are independent axes, so the "can't fist and point at once"
limitation of the first scheme is gone, and reverse is a deliberate two-hand pose instead of a
flaky motion. The earlier pointing/latch logic (`classify_gesture` + `command_to_action`) is kept
as tested utilities / an alternative mode.

## Failures / debugging
- `show`-default bug above (one-line fix).
- Stale terminal scrollback looked like the 3-D change "didn't work" -- it was pre-fix output
  (identical timestamps, `Render Mode: none`, old print format); a fresh run renders 3-D.

## One-line takeaway (the interview sentence)
> Turning the proven feedback demo into something a human actually enjoys driving was pure
> action-source / env-config work -- a rendered 3-D window with an on-screen HUD, choosable scenes,
> and a two-axis "position steers + pose throttles" hand scheme (two-hand prayer = reverse) -- none
> of which touched the world-model core, exactly because every input is just another action source.

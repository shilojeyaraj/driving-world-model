# Experiment 017 -- Capstone: feedback with a REAL reference (meaningful signals)

**Date:** 2026-06-16
**Component / change:** none new -- end-to-end demonstration of the gesture-feedback system
(GF1-GF5) with a real, trained reference instead of the untrained smoke checkpoint.

## Hypothesis (write BEFORE running)
The headless GF2+GF5 smoke (experiments/016) produced garbage feedback only because the reference
was untrained. With a real reference (WM + BC-IDM actor + eval critic from train_reference), the
3 signals should become MEANINGFUL: survival ~1 when the driver stays alive, small steer/throttle
deviation when the driver matches the expert, a real-scale critic value.

## Setup
- Trained the reference on MetaDrive IDM (`train_reference`, collect 3000 / wm 800 / bc 800 /
  critic 800): **recon=0.579, bc_loss=0.0159 (the actor cloned IDM well), critic_loss=19.1.**
- Drove a `forward` policy (throttle 0.4, steer 0) for 120 steps vs that reference, with the live
  HUD, then ran the offline report.

## Result
- **steps=120, risk_steps=0, mean_safety=1.000, mean_value=82.4, mean|steer_dev|=0.031, events {}.**
- HUD frame confirmed: full-green safety bar, "steer dev +0.03 throttle dev -0.07, value +82.29"
  over the top-down view of the car on the road.

## Hypothesis vs. reality
Matched. With a trained reference the signals are sensible: on this (near-straight) segment the
forward policy stays alive (continue head -> survival ~1), and its steering closely matches the
BC-expert (deviation 0.03), so nothing flags. On curvy segments / aggressive driving the
deviations and risk would fire -- which is exactly the intended feedback. The whole pipeline
(gesture/forward source -> MetaDrive -> world-model feedback -> HUD + report) is demonstrated
end-to-end; swapping the action source to the webcam gesture controller is the only remaining
step for a live human demo.

## Failures / debugging
None. The standard {wm, actor, critic, config} checkpoint meant train_reference's output dropped
straight into the feedback engine via load_models with zero glue.

## One-line takeaway (the interview sentence)
> With a behavior-cloned-IDM reference, the world-model feedback engine produces real signals --
> survival ~1 from the continue head, a real-scale critic value (~82), and a 0.03 steering
> deviation from the expert -- turning "drive the sim" into "here's how your driving compares to
> an expert, and where it's risky," entirely from a learned model.

# Gesture-Control + Driving-Feedback — Design Spec (v1, state mode)

Date: 2026-06-15
Owner: Shilo
Implementer: Claude (writes all code) — Shilo learns the code + architecture as it's built.

---

## 1. Goal & non-goals
**Goal.** Drive MetaDrive with hand gestures (webcam → `[steer, throttle]`), and run a
**3-signal feedback engine** that critiques the driving — (A) outcome forecast, (B) style-vs-
expert, (C) value — with a live HUD and a session report. Everything reuses the existing
contract: a gesture is just another *action source* feeding `env.step`; the feedback reads the
same obs/action stream.

**Non-goals (v1).** Real-car transfer; image-mode feedback (state mode first); a *strong* RL
policy (we use MetaDrive's IDM expert as the reference, not our actor — see `experiments/011`).

## 2. Architecture & data flow
```
webcam ─cv2─► MediaPipe landmarks ─► landmarks_to_action() ─► [steer,throttle] ─► env.step
                                                                  │                  │ obs
                                              DrivingFeedback.step(obs, action) ◄─────┘
                                                  ├ A forecast_safety (rssm.imagine + decoder cont/reward)
                                                  ├ B style_deviation (reference Actor)
                                                  └ C state_value     (Critic)
                                              → per-step signals → HUD;  finalize() → report
```
The gesture controller is an **action source** (like `random_policy` / `Actor`), so it does
**not** touch `envs/base.py`.

## 3. Conventions reused (load-bearing — identical to the codebase)
- **Action**: 2-D `[steer, throttle] ∈ [-1,1]` (`cfg.action_dim=2`); mapping output is
  `np.clip(..., -1, 1).astype(np.float32)`.
- **RSSM state-carry for feedback** mirrors `eval/closed_loop.py:_run_actor_episode`:
  `state = rssm.initial_state(1, device)`; per step `state,_,_ = rssm.obs_step(state, prev_action, e)`;
  `feat = torch.cat(state, dim=-1)`.
- **Forecast uses the prior MEAN**, not samples: `rssm.imagine(actions, state, sample=False)`
  then `decoder(feat, decode_obs=False)` (the open-loop/prediction convention, `experiments/003`).
- **Checkpoint format already fits**: `{world_model, actor, critic, config}`
  (`utils.save_checkpoint` / `load_models`). The reference stack = `actor` (reference) + `critic`
  (evaluator), so `load_models` returns exactly what the feedback engine needs. No new ckpt code.

## 4. Components (file paths + signatures, following patterns)

### 4.1 `control/gesture.py` (new package; "OK TO USE A LIBRARY / modify freely")
- `landmarks_to_action(steer_raw, throttle_raw, prev=None, *, deadzone, smoothing) -> np.ndarray`
  — PURE (deadzone + EMA smoothing + clip). Testable with synthetic numbers, no webcam — mirrors
  `envs/metadrive_env.py:adapt_obs`.
- `class GestureController:` `__init__(cfg)`, `get_action() -> np.ndarray`, `calibrate()`,
  `close()`. Lazily imports `cv2` + `mediapipe` (so the module imports without them); extracts
  hand-center-x → `steer_raw` and hand height/openness → `throttle_raw`, delegates to the pure fn.

### 4.2 `training/train_reference.py` (mirrors `training/train_behavior.py`)
`train_reference(cfg, ...) -> (wm, ref_actor, critic)`:
1. Collect IDM-expert data (MetaDrive `agent_policy=IDMPolicy`) into a `SequenceReplayBuffer`.
2. Train `WorldModel` via `assemble_loss` (existing loop).
3. **Behavior-clone** the reference `Actor` on `(feat, idm_action)` (MSE; small supervised loop).
4. Train `Critic` by **policy-evaluation of IDM** using `lambda_returns` on IDM trajectories.
5. `save_checkpoint("runs/reference/ckpt.pt", wm, ref_actor, critic, cfg)`.

### 4.3 `eval/feedback.py` (mirrors `eval/open_loop.py`; Concept/Question header)
- `forecast_safety(world_model, state, action, horizon) -> dict` — tile action over horizon,
  `imagine(sample=False)`, decode → `{"survival": Πσ(cont_logit), "pred_return": Σreward,
  "risk": survival < thr}`. **Metric A.**
- `style_deviation(reference_actor, feat, action) -> dict` — `ref,_=reference_actor(feat,
  deterministic=True)` → `{"d_steer","d_throttle"}` (L2; no model change). *Enhancement:*
  `Actor.log_prob(feat, action)` (Tanh-Normal w/ correction) for a probabilistic version. **Metric B.**
- `state_value(critic, feat) -> float`. **Metric C.**
- `class DrivingFeedback:` `__init__(wm, ref_actor, critic, cfg)`, `step(obs, action) -> dict`
  (carries RSSM state per §3), `finalize() -> report`.
- `report_from_traces(traces, cfg) -> dict` — PURE: percentile-normalize, threshold → labeled
  events (`late_brake`, `oversteer_right`, `near_off_road`, `low_value`), aggregate, stats.

### 4.4 Scripts (`sys.path` header, `main()`, `__main__`)
- `scripts/drive_gesture.py`: live loop GestureController → MetaDrive.step → topdown render + HUD
  + `DrivingFeedback.step`; saves a session GIF (reuse `scripts/record_metadrive.py:_frame`) + buffer.
- `scripts/feedback_report.py`: load recorded session + `load_models("runs/reference/ckpt.pt")` →
  `DrivingFeedback` offline → report + flagged clips.

## 5. Config additions (`config.py`, sectioned style, "modify freely")
```python
# --- gesture / feedback ---
webcam_id: int = 0
gesture_smoothing: float = 0.7
gesture_deadzone: float = 0.1
forecast_horizon: int = 15
risk_threshold: float = 0.5
```

## 6. Checkpointing
Reuse `utils.save_checkpoint`/`load_models` with the existing
`{world_model, actor, critic, config}` format (actor = reference, critic = evaluator).

## 7. Tests (`tests/`, slow-marker + `pytest.importorskip`, like `test_metadrive_adapter.py`)
- `tests/test_gesture.py` — FAST: `landmarks_to_action` range/monotonic/smoothing on synthetic
  inputs; SLOW `importorskip("mediapipe")` live-capture smoke.
- `tests/test_feedback.py` — FAST: tiny WM/Actor/Critic + synthetic obs/actions; assert
  `forecast_safety` keys + `survival∈[0,1]`, `style_deviation` shapes, `report_from_traces`
  flags an injected event. No MetaDrive.

## 8. Build phases (each ends runnable + an `experiments/NNN_*.md` log)
| Phase | Deliverable | Reuses | Gate |
|---|---|---|---|
| GF1 | `control/gesture.py` (pure map + live capture) | — | range/smoothing tests; action moves with hand |
| GF2 | `scripts/drive_gesture.py` (gesture → MetaDrive → topdown) | make_env, recorder | drive by hand; session recorded |
| GF3 | `training/train_reference.py` (WM + BC Actor + eval Critic on IDM) | collect, assemble_loss, lambda_returns | open-loop beats no-action; IDM ref sane |
| GF4 | `eval/feedback.py` (A/B/C metrics) | imagine, decoder, Actor, Critic | sane traces on a recorded session |
| GF5 | fusion + report + live HUD | recorder, load_models | HUD demo + session report |

**MVP "sick demo"** = GF1+GF2 + GF4(A via the continue head, B via deviation) + GF5 HUD.

## 9. Risks & mitigations
- **WM/critic weak on MetaDrive** → lead Metric A with the **continue head** (crash forecast — the
  signal that learned cleanly, `experiments/010`); treat reward/value as soft. Train the reference
  stack on **IDM data**, not random.
- **Reference quality** → IDM/BC, not our actor (`experiments/011`).
- **Real-time budget** → control at full fps; feedback every k frames; state mode on CPU.
- **Gesture noise/calibration** → deadzone + EMA + `calibrate()`; the pure mapping is unit-tested.
- **Sim ≠ real** → feedback on driving *in MetaDrive*, a proxy.

## 10. DreamerV3 note
The reference Critic via **policy-evaluation of IDM** reuses `lambda_returns` directly. The
symlog/return-norm stabilizers are NOT reintroduced (reverted, `experiments/012`); the reference
avoids corner-collapse by imitating IDM rather than RL-from-scratch.

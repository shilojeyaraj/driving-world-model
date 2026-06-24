# Problems We Hit & How We Dealt With Them

A running log of the real failures encountered building this driving world model — symptom, root
cause, and resolution — so we (and reviewers) don't re-learn them the hard way. Grouped by area.
Per-experiment detail lives in `experiments/NNN_*.md`; code references point at the fix.

> **The one meta-lesson:** the **prediction** side (the world model) works well; the **control**
> side (getting a *learned* policy to drive well in closed loop) is the hard, partly-open problem.
> Most pain below clusters there.

---

## A. Rendering & laptop performance

### A1. PSSM shadow crash when disabling shadows
- **Symptom:** `AssertionError: distance > 0.0 && distance < 100000.0` at `pssmCameraRig` whenever
  the 3-D window opened.
- **Cause:** we set `shadow_range=0` to cheapen rendering, but MetaDrive's PSSM shadow system always
  initializes when rendering and asserts a *positive* shadow range.
- **Fix:** never zero `shadow_range`. Disable shadows at **runtime** instead, after the engine
  exists, via `disable_shadows(env)` → `env.engine.pssm.toggle_shadows_mode()` (best-effort, never
  raises). `envs/metadrive_env.py`; tests assert `"shadow_range" not in md`.

### A2. 3-D render too choppy on a weak laptop
- **Symptom:** low frame-rate, stuttering live drive on an Intel Iris Xe (integrated GPU, no CUDA).
- **Cause:** rendering resolution + shadows/skybox + per-frame feedback forecast oversubscribe one
  integrated chip; there is no software that *adds* GPU power.
- **Fix:** ask for less — `metadrive_window_size` (default 800×600, down from 1200×900),
  `metadrive_low_graphics` (drop skybox/logo + runtime shadow-off), `metadrive_traffic_density`,
  and OS-level "Best performance" + plugged in. Documented in `docs/RUNNING.md` "weak laptop".

### A3. Live feedback forecast cost every frame
- **Symptom:** the live HUD made the loop crawl.
- **Cause:** the safety forecast is a 15-step world-model imagine — expensive — run every frame.
- **Fix:** `gesture_feedback_every` (default 3) recomputes the forecast every k frames; the HUD
  reuses it between, cheap style/value signals still update every frame. `eval/feedback.py`.

---

## B. Episode control (MetaDrive terminations)

### B1. "It freezes, then resets me at the start"
- **Symptom:** a few seconds into a hand-driven session the car snapped back to the start.
- **Cause:** MetaDrive ends the episode by default on off-road / continuous-line / any collision —
  and a human practising triggers these constantly.
- **Fix:** `metadrive_endless` turns the early-termination flags off so a human can drive through
  mistakes; added live logging (`[step N] reset -- <reason>`) so resets are explained, not silent.
  `envs/metadrive_env.py`, `scripts/drive_gesture.py`.

### B2. Endless mode *still* reset at the horizon
- **Symptom:** even with terminations off, the session reset around step 200.
- **Cause:** the episode still **truncates** at `horizon = max_episode_steps`.
- **Fix:** endless mode also sets `horizon = 1_000_000_000` → no reset at all. (commit `414315b`)

### B3. Ctrl+C threw away a long session
- **Symptom:** stopping a long drive with Ctrl+C lost all the recorded data.
- **Cause:** the save ran after the loop, so `KeyboardInterrupt` skipped it.
- **Fix:** `except KeyboardInterrupt` *inside* the loop saves what was collected; guard against an
  empty recording. `scripts/drive_gesture.py`.

---

## C. Input / gesture control (Windows)

### C1. OpenCV gesture steering "bugs out and resets the car"
- **Symptom:** turning via webcam gestures repeatedly glitched and reset the car.
- **Cause:** noisy/jittery hand tracking + termination-on-mistake made closed-loop gesture driving
  fragile on this hardware.
- **Fix (pragmatic pivot):** added **WASD keyboard driving** (`drive_gesture keyboard`, MetaDrive
  `manual_control`) as the reliable data-collection path; gesture modes kept but no longer the
  default for recording. (commit `64cddc0`)

### C2. `DLL load failed while importing _pywrap_tensorflow_internal`
- **Symptom:** gesture script crashed at import on Windows.
- **Cause:** MediaPipe pulls in TensorFlow, whose native DLLs fail to initialize if loaded **after**
  MetaDrive/panda3d.
- **Fix:** import MediaPipe (build the gesture controller) **before** creating the MetaDrive env.
  `scripts/drive_gesture.py` does this; documented in `docs/RUNNING.md` ops notes. (commit `878f915`)

### C3. MediaPipe legacy `solutions` API gone
- **Symptom:** gesture controller failed to construct.
- **Cause:** the old `mediapipe.solutions` hands API was removed.
- **Fix:** switched to the MediaPipe **Tasks** `HandLandmarker`. (commit `2d887fd`)

---

## D. Training the world model + RL actor (the control wall)

### D1. Single-shot model exploitation
- **Symptom (exp 010):** an actor trained once on a world model built from *random* data drove
  worse than random.
- **Cause:** the world model was only accurate on random-policy states; the actor exploited its
  errors elsewhere.
- **Fix:** the **iterated Dreamer loop** (`scripts/run_metadrive.py` / `training/dreamer_loop.py`):
  collect-with-policy → retrain WM → retrain actor, so the WM is grounded in policy-visited states.
  Necessary — but not sufficient (see D2).

### D2. RL actor corner-collapse — `steer=-1, throttle=+1`
- **Symptom (exp 011, and again this session):** closed-loop `actor_return` *below* random; the
  policy saturates to full-left + full-throttle. After a bigger run, `imagined_return` even fell to
  *match* the (bad) real return — the imagination gap closed, but the actor stayed collapsed.
- **Cause:** the Tanh-Normal actor's mean is driven to an extreme by an **unnormalized value
  gradient** under MetaDrive's **weak/sparse reward**; once `tanh` saturates, `d tanh/d mean ≈ 0`
  and there's no gradient to escape the corner.
- **Mitigations applied:** raised the entropy bonus `entropy_coef` 1e-3 → **1e-2** and exposed it
  (+ all training sizes) as CLI flags so it's tunable (`scripts/run_metadrive.py`, commit `2a4e35b`).
- **Status:** still the documented **open** problem for pure RL control. The principled cure is the
  full DreamerV3 stack (see D3); we pivoted to imitation (section E) for a policy that actually drives.

### D3. DreamerV3 stabilizers (simplified) backfired — reverted
- **Symptom (exp 012):** adding return-normalization + scalar `symlog` reward/value caused (1) NaNs
  and (2) corner-collapse **even on the easy toy**.
- **Cause:** `symexp` of an *unbounded* scalar head overflows → NaN; a per-batch return scale is too
  noisy and interacts badly with the iterated loop's self-collected data. DreamerV3 avoids both with
  **bounded two-hot** reward/value heads + an **EMA** return scale — substantially more machinery.
- **Fix:** **reverted** to the last green state rather than ship a fragile stand-in that regressed
  the toy. The proper full-V3 upgrade is logged as out-of-scope-for-one-step.

### D4. Model only knew **one** map
- **Symptom:** every run logged `Num Scenarios : 1`; the policy couldn't handle a new track.
- **Cause:** MetaDrive defaults to a single procedurally-generated map (seed 0); we never set the
  pool.
- **Fix:** **map randomization** — `metadrive_num_scenarios` / `metadrive_start_seed`, a `map=N`
  random-block default for varied geometry, and a **disjoint held-out eval split**
  (`train_eval_seed_split`) so `eval_driving` measures *generalization*, not memorization. (commit
  `60d443d`). *Known remaining gap:* `collect_idm` still gathers IDM data on a single map; only the
  RL/DAgger rollouts use the pool.

---

## E. Imitation learning (BC + DAgger)

### E1. Behavior cloning copies the demonstrator's mistakes
- **Symptom:** the "your-style" model trained on a hand-driven session was bold but crashy
  (off-road 100%, crash 60%) — because the recorded drive contained lots of crashes.
- **Cause:** BC blindly imitates the action taken in each state; it has no notion of good vs bad.
- **Fix / guidance:** record **cleaner** drives and **recover after mistakes** (recovery data is the
  valuable part); accumulate many drives (`runs/sessions/`, `train_on_gesture`) so one bad drive is
  diluted. Volume alone is *not* exponential improvement — quality + recovery data is the lever.

### E2. BC distribution shift — route ~3%
- **Symptom:** the IDM-cloned reference actor only completed ~3% of the route (off-road 20%).
- **Cause:** plain BC only ever sees the expert's *good* states; once the learner drifts off-center
  it has never seen a recovery → compounding error.
- **Fix:** **DAgger** — roll out the current learner, have IDM relabel the states it actually visits,
  aggregate, retrain. (`training/dagger.py`, `scripts/dagger.py`, commit `f7c5ca0`). First DAgger
  round jumped held-out route **3% → 23%**.

### E3. DAgger `bc_loss` exploded to ~12,000,000 and collapsed the policy
- **Symptom:** after a promising iter-1 (route 23%), iter-2's `bc_loss` blew up to ~12M and every
  later round idled at route ~1%.
- **Root cause (found by instrumentation, not guessing):** at the near-collision states the learner
  drives into, MetaDrive's IDM returns an **un-normalized emergency-brake action** (observed
  throttle ≈ **−198**). Stored raw as a BC target, the tanh-bounded actor (max ±1) can never match
  it, so the MSE exploded. The world model was *never* at fault (recon/feat healthy, `pred` capped
  at ±1). Earlier probes drove too gently to trigger it.
- **Fix:** `relabel_action()` — `nan_to_num` **then** clip to `[-1,1]`. The clip handles −198 (full
  brake *is* −1, which `env.step` applies anyway); `nan_to_num` handles NaN/inf from degenerate IDM
  states (`np.clip` leaves NaN as NaN → silent NaN-loss collapse). Verified end-to-end: `max|target|`
  stays 1.0 and `bc_loss` 0.01–0.05 across iterations. (commit `3488908`)

### E4. DAgger hardening (neighboring failure modes)
- **Buffer eviction:** the replay buffer evicts *oldest* episodes first — the clean iter-0 IDM base.
  `dagger_capacity()` sizes the buffer to hold all planned data so a long run never discards it.
- **Non-finite observations:** a fail-loud `assert np.all(np.isfinite(obs))` in the rollout, so a
  bad obs can't silently poison the WM/critic (dagger_train saves each iter, so failing loud is safe).

---

## F. Evaluation & measurement

### F1. Evaluated the wrong checkpoint
- **Symptom:** an eval "showed no improvement" after training.
- **Cause:** ran `eval_driving runs/reference/ckpt.pt` (the old IDM-cloned baseline) instead of the
  freshly trained `runs/gesture_reference/ckpt.pt`.
- **Fix:** awareness + clearer docs labelling which checkpoint each script writes/reads
  (`runs/reference`, `runs/gesture_reference`, `runs/metadrive`, `runs/dagger`).

### F2. `success_rate` reads 0% even for the IDM expert
- **Symptom:** success rate 0% across the board, confusing.
- **Cause:** reaching the destination needs a long enough `max_episode_steps`; short horizons can't
  complete the route even for IDM.
- **Fix:** lean on **route completion %** as the primary signal on short horizons; documented in
  `docs/RUNNING.md` §8b "Reading it".

---

## G. Tooling & workflow

### G1. `python -c "…"` `IndentationError` in PowerShell
- **Cause:** pasting a multi-line `-c` string makes PowerShell add continuation lines with leading
  spaces → Python sees indentation.
- **Fix:** one-line invocations, and ultimately **argparse CLI flags** on `run_metadrive` / `dagger`
  so no `python -c` is needed at all.

### G2. Long background runs show no interim output
- **Symptom (exp 011 + this session):** piping a long run through `grep`/`Select-String` shows
  nothing until it exits.
- **Cause:** the filter **block-buffers** the pipe.
- **Fix:** don't filter (or use line-buffering) when you need live progress; otherwise read the
  task output file directly.

### G3. `train_behavior` looked frozen
- **Symptom:** long silent training looked hung.
- **Cause:** no progress logging during the grad loop.
- **Fix:** per-stage progress prints + a documented fast (~5–8 min) command. (commit `8d25ffa`)

### G4. `dreamer_train` called `env.close()` on a DummyEnv that lacked it
- **Symptom:** crash on teardown after a 7-min run.
- **Fix:** added a no-op `close()` to the base env (real sims override). Lesson: smoke the teardown
  path too.

### G5. DonkeyGym install friction (optional path)
- **Symptom:** `np.bool8` / `np.float_` errors; `asyncore` missing on Python ≥3.12.
- **Cause:** `gym 0.26` vs NumPy 2; `asyncore` removed from the stdlib.
- **Fix:** run DonkeyGym in a Python 3.11 env with `numpy<2`; `pip install pyasyncore pyasynchat`.
  Documented in `docs/RUNNING.md` §7b.

---

## Cross-cutting lessons
1. **The world model is the reliable part; closed-loop control is the wall.** Both RL (corner
   collapse) and plain BC (distribution shift) fall short; only hand-coded IDM drives well. DAgger
   is the most promising *learned* path so far.
2. **Instrument, don't guess.** The 12M `bc_loss` looked like a WM divergence; instrumentation
   proved it was a single un-normalized target (−198). Root-cause-first beat guess-and-check.
3. **Clip/normalize at every boundary that crosses into the model.** Actions and observations from a
   physics sim can be out-of-range, un-normalized, or non-finite in edge states the happy path never
   visits.
4. **Don't ship a fragile fix.** The simplified DreamerV3 stabilizers regressed the toy, so we
   reverted rather than mask the problem (exp 012).
5. **TDD + a fast green suite** caught regressions immediately and made each fix verifiable.

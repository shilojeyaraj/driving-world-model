# DAgger (Dataset Aggregation) — Imitation Learning with Expert Relabeling — Design Spec

Date: 2026-06-24
Owner: Shilo
Implementer: Claude (writes all code) — Shilo learns the code + architecture as it's built.

---

## 1. Goal & non-goals
**Goal.** Train a *learned* driving policy that actually drives the (randomized) track in closed
loop, by fixing the **distribution-shift** failure of plain behavior cloning. Plain BC of the IDM
expert (`runs/reference/ckpt.pt`) only sees the expert's *good* states, so when the learner drifts
off-center it has never seen a recovery and fails (measured: route **3%**, off-road 20%). DAgger
repeatedly rolls out the *current learner*, lets it drift into the states it actually visits, and
has the **IDM expert relabel each of those states** with the correct action — then retrains on the
growing aggregated dataset.

**Non-goals.** Beating IDM (it's the teacher — IDM is the ceiling); RL / reward maximization (this
is pure imitation); image mode; online/in-the-loop learning. We commit to **true DAgger** (roll out
learner, query IDM at visited states); DART (noise-injected expert) is the agreed fallback **only**
if the IDM-query API can't be made to work (see §4).

## 2. Why this works (and what was missing)
- Closed-loop control fails for both learned policies we have: the RL actor collapses to saturated
  actions (`experiments/011`, the `run_metadrive` runs), and BC-from-IDM hits route 3% from
  distribution shift. Only hand-coded IDM drives well (route 99%) — but it isn't a learned model.
- DAgger is the textbook reduction of imitation learning to no-regret online learning (Ross,
  Gordon & Bagnell, 2011): aggregate `D ← D ∪ {(s visited by learner, expert(s))}`, retrain, repeat.
  It directly supplies the recovery data plain BC lacks.

## 3. Architecture & data flow
Each DAgger iteration:
```
roll out the CURRENT BC actor in MetaDrive (TRAIN-pool maps), carrying RSSM state:
    obs ─► wm.encode ─► rssm.obs_step ─► actor(feat) ─► a_actor      (the learner drives)
    query IDMPolicy.act(ego) at the SAME state      ─► a_idm         (expert's correct label)
    step env with a_actor   ← car DRIFTS like the learner (visits recovery/off-road states)
    record (obs, a_idm, reward, done)               ← obs the learner caused, IDM's fix
  ─► AGGREGATE into the growing buffer (iter-0 clean IDM data + all relabeled pairs)
  ─► retrain WM (so it can encode the new off-distribution states), BC actor, critic
  ─► eval_driving on HELD-OUT maps → track route completion
```
The car is **driven by the learner** (to collect the states it screws up in); each obs is **labeled
by IDM** (the correct recovery action). Iteration 0 is exactly `train_reference`: clean IDM data →
WM → BC actor → critic.

## 4. The technical crux — query IDM at a state it isn't driving (FIRST task)
The whole approach depends on getting `a_idm` at a learner-visited state. Plan: after `reset()`,
instantiate an `IDMPolicy` bound to the ego vehicle and call `policy.act(agent_id)` each step to get
the action IDM *would* take, while stepping the env with `a_actor`. **This is verified in a ~10-line
headless probe BEFORE writing the loop.** Acceptance: the call returns a finite 2-vector in
`[-1,1]`-ish range that varies sensibly with state. If MetaDrive won't expose this cleanly, fall
back to **DART**: drive with `agent_policy=IDMPolicy` but inject action noise so IDM visits
off-center states and records its own corrective actions (no learner-querying) — same recovery-data
payoff, lower API risk.

## 5. Components (file paths + signatures, following existing patterns)

### 5.1 `training/dagger.py` (new)
- `idm_relabel_rollout(cfg, wm, actor, steps, seed=0) -> SequenceReplayBuffer`
  — the crux. Drive the learner (RSSM state-carry mirrors `eval/closed_loop` / `eval_driving:_ActorPolicy`),
  query IDM each step, record `(obs, a_idm, reward, done)`. Headless.
- `dagger_train(cfg, iters=3, collect_steps=4000, rollout_steps=2000, wm_steps=1000, bc_steps=1000,
  critic_steps=1000, out="runs/dagger/ckpt.pt", eval_episodes=5) -> (wm, actor, critic)`
  — iter-0 = `collect_idm` + WM + BC + critic; each subsequent iter: `idm_relabel_rollout` →
  aggregate into the buffer → retrain WM + BC + critic on the aggregate → `save_checkpoint` →
  (if `eval_episodes>0`) eval on the held-out pool and print route/success/crash.

### 5.2 `scripts/dagger.py` (thin entry)
- `build_cfg(num_scenarios, road_map, traffic_density)` and `parse_args(argv)` mirroring
  `scripts/run_metadrive.py`: `--iters --collect --rollout-steps --wm-steps --bc-steps
  --critic-steps --num-scenarios --map --out --eval-episodes` (`--eval-episodes 0` to skip
  per-iter eval). `main(...)` calls `dagger_train`.

### 5.3 Reused unchanged
`collect_idm`, `bc_actor`, `eval_critic` (from `training/train_reference`); `_train_world_model`
(from `training/dreamer_loop`); `train_eval_seed_split` + the held-out forcing in `scripts/eval_driving`;
`SequenceReplayBuffer.add`; `utils.save_checkpoint`. Output is the standard `{wm, actor, critic, cfg}`
checkpoint, so `eval_driving` / `watch_metadrive_3d` work on `runs/dagger/ckpt.pt` with no changes.

## 6. Map-randomization + eval integration
Rollouts + IDM collection run across the **train pool** (seeds `0…num_scenarios-1`, via the existing
`metadrive_num_scenarios`/`metadrive_start_seed` knobs). Per-iteration eval calls the same path
`eval_driving` uses, which forces the **disjoint held-out pool** (`train_eval_seed_split`) — so
progress is measured on maps never trained on. **Per-iteration eval is ON by default**
(`eval_episodes=5`) so you can watch route completion move; `--eval-episodes 0` turns it off.

## 7. Testing (TDD)
Live rollout needs MetaDrive, so (matching the codebase) unit-test the **pure** pieces and smoke the
live one:
- `scripts/dagger.py`: `parse_args` exposes every knob with sane defaults; `build_cfg` threads the
  map pool into cfg. (no MetaDrive)
- `training/dagger.py`: a pure **aggregation** helper — combining the iter-0 buffer with a relabel
  buffer preserves total steps and episode boundaries. (no MetaDrive)
- `pytest.importorskip("metadrive")` smoke for `idm_relabel_rollout`: returns a buffer with
  `obs/action/reward/done` of the right shapes after a few steps, action finite and 2-D.

## 8. Honest expectation
DAgger is the right tool and should beat route 3% meaningfully, but it's iterative and bounded by
IDM's quality and how much of the recovery-state space we cover. Expect *improvement*, not instant
route-99%. Each iteration is a (slow, CPU) MetaDrive run; the CLI knobs keep runs short while
tuning.

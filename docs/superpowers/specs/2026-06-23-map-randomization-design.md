# Map Randomization + Held-Out Eval Split — Design Spec

Date: 2026-06-23
Owner: Shilo
Implementer: Claude (writes all code) — Shilo learns the code + architecture as it's built.

---

## 1. Goal & non-goals
**Goal.** Train and evaluate driving policies across **many procedurally-generated maps** instead
of the single fixed map we use today (`Num Scenarios : 1` in every log). This is *domain
randomization*: a policy that scores well across a varied map pool has learned to **drive**, not to
memorize one road. Add a **held-out eval split** so `eval_driving` measures true generalization —
driving maps the policy provably never trained on.

**Non-goals.** Changing the world-model / actor-critic architecture; image mode; the endless WASD
hand-drive (no resets there, so map pools don't apply); curriculum/scheduling of difficulty.

## 2. Why this works (and what was missing)
- The observation is the **259-dim state vector** (relative lidar + navigation heading + ego
  dynamics), which is *map-relative* — the policy reacts to "curve right, obstacle at 30°", features
  that exist on any map. So a policy *can* transfer; we just never gave it varied maps to learn from.
- MetaDrive is a **procedural generator**: `num_scenarios` + `start_seed` define a pool of map
  seeds, and each `reset()` samples a seed from `[start_seed, start_seed + num_scenarios)`. We never
  set these, so MetaDrive defaulted to a pool of one (seed 0).
- For varied **geometry** the map must be an **int N** (N random blocks → different
  curves/intersections/roundabouts per seed). A fixed letter-string map (e.g. `SSSS`) gives the same
  road on every seed — only spawn/traffic vary.

## 3. Config knobs (`config.py`) — all default to today's behavior (backward-compatible)
```python
metadrive_num_scenarios: int = 1     # size of the TRAIN map pool (1 = single fixed map, today)
metadrive_eval_scenarios: int = 50   # size of the held-out EVAL pool (disjoint from train)
metadrive_start_seed: int = 0        # first map seed
```
Existing scripts/tests keep `num_scenarios=1, start_seed=0` and are unaffected.

## 4. Components (file paths + signatures, following patterns)

### 4.1 `envs/metadrive_env.py`
- **`metadrive_config(cfg)`** — pass the knobs straight through to MetaDrive:
  ```python
  md["num_scenarios"] = int(getattr(cfg, "metadrive_num_scenarios", 1))
  md["start_seed"]    = int(getattr(cfg, "metadrive_start_seed", 0))
  ```
  Always present (defaults match MetaDrive's own defaults, so it's a no-op for single-map runs).
- **`train_eval_seed_split(num_train, num_eval, base=0)`** — PURE; the disjointness guarantee:
  ```python
  def train_eval_seed_split(num_train, num_eval, base=0):
      """Disjoint map-seed ranges so eval maps are NEVER trained on.
      Returns ((train_start, train_num), (eval_start, eval_num))."""
      return (base, num_train), (base + num_train, num_eval)
  ```
  Testable with plain ints, no MetaDrive.

### 4.2 `scripts/run_metadrive.py` (training entry point)
- Accept a `num_scenarios` arg (default a randomized pool, e.g. **100**); set
  `cfg.metadrive_num_scenarios = num_scenarios`, `cfg.metadrive_start_seed = 0` (the *train* range
  from `train_eval_seed_split`).
- If `road_map is None`, default `cfg.metadrive_map = 3` (3 random blocks) so training sees varied
  geometry. The checkpoint saves this, so eval inherits matching variety.
- Print the train seed range so it's visible (e.g. "training on map seeds 0–99").

### 4.3 `scripts/eval_driving.py` (eval entry point)
- After loading the checkpoint's cfg, compute the **eval** range:
  ```python
  _, (eval_start, eval_num) = train_eval_seed_split(
      cfg.metadrive_num_scenarios, cfg.metadrive_eval_scenarios)
  cfg.metadrive_start_seed, cfg.metadrive_num_scenarios = eval_start, eval_num
  ```
  This forces eval onto seeds **after** the training range — maps never seen in training. (An old
  checkpoint with `num_scenarios=1` evaluates on seeds `1 … 50`, still disjoint from trained seed 0.)
- Print the eval seed range in the header so "held-out" is visible (e.g.
  "held-out maps: seeds 100–149"). The ACTOR / RANDOM / IDM comparison is unchanged otherwise.

## 5. Tests (TDD, fast / no MetaDrive)
- `test_metadrive_adapter.py`: `metadrive_config` emits `num_scenarios` / `start_seed` from cfg;
  a default cfg yields `1` / `0` (backward compat).
- `test_metadrive_adapter.py` (or a small new test): `train_eval_seed_split` — ranges are disjoint
  (`train_start+train_num == eval_start`), correct sizes, and honor a non-zero `base`.

## 6. Scope guard
Only the RL training loop (`run_metadrive`) and `eval_driving` change behavior. `drive_gesture`
(endless WASD), `train_on_gesture`, `watch_metadrive_3d`, and `record_metadrive` are untouched —
they keep the single-map default unless their cfg says otherwise.

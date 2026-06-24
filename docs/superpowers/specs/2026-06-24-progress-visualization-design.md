# Progress Visualization — Learning Curve over Training — Design Spec

Date: 2026-06-24
Owner: Shilo
Implementer: Claude (writes all code) — Shilo learns the code + architecture as it's built.

---

## 1. Goal & non-goals
**Goal.** A presentable "is the car learning?" chart: **route completion % vs DAgger iteration**, with
dashed reference lines for the **Random** floor and the **IDM** ceiling — so the 3%→23%→… story is a
single legible image to show other people. Metrics are captured to a dependency-free CSV during
training; a separate (optional-matplotlib) script renders the PNG.

**Non-goals.** A live/streaming dashboard; multi-metric panels (chose the single curve); wiring into
`run_metadrive` (the logger is generic so it *can* later, but not now); web/interactive plots.

## 2. Architecture & data flow
```
dagger_train, each iteration:
   held-out eval (already runs) ─► summary dict
                                ─► append_progress(runs/dagger/progress.csv, progress_row(it, summary))
   (iteration 0 only)           ─► write runs/dagger/baselines.json {random_route, idm_route}
scripts/plot_progress.py:
   read_progress(progress.csv)  ─► line: route_completion (%) vs iter
   read baselines.json          ─► dashed Random (floor) + IDM (ceiling) horizontal lines
                                ─► runs/dagger/progress.png
```
Capture is **stdlib `csv`** (no new dependency); only `plot_progress` imports matplotlib (guarded).

## 3. Components (file paths + signatures)

### 3.1 `training/progress_log.py` (new; stdlib only)
- `progress_row(it, summary) -> dict` — assemble a CSV row from an eval summary:
  `{iter, route_completion, success_rate, crash_rate, off_road_rate, mean_return}`. PURE.
- `append_progress(path, row) -> None` — append `row` to the CSV, writing the header **once** (when
  the file is new/empty); creates parent dir. Columns fixed/ordered.
- `read_progress(path) -> list[dict]` — read the CSV back to dicts (numbers as floats). PURE round-trip.

### 3.2 `scripts/eval_driving.py` (small refactor — behaviour preserved)
- `main(...)` already computes `summarize_driving(...)` for ACTOR/RANDOM/IDM and prints each. Change:
  also **return** `{"actor": <summary>, "random": <summary>, "idm": <summary|None>}`. Printing is
  unchanged; existing CLI behaviour unchanged.

### 3.3 `training/dagger.py` (wire logging into the loop)
- After each iteration's `_eval_heldout`, capture the returned summaries and
  `append_progress("runs/dagger/progress.csv", progress_row(it, summaries["actor"]))`.
- At iteration 0, also write `runs/dagger/baselines.json` with the random + IDM route completion
  (IDM eval is run only at iter 0 to avoid repeating the slow baseline every round).
- Best-effort: a logging/plot-data failure must never kill training (wrap in try/except like
  `_eval_heldout`).

### 3.4 `scripts/plot_progress.py` (new)
- `main(progress_csv="runs/dagger/progress.csv", baselines="runs/dagger/baselines.json",
  out="runs/dagger/progress.png")`.
- `read_progress` → plot `route_completion*100` vs `iter` (markers+line). If `baselines.json` exists,
  add dashed horizontal lines "Random %" and "IDM %". Title "DAgger — driving skill over training",
  axis labels, legend, y in 0–100. Save PNG; print the path.
- matplotlib import **guarded**: on ImportError, print `pip install matplotlib` and exit cleanly.
- CLI: `python -m scripts.plot_progress [progress.csv] [out.png]`.

## 4. Dependency
Add `matplotlib` to `requirements.txt` under the **optional viz** section (commented, like
`metadrive`/`gym-donkeycar`), with a one-line note. Core install stays `numpy/torch/imageio`; the
plot script's guard tells the user to `pip install matplotlib` if absent.

## 5. Testing (TDD, no MetaDrive)
- `progress_log`: `append_progress` writes the header once and appends rows; a second append does
  **not** duplicate the header; `read_progress` round-trips values; `progress_row` produces the
  fixed keys from a summary dict. (uses a temp file)
- `plot_progress`: `pytest.importorskip("matplotlib")` smoke — given a 3-row CSV (+ baselines), it
  writes a non-empty PNG.

## 6. Scope guard
DAgger only. `progress_log` is generic (any `{route_completion, …}` summary), so `run_metadrive`
can adopt it later without changes here. No other entry points touched.

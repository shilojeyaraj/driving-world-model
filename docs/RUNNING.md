# Running & testing everything

Concrete commands for every part of the project. Run all of them **from the repo root** with
`python -m ...` (that's how the module imports resolve — see "Environment variables" below).

---

## 0. Environment variables — what you need

**None are required.** The code reads **no** environment variables; all configuration is in
`config.py` via `get_config(**overrides)`. You just need to run from the repo root.

- **`python -m <module>` from the repo root** puts the project on `sys.path` (the `scripts/*`
  files also self-insert the root, and `conftest.py` does it for pytest). So you do **not** need
  to set `PYTHONPATH`.
- **GPU (image mode only):** there's no env var in the code — set it in config:
  `get_config(device="cuda", ...)`. Optionally `CUDA_VISIBLE_DEVICES=0` is the standard
  torch/CUDA way to pick a GPU. State mode runs fine on CPU.
- **MetaDrive rendering on a headless Linux server:** needs an OpenGL context. Wrap the command
  in a virtual display: `xvfb-run -a python -m scripts.record_metadrive`. On Windows/desktop it
  works directly (the probe found the `wglGraphicsPipe` backend). MetaDrive auto-downloads its
  assets to its package directory on first run (no env var).
- **Webcam (gesture control):** pick the camera with `get_config(webcam_id=0)` (default 0) — not
  an env var.

---

## 1. Setup

```bash
pip install -r requirements.txt              # numpy, torch, pytest, imageio
```
Optional extras, by feature:
```bash
# real sim (state + image): the PyPI build fails on legacy gym -> use the GitHub source:
pip install "git+https://github.com/metadriverse/metadrive.git"     # see docs/METADRIVE.md
# gesture control:
pip install mediapipe opencv-python
```

## 2. Sanity check (no models, no GPU)
```bash
python scripts/smoke_test.py                 # env + replay buffer + shapes
```

## 3. Tests
```bash
pytest -m "not slow"        # fast unit/contract tests (~seconds) -- the dev loop
pytest                      # everything incl. training-based milestone gates (~minutes)
pytest tests/test_rssm.py   # one file;  add ::test_name for one test
```
Slow tests are training/sim-based gates (marked `@pytest.mark.slow` in `pytest.ini`). Live tests
that need MetaDrive / a webcam `skip` automatically when those are absent.

## 4. Core world model (state mode, DummyEnv — CPU)
```bash
python -m training.train_world_model         # collect (random) -> train the world model (ELBO)
python -m training.train_behavior            # collect -> train WM -> train policy in imagination
python -m scripts.eval_closed_loop runs/behavior/ckpt.pt   # closed-loop driving vs random
```

> **Heads-up on runtime.** The script defaults (10,000 grad steps, `seq_len=50`) take **~1.5–2h
> on CPU** — and that's overkill: the toy converges in ~1–2k steps. For a **~5–8 min** run, shrink
> `seq_len` + `steps`:
> ```bash
> python -c "from config import get_config; from training.train_world_model import train; \
> train(get_config(env='dummy', obs_type='state', seq_len=20, max_episode_steps=200), steps=1500)"
> ```
> Metrics print every 100 steps — you'll see `recon`/`reward` drop and `kl` stay healthy (>0)
> within the first few hundred. (`train_behavior` is similar; pass smaller `wm_steps`/`behavior_steps`.)

## 5. Dynamics ablation (GRU vs Mamba-style SSM)
```bash
python -m scripts.ablate_dynamics            # trains both, prints a comparison table
```

## 6. Visual mode (pixels + the "dream")
```bash
python -m scripts.dream_video                # train an image WM -> runs/dream/dream.gif (+ montage)
python -m scripts.drive_from_pixels          # policy trained + driving from pixels (toy)
```

## 7. MetaDrive (real sim)  — see docs/METADRIVE.md
```bash
python -m scripts.probe_metadrive state      # print the real obs dim -> set cfg.state_dim (259)
python -m scripts.run_metadrive              # iterated Dreamer loop on MetaDrive (state mode)
python -m scripts.record_metadrive           # WATCH it: IDM expert drives -> runs/metadrive_drive.gif
```

## 8. Gesture control + driving feedback  — see docs/superpowers/specs/2026-06-15-...
```bash
# 1) train the reference (the "expert" the feedback compares against) -- slow, needs MetaDrive:
python -m training.train_reference                                  # -> runs/reference/ckpt.pt
# 2) drive by hand with a live feedback HUD (needs a webcam + mediapipe):
python -m scripts.drive_gesture gesture runs/reference/ckpt.pt      # -> runs/drive_gesture.gif
# 3) turn a recorded session into a habits report:
python -m scripts.feedback_report                                  # -> runs/feedback_report.json

# headless smoke (no webcam): drive with a random/forward policy but full render + HUD + report:
python -m scripts.drive_gesture random runs/reference/ckpt.pt
```

## 9. Where outputs go
Everything writes under `runs/` (checkpoints `*.pt`, GIFs, `*.npz` sessions, JSON reports) — all
**gitignored**. Experiment logs live in `experiments/` (tracked).

## 10. Operational notes
- **Run heavy jobs one at a time.** Several concurrent trainings/sims oversubscribe the CPU and
  slow everything to a crawl (one run once took 3.7h under contention).
- **MetaDrive is slow on CPU** (real physics); state mode is the CPU-friendly path. Image-mode +
  many iterations want a GPU (Colab/Kaggle).
- Image-mode `image_size` must be a multiple of 16 (the strided CNN encoder/decoder).

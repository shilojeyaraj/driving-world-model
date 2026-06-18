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
> within the first few hundred.
>
> `train_behavior` is the same story (its defaults are `wm_steps=3000`, `behavior_steps=3000`).
> Fast (~5–10 min), with progress logging:
> ```bash
> python -c "from config import get_config; from training.train_behavior import train_behavior; \
> train_behavior(get_config(env='dummy', obs_type='state', seq_len=20, max_episode_steps=200), \
> wm_steps=800, behavior_steps=800)"
> ```
> It prints `[1/2] world model` then `[2/2] behavior` so you can see it progressing, and saves
> `runs/behavior/ckpt.pt`.

## 5. Dynamics ablation (GRU vs Mamba-style SSM)
```bash
python -m scripts.ablate_dynamics            # trains both, prints a comparison table
```

## 6. Visual mode (pixels + the "dream")
```bash
python -m scripts.dream_video                # train an image WM -> runs/dream/dream.gif (+ montage)
python -m scripts.drive_from_pixels          # policy trained + driving from pixels (toy)
```
> No local GPU? See **docs/KAGGLE.md** — run `dream_video(image_size=64, device="cuda")` on
> Kaggle's free GPU for a crisp 64×64 dream in minutes.

## 7. MetaDrive (real sim)  — see docs/METADRIVE.md
```bash
python -m scripts.probe_metadrive state      # print the real obs dim -> set cfg.state_dim (259)
python -m scripts.run_metadrive              # iterated Dreamer loop on MetaDrive (state mode)
python -m scripts.record_metadrive           # WATCH it (top-down GIF): IDM -> runs/metadrive_drive.gif
# rendered 3-D window (the docs look; needs a display, integrated GPU is fine):
python -m metadrive.examples.drive_in_single_agent_env        # MetaDrive's own demo (WASD to drive)
python -m scripts.watch_metadrive_3d                          # 3-D window, IDM expert drives
python -m scripts.watch_metadrive_3d runs/metadrive/ckpt.pt   # 3-D window, OUR trained policy
```

## 7b. DonkeyCar / DonkeyGym (rendered 3-D camera sim)  — see docs/DONKEYCAR.md
```bash
# install (once); on Python 3.12+ also add the asyncore backport shim:
pip install gym-donkeycar
pip install pyasyncore pyasynchat            # only needed on py>=3.12 (asyncore was removed)
# download the Unity sim "DonkeySimWin.zip" from the gym-donkeycar releases, unzip it, then:
```
```powershell
# point at the exe (PowerShell). NOTE the zip nests one level: ...\DonkeySimWin\DonkeySimWin\
$env:DONKEY_SIM_PATH = "C:\Users\shilo\Downloads\DonkeySimWin\DonkeySimWin\donkey_sim.exe"

# smoke-test the round-trip (launches the sim window, one reset + step) -> expect obs (3,64,64):
python -c "from config import get_config; from envs.base import make_env; import numpy as np; e=make_env(get_config(env='donkey', obs_type='image', image_size=64, donkey_level=3, max_episode_steps=200)); o=e.reset(); print('obs', o.shape); o,r,d,i=e.step(np.array([0.0,0.5],np.float32)); print('step', o.shape, 'reward', r, 'done', d); e.close()"

# train the world model on rendered camera frames (image mode -> wants a GPU):
python -c "from config import get_config; from training.train_world_model import train; train(get_config(env='donkey', obs_type='image', image_size=64, device='cuda', deter_dim=128, stoch_dim=32, hidden_dim=128, seq_len=20, max_episode_steps=500), steps=2000)"
```
First Unity launch may hit SmartScreen ("More info -> Run anyway"). If the smoke test throws a
**numpy type error** (`np.bool8`/`np.float_`), that's `gym 0.26` vs NumPy 2 -- run DonkeyGym in a
**Python 3.11 env with `numpy<2`** (the model code is identical there).

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
- **Windows: import MediaPipe before MetaDrive.** MediaPipe pulls in TensorFlow, whose native DLL
  fails to initialize if loaded *after* MetaDrive/panda3d (`DLL load failed while importing
  _pywrap_tensorflow_internal`). `scripts/drive_gesture.py` already does this in the right order;
  if you write your own script using both, build the gesture controller (or `import mediapipe`)
  before creating the MetaDrive env.

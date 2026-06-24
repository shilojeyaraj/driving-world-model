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
python -m scripts.run_metadrive              # Dreamer loop across 100 RANDOM maps (domain randomization)
python -m scripts.record_metadrive           # WATCH it (top-down GIF): IDM -> runs/metadrive_drive.gif
# rendered 3-D window (the docs look; needs a display, integrated GPU is fine):
python -m metadrive.examples.drive_in_single_agent_env        # MetaDrive's own demo (WASD to drive)
python -m scripts.watch_metadrive_3d                          # 3-D window, IDM expert drives
python -m scripts.watch_metadrive_3d runs/metadrive/ckpt.pt   # 3-D window, OUR trained policy
```
> **Pick the scene/map** (highway, intersection, roundabout...) — the `map` arg / config:
> ```bash
> python -m scripts.watch_metadrive_3d - X       # intersection   (S straight, C curve, X intersection,
> python -m scripts.watch_metadrive_3d - SSSS    # highway         O roundabout, T t-junction, r/R ramp; int N = N random)
> python -m scripts.run_metadrive --map X --num-scenarios 100   # FIX the scene to X across 100 seeds
> ```
> **Map randomization (general policy):** with **no** `--map`, `run_metadrive` trains across a *pool*
> of procedurally-generated maps (default **100**; `--num-scenarios 500` for more) so the policy
> learns to *drive* instead of memorizing one road. `eval_driving` then grades it on a **disjoint
> held-out** pool (seeds it never trained on) — true generalization, not memorization.
> ```bash
> # a real (slow) training run, all knobs from the CLI -- no python -c:
> python -m scripts.run_metadrive --iters 12 --wm-steps 1500 --behavior-steps 1000 --collect 2000 --seed-steps 3000
> python -m scripts.run_metadrive --entropy 0.03   # MORE exploration -> fights the full-left/full-throttle collapse
> python -m scripts.run_metadrive --help           # every knob
> ```
> Entropy (`--entropy`, default 0.01, was 0.001) is the anti-collapse lever: it keeps the actor
> trying varied actions instead of locking onto one saturated habit. Config equivalents:
> `cfg.entropy_coef`, `cfg.metadrive_num_scenarios` (train pool), `cfg.metadrive_eval_scenarios` (held-out).

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
# 2a) EASIEST: drive the 3-D window yourself with WASD (no webcam/MediaPipe -- lightest on a laptop).
#     Click the window to focus it; w=gas s=reverse a=left d=right. ENDLESS -- it never resets you
#     (off-road/crash/horizon all disabled); press Ctrl+C in the terminal to stop & save the session.
python -m scripts.drive_gesture keyboard runs/reference/ckpt.pt     # WASD + live feedback HUD, endless
# 2b) drive by hand in MetaDrive's rendered 3-D window (needs a webcam + mediapipe + a display):
python -m scripts.drive_gesture gesture runs/reference/ckpt.pt      # continuous: hand position
python -m scripts.drive_gesture gesture-discrete runs/reference/ckpt.pt   # discrete commands (3-D by default)
python -m scripts.drive_gesture gesture-discrete - SSSS            # 3-D highway, no feedback
python -m scripts.drive_gesture gesture-discrete runs/reference/ckpt.pt 2d  # force the old top-down GIF instead
# 3) turn a recorded session into a habits report:
python -m scripts.feedback_report                                  # -> runs/feedback_report.json
# 4) TRAIN the model on YOUR driving. Each drive (step 2) is auto-archived to runs/sessions/, so
#    they ACCUMULATE -- train on ALL of them (more/varied data = better world model + critic):
python -m scripts.train_on_gesture runs/sessions/                  # all archived drives -> runs/gesture_reference/ckpt.pt
python -m scripts.train_on_gesture runs/gesture_session.npz        # ...or just your latest drive
#    then re-drive critiqued against your OWN style instead of the IDM expert:
python -m scripts.drive_gesture keyboard runs/gesture_reference/ckpt.pt

# headless smoke (no webcam): drive with a random/forward policy but full render + HUD + report:
python -m scripts.drive_gesture random runs/reference/ckpt.pt
```
> **Display:** gesture modes open MetaDrive's **rendered 3-D window** by default (the real view),
> with the feedback + gesture HUD drawn on-screen via `env.render(text=...)`. Add the `2d` flag to
> fall back to the headless top-down GIF (`runs/drive_gesture.gif`) — useful with no display.
>
> **Two gesture modes** (`cfg.gesture_mode`):
> - **continuous** — hand x = steer, hand height = throttle (smooth, analog).
> - **discrete** — *position steers + pose throttles*: **move your hand left/right = steer**,
>   ✊ **closed fist = go forward**, ✋ **open palm = coast/stop**, 🙏 **two hands together (prayer)
>   = reverse**. Steer and throttle are independent, so a **fist held to the right drives forward
>   AND right at the same time** (and you can ease off speed while turning). Same pretrained
>   MediaPipe hand model, now tracking up to two hands (no training). If left/right feel swapped,
>   set `get_config(gesture_steer_sign=-1.0)`; tune `gesture_steer_mag` / `gesture_throttle_mag` /
>   `gesture_prayer_thresh` to taste.

### Running smoothly on a weak laptop (no GPU / integrated graphics)
The live loop juggles three heavy jobs on one chip — MetaDrive's 3-D render, MediaPipe hand
tracking, and the world-model feedback. There's no software you can *download* to add power
(and CUDA-PyTorch needs an NVIDIA GPU); the levers are **un-throttling the machine** and
**asking the program for less**:
- **OS (free, biggest win):** set Windows to **Best performance** and **plug in** — U-series CPUs
  throttle hard on Balanced/battery. Keep it cool (hard surface, clean vents) and close other apps.
  ```powershell
  powercfg /setactive 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c   # high performance (revert: 381b4222-...)
  ```
- **Throttle the feedback forecast** — the priciest per-step cost (a 15-step imagine) now runs only
  every `cfg.gesture_feedback_every` frames (default **3**); the HUD reuses it in between, and the
  cheap state/style/value signals still update every frame. Raise it (e.g. `5`) for more headroom.
- **Smaller webcam capture** — `cfg.gesture_cap_width` / `gesture_cap_height` (default 640×480) cap
  what MediaPipe processes each frame; drop to 480×360 if still choppy.
- **Lighter 3-D render (biggest GPU win)** — the rendered window is the single heaviest item:
  - **`cfg.metadrive_window_size`** (default **800×600**, down from MetaDrive's 1200×900) — render
    resolution is the #1 cost on an integrated GPU; drop to `(640, 480)` if still choppy.
  - **`cfg.metadrive_low_graphics`** (default **on**) — disables real-time shadows + skybox + logo
    when rendering (shadows are brutal on weak GPUs); set `False` for the pretty version.
  - **`cfg.metadrive_traffic_density`** — fewer cars = fewer models to draw + simulate (`0.05` is light).
  - Pick a **simpler scene** to draw less geometry, e.g. `... gesture-discrete <ckpt> SS` (short straight).
  - Or skip 3-D entirely with the `2d` flag (top-down) — much cheaper.
  - To find your bottleneck, run the lightest combo first: `gesture-discrete - 2d`
    (no checkpoint = no feedback, 2-D = light render).

## 8b. Evaluate a trained policy — is it actually usable?
Two axes (the world-model way: *prediction* vs *control*). Works on any actor checkpoint —
`runs/reference/ckpt.pt` (IDM-cloned), `runs/gesture_reference/ckpt.pt` (your-style), or an
RL-trained `runs/metadrive/ckpt.pt`.
```bash
# WATCH it drive (qualitative) -- 3-D window, the trained actor at the wheel (needs a display).
# Good for screen-recording. On Windows 11: click the window, then Win+Alt+R starts/stops a clip
# (saved to Videos\Captures). Runs ~2000 steps, auto-resetting at each crash/off-road/arrival.
python -m scripts.watch_metadrive_3d runs/gesture_reference/ckpt.pt  # YOUR-style model (the "current" one you trained)
python -m scripts.watch_metadrive_3d runs/reference/ckpt.pt          # the IDM-cloned reference
python -m scripts.watch_metadrive_3d runs/gesture_reference/ckpt.pt SSSS   # on a highway (cleaner-looking on camera)
python -m scripts.watch_metadrive_3d                                 # no ckpt = IDM expert drives a clean lap (best demo)

# DRIVING-USABILITY metrics (quantitative) -- route completion %, success, crash + off-road rate,
# with random + IDM-expert baselines. Forced to MetaDrive, so it works on a "your-style" ckpt too:
python -m scripts.eval_driving runs/reference/ckpt.pt                 # 5 episodes + baselines
python -m scripts.eval_driving runs/gesture_reference/ckpt.pt 10      # your-style policy, 10 episodes
python -m scripts.eval_driving runs/reference/ckpt.pt 5 SSSS          # on a highway
python -m scripts.eval_driving runs/reference/ckpt.pt 5 - noidm       # skip the slower IDM baseline

# RETURN-only closed loop (faster, less interpretable) -- actor return vs random:
python -m scripts.eval_closed_loop runs/reference/ckpt.pt
```
> **Reading it:** the bar to clear is *beat random* (it learned something) and *approach IDM* (it's
> actually good). A policy that beats random but trails IDM badly — low **route completion**, near-
> zero **success** — is *learning but not yet usable* (the documented real-sim-control limit; see
> `docs/SYSTEM_OVERVIEW.md` §7). `success_rate` needs a long enough `max_episode_steps` to reach the
> destination — short horizons read 0% even for IDM, so lean on **route completion** there.
>
> **Held-out maps:** `eval_driving` prints `held-out seeds A-B` and runs on map seeds that are
> *disjoint* from the training pool (`run_metadrive` trains on seeds `0…N-1`; eval uses `N…N+M-1`).
> So a high score here means the policy generalizes to roads it never saw — not that it memorized.

## 8c. Full script reference — every runnable entry point
All run from the repo root as `python -m <module>`. `[arg]` = optional; `-` = "skip this positional".

```bash
# --- setup / sanity (no GPU, no models) ---
python -m scripts.smoke_test                       # env + replay buffer + shapes work end-to-end
python -m scripts.probe_metadrive [state|image]    # print YOUR MetaDrive install's obs shapes (set state_dim/image_size)
python -m scripts.probe_metadrive_render           # check what 3-D rendering works on this machine

# --- train ---
python -m training.collect                         # collect trajectories into the replay buffer (dummy env, random)
python -m training.train_world_model               # train just the world model (state mode, DummyEnv)
python -m training.train_behavior                  # single-shot: WM + actor-critic in imagination (the toy)
python -m training.dreamer_loop                    # the ITERATED Dreamer loop (the real algorithm)
python -m scripts.run_metadrive [--map M] [--num-scenarios N] [--iters I] [--wm-steps W] [--behavior-steps B] [--collect C] [--seed-steps S] [--entropy E]   # Dreamer loop across a POOL of random maps -> runs/metadrive/ (see --help)
python -m scripts.drive_from_pixels                # image-mode: train a pixel WM + actor in imagination -> runs/visual/
python -m training.train_reference                 # IDM-expert reference (WM+BC actor+critic) -> runs/reference/ckpt.pt
python -m scripts.train_on_gesture [session.npz | dir | glob ...]   # learn YOUR driving; many drives accumulate (runs/sessions/)

# --- evaluate a trained policy (see 8b) ---
python -m scripts.eval_driving <ckpt> [eps] [map] [noidm]   # route/success/crash vs random + IDM (forced MetaDrive)
python -m scripts.eval_closed_loop [ckpt]          # actor return vs random (faster, less interpretable)
python -m scripts.ablate_dynamics                  # GRU vs Mamba-style SSM on the same eval harness

# --- watch / record / dream ---
python -m scripts.watch_metadrive_3d [ckpt|-] [map]   # 3-D window: IDM expert (-) or OUR trained actor
python -m scripts.record_metadrive [idm|forward] [map]   # top-down GIF of the sim -> runs/metadrive_drive.gif
python -m scripts.dream_video [ckpt]               # render the world model's DREAM (condition on real frames, roll prior)

# --- drive by hand + feedback (see 8 / 8b) ---
python -m scripts.drive_gesture [keyboard|gesture|gesture-discrete|random|forward] [ckpt|-] [map] [2d|3d]
python -m scripts.feedback_report [session.npz] [reference_ckpt.pt]   # offline habits report -> runs/feedback_report.json
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

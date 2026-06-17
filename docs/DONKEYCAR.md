# DonkeyCar (DonkeyGym) integration

Run the pipeline on **DonkeyGym** — the Unity-based DonkeyCar simulator — for **full rendered 3-D
car graphics** and **camera-image** observations. It plugs in behind the same `envs/base.py`
contract as the dummy/MetaDrive envs, so only `envs/donkey_env.py` knows about it.

DonkeyGym is **image-first**: the obs is a camera frame, the action is `[steering, throttle]`. Our
**image-mode** world model (CNN encoder + transposed-CNN decoder) consumes it directly.

---

## Status
- ✅ **Wrapper built** (`envs/donkey_env.py`): camera frame → `(3, image_size, image_size)` in
  `[0,1]` (resized to a square multiple of 16 for the CNN), action `[-1,1]` → Donkey
  `[steer∈[-1,1], throttle∈[0, donkey_throttle]]`. Defensive reset/step (old-gym 4-tuple and
  gymnasium 5-tuple). `make_env(cfg)` handles `cfg.env == "donkey"`.
- ✅ **Adapter unit-tested without the sim** (`tests/test_donkey_adapter.py`): resize/scale/CHW +
  throttle mapping; a skip-able import check that never instantiates (instantiation launches Unity).
- ⏳ **Live run not done here** — it needs the Unity sim binary + a GPU/display, which this
  headless box doesn't have. Verify on your machine.

---

## Install (read this — py3.13 is fragile; py3.11 is the clean path)

```bash
pip install gym-donkeycar          # pulls gym 0.26.2
```
Then, depending on your Python:

- **Python ≤ 3.11 (recommended — DonkeyCar's intended environment):** works as-is. DonkeyCar ships
  its own conda env (`conda create -n donkey python=3.11`); use that and `numpy<2`.
- **Python 3.12 / 3.13 (this repo's env): two extra hoops.**
  1. `pip install pyasyncore pyasynchat` — gym_donkeycar imports `asyncore`, **removed from the
     stdlib in Python 3.12**; this backport restores it. (Verified: with it, the env class imports.)
  2. ⚠️ `gym 0.26` is unmaintained and **does not support NumPy 2.0** (this repo has numpy 2.x), so
     runtime is fragile. If you hit numpy errors, run DonkeyGym in a **py3.11 env with numpy<2**.

**Download the Unity sim:** grab `donkey_sim` for your OS from the
[DonkeyGym releases](https://github.com/tawnkramer/gym-donkeycar/releases), unzip it, and set the
path (the wrapper reads it):
```bash
# Windows (PowerShell):
$env:DONKEY_SIM_PATH = "C:\path\to\DonkeySimWin\donkey_sim.exe"
# or in code: get_config(env="donkey", donkey_exe_path=r"C:\path\to\donkey_sim.exe", ...)
```
If `DONKEY_SIM_PATH` is unset, gym_donkeycar expects you to **launch the sim manually** first.

---

## Config mapping
| `config.py` field | meaning for Donkey |
|---|---|
| `env="donkey"` | selects this wrapper |
| `obs_type="image"` | Donkey is camera-only |
| `image_size` | camera frame is resized to `image_size × image_size` (multiple of 16) |
| `donkey_level` | 0 roads / 1 warehouse / 2 avc-sparkfun / 3 generated-track |
| `donkey_exe_path` | path to `donkey_sim.exe` (or set `$DONKEY_SIM_PATH`) |
| `donkey_throttle` | our throttle `+1` maps to this Donkey throttle (Donkey max is 5.0; default 1.0 = tame) |
| `action_dim=2` | `[steer, throttle] ∈ [-1,1]`; the wrapper maps throttle to Donkey's `[0, throttle]` |

---

## Running (image mode → needs a GPU)
```bash
# 1) start the sim (auto-launched if DONKEY_SIM_PATH is set), then collect + train in image mode:
python -c "from config import get_config; from training.train_world_model import train; \
train(get_config(env='donkey', obs_type='image', image_size=64, device='cuda', \
deter_dim=128, stoch_dim=32, hidden_dim=128, seq_len=20, max_episode_steps=500), steps=...)"
# 2) the dream renderer / behavior / closed-loop all work the same, with env='donkey'.
```
Image-mode + a real 3-D sim is **GPU territory** (state-mode CPU tricks don't apply to pixels at
speed). Use Colab/Kaggle or a local GPU.

---

## Known pitfalls (and how the wrapper guards them)
- **`asyncore` (py3.12+)** → install `pyasyncore pyasynchat`, or use py3.11.
- **NumPy 2.0 vs gym 0.26** → use a py3.11 env with `numpy<2` if you hit numpy errors.
- **Instantiating launches Unity** → tests/headless code must NOT construct the env (it connects to
  the sim and blocks); the unit tests only exercise the pure adapter + import.
- **Frame size** → Donkey's native camera is 120×160; the wrapper resizes to your square
  `image_size` so the CNN encoder/decoder fit.
- **Throttle range** → Donkey throttle is `[0, 5]`; we keep the codebase in `[-1,1]` and map only
  in the wrapper (recorded actions stay in our convention).

---

## Why a separate py3.11 env is cleanest
DonkeyCar/DonkeyGym is an older stack (old gym, asyncore, numpy<2). It's designed to live in its own
conda env. Running it there avoids all the shims; this repo's model/eval code is identical either
way, since everything downstream of `envs/base.py` is sim-agnostic.

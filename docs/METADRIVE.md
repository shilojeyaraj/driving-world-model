# MetaDrive integration

Run the world-model pipeline on a **real driving simulator** instead of the toy
`DummyDrivingEnv`. MetaDrive is the recommended first real sim because it's far lighter than
CARLA (no separate game-engine server); CARLA stays a later, photorealistic option.

Everything downstream is sim-agnostic — it only sees the `envs/base.py` contract
(`reset()`/`step()` returning a state vector or a `(C,H,W)` image). So adopting MetaDrive is
localized to `envs/metadrive_env.py`; the encoder, RSSM, decoder, policy, eval, and dream
renderer don't change.

---

## Status

- ✅ **Wrapper hardened** (`envs/metadrive_env.py`): defensive `reset`/`step` (handles the
  gymnasium 5-tuple and old-gym 4-tuple), action clipping to `[-1,1]`, and a pure
  `adapt_obs()` that normalizes observations to the contract.
- ✅ **`adapt_obs` unit-tested without MetaDrive** (`tests/test_metadrive_adapter.py`): covers
  flat state, multi-modal dicts, HWC↔CHW, 0–255↔0–1, and frame-stacking. A `pytest.importorskip`
  **live smoke test** runs reset/step only where MetaDrive is actually installed.
- ✅ **Probe script** (`scripts/probe_metadrive.py`) to read the real obs dims for your install.
- ✅ **Live-verified in STATE mode** (this Windows/CPU box, MetaDrive **0.4.3** via the GitHub
  source install — see below). `reset`/`step` work headless; the live smoke test passes. The
  real observation is **`state_dim = 259`** (`Box(259,)`), action `Box(2,) ∈ [-1,1]`. The full
  pipeline (collect → ELBO → backprop) runs on real MetaDrive data.
- ⏳ **Image mode not yet verified** — needs a GPU/offscreen renderer (use Colab/Kaggle).

---

## Install

```bash
pip install metadrive-simulator
```

On **first env creation** MetaDrive downloads map/vehicle assets (hundreds of MB). Image
observations render via OpenGL and generally need a **GPU or an offscreen display** — prefer
Colab/Kaggle/Linux for image mode; `state` mode is lighter and a fine place to start.

### Install issue seen here (and fixes)

On this machine the install failed while building the legacy **`gym`** dependency:

```
error in gym setup command: 'extras_require' must be a dictionary whose values are strings
or lists of strings ... Failed to build 'gym' when getting requirements to build wheel
```

This is the well-known "old `gym` won't build with modern setuptools" problem. **On this
machine, fix #3 (GitHub source install) succeeded** — it installed MetaDrive 0.4.3 with
`gymnasium` instead of the legacy `gym`, and state mode then ran headless on CPU. Fixes, in
order of preference:

1. **Use Linux / Google Colab / Kaggle** (where MetaDrive + GPU rendering are actually tested).
   This is the recommended path for image mode anyway.
2. **Fresh virtualenv with pinned build tools**, then install:
   ```bash
   python -m venv .venv && . .venv/Scripts/activate   # (Linux/mac: source .venv/bin/activate)
   pip install "setuptools<66" "wheel<0.41"
   pip install metadrive-simulator
   ```
   (Pinning build tools is why we do it in a throwaway venv — don't downgrade your global env.)
3. **Install MetaDrive from source** (newer code uses `gymnasium`, avoiding the old `gym`):
   ```bash
   pip install "git+https://github.com/metadriverse/metadrive.git"
   ```

---

## The #1 gotcha: observation shapes differ by version

MetaDrive's default observation is **not** the 35-dim vector the DummyEnv uses, and image
shapes/stacking vary by release. After installing, probe the real shapes:

```bash
python -m scripts.probe_metadrive state     # prints the state dim  -> set cfg.state_dim
python -m scripts.probe_metadrive image     # prints the (C,H,W)    -> set cfg.image_size
```

Then set the config to match (e.g. `cfg.state_dim = 259`). If you skip this, the encoder's
input layer will mismatch and training will error immediately.

---

## Config mapping

| `config.py` field | meaning for MetaDrive |
|---|---|
| `env="metadrive"` | selects this wrapper (`envs/base.py:make_env`) |
| `obs_type="state"` / `"image"` | lidar+ego vector / camera frames |
| `state_dim` | **must equal MetaDrive's state dim** — get it from the probe |
| `image_size` | target H/W for the image obs (image mode needs a GPU) |
| `action_dim=2` | MetaDrive action is `[steering, throttle/brake] ∈ [-1,1]` — matches our Tanh actor |
| `max_episode_steps` | passed as MetaDrive's `horizon` |

**Reward:** unlike the DummyEnv (`throttle − |steer|`), MetaDrive supplies its own driving
reward (progress, lane-keeping, collision/out-of-road penalties). The world model just learns
whatever reward the env emits; the reward head + closed-loop metrics carry over unchanged.

---

## Running the pipeline on MetaDrive

```bash
# 1) probe + set cfg.state_dim accordingly, then:
python -m training.collect            # collect trajectories (edit get_config(env="metadrive", ...))
python -m training.train_world_model  # train the world model
python -m training.train_behavior     # train the policy in imagination
python -m scripts.eval_closed_loop runs/behavior/ckpt.pt   # closed-loop driving
```

Start with `obs_type="state"` (CPU-friendly). For `obs_type="image"`, move to a GPU box and use
`device="cuda"` — the same image encoder/decoder validated on the toy render apply directly.

---

## Known pitfalls (and how the wrapper guards them)

- **API drift:** gymnasium returns `(obs, info)` from `reset` and a 5-tuple from `step`; old gym
  returns bare `obs` and a 4-tuple. `reset`/`step` handle both.
- **Obs nesting/stacking:** image obs may arrive as `{"image": ...}` and as `(H,W,C,stack)`;
  `adapt_obs` extracts the image and takes the most recent frame.
- **Channel order / range:** `adapt_obs` moves HWC→CHW and scales 0–255→0–1 as needed.
- **Construction signature:** some versions take the config dict positionally vs. `config=`;
  the wrapper tries both.
- **Headless rendering:** image mode needs OpenGL; on a headless server use a virtual display
  (e.g. `xvfb-run`) or a GPU instance.
- **`state_dim` mismatch:** the most common error — always probe first.

---

## Why MetaDrive before CARLA

| | weight | gives | when |
|---|---|---|---|
| toy render (`DummyDrivingEnv`) | trivial, CPU | a blob on a road | mechanism validation (done) |
| **MetaDrive** | light (CPU state / GPU image) | a real driving sim, camera obs | **next** |
| CARLA | heavy (GPU + Unreal server, tens of GB) | photorealistic camera/LiDAR | later, if photorealism is needed |

CARLA needs a dedicated GPU and a running Unreal server, and has no wrapper here yet — overkill
until the pipeline is proven on MetaDrive.

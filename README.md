# Driving World Model — a from-scratch Dreamer for model-based RL

A **complete, from-scratch Dreamer-style world model** for driving, written in PyTorch. A world
model is a learned, *differentiable simulator*: it compresses observations into a latent state,
learns the latent **dynamics**, and predicts reward/continuation — so you can "dream" futures and
train a control policy **entirely in imagination, with zero environment steps**.

Built to be *understood*: every ML core is written from scratch with "why" comments, derived in
`ARCHITECTURE.md`, and logged step-by-step in `experiments/`. It runs on a **laptop CPU** in
state mode; pixels/real-sims scale up to a GPU.

```
 obs ─Encoder─► e_t ─┐
                     ├─ RSSM (h,z) ─ Decoder ─► ô, r̂, ĉ          (world-model learning, ELBO)
 prev action ───────┘        │
                             └─ imagine (prior only) ─► Actor–Critic ─(action)─► env   (behavior)
```

## What's in it

- **World model (ELBO).** Encoder + **RSSM** (deterministic GRU memory + stochastic latent,
  reparameterized) + decoder heads (obs / reward / continue), trained on the variational bound
  with **free-bits** anti-collapse. `models/`, `ARCHITECTURE.md §2`.
- **Imagination + model-based RL.** A Tanh-Normal **actor** + value **critic** trained on imagined
  λ-returns (value gradient through the dynamics) — no env steps. `training/train_behavior.py`.
- **Two eval axes.** **Open-loop** (predict, action-conditioned beats no-action) and
  **closed-loop** (the policy actually drives). `eval/`.
- **Iterated Dreamer loop.** collect-with-policy → train WM → train policy, repeated, to ground the
  model in the states the policy visits. `training/dreamer_loop.py`.
- **Dynamics ablation.** The recurrence is a one-line `cfg.dynamics` swap behind one interface —
  **GRU** (`rssm`) vs a minimal **Mamba-style selective SSM** (`mamba`) — re-scored on the same
  harness. `models/recurrence.py`, `scripts/ablate_dynamics.py`, `experiments/006`.
- **Visual mode + "dream video".** CNN encoder + transposed-CNN decoder learn from **pixels**; the
  renderer rolls the prior under chosen actions and decodes a video of the car driving.
  `scripts/dream_video.py`, `experiments/007`.
- **Real simulators** behind one `envs/base.py` contract:
  - **MetaDrive** (`env="metadrive"`) — live-verified in state mode (real 259-dim lidar+ego);
    top-down video via `scripts/record_metadrive.py`. `docs/METADRIVE.md`.
  - **DonkeyCar / DonkeyGym** (`env="donkey"`) — Unity 3-D camera sim, image obs. `docs/DONKEYCAR.md`.
- **Gesture control + driving feedback.** Drive with your **hand** (MediaPipe HandLandmarker) and
  get live critique from the world model — **safety** (continue-head forecast), **style** (deviation
  + surprise vs a behavior-cloned IDM expert), **value** (critic) — as a HUD + a session report.
  `control/gesture.py`, `eval/feedback.py`, `scripts/drive_gesture.py`,
  `docs/superpowers/specs/2026-06-15-gesture-feedback-design.md`.
- **Full test suite** (fast unit/contract + slow training gates), built test-first throughout.

## Results & honest limitations

- ✅ **Toy (state, CPU):** world model trains (KL healthy, no collapse); open-loop beats no-action;
  a policy trained *only in imagination* drives near-optimally (**return ≈ 95 vs ≈ −51 random**).
- ✅ **Pixels:** recon → ~0; the dream video tracks reality (**per-pixel MSE ≈ 0.0002** at 64×64).
- ✅ **Real-sim learning:** MetaDrive world model trains on real lidar (**recon 245 → 0.17**).
- ⚠️ **Real-sim control is unsolved here:** the policy exploits the model / collapses into a tanh
  corner — the documented model-based RL failure mode. Fixing it needs DreamerV3-grade stabilizers
  (two-hot heads, EMA return normalization) + GPU-scale compute. `experiments/010–012`.

The honesty is the point: the repo maps exactly where a simplified, from-scratch Dreamer succeeds
and where the production tricks become necessary.

## Architecture (swappable slots)
| Slot | File | Options |
|------|------|---------|
| Encoder  | `models/encoder.py`      | MLP (state) / CNN (image) |
| Dynamics | `models/recurrence.py` (in `rssm.py`) | **GRU** ✓ / **Mamba-SSM** ✓ / Transformer (designed-for) |
| Decoder  | `models/decoder.py`      | MLP/CNN heads (obs/reward/continue); DiT designed-for |
| Policy   | `models/actor_critic.py` | actor-critic in imagination |

`config.obs_type` is the laptop/GPU switch: `"state"` (low-dim vector → CPU) vs `"image"` (pixels
→ GPU). Same code path.

## Quick start
```bash
pip install -r requirements.txt
python scripts/smoke_test.py            # env + buffer + shapes, no GPU/models
pytest -m "not slow"                    # fast tests (~seconds)

# train a policy in imagination and watch it drive the toy (CPU, ~6-8 min):
python -c "from config import get_config; from training.train_behavior import train_behavior; \
train_behavior(get_config(env='dummy', obs_type='state', seq_len=20, max_episode_steps=200), \
wm_steps=800, behavior_steps=800)"
python -m scripts.eval_closed_loop runs/behavior/ckpt.pt

python -m scripts.record_metadrive      # SEE a car drive (rendered top-down) -> runs/metadrive_drive.gif
python -m scripts.watch_metadrive_3d    # SEE it in 3-D (rendered chase-camera window; needs a display)
```
**Full command list for every feature is in `docs/RUNNING.md`** (env vars needed: none).

## Runs on your laptop vs needs a GPU
- **Laptop (CPU), works now:** the whole toy pipeline, dynamics ablation, MetaDrive **state** mode,
  the **top-down *and* 3-D rendered** driving views (`scripts/record_metadrive` /
  `scripts/watch_metadrive_3d` — the 3-D window needs a display, integrated GPU is fine), the
  gesture demo (with a webcam), and small-resolution dream video.
- **Needs a GPU:** heavy image-mode training (64×64) — use **Kaggle's free GPU** (`docs/KAGGLE.md`)
  — and the photoreal 3-D sims (DonkeyGym Unity / MetaDrive camera), which want a GPU + display.

## Docs
| Doc | What |
|---|---|
| `docs/SYSTEM_OVERVIEW.md` | the ML story + interview Q&A |
| `ARCHITECTURE.md` | the math: ELBO by hand, action timing, λ-returns, posterior collapse |
| `docs/RUNNING.md` | every run/test command (+ env vars) |
| `docs/METADRIVE.md` · `docs/DONKEYCAR.md` · `docs/KAGGLE.md` | real sims + GPU |
| `docs/superpowers/specs/` | design specs (v1 world model; gesture-feedback) |
| `experiments/001–017` | the build log (one entry per milestone) · `CONCEPTS.md` (the ML map) |

## How it was built
From-scratch, test-first, one milestone at a time. Each component carries a `Concept:` / `Question:`
header (see `CONCEPTS.md`) and a logged experiment with a one-line takeaway. The plumbing
(`envs/`, `data/`, training/eval loops) is "OK to use a library"; the ML cores were written by hand.

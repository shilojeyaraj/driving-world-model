# Driving World Model -- model-based learning, built to be learned

A from-scratch world model for driving, structured so you **learn the ML as you build it**.
The plumbing (env, replay buffer, training/eval loops) is provided and runnable today. The
ML core (encoder, dynamics, decoder, policy) is left as **deliberate stubs** for you to
implement -- that's where the understanding lives.

> Read `CONCEPTS.md` next: it maps each file to the ML concept it teaches and the one
> question you should be able to answer after implementing it.

> **Status.** v1 is complete and verified end-to-end on `DummyDrivingEnv`: world model trains
> (ELBO), open-loop beats the no-action baseline, and a policy trained purely in imagination
> drives the real env (return ≈ 95 vs ≈ −51 random). Extensions landed on top:
> - **Dynamics ablation** — the recurrence is a one-line `cfg.dynamics` swap (`rssm` GRU /
>   `mamba` selective SSM) behind one interface; `scripts/ablate_dynamics.py`, `experiments/006`.
> - **Visual mode** — a render-based image `DummyDrivingEnv`, CNN encoder + transposed-CNN
>   decoder, and a **dream-video renderer** (`scripts/dream_video.py`) that rolls the prior and
>   decodes latents into predicted frames; `experiments/007`.
>
> See **`ARCHITECTURE.md`** for the full walkthrough (ELBO by hand, action timing, λ-returns,
> posterior collapse) and `experiments/001–007` for the build log. Still designed-for: image
> mode on a GPU at full resolution, MetaDrive/CARLA, and a windowed-state Transformer recurrence.

## Architecture (three swappable slots)

```
 env -> ENCODER -> RSSM (dynamics core) -> DECODER heads
                       |
                       v
                 IMAGINE rollout -> ACTOR-CRITIC --(action)--> env
```

| Slot | File | Options |
|------|------|---------|
| Encoder  | `models/encoder.py`      | CNN / ViT |
| Dynamics | `models/rssm.py`         | RSSM / Transformer / Mamba  **<- the axis to ablate** |
| Decoder  | `models/decoder.py`      | CNN / DiT (diffusion) |
| Policy   | `models/actor_critic.py` | actor-critic in imagination |

## Two observation modes (laptop vs GPU)

`config.obs_type` switches the whole pipeline:

- `"state"` -> low-dim vector (lidar rays + ego state). No big conv nets -> trains on **CPU**.
  Use this to validate the *mechanism* on your laptop.
- `"image"` -> camera pixels. Needs a **GPU** (Kaggle: ~30 free GPU hrs/week). Same code.

Develop and debug on `state` locally; switch to `image` for the real runs.

## Quick start

```bash
pip install -r requirements.txt
python scripts/smoke_test.py        # env + buffer, no GPU, no models needed -- run this first
# ...implement the model stubs (see CONCEPTS.md)...
python -m training.collect          # collect trajectories
python -m training.train_world_model
```

The smoke test and collect script run against a built-in `DummyDrivingEnv`, so you can
confirm the plumbing on your laptop before installing MetaDrive or implementing anything.

## How the stubs are tagged

Every file's header says one of:
- `IMPLEMENT FROM SCRATCH` -- write it yourself. This is the point. Don't import a solution.
- `OK TO USE A LIBRARY`    -- plumbing; use/modify freely.

Each from-scratch stub raises `NotImplementedError` and carries a `Concept:` (the idea it
teaches) and a `Question:` (what you should be able to answer once it works).

## Suggested order

1. `scripts/smoke_test.py` passes (plumbing sanity).
2. `models/encoder.py` + `models/decoder.py` on `obs_type="state"` -- get reconstruction working.
3. `models/rssm.py` + `models/world_model.py` -- the ELBO; watch for posterior collapse.
4. `eval/open_loop.py` -- prove it predicts (action-conditioned beats not).
5. `models/actor_critic.py` + `training/train_behavior.py` -- policy in imagination.
6. `eval/closed_loop.py` -- does it actually drive?
7. Ablate the dynamics slot; switch to `obs_type="image"` on Kaggle for the real runs.

Log every step in `experiments/` using `LOG_TEMPLATE.md`.

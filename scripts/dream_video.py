"""Render the world model's DREAM (V1, step ②). Condition on a few real frames, then roll the
PRIOR forward with a chosen action sequence (sample=False -> the prior mean) and DECODE each
latent into a predicted frame. Save ground-truth vs. dreamed frames side by side -- the iconic
world-model visualization: you watch what the model thinks will happen.

By default the dream is driven with CONSTANT THROTTLE so the car clearly accelerates along the
road -- demonstrating action-conditioned prediction, not just static reconstruction.

Usage:  python -m scripts.dream_video [checkpoint.pt]
        (no checkpoint -> trains a small image-mode world model inline; run it ALONE)
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from config import get_config
from envs.base import make_env
from data.replay_buffer import SequenceReplayBuffer


def _collect(cfg, steps, seed=0):
    np.random.seed(seed)
    env = make_env(cfg)
    buf = SequenceReplayBuffer(cfg.buffer_capacity, cfg.seq_len)
    obs = env.reset()
    for _ in range(steps):
        a = np.random.uniform(-1, 1, cfg.action_dim).astype(np.float32)
        nxt, r, done, _ = env.step(a)
        buf.add(obs, a, r, done)
        obs = env.reset() if done else nxt
    return buf


def _drive_trajectory(cfg, context, horizon, drive_action, seed):
    """Roll the REAL env: `context-1` random steps to set up a state, then `horizon` steps of
    the chosen drive action. Returns obs (1, context+horizon, 3,H,W) and actions (1, *, A)."""
    np.random.seed(seed)
    env = make_env(cfg)
    obs = env.reset()
    frames, acts = [obs], []
    for t in range(context + horizon - 1):
        a = (np.random.uniform(-1, 1, cfg.action_dim).astype(np.float32) if t < context - 1
             else np.asarray(drive_action, dtype=np.float32))
        obs, _, _, _ = env.step(a)
        frames.append(obs); acts.append(a)
    return (torch.as_tensor(np.stack(frames))[None], torch.as_tensor(np.stack(acts))[None])


def _to_img(frame_chw, scale=6):
    f = np.clip(np.transpose(frame_chw, (1, 2, 0)), 0, 1)        # CHW -> HWC, [0,1]
    f = np.repeat(np.repeat(f, scale, axis=0), scale, axis=1)    # nearest-neighbour upscale
    return (f * 255).astype(np.uint8)


def dream_video(ckpt=None, image_size=32, wm_steps=800, context=5, horizon=20,
                drive_action=(0.0, 1.0), out_dir="runs/dream", seed=1):
    from models.world_model import WorldModel

    if ckpt:
        from utils import load_models
        cfg, wm, _, _ = load_models(ckpt)
    else:
        cfg = get_config(obs_type="image", env="dummy", image_size=image_size,
                         deter_dim=128, stoch_dim=32, hidden_dim=128,
                         seq_len=context + horizon + 4, max_episode_steps=300, batch_size=16)
        buf = _collect(cfg, 4000, seed=0)
        wm = WorldModel(cfg, cfg.action_dim)
        opt = torch.optim.Adam(wm.parameters(), lr=3e-3)
        for step in range(wm_steps):
            batch = {k: torch.as_tensor(v) for k, v in buf.sample(cfg.batch_size).items()}
            loss, m = wm.assemble_loss(batch)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(wm.parameters(), 100.0); opt.step()
            if step % 100 == 0:
                print(step, {k: round(float(v), 4) for k, v in m.items()})

    # Condition on `context` real frames, then DREAM `horizon` steps with the drive action.
    obs, actions = _drive_trajectory(cfg, context, horizon, drive_action, seed)
    wm.eval()
    with torch.no_grad():
        emb = wm.encoder(obs[:, :context].reshape(context, *obs.shape[2:])).reshape(1, context, -1)
        state = wm.rssm.observe(emb, actions[:, :context], wm.rssm.initial_state(1, obs.device))["state"]
        ia = actions[:, context - 1: context - 1 + horizon]
        feat = wm.rssm.imagine(ia, state, sample=False)["feat"]
        pred = wm.decoder(feat.reshape(horizon, feat.shape[-1]))["obs"].reshape(horizon, *obs.shape[2:])
    true = obs[0, context: context + horizon]

    import imageio.v2 as imageio
    os.makedirs(out_dir, exist_ok=True)
    t = true.cpu().numpy(); p = pred.cpu().numpy()
    row_true = np.concatenate([_to_img(t[i]) for i in range(horizon)], axis=1)
    row_pred = np.concatenate([_to_img(p[i]) for i in range(horizon)], axis=1)
    sep_h = np.full((4, row_true.shape[1], 3), 64, np.uint8)
    imageio.imwrite(os.path.join(out_dir, "dream_montage.png"),
                    np.concatenate([row_true, sep_h, row_pred], axis=0))

    sep_w = np.full((_to_img(t[0]).shape[0], 4, 3), 64, np.uint8)
    gif = [np.concatenate([_to_img(t[i]), sep_w, _to_img(p[i])], axis=1) for i in range(horizon)]
    imageio.mimsave(os.path.join(out_dir, "dream.gif"), gif, duration=0.15, loop=0)

    mse = float(((pred - true) ** 2).mean())
    print(f"saved {out_dir}/dream_montage.png and dream.gif  (top/left=truth, bottom/right=dream; "
          f"drive_action={drive_action}; per-pixel dream MSE={mse:.4f})")


if __name__ == "__main__":
    dream_video(sys.argv[1] if len(sys.argv) > 1 else None)

"""Train on a recorded GESTURE-driven session (e.g. you hand-driving MetaDrive via
scripts/drive_gesture.py). Builds a world model + a "your-style" reference (behavior-cloned from
YOUR driving) + a critic, saved as a normal checkpoint -- so the feedback engine then critiques
your driving against YOUR OWN style (not the IDM expert).

Flow:
  1) python -m scripts.drive_gesture gesture           # hand-drive MetaDrive -> runs/gesture_session.npz
  2) python -m scripts.train_on_gesture                # -> runs/gesture_reference/ckpt.pt
  3) python -m scripts.drive_gesture gesture runs/gesture_reference/ckpt.pt   # drive w/ feedback vs YOU
  4) python -m scripts.feedback_report runs/gesture_session.npz runs/gesture_reference/ckpt.pt

Usage:  python -m scripts.train_on_gesture [session.npz]
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from config import get_config
from data.replay_buffer import buffer_from_session
from models.world_model import WorldModel
from training.dreamer_loop import _train_world_model
from training.train_reference import bc_actor, eval_critic
from utils import save_checkpoint


def main(session="runs/gesture_session.npz", out="runs/gesture_reference/ckpt.pt",
         wm_steps=1000, bc_steps=1000, critic_steps=1000):
    data = np.load(session)
    obs = data["obs"]
    common = dict(action_dim=2, deter_dim=128, stoch_dim=32, hidden_dim=128, seq_len=10,
                  gamma=0.99, lambda_=0.95, lr=3e-4, actor_lr=3e-4, critic_lr=3e-4)
    if obs.ndim == 2:                                    # state vector (e.g. MetaDrive 259-dim)
        cfg = get_config(obs_type="state", state_dim=obs.shape[1], **common)
    else:                                                # image (C,H,W)
        cfg = get_config(obs_type="image", image_size=obs.shape[-1], **common)

    buf = buffer_from_session(obs, data["action"], data["reward"], data["done"],
                              cfg.buffer_capacity, cfg.seq_len)
    print(f"loaded {session}: {obs.shape[0]} steps -> {len(buf._episodes)} usable episodes", flush=True)
    if not buf.can_sample():
        raise SystemExit("no usable episodes -- drive longer, or lower seq_len in this script.")

    wm = WorldModel(cfg, cfg.action_dim)
    opt = torch.optim.Adam(wm.parameters(), lr=cfg.lr)
    print(f"[1/3] world model ({wm_steps} steps)...", flush=True)
    _train_world_model(cfg, wm, opt, buf, wm_steps)
    print(f"[2/3] behavior-cloning YOUR style ({bc_steps} steps)...", flush=True)
    actor, bc_loss = bc_actor(cfg, wm, buf, bc_steps)
    print(f"[3/3] critic ({critic_steps} steps)...", flush=True)
    critic, critic_loss = eval_critic(cfg, wm, buf, critic_steps)

    save_checkpoint(out, wm, actor, critic, cfg)
    print(f"YOUR-style reference saved -> {out}  bc_loss={bc_loss:.4f} critic_loss={critic_loss:.4f}", flush=True)
    print(f"drive with feedback vs YOUR style:  python -m scripts.drive_gesture gesture {out}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "runs/gesture_session.npz")

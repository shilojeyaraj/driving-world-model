"""Train on recorded GESTURE/KEYBOARD-driven sessions (you hand-driving MetaDrive via
scripts/drive_gesture.py). Builds a world model + a "your-style" reference (behavior-cloned from
YOUR driving) + a critic, saved as a normal checkpoint -- so the feedback engine then critiques
your driving against YOUR OWN style (not the IDM expert).

Accepts ONE OR MANY sessions (files, a directory, or a glob), so multiple drives ACCUMULATE into
a bigger dataset -- drive_gesture archives each drive to runs/sessions/.

Flow:
  1) python -m scripts.drive_gesture keyboard runs/reference/ckpt.pt   # drive -> runs/sessions/session_*.npz
  2) python -m scripts.train_on_gesture runs/sessions/                 # train on ALL drives
  3) python -m scripts.drive_gesture keyboard runs/gesture_reference/ckpt.pt   # critiqued vs YOU
  4) python -m scripts.feedback_report runs/gesture_session.npz runs/gesture_reference/ckpt.pt

Usage:  python -m scripts.train_on_gesture [session.npz | dir | glob] ...
"""
import os, sys, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from config import get_config
from data.replay_buffer import buffer_from_session
from models.world_model import WorldModel
from training.dreamer_loop import _train_world_model
from training.train_reference import bc_actor, eval_critic
from utils import save_checkpoint


def concat_sessions(sessions):
    """Concatenate recorded sessions into one. Each session's LAST step is forced done=True so
    episodes never bleed across separate drives (the boundary isn't a real transition). PURE."""
    obs, act, rew, done = [], [], [], []
    for s in sessions:
        d = np.asarray(s["done"], np.float32).copy()
        if len(d):
            d[-1] = 1.0                                  # end of this drive = episode boundary
        obs.append(np.asarray(s["obs"]))
        act.append(np.asarray(s["action"]))
        rew.append(np.asarray(s["reward"], np.float32))
        done.append(d)
    return {"obs": np.concatenate(obs), "action": np.concatenate(act),
            "reward": np.concatenate(rew), "done": np.concatenate(done)}


def load_sessions(paths):
    """Expand `paths` (any mix of .npz files, directories, or globs) to session files and
    concatenate them (see concat_sessions). Returns (combined_data, sorted_file_list)."""
    files = []
    for p in paths:
        if os.path.isdir(p):
            files += glob.glob(os.path.join(p, "*.npz"))
        else:
            files += glob.glob(p)                        # handles both literal paths and wildcards
    files = sorted(dict.fromkeys(files))                 # dedup, deterministic order
    if not files:
        raise SystemExit(f"no .npz sessions found in: {list(paths)}")
    return concat_sessions([dict(np.load(f)) for f in files]), files


def main(sessions=("runs/gesture_session.npz",), out="runs/gesture_reference/ckpt.pt",
         wm_steps=1000, bc_steps=1000, critic_steps=1000):
    if isinstance(sessions, str):
        sessions = [sessions]
    data, files = load_sessions(sessions)
    obs = data["obs"]
    common = dict(action_dim=2, deter_dim=128, stoch_dim=32, hidden_dim=128, seq_len=10,
                  gamma=0.99, lambda_=0.95, lr=3e-4, actor_lr=3e-4, critic_lr=3e-4)
    if obs.ndim == 2:                                    # state vector (e.g. MetaDrive 259-dim)
        cfg = get_config(obs_type="state", state_dim=obs.shape[1], **common)
    else:                                                # image (C,H,W)
        cfg = get_config(obs_type="image", image_size=obs.shape[-1], **common)

    buf = buffer_from_session(obs, data["action"], data["reward"], data["done"],
                              cfg.buffer_capacity, cfg.seq_len)
    print(f"loaded {len(files)} session file(s): {obs.shape[0]} steps -> {len(buf._episodes)} usable episodes",
          flush=True)
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
    print(f"drive with feedback vs YOUR style:  python -m scripts.drive_gesture keyboard {out}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or ["runs/gesture_session.npz"])

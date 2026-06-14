"""Real MetaDrive run (state mode): collect -> train world model -> train policy in
imagination -> closed-loop in the REAL sim. Headless on CPU, but slow (MetaDrive stepping is
real physics) -- run it ALONE. See docs/METADRIVE.md.

This is a SINGLE-SHOT pipeline (collect once with a random policy, then train). MetaDrive is far
harder than the toy and the world model only sees random-policy (crash-prone) data, so expect a
modest first-pass policy -- a full Dreamer loop would alternate collect-with-policy and retrain.

Usage:  python -m scripts.run_metadrive
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from config import get_config
from envs.base import make_env
from data.replay_buffer import SequenceReplayBuffer
from models.world_model import WorldModel
from models.actor_critic import Actor, Critic
from training.train_behavior import train_behavior_in_imagination
from eval.closed_loop import closed_loop_eval
from utils import save_checkpoint


def main(collect_steps=4000, wm_steps=1500, behavior_steps=1500, out="runs/metadrive/ckpt.pt"):
    cfg = get_config(env="metadrive", obs_type="state", state_dim=259, action_dim=2,
                     deter_dim=128, stoch_dim=32, hidden_dim=128, seq_len=10, imagine_horizon=15,
                     gamma=0.99, lambda_=0.95, entropy_coef=1e-3, actor_lr=3e-4, critic_lr=3e-4,
                     batch_size=16, max_episode_steps=200)
    torch.manual_seed(0); np.random.seed(0)
    env = make_env(cfg)                                  # one sim instance, reused for eval

    # --- collect (random policy) ---
    buf = SequenceReplayBuffer(cfg.buffer_capacity, cfg.seq_len)
    obs = env.reset()
    for _ in range(collect_steps):
        a = np.random.uniform(-1, 1, cfg.action_dim).astype(np.float32)
        nxt, r, done, _ = env.step(a)
        buf.add(obs, a, r, done)
        obs = env.reset() if done else nxt
    print(f"collected {len(buf)} steps across {len(buf._episodes)} usable episodes", flush=True)

    # --- world model ---
    wm = WorldModel(cfg, cfg.action_dim)
    opt = torch.optim.Adam(wm.parameters(), lr=3e-4)
    for i in range(wm_steps):
        if not buf.can_sample():
            break
        batch = {k: torch.as_tensor(v) for k, v in buf.sample(cfg.batch_size).items()}
        loss, m = wm.assemble_loss(batch)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(wm.parameters(), 100.0); opt.step()
        if i % 300 == 0:
            print("wm", i, {k: round(float(v), 3) for k, v in m.items()}, flush=True)

    # --- behavior in imagination ---
    feat_dim = cfg.deter_dim + cfg.stoch_dim
    actor, critic = Actor(cfg, feat_dim, cfg.action_dim), Critic(cfg, feat_dim)
    train_behavior_in_imagination(cfg, wm, buf, actor, critic, steps=behavior_steps, log_every=300)

    # --- closed-loop in the REAL sim ---
    out_m = closed_loop_eval(actor, wm, env, episodes=5, max_steps=cfg.max_episode_steps)
    save_checkpoint(out, wm, actor, critic, cfg)
    print("MetaDrive closed-loop:", {k: round(v, 3) for k, v in out_m.items()}, flush=True)
    print(f"saved {out}", flush=True)
    env.close()


if __name__ == "__main__":
    main()

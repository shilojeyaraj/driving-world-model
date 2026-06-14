"""Drive from pixels (V1 step ①): train an IMAGE world model, train the actor-critic purely
in imagination, then run closed-loop in the visual env. Saves a checkpoint for reuse.

Heavy (image training) -> a script, not a routine test. Run it ALONE: several concurrent
training jobs oversubscribe the CPU and slow everything to a crawl.

Usage:  python -m scripts.drive_from_pixels
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


def pool(cfg, steps, seed=0):
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


def main(image_size=16, wm_steps=400, behavior_steps=400, out="runs/visual/ckpt.pt"):
    cfg = get_config(obs_type="image", env="dummy", image_size=image_size, deter_dim=64,
                     stoch_dim=16, hidden_dim=64, seq_len=12, imagine_horizon=10, gamma=0.99,
                     lambda_=0.95, entropy_coef=1e-3, actor_lr=3e-3, critic_lr=3e-3,
                     batch_size=16, max_episode_steps=100)
    torch.manual_seed(0)
    buf = pool(cfg, 2500, seed=0)

    wm = WorldModel(cfg, cfg.action_dim)
    opt = torch.optim.Adam(wm.parameters(), lr=3e-3)
    for step in range(wm_steps):
        batch = {k: torch.as_tensor(v) for k, v in buf.sample(cfg.batch_size).items()}
        loss, m = wm.assemble_loss(batch)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(wm.parameters(), 100.0); opt.step()
        if step % 100 == 0:
            print("wm", step, {k: round(float(v), 4) for k, v in m.items()})

    feat_dim = cfg.deter_dim + cfg.stoch_dim
    actor, critic = Actor(cfg, feat_dim, cfg.action_dim), Critic(cfg, feat_dim)
    train_behavior_in_imagination(cfg, wm, buf, actor, critic, steps=behavior_steps, log_every=100)

    out_metrics = closed_loop_eval(actor, wm, make_env(cfg), episodes=5, max_steps=100)
    save_checkpoint(out, wm, actor, critic, cfg)
    print("closed-loop (from pixels):", {k: round(v, 3) for k, v in out_metrics.items()})
    print(f"saved {out}")


if __name__ == "__main__":
    main()

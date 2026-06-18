"""Watch MetaDrive drive in the 3-D render window (the rendered "docs look").

Default: MetaDrive's IDM expert drives a clean lap. Pass a checkpoint to watch OUR trained actor
(from runs/metadrive/ckpt.pt). Needs a DISPLAY (your laptop -- not a headless server/Kaggle).
Integrated graphics is fine. See docs/METADRIVE.md.

Usage:  python -m scripts.watch_metadrive_3d                       # IDM expert, 3-D window
        python -m scripts.watch_metadrive_3d runs/metadrive/ckpt.pt   # our policy
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


def main(ckpt=None, steps=2000):
    from metadrive.envs import MetaDriveEnv

    if ckpt:
        import torch
        from utils import load_models
        from envs.metadrive_env import adapt_obs
        cfg, wm, actor, critic = load_models(ckpt)
        device = torch.device(cfg.device)
        env = MetaDriveEnv(dict(use_render=True, horizon=cfg.max_episode_steps))
        rssm = wm.rssm
        state = rssm.initial_state(1, device)
        prev = torch.zeros(1, cfg.action_dim, device=device)
        obs = adapt_obs(env.reset()[0], "state")
        wm.eval(); actor.eval()
        with torch.no_grad():
            for _ in range(steps):
                e = wm.encoder(torch.as_tensor(obs, device=device).float().unsqueeze(0))
                state, _, _ = rssm.obs_step(state, prev, e)
                action, _ = actor(torch.cat(state, dim=-1), deterministic=True)
                raw, r, terminated, truncated, info = env.step(action.squeeze(0).cpu().numpy())
                prev = action
                if terminated or truncated:
                    obs = adapt_obs(env.reset()[0], "state")
                    state = rssm.initial_state(1, device)
                    prev = torch.zeros(1, cfg.action_dim, device=device)
                else:
                    obs = adapt_obs(raw, "state")
        env.close()
    else:
        from metadrive.policy.idm_policy import IDMPolicy
        env = MetaDriveEnv(dict(use_render=True, agent_policy=IDMPolicy, horizon=1000))
        env.reset()
        dummy = np.zeros(2, dtype=np.float32)
        for _ in range(steps):
            _, r, terminated, truncated, info = env.step(dummy)
            if terminated or truncated:
                env.reset()
        env.close()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)

"""Watch MetaDrive drive in the 3-D render window (the rendered "docs look").

Default: MetaDrive's IDM expert drives a clean lap. Pass a checkpoint to watch OUR trained actor
(from runs/metadrive/ckpt.pt). Needs a DISPLAY (your laptop -- not a headless server/Kaggle).
Integrated graphics is fine. See docs/METADRIVE.md.

Usage:  python -m scripts.watch_metadrive_3d                       # IDM expert, 3-D window, default map
        python -m scripts.watch_metadrive_3d - X                   # IDM on an intersection
        python -m scripts.watch_metadrive_3d - SSSS                # IDM on a highway (4 straights)
        python -m scripts.watch_metadrive_3d runs/metadrive/ckpt.pt   # our policy

Scene (`road_map`): int N (N random blocks) or block letters -- S straight, C curve, X
intersection, O roundabout, T t-junction, r/R ramp. `traffic_density` 0..~0.3.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


def main(ckpt=None, steps=2000, road_map=None, traffic_density=0.1):
    from metadrive.envs import MetaDriveEnv
    from envs.metadrive_env import metadrive_config
    if isinstance(road_map, str) and road_map.isdigit():
        road_map = int(road_map)

    if ckpt:
        import torch
        from utils import load_models
        from envs.metadrive_env import adapt_obs
        cfg, wm, actor, critic = load_models(ckpt)
        cfg.metadrive_render = True
        if road_map is not None:
            cfg.metadrive_map = road_map
        cfg.metadrive_traffic_density = traffic_density
        device = torch.device(cfg.device)
        env = MetaDriveEnv(metadrive_config(cfg))
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
        from config import get_config
        from metadrive.policy.idm_policy import IDMPolicy
        cfg = get_config(env="metadrive", obs_type="state", metadrive_render=True,
                         metadrive_map=road_map, metadrive_traffic_density=traffic_density,
                         max_episode_steps=1000)
        md = metadrive_config(cfg)
        md["agent_policy"] = IDMPolicy
        env = MetaDriveEnv(md)
        env.reset()
        dummy = np.zeros(2, dtype=np.float32)
        for _ in range(steps):
            _, r, terminated, truncated, info = env.step(dummy)
            if terminated or truncated:
                env.reset()
        env.close()


if __name__ == "__main__":
    a = sys.argv[1:]
    ck = a[0] if len(a) > 0 and a[0] not in ("-", "idm", "none") else None
    rm = a[1] if len(a) > 1 else None
    main(ckpt=ck, road_map=rm)

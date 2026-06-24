"""ABLATION: does cloning the expert DIRECTLY from the state vector (a plain MLP, no world model)
lane-keep better than cloning in the WM latent? Both policies are trained on the SAME IDM data and
evaluated on the SAME held-out maps, so the only difference is the representation.

If DIRECT off-road rate << LATENT off-road rate (~100%), the under-trained world-model latent is the
bottleneck for control -- the single most informative experiment for where to invest next.

Usage:  python -m scripts.ablate_direct_bc
        python -m scripts.ablate_direct_bc 4000 50 10        # collect_steps num_scenarios eval_episodes
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from scripts.dagger import build_cfg
from scripts.eval_driving import _run_episodes, _ActorPolicy, _FixedPolicy, _fmt
from eval.closed_loop import summarize_driving
from envs.metadrive_env import train_eval_seed_split
from training.train_reference import collect_idm, bc_actor
from training.dreamer_loop import _train_world_model
from training.direct_bc import DirectPolicy, train_direct_bc
from models.world_model import WorldModel


class _DirectAct:
    """Stateless eval wrapper: action = MLP(obs). No RSSM state to carry (the whole point)."""
    def __init__(self, policy):
        self.p = policy.eval()

    def reset(self):
        pass

    def __call__(self, obs):
        with torch.no_grad():
            return self.p(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy()


def main(collect_steps=4000, num_scenarios=50, episodes=10, wm_steps=1000, bc_steps=1000,
         direct_steps=4000):
    cfg = build_cfg(num_scenarios=num_scenarios)
    print(f"ablation: latent-BC vs direct-BC, same IDM data ({collect_steps} steps, {num_scenarios} maps), "
          f"same held-out eval ({episodes} eps)", flush=True)

    buf = collect_idm(cfg, collect_steps); buf._flush()
    obs = np.concatenate([ep["obs"] for ep in buf._episodes]).astype(np.float32)
    act = np.concatenate([ep["action"] for ep in buf._episodes]).astype(np.float32)
    print(f"collected {len(obs)} (obs,action) pairs across {len(buf._episodes)} episodes", flush=True)

    # --- LATENT-BC: the current pipeline (world model + behavior-cloned actor on WM features) ---
    wm = WorldModel(cfg, cfg.action_dim).to(cfg.device)
    wm_opt = torch.optim.Adam(wm.parameters(), lr=cfg.lr)
    _train_world_model(cfg, wm, wm_opt, buf, wm_steps)
    latent_actor, latent_loss = bc_actor(cfg, wm, buf, bc_steps)

    # --- DIRECT-BC: the ablation (plain obs -> action MLP, no world model) ---
    direct = DirectPolicy(cfg.state_dim, cfg.action_dim)
    direct_loss = train_direct_bc(direct, obs, act, direct_steps, device=cfg.device)
    print(f"trained  latent bc_loss={latent_loss:.4f}  direct bc_loss={direct_loss:.4f}", flush=True)

    # --- eval BOTH on the SAME disjoint held-out maps ---
    _, (eval_start, eval_num) = train_eval_seed_split(int(getattr(cfg, "metadrive_num_scenarios", 1)),
                                                      int(getattr(cfg, "metadrive_eval_scenarios", 50)))
    cfg.metadrive_start_seed, cfg.metadrive_num_scenarios = eval_start, eval_num
    cfg.metadrive_render, cfg.metadrive_endless = False, False
    print(f"held-out seeds {eval_start}-{eval_start + eval_num - 1}", flush=True)

    latent = summarize_driving(_run_episodes(cfg, episodes, _ActorPolicy(cfg, wm, latent_actor)))
    print(f"  LATENT-BC : {_fmt(latent)}", flush=True)
    direct_s = summarize_driving(_run_episodes(cfg, episodes, _DirectAct(direct)))
    print(f"  DIRECT-BC : {_fmt(direct_s)}", flush=True)
    rnd = _FixedPolicy(lambda o: np.random.uniform(-1, 1, cfg.action_dim).astype(np.float32))
    print(f"  RANDOM    : {_fmt(summarize_driving(_run_episodes(cfg, episodes, rnd)))}", flush=True)

    better_route = direct_s["route_completion"] > latent["route_completion"] + 0.02
    less_offroad = direct_s["off_road_rate"] < latent["off_road_rate"] - 0.1
    verdict = ("DIRECT-BC outperforms LATENT-BC -> the under-trained WM latent is hurting control; "
               "cloning direct from the state vector is better"
               if (better_route or less_offroad)
               else "no clear win for DIRECT -> the latent is NOT the main bottleneck (look elsewhere)")
    print(f"VERDICT: {verdict}", flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    cs = int(a[0]) if len(a) > 0 else 4000
    ns = int(a[1]) if len(a) > 1 else 50
    ep = int(a[2]) if len(a) > 2 else 10
    main(collect_steps=cs, num_scenarios=ns, episodes=ep)

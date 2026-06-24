"""Roadmap A experiment: does adding RECOVERY data (perturbed IDM collection) fix the direct policy's
off-road drift? Trains two direct-BC policies on the SAME budget -- one on clean demos only, one on
clean + recovery demos -- and evaluates both on the SAME held-out maps. Only the data differs.

Success = clean+recovery has LOWER off-road and/or HIGHER route than clean-only. The recovery policy
is saved to runs/direct_bc/policy_recovery.pt so you can watch it:
    python -m scripts.watch_direct_bc runs/direct_bc/policy_recovery.pt

Usage:  python -m scripts.recovery_bc
        python -m scripts.recovery_bc 4000 4000 50 10     # clean_steps recovery_steps num_scenarios episodes
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from scripts.dagger import build_cfg
from scripts.ablate_direct_bc import _DirectAct
from scripts.eval_driving import _run_episodes, _FixedPolicy, _fmt
from eval.closed_loop import summarize_driving
from envs.metadrive_env import train_eval_seed_split
from training.train_reference import collect_idm
from training.recovery import collect_idm_perturbed
from training.direct_bc import DirectPolicy, train_direct_bc, save_direct


def _flat(buf):
    obs = np.concatenate([e["obs"] for e in buf._episodes]).astype(np.float32)
    act = np.concatenate([e["action"] for e in buf._episodes]).astype(np.float32)
    return obs, act


def main(clean_steps=4000, recovery_steps=4000, num_scenarios=50, episodes=10, direct_steps=6000):
    cfg = build_cfg(num_scenarios=num_scenarios)
    print(f"recovery experiment: clean-only vs clean+recovery direct-BC "
          f"(clean {clean_steps}, recovery {recovery_steps}, {num_scenarios} maps, {episodes} eps)", flush=True)

    clean = collect_idm(cfg, clean_steps); clean._flush()
    rec = collect_idm_perturbed(cfg, recovery_steps)
    co, ca = _flat(clean)
    ro, ra = _flat(rec)
    print(f"clean {len(co)} pairs ({len(clean._episodes)} eps), recovery {len(ro)} pairs "
          f"({len(rec._episodes)} eps)", flush=True)

    pol_clean = DirectPolicy(cfg.state_dim, cfg.action_dim)
    train_direct_bc(pol_clean, co, ca, direct_steps, device=cfg.device)
    pol_rec = DirectPolicy(cfg.state_dim, cfg.action_dim)
    train_direct_bc(pol_rec, np.concatenate([co, ro]), np.concatenate([ca, ra]), direct_steps, device=cfg.device)
    save_direct("runs/direct_bc/policy_recovery.pt", pol_rec, cfg.state_dim, cfg.action_dim)

    # eval BOTH on the SAME disjoint held-out maps
    _, (eval_start, eval_num) = train_eval_seed_split(int(getattr(cfg, "metadrive_num_scenarios", 1)),
                                                      int(getattr(cfg, "metadrive_eval_scenarios", 50)))
    cfg.metadrive_start_seed, cfg.metadrive_num_scenarios = eval_start, eval_num
    cfg.metadrive_render, cfg.metadrive_endless = False, False
    print(f"held-out seeds {eval_start}-{eval_start + eval_num - 1}", flush=True)

    clean_s = summarize_driving(_run_episodes(cfg, episodes, _DirectAct(pol_clean)))
    print(f"  CLEAN-only      : {_fmt(clean_s)}", flush=True)
    rec_s = summarize_driving(_run_episodes(cfg, episodes, _DirectAct(pol_rec)))
    print(f"  CLEAN+RECOVERY  : {_fmt(rec_s)}", flush=True)
    rnd = _FixedPolicy(lambda o: np.random.uniform(-1, 1, cfg.action_dim).astype(np.float32))
    print(f"  RANDOM          : {_fmt(summarize_driving(_run_episodes(cfg, episodes, rnd)))}", flush=True)

    helped = (rec_s["off_road_rate"] < clean_s["off_road_rate"] - 0.05
              or rec_s["route_completion"] > clean_s["route_completion"] + 0.02)
    print(f"VERDICT: {'recovery data HELPS (off-road down / route up) -> keep it' if helped else 'no clear gain from recovery data this run (try more perturbation / more data)'}",
          flush=True)
    print("watch the recovery policy:  python -m scripts.watch_direct_bc runs/direct_bc/policy_recovery.pt", flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    cs = int(a[0]) if len(a) > 0 else 4000
    rs = int(a[1]) if len(a) > 1 else 4000
    ns = int(a[2]) if len(a) > 2 else 50
    ep = int(a[3]) if len(a) > 3 else 10
    main(clean_steps=cs, recovery_steps=rs, num_scenarios=ns, episodes=ep)

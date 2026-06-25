"""Train the production direct obs->action driving policy (roadmap C): clean IDM demos + perturbation
recovery demos, scaled up and tunable. Saves runs/direct_bc/policy.pt and reports the FULL held-out
eval -- route/crash/off-road (deterministic seeds) AND the targeted recovery_rate. This is the
"train the best learned driver" entry point; ablate_direct_bc / recovery_bc are the diagnostics.

More data is the lever (direct-BC fits easily, bc_loss ~0.05 -> coverage-limited, not capacity).
--perturb-prob / --gamma tune how much recovery coverage the perturbation generates.

Usage:  python -m scripts.train_direct_policy                              # defaults (8k clean + 8k recovery)
        python -m scripts.train_direct_policy --clean 16000 --recovery 16000
        python -m scripts.train_direct_policy --perturb-prob 0.1 --gamma 0.25
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from scripts.dagger import build_cfg
from scripts.ablate_direct_bc import _DirectAct
from scripts.eval_driving import _run_episodes, _fmt
from scripts.eval_recovery import eval_recovery
from eval.closed_loop import summarize_driving, summarize_recovery
from envs.metadrive_env import train_eval_seed_split
from training.train_reference import collect_idm
from training.recovery import collect_idm_perturbed
from training.direct_bc import (DirectPolicy, DirectPolicyAux, train_direct_bc, train_direct_bc_aux,
                                save_direct, flatten_buffer)


def _rewards(buf):
    return np.concatenate([e["reward"] for e in buf._episodes]).astype(np.float32)


def _collect_pair(cfg, clean_steps, recovery_steps, perturb_prob, gamma):
    """Collect clean IDM + perturbation-recovery data for `cfg` and flatten to (obs, act, rew)."""
    clean = collect_idm(cfg, clean_steps); clean._flush()
    rec = collect_idm_perturbed(cfg, recovery_steps, perturb_prob=perturb_prob, gamma=gamma)
    co, ca = flatten_buffer(clean)
    ro, ra = flatten_buffer(rec)
    return (np.concatenate([co, ro]), np.concatenate([ca, ra]),
            np.concatenate([_rewards(clean), _rewards(rec)]))


def main(clean_steps=8000, recovery_steps=8000, direct_steps=8000, num_scenarios=50, episodes=10,
         perturb_prob=0.08, gamma=0.2, aux_weight=0.0, boost_scene=None, boost_steps=4000,
         out="runs/direct_bc/policy.pt"):
    cfg = build_cfg(num_scenarios=num_scenarios)
    print(f"train direct policy: clean {clean_steps} + recovery {recovery_steps} ({num_scenarios} maps), "
          f"perturb p={perturb_prob} gamma={gamma} aux_weight={aux_weight} "
          f"boost={boost_scene or 'none'} -> {out}", flush=True)

    obs, act, rew = _collect_pair(cfg, clean_steps, recovery_steps, perturb_prob, gamma)
    if boost_scene:                                  # extra clean+recovery on a weak geometry (e.g. "O")
        bcfg = build_cfg(num_scenarios=num_scenarios, road_map=boost_scene)
        bo, ba, br = _collect_pair(bcfg, boost_steps, boost_steps, perturb_prob, gamma)
        obs, act, rew = np.concatenate([obs, bo]), np.concatenate([act, ba]), np.concatenate([rew, br])
        print(f"boosted with {len(bo)} extra '{boost_scene}' pairs", flush=True)
    print(f"training on {len(obs)} pairs", flush=True)

    if aux_weight > 0:                                   # roadmap D: + auxiliary progress (reward) head
        policy = DirectPolicyAux(cfg.state_dim, cfg.action_dim)
        losses = train_direct_bc_aux(policy, obs, act, rew, direct_steps, aux_weight=aux_weight, device=cfg.device)
        print(f"trained AUX (action {losses['action']:.4f} aux {losses['aux']:.4f}) -> {out}", flush=True)
    else:
        policy = DirectPolicy(cfg.state_dim, cfg.action_dim)
        loss = train_direct_bc(policy, obs, act, direct_steps, device=cfg.device)
        print(f"trained (bc_loss {loss:.4f}) -> {out}", flush=True)
    save_direct(out, policy, cfg.state_dim, cfg.action_dim)

    # full held-out eval: driving metrics + targeted recovery (both on the same fixed seeds)
    _, (eval_start, eval_num) = train_eval_seed_split(int(cfg.metadrive_num_scenarios),
                                                      int(cfg.metadrive_eval_scenarios))
    cfg.metadrive_start_seed, cfg.metadrive_num_scenarios = eval_start, eval_num
    cfg.metadrive_render, cfg.metadrive_endless = False, False
    print(f"held-out seeds {eval_start}-{eval_start + eval_num - 1}", flush=True)
    drive = summarize_driving(_run_episodes(cfg, episodes, _DirectAct(policy)))
    print(f"  DRIVE    : {_fmt(drive)}", flush=True)
    recov = summarize_recovery(eval_recovery(cfg, policy, episodes))
    print(f"  RECOVERY : recovery_rate {recov['recovery_rate']:.0%}  mean_route {recov['mean_route']:.0%}  "
          f"(n={recov['n']})", flush=True)
    # log this run as a milestone so the accuracy-over-the-project chart grows (scripts.plot_milestones)
    try:
        from training.progress_log import append_milestone
        tag = (f"direct+rec{'+aux' if aux_weight > 0 else ''}"
               f"{'+' + boost_scene if boost_scene else ''} {(clean_steps + recovery_steps) // 1000}k")
        append_milestone("runs/milestones.csv", tag, drive)
    except Exception as e:
        print(f"  (milestone log skipped: {e!r})", flush=True)
    print(f"watch it:  python -m scripts.watch_direct_bc {out}", flush=True)


def parse_args(argv):
    p = argparse.ArgumentParser(prog="train_direct_policy")
    p.add_argument("--clean", dest="clean_steps", type=int, default=8000)
    p.add_argument("--recovery", dest="recovery_steps", type=int, default=8000)
    p.add_argument("--bc-steps", dest="direct_steps", type=int, default=8000)
    p.add_argument("--num-scenarios", dest="num_scenarios", type=int, default=50)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--perturb-prob", dest="perturb_prob", type=float, default=0.08)
    p.add_argument("--gamma", type=float, default=0.2)
    p.add_argument("--aux-weight", dest="aux_weight", type=float, default=0.0,
                   help="roadmap D: weight on the auxiliary progress (reward) head; 0 = plain direct-BC")
    p.add_argument("--boost-scene", dest="boost_scene", default=None,
                   help="mix in extra clean+recovery data on this block type (e.g. O for roundabout)")
    p.add_argument("--boost-steps", dest="boost_steps", type=int, default=4000,
                   help="clean AND recovery steps to collect for --boost-scene")
    p.add_argument("--out", default="runs/direct_bc/policy.pt")
    return p.parse_args(argv)


if __name__ == "__main__":
    a = parse_args(sys.argv[1:])
    main(clean_steps=a.clean_steps, recovery_steps=a.recovery_steps, direct_steps=a.direct_steps,
         num_scenarios=a.num_scenarios, episodes=a.episodes, perturb_prob=a.perturb_prob,
         gamma=a.gamma, aux_weight=a.aux_weight, boost_scene=a.boost_scene, boost_steps=a.boost_steps,
         out=a.out)

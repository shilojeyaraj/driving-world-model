"""Direct-policy DAgger: fix behavioral cloning's distribution shift by rolling out the CURRENT
LEARNED DIRECT POLICY (no world model), collecting its failure states, relabeling with IDM expert
actions, and retraining iteratively. Combines the direct policy's proven state encoding (no WM
bottleneck, route 39%+ from plain BC) with DAgger's theoretically optimal covariate-shift fix.

Each iteration: roll out the current DirectPolicy so it drifts into the states IT actually
fails at, query IDM for the correct action at each of those states, aggregate into the growing
dataset, and retrain. The dataset grows monotonically; DAgger's no-regret guarantee applies.

See training/dagger.py for the WM-based DAgger (uses RSSM state carry).
"""
import dataclasses
import os
import shutil

import numpy as np
import torch

from data.replay_buffer import SequenceReplayBuffer
from training.dagger import relabel_action, extend_buffer
from training.direct_bc import DirectPolicy, train_direct_bc, save_direct, flatten_buffer


def direct_dagger_rollout(cfg, policy, steps, seed=0):
    """Roll out DirectPolicy in MetaDrive. Policy drives (visits its own failure states); IDM
    relabels each state with the correct expert action. Returns a SequenceReplayBuffer.
    Stateless -- no RSSM needed (that's the whole point of the direct policy)."""
    from metadrive.envs import MetaDriveEnv
    from metadrive.policy.idm_policy import IDMPolicy
    from envs.metadrive_env import adapt_obs, metadrive_config

    np.random.seed(seed)
    torch.manual_seed(seed)
    md = metadrive_config(cfg)
    md["use_render"] = False
    env = MetaDriveEnv(md)
    policy.eval()
    buf = SequenceReplayBuffer(steps + cfg.seq_len, cfg.seq_len)

    obs = adapt_obs(env.reset()[0], "state")
    idm = IDMPolicy(env.agent, 0)
    obs_l, act_l, rew_l, done_l = [], [], [], []
    try:
        for _ in range(steps):
            with torch.no_grad():
                a_policy = policy(
                    torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                ).squeeze(0).numpy()
            a_idm = relabel_action(idm.act(env.agent.id))
            raw, r, term, trunc, _ = env.step(a_policy)
            done = bool(term or trunc)
            obs_l.append(obs)
            act_l.append(a_idm)
            rew_l.append(float(r))
            done_l.append(done)
            if done:
                obs = adapt_obs(env.reset()[0], "state")
                idm = IDMPolicy(env.agent, 0)
            else:
                obs = adapt_obs(raw, "state")
    finally:
        env.close()
    return extend_buffer(buf, obs_l, act_l, rew_l, done_l)


def direct_dagger_train(cfg, iters=3, clean_steps=8000, recovery_steps=8000, rollout_steps=2000,
                        direct_steps=8000, perturb_prob=0.08, gamma=0.2,
                        boost_scene=None, boost_steps=4000,
                        out="runs/direct_dagger/policy.pt", eval_episodes=5):
    """Start from clean+recovery data, then iterate: roll out current policy, collect failure
    states relabeled by IDM, aggregate, retrain. Saves every iteration + the best held-out model.

    Args:
        iters: DAgger rounds after the iter-0 BC baseline.
        rollout_steps: learner-driven, IDM-relabeled steps per round.
        boost_scene: road_map code for a weak geometry, e.g. 'O' for roundabout -- adds extra
            clean+recovery data for that scene at iter 0 (same semantics as train_direct_policy).
    """
    from training.train_reference import collect_idm
    from training.recovery import collect_idm_perturbed
    from scripts.train_direct_policy import _collect_pair, _rewards
    from scripts.ablate_direct_bc import _DirectAct
    from scripts.eval_driving import _run_episodes, _fmt
    from scripts.eval_recovery import eval_recovery
    from eval.closed_loop import summarize_driving, summarize_recovery
    from envs.metadrive_env import train_eval_seed_split
    from training.progress_log import append_milestone, append_progress, progress_row

    run_dir = os.path.dirname(out) or "runs/direct_dagger"
    os.makedirs(run_dir, exist_ok=True)

    # Iter 0: clean + perturbation-recovery data (the proven direct-policy baseline)
    obs, act, rew = _collect_pair(cfg, clean_steps, recovery_steps, perturb_prob, gamma)
    if boost_scene:
        from scripts.dagger import build_cfg as _build_cfg
        bcfg = _build_cfg(int(cfg.metadrive_num_scenarios), road_map=boost_scene)
        bo, ba, br = _collect_pair(bcfg, boost_steps, boost_steps, perturb_prob, gamma)
        obs = np.concatenate([obs, bo])
        act = np.concatenate([act, ba])
        rew = np.concatenate([rew, br])
        print(f"boosted with {len(bo)} extra '{boost_scene}' pairs", flush=True)
    print(f"iter 0: {len(obs)} initial pairs (clean+recovery)", flush=True)

    policy = DirectPolicy(cfg.state_dim, cfg.action_dim)
    loss = train_direct_bc(policy, obs, act, direct_steps, device=cfg.device)
    save_direct(out, policy, cfg.state_dim, cfg.action_dim)
    shutil.copyfile(out, os.path.join(run_dir, "policy_iter0.pt"))
    print(f"iter 0: bc_loss={loss:.4f} -> {out}", flush=True)

    _, (eval_start, eval_num) = train_eval_seed_split(
        int(cfg.metadrive_num_scenarios), int(cfg.metadrive_eval_scenarios))
    eval_cfg = dataclasses.replace(cfg, metadrive_start_seed=eval_start,
                                   metadrive_num_scenarios=eval_num,
                                   metadrive_render=False, metadrive_endless=False)

    best_route, best_iter = -1.0, -1

    def _eval_iter(it):
        nonlocal best_route, best_iter
        if eval_episodes <= 0:
            return
        drive = summarize_driving(_run_episodes(eval_cfg, eval_episodes, _DirectAct(policy)))
        recov = summarize_recovery(eval_recovery(eval_cfg, policy, eval_episodes))
        print(f"  iter {it} DRIVE    : {_fmt(drive)}", flush=True)
        print(f"  iter {it} RECOVERY : recovery_rate {recov['recovery_rate']:.0%}  "
              f"mean_route {recov['mean_route']:.0%}", flush=True)
        try:
            append_progress(os.path.join(run_dir, "progress.csv"), progress_row(it, drive))
        except Exception as e:
            print(f"  (progress log skipped: {e!r})", flush=True)
        route = float(drive.get("route_completion", -1.0))
        if route > best_route:
            best_route, best_iter = route, it
            shutil.copyfile(out, os.path.join(run_dir, "policy_best.pt"))
            print(f"  iter {it}: new BEST route {route:.0%} -> {run_dir}/policy_best.pt", flush=True)

    _eval_iter(0)

    for it in range(1, iters + 1):
        buf = direct_dagger_rollout(cfg, policy, rollout_steps, seed=it)
        ro, ra = flatten_buffer(buf)
        obs = np.concatenate([obs, ro])
        act = np.concatenate([act, ra])
        print(f"iter {it}: aggregated -> {len(obs)} pairs ({len(ro)} new from rollout)", flush=True)

        loss = train_direct_bc(policy, obs, act, direct_steps, device=cfg.device)
        save_direct(out, policy, cfg.state_dim, cfg.action_dim)
        shutil.copyfile(out, os.path.join(run_dir, f"policy_iter{it}.pt"))
        print(f"iter {it}: bc_loss={loss:.4f} -> {out}", flush=True)
        _eval_iter(it)

    if best_iter >= 0:
        print(f"BEST model = iter {best_iter} (route {best_route:.0%}) at "
              f"{os.path.join(run_dir, 'policy_best.pt')}", flush=True)
    try:
        tag = f"direct-dagger+{'O' if boost_scene == 'O' else boost_scene or 'noboost'} {iters}it"
        from training.progress_log import read_progress
        rows = read_progress(os.path.join(run_dir, "progress.csv"))
        if rows:
            best_row = max(rows, key=lambda r: r["route_completion"])
            append_milestone("runs/milestones.csv", tag,
                             {"route_completion": best_row["route_completion"],
                              "crash_rate": best_row["crash_rate"],
                              "off_road_rate": best_row["off_road_rate"]})
    except Exception as e:
        print(f"  (milestone log skipped: {e!r})", flush=True)
    print(f"plot the learning curve:  python -m scripts.plot_progress "
          f"{os.path.join(run_dir, 'progress.csv')}", flush=True)
    return policy

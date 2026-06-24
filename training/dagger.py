"""DAgger (Dataset Aggregation) for driving: fix behavior-cloning's distribution-shift failure
(plain BC of IDM hits route ~3% -- it never saw a recovery, so it can't recover). Each iteration we
ROLL OUT THE CURRENT LEARNER so it drifts into the states it actually visits, ask the IDM EXPERT to
relabel each of those states with the correct action, AGGREGATE into a growing buffer, and retrain.
See docs/superpowers/specs/2026-06-24-dagger-design.md.

The crux (verified): query IDM at a state it isn't driving by instantiating an IDMPolicy bound to
the ego vehicle and calling .act(agent_id) -- while stepping the sim with the LEARNER's action.

Reuses train_reference's BC machinery (collect_idm / bc_actor / eval_critic) + dreamer_loop's
_train_world_model. Output is the standard {wm, actor, critic, cfg} checkpoint, so eval_driving and
watch_metadrive_3d work on runs/dagger/ckpt.pt unchanged.
"""
import os
import json
import shutil

import numpy as np
import torch

from data.replay_buffer import SequenceReplayBuffer
from models.world_model import WorldModel
from models.actor_critic import Actor
from training.train_reference import collect_idm, bc_actor, eval_critic
from training.dreamer_loop import _train_world_model
from utils import save_checkpoint


def relabel_action(raw_action):
    """The expert label stored for behavior cloning: FINITE and in the env's action range [-1,1]. PURE.
    MetaDrive's IDM returns an UN-normalized longitudinal command, and at near-collision states the
    learner drives into it emits a huge emergency-brake value (observed throttle ~ -198) or, in
    degenerate states (speed~0, no lead vehicle), a NaN/inf. The agent's action space is [-1,1] and
    env.step clips to it, so -198 *is* full brake = -1. An unclipped target the tanh-bounded actor
    can never match blows bc_loss up (~12M); a NaN target makes the loss NaN -> SILENT collapse.
    nan_to_num first (NaN->0 neutral, +-inf->large) THEN clip -- np.clip alone leaves NaN as NaN."""
    a = np.nan_to_num(np.asarray(raw_action, dtype=np.float32), nan=0.0)
    return np.clip(a, -1.0, 1.0)


def dagger_capacity(collect_steps, iters, rollout_steps, margin=0):
    """Replay capacity that holds ALL planned data (iter-0 IDM base + every rollout), so the buffer's
    oldest-first eviction never discards the clean expert demonstrations over a long run. PURE."""
    return collect_steps + iters * rollout_steps + margin


def extend_buffer(buf, obs, actions, rewards, dones):
    """Add a collected trajectory to `buf`, then flush so its trailing steps form a CLOSED episode
    and can't bleed into the next rollout added to the same buffer. DAgger pours many rollouts into
    one growing buffer; without the flush, the tail of one rollout and the head of the next would be
    stitched into a single 'episode' containing a transition that never happened. PURE (no sim)."""
    for o, a, r, d in zip(obs, actions, rewards, dones):
        buf.add(o, a, float(r), bool(d))
    buf._flush()                                   # close the trailing run -> episode boundary
    return buf


def idm_relabel_rollout(cfg, wm, actor, steps, seed=0, buf=None):
    """Drive MetaDrive with the LEARNER (wm+actor) so it drifts, and relabel every visited state with
    the IDM expert's action -> (obs, IDM_action, reward, done). Adds the rollout to `buf` (a new one
    if None) and returns it. The learner controls the car (collect its own failure states); IDM only
    advises (the correct recovery action). Headless. RSSM state-carry mirrors eval_driving._ActorPolicy."""
    from metadrive.envs import MetaDriveEnv
    from metadrive.policy.idm_policy import IDMPolicy
    from envs.metadrive_env import adapt_obs, metadrive_config

    np.random.seed(seed); torch.manual_seed(seed)
    md = metadrive_config(cfg); md["use_render"] = False
    env = MetaDriveEnv(md)
    device = torch.device(cfg.device)
    wm.eval(); actor.eval()
    if buf is None:
        buf = SequenceReplayBuffer(cfg.buffer_capacity, cfg.seq_len)

    def reset_state():
        return wm.rssm.initial_state(1, device), torch.zeros(1, cfg.action_dim, device=device)

    obs = adapt_obs(env.reset()[0], "state")
    idm = IDMPolicy(env.agent, 0)                  # expert advisor bound to the ego (NOT driving)
    state, prev = reset_state()
    obs_l, act_l, rew_l, done_l = [], [], [], []
    try:
        for _ in range(steps):
            with torch.no_grad():
                e = wm.encoder(torch.as_tensor(obs, device=device).float().unsqueeze(0))
                state, _, _ = wm.rssm.obs_step(state, prev, e)
                a_actor, _ = actor(torch.cat(state, dim=-1), deterministic=True)
            a_idm = relabel_action(idm.act(env.agent.id))     # expert label: finite + clipped to [-1,1]
            raw, r, term, trunc, info = env.step(a_actor.squeeze(0).cpu().numpy())  # LEARNER drives
            done = bool(term or trunc)
            # Guard #3: a non-finite obs would silently poison the WM/critic with NaN. Fail loud --
            # dagger_train saves every iteration, so prior progress is preserved.
            assert np.all(np.isfinite(obs)), "non-finite observation from MetaDrive in DAgger rollout"
            obs_l.append(obs); act_l.append(a_idm); rew_l.append(float(r)); done_l.append(done)
            prev = a_actor
            if done:
                obs = adapt_obs(env.reset()[0], "state")
                idm = IDMPolicy(env.agent, 0)      # ego is recreated on reset -> rebind the advisor
                state, prev = reset_state()
            else:
                obs = adapt_obs(raw, "state")
    finally:
        env.close()
    return extend_buffer(buf, obs_l, act_l, rew_l, done_l)


def _eval_heldout(out, episodes, with_idm=False):
    """Per-iteration progress: route/success/crash on the DISJOINT held-out maps (eval_driving forces
    them). Returns the {actor, random, idm} summaries (or None on failure). Best-effort -- a flaky
    eval must never kill the training run. IDM is slow, so it's only run when with_idm (iter 0)."""
    try:
        from scripts.eval_driving import main as eval_main
        return eval_main(ckpt=out, episodes=episodes, with_idm=with_idm)
    except Exception as e:
        print(f"  (held-out eval skipped: {e!r})", flush=True)
        return None


def _log_progress(it, summaries, run_dir):
    """Append the iteration's ACTOR metrics to progress.csv (a learning curve), and at iter 0 record
    the Random/IDM route baselines. Best-effort -- visualization must never kill training.
    Plot with: python -m scripts.plot_progress"""
    if not summaries or not summaries.get("actor"):
        return
    try:
        from training.progress_log import append_progress, progress_row
        append_progress(os.path.join(run_dir, "progress.csv"), progress_row(it, summaries["actor"]))
        if it == 0:
            bl = {}
            if summaries.get("random"):
                bl["random_route"] = float(summaries["random"]["route_completion"])
            if summaries.get("idm"):
                bl["idm_route"] = float(summaries["idm"]["route_completion"])
            with open(os.path.join(run_dir, "baselines.json"), "w") as f:
                json.dump(bl, f)
    except Exception as e:
        print(f"  (progress log skipped: {e!r})", flush=True)


def dagger_train(cfg, iters=3, collect_steps=4000, rollout_steps=2000, wm_steps=1000,
                 bc_steps=1000, critic_steps=1000, out="runs/dagger/ckpt.pt", eval_episodes=5):
    """Iteration 0 = train_reference (clean IDM data -> WM -> BC actor -> critic). Each later iter:
    relabel-rollout the current learner -> aggregate -> retrain WM + actor + critic -> save -> eval."""
    run_dir = os.path.dirname(out) or "runs/dagger"
    os.makedirs(run_dir, exist_ok=True)
    buf = collect_idm(cfg, collect_steps)          # iter-0 expert demonstrations
    buf._flush()                                   # close trailing run so relabel data can't bleed in
    buf.capacity = max(buf.capacity, dagger_capacity(collect_steps, iters, rollout_steps, cfg.seq_len))
    print(f"iter 0: collected {len(buf)} IDM steps across {len(buf._episodes)} episodes", flush=True)

    wm = WorldModel(cfg, cfg.action_dim).to(cfg.device)
    wm_opt = torch.optim.Adam(wm.parameters(), lr=cfg.lr)
    actor = critic = None
    best_route, best_iter = -1.0, -1                # DAgger can REGRESS -- never lose the best model
    for it in range(iters + 1):
        if it > 0:                                 # grow the dataset with learner-visited, IDM-labeled states
            idm_relabel_rollout(cfg, wm, actor, rollout_steps, seed=it, buf=buf)
            print(f"iter {it}: aggregated -> {len(buf)} steps, {len(buf._episodes)} episodes", flush=True)
        wm_m = _train_world_model(cfg, wm, wm_opt, buf, wm_steps)
        actor, bc_loss = bc_actor(cfg, wm, buf, bc_steps)
        critic, critic_loss = eval_critic(cfg, wm, buf, critic_steps)
        save_checkpoint(out, wm, actor, critic, cfg)
        shutil.copyfile(out, os.path.join(run_dir, f"ckpt_iter{it}.pt"))   # keep EVERY iteration
        print(f"iter {it}: recon={float(wm_m.get('recon', 0)):.3f} bc_loss={bc_loss:.4f} "
              f"critic_loss={critic_loss:.4f} -> saved {out} (+ ckpt_iter{it}.pt)", flush=True)
        if eval_episodes > 0:
            sums = _eval_heldout(out, eval_episodes, with_idm=(it == 0))   # IDM baseline once, at iter 0
            _log_progress(it, sums, run_dir)                               # -> progress.csv (+ baselines.json)
            route = float((sums or {}).get("actor", {}).get("route_completion", -1.0)) if sums else -1.0
            if route > best_route:                                         # track the best held-out model
                best_route, best_iter = route, it
                shutil.copyfile(out, os.path.join(run_dir, "ckpt_best.pt"))
                print(f"iter {it}: new BEST route {route:.0%} -> {os.path.join(run_dir, 'ckpt_best.pt')}", flush=True)
    if best_iter >= 0:
        print(f"BEST model = iter {best_iter} (route {best_route:.0%}) at {os.path.join(run_dir, 'ckpt_best.pt')}", flush=True)
    print(f"done. plot the learning curve:  python -m scripts.plot_progress "
          f"{os.path.join(run_dir, 'progress.csv')}", flush=True)
    return wm, actor, critic

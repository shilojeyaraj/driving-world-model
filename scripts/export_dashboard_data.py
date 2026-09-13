"""Offline export for the web dashboard: run the REAL pipelines (ablation, direct-policy DAgger
with recovery data, the documented WM-exploitation failure) and dump every artifact a static
dashboard needs -- rollout frames + per-step telemetry, learning curves, per-scene breakdown,
and the honest failure case study -- as plain JSON/JPEG under `dashboard_data/`.

No fabricated numbers: every value here comes from an actual MetaDrive episode or gradient step
run in this process. Scaled down from the historical multi-hour runs (see README/experiments/*.md
for the fully-scaled numbers) so the whole export finishes in minutes on a laptop CPU -- the
dashboard clearly labels which numbers are "live-reproduced this run" vs "historical, documented".

Usage:  python -m scripts.export_dashboard_data                      # full export, defaults
        python -m scripts.export_dashboard_data --quick              # smaller sizes, for a dry run
"""
import os, sys, json, argparse, time, dataclasses

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

OUT_ROOT = "dashboard_data"
SCENES = {"S": "straight", "C": "curve", "X": "intersection", "O": "roundabout"}


def _out(*parts):
    p = os.path.join(OUT_ROOT, *parts)
    os.makedirs(os.path.dirname(p) if "." in os.path.basename(p) else p, exist_ok=True)
    return p


def _dump(name, obj):
    path = _out(f"{name}.json")
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"  wrote {path}", flush=True)
    return path


# ---------------------------------------------------------------------------------------
# Phase 1: the direct-policy ablation (RANDOM vs LATENT-BC vs DIRECT-BC) -- experiments/*, ARCHITECTURE.md §10
# ---------------------------------------------------------------------------------------
def run_ablation(collect_steps, num_scenarios, episodes, wm_steps, bc_steps, direct_steps):
    from scripts.dagger import build_cfg
    from scripts.eval_driving import _run_episodes, _ActorPolicy, _FixedPolicy
    from scripts.ablate_direct_bc import _DirectAct
    from eval.closed_loop import summarize_driving
    from envs.metadrive_env import train_eval_seed_split
    from training.train_reference import collect_idm, bc_actor
    from training.dreamer_loop import _train_world_model
    from training.direct_bc import DirectPolicy, train_direct_bc
    from models.world_model import WorldModel

    print(f"[ablation] collecting {collect_steps} IDM steps on {num_scenarios} maps...", flush=True)
    cfg = build_cfg(num_scenarios=num_scenarios)
    buf = collect_idm(cfg, collect_steps); buf._flush()
    obs = np.concatenate([ep["obs"] for ep in buf._episodes]).astype(np.float32)
    act = np.concatenate([ep["action"] for ep in buf._episodes]).astype(np.float32)

    wm = WorldModel(cfg, cfg.action_dim).to(cfg.device)
    wm_opt = torch.optim.Adam(wm.parameters(), lr=cfg.lr)
    print(f"[ablation] training world model ({wm_steps} steps) + latent BC actor ({bc_steps})...", flush=True)
    _train_world_model(cfg, wm, wm_opt, buf, wm_steps)
    latent_actor, latent_loss = bc_actor(cfg, wm, buf, bc_steps)

    print(f"[ablation] training direct BC ({direct_steps} steps)...", flush=True)
    direct = DirectPolicy(cfg.state_dim, cfg.action_dim)
    direct_loss = train_direct_bc(direct, obs, act, direct_steps, device=cfg.device)

    _, (eval_start, eval_num) = train_eval_seed_split(int(cfg.metadrive_num_scenarios),
                                                       int(cfg.metadrive_eval_scenarios))
    cfg.metadrive_start_seed, cfg.metadrive_num_scenarios = eval_start, eval_num
    cfg.metadrive_render, cfg.metadrive_endless = False, False

    print(f"[ablation] held-out eval ({episodes} episodes/policy)...", flush=True)
    latent = summarize_driving(_run_episodes(cfg, episodes, _ActorPolicy(cfg, wm, latent_actor)))
    direct_s = summarize_driving(_run_episodes(cfg, episodes, _DirectAct(direct)))
    rnd = _FixedPolicy(lambda o: np.random.uniform(-1, 1, cfg.action_dim).astype(np.float32))
    random_s = summarize_driving(_run_episodes(cfg, episodes, rnd))

    result = {
        "bc_loss": {"latent": latent_loss, "direct": direct_loss},
        "driving": {"random": random_s, "latent_bc": latent, "direct_bc": direct_s},
        "config": {"collect_steps": collect_steps, "num_scenarios": num_scenarios,
                   "episodes": episodes, "wm_steps": wm_steps, "bc_steps": bc_steps,
                   "direct_steps": direct_steps},
    }
    print(f"[ablation] bc_loss latent={latent_loss:.4f} direct={direct_loss:.4f}  "
          f"route random={random_s['route_completion']:.0%} latent={latent['route_completion']:.0%} "
          f"direct={direct_s['route_completion']:.0%}", flush=True)
    _dump("ablation", result)
    return result, direct  # hand back the direct policy so later phases can start from something real


# ---------------------------------------------------------------------------------------
# Phase 2: the documented WM model-exploitation failure -- experiments/010
# ---------------------------------------------------------------------------------------
def _collect_random(cfg, steps):
    """Like training.collect.collect, but closes its env -- MetaDrive's engine is a process-wide
    singleton that asserts if a second env is created while an earlier one is still open."""
    from envs.base import make_env
    from data.replay_buffer import SequenceReplayBuffer

    env = make_env(cfg)
    buf = SequenceReplayBuffer(cfg.buffer_capacity, cfg.seq_len)
    try:
        obs = env.reset()
        for _ in range(steps):
            a = np.random.uniform(-1, 1, cfg.action_dim).astype(np.float32)
            nxt, r, done, _ = env.step(a)
            buf.add(obs, a, r, done)
            obs = env.reset() if done else nxt
    finally:
        env.close()
    return buf


def run_exploitation_case_study(collect_steps, wm_steps, behavior_steps, eval_episodes):
    from scripts.dagger import build_cfg
    from models.world_model import WorldModel
    from models.actor_critic import Actor, Critic
    from eval.closed_loop import closed_loop_eval
    from envs.base import make_env

    print(f"[exploitation] single-shot pipeline: collect {collect_steps} random steps -> "
          f"train WM ({wm_steps}) -> train policy in imagination ({behavior_steps})...", flush=True)
    cfg = build_cfg(num_scenarios=50)
    cfg.metadrive_endless = False
    buf = _collect_random(cfg, collect_steps)
    print(f"  collected {len(buf)} steps across {len(buf._episodes)} usable episodes", flush=True)

    wm = WorldModel(cfg, cfg.action_dim)
    opt = torch.optim.Adam(wm.parameters(), lr=cfg.lr)
    recon_before = None
    for step in range(wm_steps):
        if not buf.can_sample():
            break
        batch = {k: torch.as_tensor(v) for k, v in buf.sample(cfg.batch_size).items()}
        loss, m = wm.assemble_loss(batch)
        if recon_before is None:
            recon_before = float(m.get("recon", 0.0))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(wm.parameters(), 100.0); opt.step()
    recon_after = float(m.get("recon", 0.0))
    kl_after = float(m.get("kl", 0.0))

    feat_dim = cfg.deter_dim + cfg.stoch_dim
    actor, critic = Actor(cfg, feat_dim, cfg.action_dim), Critic(cfg, feat_dim)
    from training.train_behavior import train_behavior_in_imagination
    beh_m = train_behavior_in_imagination(cfg, wm, buf, actor, critic, steps=behavior_steps, log_every=behavior_steps)
    imagined_return = float(beh_m.get("imagined_return", 0.0))

    env = make_env(cfg)
    print(f"[exploitation] closed-loop eval in the REAL sim ({eval_episodes} episodes)...", flush=True)
    real = closed_loop_eval(actor, wm, env, episodes=eval_episodes, max_steps=cfg.max_episode_steps)
    env.close()

    result = {
        "recon_before": recon_before, "recon_after": recon_after, "kl_after": kl_after,
        "imagined_return": imagined_return,
        "real_actor_return": real["actor_return"], "real_random_return": real["random_return"],
        "actor_steer": real["actor_steer"], "actor_throttle": real["actor_throttle"],
        "exploits_model": bool(real["actor_return"] < real["random_return"]),
        "config": {"collect_steps": collect_steps, "wm_steps": wm_steps, "behavior_steps": behavior_steps,
                   "eval_episodes": eval_episodes},
    }
    print(f"[exploitation] imagined_return={imagined_return:.2f}  real_actor={real['actor_return']:.2f}  "
          f"real_random={real['random_return']:.2f}  steer={real['actor_steer']:.3f} "
          f"throttle={real['actor_throttle']:.3f}", flush=True)
    _dump("exploitation_case_study", result)
    from utils import save_checkpoint
    ckpt_path = "runs/dashboard_export/exploit/ckpt.pt"
    save_checkpoint(ckpt_path, wm, actor, critic, cfg)
    print(f"  saved {ckpt_path}", flush=True)
    return result, wm, actor


# ---------------------------------------------------------------------------------------
# Phase 3: direct-policy DAgger with roundabout recovery boost -- README's headline route numbers
# ---------------------------------------------------------------------------------------
def run_direct_dagger(iters, clean_steps, recovery_steps, rollout_steps, direct_steps,
                       boost_scene, boost_steps, eval_episodes, num_scenarios):
    from scripts.dagger import build_cfg
    from training.direct_dagger import direct_dagger_train
    from training.progress_log import read_progress

    cfg = build_cfg(num_scenarios=num_scenarios)
    out = "runs/dashboard_export/direct_dagger/policy.pt"
    print(f"[dagger] {iters} rounds, boost={boost_scene}, out={out}", flush=True)
    direct_dagger_train(cfg, iters=iters, clean_steps=clean_steps, recovery_steps=recovery_steps,
                        rollout_steps=rollout_steps, direct_steps=direct_steps,
                        boost_scene=boost_scene, boost_steps=boost_steps,
                        out=out, eval_episodes=eval_episodes)
    progress = read_progress(os.path.join(os.path.dirname(out), "progress.csv"))
    _dump("dagger_progress", {"iterations": progress,
                              "config": {"iters": iters, "clean_steps": clean_steps,
                                         "recovery_steps": recovery_steps, "rollout_steps": rollout_steps,
                                         "direct_steps": direct_steps, "boost_scene": boost_scene,
                                         "boost_steps": boost_steps, "eval_episodes": eval_episodes}})
    best_path = os.path.join(os.path.dirname(out), "policy_best.pt")
    return best_path if os.path.exists(best_path) else out


# ---------------------------------------------------------------------------------------
# Phase 4: per-scene breakdown of the final policy
# ---------------------------------------------------------------------------------------
def run_by_scene(policy_path, episodes, scenes, num_scenarios):
    from scripts.dagger import build_cfg
    from scripts.ablate_direct_bc import _DirectAct
    from scripts.eval_driving import _run_episodes
    from eval.closed_loop import summarize_driving
    from envs.metadrive_env import train_eval_seed_split
    from training.direct_bc import load_direct

    policy = load_direct(policy_path)
    results = {}
    for scene in scenes:
        cfg = build_cfg(num_scenarios=num_scenarios, road_map=scene)
        _, (eval_start, eval_num) = train_eval_seed_split(int(cfg.metadrive_num_scenarios), 50)
        cfg.metadrive_start_seed, cfg.metadrive_num_scenarios = eval_start, eval_num
        cfg.metadrive_render, cfg.metadrive_endless = False, False
        summary = summarize_driving(_run_episodes(cfg, episodes, _DirectAct(policy)))
        results[scene] = {"name": SCENES.get(scene, scene), **summary}
        print(f"[by_scene] {scene} ({SCENES.get(scene, scene)}): "
              f"route={summary['route_completion']:.0%} off_road={summary['off_road_rate']:.0%}", flush=True)
    _dump("by_scene", {"scenes": results, "config": {"episodes": episodes}})
    return results


# ---------------------------------------------------------------------------------------
# Phase 5: rollout recording -- a compact MP4 (native scrubbing, tiny payload vs a JPEG
# sequence) + per-step telemetry JSON, for the dashboard's rollout viewer.
# ---------------------------------------------------------------------------------------
def _frame(env, size=320):
    from PIL import Image
    rgb = np.asarray(env.render(mode="topdown", window=False))
    return np.asarray(Image.fromarray(rgb).resize((size, size)))


def _save_video(frames, path, fps=15):
    import imageio
    with imageio.get_writer(path, fps=fps, codec="libx264", quality=None,
                            output_params=["-pix_fmt", "yuv420p", "-crf", "28"]) as w:
        for f in frames:
            w.append_data(f)


def record_rollout(name, cfg_builder, act_fn, steps, frame_size=256, fps=15, use_idm=False):
    from metadrive.envs import MetaDriveEnv
    from envs.metadrive_env import metadrive_config, adapt_obs

    cfg = cfg_builder()
    md = metadrive_config(cfg); md["use_render"] = False
    if use_idm:
        from metadrive.policy.idm_policy import IDMPolicy
        md["agent_policy"] = IDMPolicy
    env = MetaDriveEnv(md)
    raw, _ = env.reset()
    obs = adapt_obs(raw, "state")

    frames, telemetry = [], []
    try:
        for t in range(steps):
            action = act_fn(obs)
            raw, r, terminated, truncated, info = env.step(np.asarray(action, dtype=np.float32))
            frames.append(_frame(env, size=frame_size))
            telemetry.append({
                "t": t, "steer": float(action[0]), "throttle": float(action[1]), "reward": float(r),
                "route_completion": float(info.get("route_completion", 0.0)),
                "out_of_road": bool(info.get("out_of_road", False)),
                "crash": bool(info.get("crash") or info.get("crash_vehicle") or info.get("crash_object")),
            })
            if terminated or truncated:
                break
            obs = adapt_obs(raw, "state")
    finally:
        env.close()

    video_path = _out("rollouts", name, "clip.mp4")
    _save_video(frames, video_path, fps=fps)
    manifest = {"name": name, "frame_count": len(frames), "fps": fps,
                "frame_size": frame_size, "steps": len(telemetry), "telemetry": telemetry}
    _dump(f"rollouts/{name}/manifest", manifest)
    size_kb = os.path.getsize(video_path) / 1024
    print(f"[rollout] {name}: {len(frames)} frames -> {video_path} ({size_kb:.0f}KB)", flush=True)
    return manifest


def record_all_rollouts(direct_policy_path, exploit_wm=None, exploit_actor=None,
                          steps_per_scene=250, frame_size=256):
    from scripts.dagger import build_cfg
    from training.direct_bc import load_direct

    direct_policy = load_direct(direct_policy_path)
    direct_policy.eval()

    def _direct_act(obs):
        with torch.no_grad():
            return direct_policy(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy()

    zero_act = lambda obs: np.zeros(2, dtype=np.float32)  # IDM drives itself; action is ignored

    for scene in SCENES:
        cfg_builder = lambda scene=scene: build_cfg(num_scenarios=50, road_map=scene)
        # IDM baseline (drives itself; action is ignored)
        record_rollout(f"idm_{scene}", cfg_builder, zero_act, steps_per_scene, frame_size, use_idm=True)
        # our best direct policy
        record_rollout(f"direct_{scene}", cfg_builder, _direct_act, steps_per_scene, frame_size)

    if exploit_wm is not None and exploit_actor is not None:
        record_exploitation_rollout(exploit_wm, exploit_actor, steps_per_scene, frame_size)


def record_exploitation_rollout(wm, actor, steps, frame_size):
    """Record the degenerate single-shot policy actually driving (or failing to) in the real sim --
    the honest failure clip for the case-study section."""
    from scripts.dagger import build_cfg
    from metadrive.envs import MetaDriveEnv
    from envs.metadrive_env import metadrive_config, adapt_obs

    cfg = build_cfg(num_scenarios=50)
    md = metadrive_config(cfg); md["use_render"] = False
    env = MetaDriveEnv(md)
    raw, _ = env.reset()
    obs = adapt_obs(raw, "state")

    rssm = wm.rssm
    state = rssm.initial_state(1, torch.device(cfg.device))
    prev_action = torch.zeros(1, cfg.action_dim)

    frames, telemetry = [], []
    actor.eval(); wm.eval()
    try:
        with torch.no_grad():
            for t in range(steps):
                e = wm.encoder(torch.as_tensor(obs).float().unsqueeze(0))
                state, _, _ = rssm.obs_step(state, prev_action, e)
                feat = torch.cat(state, dim=-1)
                action, _ = actor(feat, deterministic=True)
                a = action.squeeze(0).cpu().numpy()
                raw, r, terminated, truncated, info = env.step(np.asarray(a, dtype=np.float32))
                frames.append(_frame(env, size=frame_size))
                telemetry.append({"t": t, "steer": float(a[0]), "throttle": float(a[1]), "reward": float(r),
                                  "route_completion": float(info.get("route_completion", 0.0)),
                                  "out_of_road": bool(info.get("out_of_road", False))})
                prev_action = action
                if terminated or truncated:
                    break
                obs = adapt_obs(raw, "state")
    finally:
        env.close()

    fps = 8  # slower than the scene clips -- this one is only ~2s of real time, stretch it for legibility
    video_path = _out("rollouts", "exploit_wm", "clip.mp4")
    _save_video(frames, video_path, fps=fps)
    manifest = {"name": "exploit_wm", "frame_count": len(frames), "fps": fps,
               "frame_size": frame_size, "steps": len(telemetry), "telemetry": telemetry}
    _dump("rollouts/exploit_wm/manifest", manifest)
    size_kb = os.path.getsize(video_path) / 1024
    print(f"[rollout] exploit_wm: {len(frames)} frames -> {video_path} ({size_kb:.0f}KB)", flush=True)


# ---------------------------------------------------------------------------------------
DIRECT_POLICY_PATH = "runs/dashboard_export/direct_dagger/policy_best.pt"
EXPLOIT_CKPT_PATH = "runs/dashboard_export/exploit/ckpt.pt"


def run_phase(phase, scale):
    """Each phase creates its own MetaDrive engine. MetaDrive's engine is a process-wide singleton
    that doesn't always tear down cleanly after many envs in one process (we hit
    `AssertionError: Can not call this API after engine initialization!` running phases back to
    back) -- so the CLI driver below runs each phase in its OWN subprocess instead."""
    if phase == "ablation":
        run_ablation(collect_steps=int(4000 * scale), num_scenarios=50, episodes=max(3, int(10 * scale)),
                    wm_steps=int(1000 * scale), bc_steps=int(1000 * scale), direct_steps=int(4000 * scale))
    elif phase == "exploit":
        run_exploitation_case_study(collect_steps=int(4000 * scale), wm_steps=int(1500 * scale),
                                    behavior_steps=int(1500 * scale), eval_episodes=5)
    elif phase == "dagger":
        run_direct_dagger(iters=3, clean_steps=int(8000 * scale), recovery_steps=int(8000 * scale),
                          rollout_steps=int(2000 * scale), direct_steps=int(8000 * scale),
                          boost_scene="O", boost_steps=int(4000 * scale), eval_episodes=max(3, int(5 * scale)),
                          num_scenarios=50)
    elif phase == "scene":
        run_by_scene(DIRECT_POLICY_PATH, episodes=max(3, int(10 * scale)), scenes="SCXO", num_scenarios=50)
    elif phase == "rollouts":
        exploit_wm = exploit_actor = None
        if os.path.exists(EXPLOIT_CKPT_PATH):
            from utils import load_models
            _, exploit_wm, exploit_actor, _ = load_models(EXPLOIT_CKPT_PATH)
        record_all_rollouts(DIRECT_POLICY_PATH, exploit_wm, exploit_actor,
                            steps_per_scene=int(250 * scale), frame_size=256)
    else:
        raise ValueError(f"unknown phase: {phase}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true", help="small sizes, for a dry run")
    p.add_argument("--phase", required=True,
                   choices=["ablation", "exploit", "dagger", "scene", "rollouts"],
                   help="run exactly one phase (each phase = its own MetaDrive engine)")
    a = p.parse_args()
    os.makedirs(OUT_ROOT, exist_ok=True)
    scale = 0.25 if a.quick else 1.0
    t0 = time.time()
    run_phase(a.phase, scale)
    print(f"\n[{a.phase}] done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

"""Driving-usability eval: is a trained policy ACTUALLY usable -- does it get down the route
without crashing? Runs the actor in MetaDrive for N episodes and reports route completion %,
success rate (reached destination), crash rate, and off-road rate, with random + IDM-expert
baselines to compare against.

Unlike scripts/eval_closed_loop.py (which reports raw return in whatever env the checkpoint's cfg
names), this FORCES MetaDrive -- so it works on a `train_on_gesture` "your-style" checkpoint too,
whose saved cfg says env="dummy". Real terminations are kept on (NOT endless), so crash/off-road
are detected.

Usage:  python -m scripts.eval_driving runs/reference/ckpt.pt              # 5 episodes + baselines
        python -m scripts.eval_driving runs/gesture_reference/ckpt.pt 10   # your-style policy, 10 eps
        python -m scripts.eval_driving runs/reference/ckpt.pt 5 SSSS       # on a highway
        python -m scripts.eval_driving runs/reference/ckpt.pt 5 - noidm    # skip the (slower) IDM baseline
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


def _metadrive(cfg, idm=False):
    from metadrive.envs import MetaDriveEnv
    from envs.metadrive_env import metadrive_config
    md = metadrive_config(cfg)
    md["use_render"] = False                       # headless eval (metrics, not a window)
    if idm:
        from metadrive.policy.idm_policy import IDMPolicy
        md["agent_policy"] = IDMPolicy
    return MetaDriveEnv(md)


def _run_episodes(cfg, episodes, act, idm=False):
    """Run `episodes` MetaDrive episodes, `act(obs)->action` choosing each step (ignored under IDM,
    which drives itself). Returns a list of episode_outcome records."""
    from envs.metadrive_env import adapt_obs
    from eval.closed_loop import episode_outcome
    env = _metadrive(cfg, idm=idm)
    max_steps = cfg.max_episode_steps
    recs = []
    try:
        for _ in range(episodes):
            obs = adapt_obs(env.reset()[0], "state")
            total, info, steps = 0.0, {}, 0
            act.reset()
            for t in range(max_steps):
                raw, r, term, trunc, info = env.step(act(obs))
                total += float(r); steps = t + 1
                if term or trunc:
                    break
                obs = adapt_obs(raw, "state")
            recs.append(episode_outcome(info, total, steps))
    finally:
        env.close()
    return recs


class _ActorPolicy:
    """Carries the RSSM posterior across steps (like eval/closed_loop._run_actor_episode) and acts
    on the actor's mode. `reset()` re-inits the latent state at each episode boundary."""
    def __init__(self, cfg, wm, actor):
        import torch
        self.t = torch
        self.cfg, self.wm, self.actor = cfg, wm, actor
        self.device = torch.device(cfg.device)
        wm.eval(); actor.eval()
        self.reset()

    def reset(self):
        self.state = self.wm.rssm.initial_state(1, self.device)
        self.prev = self.t.zeros(1, self.cfg.action_dim, device=self.device)

    def __call__(self, obs):
        with self.t.no_grad():
            e = self.wm.encoder(self.t.as_tensor(obs, device=self.device).float().unsqueeze(0))
            self.state, _, _ = self.wm.rssm.obs_step(self.state, self.prev, e)
            action, _ = self.actor(self.t.cat(self.state, dim=-1), deterministic=True)
            self.prev = action
            return action.squeeze(0).cpu().numpy()


class _FixedPolicy:
    """A stateless action source (random / IDM-dummy)."""
    def __init__(self, fn):
        self._fn = fn

    def reset(self):
        pass

    def __call__(self, obs):
        return self._fn(obs)


def _fmt(s):
    if s.get("n", 0) == 0:
        return "no episodes"
    return (f"route {s['route_completion']:.0%}±{s.get('route_completion_std', 0):.0%}  "
            f"success {s['success_rate']:.0%}  crash {s['crash_rate']:.0%}  off-road {s['off_road_rate']:.0%}  "
            f"return {s['mean_return']:+.1f}±{s.get('return_std', 0):.1f}  "
            f"steps {s['mean_steps']:.0f}  (n={s['n']})")


def main(ckpt, episodes=5, road_map=None, with_idm=True):
    from utils import load_models
    from eval.closed_loop import summarize_driving
    cfg, wm, actor, critic = load_models(ckpt)
    if actor is None:
        raise SystemExit(f"checkpoint {ckpt} has no actor; train a policy first.")
    cfg.env, cfg.obs_type, cfg.metadrive_render = "metadrive", "state", False   # force MetaDrive
    cfg.metadrive_endless = False                                               # real terminations
    if isinstance(road_map, str) and road_map.isdigit():
        road_map = int(road_map)
    if road_map is not None:
        cfg.metadrive_map = road_map
    # HELD-OUT maps: grade on the eval seed range, which is DISJOINT from the training range the
    # checkpoint used -> measures generalization (driving roads it never trained on), not memorization.
    from envs.metadrive_env import train_eval_seed_split
    _, (eval_start, eval_num) = train_eval_seed_split(int(getattr(cfg, "metadrive_num_scenarios", 1)),
                                                      int(getattr(cfg, "metadrive_eval_scenarios", 50)))
    cfg.metadrive_start_seed, cfg.metadrive_num_scenarios = eval_start, eval_num

    print(f"driving-usability eval: {ckpt}  ({episodes} episodes, map={cfg.metadrive_map}, "
          f"held-out seeds {eval_start}-{eval_start + eval_num - 1})", flush=True)
    actor_sum = summarize_driving(_run_episodes(cfg, episodes, _ActorPolicy(cfg, wm, actor)))
    print(f"  ACTOR : {_fmt(actor_sum)}", flush=True)
    rnd = _FixedPolicy(lambda obs: np.random.uniform(-1, 1, cfg.action_dim).astype(np.float32))
    random_sum = summarize_driving(_run_episodes(cfg, episodes, rnd))
    print(f"  RANDOM: {_fmt(random_sum)}", flush=True)
    idm_sum = None
    if with_idm:
        dummy = _FixedPolicy(lambda obs: np.zeros(cfg.action_dim, np.float32))
        idm_sum = summarize_driving(_run_episodes(cfg, episodes, dummy, idm=True))
        print(f"  IDM   : {_fmt(idm_sum)}", flush=True)
    return {"actor": actor_sum, "random": random_sum, "idm": idm_sum}    # for progress logging


if __name__ == "__main__":
    a = sys.argv[1:]
    ck = a[0] if len(a) > 0 else "runs/reference/ckpt.pt"
    eps = int(a[1]) if len(a) > 1 and a[1].isdigit() else 5
    rm = a[2] if len(a) > 2 and a[2] not in ("-", "none") else None
    idm = "noidm" not in a
    main(ckpt=ck, episodes=eps, road_map=rm, with_idm=idm)

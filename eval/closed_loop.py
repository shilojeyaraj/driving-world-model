"""Closed-loop eval: run the trained actor in the env and measure DRIVING, not prediction.

Concept:  The second eval axis. A model can ace open-loop prediction and still drive badly.
Question: If open-loop looks great but closed-loop fails, what are the likely causes?

Metrics: route completion %, collisions/km, lane-keeping error, interventions/km.
Baselines to beat: random, the data-collection policy, behavior-cloning (no world model).

----------------------------------------------------------------------------------------
WHY this is a DIFFERENT axis from open-loop:
  Open-loop feeds the model TRUE actions and asks "can you predict?". Closed-loop lets the
  policy CHOOSE the actions and asks "do those choices produce good outcomes in the REAL env?".
  Crucially, the policy now visits states IT drives itself into -- which can drift away from
  the data the world model was trained on (distribution shift), and small per-step errors can
  COMPOUND. Great open-loop + bad closed-loop usually means exactly that: the model predicts
  in-distribution but the policy steers into states the model gets wrong (or the actor learned
  to exploit model errors -- the imagination failure mode from train_behavior).

Running the actor here mirrors `observe`: keep the RSSM posterior state across env steps,
feeding the PREVIOUS action and the CURRENT observation (a_{-1}=0 at reset).
"""
import numpy as np
import torch


def _run_actor_episode(actor, world_model, env, max_steps):
    cfg = world_model.cfg
    device = torch.device(cfg.device)
    rssm = world_model.rssm

    state = rssm.initial_state(1, device)
    prev_action = torch.zeros(1, cfg.action_dim, device=device)
    obs = env.reset()
    total, steers, throttles = 0.0, [], []
    with torch.no_grad():
        for _ in range(max_steps):
            e = world_model.encoder(torch.as_tensor(obs, device=device).float().unsqueeze(0))
            state, _, _ = rssm.obs_step(state, prev_action, e)     # posterior step from real obs
            feat = torch.cat(state, dim=-1)                         # [h; z]
            action, _ = actor(feat, deterministic=True)            # act on the mode
            a = action.squeeze(0).cpu().numpy()
            obs, reward, done, _ = env.step(a)
            total += reward
            steers.append(float(a[0]))                              # action = [steer, throttle]
            throttles.append(float(a[1]))
            prev_action = action
            if done:
                break
    return total, float(np.mean(steers)), float(np.mean(throttles))


def _run_random_episode(env, action_dim, max_steps):
    env.reset()
    total = 0.0
    for _ in range(max_steps):
        _, reward, done, _ = env.step(np.random.uniform(-1, 1, action_dim).astype(np.float32))
        total += reward
        if done:
            break
    return total


def closed_loop_eval(actor, world_model, env, episodes=10, max_steps=None):
    """Run `episodes` episodes of the actor in the env (RSSM state carried across steps) and
    of a random baseline (the data-collection policy is random in DummyEnv). Returns mean
    episode return for each plus the actor's mean steer/throttle."""
    cfg = world_model.cfg
    max_steps = max_steps or cfg.max_episode_steps
    actor.eval()
    world_model.eval()

    a_ret, a_steer, a_thr = [], [], []
    for _ in range(episodes):
        ret, st, th = _run_actor_episode(actor, world_model, env, max_steps)
        a_ret.append(ret); a_steer.append(st); a_thr.append(th)
    r_ret = [_run_random_episode(env, cfg.action_dim, max_steps) for _ in range(episodes)]

    return {
        "actor_return": float(np.mean(a_ret)),
        "random_return": float(np.mean(r_ret)),
        "actor_steer": float(np.mean(a_steer)),
        "actor_throttle": float(np.mean(a_thr)),
    }

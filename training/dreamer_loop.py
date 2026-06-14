"""The iterated Dreamer training loop (the real algorithm, vs. the single-shot in
train_behavior.train_behavior).

    seed the buffer with random data, then repeat:
      1. COLLECT with the current policy (+ exploration) -> append to the replay buffer
      2. train the WORLD MODEL on the buffer
      3. train the POLICY (actor-critic) in imagination on the frozen world model

Why iterate: a policy trained inside a model built only from RANDOM data exploits the model's
errors -- it's confident in imagination but drives off-road in reality (experiments/010).
Collecting WITH the policy grounds the world model in the states the policy actually visits,
closing the imagination-vs-reality gap. This is what makes the agent actually drive.

Run:  python -m training.dreamer_loop          (DummyEnv demo)
"""
import numpy as np
import torch

from config import get_config
from envs.base import make_env
from data.replay_buffer import SequenceReplayBuffer
from models.world_model import WorldModel
from models.actor_critic import Actor, Critic
from training.train_behavior import behavior_losses, start_states_from_batch


def collect_random(env, buf, steps, action_dim):
    """Seed the buffer with a random policy (the world model needs something to start)."""
    obs = env.reset()
    for _ in range(steps):
        a = np.random.uniform(-1, 1, action_dim).astype(np.float32)
        nxt, r, done, _ = env.step(a)
        buf.add(obs, a, r, done)
        obs = env.reset() if done else nxt
    return buf


def collect_with_policy(env, world_model, actor, buf, steps, explore_std=0.3, device=None):
    """Run the actor IN the env (RSSM posterior carried across steps), with exploration noise,
    appending transitions to the buffer. This is the data that grounds the next world model."""
    device = device or torch.device(world_model.cfg.device)
    rssm = world_model.rssm
    A = world_model.cfg.action_dim
    state = rssm.initial_state(1, device)
    prev_action = torch.zeros(1, A, device=device)
    obs = env.reset()
    world_model.eval()
    with torch.no_grad():
        for _ in range(steps):
            e = world_model.encoder(torch.as_tensor(obs, device=device).float().unsqueeze(0))
            state, _, _ = rssm.obs_step(state, prev_action, e)
            action, _ = actor(torch.cat(state, dim=-1))           # stochastic sample (exploration)
            a = action.squeeze(0).cpu().numpy()
            if explore_std > 0:
                a = a + np.random.normal(0, explore_std, size=A).astype(np.float32)
            a = np.clip(a, -1.0, 1.0).astype(np.float32)
            nxt, r, done, _ = env.step(a)
            buf.add(obs, a, r, done)
            prev_action = torch.as_tensor(a, device=device).unsqueeze(0)
            if done:
                obs = env.reset()
                state = rssm.initial_state(1, device)
                prev_action = torch.zeros(1, A, device=device)
            else:
                obs = nxt
    return buf


def _train_world_model(cfg, wm, wm_opt, buf, steps):
    wm.requires_grad_(True)
    wm.train()
    last = {}
    for _ in range(steps):
        if not buf.can_sample():
            break
        batch = {k: torch.as_tensor(v) for k, v in buf.sample(cfg.batch_size).items()}
        loss, last = wm.assemble_loss(batch)
        wm_opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(wm.parameters(), 100.0); wm_opt.step()
    return last


def _train_behavior(cfg, wm, actor, critic, actor_opt, critic_opt, buf, steps, device):
    wm.requires_grad_(False)            # frozen during behavior; grads still flow to the actor
    wm.eval()
    last = {}
    for _ in range(steps):
        if not buf.can_sample():
            break
        batch = {k: torch.as_tensor(v) for k, v in buf.sample(cfg.batch_size).items()}
        start = start_states_from_batch(wm, batch, device)
        a_loss, c_loss, last = behavior_losses(cfg, wm, actor, critic, start)
        actor_opt.zero_grad(); critic_opt.zero_grad()
        (a_loss + c_loss).backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), 100.0)
        torch.nn.utils.clip_grad_norm_(critic.parameters(), 100.0)
        actor_opt.step(); critic_opt.step()
    return last


def dreamer_train(cfg, env=None, iters=10, seed_steps=1000, collect_per_iter=500,
                  wm_steps=300, behavior_steps=300, explore_std=0.3, eval_fn=None, eval_every=0):
    """Run the iterated loop. Returns (world_model, actor, critic, buffer). Pass `env` to reuse a
    single (e.g. MetaDrive) sim instance; otherwise one is created and closed."""
    device = torch.device(cfg.device)
    own_env = env is None
    env = env or make_env(cfg)
    buf = SequenceReplayBuffer(cfg.buffer_capacity, cfg.seq_len)

    wm = WorldModel(cfg, cfg.action_dim).to(device)
    feat_dim = cfg.deter_dim + cfg.stoch_dim
    actor = Actor(cfg, feat_dim, cfg.action_dim).to(device)
    critic = Critic(cfg, feat_dim).to(device)
    wm_opt = torch.optim.Adam(wm.parameters(), lr=cfg.lr)
    actor_opt = torch.optim.Adam(actor.parameters(), lr=cfg.actor_lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=cfg.critic_lr)

    collect_random(env, buf, seed_steps, cfg.action_dim)
    for it in range(iters):
        collect_with_policy(env, wm, actor, buf, collect_per_iter, explore_std, device)
        wm_m = _train_world_model(cfg, wm, wm_opt, buf, wm_steps)
        beh_m = _train_behavior(cfg, wm, actor, critic, actor_opt, critic_opt, buf, behavior_steps, device)
        print(f"iter {it+1}/{iters}  buf={len(buf)}  "
              f"recon={float(wm_m.get('recon', 0)):.3f} kl={float(wm_m.get('kl', 0)):.3f}  "
              f"imagined_return={float(beh_m.get('imagined_return', 0)):.3f}", flush=True)
        if eval_every and eval_fn and (it + 1) % eval_every == 0:
            print(f"  eval: {eval_fn(actor, wm, env)}", flush=True)

    if own_env:
        env.close()
    return wm, actor, critic, buf


if __name__ == "__main__":
    from eval.closed_loop import closed_loop_eval
    cfg = get_config(env="dummy", obs_type="state", device="cpu", max_episode_steps=100,
                     deter_dim=64, stoch_dim=16, hidden_dim=64, seq_len=16, imagine_horizon=15,
                     actor_lr=3e-3, critic_lr=3e-3)
    wm, actor, critic, _ = dreamer_train(cfg, iters=6, seed_steps=800, collect_per_iter=400,
                                         wm_steps=200, behavior_steps=200)
    print(closed_loop_eval(actor, wm, make_env(cfg), episodes=5, max_steps=100))

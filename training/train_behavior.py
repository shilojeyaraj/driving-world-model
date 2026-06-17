"""Behavior (actor-critic) training in imagination -- spec §4.6.

The policy never touches the env here: it rolls the RSSM forward with img_step (SAMPLED,
reparameterized -- unlike open-loop prediction, which uses the mean), the critic scores
imagined states, and we optimize lambda-returns. The world model is frozen.

See models/actor_critic.py.   Run: python -m training.train_behavior

----------------------------------------------------------------------------------------
THE PAYOFF (why this works with zero env steps):
  Because the world model is a differentiable simulator. From a real start state we dream a
  trajectory; the imagined REWARD for action a_i is read from feat_{i+1} (the state that
  consumed a_i -- the same action-timing alignment as world_model.assemble_loss). Maximizing
  the lambda-return backpropagates that return THROUGH the dynamics + reward head into the
  actor (a value gradient). Failure mode: if the world model is wrong, the actor learns to
  exploit the model's errors -- it optimizes a fantasy. Mitigations: detach start states,
  entropy bonus, grad clipping.
"""
import torch

from config import get_config


def lambda_returns(rewards, conts, values, gamma, lam):
    """rewards (N,H), conts (N,H), values (N,H+1) -> lambda-returns (N,H).

    V^lambda_t = r_t + gamma*c_t*[(1-lam)*v_{t+1} + lam*V^lambda_{t+1}],  V^lambda_H = v_H.
    Computed backward from the bootstrap value at the horizon end."""
    H = rewards.shape[1]
    out = [None] * H
    future = values[:, H]                                   # V^lambda_H = v_H (bootstrap)
    for t in reversed(range(H)):
        future = rewards[:, t] + gamma * conts[:, t] * ((1 - lam) * values[:, t + 1] + lam * future)
        out[t] = future
    return torch.stack(out, dim=1)


def imagine_rollout(world_model, actor, start_state, horizon):
    """Dream `horizon` steps from start_state=(h,z) using the actor's reparameterized actions.
    Returns feats (N,H+1,F), rewards (N,H), conts (N,H), entropies (N,H).
    reward/cont for action a_i are decoded from feat_{i+1} (the state that consumed a_i)."""
    rssm, decoder = world_model.rssm, world_model.decoder
    h, z = start_state
    feat = torch.cat([h, z], dim=-1)
    feats, entropies = [feat], []
    for _ in range(horizon):
        action, entropy = actor(feat)                       # reparameterized sample
        entropies.append(entropy)
        (h, z), _ = rssm.img_step((h, z), action, sample=True)
        feat = torch.cat([h, z], dim=-1)
        feats.append(feat)
    feats = torch.stack(feats, dim=1)                       # (N, H+1, F)
    N, _, Fdim = feats.shape
    # Only reward/continue are needed in imagination; skip the (image) obs head entirely.
    dec = decoder(feats[:, 1:].reshape(N * horizon, Fdim), decode_obs=False)
    rewards = dec["reward"].reshape(N, horizon)
    conts = torch.sigmoid(dec["cont_logit"]).reshape(N, horizon)
    return feats, rewards, conts, torch.stack(entropies, dim=1)


def behavior_losses(cfg, world_model, actor, critic, start_state):
    """One imagination step -> (actor_loss, critic_loss, metrics)."""
    H = cfg.imagine_horizon
    feats, rewards, conts, entropies = imagine_rollout(world_model, actor, start_state, H)
    N, Hp1, Fdim = feats.shape

    # Critic reads DETACHED feats -> its gradient touches only critic params (not the actor).
    values = critic(feats.reshape(N * Hp1, Fdim).detach()).reshape(N, Hp1)

    # Actor return: bootstrap values are detached (constants), so the actor's gradient flows
    # only through the rewards -> dynamics -> actions (the value gradient).
    returns = lambda_returns(rewards, conts, values.detach(), cfg.gamma, cfg.lambda_)  # (N,H)

    actor_loss = -returns.mean() - cfg.entropy_coef * entropies.mean()
    critic_loss = ((values[:, :H] - returns.detach()) ** 2).mean()

    metrics = {
        "actor_loss": actor_loss.detach(),
        "critic_loss": critic_loss.detach(),
        "imagined_return": returns[:, 0].mean().detach(),   # full-horizon return from t=0
        "entropy": entropies.mean().detach(),
    }
    return actor_loss, critic_loss, metrics


def start_states_from_batch(world_model, batch, device):
    """Encode + observe a real batch, then use every per-step (h,z) as an imagination start
    state (DETACHED, so behavior grads never flow into the world model's training graph)."""
    obs, actions = batch["obs"].to(device), batch["action"].to(device)
    B, T = obs.shape[:2]
    deter = world_model.cfg.deter_dim
    with torch.no_grad():
        embeds = world_model.encoder(obs.reshape(B * T, *obs.shape[2:])).reshape(B, T, -1)
        feat = world_model.rssm.observe(
            embeds, actions, world_model.rssm.initial_state(B, device))["feat"]
    feat = feat.reshape(B * T, feat.shape[-1])
    return feat[:, :deter].detach(), feat[:, deter:].detach()


def train_behavior_in_imagination(cfg, world_model, buffer, actor, critic, steps=1000, log_every=100):
    """Train actor + critic purely in imagination from a frozen world model + a replay buffer."""
    device = torch.device(cfg.device)
    world_model.to(device).requires_grad_(False)            # frozen simulator
    world_model.eval()
    actor.to(device)
    critic.to(device)
    actor_opt = torch.optim.Adam(actor.parameters(), lr=cfg.actor_lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=cfg.critic_lr)

    last = {}
    for step in range(steps):
        batch = {k: torch.as_tensor(v) for k, v in buffer.sample(cfg.batch_size).items()}
        start = start_states_from_batch(world_model, batch, device)

        actor_loss, critic_loss, metrics = behavior_losses(cfg, world_model, actor, critic, start)
        actor_opt.zero_grad()
        critic_opt.zero_grad()
        (actor_loss + critic_loss).backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), 100.0)
        torch.nn.utils.clip_grad_norm_(critic.parameters(), 100.0)
        actor_opt.step()
        critic_opt.step()

        last = metrics
        if step % log_every == 0:
            print(step, {k: round(float(v), 4) for k, v in metrics.items()})
    return last


def train_behavior(cfg, wm_steps=3000, behavior_steps=3000):
    """End-to-end demo: collect -> train world model -> train behavior in imagination."""
    from training.collect import collect
    from models.world_model import WorldModel
    from models.actor_critic import Actor, Critic

    device = torch.device(cfg.device)
    buf = collect(cfg, num_steps=5000)

    wm = WorldModel(cfg, cfg.action_dim).to(device)
    opt = torch.optim.Adam(wm.parameters(), lr=cfg.lr)
    print(f"[1/2] training world model ({wm_steps} steps)...", flush=True)
    for step in range(wm_steps):
        if not buf.can_sample():
            break
        batch = {k: torch.as_tensor(v, device=device) for k, v in buf.sample(cfg.batch_size).items()}
        loss, m = wm.assemble_loss(batch)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(wm.parameters(), 100.0)
        opt.step()
        if step % 200 == 0:
            print(f"  wm {step}", {k: round(float(v), 4) for k, v in m.items()}, flush=True)

    feat_dim = cfg.deter_dim + cfg.stoch_dim
    actor, critic = Actor(cfg, feat_dim, cfg.action_dim), Critic(cfg, feat_dim)
    print(f"[2/2] training behavior in imagination ({behavior_steps} steps)...", flush=True)
    train_behavior_in_imagination(cfg, wm, buf, actor, critic, steps=behavior_steps)

    from utils import save_checkpoint
    ckpt_path = f"{cfg.log_dir}/behavior/ckpt.pt"
    save_checkpoint(ckpt_path, wm, actor, critic, cfg)
    print(f"saved checkpoint -> {ckpt_path}  (eval: python -m scripts.eval_closed_loop {ckpt_path})")
    return wm, actor, critic


if __name__ == "__main__":
    train_behavior(get_config(env="dummy", obs_type="state", device="cpu", max_episode_steps=200))

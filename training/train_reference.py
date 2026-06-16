"""Reference stack for driving feedback (GF3): a world model + a behavior-cloned REFERENCE actor
+ a policy-evaluation CRITIC, trained on MetaDrive's IDM expert. Saved in the standard
{world_model, actor, critic, config} checkpoint so the feedback engine just calls
utils.load_models (the actor = "how an expert drives here", the critic = "how good is this state").

Why IDM, not our trained actor: our RL actor collapses on MetaDrive (experiments/011); the
reference must actually drive well, so we imitate IDM.

Run:  python -m training.train_reference        (needs MetaDrive -- see docs/METADRIVE.md)
"""
import numpy as np
import torch

from config import get_config
from data.replay_buffer import SequenceReplayBuffer
from models.world_model import WorldModel
from models.actor_critic import Actor, Critic
from training.train_behavior import lambda_returns
from training.dreamer_loop import _train_world_model
from utils import save_checkpoint


def collect_idm(cfg, steps, seed=0):
    """Drive MetaDrive with the built-in IDM expert and record (obs, IDM-action, reward, done).
    The action the env actually applied is `info["action"]` (the dummy action we pass is ignored
    when agent_policy=IDMPolicy)."""
    from metadrive.envs import MetaDriveEnv
    from metadrive.policy.idm_policy import IDMPolicy
    from envs.metadrive_env import adapt_obs

    np.random.seed(seed)
    env = MetaDriveEnv(dict(use_render=False, horizon=cfg.max_episode_steps, agent_policy=IDMPolicy))
    buf = SequenceReplayBuffer(cfg.buffer_capacity, cfg.seq_len)
    dummy = np.zeros(cfg.action_dim, dtype=np.float32)
    obs = adapt_obs(env.reset()[0], cfg.obs_type)
    for _ in range(steps):
        raw, r, terminated, truncated, info = env.step(dummy)
        action = np.asarray(info["action"], dtype=np.float32)        # the IDM action actually applied
        done = bool(terminated or truncated)
        buf.add(obs, action, float(r), done)
        obs = adapt_obs(env.reset()[0], cfg.obs_type) if done else adapt_obs(raw, cfg.obs_type)
    env.close()
    return buf


def _feats(cfg, wm, batch):
    """Encode + observe a batch -> per-step features (B,T,F), no grad (WM frozen)."""
    obs, actions = batch["obs"], batch["action"]
    B, T = obs.shape[:2]
    with torch.no_grad():
        embeds = wm.encoder(obs.reshape(B * T, *obs.shape[2:])).reshape(B, T, -1)
        feat = wm.rssm.observe(embeds, actions, wm.rssm.initial_state(B, obs.device))["feat"]
    return feat, B, T


def bc_actor(cfg, wm, buf, steps, lr=None):
    """Behavior-clone an Actor on the demonstrator's (feat_t -> action_t) pairs (MSE on the
    deterministic action). The WM is frozen; only the actor learns."""
    wm.requires_grad_(False); wm.eval()
    feat_dim = cfg.deter_dim + cfg.stoch_dim
    actor = Actor(cfg, feat_dim, cfg.action_dim).to(cfg.device)
    opt = torch.optim.Adam(actor.parameters(), lr=lr or cfg.actor_lr)
    last = 0.0
    for _ in range(steps):
        if not buf.can_sample():
            break
        batch = {k: torch.as_tensor(v) for k, v in buf.sample(cfg.batch_size).items()}
        feat, B, T = _feats(cfg, wm, batch)
        pred, _ = actor(feat.reshape(B * T, feat_dim), deterministic=True)
        target = batch["action"].reshape(B * T, cfg.action_dim)
        loss = ((pred - target) ** 2).mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), 100.0); opt.step()
        last = float(loss.detach())
    return actor, last


def eval_critic(cfg, wm, buf, steps, lr=None):
    """Policy-evaluate the demonstrator: regress a Critic to lambda-returns of the RECORDED
    rewards along the real trajectories (so V(s) ~ 'how good is this state under expert driving').
    Reuses training.train_behavior.lambda_returns."""
    wm.requires_grad_(False); wm.eval()
    feat_dim = cfg.deter_dim + cfg.stoch_dim
    critic = Critic(cfg, feat_dim).to(cfg.device)
    opt = torch.optim.Adam(critic.parameters(), lr=lr or cfg.critic_lr)
    last = 0.0
    for _ in range(steps):
        if not buf.can_sample():
            break
        batch = {k: torch.as_tensor(v) for k, v in buf.sample(cfg.batch_size).items()}
        feat, B, T = _feats(cfg, wm, batch)
        values = critic(feat.reshape(B * T, feat_dim)).reshape(B, T)              # (B,T), grad to critic
        with torch.no_grad():
            rewards, done = batch["reward"], batch["done"]
            # lambda-returns over the real sequence: H=T-1, bootstrap with the critic's own values.
            returns = lambda_returns(rewards[:, :T - 1], 1.0 - done[:, :T - 1],
                                     values.detach(), cfg.gamma, cfg.lambda_)      # (B,T-1)
        loss = ((values[:, :T - 1] - returns) ** 2).mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), 100.0); opt.step()
        last = float(loss.detach())
    return critic, last


def train_reference(cfg, collect_steps=4000, wm_steps=1000, bc_steps=1000, critic_steps=1000,
                    out="runs/reference/ckpt.pt"):
    """Full reference stack: collect IDM data -> train WM -> BC actor -> eval critic -> save."""
    buf = collect_idm(cfg, collect_steps)
    print(f"collected {len(buf)} steps across {len(buf._episodes)} usable episodes", flush=True)

    wm = WorldModel(cfg, cfg.action_dim).to(cfg.device)
    wm_opt = torch.optim.Adam(wm.parameters(), lr=cfg.lr)
    wm_m = _train_world_model(cfg, wm, wm_opt, buf, wm_steps)
    actor, bc_loss = bc_actor(cfg, wm, buf, bc_steps)
    critic, critic_loss = eval_critic(cfg, wm, buf, critic_steps)

    save_checkpoint(out, wm, actor, critic, cfg)
    print(f"reference saved -> {out}  recon={float(wm_m.get('recon', 0)):.3f} "
          f"bc_loss={bc_loss:.4f} critic_loss={critic_loss:.4f}", flush=True)
    return wm, actor, critic


if __name__ == "__main__":
    train_reference(get_config(env="metadrive", obs_type="state", state_dim=259, action_dim=2,
                               deter_dim=128, stoch_dim=32, hidden_dim=128, seq_len=10,
                               max_episode_steps=200, lr=3e-4))

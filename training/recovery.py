"""Recovery-data collection (roadmap A): fix the direct policy's off-road drift (BC distribution
shift) by giving it RECOVERY demonstrations. We drive the car with the IDM expert but inject a
triangular steering perturbation so it drifts off the centerline; at each (drifted) state we record
IDM's CLEAN action as the label. The data then covers a tube around the lane, and the labels teach
"when off-center, steer back" -- exactly what pure centerline cloning never shows.

DART (Laskey 2017) / Codevilla 2019 §A.3 / ChauffeurNet -- the canonical, off-policy (no live expert
recovery needed) covariate-shift fix.
"""
import numpy as np


def triangular_impulse(t, t0, tau, sigma, gamma):
    """Codevilla et al. (2019) Eq 3: a triangular perturbation over [t0, t0+tau], peaking at
    sigma*gamma at the midpoint and zero outside. PURE. `t` and `tau` are in the same unit (steps)."""
    if tau <= 0:
        return 0.0
    val = 1.0 - abs(2.0 * (t - t0) / tau - 1.0)
    return float(sigma * gamma * max(0.0, val))


def _clean_label(idm, agent_id):
    """IDM's recommended action at the current state, finite + clipped to [-1,1] (its raw longitudinal
    command can be a huge emergency-brake or NaN in degenerate states; see training/dagger.py)."""
    a = np.nan_to_num(np.asarray(idm.act(agent_id), dtype=np.float32), nan=0.0)
    return np.clip(a, -1.0, 1.0)


def collect_idm_perturbed(cfg, steps, seed=0, perturb_prob=0.05, gamma=0.15, tau_range=(10, 40)):
    """Collect IDM data WITH recovery coverage. Headless. We drive the car ourselves: each step the
    LABEL is IDM's clean action; we EXECUTE label + a triangular STEERING impulse so the car drifts,
    then record (obs, clean IDM action, reward, done). `perturb_prob` starts a new impulse per step
    (when none active); `tau_range` is its duration in steps. Returns a SequenceReplayBuffer."""
    from metadrive.envs import MetaDriveEnv
    from metadrive.policy.idm_policy import IDMPolicy
    from envs.metadrive_env import adapt_obs, metadrive_config
    from data.replay_buffer import SequenceReplayBuffer

    np.random.seed(seed)
    md = metadrive_config(cfg); md["use_render"] = False
    env = MetaDriveEnv(md)
    buf = SequenceReplayBuffer(cfg.buffer_capacity, cfg.seq_len)
    obs = adapt_obs(env.reset()[0], cfg.obs_type)
    idm = IDMPolicy(env.agent, 0)
    impulse = None                                   # (t0, tau, sigma) of the active steering impulse
    try:
        for t in range(steps):
            label = _clean_label(idm, env.agent.id)  # IDM's correct action at the (drifted) state
            if impulse is None and np.random.rand() < perturb_prob:
                impulse = (t, int(np.random.randint(tau_range[0], tau_range[1] + 1)),
                           float(np.random.choice([-1.0, 1.0])))
            steer_perturb = 0.0
            if impulse is not None:
                t0, tau, sigma = impulse
                steer_perturb = triangular_impulse(t, t0, tau, sigma, gamma)
                if t >= t0 + tau:
                    impulse = None
            executed = label.copy()
            executed[0] = float(np.clip(executed[0] + steer_perturb, -1.0, 1.0))   # perturb STEERING only
            raw, r, terminated, truncated, info = env.step(executed)
            done = bool(terminated or truncated)
            buf.add(obs, label, float(r), done)      # record the CLEAN label at this state
            if done:
                obs = adapt_obs(env.reset()[0], cfg.obs_type)
                idm = IDMPolicy(env.agent, 0); impulse = None
            else:
                obs = adapt_obs(raw, cfg.obs_type)
    finally:
        env.close()
    buf._flush()
    return buf

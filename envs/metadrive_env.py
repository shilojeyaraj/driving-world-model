"""MetaDrive wrapper -- a REAL driving sim behind the same envs/base.py contract.

The rest of the codebase never imports MetaDrive; it only sees reset()/step() returning a
state vector or a (C,H,W) image. So adopting MetaDrive is localized to this file.

INSTALL + USAGE + GOTCHAS: see docs/METADRIVE.md. The #1 thing that breaks across MetaDrive
versions is the OBSERVATION SHAPE/DIM -- run `python -m scripts.probe_metadrive` to print the
actual state_dim / image shape for YOUR install, then set cfg.state_dim / cfg.image_size to match.

OK TO USE A LIBRARY / modify freely.
Install: pip install metadrive-simulator      Docs: https://metadrive-simulator.readthedocs.io
"""
import numpy as np

from .base import DrivingEnv


def adapt_obs(raw, obs_type):
    """Normalize a raw MetaDrive observation to the envs/base.py contract. Pure + dependency-free
    so it can be unit-tested without MetaDrive (this is where version differences bite):

    - state -> a flat float32 vector. (raw may be a dict for multi-modal obs.)
    - image -> a float32 (C,H,W) in [0,1]. MetaDrive may return HWC, may STACK frames as a
      trailing axis (H,W,C,stack), and may be 0..255; we take the last frame, move to CHW, scale.
    """
    if obs_type == "state":
        if isinstance(raw, dict):                       # multi-modal -> pick the vector entry
            raw = raw.get("state", next(iter(raw.values())))
        return np.asarray(raw, dtype=np.float32).reshape(-1)

    if isinstance(raw, dict):                            # image obs often nested under "image"
        raw = raw.get("image", next(iter(raw.values())))
    img = np.asarray(raw, dtype=np.float32)
    if img.ndim == 4:                                   # (H,W,C,stack) -> most recent frame
        img = img[..., -1]
    if img.ndim == 3 and img.shape[0] not in (1, 3):    # HWC -> CHW
        img = np.transpose(img, (2, 0, 1))
    if img.max() > 1.5:                                 # 0..255 -> 0..1
        img = img / 255.0
    return np.ascontiguousarray(img, dtype=np.float32)


def metadrive_config(cfg):
    """Build the MetaDrive config dict from our cfg -- the SCENE knobs live here so every entry
    point (wrapper, drive_gesture, watch/record scripts) picks the same map/traffic. `map` is
    int N (N random blocks) or a block-letter string (S straight, C curve, X intersection,
    O roundabout, T t-junction, r/R ramp); traffic_density sets how many other cars."""
    md = dict(use_render=bool(getattr(cfg, "metadrive_render", False)),
              horizon=cfg.max_episode_steps,
              traffic_density=float(getattr(cfg, "metadrive_traffic_density", 0.1)))
    m = getattr(cfg, "metadrive_map", None)
    if m is not None:
        md["map"] = m
    if cfg.obs_type == "image":
        md["image_observation"] = True              # VERSION-SPECIFIC; see docs/METADRIVE.md
    if getattr(cfg, "metadrive_endless", False):    # let a human drive through mistakes (no reset)
        md.update(out_of_road_done=False, on_continuous_line_done=False,
                  crash_vehicle_done=False, crash_object_done=False, crash_human_done=False)
    if md["use_render"]:                             # 3-D window: cut what a weak GPU has to draw
        md["window_size"] = tuple(getattr(cfg, "metadrive_window_size", (800, 600)))  # fewer pixels
        if getattr(cfg, "metadrive_low_graphics", True):
            md.update(show_skybox=False, show_logo=False)   # cheaper scene. NOTE: do NOT set
            # shadow_range=0 -- MetaDrive's PSSM asserts distance>0 and crashes. Shadows are turned
            # off at runtime instead (disable_shadows below), after the engine/pssm exists.
        if getattr(cfg, "metadrive_manual_control", False):
            md.update(manual_control=True, controller="keyboard")   # drive with WASD in the window
    return md


def applied_action(env, proposed, manual):
    """The action the vehicle ACTUALLY executed this step. In keyboard/manual mode MetaDrive
    overrides our env-input action with the controller's (WASD), so the real action lives on the
    agent (`last_current_action[-1]`) -- recording that is what makes the session the human's own
    driving. Non-manual modes (gesture/random) just executed `proposed`, so we pass it through.
    Defensive: any malformed env falls back to `proposed` rather than raising."""
    if not manual:
        return np.asarray(proposed, dtype=np.float32)
    try:
        return np.asarray(env.agent.last_current_action[-1], dtype=np.float32)
    except Exception:
        return np.asarray(proposed, dtype=np.float32)


def disable_shadows(env):
    """Best-effort: switch off PSSM shadow sampling on a live MetaDrive env (a real GPU cost on a
    weak/integrated card). Must run AFTER reset() created the engine. MetaDrive has no config flag
    for this and exposes only pssm.toggle_shadows_mode(), so we call it defensively -- a no-op if
    the internals differ across versions, never a crash."""
    try:
        pssm = getattr(getattr(env, "engine", None), "pssm", None)
        if pssm is not None and getattr(pssm, "use_pssm", False):
            pssm.toggle_shadows_mode()              # use_pssm True -> False (skip shadow lookup)
            return True
    except Exception:
        pass
    return False


class MetaDriveDrivingEnv(DrivingEnv):
    def __init__(self, cfg):
        self.cfg = cfg
        self.obs_type = cfg.obs_type
        self.action_dim = cfg.action_dim                # MetaDrive action = [steering, throttle], in [-1,1]
        from metadrive.envs import MetaDriveEnv         # imported here so the dummy env runs without it

        # Scene/render/traffic config (use_render=True opens the 3-D window; map sets the road).
        md_cfg = metadrive_config(cfg)
        try:
            self._env = MetaDriveEnv(config=md_cfg)
        except TypeError:                               # some versions take the dict positionally
            self._env = MetaDriveEnv(md_cfg)
        self.observation_space = getattr(self._env, "observation_space", None)
        self.metadrive_action_space = getattr(self._env, "action_space", None)

    def reset(self):
        out = self._env.reset()
        raw = out[0] if isinstance(out, tuple) else out  # gymnasium (obs, info) vs old gym obs
        return adapt_obs(raw, self.obs_type)

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        out = self._env.step(action)
        if len(out) == 5:                                # gymnasium: obs, r, terminated, truncated, info
            raw, reward, terminated, truncated, info = out
            done = bool(terminated or truncated)
        else:                                            # old gym: obs, r, done, info
            raw, reward, done, info = out
        return adapt_obs(raw, self.obs_type), float(reward), bool(done), info

    def close(self):
        if getattr(self, "_env", None) is not None:
            self._env.close()

"""Central config. The one knob that matters most: obs_type (the laptop/GPU switch).

OK TO USE A LIBRARY / modify freely.
"""
from dataclasses import dataclass


@dataclass
class Config:
    # --- observation mode: the laptop/GPU switch ---
    obs_type: str = "state"      # "state" (CPU-friendly) or "image" (needs a GPU)
    image_size: int = 64         # used only when obs_type == "image"
    state_dim: int = 35          # used only when obs_type == "state" (verify vs your env)

    # --- env ---
    env: str = "dummy"           # "dummy" | "metadrive" | "donkey"
    action_dim: int = 2          # [steer, throttle]
    max_episode_steps: int = 1000

    metadrive_render: bool = False   # True -> open MetaDrive's 3-D render window (needs a display)
    metadrive_map: object = None     # scene: None=default | int N (N random blocks) | str of block
                                     # letters: S straight, C curve, X intersection, O roundabout,
                                     # T t-junction, r/R on/off ramp, y merge. e.g. "SSSS"=highway, "X"=intersection
    metadrive_traffic_density: float = 0.1   # other-vehicle density (0.0 empty ... ~0.3 busy)
    metadrive_window_size: tuple = (800, 600) # 3-D render resolution; smaller = far smoother on a
                                              # weak/integrated GPU (MetaDrive's own default is 1200x900)
    metadrive_low_graphics: bool = True      # when rendering, drop shadows/skybox/logo for FPS
    metadrive_manual_control: bool = False   # drive the 3-D window yourself with WASD (no webcam)
    metadrive_endless: bool = False          # NEVER reset: off-road/line/crash terminations off AND
                                             # horizon unbounded -- a human drives continuously, no resets
    # map randomization (domain randomization): a pool of procedurally-generated maps so a policy
    # learns to DRIVE rather than memorize one road. Each reset() samples a seed from
    # [start_seed, start_seed+num_scenarios). Default 1 = the single fixed map we trained on before.
    metadrive_num_scenarios: int = 1         # size of the TRAIN map pool (1 = single fixed map)
    metadrive_eval_scenarios: int = 50       # size of the held-out EVAL pool (disjoint from train)
    metadrive_start_seed: int = 0            # first map seed

    # --- donkey (DonkeyGym Unity sim; image mode) -- see docs/DONKEYCAR.md ---
    donkey_level: int = 3        # 0=roads 1=warehouse 2=avc-sparkfun 3=generated-track
    donkey_exe_path: str = None  # path to donkey_sim.exe; None -> $DONKEY_SIM_PATH or manual launch
    donkey_throttle: float = 1.0 # our throttle +1 maps to this Donkey throttle (Donkey max is 5.0)

    # --- world model ---
    deter_dim: int = 256         # h_t, the deterministic recurrent state
    stoch_dim: int = 32          # z_t, the stochastic latent
    hidden_dim: int = 256        # width of the MLP heads (encoder/decoder/prior/posterior)
    min_std: float = 0.1         # std floor for Gaussian prior/posterior (softplus + min_std)
    dynamics: str = "rssm"       # "rssm" | "transformer" | "mamba"  <- the axis to ablate
    encoder: str = "cnn"         # "cnn" | "vit"  (only matters for obs_type == "image")

    # --- training ---
    seq_len: int = 50
    batch_size: int = 16
    lr: float = 3e-4
    kl_scale: float = 1.0
    free_bits: float = 1.0       # see CONCEPTS.md: guards against posterior collapse
    imagine_horizon: int = 15

    # --- behavior (actor-critic in imagination) ---
    gamma: float = 0.99          # discount factor for returns
    lambda_: float = 0.95        # lambda for lambda-returns (bias/variance trade-off)
    entropy_coef: float = 1e-3   # actor entropy bonus (exploration / anti-collapse)
    actor_lr: float = 8e-5
    critic_lr: float = 8e-5

    # --- gesture control / driving feedback (see docs/superpowers/specs/2026-06-15-...) ---
    webcam_id: int = 0
    gesture_smoothing: float = 0.7    # EMA weight on the previous action (anti-jitter)
    gesture_deadzone: float = 0.1     # zero out |signal| below this
    forecast_horizon: int = 15        # steps the safety metric imagines your action forward
    risk_threshold: float = 0.5       # predicted survival below this -> safety alert
    # --- performance (weak laptop / no GPU): cut what the live drive loop asks for ---
    gesture_feedback_every: int = 3   # recompute the EXPENSIVE safety forecast every k live frames
                                      # (HUD reuses it between); 1 = every frame. Raise on a slow CPU.
    gesture_cap_width: int = 640      # cap webcam capture width (fewer pixels for MediaPipe); 0 = default
    gesture_cap_height: int = 480     # cap webcam capture height; 0 = camera default
    # discrete mode: hand x-position=steer, fist=go forward, open palm=coast/stop,
    # two hands together (prayer)=reverse -- steer + throttle combine (forward + turn at once)
    gesture_mode: str = "continuous"  # "continuous" (hand position) | "discrete" (position+pose)
    gesture_mirror: bool = True       # mirror the webcam frame so steering feels like a mirror
    gesture_steer_mag: float = 0.8    # max steer at full left/right hand deflection
    gesture_throttle_mag: float = 0.5 # throttle for a closed-fist "go forward"
    gesture_reverse_mag: float = 0.4  # reverse throttle for the two-hands "prayer" sign
    gesture_steer_sign: float = 1.0   # set to -1.0 if left/right come out reversed for you
    gesture_prayer_thresh: float = 0.15  # how close two hands must be (normalized) to mean "reverse"
    gesture_backward_dy: float = 0.06 # (legacy: down-swipe reverse, used by classify_gesture)

    # --- infra ---
    device: str = "cpu"          # "cpu" for state runs; "cuda" for image runs on Kaggle
    seed: int = 0
    buffer_capacity: int = 100_000
    log_dir: str = "runs"


def get_config(**overrides) -> Config:
    cfg = Config()
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg

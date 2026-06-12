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
    env: str = "dummy"           # "dummy" | "metadrive"
    action_dim: int = 2          # [steer, throttle]
    max_episode_steps: int = 1000

    # --- world model ---
    deter_dim: int = 256         # h_t, the deterministic recurrent state
    stoch_dim: int = 32          # z_t, the stochastic latent
    dynamics: str = "rssm"       # "rssm" | "transformer" | "mamba"  <- the axis to ablate
    encoder: str = "cnn"         # "cnn" | "vit"  (only matters for obs_type == "image")

    # --- training ---
    seq_len: int = 50
    batch_size: int = 16
    lr: float = 3e-4
    kl_scale: float = 1.0
    free_bits: float = 1.0       # see CONCEPTS.md: guards against posterior collapse
    imagine_horizon: int = 15

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

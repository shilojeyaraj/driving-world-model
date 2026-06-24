"""DAgger entry point: train a learned driving policy that actually recovers, by aggregating
learner-visited states relabeled by the IDM expert. See training/dagger.py and
docs/superpowers/specs/2026-06-24-dagger-design.md.

All knobs are CLI flags (see --help). Output -> runs/dagger/ckpt.pt (standard checkpoint), so:
    python -m scripts.eval_driving runs/dagger/ckpt.pt          # route/success on HELD-OUT maps
    python -m scripts.watch_metadrive_3d runs/dagger/ckpt.pt    # watch it drive

Usage:  python -m scripts.dagger                                       # defaults: 3 iters, 50-map pool
        python -m scripts.dagger --iters 5 --rollout-steps 3000        # more DAgger rounds
        python -m scripts.dagger --eval-episodes 0                     # skip per-iter held-out eval (faster)
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import get_config
from training.dagger import dagger_train


def build_cfg(num_scenarios=50, road_map=None, traffic_density=0.1):
    """Training cfg for DAgger. Extracted so the map-pool wiring is unit-testable without the sim.
    road_map None -> 3 random blocks (varied geometry per seed); digit string -> that many blocks;
    letter string (e.g. "SSSS") -> a fixed scene. REAL terminations (endless OFF) so the learner's
    drift actually ends the episode -- that's the recovery signal DAgger collects. Trains on seeds
    [0, num_scenarios); eval_driving derives the disjoint held-out range from this."""
    if isinstance(road_map, str) and road_map.isdigit():
        road_map = int(road_map)
    if road_map is None:
        road_map = 3
    return get_config(env="metadrive", obs_type="state", state_dim=259, action_dim=2,
                      deter_dim=128, stoch_dim=32, hidden_dim=128, seq_len=10,
                      gamma=0.99, lambda_=0.95, lr=3e-4, actor_lr=3e-4, critic_lr=3e-4,
                      batch_size=16, max_episode_steps=200, metadrive_endless=False,
                      metadrive_map=road_map, metadrive_traffic_density=traffic_density,
                      metadrive_num_scenarios=int(num_scenarios), metadrive_start_seed=0)


def parse_args(argv):
    p = argparse.ArgumentParser(prog="dagger", description="DAgger imitation learning on MetaDrive.")
    p.add_argument("--map", dest="road_map", default=None,
                   help='scene: int N (random blocks) or letters e.g. "SSSS"; default 3 blocks')
    p.add_argument("--num-scenarios", dest="num_scenarios", type=int, default=50,
                   help="size of the TRAIN map pool")
    p.add_argument("--iters", type=int, default=3, help="DAgger rounds after the initial BC")
    p.add_argument("--collect", dest="collect_steps", type=int, default=4000,
                   help="iter-0 IDM expert steps")
    p.add_argument("--rollout-steps", dest="rollout_steps", type=int, default=2000,
                   help="learner-driven, IDM-relabeled steps per DAgger round")
    p.add_argument("--wm-steps", dest="wm_steps", type=int, default=1000, help="world-model grad steps/iter")
    p.add_argument("--bc-steps", dest="bc_steps", type=int, default=1000, help="behavior-clone grad steps/iter")
    p.add_argument("--critic-steps", dest="critic_steps", type=int, default=1000, help="critic grad steps/iter")
    p.add_argument("--eval-episodes", dest="eval_episodes", type=int, default=5,
                   help="held-out eval episodes per iter (0 = skip, faster)")
    p.add_argument("--out", default="runs/dagger/ckpt.pt", help="checkpoint path")
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args([] if argv is None else argv)
    cfg = build_cfg(a.num_scenarios, a.road_map)
    print(f"DAgger: {a.iters} rounds on {a.num_scenarios} maps (seeds 0-{a.num_scenarios - 1}), "
          f"map={cfg.metadrive_map} blocks, rollout {a.rollout_steps}/iter -> {a.out}", flush=True)
    dagger_train(cfg, iters=a.iters, collect_steps=a.collect_steps, rollout_steps=a.rollout_steps,
                 wm_steps=a.wm_steps, bc_steps=a.bc_steps, critic_steps=a.critic_steps,
                 out=a.out, eval_episodes=a.eval_episodes)


if __name__ == "__main__":
    main(sys.argv[1:])

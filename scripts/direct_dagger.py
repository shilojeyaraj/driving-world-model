"""Direct-policy DAgger entry point: iterative imitation learning that rolls out the current
learned DirectPolicy (no world model) to collect failure states, relabels with IDM, and retrains.
Each DAgger round the policy sees its own mistakes and learns to fix them.

Saves runs/direct_dagger/policy.pt (+ policy_iterN.pt every round, policy_best.pt best held-out).
The output is a DirectPolicy checkpoint, so:
    python -m scripts.eval_by_scene runs/direct_dagger/policy.pt    # per-geometry breakdown
    python -m scripts.watch_direct_bc runs/direct_dagger/policy.pt  # watch it drive
    python -m scripts.plot_progress runs/direct_dagger/progress.csv # learning curve

Usage:  python -m scripts.direct_dagger                              # defaults (3 iters)
        python -m scripts.direct_dagger --iters 5 --rollout-steps 3000
        python -m scripts.direct_dagger --boost-scene O              # extra roundabout data at iter 0
        python -m scripts.direct_dagger --eval-episodes 0            # skip per-iter eval (faster)
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.dagger import build_cfg
from training.direct_dagger import direct_dagger_train


def parse_args(argv):
    p = argparse.ArgumentParser(prog="direct_dagger",
                                description="Direct-policy DAgger on MetaDrive.")
    p.add_argument("--map", dest="road_map", default=None,
                   help='scene: int N (random blocks) or letters e.g. "SSSS"; default 3 blocks')
    p.add_argument("--num-scenarios", dest="num_scenarios", type=int, default=50,
                   help="size of the TRAIN map pool")
    p.add_argument("--iters", type=int, default=3, help="DAgger rounds after the iter-0 BC baseline")
    p.add_argument("--clean", dest="clean_steps", type=int, default=8000,
                   help="clean IDM steps for the iter-0 dataset")
    p.add_argument("--recovery", dest="recovery_steps", type=int, default=8000,
                   help="perturbation-recovery steps for the iter-0 dataset")
    p.add_argument("--rollout-steps", dest="rollout_steps", type=int, default=2000,
                   help="policy-driven, IDM-relabeled steps per DAgger round")
    p.add_argument("--bc-steps", dest="direct_steps", type=int, default=8000,
                   help="gradient steps for retraining the direct policy each round")
    p.add_argument("--perturb-prob", dest="perturb_prob", type=float, default=0.08)
    p.add_argument("--gamma", type=float, default=0.2)
    p.add_argument("--boost-scene", dest="boost_scene", default=None,
                   help="mix in extra clean+recovery data for this geometry at iter 0 (e.g. O)")
    p.add_argument("--boost-steps", dest="boost_steps", type=int, default=4000,
                   help="clean AND recovery steps for --boost-scene")
    p.add_argument("--eval-episodes", dest="eval_episodes", type=int, default=5,
                   help="held-out eval episodes per iter (0 = skip, faster)")
    p.add_argument("--out", default="runs/direct_dagger/policy.pt")
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args([] if argv is None else argv)
    cfg = build_cfg(a.num_scenarios, a.road_map)
    print(f"direct-DAgger: {a.iters} rounds on {a.num_scenarios} maps, "
          f"rollout {a.rollout_steps}/iter, boost={a.boost_scene or 'none'} -> {a.out}", flush=True)
    direct_dagger_train(cfg, iters=a.iters, clean_steps=a.clean_steps,
                        recovery_steps=a.recovery_steps, rollout_steps=a.rollout_steps,
                        direct_steps=a.direct_steps, perturb_prob=a.perturb_prob, gamma=a.gamma,
                        boost_scene=a.boost_scene, boost_steps=a.boost_steps,
                        out=a.out, eval_episodes=a.eval_episodes)


if __name__ == "__main__":
    main(sys.argv[1:])

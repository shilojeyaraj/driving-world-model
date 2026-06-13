"""Load a trained checkpoint and run closed-loop eval (spec §6: closed-loop loads from there).

Usage:  python -m scripts.eval_closed_loop [runs/behavior/ckpt.pt]
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import load_models
from envs.base import make_env
from eval.closed_loop import closed_loop_eval


def main(path="runs/behavior/ckpt.pt", episodes=10):
    cfg, world_model, actor, critic = load_models(path)
    if actor is None:
        raise SystemExit(f"checkpoint {path} has no actor; train behavior first.")
    env = make_env(cfg)
    out = closed_loop_eval(actor, world_model, env, episodes=episodes)
    print({k: round(v, 3) for k, v in out.items()})


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "runs/behavior/ckpt.pt")

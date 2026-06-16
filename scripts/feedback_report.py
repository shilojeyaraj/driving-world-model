"""Offline driving-feedback report (GF5): replay a recorded session through the 3-signal
DrivingFeedback engine against a reference checkpoint, and write a habits report (summary stats
+ labeled events). Pairs with scripts/drive_gesture.py (which saves the session) and
training/train_reference.py (which trains the reference).

Usage:  python -m scripts.feedback_report [session.npz] [reference_ckpt.pt]
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


def main(session="runs/gesture_session.npz", ckpt="runs/reference/ckpt.pt",
         out="runs/feedback_report.json"):
    from utils import load_models
    from eval.feedback import DrivingFeedback

    data = np.load(session)
    cfg, wm, ref_actor, critic = load_models(ckpt)
    fb = DrivingFeedback(wm, ref_actor, critic, cfg)
    for obs, action in zip(data["obs"], data["action"]):
        fb.step(obs, action)
    report = fb.finalize()

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2, default=float)

    print("=== driving feedback ===", flush=True)
    print(f"steps={report['n_steps']}  risk_steps={report['n_risk']}  "
          f"mean_safety={report['mean_survival']:.3f}  mean_value={report['mean_value']:.3f}  "
          f"mean|steer_dev|={report['mean_abs_steer_dev']:.3f}", flush=True)
    print("event counts:", report["event_counts"], flush=True)
    print(f"(full report -> {out})", flush=True)
    return report


if __name__ == "__main__":
    s = sys.argv[1] if len(sys.argv) > 1 else "runs/gesture_session.npz"
    c = sys.argv[2] if len(sys.argv) > 2 else "runs/reference/ckpt.pt"
    main(s, c)

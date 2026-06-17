"""Drive MetaDrive and (optionally) overlay live driving feedback (GF2 + GF5).

Action source is pluggable so the whole pipeline is verifiable WITHOUT a camera:
  policy="gesture"  -> your hand via the webcam (needs cv2 + mediapipe + a camera)
  policy="random" / "forward"  -> headless smoke of the render + feedback + HUD + recording

If a reference checkpoint is given (runs/reference/ckpt.pt from train_reference), the 3-signal
DrivingFeedback runs live and a HUD is drawn on the top-down view. The session (obs/action/...)
is saved so scripts/feedback_report.py can analyze it offline.

Usage:  python -m scripts.drive_gesture                       # gesture, no feedback
        python -m scripts.drive_gesture gesture runs/reference/ckpt.pt   # + live feedback HUD
        python -m scripts.drive_gesture random  runs/reference/ckpt.pt   # headless smoke
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


def _action_source(policy, cfg):
    """Return (get_action(obs)->np.ndarray, close()). Gesture uses the webcam; others are headless."""
    if policy == "gesture":
        from control.gesture import GestureController
        ctrl = GestureController(cfg)
        ctrl.calibrate()
        return (lambda obs: ctrl.get_action()), ctrl.close
    if policy == "forward":
        return (lambda obs: np.array([0.0, 0.4], np.float32)), (lambda: None)
    return (lambda obs: np.random.uniform(-1, 1, cfg.action_dim).astype(np.float32)), (lambda: None)


def _resize(frame, size=420):
    from PIL import Image
    return np.asarray(Image.fromarray(frame).resize((size, size)))


def _draw_hud(frame, fb):
    """Overlay the 3 feedback signals on an (already-resized) top-down frame."""
    from PIL import Image, ImageDraw
    img = Image.fromarray(frame)
    d = ImageDraw.Draw(img)
    surv = float(fb["survival"])
    d.rectangle([10, 10, 210, 28], outline=(0, 0, 0))
    d.rectangle([10, 10, 10 + int(200 * surv), 28],
                fill=(int(255 * (1 - surv)), int(255 * surv), 0))                # red->green
    risk = "  RISK!" if fb["risk"] else ""
    d.text((12, 32), f"safety {surv:.2f}{risk}", fill=(0, 0, 0))
    d.text((12, 48), f"steer dev {fb['d_steer']:+.2f}   throttle dev {fb['d_throttle']:+.2f}", fill=(0, 0, 0))
    d.text((12, 64), f"value {fb['value']:+.2f}", fill=(0, 0, 0))
    return np.asarray(img)


def drive_gesture(policy="gesture", ckpt=None, steps=400, out_gif="runs/drive_gesture.gif",
                  session="runs/gesture_session.npz", show=None):
    import imageio.v2 as imageio
    from envs.metadrive_env import adapt_obs          # pure (no panda3d at import time)

    show = (policy == "gesture") if show is None else show

    feedback = None
    if ckpt:
        from utils import load_models                  # torch only -- no native-DLL clash
        from eval.feedback import DrivingFeedback
        cfg, wm, ref_actor, critic = load_models(ckpt)
        feedback = DrivingFeedback(wm, ref_actor, critic, cfg)
    else:
        from config import get_config
        cfg = get_config(env="metadrive", obs_type="state", state_dim=259, action_dim=2,
                         max_episode_steps=300)

    # WINDOWS LOAD-ORDER FIX (important): build the action source -- which imports MediaPipe and
    # therefore TensorFlow's native DLLs -- and open the cv2 window BEFORE loading MetaDrive /
    # panda3d. The reverse order makes TF's DLL init fail ("DLL load failed while importing
    # _pywrap_tensorflow_internal"). Verified: metadrive-then-mediapipe fails, mediapipe-then-
    # metadrive is fine.
    get_action, close_src = _action_source(policy, cfg)
    cv2 = None
    if show:
        import cv2
        cv2.namedWindow("drive (press q to quit)", cv2.WINDOW_NORMAL)

    from metadrive.envs import MetaDriveEnv             # panda3d loads here, AFTER TensorFlow
    env = MetaDriveEnv(dict(use_render=False, horizon=cfg.max_episode_steps))
    obs = adapt_obs(env.reset()[0], "state")
    frames, rec = [], {"obs": [], "action": [], "reward": [], "done": []}
    try:
        for _ in range(steps):
            action = np.asarray(get_action(obs), dtype=np.float32)
            fb = feedback.step(obs, action) if feedback else None
            raw, r, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)
            for k, v in zip(rec, (obs, action, float(r), done)):
                rec[k].append(v)
            frame = _resize(np.asarray(env.render(mode="topdown", window=False)))
            if fb:
                frame = _draw_hud(frame, fb)
            frames.append(frame)
            if show:
                cv2.imshow("drive (press q to quit)", frame[:, :, ::-1])   # RGB -> BGR for cv2
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            obs = adapt_obs(env.reset()[0], "state") if done else adapt_obs(raw, "state")
    finally:
        close_src()
        env.close()
        if show:
            cv2.destroyAllWindows()

    os.makedirs(os.path.dirname(out_gif) or ".", exist_ok=True)
    imageio.mimsave(out_gif, frames, duration=0.06, loop=0)
    np.savez(session, obs=np.stack(rec["obs"]), action=np.stack(rec["action"]),
             reward=np.asarray(rec["reward"], np.float32), done=np.asarray(rec["done"], np.float32))
    print(f"saved {out_gif} ({len(frames)} frames) and {session}  policy={policy} feedback={bool(feedback)}",
          flush=True)


if __name__ == "__main__":
    pol = sys.argv[1] if len(sys.argv) > 1 else "gesture"
    ck = sys.argv[2] if len(sys.argv) > 2 else None
    drive_gesture(policy=pol, ckpt=ck)

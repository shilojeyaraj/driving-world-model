"""Drive MetaDrive and (optionally) overlay live driving feedback (GF2 + GF5).

Action source is pluggable so the whole pipeline is verifiable WITHOUT a camera:
  policy="gesture"           -> hand POSITION drives (continuous): x=steer, height=throttle
  policy="gesture-discrete"  -> hand COMMANDS: point left/right=turn, fist=go, palm=stop, swipe-down=reverse
  policy="random" / "forward"  -> headless smoke of the render + feedback + HUD + recording

If a reference checkpoint is given (runs/reference/ckpt.pt from train_reference), the 3-signal
DrivingFeedback runs live and a HUD is drawn on the top-down view. The session (obs/action/...)
is saved so scripts/feedback_report.py can analyze it offline.

Usage:  python -m scripts.drive_gesture                                  # continuous gesture, no feedback
        python -m scripts.drive_gesture gesture-discrete runs/reference/ckpt.pt  # point/fist/palm + HUD
        python -m scripts.drive_gesture gesture runs/reference/ckpt.pt   # continuous + live feedback HUD
        python -m scripts.drive_gesture gesture-discrete - SSSS          # discrete commands on a highway
        python -m scripts.drive_gesture random  runs/reference/ckpt.pt   # headless smoke
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


def _action_source(policy, cfg):
    """Return (get_action(obs)->np.ndarray, close(), label()->str). Gesture uses the webcam;
    others are headless. label() reports the discrete command (for the HUD) or ""."""
    if policy in ("gesture", "gesture-discrete"):
        from control.gesture import GestureController
        if policy == "gesture-discrete":
            cfg.gesture_mode = "discrete"
        ctrl = GestureController(cfg)
        ctrl.calibrate()
        return (lambda obs: ctrl.get_action()), ctrl.close, (lambda: getattr(ctrl, "last_command", ""))
    if policy == "forward":
        return (lambda obs: np.array([0.0, 0.4], np.float32)), (lambda: None), (lambda: "")
    return (lambda obs: np.random.uniform(-1, 1, cfg.action_dim).astype(np.float32)), (lambda: None), (lambda: "")


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


def _draw_command(frame, cmd):
    """Top-right label showing the discrete gesture command (forward/left/right/stop/backward)."""
    from PIL import Image, ImageDraw
    img = Image.fromarray(frame)
    d = ImageDraw.Draw(img)
    w = frame.shape[1]
    d.rectangle([w - 132, 8, w - 8, 26], fill=(255, 255, 255), outline=(0, 0, 0))
    d.text((w - 128, 11), f"gesture: {cmd}", fill=(0, 0, 0))
    return np.asarray(img)


def _hud_text(fb, cmd):
    """Build the on-screen text dict for MetaDrive's 3-D render(text=...) HUD."""
    text = {}
    if cmd:
        text["gesture"] = cmd
    if fb:
        text["safety"] = f"{fb['survival']:.2f}" + ("  RISK!" if fb["risk"] else "")
        text["style"] = f"steer {fb['d_steer']:+.2f}  thr {fb['d_throttle']:+.2f}"
        text["value"] = f"{fb['value']:+.2f}"
    return text


def drive_gesture(policy="gesture", ckpt=None, steps=400, out_gif="runs/drive_gesture.gif",
                  session="runs/gesture_session.npz", show=None, road_map=None, traffic_density=0.1,
                  render_3d=None):
    import imageio.v2 as imageio
    from envs.metadrive_env import adapt_obs, metadrive_config   # pure (no panda3d at import time)

    # render_3d -> MetaDrive's rendered 3-D window (the "real" view); else top-down cv2/GIF (headless-ok).
    # Default: live gesture modes -> 3-D window; scripted policies (random/forward) stay headless.
    if render_3d is None:
        render_3d = policy.startswith("gesture")
    if show is None:
        show = policy.startswith("gesture") and not render_3d   # cv2 top-down only when NOT 3-D
    if isinstance(road_map, str) and road_map.isdigit():
        road_map = int(road_map)

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
    if road_map is not None:
        cfg.metadrive_map = road_map
    cfg.metadrive_traffic_density = traffic_density
    cfg.metadrive_render = bool(render_3d)               # -> use_render in metadrive_config

    # WINDOWS LOAD-ORDER FIX (important): build the action source -- which imports MediaPipe and
    # therefore TensorFlow's native DLLs -- and open the cv2 window BEFORE loading MetaDrive /
    # panda3d. The reverse order makes TF's DLL init fail ("DLL load failed while importing
    # _pywrap_tensorflow_internal"). Verified: metadrive-then-mediapipe fails, mediapipe-then-
    # metadrive is fine.
    get_action, close_src, label = _action_source(policy, cfg)
    cv2 = None
    if show:
        import cv2
        cv2.namedWindow("drive (press q to quit)", cv2.WINDOW_NORMAL)

    from metadrive.envs import MetaDriveEnv             # panda3d loads here, AFTER TensorFlow
    md = metadrive_config(cfg)                          # use_render=True iff render_3d
    if not render_3d:
        md["use_render"] = False                        # top-down cv2/GIF path
    env = MetaDriveEnv(md)
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
            cmd = label()
            if render_3d:
                env.render(text=_hud_text(fb, cmd))           # HUD overlay on the 3-D window
            else:
                frame = _resize(np.asarray(env.render(mode="topdown", window=False)))
                if fb:
                    frame = _draw_hud(frame, fb)
                if cmd:
                    frame = _draw_command(frame, cmd)
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

    if frames:                                           # only the top-down path records frames
        os.makedirs(os.path.dirname(out_gif) or ".", exist_ok=True)
        imageio.mimsave(out_gif, frames, duration=0.06, loop=0)
    np.savez(session, obs=np.stack(rec["obs"]), action=np.stack(rec["action"]),
             reward=np.asarray(rec["reward"], np.float32), done=np.asarray(rec["done"], np.float32))
    gif_msg = f"{out_gif} ({len(frames)} frames) and " if frames else ""
    print(f"saved {gif_msg}{session}  policy={policy} render={'3d' if render_3d else 'topdown'} "
          f"feedback={bool(feedback)}", flush=True)


if __name__ == "__main__":
    # flags: "2d" forces top-down/GIF, "3d" forces the rendered window (default for gesture modes)
    a = [x for x in sys.argv[1:] if x not in ("2d", "3d")]
    r3d = True if "3d" in sys.argv[1:] else (False if "2d" in sys.argv[1:] else None)
    pol = a[0] if len(a) > 0 else "gesture"
    ck = a[1] if len(a) > 1 and a[1] not in ("-", "none") else None
    rm = a[2] if len(a) > 2 else None
    drive_gesture(policy=pol, ckpt=ck, road_map=rm, render_3d=r3d)

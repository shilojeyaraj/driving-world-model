"""Gesture control (GF1): turn webcam hand tracking into a [steer, throttle] action -- another
ACTION SOURCE feeding env.step (like random_policy / the Actor), so it never touches envs/base.py.

OK TO USE A LIBRARY / modify freely.   Needs cv2 + mediapipe for live use (both optional; the
mapping below is pure and dep-free). Uses MediaPipe's pretrained HandLandmarker (Tasks API; the
legacy `mediapipe.solutions` API is gone in recent versions) -- the model file is downloaded once.
See docs/superpowers/specs/2026-06-15-gesture-feedback-design.md.

The hard, fragile part of any sensor->action layer is the MAPPING (range, deadzone, smoothing),
so that is a PURE function unit-tested without a camera -- mirrors envs/metadrive_env.py:adapt_obs.
"""
import os
import urllib.request

import numpy as np

_HAND_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
                   "hand_landmarker/float16/1/hand_landmarker.task")


def landmarks_to_action(steer_raw, throttle_raw, prev=None, *, deadzone=0.1, smoothing=0.7):
    """Map raw gesture signals in ~[-1,1] (steer_raw: hand left/right; throttle_raw: hand
    up/openness) to a clean [steer, throttle] action in [-1,1].

    deadzone zeros out small jitter; smoothing is an EMA toward the previous action (gestures are
    noisy -- without this the car twitches); the result is clipped to the env's action range."""
    a = np.array([steer_raw, throttle_raw], dtype=np.float32)
    a = np.where(np.abs(a) < deadzone, 0.0, a)               # deadzone
    if prev is not None:
        prev = np.asarray(prev, dtype=np.float32)
        a = smoothing * prev + (1.0 - smoothing) * a          # EMA smoothing
    return np.clip(a, -1.0, 1.0).astype(np.float32)


# ----- discrete gesture mode (GF1b): pointing/fist/palm/swipe -> a driving COMMAND -----
# All of this is PURE (operates on a (21,2) landmark array) so it's unit-tested with no camera.
# MediaPipe hand-landmark indices: wrist=0; per finger (mcp,pip,tip): thumb 1/2/4,
# index 5/6/8, middle 9/10/12, ring 13/14/16, pinky 17/18/20.

def hand_center(pts):
    """Mean (x, y) of the 21 landmarks, in the frame's normalized [0,1] coordinates."""
    return np.asarray(pts, dtype=float)[:, :2].mean(axis=0)


def extended_fingers(pts, margin=1.10):
    """[thumb, index, middle, ring, pinky] booleans: a finger is 'extended' when its tip is
    farther from the wrist than its pip joint (orientation-agnostic, so it works for a hand held
    upright or sideways)."""
    pts = np.asarray(pts, dtype=float)
    wrist = pts[0]
    tips, pips = [4, 8, 12, 16, 20], [2, 6, 10, 14, 18]
    out = [np.linalg.norm(pts[t] - wrist) > np.linalg.norm(pts[p] - wrist) * margin
           for t, p in zip(tips, pips)]
    return np.array(out, dtype=bool)


def classify_gesture(pts, prev_center=None, *, backward_dy=0.06, point_dead=0.03):
    """Map one hand pose (+ its motion since last frame) to a driving COMMAND:
      closed fist          -> "forward"      open palm     -> "stop"
      index pointing left  -> "left"         pointing right-> "right"
      hand swiped downward -> "backward"     anything else -> "none"
    `prev_center` is last frame's hand_center; a downward jump > backward_dy is the reverse swipe."""
    pts = np.asarray(pts, dtype=float)
    if prev_center is not None and (hand_center(pts)[1] - prev_center[1]) > backward_dy:
        return "backward"                                 # deliberate downward motion = reverse
    ext = extended_fingers(pts)
    n_main = int(ext[1:5].sum())                          # index/middle/ring/pinky
    if n_main == 0:
        return "forward"                                  # fist
    if n_main >= 3:
        return "stop"                                     # open palm
    if n_main == 1 and ext[1]:                            # only the index finger -> pointing
        dx = float(pts[8][0] - pts[5][0])                 # index tip vs knuckle (image x)
        if dx > point_dead:
            return "right"
        if dx < -point_dead:
            return "left"
    return "none"                                         # pointing up / 2 fingers / unclear


def command_to_action(command, prev=None, *, steer_mag=0.6, throttle_mag=0.5, reverse_mag=0.4,
                      steer_sign=1.0, smoothing=0.5, straighten=0.5):
    """Turn a discrete COMMAND into a smoothed [steer, throttle] action. It's a small state
    machine: steering and throttle are separate axes, and a command updates one while HOLDING the
    other (so you fist to go, then point to steer while still moving). EMA-smoothed like the
    continuous mapping. `steer_sign` flips left/right if they come out reversed on your setup."""
    prev = np.zeros(2, dtype=np.float32) if prev is None else np.asarray(prev, dtype=np.float32)
    s, t = float(prev[0]), float(prev[1])
    if command == "left":
        s_t, t_t = -steer_mag * steer_sign, t
    elif command == "right":
        s_t, t_t = steer_mag * steer_sign, t
    elif command == "forward":
        s_t, t_t = s, throttle_mag
    elif command == "backward":
        s_t, t_t = s, -reverse_mag
    elif command == "stop":
        s_t, t_t = 0.0, 0.0
    else:                                                 # "none": coast, straighten the wheel
        s_t, t_t = s * (1.0 - straighten), t
    target = np.array([s_t, t_t], dtype=np.float32)
    return np.clip(smoothing * prev + (1.0 - smoothing) * target, -1.0, 1.0).astype(np.float32)


def _ensure_hand_model(path):
    """Download MediaPipe's pretrained hand_landmarker.task once (cached at `path`)."""
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        print(f"downloading hand landmark model -> {path} ...", flush=True)
        urllib.request.urlretrieve(_HAND_MODEL_URL, path)
    return path


class GestureController:
    """Live webcam controller. Reads one hand with MediaPipe's pretrained HandLandmarker and
    returns a smoothed [steer, throttle] action in [-1,1]. Two modes (cfg.gesture_mode):
      "continuous" -> steer from the hand's x position, throttle from its height.
      "discrete"   -> point left/right = turn, closed fist = go, open palm = stop,
                      downward swipe = reverse  (classify_gesture + command_to_action).

    Opens the webcam FIRST (fail fast if absent) before importing MediaPipe / downloading the model.
    On Windows, build this controller BEFORE creating a MetaDrive env -- importing MediaPipe pulls
    in TensorFlow, whose native DLL fails to load after panda3d (see scripts/drive_gesture.py)."""

    def __init__(self, cfg):
        import cv2
        self._cv2 = cv2
        self._cap = cv2.VideoCapture(cfg.webcam_id)
        if not self._cap.isOpened():
            raise RuntimeError(f"could not open webcam {cfg.webcam_id}")

        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode
        self._mp = mp
        model = _ensure_hand_model(os.path.join(cfg.log_dir, "hand_landmarker.task"))
        opts = HandLandmarkerOptions(base_options=BaseOptions(model_asset_path=model),
                                     num_hands=1, running_mode=RunningMode.IMAGE)
        self._landmarker = HandLandmarker.create_from_options(opts)
        self.cfg = cfg
        self.mode = getattr(cfg, "gesture_mode", "continuous")
        self._prev = None
        self._prev_center = None
        self.last_command = "none"            # for HUDs: the discrete command read this frame

    def _detect(self, frame_bgr):
        """One hand's 21 landmarks as an (21,2) array in [0,1], or None if no hand seen."""
        rgb = self._cv2.cvtColor(frame_bgr, self._cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        res = self._landmarker.detect(mp_image)
        if not res.hand_landmarks:
            return None
        return np.array([[p.x, p.y] for p in res.hand_landmarks[0]], dtype=float)

    def get_action(self):
        ok, frame = self._cap.read()
        if not ok:
            return self._prev if self._prev is not None else np.zeros(2, dtype=np.float32)
        if getattr(self.cfg, "gesture_mirror", True):
            frame = self._cv2.flip(frame, 1)              # mirror -> control feels natural
        pts = self._detect(frame)

        if self.mode == "discrete":
            cmd = "none" if pts is None else classify_gesture(
                pts, self._prev_center, backward_dy=getattr(self.cfg, "gesture_backward_dy", 0.06))
            self.last_command = cmd
            self._prev_center = None if pts is None else hand_center(pts)
            self._prev = command_to_action(
                cmd, prev=self._prev, steer_mag=getattr(self.cfg, "gesture_steer_mag", 0.6),
                throttle_mag=getattr(self.cfg, "gesture_throttle_mag", 0.5),
                reverse_mag=getattr(self.cfg, "gesture_reverse_mag", 0.4),
                steer_sign=getattr(self.cfg, "gesture_steer_sign", 1.0),
                smoothing=self.cfg.gesture_smoothing)
            return self._prev

        # continuous: hand-center x -> steer, hand height -> throttle
        if pts is None:
            steer_raw, throttle_raw = 0.0, 0.0
        else:
            cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
            steer_raw, throttle_raw = 2.0 * cx - 1.0, 1.0 - 2.0 * cy
        self._prev = landmarks_to_action(
            steer_raw, throttle_raw, prev=self._prev,
            deadzone=self.cfg.gesture_deadzone, smoothing=self.cfg.gesture_smoothing)
        return self._prev

    def calibrate(self, frames=30):
        """Warm up the camera (drop the first few frames)."""
        for _ in range(frames):
            self._cap.read()
        self._prev = None
        self._prev_center = None

    def close(self):
        if getattr(self, "_cap", None) is not None:
            self._cap.release()
        if getattr(self, "_landmarker", None) is not None:
            self._landmarker.close()

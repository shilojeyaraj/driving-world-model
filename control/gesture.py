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


# ----- chosen scheme (GF1c): position steers + pose throttles -> forward + turn AT THE SAME TIME -----
# Steering comes from the hand's x-position (continuous), throttle from its pose, so the two axes are
# independent and simultaneous. Reverse = a two-hands-together "prayer" sign. All pure / camera-free.

def hands_together(hand_a, hand_b, thresh=0.15):
    """True when two hands' centers are within `thresh` (normalized) -- the 'prayer' / reverse sign."""
    return bool(np.linalg.norm(hand_center(hand_a) - hand_center(hand_b)) < thresh)


def throttle_command(hands, *, prayer_thresh=0.15):
    """Throttle intent from the visible hands (list of (21,2) arrays):
      two hands together -> "reverse",  one closed fist -> "forward",  else -> "coast"
    (open palm / relaxed / no hand all release the throttle)."""
    if len(hands) >= 2 and hands_together(hands[0], hands[1], prayer_thresh):
        return "reverse"
    if len(hands) >= 1:
        return "forward" if int(extended_fingers(hands[0])[1:5].sum()) == 0 else "coast"
    return "coast"


def steer_from_position(pts, *, deadzone=0.1):
    """Steering in [-1,1] from the hand's horizontal position (left of frame -> -1, right -> +1)."""
    raw = 2.0 * float(np.asarray(pts, dtype=float)[:, 0].mean()) - 1.0
    return 0.0 if abs(raw) < deadzone else float(np.clip(raw, -1.0, 1.0))


def position_pose_action(hands, prev=None, *, steer_mag=0.8, throttle_mag=0.5, reverse_mag=0.4,
                         steer_sign=1.0, deadzone=0.1, smoothing=0.5, straighten=0.5,
                         prayer_thresh=0.15):
    """Hand x-position sets STEER, hand pose sets THROTTLE -- the two combine, so a fist held to the
    right drives forward AND right at once. `hands` = list of (21,2) arrays (0/1/2 hands). EMA-
    smoothed like the other mappings; with no hand the wheel straightens and throttle releases.
    Returns (action[steer,throttle] in [-1,1], command-str for the HUD)."""
    prev = np.zeros(2, dtype=np.float32) if prev is None else np.asarray(prev, dtype=np.float32)
    cmd = throttle_command(hands, prayer_thresh=prayer_thresh)
    if hands:
        s_t = steer_sign * steer_mag * steer_from_position(hands[0], deadzone=deadzone)
    else:
        s_t = float(prev[0]) * (1.0 - straighten)         # no hand seen -> straighten the wheel
    t_t = {"forward": throttle_mag, "reverse": -reverse_mag, "coast": 0.0}[cmd]
    target = np.array([s_t, t_t], dtype=np.float32)
    action = np.clip(smoothing * prev + (1.0 - smoothing) * target, -1.0, 1.0).astype(np.float32)
    return action, cmd


def _ensure_hand_model(path):
    """Download MediaPipe's pretrained hand_landmarker.task once (cached at `path`)."""
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        print(f"downloading hand landmark model -> {path} ...", flush=True)
        urllib.request.urlretrieve(_HAND_MODEL_URL, path)
    return path


class GestureController:
    """Live webcam controller. Tracks up to TWO hands with MediaPipe's pretrained HandLandmarker
    and returns a smoothed [steer, throttle] action in [-1,1]. Two modes (cfg.gesture_mode):
      "continuous" -> steer from the hand's x position, throttle from its height.
      "discrete"   -> position steers + pose throttles (position_pose_action): hand x-position =
                      steer, closed fist = go forward, open palm = coast/stop, two hands together
                      (prayer) = reverse. Steer + throttle are independent, so forward + turn happen
                      at the same time.

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
                                     num_hands=2, running_mode=RunningMode.IMAGE)  # 2 -> prayer/reverse
        self._landmarker = HandLandmarker.create_from_options(opts)
        self.cfg = cfg
        self.mode = getattr(cfg, "gesture_mode", "continuous")
        self._prev = None
        self.last_command = "none"            # for HUDs: the command read this frame

    def _detect_hands(self, frame_bgr):
        """All detected hands (up to 2) as a list of (21,2) arrays in [0,1]; [] if none."""
        rgb = self._cv2.cvtColor(frame_bgr, self._cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        res = self._landmarker.detect(mp_image)
        return [np.array([[p.x, p.y] for p in h], dtype=float) for h in res.hand_landmarks]

    def get_action(self):
        ok, frame = self._cap.read()
        if not ok:
            return self._prev if self._prev is not None else np.zeros(2, dtype=np.float32)
        if getattr(self.cfg, "gesture_mirror", True):
            frame = self._cv2.flip(frame, 1)              # mirror -> control feels natural
        hands = self._detect_hands(frame)

        if self.mode == "discrete":
            self._prev, cmd = position_pose_action(
                hands, prev=self._prev, steer_mag=getattr(self.cfg, "gesture_steer_mag", 0.8),
                throttle_mag=getattr(self.cfg, "gesture_throttle_mag", 0.5),
                reverse_mag=getattr(self.cfg, "gesture_reverse_mag", 0.4),
                steer_sign=getattr(self.cfg, "gesture_steer_sign", 1.0),
                deadzone=self.cfg.gesture_deadzone, smoothing=self.cfg.gesture_smoothing,
                prayer_thresh=getattr(self.cfg, "gesture_prayer_thresh", 0.15))
            self.last_command = cmd
            return self._prev

        # continuous: first hand's center x -> steer, height -> throttle
        if not hands:
            steer_raw, throttle_raw = 0.0, 0.0
        else:
            cx, cy = hands[0][:, 0].mean(), hands[0][:, 1].mean()
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

    def close(self):
        if getattr(self, "_cap", None) is not None:
            self._cap.release()
        if getattr(self, "_landmarker", None) is not None:
            self._landmarker.close()

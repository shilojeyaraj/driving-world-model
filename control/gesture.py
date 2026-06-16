"""Gesture control (GF1): turn webcam hand tracking into a [steer, throttle] action -- another
ACTION SOURCE feeding env.step (like random_policy / the Actor), so it never touches envs/base.py.

OK TO USE A LIBRARY / modify freely.   Needs cv2 + mediapipe for live use (both optional; the
mapping below is pure and dep-free). See docs/superpowers/specs/2026-06-15-gesture-feedback-design.md.

The hard, fragile part of any sensor->action layer is the MAPPING (range, deadzone, smoothing),
so that is a PURE function unit-tested without a camera -- mirrors envs/metadrive_env.py:adapt_obs.
"""
import numpy as np


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


class GestureController:
    """Live webcam controller. Reads one hand with MediaPipe, derives steer from the hand's
    horizontal position and throttle from its height, and returns a smoothed action.

    Imports cv2 + mediapipe lazily so this module imports fine without them (the dummy/state
    pipeline never needs a camera). Action convention matches the env: [steer, throttle] in [-1,1]."""

    def __init__(self, cfg):
        import cv2
        import mediapipe as mp
        self.cfg = cfg
        self._cap = cv2.VideoCapture(cfg.webcam_id)
        if not self._cap.isOpened():
            raise RuntimeError(f"could not open webcam {cfg.webcam_id}")
        self._hands = mp.solutions.hands.Hands(max_num_hands=1, min_detection_confidence=0.5)
        self._cv2 = cv2
        self._prev = None

    def _signals(self, frame):
        """Extract (steer_raw, throttle_raw) in ~[-1,1] from one BGR frame, or (0,0) if no hand.
        steer = hand-center x mapped left/right; throttle = hand height (higher hand = accelerate)."""
        rgb = self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGB)
        res = self._hands.process(rgb)
        if not res.multi_hand_landmarks:
            return 0.0, 0.0
        lm = res.multi_hand_landmarks[0].landmark
        xs = [p.x for p in lm]; ys = [p.y for p in lm]
        cx = float(np.mean(xs))                               # normalized [0,1], 0=left, 1=right
        cy = float(np.mean(ys))                               # normalized [0,1], 0=top, 1=bottom
        steer_raw = 2.0 * cx - 1.0                            # left hand -> steer left (-)
        throttle_raw = 1.0 - 2.0 * cy                         # higher hand (small y) -> accelerate
        return steer_raw, throttle_raw

    def get_action(self):
        ok, frame = self._cap.read()
        if not ok:
            return self._prev if self._prev is not None else np.zeros(2, dtype=np.float32)
        steer_raw, throttle_raw = self._signals(frame)
        self._prev = landmarks_to_action(
            steer_raw, throttle_raw, prev=self._prev,
            deadzone=self.cfg.gesture_deadzone, smoothing=self.cfg.gesture_smoothing)
        return self._prev

    def calibrate(self, frames=30):
        """Warm up the camera/detector (drop the first few frames). Returns nothing."""
        for _ in range(frames):
            self._cap.read()
        self._prev = None

    def close(self):
        if getattr(self, "_cap", None) is not None:
            self._cap.release()
        if getattr(self, "_hands", None) is not None:
            self._hands.close()

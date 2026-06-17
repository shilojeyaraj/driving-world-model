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


def _ensure_hand_model(path):
    """Download MediaPipe's pretrained hand_landmarker.task once (cached at `path`)."""
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        print(f"downloading hand landmark model -> {path} ...", flush=True)
        urllib.request.urlretrieve(_HAND_MODEL_URL, path)
    return path


class GestureController:
    """Live webcam controller. Reads one hand with MediaPipe's pretrained HandLandmarker, derives
    steer from the hand's horizontal position and throttle from its height, and returns a smoothed
    action. Action convention matches the env: [steer, throttle] in [-1,1].

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
        self._prev = None

    def _signals(self, frame_bgr):
        """(steer_raw, throttle_raw) in ~[-1,1] from one BGR frame, or (0,0) if no hand.
        steer = hand-center x (left/right); throttle = hand height (higher hand = accelerate)."""
        rgb = self._cv2.cvtColor(frame_bgr, self._cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        res = self._landmarker.detect(mp_image)
        if not res.hand_landmarks:
            return 0.0, 0.0
        lm = res.hand_landmarks[0]                            # 21 NormalizedLandmark (x,y in [0,1])
        cx = sum(p.x for p in lm) / len(lm)
        cy = sum(p.y for p in lm) / len(lm)
        return 2.0 * cx - 1.0, 1.0 - 2.0 * cy

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
        """Warm up the camera (drop the first few frames)."""
        for _ in range(frames):
            self._cap.read()
        self._prev = None

    def close(self):
        if getattr(self, "_cap", None) is not None:
            self._cap.release()
        if getattr(self, "_landmarker", None) is not None:
            self._landmarker.close()

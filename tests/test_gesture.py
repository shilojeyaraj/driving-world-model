"""Gesture -> action mapping (GF1). The mapping is a PURE function (deadzone + EMA smoothing +
clip), testable with synthetic numbers and NO webcam -- the cv2/MediaPipe capture is exercised
only by a live smoke test that skips where the deps/camera are absent (mirrors
tests/test_metadrive_adapter.py)."""
import numpy as np
import pytest

from control.gesture import landmarks_to_action


def test_action_in_range_and_shape():
    a = landmarks_to_action(0.4, -0.9, deadzone=0.0, smoothing=0.0)
    assert a.shape == (2,) and a.dtype == np.float32
    assert a.min() >= -1.0 and a.max() <= 1.0


def test_clips_out_of_range_signals():
    a = landmarks_to_action(5.0, -5.0, deadzone=0.0, smoothing=0.0)
    assert np.allclose(a, [1.0, -1.0])


def test_deadzone_zeros_small_signals():
    a = landmarks_to_action(0.05, -0.05, deadzone=0.1, smoothing=0.0)
    assert np.allclose(a, [0.0, 0.0])


def test_steer_is_monotonic_in_signal():
    left = landmarks_to_action(-0.6, 0.0, deadzone=0.0, smoothing=0.0)[0]
    right = landmarks_to_action(0.6, 0.0, deadzone=0.0, smoothing=0.0)[0]
    assert right > left                                  # more-right signal -> larger steer


def test_smoothing_blends_toward_previous():
    prev = np.array([0.0, 0.0], dtype=np.float32)
    a = landmarks_to_action(1.0, 1.0, prev=prev, deadzone=0.0, smoothing=0.8)
    # 0.8*prev + 0.2*new = 0.2
    assert np.allclose(a, [0.2, 0.2], atol=1e-5)


@pytest.mark.slow
def test_live_controller_smoke():
    """Runs only where cv2 + mediapipe (and a camera) exist; otherwise skipped. Marked slow
    because importing mediapipe is heavy -- keeps the fast loop fast."""
    pytest.importorskip("cv2")
    pytest.importorskip("mediapipe")
    from config import get_config
    from control.gesture import GestureController
    try:
        ctrl = GestureController(get_config())
    except Exception as e:                               # no camera in CI/headless
        pytest.skip(f"no webcam available: {type(e).__name__}")
    try:
        a = ctrl.get_action()
        assert a.shape == (2,) and a.min() >= -1.0 and a.max() <= 1.0
    finally:
        ctrl.close()

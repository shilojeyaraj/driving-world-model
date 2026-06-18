"""Gesture -> action mapping (GF1). The mapping is a PURE function (deadzone + EMA smoothing +
clip), testable with synthetic numbers and NO webcam -- the cv2/MediaPipe capture is exercised
only by a live smoke test that skips where the deps/camera are absent (mirrors
tests/test_metadrive_adapter.py)."""
import numpy as np
import pytest

from control.gesture import (landmarks_to_action, extended_fingers, hand_center,
                             classify_gesture, command_to_action,
                             hands_together, throttle_command, steer_from_position,
                             position_pose_action)


# --- synthetic hand builder (no camera): 21 MediaPipe landmarks as an (21,2) array ---
# Finger i extended => its tip sits far from the wrist; folded => tip near the pip joint.
def _hand(ext, index_dx=0.0, center=None):
    """ext = [thumb, index, middle, ring, pinky] booleans. index_dx sets the index-finger
    pointing direction (>0 right, <0 left in image x). center optionally re-positions the hand."""
    pts = np.zeros((21, 2), dtype=float)
    wrist = np.array([0.5, 0.9])
    pts[0] = wrist
    groups = {0: (1, 2, 4), 1: (5, 6, 8), 2: (9, 10, 12), 3: (13, 14, 16), 4: (17, 18, 20)}
    for fi, (mcp, pip, tip) in groups.items():
        pts[mcp] = wrist + [0.0, -0.10]
        pts[pip] = wrist + [0.0, -0.20]
        pts[tip] = wrist + [0.0, (-0.40 if ext[fi] else -0.12)]
    if ext[1]:                                            # index: encode pointing direction
        pts[5] = [0.5, 0.5]
        pts[8] = [0.5 + index_dx, 0.2]
    if center is not None:
        pts = pts - pts.mean(0) + np.asarray(center, dtype=float)
    return pts


def test_extended_fingers_open_vs_fist():
    assert extended_fingers(_hand([True] * 5))[1:].all()          # open palm: 4 main fingers up
    assert not extended_fingers(_hand([False] * 5))[1:].any()     # fist: none up


def test_classify_open_palm_is_stop():
    assert classify_gesture(_hand([True] * 5)) == "stop"


def test_classify_fist_is_forward():
    assert classify_gesture(_hand([False] * 5)) == "forward"


def test_classify_pointing_left_and_right():
    assert classify_gesture(_hand([False, True, False, False, False], index_dx=+0.2)) == "right"
    assert classify_gesture(_hand([False, True, False, False, False], index_dx=-0.2)) == "left"


def test_classify_pointing_straight_up_is_none():
    assert classify_gesture(_hand([False, True, False, False, False], index_dx=0.0)) == "none"


def test_classify_ambiguous_two_fingers_is_none():
    assert classify_gesture(_hand([False, True, True, False, False])) == "none"


def test_classify_backward_on_downward_motion():
    pts = _hand([False] * 5, center=(0.5, 0.5))           # would be "forward" if still
    prev = hand_center(_hand([False] * 5, center=(0.5, 0.38)))   # hand was higher last frame
    assert classify_gesture(pts, prev_center=prev, backward_dy=0.06) == "backward"
    # tiny motion does NOT trigger reverse:
    near = hand_center(_hand([False] * 5, center=(0.5, 0.49)))
    assert classify_gesture(pts, prev_center=near, backward_dy=0.06) == "forward"


def test_command_to_action_left_right_throttle():
    fwd = command_to_action("forward", prev=[0.3, 0.0], smoothing=0.0, throttle_mag=0.5)
    assert np.allclose(fwd, [0.3, 0.5])                   # throttle set, steer held
    left = command_to_action("left", prev=[0.0, 0.4], smoothing=0.0, steer_mag=0.6)
    assert np.allclose(left, [-0.6, 0.4])                 # steer set, throttle held
    right = command_to_action("right", prev=[0.0, 0.4], smoothing=0.0, steer_mag=0.6)
    assert np.allclose(right, [0.6, 0.4])
    back = command_to_action("backward", prev=[0.2, 0.5], smoothing=0.0, reverse_mag=0.4)
    assert np.allclose(back, [0.2, -0.4])                 # reverse throttle, steer held
    stop = command_to_action("stop", prev=[0.5, 0.5], smoothing=0.0)
    assert np.allclose(stop, [0.0, 0.0])


def test_command_to_action_steer_sign_flips_direction():
    r = command_to_action("right", prev=[0, 0], smoothing=0.0, steer_mag=0.6, steer_sign=-1.0)
    assert np.allclose(r, [-0.6, 0.0])                    # sign flip swaps left/right


def test_command_to_action_none_straightens_steer():
    a = command_to_action("none", prev=[0.8, 0.5], smoothing=0.0, straighten=0.5)
    assert np.allclose(a, [0.4, 0.5])                     # steer decays toward 0, throttle held


# --- chosen scheme: position steers + pose throttles (forward+turn at the same time) ---
def test_hands_together_detects_prayer():
    a = _hand([True] * 5, center=(0.50, 0.5))
    b = _hand([True] * 5, center=(0.54, 0.5))
    assert hands_together(a, b, thresh=0.15)
    far = _hand([True] * 5, center=(0.88, 0.5))
    assert not hands_together(a, far, thresh=0.15)


def test_throttle_command_fist_palm_none_prayer():
    fist = _hand([False] * 5, center=(0.5, 0.5))
    palm = _hand([True] * 5, center=(0.5, 0.5))
    assert throttle_command([fist]) == "forward"          # closed fist -> go
    assert throttle_command([palm]) == "coast"            # open palm -> coast/stop
    assert throttle_command([]) == "coast"                # no hand -> coast
    a = _hand([True] * 5, center=(0.50, 0.5))
    b = _hand([True] * 5, center=(0.54, 0.5))
    assert throttle_command([a, b]) == "reverse"          # two hands together (prayer) -> reverse


def test_steer_from_position_left_right_center():
    right = steer_from_position(_hand([False] * 5, center=(0.85, 0.5)), deadzone=0.1)
    left = steer_from_position(_hand([False] * 5, center=(0.15, 0.5)), deadzone=0.1)
    center = steer_from_position(_hand([False] * 5, center=(0.50, 0.5)), deadzone=0.1)
    assert right > 0.3 and left < -0.3 and center == 0.0


def test_position_pose_forward_and_right_at_once():
    # closed fist held to the RIGHT -> steer right AND throttle forward simultaneously
    fist_right = _hand([False] * 5, center=(0.85, 0.5))
    act, cmd = position_pose_action([fist_right], prev=[0, 0], smoothing=0.0,
                                    steer_mag=1.0, throttle_mag=0.5)
    assert cmd == "forward" and act[0] > 0.3 and act[1] > 0.3   # BOTH axes active at once


def test_position_pose_prayer_reverses():
    a = _hand([True] * 5, center=(0.50, 0.5))
    b = _hand([True] * 5, center=(0.54, 0.5))
    act, cmd = position_pose_action([a, b], prev=[0, 0.5], smoothing=0.0, reverse_mag=0.4)
    assert cmd == "reverse" and act[1] < 0


def test_position_pose_no_hands_coasts_and_straightens():
    act, cmd = position_pose_action([], prev=[0.8, 0.5], smoothing=0.0, straighten=0.5)
    assert cmd == "coast" and abs(act[0]) < 0.8 and act[1] == 0.0


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

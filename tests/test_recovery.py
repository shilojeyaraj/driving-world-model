"""Recovery-data collection (roadmap A). The triangular steering impulse (Codevilla Eq 3) is pure and
unit-tested here; the perturbed IDM collector that uses it is exercised by an importorskip smoke (it
needs MetaDrive). The impulse pushes the IDM expert off the centerline so the recorded data covers a
tube around the lane, with IDM's clean action as the recovery label.
"""
import numpy as np
import pytest

from training.recovery import triangular_impulse


def test_triangular_impulse_shape():
    # impulse over [t0, t0+tau], peak sigma*gamma at the midpoint, zero at edges and outside
    assert triangular_impulse(t=5.0, t0=5.0, tau=2.0, sigma=1.0, gamma=0.15) == pytest.approx(0.0)   # start
    assert triangular_impulse(6.0, 5.0, 2.0, 1.0, 0.15) == pytest.approx(0.15)                        # peak
    assert triangular_impulse(7.0, 5.0, 2.0, 1.0, 0.15) == pytest.approx(0.0)                         # end
    assert triangular_impulse(4.0, 5.0, 2.0, 1.0, 0.15) == 0.0                                        # before
    assert triangular_impulse(9.0, 5.0, 2.0, 1.0, 0.15) == 0.0                                        # after


def test_triangular_impulse_sign_and_intensity():
    assert triangular_impulse(6.0, 5.0, 2.0, -1.0, 0.15) == pytest.approx(-0.15)   # negative sign
    assert triangular_impulse(6.0, 5.0, 2.0, 1.0, 0.30) == pytest.approx(0.30)     # intensity scales peak


def test_triangular_impulse_degenerate_duration_is_zero():
    assert triangular_impulse(5.0, 5.0, 0.0, 1.0, 0.15) == 0.0     # tau<=0 -> no impulse


def test_collect_idm_perturbed_smoke():
    """Live (needs MetaDrive): the perturbed collector returns a usable buffer with bounded actions
    (the recovery labels are IDM's clean actions, clipped to the action range)."""
    pytest.importorskip("metadrive")
    from scripts.dagger import build_cfg
    from training.recovery import collect_idm_perturbed
    buf = collect_idm_perturbed(build_cfg(num_scenarios=2), steps=60, perturb_prob=0.2)
    assert len(buf) == 60
    if buf.can_sample():
        batch = buf.sample(2)
        assert batch["action"].shape[-1] == 2
        assert np.abs(batch["action"]).max() <= 1.0          # labels stay in the action range

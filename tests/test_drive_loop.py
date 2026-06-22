"""Live-logging helper for the drive loop: turn MetaDrive's info dict into a human-readable
termination reason (why the car reset). Pure -> testable without MetaDrive or a window."""
from scripts.drive_gesture import _termination_reason


def test_termination_reason_picks_the_active_cause():
    assert _termination_reason({"out_of_road": True, "crash_vehicle": False, "max_step": False}) == "out_of_road"
    assert _termination_reason({"crash_vehicle": True, "out_of_road": False}) == "crash_vehicle"
    assert _termination_reason({"max_step": True}) == "max_step"
    assert _termination_reason({"arrive_dest": True}) == "arrive_dest"


def test_termination_reason_defaults_when_unknown():
    assert _termination_reason({}) == "done"
    assert _termination_reason({"out_of_road": False, "crash_vehicle": False}) == "done"

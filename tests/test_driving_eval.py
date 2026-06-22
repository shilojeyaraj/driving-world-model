"""Driving-usability metrics (eval/closed_loop.py): is a trained policy actually usable -- does it
get down the route without crashing? The aggregation is PURE so it's testable without MetaDrive;
the live episode runners (which need MetaDrive) are exercised by scripts/eval_driving.py."""
from eval.closed_loop import episode_outcome, summarize_driving


def test_episode_outcome_classifies_crash_route_and_return():
    rec = episode_outcome({"crash_vehicle": True, "route_completion": 0.4,
                           "arrive_dest": False, "out_of_road": False}, total_return=12.3, steps=87)
    assert rec["crash"] is True            # any crash_* flag -> crash
    assert rec["route_completion"] == 0.4
    assert rec["arrive_dest"] is False
    assert rec["return"] == 12.3 and rec["steps"] == 87


def test_episode_outcome_arrive_dest_is_success_not_crash():
    rec = episode_outcome({"arrive_dest": True, "route_completion": 1.0}, 50.0, 200)
    assert rec["arrive_dest"] is True and rec["crash"] is False and rec["out_of_road"] is False


def test_summarize_driving_aggregates_rates_and_means():
    recs = [
        {"return": 10.0, "route_completion": 1.0, "arrive_dest": True,  "crash": False, "out_of_road": False, "steps": 200},
        {"return": 2.0,  "route_completion": 0.2, "arrive_dest": False, "crash": True,  "out_of_road": False, "steps": 30},
        {"return": 4.0,  "route_completion": 0.5, "arrive_dest": False, "crash": False, "out_of_road": True,  "steps": 60},
        {"return": 8.0,  "route_completion": 1.0, "arrive_dest": True,  "crash": False, "out_of_road": False, "steps": 180},
    ]
    s = summarize_driving(recs)
    assert s["n"] == 4
    assert s["success_rate"] == 0.5        # 2/4 arrived
    assert s["crash_rate"] == 0.25         # 1/4 crashed
    assert s["off_road_rate"] == 0.25      # 1/4 left the road
    assert abs(s["route_completion"] - 0.675) < 1e-6   # mean of route_completion
    assert abs(s["mean_return"] - 6.0) < 1e-6


def test_summarize_driving_empty_is_safe():
    assert summarize_driving([])["n"] == 0

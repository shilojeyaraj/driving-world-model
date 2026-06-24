"""Driving-usability metrics (eval/closed_loop.py): is a trained policy actually usable -- does it
get down the route without crashing? The aggregation is PURE so it's testable without MetaDrive;
the live episode runners (which need MetaDrive) are exercised by scripts/eval_driving.py."""
from eval.closed_loop import episode_outcome, summarize_driving, eval_seeds, summarize_recovery


def test_eval_seeds_are_deterministic_and_in_range():
    """Fixed, in-range map seeds so two eval runs see the SAME maps (comparable, not noisy)."""
    assert eval_seeds(50, 50, 10) == list(range(50, 60))
    assert eval_seeds(50, 3, 5) == [50, 51, 52, 50, 51]      # cycles within the pool
    s = eval_seeds(100, 7, 20)
    assert len(s) == 20 and all(100 <= x < 107 for x in s)   # never leaves [start, start+num)


def test_summarize_recovery_rate():
    """Targeted recovery metric: fraction of perturbed-start episodes that got back (didn't end off-road)."""
    recs = [{"recovered": True, "route_completion": 0.5},
            {"recovered": False, "route_completion": 0.1},
            {"recovered": True, "route_completion": 0.4},
            {"recovered": True, "route_completion": 0.6}]
    s = summarize_recovery(recs)
    assert s["recovery_rate"] == 0.75
    assert s["n"] == 4
    assert abs(s["mean_route"] - 0.4) < 1e-6


def test_summarize_recovery_empty_is_safe():
    assert summarize_recovery([])["n"] == 0


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


def test_summarize_driving_reports_variability():
    """n=5 evals swing ~±40% (Codevilla); the summary must report SPREAD so we can tell a real
    change from noise -- std of route completion and return across episodes."""
    import pytest
    recs = [
        {"return": 10.0, "route_completion": 0.2, "arrive_dest": False, "crash": False, "out_of_road": True, "steps": 100},
        {"return": 30.0, "route_completion": 0.4, "arrive_dest": False, "crash": True,  "out_of_road": True, "steps": 50},
    ]
    s = summarize_driving(recs)
    assert s["route_completion"] == pytest.approx(0.3)
    assert s["route_completion_std"] == pytest.approx(0.1)   # population std of [0.2, 0.4]
    assert s["return_std"] == pytest.approx(10.0)            # population std of [10, 30]


def test_summarize_driving_empty_is_safe():
    s = summarize_driving([])
    assert s["n"] == 0

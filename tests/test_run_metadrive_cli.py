"""run_metadrive CLI + cfg-building tests. The training itself needs MetaDrive (live, slow), but
the ARGUMENT PARSING and CFG WIRING are pure -- so the tunable knobs (iters/wm_steps/entropy/pool)
are unit-tested here without running the sim. This is what makes 'tune against the action-collapse
from the command line' a real, verified feature rather than hardcoded constants.
"""
from scripts.run_metadrive import parse_args, build_cfg


def test_parse_args_exposes_training_knobs():
    """Every training size + the entropy bonus is settable from the CLI (no more python -c editing)."""
    a = parse_args(["--iters", "12", "--wm-steps", "1500", "--behavior-steps", "1000",
                    "--seed-steps", "3000", "--collect", "2000", "--entropy", "0.02",
                    "--num-scenarios", "250"])
    assert a.iters == 12
    assert a.wm_steps == 1500
    assert a.behavior_steps == 1000
    assert a.seed_steps == 3000
    assert a.collect_per_iter == 2000
    assert a.entropy_coef == 0.02
    assert a.num_scenarios == 250


def test_parse_args_defaults_bump_entropy():
    """Defaults: 100-map pool, random-block geometry, and entropy bumped above the old 1e-3 that
    let the actor collapse to saturated full-left/full-throttle actions."""
    a = parse_args([])
    assert a.num_scenarios == 100
    assert a.road_map is None
    assert a.entropy_coef > 1e-3              # raised from the collapse-prone default


def test_build_cfg_threads_entropy_and_pool():
    """build_cfg wires the CLI knobs into the actual training cfg (cfg.entropy_coef is consumed by
    the actor loss in train_behavior)."""
    cfg = build_cfg(num_scenarios=250, road_map=None, entropy_coef=0.02)
    assert cfg.entropy_coef == 0.02
    assert cfg.metadrive_num_scenarios == 250
    assert cfg.metadrive_start_seed == 0
    assert cfg.metadrive_map == 3            # None -> 3 random blocks (varied geometry per seed)


def test_build_cfg_preserves_letter_and_int_maps():
    """A fixed scene (letter string) passes through; a digit string becomes an int block count."""
    assert build_cfg(road_map="SSSS").metadrive_map == "SSSS"
    assert build_cfg(road_map="5").metadrive_map == 5

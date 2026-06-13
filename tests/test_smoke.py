"""Wrap the existing plumbing smoke test (spec §7: test_smoke.py) so it runs under pytest."""
from scripts.smoke_test import main


def test_smoke_plumbing_runs():
    # main() asserts batch shapes internally for both state and image obs; no return value.
    main()

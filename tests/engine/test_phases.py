"""Unit tests for phase sum check engine functions."""

from custom_components.powerguesser.engine.phases import (
    PHASE_TOLERANCE_FRAC,
    PHASE_TOLERANCE_W,
    PhaseSumResult,
    check_phase_sum,
)


def test_check_phase_sum_ok() -> None:
    """Phases summing to grid power within tolerance return OK."""
    assert check_phase_sum([300.0, 300.0, 400.0], 1000.0) == PhaseSumResult.OK
    # Boundary: tolerance at 1000 W is max(200, 100) = 200 W
    assert check_phase_sum([1200.0], 1000.0) == PhaseSumResult.OK
    assert check_phase_sum([800.0], 1000.0) == PhaseSumResult.OK
    # Negative grid: -1000 W with sum -1000 W
    assert check_phase_sum([-1000.0], -1000.0) == PhaseSumResult.OK
    # Zero grid: sum 0 W with grid 0 W -> OK
    assert check_phase_sum([0.0], 0.0) == PhaseSumResult.OK


def test_check_phase_sum_inverted() -> None:
    """Phases summing to negative grid power within tolerance return INVERTED."""
    assert check_phase_sum([-300.0, -300.0, -400.0], 1000.0) == PhaseSumResult.INVERTED
    # Boundary: -1000 W inverted is 1000 W
    assert check_phase_sum([1000.0], -1000.0) == PhaseSumResult.INVERTED
    assert check_phase_sum([-1200.0], 1000.0) == PhaseSumResult.INVERTED
    assert check_phase_sum([-800.0], 1000.0) == PhaseSumResult.INVERTED


def test_check_phase_sum_mismatch() -> None:
    """Phases summing outside tolerance return MISMATCH."""
    assert check_phase_sum([100.0, 100.0, 100.0], 1000.0) == PhaseSumResult.MISMATCH
    assert check_phase_sum([1201.0], 1000.0) == PhaseSumResult.MISMATCH
    assert check_phase_sum([799.0], 1000.0) == PhaseSumResult.MISMATCH
    assert check_phase_sum([-1201.0], 1000.0) == PhaseSumResult.MISMATCH
    assert check_phase_sum([-799.0], 1000.0) == PhaseSumResult.MISMATCH


def test_check_phase_sum_skipped() -> None:
    """Missing grid, empty phases, or any None phase value returns SKIPPED."""
    assert check_phase_sum([], 1000.0) == PhaseSumResult.SKIPPED
    assert check_phase_sum([300.0, 300.0, 400.0], None) == PhaseSumResult.SKIPPED
    assert check_phase_sum([300.0, None, 400.0], 1000.0) == PhaseSumResult.SKIPPED
    assert check_phase_sum([None], 1000.0) == PhaseSumResult.SKIPPED
    assert check_phase_sum([], None) == PhaseSumResult.SKIPPED


def test_check_phase_sum_large_grid_fractional_tolerance() -> None:
    """For large grid values, tolerance is 10% of abs(grid_w)."""
    # 5000 W -> 10% is 500 W (> 200 W)
    assert check_phase_sum([5500.0], 5000.0) == PhaseSumResult.OK
    assert check_phase_sum([5501.0], 5000.0) == PhaseSumResult.MISMATCH
    assert check_phase_sum([-5500.0], 5000.0) == PhaseSumResult.INVERTED
    assert check_phase_sum([-5501.0], 5000.0) == PhaseSumResult.MISMATCH

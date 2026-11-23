"""Unit tests for train/validation/holdout day split logic."""

import numpy as np
import pytest

from custom_components.powerguesser.engine.learning.split import (
    DaySplit,
    block_of,
    calendar_holdout_start,
    data_days,
    split_days,
)


def test_split_days_proportions_and_holdout() -> None:
    """split_days creates 70/30 train/val blocks and isolates holdout days."""
    # 10 data days, holdout 0 -> 7 train, 3 val
    split10 = split_days(list(range(10)), train_fraction=0.7, holdout_days=0)
    assert split10.train == (0, 1, 2, 3, 4, 5, 6)
    assert split10.val == (7, 8, 9)
    assert split10.holdout == ()

    # 7 data days, holdout 0 -> 4 train, 3 val (floor(0.7 * 7) = 4)
    split7 = split_days(list(range(7)), train_fraction=0.7, holdout_days=0)
    assert split7.train == (0, 1, 2, 3)
    assert split7.val == (4, 5, 6)
    assert split7.holdout == ()

    # 10 data days, holdout 2 -> removes last 2 (8, 9), 8 remaining -> 5 train, 3 val
    split_hold = split_days(list(range(10)), train_fraction=0.7, holdout_days=2)
    assert split_hold.train == (0, 1, 2, 3, 4)
    assert split_hold.val == (5, 6, 7)
    assert split_hold.holdout == (8, 9)


def test_split_days_fewer_than_two_days_raises_value_error() -> None:
    """split_days raises ValueError naming data days and dropped count when n < 2."""
    with pytest.raises(ValueError, match=r"need at least 2 data days, have 1 \(3 dropped below 50 % coverage\)"):
        split_days([0], train_fraction=0.7, holdout_days=0, n_dropped=3)


def test_data_days_drops_under_covered_days() -> None:
    """data_days excludes days with less than 50% finite samples."""
    # 3 days, 10 samples per day
    t0_ms = 0
    step_ms = 1000
    day_bounds_ms = [0, 10_000, 20_000, 30_000]

    # Day 0: 8 finite, 2 NaN (80% -> kept)
    # Day 1: 4 finite, 6 NaN (40% -> dropped)
    # Day 2: 10 finite (100% -> kept)
    r = np.array(
        [1.0] * 8 + [np.nan] * 2 + [1.0] * 4 + [np.nan] * 6 + [1.0] * 10,
        dtype=np.float64,
    )

    valid = data_days(r, t0_ms, step_ms, day_bounds_ms)
    assert valid == [0, 2]


def test_block_of_assigns_by_start_day() -> None:
    """block_of maps timestamps to block id and tags episodes spanning midnight by start day."""
    day_bounds_ms = [0, 86_400_000, 2 * 86_400_000, 3 * 86_400_000, 4 * 86_400_000]
    split = DaySplit(train=(0, 1), val=(2,), holdout=(3,))

    # Episode starting 2 min before midnight on day 0 (23:58)
    ts_23_58_day0 = 86_400_000 - 120_000
    # Episode starting on day 1
    ts_day1 = 86_400_000 + 3600_000
    # Episode on day 2 (val)
    ts_day2 = 2 * 86_400_000 + 3600_000
    # Episode on day 3 (holdout)
    ts_day3 = 3 * 86_400_000 + 3600_000
    # Timestamp outside bounds
    ts_outside = 5 * 86_400_000

    start_ts = np.array([ts_23_58_day0, ts_day1, ts_day2, ts_day3, ts_outside], dtype=np.int64)
    blocks = block_of(start_ts, day_bounds_ms, split)

    # 0=train, 1=val, 2=holdout, -1=none
    np.testing.assert_array_equal(blocks, [0, 0, 1, 2, -1])


def test_calendar_holdout_start_window() -> None:
    """calendar_holdout_start anchors the D-18 window to the calendar, clamped at 0."""
    assert calendar_holdout_start(35, 7) == 28
    assert calendar_holdout_start(5, 7) == 0
    assert calendar_holdout_start(7, 7) == 0


def test_split_days_holdout_from_day() -> None:
    """holdout_from_day takes every day at or after the index; the rest splits 70/30."""
    split = split_days(range(35), holdout_from_day=28)
    assert split.holdout == tuple(range(28, 35))
    # 28 remaining days: floor(0.7 * 28) = 19 train, 9 val
    assert split.train == tuple(range(0, 19))
    assert split.val == tuple(range(19, 28))


def test_split_days_holdout_from_day_with_calendar_gaps() -> None:
    """Missing calendar days stay missing: the holdout is (28, 29, 32, 33, 34)."""
    days = [d for d in range(35) if d not in (30, 31)]
    split = split_days(days, holdout_from_day=28)
    assert split.holdout == (28, 29, 32, 33, 34)
    assert split.train == tuple(range(0, 19))
    assert split.val == tuple(range(19, 28))


def test_split_days_holdout_days_and_from_day_are_mutually_exclusive() -> None:
    """holdout_days (last N valid days) and holdout_from_day (calendar) never combine."""
    with pytest.raises(ValueError, match="holdout_days.*holdout_from_day"):
        split_days(range(35), holdout_days=7, holdout_from_day=28)


def test_split_days_without_holdout_from_day_unchanged() -> None:
    """Callers that pass nothing get the same DaySplit as before (D-18 additive)."""
    split = split_days(list(range(10)), holdout_days=2)
    assert split == DaySplit(
        train=(0, 1, 2, 3, 4),
        val=(5, 6, 7),
        holdout=(8, 9),
    )

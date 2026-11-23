"""Unit tests for coverage calculation and segment operations."""

import sqlite3
from pathlib import Path

from custom_components.powerguesser.engine.buffer import BufferDB
from custom_components.powerguesser.engine.coverage import (
    MS_PER_DAY,
    ChannelCoverage,
    channel_coverage,
    intersect,
    segments,
    usable_days,
)
from custom_components.powerguesser.engine.learning.series import load_residual
from custom_components.powerguesser.engine.learning.split import data_days
from tests.fixtures.learning.synth import (
    ENTITY_BATTERY,
    ENTITY_GRID,
    ENTITY_SOLAR,
    SynthSpec,
    build_buffer,
    utc_day_bounds,
)


def test_segments_empty_and_invalid_bounds() -> None:
    """Empty rows or non-positive window returns empty list."""
    assert segments([], 1000, 5000) == []
    assert segments([(2000, 100)], 5000, 1000) == []
    assert segments([(2000, 100)], 1000, 1000) == []


def test_segments_clipping_and_merging() -> None:
    """Segments are clipped to [start_ms, end_ms) and consecutive spans are merged."""
    # Rows:
    # (500, 50): starts before start_ms (1000)
    # (2000, None): gap starts at 2000 -> clipped span [1000, 2000)
    # (3000, 100): span starts at 3000
    # (4000, 200): continues
    # (6000, None): ends at 6000 -> clipped span [3000, 5000) with end_ms=5000
    rows = [
        (500, 50),
        (2000, None),
        (3000, 100),
        (4000, 200),
        (6000, None),
    ]
    segs = segments(rows, 1000, 5000)
    assert segs == [(1000, 2000), (3000, 5000)]


def test_segments_last_non_null_extends_to_end_ms() -> None:
    """A trailing non-NULL row extends to end_ms."""
    rows = [
        (1000, 50),
        (2000, 100),
    ]
    segs = segments(rows, 0, 4000)
    assert segs == [(1000, 4000)]


def test_intersect_multiple_lists() -> None:
    """Intersect finds common spans across lists."""
    list1 = [(1000, 3000), (4000, 6000)]
    list2 = [(2000, 5000)]
    list3 = [(1500, 2500), (4500, 5500)]

    assert intersect([list1, list2]) == [(2000, 3000), (4000, 5000)]
    assert intersect([list1, list2, list3]) == [(2000, 2500), (4500, 5000)]


def test_intersect_empty_cases() -> None:
    """Intersect returns empty when any list is empty or no lists given."""
    assert intersect([]) == []
    assert intersect([[]]) == []
    assert intersect([[(1000, 2000)], []]) == []


def test_usable_days_solar_excluded() -> None:
    """Solar NULLs do not reduce usable days when other gross channels exist."""
    # Day is 86_400_000 ms.
    # Grid (ch=1) has 2 full days of continuous data.
    # Solar (ch=2) has night gaps every 12 hours.
    one_day = MS_PER_DAY
    two_days = 2 * MS_PER_DAY

    segments_by_channel = {
        1: [(0, two_days)],
        2: [(0, one_day // 2), (one_day, one_day + one_day // 2)],  # half days
    }
    roles = {1: "grid", 2: "solar"}

    # Grid alone decides usable days (solar NULLs ignored)
    days = usable_days(segments_by_channel, roles)
    assert abs(days - 2.0) < 1e-4


def test_usable_days_when_all_solar() -> None:
    """When every gross channel is solar, solar gaps reduce usable days."""
    half_day = MS_PER_DAY // 2
    segments_by_channel = {
        2: [(0, half_day)],
    }
    roles = {2: "solar"}

    days = usable_days(segments_by_channel, roles)
    assert abs(days - 0.5) < 1e-4


def test_usable_days_plugs_do_not_enter() -> None:
    """Plugs / metered devices (not in roles) never enter usable_days."""
    two_days = 2 * MS_PER_DAY
    segments_by_channel = {
        1: [(0, two_days)],
        3: [(0, 1000)],  # plug with almost no history
    }
    roles = {1: "grid"}  # plug ch=3 not in roles

    days = usable_days(segments_by_channel, roles)
    assert abs(days - 2.0) < 1e-4


def test_usable_days_empty_inputs() -> None:
    """Empty roles or empty segments return 0.0."""
    assert usable_days({}, {1: "grid"}) == 0.0
    assert usable_days({1: [(0, 1000)]}, {}) == 0.0


def test_channel_coverage_stats() -> None:
    """Channel coverage computes days, first_ms, last_ms, and gap_count."""
    # Window: [1000, 5000)
    # Seed row at 500 (before window)
    # Non-NULL at 1500, NULL at 2500 (gap 1)
    # Non-NULL at 3500, NULL at 4500 (gap 2)
    rows = [
        (500, 10),
        (1500, 20),
        (2500, None),
        (3500, 30),
        (4500, None),
    ]
    cov = channel_coverage(rows, 1000, 5000)

    # First and last non-NULL inside [1000, 5000):
    assert cov.first_ms == 1500
    assert cov.last_ms == 3500

    # Spans in [1000, 5000):
    # From 500 to 2500 -> clipped [1000, 2500) = 1500 ms
    # From 3500 to 4500 -> clipped [3500, 4500) = 1000 ms
    # Total ms = 2500 ms
    expected_days = 2500 / MS_PER_DAY
    assert abs(cov.days - expected_days) < 1e-6

    # Transitions from non-NULL -> NULL:
    # 1500 -> 2500 is 1
    # 3500 -> 4500 is 2
    assert cov.gap_count == 2


def test_channel_coverage_empty_rows() -> None:
    """Channel coverage on empty rows returns 0 days, None bounds, 0 gaps."""
    cov = channel_coverage([], 1000, 5000)
    assert cov == ChannelCoverage(
        days=0.0,
        first_ms=None,
        last_ms=None,
        gap_count=0,
    )


# --- usable-days gate vs extract parity (folded todo 2026-11-18) ---------------

DAYS = 11
STEP_MS = 30_000
END_MS = 1_793_491_200_000  # UTC midnight


def _blank_channel_day(buf_path: Path, channel: int, start_ms: int, end_ms: int) -> None:
    """Set every sample of one channel inside [start_ms, end_ms) to NULL."""
    conn = sqlite3.connect(str(buf_path))
    try:
        conn.execute(
            "UPDATE samples SET w = NULL WHERE ch = ? AND ts >= ? AND ts < ?",
            (channel, start_ms, end_ms),
        )
        conn.commit()
    finally:
        conn.close()


def _gate_days(buf_path: Path, bounds: list[int], channel_ids: dict[str, int]) -> float:
    """Usable days as the learning gate computes them (BufferDB.coverage_for)."""
    start_ms, end_ms = bounds[0], bounds[-1]
    windows = {
        channel_ids[ENTITY_GRID]: (start_ms, end_ms),
        channel_ids[ENTITY_SOLAR]: (start_ms, end_ms),
        channel_ids[ENTITY_BATTERY]: (start_ms, end_ms),
    }
    roles = {
        channel_ids[ENTITY_GRID]: "grid",
        channel_ids[ENTITY_SOLAR]: "solar",
        channel_ids[ENTITY_BATTERY]: "battery",
    }
    db = BufferDB(str(buf_path))
    db.open()
    try:
        usable, _ = db.coverage_for(windows, roles)
        return usable
    finally:
        db.close_clean()


def _extract_days(buf_path: Path, bounds: list[int]) -> int:
    """Data days as extract counts them (load_residual + split.data_days)."""
    series = load_residual(str(buf_path), bounds[0], bounds[-1], STEP_MS, [], {})
    return len(data_days(series.residual, bounds[0], STEP_MS, bounds))


def test_usable_days_matches_extract_on_solar_outage(tmp_path: Path) -> None:
    """One day whose solar channel is all NULL still counts in gate and extract.

    The gate intersects only the non-solar gross channels, and extract's
    gross_on_read maps a NULL solar row to 0 W, so both count the day.
    """
    buf = tmp_path / "buffer.sqlite"
    data = build_buffer(buf, SynthSpec(days=DAYS), END_MS)
    bounds = utc_day_bounds(END_MS, DAYS)
    outage_day = 5
    _blank_channel_day(
        buf,
        data.channel_ids[ENTITY_SOLAR],
        bounds[outage_day],
        bounds[outage_day + 1],
    )

    gate = _gate_days(buf, bounds, data.channel_ids)
    assert abs(gate - DAYS) < 0.05
    assert _extract_days(buf, bounds) == DAYS


def test_usable_days_matches_extract_on_grid_outage(tmp_path: Path) -> None:
    """One day whose grid channel is all NULL drops that day in gate and extract.

    A NULL grid row makes gross NaN in gross_on_read, and the gate's non-solar
    intersection loses the day too, so both report DAYS - 1.
    """
    buf = tmp_path / "buffer.sqlite"
    data = build_buffer(buf, SynthSpec(days=DAYS), END_MS)
    bounds = utc_day_bounds(END_MS, DAYS)
    outage_day = 5
    _blank_channel_day(
        buf,
        data.channel_ids[ENTITY_GRID],
        bounds[outage_day],
        bounds[outage_day + 1],
    )

    gate = _gate_days(buf, bounds, data.channel_ids)
    assert abs(gate - (DAYS - 1)) < 0.05
    assert _extract_days(buf, bounds) == DAYS - 1

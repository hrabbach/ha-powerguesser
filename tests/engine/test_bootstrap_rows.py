"""Unit tests for pure engine recorder history conversion."""

from custom_components.powerguesser.engine.bootstrap import (
    GAP_FACTOR,
    rows_from_history,
)


def test_gap_factor_value() -> None:
    """Verify GAP_FACTOR is 3.0."""
    assert GAP_FACTOR == 3.0


def test_rows_from_history_empty() -> None:
    """Empty states list produces empty rows."""
    assert rows_from_history([], ch=1, unit_factor=1.0, mode="on_change", interval_s=30.0) == []


def test_rows_from_history_lu_lr_pairs() -> None:
    """Held values with lr > lu emit both lu and lr rows."""
    states = [
        ("100", 10.0, 25.0),
        ("200", 30.0, 30.0),
    ]
    rows = rows_from_history(states, ch=1, unit_factor=1.0, mode="on_change", interval_s=30.0)
    assert rows == [
        (10_000, 1, 100),
        (25_000, 1, 100),
        (30_000, 1, 200),
    ]


def test_rows_from_history_null_for_unavailable_and_non_numeric() -> None:
    """Unavailable, unknown, and non-numeric values emit NULL and no lr row."""
    states = [
        ("unavailable", 10.0, 20.0),
        ("unknown", 30.0, 40.0),
        ("bad_value", 50.0, 60.0),
    ]
    rows = rows_from_history(states, ch=2, unit_factor=1.0, mode="on_change", interval_s=30.0)
    assert rows == [
        (10_000, 2, None),
        (30_000, 2, None),
        (50_000, 2, None),
    ]


def test_rows_from_history_kw_unit_factor() -> None:
    """kW unit factor (1000) converts float states correctly."""
    states = [
        ("1.5", 10.0, 10.0),
        ("0.25", 20.0, 20.0),
    ]
    rows = rows_from_history(states, ch=3, unit_factor=1000.0, mode="on_change", interval_s=30.0)
    assert rows == [
        (10_000, 3, 1500),
        (20_000, 3, 250),
    ]


def test_rows_from_history_on_change_without_gap_markers() -> None:
    """On-change sources hold until next row with no NULL gap marker even with large delay."""
    states = [
        ("100", 0.0, 60.0),
        ("250", 300.0, 300.0),
    ]
    # D-23 change-cadence median such as 31.0
    rows = rows_from_history(states, ch=1, unit_factor=1.0, mode="on_change", interval_s=31.0)
    assert rows == [
        (0, 1, 100),
        (60_000, 1, 100),
        (300_000, 1, 250),
    ]


def test_rows_from_history_re_reporting_gap_marker() -> None:
    """Re-reporting sources emit NULL at lr + 3x interval when next change is later."""
    states = [
        ("100", 0.0, 60.0),
        ("250", 300.0, 300.0),
    ]
    # GAP_FACTOR = 3.0, interval = 30 -> gap at 60 + 90 = 150 s (150_000 ms)
    rows = rows_from_history(states, ch=1, unit_factor=1.0, mode="re_reporting", interval_s=30.0)
    assert rows == [
        (0, 1, 100),
        (60_000, 1, 100),
        (150_000, 1, None),
        (300_000, 1, 250),
    ]


def test_rows_from_history_re_reporting_no_gap_when_next_change_within_threshold() -> None:
    """Re-reporting sources emit no gap marker when next change is before lr + 3x interval."""
    states = [
        ("100", 0.0, 60.0),
        ("250", 140.0, 140.0),
    ]
    # 60 + 3 * 30 = 150. Next change is at 140 < 150 -> no gap marker
    rows = rows_from_history(states, ch=1, unit_factor=1.0, mode="re_reporting", interval_s=30.0)
    assert rows == [
        (0, 1, 100),
        (60_000, 1, 100),
        (140_000, 1, 250),
    ]


def test_rows_from_history_live_patch() -> None:
    """Last recorder row is extended to live_lr_s when given and larger."""
    states = [("100", 0.0, 0.0)]
    rows = rows_from_history(
        states,
        ch=1,
        unit_factor=1.0,
        mode="re_reporting",
        interval_s=30.0,
        live_lr_s=90.0,
    )
    assert rows == [
        (0, 1, 100),
        (90_000, 1, 100),
    ]


def test_rows_from_history_duplicate_timestamps_collapsed() -> None:
    """Duplicate timestamps are collapsed with last-wins, returning sorted by ts."""
    states = [
        ("100", 10.0, 20.0),
        ("200", 20.0, 30.0),
    ]
    # First tuple emits (10_000, 1, 100) and (20_000, 1, 100).
    # Second tuple emits (20_000, 1, 200) and (30_000, 1, 200).
    # At ts=20_000, last wins -> 200.
    rows = rows_from_history(states, ch=1, unit_factor=1.0, mode="on_change", interval_s=30.0)
    assert rows == [
        (10_000, 1, 100),
        (20_000, 1, 200),
        (30_000, 1, 200),
    ]


def test_rows_from_history_re_reporting_trailing_gap() -> None:
    """When end_s is provided and exceeds gap threshold, emit trailing NULL."""
    states = [
        ("100", 0.0, 60.0),
    ]
    # interval 30, gap threshold 60 + 90 = 150 s. end_s is 300 s.
    rows = rows_from_history(
        states,
        ch=1,
        unit_factor=1.0,
        mode="re_reporting",
        interval_s=30.0,
        end_s=300.0,
    )
    assert rows == [
        (0, 1, 100),
        (60_000, 1, 100),
        (150_000, 1, None),
    ]


def test_rows_from_history_re_reporting_no_trailing_gap_when_within_threshold() -> None:
    """When end_s is within gap threshold, no trailing NULL is emitted."""
    states = [
        ("100", 0.0, 60.0),
    ]
    # interval 30, gap threshold 150 s. end_s is 120 s.
    rows = rows_from_history(
        states,
        ch=1,
        unit_factor=1.0,
        mode="re_reporting",
        interval_s=30.0,
        end_s=120.0,
    )
    assert rows == [
        (0, 1, 100),
        (60_000, 1, 100),
    ]


def test_rows_from_history_gap_markers_off_holds_without_age_limit() -> None:
    """Metered plugs (P1 D-07): no inner or trailing gap marker, unavailable still NULL."""
    states = [
        ("100", 0.0, 60.0),
        ("250", 300.0, 300.0),
        ("unavailable", 400.0, 400.0),
        ("300", 500.0, 500.0),
    ]
    rows = rows_from_history(
        states,
        ch=1,
        unit_factor=1.0,
        mode="re_reporting",
        interval_s=30.0,
        end_s=900.0,
        gap_markers=False,
    )
    assert rows == [
        (0, 1, 100),
        (60_000, 1, 100),
        (300_000, 1, 250),
        (400_000, 1, None),
        (500_000, 1, 300),
    ]


"""Engine tests for the HA-free capture window rules (CAP-01 refusals, DST).

Plan 08-03 Task 1. ``resolve_window`` validates a retroactive window before any
snap runs: it refuses a window outside the buffer, across a sample gap or too
short for the meter's grid, each with its own stated reason. ``local_to_utc_ms``
resolves a naive local time in a named time zone and refuses a wall time in the
repeated (ambiguous) or skipped (nonexistent) DST hour.
"""

from datetime import datetime, timezone
from pathlib import Path
import sqlite3
from typing import Any
from zoneinfo import ZoneInfo

import pytest

from custom_components.powerguesser.engine.capture import window as cap_window
from custom_components.powerguesser.engine.capture.window import (
    REFUSAL_CODES,
    CaptureRefused,
    local_to_utc_ms,
    resolve_window,
)
from tests.fixtures.learning.synth import (
    ENTITY_BATTERY,
    ENTITY_GRID,
    ENTITY_SOLAR,
    SynthSpec,
    build_buffer,
)

TZ = "Europe/Berlin"
END_MS = 1_700_000_000_000
GRID_S = 30
GRID_MS = GRID_S * 1000
HOUR_MS = 3_600_000


def _utc(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)


def _build(tmp_path: Path, spec: SynthSpec | None = None) -> tuple[str, Any]:
    path = str(tmp_path / "buffer.sqlite")
    data = build_buffer(path, spec or SynthSpec(days=4), END_MS)
    return path, data


def _resolve(path: str, start: int, end: int, *, now_ms: int | None = None,
             enabled: list[str] | None = None, grid_s: int = GRID_S) -> tuple[int, int]:
    return resolve_window(
        _utc(start),
        _utc(end),
        tz_name=TZ,
        now_ms=now_ms if now_ms is not None else END_MS + 10 * HOUR_MS,
        grid_s=grid_s,
        enabled_event_types=enabled if enabled is not None else ["step", "cycle_shape", "continuous"],
        buffer_path=path,
    )


# --- local_to_utc_ms / DST -------------------------------------------------


def test_local_time_dst_rules() -> None:
    """A wall time in a repeated hour is ambiguous, in a skipped hour nonexistent."""
    with pytest.raises(CaptureRefused) as amb:
        local_to_utc_ms(datetime(2026, 10, 25, 2, 30), TZ)
    assert amb.value.code == "ambiguous_local_time"

    with pytest.raises(CaptureRefused) as non:
        local_to_utc_ms(datetime(2026, 3, 29, 2, 30), TZ)
    assert non.value.code == "nonexistent_local_time"

    # 01:30 local on the fall-back day is unambiguous (CEST, +2 -> 23:30Z prior day).
    got = local_to_utc_ms(datetime(2026, 10, 25, 1, 30), TZ)
    assert got == int(
        datetime(2026, 10, 24, 23, 30, tzinfo=timezone.utc).timestamp() * 1000
    )

    # An aware datetime is used as given.
    aware = datetime(2026, 10, 25, 1, 30, tzinfo=timezone.utc)
    assert local_to_utc_ms(aware, TZ) == int(aware.timestamp() * 1000)


def test_refusal_codes_vocabulary() -> None:
    """REFUSAL_CODES holds exactly the nine documented codes."""
    assert set(REFUSAL_CODES) == {
        "invalid_window",
        "future_end",
        "window_too_long",
        "too_short_for_rate",
        "interval_too_coarse",
        "outside_buffer",
        "across_gap",
        "ambiguous_local_time",
        "nonexistent_local_time",
    }
    assert cap_window.MAX_PAIR_S == 21600


# --- resolve_window refusals ----------------------------------------------


def test_window_refusal_order_and_codes(tmp_path: Path) -> None:
    """Each refusal case raises its own code in the documented order."""
    path, data = _build(tmp_path)
    first_ms = min(data.rows[ENTITY_GRID][0][0], data.rows[ENTITY_BATTERY][0][0])
    last_ms = max(data.rows[ENTITY_GRID][-1][0], data.rows[ENTITY_BATTERY][-1][0])
    base = first_ms + 2 * HOUR_MS

    with pytest.raises(CaptureRefused) as e1:
        _resolve(path, base, base)  # end == start
    assert e1.value.code == "invalid_window"

    with pytest.raises(CaptureRefused) as e2:
        _resolve(path, base, base + HOUR_MS, now_ms=base + 60_000)  # future end
    assert e2.value.code == "future_end"

    with pytest.raises(CaptureRefused) as e3:
        _resolve(path, base, base + 6 * HOUR_MS + 1000)  # 6 h + 1 s
    assert e3.value.code == "window_too_long"

    with pytest.raises(CaptureRefused) as e4:
        _resolve(path, base, base + GRID_MS)  # one grid step long
    assert e4.value.code == "too_short_for_rate"

    with pytest.raises(CaptureRefused) as e5:
        _resolve(path, base, base + 5 * 60_000, enabled=["cycle_shape", "continuous"])
    assert e5.value.code == "interval_too_coarse"

    with pytest.raises(CaptureRefused) as e6:
        _resolve(path, first_ms - HOUR_MS, first_ms - HOUR_MS + 5 * 60_000)
    assert e6.value.code == "outside_buffer"

    with pytest.raises(CaptureRefused) as e7:
        end = last_ms + GRID_MS + 1000
        _resolve(path, end - 5 * 60_000, end)
    assert e7.value.code == "outside_buffer"

    # A 400 s gross hole inside a window in its own slot -> across_gap.
    slot = first_ms + 8 * HOUR_MS
    conn = sqlite3.connect(path)
    grid_ch = data.channel_ids[ENTITY_GRID]
    conn.execute(
        "DELETE FROM samples WHERE ch = ? AND ts >= ? AND ts <= ?",
        (grid_ch, slot, slot + 400_000),
    )
    conn.commit()
    conn.close()
    with pytest.raises(CaptureRefused) as e8:
        _resolve(path, slot + 10_000, slot + 300_000)
    assert e8.value.code == "across_gap"

    # A NULL gross row inside a window in another slot -> across_gap.
    slot2 = first_ms + 12 * HOUR_MS
    conn = sqlite3.connect(path)
    conn.execute(
        "INSERT OR REPLACE INTO samples(ts, ch, w) VALUES(?, ?, NULL)",
        (slot2 + 150_000, grid_ch),
    )
    conn.commit()
    conn.close()
    with pytest.raises(CaptureRefused) as e9:
        _resolve(path, slot2 + 10_000, slot2 + 300_000)
    assert e9.value.code == "across_gap"


def test_window_accepts_edge_values(tmp_path: Path) -> None:
    """Boundary values (6 h, 2 grid steps, a 150 s spacing, a NULL solar row) pass."""
    path, data = _build(tmp_path)
    grid_rows = data.rows[ENTITY_GRID]
    first_ms = grid_rows[0][0]
    base = first_ms + 18 * HOUR_MS

    # Exactly 6 h.
    s, e = _resolve(path, base, base + 6 * HOUR_MS)
    assert (s, e) == (base, base + 6 * HOUR_MS)

    # Exactly 2 grid steps.
    s2, e2 = _resolve(path, base, base + 2 * GRID_MS)
    assert e2 - s2 == 2 * GRID_MS

    # A NULL solar row inside a window is accepted (solar NULL reads as 0 W).
    solar_ch = data.channel_ids[ENTITY_SOLAR]
    slot = base + 60 * 60_000
    conn = sqlite3.connect(path)
    conn.execute(
        "INSERT OR REPLACE INTO samples(ts, ch, w) VALUES(?, ?, NULL)",
        (slot + 150_000, solar_ch),
    )
    conn.commit()
    conn.close()
    s3, _ = _resolve(path, slot, slot + 300_000)
    assert s3 == slot

    # A 150 s spacing at grid 30 (= GAP_MAX_FLOOR_S) is accepted.
    slow_path, slow_data = _build(
        tmp_path / "slow", SynthSpec(days=3, interval_s=(150.0, 150.0))
    )
    slow_first = slow_data.rows[ENTITY_GRID][0][0]
    slow_base = slow_first + 2 * HOUR_MS
    s4, e4 = _resolve(slow_path, slow_base, slow_base + 300_000)
    assert e4 - s4 == 300_000


def test_window_refuses_gap_at_either_scan_edge(tmp_path: Path) -> None:
    """A gross hole straddling the window's leading or trailing scan edge is refused.

    ``read_range`` also returns the seed row (the last sample before the scan's
    lower bound); it must seed the gap check, and the trailing edge from the last
    row to the scan's upper bound must be checked too, or a hole at either edge
    slips through (CAP-01).
    """
    path, data = _build(tmp_path)
    grid_ch = data.channel_ids[ENTITY_GRID]
    first_ms = min(data.rows[ENTITY_GRID][0][0], data.rows[ENTITY_BATTERY][0][0])
    margin_ms = 120 * 1000  # DEFAULT_CONFIG.snap_margin_s

    # Leading edge: a hole ending exactly at the scan's lower bound.
    base = first_ms + 10 * HOUR_MS
    win_lo = base - margin_ms
    conn = sqlite3.connect(path)
    conn.execute(
        "DELETE FROM samples WHERE ch = ? AND ts >= ? AND ts < ?",
        (grid_ch, win_lo - 6 * GRID_MS, win_lo),
    )
    conn.commit()
    conn.close()
    with pytest.raises(CaptureRefused) as lead:
        _resolve(path, base, base + 300_000)
    assert lead.value.code == "across_gap"

    # Trailing edge: a hole from the scan's start running past its upper bound,
    # so no pair of in-range rows straddles it.
    base2 = first_ms + 14 * HOUR_MS
    conn = sqlite3.connect(path)
    conn.execute(
        "DELETE FROM samples WHERE ch = ? AND ts >= ? AND ts < ?",
        (grid_ch, base2, base2 + 900_000),
    )
    conn.commit()
    conn.close()
    with pytest.raises(CaptureRefused) as trail:
        _resolve(path, base2, base2 + 300_000)
    assert trail.value.code == "across_gap"

"""Engine tests for the capture snap oracle and the live/retro search ranges.

Plan 08-03 Task 2. The oracle test proves the snap met the roadmap bar on
synthetic kettle runs (rise/fall within one grid step of truth for >= 98 % of
runs at the default margin); the non-default test proves the D-09 tuning path is
live (a 60 s margin misses a run the 120 s default snaps).
"""

from datetime import datetime, timezone
from pathlib import Path
import sqlite3
from zoneinfo import ZoneInfo

import numpy as np

from custom_components.powerguesser.engine.capture import snap
from custom_components.powerguesser.engine.capture.config import CaptureConfig
from custom_components.powerguesser.engine.learning import episodes
from tests.fixtures.learning.synth import (
    ENTITY_BATTERY,
    ENTITY_GRID,
    SynthSpec,
    build_buffer,
)

END_MS = 1_700_000_000_000
GRID_S = 30
TOL_MS = GRID_S * 1000 + 3000


def _context(buffer_path: str) -> dict:
    return {
        "buffer_path": buffer_path,
        "grid_s": GRID_S,
        "enabled_event_types": ["step", "cycle_shape", "continuous"],
        "metered_channels": [],
        "phase_channels": {},
        "session_windows": [],
        "tz_name": "UTC",
    }


def test_snap_oracle_on_synthetic_kettle_runs(tmp_path: Path) -> None:
    """Jittered user windows snap rise and fall within one grid step for >= 98 % of runs."""
    path = str(tmp_path / "buffer.sqlite")
    data = build_buffer(path, SynthSpec(days=10, interval_s=(30.0, 30.0)), END_MS)
    context = _context(path)
    rng = np.random.default_rng(8)

    snapped = 0
    none_count = 0
    total = 0
    misses: list[str] = []
    for on_ms, off_ms in data.kettle_runs:
        start_ms = int(on_ms + rng.uniform(-60, 60) * 1000)
        end_ms = int(off_ms + rng.uniform(-60, 60) * 1000)
        rise_range, fall_range = snap.retro_ranges(start_ms, end_ms)
        out = snap.snap_run(
            context,
            rise_range_ms=rise_range,
            fall_range_ms=fall_range,
            ref_start_ms=start_ms,
            ref_end_ms=end_ms,
        )
        res = out["result"]
        total += 1
        if res["verdict"] == "none":
            none_count += 1
            misses.append(f"none on={on_ms} off={off_ms}")
            continue
        if (
            abs(res["rise_ms"] - on_ms) <= TOL_MS
            and abs(res["fall_ms"] - off_ms) <= TOL_MS
        ):
            snapped += 1
        else:
            misses.append(
                f"off-by rise={res['rise_ms'] - on_ms} fall={res['fall_ms'] - off_ms} "
                f"verdict={res['verdict']} on={on_ms}"
            )

    fraction = snapped / total if total else 0.0
    assert fraction >= 0.98, (
        f"snapped fraction {fraction:.3f} ({snapped}/{total}); misses={misses}"
    )
    assert none_count == 0, f"{none_count} of {total} windows returned none: {misses}"


def test_non_default_config_changes_outcome(tmp_path: Path) -> None:
    """A 60 s margin misses a run the 120 s default snaps (the tuning path is live)."""
    path = str(tmp_path / "buffer.sqlite")
    data = build_buffer(path, SynthSpec(days=6, interval_s=(30.0, 30.0)), END_MS)
    context = _context(path)
    on_ms, off_ms = data.kettle_runs[-2]

    start_ms = on_ms - 100_000
    end_ms = off_ms + 100_000

    rise_range, fall_range = snap.retro_ranges(start_ms, end_ms)
    default_out = snap.snap_run(
        context,
        rise_range_ms=rise_range,
        fall_range_ms=fall_range,
        ref_start_ms=start_ms,
        ref_end_ms=end_ms,
    )
    assert default_out["result"]["verdict"] != "none"
    assert abs(default_out["result"]["rise_ms"] - on_ms) <= TOL_MS

    tight = CaptureConfig(snap_margin_s=60)
    tight_rise, tight_fall = snap.retro_ranges(start_ms, end_ms, tight)
    tight_out = snap.snap_run(
        context,
        rise_range_ms=tight_rise,
        fall_range_ms=tight_fall,
        ref_start_ms=start_ms,
        ref_end_ms=end_ms,
        config=tight,
    )
    assert tight_out["result"]["verdict"] == "none"


# --- DST-safe local-day bounds (CAP-01) ------------------------------------


def test_day_bounds_advance_by_calendar_across_dst() -> None:
    """Local midnights step by calendar date: exactly one per day across a DST change.

    The previous +86400 s advance re-snapped into the same local day on the
    autumn fall-back and emitted a duplicate midnight, shifting every later day
    index (wrong weekday/hour features) for any capture crossing the change.
    """
    tz = ZoneInfo("Europe/Berlin")

    def _utc_ms(y: int, m: int, d: int, h: int) -> int:
        return int(datetime(y, m, d, h, tzinfo=timezone.utc).timestamp() * 1000)

    for start, end, dst_hours in (
        ((2026, 10, 21, 12), (2026, 10, 29, 12), 25.0),  # Oct 25 autumn fall-back
        ((2026, 3, 26, 12), (2026, 4, 3, 12), 23.0),  # Mar 29 spring-forward
    ):
        bounds, first_weekday = snap._day_bounds_ms(
            _utc_ms(*start), _utc_ms(*end), "Europe/Berlin"
        )
        assert all(a < b for a, b in zip(bounds, bounds[1:]))  # strictly increasing
        local = [datetime.fromtimestamp(b / 1000, tz=tz) for b in bounds]
        assert all(
            (d.hour, d.minute, d.second, d.microsecond) == (0, 0, 0, 0) for d in local
        )
        dates = [d.date() for d in local]
        # One bound per calendar day: no duplicated and no skipped local day.
        assert all((b - a).days == 1 for a, b in zip(dates, dates[1:])), dates
        assert first_weekday == local[0].weekday()
        gaps_h = [(b - a) / 3_600_000 for a, b in zip(bounds, bounds[1:])]
        assert dst_hours in gaps_h, gaps_h


def test_day_bounds_features_after_fall_back() -> None:
    """compute_features reads the right local hour/weekday on a day after the fall-back."""
    tz = ZoneInfo("Europe/Berlin")
    step_ms = GRID_S * 1000
    ctx_start = int(datetime(2026, 10, 21, 12, tzinfo=timezone.utc).timestamp() * 1000)
    ctx_end = int(datetime(2026, 10, 29, 12, tzinfo=timezone.utc).timestamp() * 1000)
    bounds, first_weekday = snap._day_bounds_ms(ctx_start, ctx_end, "Europe/Berlin")

    event = datetime(2026, 10, 27, 14, 0, tzinfo=tz)  # a normal 24 h day after the change
    event_ms = int(event.timestamp() * 1000)
    assert (event_ms - ctx_start) % step_ms == 0
    start_i = (event_ms - ctx_start) // step_ms
    table = {
        "start_i": np.array([start_i], dtype=np.int64),
        "end_i": np.array([start_i + 4], dtype=np.int64),
    }
    feat = episodes.compute_features(
        table, ctx_start, step_ms, bounds, first_weekday, {}
    )
    assert int(feat["hour"][0]) == 14
    assert int(feat["dow"][0]) == event.weekday()


# --- native max gap measures the run edges (a boundary gap must flag) -------


def _delete_samples(
    path: str, channels: list[int], lo_ms: int, hi_ms: int | None
) -> None:
    """Delete a channel's rows in [lo_ms, hi_ms) (or from lo_ms on when hi is None)."""
    placeholders = ",".join("?" * len(channels))
    conn = sqlite3.connect(path)
    if hi_ms is None:
        conn.execute(
            f"DELETE FROM samples WHERE ch IN ({placeholders}) AND ts >= ?",
            (*channels, lo_ms),
        )
    else:
        conn.execute(
            f"DELETE FROM samples WHERE ch IN ({placeholders}) AND ts >= ? AND ts < ?",
            (*channels, lo_ms, hi_ms),
        )
    conn.commit()
    conn.close()


def test_native_max_gap_measures_leading_edge(tmp_path: Path) -> None:
    """A missing native sample straddling the run's leading edge raises the gap.

    ``read_range``'s seed row (the last sample before the scan's lower bound) is
    the only witness of a hole at the leading edge; dropping it hid the gap.
    """
    path = str(tmp_path / "buffer.sqlite")
    data = build_buffer(path, SynthSpec(days=6, interval_s=(30.0, 30.0)), END_MS)
    on_ms, off_ms = data.kettle_runs[-3]
    grid_ms = GRID_S * 1000
    chs = [data.channel_ids[ENTITY_GRID], data.channel_ids[ENTITY_BATTERY]]

    base = snap._native_max_gap_s(path, on_ms, off_ms, grid_ms)
    assert base is not None and base <= grid_ms / 1000.0 + 1e-9

    lo = on_ms - grid_ms
    # Remove the rows just before the scan's lower bound: the seed and the first
    # in-range row are now far apart, which only the seed-seeded check sees.
    _delete_samples(path, chs, lo - 6 * grid_ms, lo)
    lead = snap._native_max_gap_s(path, on_ms, off_ms, grid_ms)
    assert lead is not None and lead >= 1.5 * grid_ms / 1000.0


def test_native_max_gap_measures_trailing_edge(tmp_path: Path) -> None:
    """A missing native sample straddling the run's fall boundary raises the gap.

    The trailing edge from the last in-range row to the scan's upper bound was
    never measured, so a hole running off the fall boundary slipped through.
    """
    path = str(tmp_path / "buffer.sqlite")
    data = build_buffer(path, SynthSpec(days=6, interval_s=(30.0, 30.0)), END_MS)
    on_ms, off_ms = data.kettle_runs[-3]
    grid_ms = GRID_S * 1000
    chs = [data.channel_ids[ENTITY_GRID], data.channel_ids[ENTITY_BATTERY]]
    hi = off_ms + grid_ms

    # Remove everything from near the fall boundary on: the last in-range row is
    # now far before hi, which only the trailing-edge check sees.
    _delete_samples(path, chs, hi - 3 * grid_ms, None)
    trail = snap._native_max_gap_s(path, on_ms, off_ms, grid_ms)
    assert trail is not None and trail >= 1.5 * grid_ms / 1000.0

"""Learning residual rule for a missing metered value (debug extract-zero-data-days).

Owner pick C (amends P3 A7 / P2 D-03 for metered channels in the learning path): a
missing metered value holds the last known value for at most 300 s, then counts as 0 W
in the residual. The plug series used as training examples stays NaN beyond the hold
limit, so no fake off edges appear. Synthetic data only, shaped like the live evidence:
gross fully covered, one plug offline for days, one 1.6 kW plug on a 300 s cadence whose
rows are each followed by a stale NULL marker 90 s later.
"""

from pathlib import Path

import numpy as np
import pytest

from custom_components.powerguesser.engine.buffer import (
    KIND_GROSS,
    KIND_METERED,
    KIND_SESSION,
    ROLE_METERED,
    ROLE_SESSION,
    BufferDB,
    ChannelSpec,
    GrossLayoutEntry,
)
from custom_components.powerguesser.engine.learning.series import load_residual
from custom_components.powerguesser.engine.learning.split import data_days
from custom_components.powerguesser.engine.learning.worker import extract

DAY_MS = 86_400_000
END_MS = 1_793_491_200_000  # UTC midnight
STEP_MS = 30_000
GROSS_W = 2000
HEATER_W = 300
TASMOTA_W = 1600

ENTITY_GRID = "sensor.grid_power"
ENTITY_HEATER = "sensor.heater_plug_power"
ENTITY_TASMOTA = "sensor.tasmota_plug_power"


def _bounds(days: int) -> list[int]:
    start = END_MS - days * DAY_MS
    return [start + i * DAY_MS for i in range(days + 1)]


def _build(
    path: Path,
    days: int,
    *,
    heater_rows: list[tuple[int, int | None]] | None = None,
    tasmota: bool = False,
) -> dict[str, int]:
    """Write a buffer with a fully covered grid channel and the requested plugs."""
    start = END_MS - days * DAY_MS
    specs = [ChannelSpec(ENTITY_GRID, "grid", KIND_GROSS)]
    if heater_rows is not None:
        specs.append(ChannelSpec(ENTITY_HEATER, ROLE_METERED, KIND_METERED))
    if tasmota:
        specs.append(ChannelSpec(ENTITY_TASMOTA, ROLE_METERED, KIND_METERED))
    layout = [GrossLayoutEntry("grid", "direct", (ENTITY_GRID,))]

    db = BufferDB(str(path))
    db.open()
    id_map = db.ensure_channels(specs, layout)
    ids = {ENTITY_GRID: id_map[(ENTITY_GRID, "grid")]}

    rows: list[tuple[int, int, int | None]] = []
    # Gross: a row every 31 s from before the window to its end, never missing.
    ts = start - 31_000
    while ts < END_MS:
        rows.append((ts, ids[ENTITY_GRID], GROSS_W))
        ts += 31_000

    if heater_rows is not None:
        ids[ENTITY_HEATER] = id_map[(ENTITY_HEATER, ROLE_METERED)]
        rows.extend((t, ids[ENTITY_HEATER], w) for t, w in heater_rows)

    if tasmota:
        # 300 s cadence; the (premature) stale marker lands 90 s after every row.
        ids[ENTITY_TASMOTA] = id_map[(ENTITY_TASMOTA, ROLE_METERED)]
        ts = start - 293_000
        while ts < END_MS:
            rows.append((ts, ids[ENTITY_TASMOTA], TASMOTA_W))
            rows.append((ts + 90_000, ids[ENTITY_TASMOTA], None))
            ts += 300_000

    rows.sort(key=lambda r: (r[0], r[1]))
    db.write_batch(rows)
    db.close_clean()
    return ids


def _heater_outage_rows(start: int, null_ms: int, resume_ms: int) -> list[tuple[int, int | None]]:
    """Grid-aligned 60 s heater rows with one NULL at null_ms and none until resume_ms."""
    out: list[tuple[int, int | None]] = []
    ts = start - 60_000
    while ts < null_ms:
        out.append((ts, HEATER_W))
        ts += 60_000
    out.append((null_ms, None))
    ts = resume_ms
    while ts < END_MS:
        out.append((ts, HEATER_W))
        ts += 60_000
    return out


def _idx(start: int, ts: int) -> int:
    return (ts - start) // STEP_MS


def test_extract_finds_days_with_offline_and_stale_marked_plugs(tmp_path: Path) -> None:
    """Extract keeps every gross-covered day although plugs are offline or stale-marked.

    Before the fix the residual was NaN wherever any plug was NaN, so the 210 s of every
    300 s Tasmota cycle and the multi-day heater outage dropped every day and split_days
    raised 'need at least 2 data days, have 0'.
    """
    days = 4
    bounds = _bounds(days)
    start = bounds[0]
    buf = tmp_path / "buffer.sqlite"
    ids = _build(
        buf,
        days,
        heater_rows=_heater_outage_rows(start, start + DAY_MS // 2, start + 7 * DAY_MS // 2),
        tasmota=True,
    )

    job = {
        "schema": 1,
        "buffer_path": str(buf),
        "stage_dir": str(tmp_path / "stage"),
        "grid_s": 30,
        "day_bounds_ms": bounds,
        "first_weekday": 0,
        "holdout_days": 0,
        "train_fraction": 0.7,
        "enabled_event_types": ["step", "cycle_shape", "continuous"],
        "metered_channels": [ids[ENTITY_HEATER], ids[ENTITY_TASMOTA]],
        "phase_channels": {},
    }
    result = extract(job)

    split = result["meta"]["days"]
    assert split["dropped"] == []
    assert sorted(split["train"] + split["val"] + split["holdout"]) == list(range(days))


def test_short_null_holds_last_value_without_phantom_pulses(tmp_path: Path) -> None:
    """A NULL span shorter than 300 s holds the plug's last value in plug and residual."""
    days = 2
    bounds = _bounds(days)
    start = bounds[0]
    buf = tmp_path / "buffer.sqlite"
    ids = _build(buf, days, tasmota=True)
    tch = ids[ENTITY_TASMOTA]

    series = load_residual(str(buf), bounds[0], bounds[-1], STEP_MS, [tch], {})

    assert len(data_days(series.residual, start, STEP_MS, bounds)) == days
    # Held, never 0 W: a 0 W fill would leave a 1.6 kW, 210 s on / 90 s off pulse train.
    assert np.all(series.plugs[tch] == TASMOTA_W)
    assert np.all(series.residual == GROSS_W - TASMOTA_W)


def test_null_beyond_hold_limit_is_zero_in_residual_and_nan_in_plug(tmp_path: Path) -> None:
    """Boundary: held up to exactly 300 s after the NULL row, 0 W / NaN one step later."""
    days = 2
    bounds = _bounds(days)
    start = bounds[0]
    null_ms = start + 12 * 3_600_000
    resume_ms = start + DAY_MS + 6 * 3_600_000
    buf = tmp_path / "buffer.sqlite"
    ids = _build(buf, days, heater_rows=_heater_outage_rows(start, null_ms, resume_ms))
    hch = ids[ENTITY_HEATER]

    series = load_residual(str(buf), bounds[0], bounds[-1], STEP_MS, [hch], {})
    plug = series.plugs[hch]
    r = series.residual

    i_null = _idx(start, null_ms)
    assert plug[i_null - 1] == HEATER_W
    # Inside the hold: 0 s .. 300 s after the NULL row.
    held = slice(i_null, i_null + 11)
    assert np.all(plug[held] == HEATER_W)
    assert np.all(r[held] == GROSS_W - HEATER_W)
    # One step past the limit (330 s): residual counts the plug as 0 W, plug stays NaN.
    beyond = i_null + 11
    assert np.isnan(plug[beyond])
    assert r[beyond] == GROSS_W
    # Through the whole outage the residual stays finite and the plug NaN.
    outage = slice(beyond, _idx(start, resume_ms))
    assert np.all(np.isnan(plug[outage]))
    assert np.all(r[outage] == GROSS_W)
    # Reporting again after the outage.
    assert plug[_idx(start, resume_ms)] == HEATER_W
    assert len(data_days(r, start, STEP_MS, bounds)) == days


@pytest.mark.parametrize("step_ms", [30_000, 32_000])
def test_bursty_plug_with_stale_markers_is_fully_held(tmp_path: Path, step_ms: int) -> None:
    """A plug visible for 1.5 s, then NULL for 56.5 s, holds its last value at every grid point.

    The last known value is the last non-NULL row, not the last finite grid point: the
    grid almost never lands inside a burst (live: fridge plug, 0.7 s median, stale NULL
    at 2.5x median).
    """
    days = 1
    bounds = _bounds(days)
    start = bounds[0]
    burst_w = 120
    rows: list[tuple[int, int | None]] = []
    ts = start - 600_007
    while ts < END_MS:
        rows.append((ts, burst_w))
        rows.append((ts + 1_500, None))
        ts += 58_000
    buf = tmp_path / "buffer.sqlite"
    ids = _build(buf, days, heater_rows=rows)
    hch = ids[ENTITY_HEATER]

    series = load_residual(str(buf), bounds[0], bounds[-1], step_ms, [hch], {})

    assert np.all(series.plugs[hch] == burst_w)
    assert np.all(series.residual == GROSS_W - burst_w)


def test_null_run_hold_counts_from_first_null(tmp_path: Path) -> None:
    """Repeated NULL rows (unavailable re-reported every 30 s) do not restart the hold."""
    days = 1
    bounds = _bounds(days)
    start = bounds[0]
    null_ms = start + 6 * 3_600_000 + 7_000
    resume_ms = null_ms + 3_600_000
    rows = [r for r in _heater_outage_rows(start, null_ms, resume_ms)]
    rows.extend((t, None) for t in range(null_ms + 30_000, resume_ms, 30_000))
    buf = tmp_path / "buffer.sqlite"
    ids = _build(buf, days, heater_rows=sorted(rows))
    hch = ids[ENTITY_HEATER]

    series = load_residual(str(buf), bounds[0], bounds[-1], STEP_MS, [hch], {})
    plug = series.plugs[hch]
    grid = start + np.arange(len(plug)) * STEP_MS
    in_run = (grid >= null_ms) & (grid < resume_ms)
    held = in_run & (grid - null_ms <= 300_000)
    expired = in_run & (grid - null_ms > 300_000)

    assert held.sum() == 10 and expired.sum() > 100
    assert np.all(plug[held] == HEATER_W)
    assert np.all(np.isnan(plug[expired]))
    assert np.all(series.residual[expired] == GROSS_W)


def test_plug_offline_since_before_window_counts_zero(tmp_path: Path) -> None:
    """No last known value inside the window: residual counts 0 W, plug series all NaN."""
    days = 2
    bounds = _bounds(days)
    start = bounds[0]
    buf = tmp_path / "buffer.sqlite"
    ids = _build(
        buf,
        days,
        heater_rows=[(start - DAY_MS, HEATER_W), (start - DAY_MS + 60_000, None)],
    )
    hch = ids[ENTITY_HEATER]

    series = load_residual(str(buf), bounds[0], bounds[-1], STEP_MS, [hch], {})

    assert np.all(np.isnan(series.plugs[hch]))
    assert np.all(series.residual == GROSS_W)


@pytest.mark.parametrize("gap_s", [270, 300, 330])
def test_hold_limit_neighbours(tmp_path: Path, gap_s: int) -> None:
    """NULL gaps around the 300 s limit: the last NaN point is held iff it is <= 300 s after the NULL."""
    days = 1
    bounds = _bounds(days)
    start = bounds[0]
    null_ms = start + 6 * 3_600_000
    # The gap's last grid point lies gap_s after the NULL row.
    resume_ms = null_ms + gap_s * 1000 + STEP_MS
    buf = tmp_path / "buffer.sqlite"
    ids = _build(buf, days, heater_rows=_heater_outage_rows(start, null_ms, resume_ms))
    hch = ids[ENTITY_HEATER]

    series = load_residual(str(buf), bounds[0], bounds[-1], STEP_MS, [hch], {})
    last_nan = _idx(start, resume_ms) - 1

    if gap_s <= 300:
        assert series.plugs[hch][last_nan] == HEATER_W
        assert series.residual[last_nan] == GROSS_W - HEATER_W
    else:
        assert np.isnan(series.plugs[hch][last_nan])
        assert series.residual[last_nan] == GROSS_W


# --- Plan 06-09 (REV-04, D-25): window-bounded session subtraction ---

ENTITY_SESSION_PLUG = "sensor.session_plug_power"
PLUG_W = 400


def _build_session_buffer(path: Path, days: int) -> tuple[int, int]:
    """Write a buffer whose gross includes a 400 W session plug; return (grid_ch, plug_ch)."""
    start = END_MS - days * DAY_MS
    specs = [
        ChannelSpec(ENTITY_GRID, "grid", KIND_GROSS),
        ChannelSpec(ENTITY_SESSION_PLUG, ROLE_SESSION, KIND_SESSION),
    ]
    layout = [GrossLayoutEntry("grid", "direct", (ENTITY_GRID,))]
    db = BufferDB(str(path))
    db.open()
    id_map = db.ensure_channels(specs, layout)
    grid_ch = id_map[(ENTITY_GRID, "grid")]
    plug_ch = id_map[(ENTITY_SESSION_PLUG, ROLE_SESSION)]
    rows: list[tuple[int, int, int | None]] = []
    ts = start - 31_000
    while ts < END_MS:
        rows.append((ts, grid_ch, GROSS_W + PLUG_W))
        rows.append((ts, plug_ch, PLUG_W))
        ts += 31_000
    rows.sort(key=lambda r: (r[0], r[1]))
    db.write_batch(rows)
    db.close_clean()
    return grid_ch, plug_ch


def test_load_residual_session_windows_subtract_only_in_window(tmp_path: Path) -> None:
    """A session channel is subtracted inside its window only; outside is untouched."""
    days = 2
    bounds = _bounds(days)
    start = bounds[0]
    buf = tmp_path / "buffer.sqlite"
    _grid_ch, plug_ch = _build_session_buffer(buf, days)
    window = (start + 3_600_000, start + 3 * 3_600_000)

    series = load_residual(
        str(buf),
        start,
        bounds[-1],
        STEP_MS,
        [],
        {},
        session_windows={plug_ch: [window]},
    )
    plain = load_residual(str(buf), start, bounds[-1], STEP_MS, [], {})
    r = series.residual
    grid = start + np.arange(len(r)) * STEP_MS
    inside = (grid >= window[0]) & (grid < window[1])

    assert inside.any() and (~inside).any()
    # Inside: the 400 W plug is subtracted. Outside: byte-identical to no windows.
    assert np.all(r[inside] == GROSS_W)
    assert np.all(np.array_equal(r[~inside], plain.residual[~inside]))
    # The session series is NaN outside its window.
    masked = series.sessions[plug_ch]
    assert np.all(np.isnan(masked[~inside]))
    assert np.all(masked[inside] == PLUG_W)


def test_load_residual_metered_and_session_channel_subtracts_once(tmp_path: Path) -> None:
    """A channel both metered and windowed is subtracted once (the metered rule wins)."""
    days = 2
    bounds = _bounds(days)
    start = bounds[0]
    buf = tmp_path / "buffer.sqlite"
    _grid_ch, plug_ch = _build_session_buffer(buf, days)
    window = (start + 3_600_000, start + 3 * 3_600_000)

    series = load_residual(
        str(buf),
        start,
        bounds[-1],
        STEP_MS,
        [plug_ch],
        {},
        session_windows={plug_ch: [window]},
    )

    assert np.all(series.residual == GROSS_W)  # 400 subtracted exactly once
    assert plug_ch not in series.sessions


def test_load_residual_without_windows_is_unchanged(tmp_path: Path) -> None:
    """A plain call (no session_windows) leaves residual and sessions empty."""
    days = 1
    bounds = _bounds(days)
    buf = tmp_path / "buffer.sqlite"
    _build_session_buffer(buf, days)

    series = load_residual(str(buf), bounds[0], bounds[-1], STEP_MS, [], {})
    assert np.all(series.residual == GROSS_W + PLUG_W)
    assert series.sessions == {}

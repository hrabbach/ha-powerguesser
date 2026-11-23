"""Synthetic household buffer generator for testing."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np

from custom_components.powerguesser.engine.buffer import (
    KIND_GROSS,
    KIND_METERED,
    KIND_PHASE,
    KIND_SESSION,
    ROLE_METERED,
    ROLE_SESSION,
    BufferDB,
    ChannelSpec,
    GrossLayoutEntry,
)

ENTITY_GRID = "sensor.grid_power"
ENTITY_SOLAR = "sensor.solar_power"
ENTITY_BATTERY = "sensor.battery_power"
ENTITY_WASHER = "sensor.washer_power"
ENTITY_KETTLE = "sensor.kettle_power"
# Plan 06-09: a spare plug lent to the kettle for a session; it records the
# kettle's own 1 s signal on a session channel (no Energy prefs entry).
ENTITY_KETTLE_SESSION = "sensor.kettle_session_power"
ENTITY_L1 = "sensor.l1_power"
ENTITY_L2 = "sensor.l2_power"
ENTITY_L3 = "sensor.l3_power"


@dataclass(frozen=True, slots=True)
class SynthSpec:
    """Specification for synthetic power signal generation."""

    days: int = 22
    seed: int = 7
    base_w: float = 250.0
    noise_w: float = 15.0
    kettle_w: float = 2000.0
    kettle_s: int = 180
    kettle_per_day: tuple[int, int] = (2, 4)
    solar_peak_w: float = 3000.0
    interval_s: tuple[float, float] = (30.0, 33.0)
    oven: bool = False
    washer: bool = False
    washer_daily: bool = False
    kettle_during_washer: bool = False
    water_heater: bool = False
    water_heater_w: float = 3000.0
    dryer: bool = False
    dryer_element_w: float = 2400.0
    dryer_motor_w: float = 250.0
    plug_start_day: int = 0
    kettle_plug: bool = False
    # Plan 06-09 (REV-04): a plug lent to the kettle for a device-target
    # session. It records the kettle's own 1 s signal on a session channel so
    # the session can be validated against the confirmed device.
    kettle_session: bool = False
    spikes: bool = False
    gaps: bool = False
    phases: bool = False


@dataclass(frozen=True, slots=True)
class SynthData:
    """Generated synthetic power data and ground truth runs."""

    rows: dict[str, list[tuple[int, int | None]]]
    kettle_runs: list[tuple[int, int]]
    channel_ids: dict[str, int] = field(default_factory=dict)


def utc_day_bounds(end_ms: int, days: int) -> list[int]:
    """Return days + 1 UTC-midnight bounds ending at end_ms."""
    ms_per_day = 86_400_000
    start_ms = end_ms - days * ms_per_day
    return [start_ms + i * ms_per_day for i in range(days + 1)]


def generate(spec: SynthSpec, end_ms: int) -> SynthData:
    """Simulate household loads at 1 s resolution and sample with random intervals."""
    rng = np.random.default_rng(spec.seed)
    ms_per_day = 86_400_000
    start_ms = end_ms - spec.days * ms_per_day

    pad_s = 120
    total_s = spec.days * 86400 + 2 * pad_s
    t0_ms = start_ms - pad_s * 1000

    base_1s = np.full(total_s, spec.base_w, dtype=np.float64)
    base_1s += rng.normal(0, spec.noise_w, total_s)

    kettle_1s = np.zeros(total_s, dtype=np.float64)
    oven_1s = np.zeros(total_s, dtype=np.float64)
    washer_1s = np.zeros(total_s, dtype=np.float64)
    water_heater_1s = np.zeros(total_s, dtype=np.float64)
    dryer_1s = np.zeros(total_s, dtype=np.float64)

    # The Plan 04-06 SC2 generators draw only from their own RNG and only when
    # enabled, so every existing synthetic buffer stays byte-identical (T-04-31).
    rng2 = np.random.default_rng(spec.seed + 1009) if (spec.water_heater or spec.dryer) else None

    kettle_runs: list[tuple[int, int]] = []

    for d in range(spec.days):
        day_base_s = pad_s + d * 86400
        n_runs = rng.integers(spec.kettle_per_day[0], spec.kettle_per_day[1] + 1)
        chosen: list[int] = []
        attempts = 0
        while len(chosen) < n_runs and attempts < 100:
            attempts += 1
            if rng.random() < 0.5:
                window_start = 6 * 3600
                window_end = int(9.5 * 3600)
            else:
                window_start = 17 * 3600
                window_end = 21 * 3600

            sec_offset = int(rng.integers(window_start, window_end))
            if all(abs(sec_offset - c) >= spec.kettle_s + 60 for c in chosen):
                chosen.append(sec_offset)

        for sec_offset in sorted(chosen):
            run_start_s = day_base_s + sec_offset
            run_end_s = run_start_s + spec.kettle_s

            kettle_1s[run_start_s:run_end_s] += spec.kettle_w
            kettle_runs.append((t0_ms + run_start_s * 1000, t0_ms + run_end_s * 1000))

        # Oven: 2200 W thermostat cycling on evenings
        if spec.oven and rng.random() < 0.7:
            oven_start_s = day_base_s + rng.integers(18 * 3600, 19 * 3600)
            # 10 min preheat
            oven_1s[oven_start_s : oven_start_s + 600] = 2200.0
            curr_s = oven_start_s + 600
            total_oven_dur = rng.integers(30 * 60, 40 * 60)
            while curr_s < oven_start_s + total_oven_dur:
                on_s = rng.integers(120, 180)
                off_s = rng.integers(120, 240)
                oven_1s[curr_s : min(curr_s + on_s, oven_start_s + total_oven_dur)] = 2200.0
                curr_s += on_s + off_s

        # Washer: 2000 W heat 15 min, 200 W wash 40 min, 500 W spin 5 min, every 2-3 days
        # (washer_daily adds the odd days too, so a 7-day buffer holds one run per day
        # — Plan 06-07's daily-washer validation case; the default False leaves every
        # existing buffer byte-identical, plan 06-07 Task 1).
        if spec.washer and (spec.washer_daily or d % 2 == 0):
            wash_start_s = day_base_s + rng.integers(9 * 3600, 14 * 3600)
            washer_1s[wash_start_s : wash_start_s + 900] = 2000.0
            washer_1s[wash_start_s + 900 : wash_start_s + 3300] = 200.0
            washer_1s[wash_start_s + 3300 : wash_start_s + 3600] = 500.0
            # Plan 06-08: one kettle run starting 5 minutes into every washer
            # run's 200 W stage, on top of the regular kettle runs, so the two
            # plugs overlap by ground truth (SC2's kettle-during-wash case). It
            # needs the kettle plug channel so the overlay is also part of the
            # kettle's own series. Default False keeps every existing buffer
            # byte-identical.
            if spec.kettle_during_washer and spec.kettle_plug:
                overlay_start_s = wash_start_s + 900 + 300
                kettle_1s[overlay_start_s : overlay_start_s + spec.kettle_s] += spec.kettle_w
                kettle_runs.append(
                    (
                        t0_ms + overlay_start_s * 1000,
                        t0_ms + (overlay_start_s + spec.kettle_s) * 1000,
                    )
                )

        # Water heater (Plan 04-01 parity: 3000 W, 20-40 min): one run starting
        # 06:00-08:00 plus a second starting 19:00-21:00 on odd days.
        if spec.water_heater and rng2 is not None:
            wh_start_s = day_base_s + int(rng2.integers(6 * 3600, 8 * 3600))
            wh_dur_s = int(rng2.integers(20 * 60, 40 * 60))
            water_heater_1s[wh_start_s : wh_start_s + wh_dur_s] = spec.water_heater_w
            if d % 2 == 1:
                wh2_start_s = day_base_s + int(rng2.integers(19 * 3600, 21 * 3600))
                wh2_dur_s = int(rng2.integers(20 * 60, 40 * 60))
                water_heater_1s[wh2_start_s : wh2_start_s + wh2_dur_s] = spec.water_heater_w

        # Dryer (Plan 04-01 parity: 2400 W element cycling 2-5 min on / 1-3 min
        # off on a 250 W drum-motor step), every other day starting 10:00-15:00.
        if spec.dryer and rng2 is not None and (d % 2 == 0):
            motor_start_s = day_base_s + int(rng2.integers(10 * 3600, 15 * 3600))
            motor_dur_s = int(rng2.integers(60 * 60, 90 * 60))
            motor_end_s = motor_start_s + motor_dur_s
            dryer_1s[motor_start_s:motor_end_s] = spec.dryer_motor_w
            # Element cycles on top of the motor, from 2 min after its start
            # until 5 min before its end; only full cycles are emitted.
            elem_from_s = motor_start_s + 120
            elem_to_s = motor_end_s - 300
            curr_s = elem_from_s
            while curr_s + 120 <= elem_to_s:
                on_s = int(rng2.integers(2 * 60, 5 * 60))
                off_s = int(rng2.integers(1 * 60, 3 * 60))
                if curr_s + on_s > elem_to_s:
                    break
                dryer_1s[curr_s : curr_s + on_s] += spec.dryer_element_w
                curr_s += on_s + off_s

    load_1s = base_1s + kettle_1s + oven_1s + washer_1s + water_heater_1s + dryer_1s

    # Solar: 06:00 to 18:00 bell curve
    solar_1s = np.zeros(total_s, dtype=np.float64)
    for d in range(spec.days):
        day_base_s = pad_s + d * 86400
        sunrise = day_base_s + 6 * 3600
        sunset = day_base_s + 18 * 3600
        daylight_s = np.arange(sunrise, sunset)
        solar_1s[daylight_s] = spec.solar_peak_w * np.sin(
            np.pi * (daylight_s - sunrise) / (sunset - sunrise)
        )

    battery_1s = np.zeros(total_s, dtype=np.float64)
    grid_1s = load_1s - solar_1s

    # Phases: kettle on L1, oven and dryer split over L2/L3, water heater on L3,
    # base split evenly
    l1_1s = base_1s / 3.0 + kettle_1s
    l2_1s = base_1s / 3.0 + (oven_1s / 2.0) + (washer_1s / 2.0) + (dryer_1s / 2.0)
    l3_1s = (
        base_1s / 3.0
        + (oven_1s / 2.0)
        + (washer_1s / 2.0)
        + (dryer_1s / 2.0)
        + water_heater_1s
    )

    rows: dict[str, list[tuple[int, int | None]]] = {
        ENTITY_GRID: [],
        ENTITY_SOLAR: [],
        ENTITY_BATTERY: [],
    }
    signals: dict[str, np.ndarray] = {
        ENTITY_GRID: grid_1s,
        ENTITY_SOLAR: solar_1s,
        ENTITY_BATTERY: battery_1s,
    }

    if spec.washer:
        signals[ENTITY_WASHER] = washer_1s
        rows[ENTITY_WASHER] = []

    if spec.kettle_plug:
        signals[ENTITY_KETTLE] = kettle_1s
        rows[ENTITY_KETTLE] = []

    if spec.phases:
        signals[ENTITY_L1] = l1_1s
        signals[ENTITY_L2] = l2_1s
        signals[ENTITY_L3] = l3_1s
        rows[ENTITY_L1] = []
        rows[ENTITY_L2] = []
        rows[ENTITY_L3] = []

    # Plan 06-09: the kettle session plug records the kettle's 1 s signal,
    # sampled like every other channel. Added after every existing entity so a
    # buffer without kettle_session stays byte-identical.
    if spec.kettle_session:
        signals[ENTITY_KETTLE_SESSION] = kettle_1s
        rows[ENTITY_KETTLE_SESSION] = []

    for entity, sig in signals.items():
        ch_rows: list[tuple[int, int | None]] = []
        curr_t_ms = t0_ms
        prev_idx = 0

        entity_start_ms = t0_ms
        if entity == ENTITY_WASHER and spec.plug_start_day > 0:
            entity_start_ms = start_ms + spec.plug_start_day * ms_per_day

        while curr_t_ms < end_ms + pad_s * 1000:
            dt_s = float(rng.uniform(spec.interval_s[0], spec.interval_s[1]))
            next_t_ms = curr_t_ms + int(round(dt_s * 1000))
            curr_idx = min(total_s, max(prev_idx + 1, int(round((next_t_ms - t0_ms) / 1000))))
            if curr_idx <= prev_idx:
                # Simulated signal exhausted (sampling jitter overran the end pad):
                # stop instead of averaging an empty slice into NaN.
                break

            if next_t_ms >= entity_start_ms:
                mean_val = float(np.mean(sig[prev_idx:curr_idx]))
                ch_rows.append((next_t_ms, int(round(mean_val))))

            curr_t_ms = next_t_ms
            prev_idx = curr_idx

        # Alignment spikes on grid: 5 per day, one-sample +400 W excursion
        if spec.spikes and entity == ENTITY_GRID:
            for d in range(spec.days):
                day_start_t = start_ms + d * ms_per_day
                day_end_t = day_start_t + ms_per_day
                day_indices = [
                    idx
                    for idx, (t, _) in enumerate(ch_rows)
                    if day_start_t <= t < day_end_t and idx < len(ch_rows) - 1
                ]
                if len(day_indices) >= 5:
                    chosen_indices = rng.choice(day_indices, size=5, replace=False)
                    for c_idx in chosen_indices:
                        t, w = ch_rows[c_idx]
                        if w is not None:
                            ch_rows[c_idx] = (t, w + 400)

        # 2-hour NULL gap on day 3 for grid
        if spec.gaps and entity == ENTITY_GRID and spec.days > 3:
            gap_start_ms = start_ms + 3 * ms_per_day + 12 * 3600_000
            gap_end_ms = gap_start_ms + 2 * 3600_000
            filtered: list[tuple[int, int | None]] = []
            gap_marker_inserted = False
            for t, w in ch_rows:
                if t < gap_start_ms:
                    filtered.append((t, w))
                elif gap_start_ms <= t < gap_end_ms:
                    if not gap_marker_inserted:
                        filtered.append((gap_start_ms, None))
                        gap_marker_inserted = True
                else:
                    filtered.append((t, w))
            ch_rows = filtered

        rows[entity] = ch_rows

    return SynthData(rows=rows, kettle_runs=kettle_runs)


def build_buffer(path: str | Path, spec: SynthSpec, end_ms: int) -> SynthData:
    """Build a SQLite buffer file with synthetic household data."""
    db = BufferDB(str(path))
    db.open()
    data = generate(spec, end_ms)

    specs = [
        ChannelSpec(ENTITY_GRID, "grid", KIND_GROSS),
        ChannelSpec(ENTITY_SOLAR, "solar", KIND_GROSS),
        ChannelSpec(ENTITY_BATTERY, "battery", KIND_GROSS),
    ]
    layout = [
        GrossLayoutEntry("grid", "direct", (ENTITY_GRID,)),
        GrossLayoutEntry("solar", "direct", (ENTITY_SOLAR,)),
        GrossLayoutEntry("battery", "direct", (ENTITY_BATTERY,)),
    ]

    if spec.washer:
        specs.append(ChannelSpec(ENTITY_WASHER, ROLE_METERED, KIND_METERED))

    if spec.kettle_plug:
        specs.append(ChannelSpec(ENTITY_KETTLE, ROLE_METERED, KIND_METERED))

    if spec.phases:
        specs.append(ChannelSpec(ENTITY_L1, "l1", KIND_PHASE))
        specs.append(ChannelSpec(ENTITY_L2, "l2", KIND_PHASE))
        specs.append(ChannelSpec(ENTITY_L3, "l3", KIND_PHASE))

    if spec.kettle_session:
        specs.append(
            ChannelSpec(ENTITY_KETTLE_SESSION, ROLE_SESSION, KIND_SESSION)
        )

    id_map = db.ensure_channels(specs, layout)

    channel_ids = {
        ENTITY_GRID: id_map[(ENTITY_GRID, "grid")],
        ENTITY_SOLAR: id_map[(ENTITY_SOLAR, "solar")],
        ENTITY_BATTERY: id_map[(ENTITY_BATTERY, "battery")],
    }
    if spec.washer:
        channel_ids[ENTITY_WASHER] = id_map[(ENTITY_WASHER, ROLE_METERED)]
    if spec.kettle_plug:
        channel_ids[ENTITY_KETTLE] = id_map[(ENTITY_KETTLE, ROLE_METERED)]
    if spec.phases:
        channel_ids[ENTITY_L1] = id_map[(ENTITY_L1, "l1")]
        channel_ids[ENTITY_L2] = id_map[(ENTITY_L2, "l2")]
        channel_ids[ENTITY_L3] = id_map[(ENTITY_L3, "l3")]
    if spec.kettle_session:
        channel_ids[ENTITY_KETTLE_SESSION] = id_map[
            (ENTITY_KETTLE_SESSION, ROLE_SESSION)
        ]

    all_rows: list[tuple[int, int, int | None]] = []
    for entity, rows in data.rows.items():
        if entity in channel_ids:
            ch_id = channel_ids[entity]
            for ts, w in rows:
                all_rows.append((ts, ch_id, w))

    all_rows.sort(key=lambda r: r[0])
    chunk_size = 20_000
    for i in range(0, len(all_rows), chunk_size):
        db.write_batch(all_rows[i : i + chunk_size])

    db.close_clean()
    return SynthData(rows=data.rows, kettle_runs=data.kettle_runs, channel_ids=channel_ids)


def fill_buffer(
    db: BufferDB,
    channel_ids: Mapping[str, int],
    spec: SynthSpec,
    end_ms: int,
) -> SynthData:
    """Fill an open BufferDB with synthetic data for the given channel mapping."""
    data = generate(spec, end_ms)
    all_rows: list[tuple[int, int, int | None]] = []
    for entity, rows in data.rows.items():
        if entity in channel_ids:
            ch_id = channel_ids[entity]
            for ts, w in rows:
                all_rows.append((ts, ch_id, w))

    all_rows.sort(key=lambda r: r[0])
    chunk_size = 20_000
    for i in range(0, len(all_rows), chunk_size):
        db.write_batch(all_rows[i : i + chunk_size])

    return SynthData(rows=data.rows, kettle_runs=data.kettle_runs, channel_ids=dict(channel_ids))

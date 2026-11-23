"""Tests for PowerGuesser recorder bootstrap import."""

import asyncio
from datetime import timedelta
from typing import Any
from unittest.mock import AsyncMock, patch
import pytest

from homeassistant.components.energy.data import EnergyManager
from homeassistant.components.recorder import get_instance
from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util
from pytest_homeassistant.common import async_fire_time_changed
from pytest_homeassistant.components.recorder.common import async_wait_recording_done

import custom_components.powerguesser.bootstrap as bootstrap_module
from custom_components.powerguesser.bootstrap import (
    DONE_IMPORTED,
    DONE_INVALID_UNIT,
    BOOTSTRAP_DAY_MS,
    BootstrapRunner,
    _query,
)
from custom_components.powerguesser.const import (
    BOOTSTRAP_START_TIMEOUT_S,
    BUFFER_DIR,
    BUFFER_FILE,
)
from custom_components.powerguesser.engine.reader import open_readonly
from tests.conftest import async_set_power, async_setup_via_flow


async def test_bootstrap_disabled_without_marker(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
) -> None:
    """Without bootstrap_enabled marker, conftest switch disables bootstrap."""
    energy_manager.data = std_prefs
    async_set_power(hass, "sensor.grid_power", 500)
    async_set_power(hass, "sensor.solar_power", 1000)
    async_set_power(hass, "sensor.battery_power", 200)
    async_set_power(hass, "sensor.washer_power", 50)

    entry = await async_setup_via_flow(hass)
    hub = entry.runtime_data

    async_fire_time_changed(
        hass, dt_util.utcnow() + timedelta(seconds=BOOTSTRAP_START_TIMEOUT_S + 1)
    )
    await hass.async_block_till_done()

    assert hub.bootstrap_task is None
    assert hub.bootstrap_coordinator.data["status"] is None


@pytest.mark.bootstrap_enabled
async def test_bootstrap_fills_buffer(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    freezer: Any,
) -> None:
    """Tracer test: recorder history is imported into buffer up to live_start."""
    energy_manager.data = std_prefs
    now = dt_util.utcnow()
    start_time = now - timedelta(days=3)
    freezer.move_to(start_time)

    for h in range(72):
        if 20 <= h < 22:
            async_set_power(hass, "sensor.grid_power", "unavailable")
        else:
            async_set_power(hass, "sensor.grid_power", 500 + h)
        async_set_power(hass, "sensor.solar_power", 1000 + h)
        async_set_power(hass, "sensor.battery_power", 200 + h)
        async_set_power(hass, "sensor.washer_power", 50 + h)
        freezer.tick(timedelta(hours=1))

    await async_wait_recording_done(hass)

    freezer.move_to(now)
    async_set_power(hass, "sensor.grid_power", 600)
    async_set_power(hass, "sensor.solar_power", 1500)
    async_set_power(hass, "sensor.battery_power", 100)
    async_set_power(hass, "sensor.washer_power", 75)

    entry = await async_setup_via_flow(hass)
    hub = entry.runtime_data
    assert hub.bootstrap_task is not None

    # Gate starts in waiting status
    assert hub.bootstrap_coordinator.data["status"] == "waiting"

    # Release gate via timeout timer
    async_fire_time_changed(
        hass, dt_util.utcnow() + timedelta(seconds=BOOTSTRAP_START_TIMEOUT_S + 1)
    )
    await hub.bootstrap_task

    assert hub.bootstrap_coordinator.data["status"] == "done"

    conn = open_readonly(hass.config.path(BUFFER_DIR, BUFFER_FILE))
    cursor = conn.execute("SELECT id, entity_id FROM channels")
    ch_map = {row[1]: row[0] for row in cursor.fetchall()}

    for entity_id in [
        "sensor.grid_power",
        "sensor.solar_power",
        "sensor.battery_power",
        "sensor.washer_power",
    ]:
        ch = ch_map[entity_id]
        done_val = await hass.async_add_executor_job(
            hub.writer.db.get_meta, f"bootstrap_done:{ch}"
        )
        assert done_val == DONE_IMPORTED

        live_start_str = await hass.async_add_executor_job(
            hub.writer.db.get_meta, f"live_start_ms:{ch}"
        )
        assert live_start_str is not None
        live_start_ms = int(live_start_str)

        rows = conn.execute(
            "SELECT ts, w FROM samples WHERE ch = ? AND ts < ? ORDER BY ts ASC",
            (ch, live_start_ms),
        ).fetchall()
        assert len(rows) > 0, f"No bootstrap rows for {entity_id}"

        if entity_id == "sensor.grid_power":
            null_rows = [r for r in rows if r[1] is None]
            assert len(null_rows) > 0
            unavail_start_ms = int(
                round((start_time + timedelta(hours=20)).timestamp() * 1000)
            )
            unavail_end_ms = int(
                round((start_time + timedelta(hours=22)).timestamp() * 1000)
            )
            stretch_rows = [
                r for r in rows if unavail_start_ms <= r[0] < unavail_end_ms
            ]
            assert len(stretch_rows) > 0
            assert all(r[1] is None for r in stretch_rows)


@pytest.mark.bootstrap_enabled
@pytest.mark.parametrize(
    "recorder_config",
    [{"exclude": {"entities": ["sensor.washer_power"]}}],
)
async def test_excluded_channel_imports_nothing(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    freezer: Any,
) -> None:
    """Recorder-excluded channel produces no rows and is marked done."""
    energy_manager.data = std_prefs
    now = dt_util.utcnow()
    start_time = now - timedelta(days=2)
    freezer.move_to(start_time)

    for h in range(48):
        async_set_power(hass, "sensor.grid_power", 500)
        async_set_power(hass, "sensor.washer_power", 50)
        freezer.tick(timedelta(hours=1))

    await async_wait_recording_done(hass)

    freezer.move_to(now)
    async_set_power(hass, "sensor.grid_power", 500)
    async_set_power(hass, "sensor.solar_power", 0)
    async_set_power(hass, "sensor.battery_power", 0)
    async_set_power(hass, "sensor.washer_power", 50)

    entry = await async_setup_via_flow(hass)
    hub = entry.runtime_data
    assert hub.bootstrap_task is not None

    async_fire_time_changed(
        hass, dt_util.utcnow() + timedelta(seconds=BOOTSTRAP_START_TIMEOUT_S + 1)
    )
    await hub.bootstrap_task

    assert hub.bootstrap_coordinator.data["status"] == "done"

    conn = open_readonly(hass.config.path(BUFFER_DIR, BUFFER_FILE))
    cursor = conn.execute("SELECT id, entity_id FROM channels")
    ch_map = {row[1]: row[0] for row in cursor.fetchall()}

    washer_ch = ch_map["sensor.washer_power"]
    done_val = await hass.async_add_executor_job(
        hub.writer.db.get_meta, f"bootstrap_done:{washer_ch}"
    )
    assert done_val == DONE_IMPORTED

    live_start_str = await hass.async_add_executor_job(
        hub.writer.db.get_meta, f"live_start_ms:{washer_ch}"
    )
    assert live_start_str is not None
    live_start_ms = int(live_start_str)

    rows = conn.execute(
        "SELECT ts, w FROM samples WHERE ch = ? AND ts < ?",
        (washer_ch, live_start_ms),
    ).fetchall()
    assert len(rows) == 0


@pytest.mark.bootstrap_enabled
async def test_invalid_unit_channel_marked_terminal(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    freezer: Any,
) -> None:
    """A channel with unconvertible unit is marked invalid_unit and never retried."""
    energy_manager.data = std_prefs
    now = dt_util.utcnow()
    start_time = now - timedelta(days=1)
    freezer.move_to(start_time)

    for h in range(24):
        async_set_power(hass, "sensor.grid_power", 500)
        async_set_power(hass, "sensor.washer_power", 50)
        freezer.tick(timedelta(hours=1))

    await async_wait_recording_done(hass)

    freezer.move_to(now)
    async_set_power(hass, "sensor.grid_power", 500)
    async_set_power(hass, "sensor.solar_power", 0)
    async_set_power(hass, "sensor.battery_power", 0)
    async_set_power(hass, "sensor.washer_power", 50)

    entry = await async_setup_via_flow(hass)
    hub = entry.runtime_data
    assert hub.bootstrap_task is not None

    # Before releasing the gate, re-set washer with unconvertible unit 'VA'
    async_set_power(hass, "sensor.washer_power", 50, unit="VA")

    async_fire_time_changed(
        hass, dt_util.utcnow() + timedelta(seconds=BOOTSTRAP_START_TIMEOUT_S + 1)
    )
    await hub.bootstrap_task

    assert hub.bootstrap_coordinator.data["status"] == "done"

    conn = open_readonly(hass.config.path(BUFFER_DIR, BUFFER_FILE))
    cursor = conn.execute("SELECT id, entity_id FROM channels")
    ch_map = {row[1]: row[0] for row in cursor.fetchall()}

    washer_ch = ch_map["sensor.washer_power"]
    done_val = await hass.async_add_executor_job(
        hub.writer.db.get_meta, f"bootstrap_done:{washer_ch}"
    )
    assert done_val == DONE_INVALID_UNIT

    live_start_str = await hass.async_add_executor_job(
        hub.writer.db.get_meta, f"live_start_ms:{washer_ch}"
    )
    assert live_start_str is not None
    live_start_ms = int(live_start_str)

    rows = conn.execute(
        "SELECT ts, w FROM samples WHERE ch = ? AND ts < ?",
        (washer_ch, live_start_ms),
    ).fetchall()
    assert len(rows) == 0

    # Second run: verify no recorder query is issued for sensor.washer_power
    with patch.object(
        bootstrap_module,
        "_query",
        wraps=bootstrap_module._query,
    ) as mock_query:
        await BootstrapRunner(hass, hub).async_run()
        queried = [call.args[1] for call in mock_query.call_args_list]
        assert "sensor.washer_power" not in queried


@pytest.mark.bootstrap_enabled
async def test_silent_plug_imported(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    freezer: Any,
) -> None:
    """A silent plug that never changes gets live_start_ms from activation and imports history."""
    energy_manager.data = std_prefs
    now = dt_util.utcnow()
    start_time = now - timedelta(days=2)
    freezer.move_to(start_time)

    for h in range(48):
        async_set_power(hass, "sensor.grid_power", 500)
        async_set_power(hass, "sensor.washer_power", 50)
        freezer.tick(timedelta(hours=1))

    await async_wait_recording_done(hass)

    freezer.move_to(now)
    async_set_power(hass, "sensor.grid_power", 500)
    async_set_power(hass, "sensor.solar_power", 0)
    async_set_power(hass, "sensor.battery_power", 0)
    # Washer state remains 50 W and never updates again
    async_set_power(hass, "sensor.washer_power", 50)

    entry = await async_setup_via_flow(hass)
    hub = entry.runtime_data

    async_fire_time_changed(
        hass, dt_util.utcnow() + timedelta(seconds=BOOTSTRAP_START_TIMEOUT_S + 1)
    )
    await hub.bootstrap_task

    conn = open_readonly(hass.config.path(BUFFER_DIR, BUFFER_FILE))
    cursor = conn.execute("SELECT id, entity_id FROM channels")
    ch_map = {row[1]: row[0] for row in cursor.fetchall()}
    washer_ch = ch_map["sensor.washer_power"]

    live_start_str = await hass.async_add_executor_job(
        hub.writer.db.get_meta, f"live_start_ms:{washer_ch}"
    )
    assert live_start_str is not None
    assert (
        await hass.async_add_executor_job(
            hub.writer.db.get_meta, f"bootstrap_done:{washer_ch}"
        )
        == DONE_IMPORTED
    )

    rows = conn.execute(
        "SELECT ts, w FROM samples WHERE ch = ? AND ts < ?",
        (washer_ch, int(live_start_str)),
    ).fetchall()
    assert len(rows) > 0


@pytest.mark.bootstrap_enabled
async def test_bootstrap_gate_releases_on_gross_deltas_before_timeout(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    freezer: Any,
) -> None:
    """Gross channels changing every 30 s reach 20 deltas and open the gate before 900 s."""
    energy_manager.data = std_prefs
    now = dt_util.utcnow()
    freezer.move_to(now)

    async_set_power(hass, "sensor.grid_power", 500)
    async_set_power(hass, "sensor.solar_power", 1000)
    async_set_power(hass, "sensor.battery_power", 200)
    async_set_power(hass, "sensor.washer_power", 50)

    entry = await async_setup_via_flow(hass)
    hub = entry.runtime_data
    assert hub.bootstrap_coordinator.data["status"] == "waiting"

    # Drive 20 state changes for grid, solar, and battery every 30 seconds
    # (21 samples yield 20 deltas)
    for step in range(1, 22):
        freezer.tick(timedelta(seconds=30))
        async_fire_time_changed(hass, dt_util.utcnow())
        async_set_power(hass, "sensor.grid_power", 500 + step * 10)
        async_set_power(hass, "sensor.solar_power", 1000 + step * 10)
        async_set_power(hass, "sensor.battery_power", 200 + step * 10)
        await hass.async_block_till_done()

    # Gate should have opened because all gross profilers reached >= 20 deltas
    # Elapsed time is 20 * 30 = 600 s < 900 s
    assert hub._bootstrap_gate.is_set()
    await hub.bootstrap_task
    assert hub.bootstrap_coordinator.data["status"] == "done"


@pytest.mark.bootstrap_enabled
async def test_bootstrap_gate_waits_for_timeout_when_solar_flat(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    freezer: Any,
) -> None:
    """When a gross channel stays at 0 W (deltas not counted), gate waits for 900 s timeout."""
    energy_manager.data = std_prefs
    now = dt_util.utcnow()
    freezer.move_to(now)

    async_set_power(hass, "sensor.grid_power", 500)
    async_set_power(hass, "sensor.solar_power", 0)
    async_set_power(hass, "sensor.battery_power", 200)
    async_set_power(hass, "sensor.washer_power", 50)

    entry = await async_setup_via_flow(hass)
    hub = entry.runtime_data
    assert hub.bootstrap_coordinator.data["status"] == "waiting"

    # Grid and battery change 20 times, but solar stays at 0 W (no non-zero deltas)
    for step in range(1, 21):
        freezer.tick(timedelta(seconds=30))
        async_fire_time_changed(hass, dt_util.utcnow())
        async_set_power(hass, "sensor.grid_power", 500 + step * 10)
        async_set_power(hass, "sensor.battery_power", 200 + step * 10)
        await hass.async_block_till_done()

    # Solar stayed at 0 W, so delta_count < 20 -> gate is still unset at 600 s!
    assert not hub._bootstrap_gate.is_set()
    assert hub.bootstrap_coordinator.data["status"] == "waiting"

    # Advance time past 900 s
    freezer.tick(timedelta(seconds=301))
    async_fire_time_changed(hass, dt_util.utcnow())
    await hub.bootstrap_task

    assert hub.bootstrap_coordinator.data["status"] == "done"


@pytest.mark.bootstrap_enabled
async def test_chunk_boundary_gap_reconstruction(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    freezer: Any,
) -> None:
    """A re-reporting gap spanning a chunk boundary produces its NULL row via chunk carry."""
    energy_manager.data = std_prefs
    now = dt_util.utcnow()
    start_time = now - timedelta(days=2)
    freezer.move_to(start_time)

    # State before boundary: at start_time + 23h 58m
    freezer.tick(timedelta(hours=23, minutes=58))
    async_set_power(hass, "sensor.grid_power", 500)
    # Re-report at 23h 59m
    freezer.tick(timedelta(minutes=1))
    async_set_power(hass, "sensor.grid_power", 500)

    # Next state is at 24h 05m (5 minutes past boundary, exceeding 3x 30s = 90s)
    freezer.tick(timedelta(minutes=6))
    async_set_power(hass, "sensor.grid_power", 600)

    await async_wait_recording_done(hass)

    freezer.move_to(now)
    async_set_power(hass, "sensor.grid_power", 600)
    async_set_power(hass, "sensor.solar_power", 0)
    async_set_power(hass, "sensor.battery_power", 0)
    async_set_power(hass, "sensor.washer_power", 0)

    entry = await async_setup_via_flow(hass)
    hub = entry.runtime_data
    # Force grid mode to re_reporting so it generates gap markers
    hub.modes["sensor.grid_power"].re_reporting = True
    if "sensor.grid_power" in hub.profilers:
        hub.profilers["sensor.grid_power"]._published_median = 30.0

    async_fire_time_changed(
        hass, dt_util.utcnow() + timedelta(seconds=BOOTSTRAP_START_TIMEOUT_S + 1)
    )
    await hub.bootstrap_task

    conn = open_readonly(hass.config.path(BUFFER_DIR, BUFFER_FILE))
    cursor = conn.execute("SELECT id FROM channels WHERE entity_id = 'sensor.grid_power'")
    grid_ch = cursor.fetchone()[0]

    live_start_str = await hass.async_add_executor_job(
        hub.writer.db.get_meta, f"live_start_ms:{grid_ch}"
    )
    rows = conn.execute(
        "SELECT ts, w FROM samples WHERE ch = ? AND ts < ? ORDER BY ts ASC",
        (grid_ch, int(live_start_str)),
    ).fetchall()

    null_rows = [r for r in rows if r[1] is None]
    assert len(null_rows) > 0


@pytest.mark.bootstrap_enabled
async def test_metered_plug_bootstrap_has_no_synthesized_gap(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    freezer: Any,
) -> None:
    """A re-reporting metered plug silent past 3x its interval imports with no NULL row.

    P1 D-07: plugs are held with no age limit unless unavailable. The bootstrap used
    lr + 3x interval (30 s fallback) and so blanked 300 s Tasmota plugs 90 s after
    every report.
    """
    energy_manager.data = std_prefs
    now = dt_util.utcnow()
    freezer.move_to(now - timedelta(days=1))

    async_set_power(hass, "sensor.washer_power", 200)
    freezer.tick(timedelta(minutes=1))
    async_set_power(hass, "sensor.washer_power", 200)
    # Next change 5 min after the re-report, well past lr + 3x 30 s
    freezer.tick(timedelta(minutes=5))
    async_set_power(hass, "sensor.washer_power", 1600)
    freezer.tick(timedelta(minutes=5))
    async_set_power(hass, "sensor.washer_power", 1600)

    await async_wait_recording_done(hass)

    freezer.move_to(now)
    async_set_power(hass, "sensor.grid_power", 600)
    async_set_power(hass, "sensor.solar_power", 0)
    async_set_power(hass, "sensor.battery_power", 0)
    async_set_power(hass, "sensor.washer_power", 1600)

    entry = await async_setup_via_flow(hass)
    hub = entry.runtime_data
    hub.modes["sensor.washer_power"].re_reporting = True
    if "sensor.washer_power" in hub.profilers:
        hub.profilers["sensor.washer_power"]._published_median = 30.0

    async_fire_time_changed(
        hass, dt_util.utcnow() + timedelta(seconds=BOOTSTRAP_START_TIMEOUT_S + 1)
    )
    await hub.bootstrap_task

    conn = open_readonly(hass.config.path(BUFFER_DIR, BUFFER_FILE))
    washer_ch = conn.execute(
        "SELECT id FROM channels WHERE entity_id = 'sensor.washer_power'"
    ).fetchone()[0]
    live_start_str = await hass.async_add_executor_job(
        hub.writer.db.get_meta, f"live_start_ms:{washer_ch}"
    )
    rows = conn.execute(
        "SELECT ts, w FROM samples WHERE ch = ? AND ts < ? ORDER BY ts ASC",
        (washer_ch, int(live_start_str)),
    ).fetchall()
    conn.close()

    assert [w for _, w in rows if w is not None] != []
    assert [r for r in rows if r[1] is None] == []


@pytest.mark.bootstrap_enabled
async def test_bootstrap_resumes_from_cursor_after_unload(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    freezer: Any,
) -> None:
    """Interrupted import resumes from stored cursor and matches uninterrupted run."""
    energy_manager.data = std_prefs
    now = dt_util.utcnow()
    start_time = now - timedelta(days=3)
    freezer.move_to(start_time)

    for h in range(72):
        async_set_power(hass, "sensor.grid_power", 500 + h)
        async_set_power(hass, "sensor.solar_power", 1000 + h)
        async_set_power(hass, "sensor.battery_power", 200 + h)
        async_set_power(hass, "sensor.washer_power", 50 + h)
        freezer.tick(timedelta(hours=1))

    await async_wait_recording_done(hass)

    freezer.move_to(now)
    async_set_power(hass, "sensor.grid_power", 600)
    async_set_power(hass, "sensor.solar_power", 1500)
    async_set_power(hass, "sensor.battery_power", 100)
    async_set_power(hass, "sensor.washer_power", 75)

    entry = await async_setup_via_flow(hass)
    hub = entry.runtime_data

    conn = open_readonly(hass.config.path(BUFFER_DIR, BUFFER_FILE))
    cursor = conn.execute("SELECT id FROM channels WHERE entity_id = 'sensor.grid_power'")
    grid_ch = cursor.fetchone()[0]

    # Run full uninterrupted bootstrap to completion
    async_fire_time_changed(
        hass, dt_util.utcnow() + timedelta(seconds=BOOTSTRAP_START_TIMEOUT_S + 1)
    )
    await hub.bootstrap_task

    live_start_str = await hass.async_add_executor_job(
        hub.writer.db.get_meta, f"live_start_ms:{grid_ch}"
    )
    uninterrupted_rows = conn.execute(
        "SELECT ts, w FROM samples WHERE ch = ? AND ts < ? ORDER BY ts ASC",
        (grid_ch, int(live_start_str)),
    ).fetchall()

    # Now clear bootstrap_done and set cursor back by 1 day to test resume
    stored_cursor = int(
        await hass.async_add_executor_job(
            hub.writer.db.get_meta, f"bootstrap_cursor_ms:{grid_ch}"
        )
    )
    one_day_back = stored_cursor - BOOTSTRAP_DAY_MS
    await hass.async_add_executor_job(
        hub.writer.db.set_meta, f"bootstrap_cursor_ms:{grid_ch}", str(one_day_back)
    )
    await hass.async_add_executor_job(
        hub.writer.db.set_meta, f"bootstrap_done:{grid_ch}", ""
    )

    # Resume
    runner = BootstrapRunner(hass, hub)
    await runner.async_run()

    resumed_rows = conn.execute(
        "SELECT ts, w FROM samples WHERE ch = ? AND ts < ? ORDER BY ts ASC",
        (grid_ch, int(live_start_str)),
    ).fetchall()

    assert len(resumed_rows) == len(uninterrupted_rows)
    assert resumed_rows == uninterrupted_rows


@pytest.mark.bootstrap_enabled
async def test_late_channel_imported_on_rebuild(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    freezer: Any,
) -> None:
    """Adding a device_consumption plug after initial import imports that plug's history."""
    initial_prefs = {
        "energy_sources": std_prefs["energy_sources"],
        "device_consumption": [],
    }
    energy_manager.data = initial_prefs

    now = dt_util.utcnow()
    start_time = now - timedelta(days=2)
    freezer.move_to(start_time)

    for h in range(48):
        async_set_power(hass, "sensor.grid_power", 500)
        async_set_power(hass, "sensor.washer_power", 50)
        freezer.tick(timedelta(hours=1))

    await async_wait_recording_done(hass)

    freezer.move_to(now)
    async_set_power(hass, "sensor.grid_power", 500)
    async_set_power(hass, "sensor.solar_power", 0)
    async_set_power(hass, "sensor.battery_power", 0)
    async_set_power(hass, "sensor.washer_power", 50)

    entry = await async_setup_via_flow(hass)
    hub = entry.runtime_data

    async_fire_time_changed(
        hass, dt_util.utcnow() + timedelta(seconds=BOOTSTRAP_START_TIMEOUT_S + 1)
    )
    await hub.bootstrap_task
    assert hub.bootstrap_coordinator.data["status"] == "done"

    # Now add washer to device_consumption
    energy_manager.data = std_prefs
    await hub.async_rebuild()

    assert hub.bootstrap_task is not None
    await hub.bootstrap_task

    conn = open_readonly(hass.config.path(BUFFER_DIR, BUFFER_FILE))
    cursor = conn.execute("SELECT id FROM channels WHERE entity_id = 'sensor.washer_power'")
    washer_ch = cursor.fetchone()[0]

    assert (
        await hass.async_add_executor_job(
            hub.writer.db.get_meta, f"bootstrap_done:{washer_ch}"
        )
        == DONE_IMPORTED
    )

    live_start_str = await hass.async_add_executor_job(
        hub.writer.db.get_meta, f"live_start_ms:{washer_ch}"
    )
    rows = conn.execute(
        "SELECT ts, w FROM samples WHERE ch = ? AND ts < ?",
        (washer_ch, int(live_start_str)),
    ).fetchall()
    assert len(rows) > 0


@pytest.mark.bootstrap_enabled
async def test_bootstrap_does_not_feed_profiler_and_recorder_is_read_only(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    freezer: Any,
) -> None:
    """Profiler delta count is unchanged by import (D-21), and recorder history is unmodified."""
    energy_manager.data = std_prefs
    now = dt_util.utcnow()
    start_time = now - timedelta(days=2)
    freezer.move_to(start_time)

    for h in range(48):
        async_set_power(hass, "sensor.grid_power", 500 + h)
        freezer.tick(timedelta(hours=1))

    await async_wait_recording_done(hass)

    freezer.move_to(now)
    async_set_power(hass, "sensor.grid_power", 600)
    async_set_power(hass, "sensor.solar_power", 0)
    async_set_power(hass, "sensor.battery_power", 0)
    async_set_power(hass, "sensor.washer_power", 0)

    entry = await async_setup_via_flow(hass)
    hub = entry.runtime_data

    # Snapshot recorder history before bootstrap
    states_before = await get_instance(hass).async_add_executor_job(
        _query, hass, "sensor.grid_power", start_time.timestamp(), now.timestamp()
    )

    grid_profiler = hub.profilers["sensor.grid_power"]
    deltas_before = grid_profiler.delta_count

    async_fire_time_changed(
        hass, dt_util.utcnow() + timedelta(seconds=BOOTSTRAP_START_TIMEOUT_S + 1)
    )
    await hub.bootstrap_task

    # Profiler delta count must not have been modified by bootstrap rows
    assert grid_profiler.delta_count == deltas_before

    # Recorder history must be identical before and after
    states_after = await get_instance(hass).async_add_executor_job(
        _query, hass, "sensor.grid_power", start_time.timestamp(), now.timestamp()
    )
    assert states_before == states_after


@pytest.mark.bootstrap_enabled
async def test_recorder_unavailable_when_async_db_ready_false(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
) -> None:
    """When async_db_ready resolves False, status is recorder_unavailable and setup stays loaded."""
    energy_manager.data = std_prefs
    async_set_power(hass, "sensor.grid_power", 500)
    async_set_power(hass, "sensor.solar_power", 0)
    async_set_power(hass, "sensor.battery_power", 0)
    async_set_power(hass, "sensor.washer_power", 0)

    with patch.object(
        get_instance(hass),
        "async_db_ready",
        new_callable=lambda: asyncio.Future(),
    ) as mock_ready_fut:
        mock_ready_fut.set_result(False)

        entry = await async_setup_via_flow(hass)
        hub = entry.runtime_data

        async_fire_time_changed(
            hass, dt_util.utcnow() + timedelta(seconds=BOOTSTRAP_START_TIMEOUT_S + 1)
        )
        await hub.bootstrap_task

        assert hub.bootstrap_coordinator.data["status"] == "recorder_unavailable"
        assert len(hass.config_entries.async_entries("powerguesser")) == 1


@pytest.mark.bootstrap_enabled
async def test_newest_recorder_row_extended_to_live_last_reported(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    freezer: Any,
) -> None:
    """The newest recorder row is extended to live last_reported when last_updated matches."""
    energy_manager.data = std_prefs
    now = dt_util.utcnow()
    start_time = now - timedelta(days=1)
    freezer.move_to(start_time)

    # State recorded at start_time
    async_set_power(hass, "sensor.grid_power", 500)
    async_set_power(hass, "sensor.solar_power", 0)
    async_set_power(hass, "sensor.battery_power", 0)
    async_set_power(hass, "sensor.washer_power", 0)
    await async_wait_recording_done(hass)

    # Advance time: grid power remains at 500 W (state_reported, last_updated unchanged, last_reported updated)
    freezer.tick(timedelta(minutes=15))
    live_time = dt_util.utcnow()
    state = hass.states.get("sensor.grid_power")
    assert state is not None
    state.last_reported = live_time
    state._cache["last_reported_timestamp"] = live_time.timestamp()

    # Move to setup time 15 minutes later
    freezer.tick(timedelta(minutes=15))
    entry = await async_setup_via_flow(hass)
    hub = entry.runtime_data

    async_fire_time_changed(
        hass, dt_util.utcnow() + timedelta(seconds=BOOTSTRAP_START_TIMEOUT_S + 1)
    )
    await hub.bootstrap_task

    conn = open_readonly(hass.config.path(BUFFER_DIR, BUFFER_FILE))
    cursor = conn.execute("SELECT id FROM channels WHERE entity_id = 'sensor.grid_power'")
    grid_ch = cursor.fetchone()[0]

    live_start_str = await hass.async_add_executor_job(
        hub.writer.db.get_meta, f"live_start_ms:{grid_ch}"
    )
    rows = conn.execute(
        "SELECT ts, w FROM samples WHERE ch = ? AND ts < ? ORDER BY ts ASC",
        (grid_ch, int(live_start_str)),
    ).fetchall()

    expected_lr_ms = int(round(state.last_reported_timestamp * 1000))
    ts_list = [r[0] for r in rows]
    assert expected_lr_ms in ts_list

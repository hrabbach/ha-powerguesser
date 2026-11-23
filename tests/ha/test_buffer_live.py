"""Tracer and gap tests for PowerGuesser history buffer."""

from datetime import timedelta
import logging
from typing import Any
import pytest

from homeassistant.components.energy.data import EnergyManager
from homeassistant.const import EVENT_HOMEASSISTANT_STOP, STATE_UNAVAILABLE
from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util
from pytest_homeassistant.common import async_fire_time_changed

from custom_components.powerguesser.const import BUFFER_DIR, BUFFER_FILE, DOMAIN
from custom_components.powerguesser.engine.reader import (
    gross_on_read,
    open_readonly,
    read_range,
    resample_zoh,
)
from tests.conftest import (
    async_flush_buffer,
    async_read_buffer,
    async_set_power,
    async_setup_via_flow,
)


async def test_live_samples_buffered_and_gross_on_read_matches_sensor(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
) -> None:
    """Tracer test: live samples travel from hub to SQLite and gross on read matches sensor.

    Note: The equality between gross_on_read and sensor.powerguesser_gross_power
    depends on the live sensor having settled (grid change plus solar and battery
    re-reports, P1 D-05). Any future change to settle semantics must update this
    drive sequence, not the buffer assertion.
    """
    energy_manager.data = std_prefs
    async_set_power(hass, "sensor.grid_power", 500)
    async_set_power(hass, "sensor.solar_power", 1200)
    async_set_power(hass, "sensor.battery_power", -300)
    async_set_power(hass, "sensor.washer_power", 200)

    entry = await async_setup_via_flow(hass)
    assert len(hass.config_entries.async_entries(DOMAIN)) == 1

    # Change grid to 800 W (fires state_changed)
    async_set_power(hass, "sensor.grid_power", 800)
    await hass.async_block_till_done()

    # Re-report solar 1200 W and battery -300 W unchanged (fires state_reported)
    async_set_power(hass, "sensor.solar_power", 1200)
    await hass.async_block_till_done()
    async_set_power(hass, "sensor.battery_power", -300)
    await hass.async_block_till_done()

    # Set washer to 250 W (metered device)
    async_set_power(hass, "sensor.washer_power", 250)
    await hass.async_block_till_done()

    # Settle reached: 800 + 1200 - 300 = 1700 W
    gross_sensor = hass.states.get("sensor.powerguesser_gross_power")
    assert gross_sensor is not None
    assert float(gross_sensor.state) == 1700.0

    # Flush pending in-memory buffer batch to SQLite
    await async_flush_buffer(hass, entry)

    # Read back samples and gross layout via helper
    data, layout = await async_read_buffer(hass)

    # Verify channels present
    assert ("sensor.grid_power", "grid") in data
    assert ("sensor.solar_power", "solar") in data
    assert ("sensor.battery_power", "battery") in data
    assert ("sensor.washer_power", "metered") in data

    # Verify grid row with w = 800 at round(last_reported * 1000)
    grid_state = hass.states.get("sensor.grid_power")
    assert grid_state is not None
    expected_grid_ts = round(grid_state.last_reported_timestamp * 1000)
    grid_rows = data[("sensor.grid_power", "grid")]
    assert any(ts == expected_grid_ts and w == 800 for ts, w in grid_rows)

    # Verify solar and battery rows at their re-report timestamps (D-02)
    solar_state = hass.states.get("sensor.solar_power")
    assert solar_state is not None
    expected_solar_ts = round(solar_state.last_reported_timestamp * 1000)
    solar_rows = data[("sensor.solar_power", "solar")]
    assert any(ts == expected_solar_ts and w == 1200 for ts, w in solar_rows)

    battery_state = hass.states.get("sensor.battery_power")
    assert battery_state is not None
    expected_battery_ts = round(battery_state.last_reported_timestamp * 1000)
    battery_rows = data[("sensor.battery_power", "battery")]
    assert any(ts == expected_battery_ts and w == -300 for ts, w in battery_rows)

    # Verify washer row with w = 250 under role "metered"
    washer_state = hass.states.get("sensor.washer_power")
    assert washer_state is not None
    expected_washer_ts = round(washer_state.last_reported_timestamp * 1000)
    washer_rows = data[("sensor.washer_power", "metered")]
    assert any(ts == expected_washer_ts and w == 250 for ts, w in washer_rows)

    # Reconstruct gross using reader engine functions
    db_path = hass.config.path(BUFFER_DIR, BUFFER_FILE)
    conn = open_readonly(db_path)
    try:
        cur = conn.execute("SELECT id FROM channels ORDER BY id")
        all_channels = [r[0] for r in cur.fetchall()]

        all_ts = [
            ts
            for rows in data.values()
            for ts, _ in rows
        ]
        start_ms = min(all_ts)
        end_ms = max(all_ts) + 1000

        range_data = read_range(conn, all_channels, start_ms, end_ms)
        series_map = {
            ch: resample_zoh(range_data[ch], start_ms, end_ms, 1000)
            for ch in all_channels
        }
        rebuilt_gross = gross_on_read(series_map, layout)

        # Evaluated at the grid point just after the last settle
        assert rebuilt_gross[-1] == 1700.0
        assert rebuilt_gross[-1] == float(gross_sensor.state)
    finally:
        conn.close()


async def test_gap_unavailable_source_writes_null_row(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
) -> None:
    """An unavailable/unknown state on a tracked channel writes a NULL row."""
    energy_manager.data = std_prefs
    async_set_power(hass, "sensor.grid_power", 500)
    async_set_power(hass, "sensor.solar_power", 1000)
    async_set_power(hass, "sensor.battery_power", 0)
    async_set_power(hass, "sensor.washer_power", 200)

    entry = await async_setup_via_flow(hass)

    # Set grid to unavailable
    hass.states.async_set("sensor.grid_power", STATE_UNAVAILABLE)
    await hass.async_block_till_done()

    await async_flush_buffer(hass, entry)
    data, _ = await async_read_buffer(hass)

    grid_rows = data[("sensor.grid_power", "grid")]
    # The last row must be NULL
    assert grid_rows[-1][1] is None


async def test_gap_stale_rereporting_source_writes_null_row(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    freezer: Any,
) -> None:
    """A channel classified re-reporting silent for > 90s gets exactly one NULL row at last_ts + 90s."""
    energy_manager.data = std_prefs
    async_set_power(hass, "sensor.grid_power", 500)
    async_set_power(hass, "sensor.solar_power", 1000)
    async_set_power(hass, "sensor.battery_power", -300)
    async_set_power(hass, "sensor.washer_power", 200)

    entry = await async_setup_via_flow(hass)

    # Re-report battery 3 times 10s apart to classify under D-24
    for _ in range(3):
        freezer.tick(timedelta(seconds=10))
        async_set_power(hass, "sensor.battery_power", -300)
        await hass.async_block_till_done()

    battery_state = hass.states.get("sensor.battery_power")
    assert battery_state is not None
    last_reported_ms = round(battery_state.last_reported_timestamp * 1000)

    # Advance time by 91s (> 90s staleness threshold)
    freezer.tick(timedelta(seconds=91))
    async_fire_time_changed(hass, dt_util.utcnow(), fire_all=True)
    await hass.async_block_till_done()

    await async_flush_buffer(hass, entry)
    data, _ = await async_read_buffer(hass)
    battery_rows = data[("sensor.battery_power", "battery")]

    # Last row is a stale marker at exactly last_reported_ms + 90_000 ms
    expected_stale_ts = last_reported_ms + 90_000
    assert battery_rows[-1] == (expected_stale_ts, None)

    # Next tick doesn't duplicate the stale marker
    freezer.tick(timedelta(seconds=10))
    async_fire_time_changed(hass, dt_util.utcnow(), fire_all=True)
    await hass.async_block_till_done()

    await async_flush_buffer(hass, entry)
    data2, _ = await async_read_buffer(hass)
    battery_rows2 = data2[("sensor.battery_power", "battery")]
    stale_markers = [r for r in battery_rows2 if r[1] is None and r[0] == expected_stale_ts]
    assert len(stale_markers) == 1


async def test_gap_rereporting_metered_plug_never_stale(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    freezer: Any,
) -> None:
    """A metered plug classified re-reporting and then silent gets no stale NULL row.

    P1 D-07: metered plugs are held at their last value with no age limit unless
    unavailable. A stale marker after 2.5x a bursty plug's sub-second median (or the
    90 s default on a 300 s plug) blanked the plug in the learning path.
    """
    energy_manager.data = std_prefs
    async_set_power(hass, "sensor.grid_power", 500)
    async_set_power(hass, "sensor.solar_power", 1000)
    async_set_power(hass, "sensor.battery_power", -300)
    async_set_power(hass, "sensor.washer_power", 200)

    entry = await async_setup_via_flow(hass)

    # Re-report the washer 3 times 10 s apart to classify it re-reporting under D-24
    for _ in range(3):
        freezer.tick(timedelta(seconds=10))
        async_set_power(hass, "sensor.washer_power", 200)
        await hass.async_block_till_done()
    assert entry.runtime_data.modes["sensor.washer_power"].re_reporting

    # Silent well past the 90 s default staleness timeout, ticking along the way
    for _ in range(4):
        freezer.tick(timedelta(seconds=91))
        async_fire_time_changed(hass, dt_util.utcnow(), fire_all=True)
        await hass.async_block_till_done()

    await async_flush_buffer(hass, entry)
    data, _ = await async_read_buffer(hass)
    washer_rows = data[("sensor.washer_power", "metered")]

    # Last row is still the held value; the only NULL row is the activation marker
    assert washer_rows[-1][1] == 200
    null_rows = [r for r in washer_rows if r[1] is None]
    assert null_rows == [washer_rows[0]]


async def test_gap_single_rereport_never_stale(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    freezer: Any,
) -> None:
    """A channel with fewer than 3 re-reports stays on-change and never gets a stale NULL marker."""
    energy_manager.data = std_prefs
    async_set_power(hass, "sensor.grid_power", 500)
    async_set_power(hass, "sensor.solar_power", 1000)
    async_set_power(hass, "sensor.battery_power", -300)
    async_set_power(hass, "sensor.washer_power", 200)

    entry = await async_setup_via_flow(hass)

    # Single re-report
    freezer.tick(timedelta(seconds=10))
    async_set_power(hass, "sensor.battery_power", -300)
    await hass.async_block_till_done()

    # Advance time by 120s
    freezer.tick(timedelta(seconds=120))
    async_fire_time_changed(hass, dt_util.utcnow(), fire_all=True)
    await hass.async_block_till_done()

    await async_flush_buffer(hass, entry)
    data, _ = await async_read_buffer(hass)
    battery_rows = data[("sensor.battery_power", "battery")]
    # Last row is still the held value (-300), no stale NULL row was appended
    assert battery_rows[-1][1] == -300
    # The only NULL row for battery was the initial activation marker at startup
    null_rows = [r for r in battery_rows if r[1] is None]
    assert len(null_rows) == 1
    assert null_rows[0] == battery_rows[0]


async def test_gap_stop_writes_stop_markers_and_clean_stop(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
) -> None:
    """HA stop writes a NULL row per active channel, flushes, and records clean_stop '1'."""
    energy_manager.data = std_prefs
    async_set_power(hass, "sensor.grid_power", 500)
    async_set_power(hass, "sensor.solar_power", 1000)
    async_set_power(hass, "sensor.battery_power", -300)
    async_set_power(hass, "sensor.washer_power", 200)

    await async_setup_via_flow(hass)

    # Fire HA stop event
    hass.bus.async_fire(EVENT_HOMEASSISTANT_STOP)
    await hass.async_block_till_done()

    # Open db read-only and verify clean_stop is 1 and all channels have NULL as last row
    db_path = hass.config.path(BUFFER_DIR, BUFFER_FILE)
    conn = open_readonly(db_path)
    try:
        clean_stop = conn.execute("SELECT value FROM meta WHERE key = 'clean_stop'").fetchone()
        assert clean_stop is not None and clean_stop[0] == "1"

        channels = conn.execute("SELECT id FROM channels").fetchall()
        for (ch_id,) in channels:
            last_sample = conn.execute(
                "SELECT w FROM samples WHERE ch = ? ORDER BY ts DESC LIMIT 1",
                (ch_id,),
            ).fetchone()
            assert last_sample is not None
            assert last_sample[0] is None
    finally:
        conn.close()


async def test_gap_startup_and_seed_after_unload(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
) -> None:
    """Unload then reload writes an activation NULL row followed by a seed row with current state."""
    energy_manager.data = std_prefs
    async_set_power(hass, "sensor.grid_power", 500)
    async_set_power(hass, "sensor.solar_power", 1000)
    async_set_power(hass, "sensor.battery_power", -300)
    async_set_power(hass, "sensor.washer_power", 200)

    entry = await async_setup_via_flow(hass)
    await async_flush_buffer(hass, entry)

    # Unload entry (which closes buffer cleanly)
    await hass.config_entries.async_unload(entry.entry_id)
    await hass.async_block_till_done()

    # Reload entry
    await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    await async_flush_buffer(hass, entry)
    data, _ = await async_read_buffer(hass)

    # For each active channel, after unload there was a stop NULL marker,
    # then reload added an activation NULL marker, followed by a seed row!
    for key in [
        ("sensor.grid_power", "grid"),
        ("sensor.solar_power", "solar"),
        ("sensor.battery_power", "battery"),
        ("sensor.washer_power", "metered"),
    ]:
        rows = data[key]
        # Last row should be the non-null seed row
        assert rows[-1][1] is not None
        # Penultimate row should be the activation NULL row
        assert rows[-2][1] is None
        # Seed timestamp > activation timestamp
        assert rows[-1][0] > rows[-2][0]


async def test_gap_removed_channel_writes_marker_and_drops_inflight(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
) -> None:
    """A metered device removed from Energy prefs gets one NULL row and subsequent samples are dropped."""
    energy_manager.data = std_prefs
    async_set_power(hass, "sensor.grid_power", 500)
    async_set_power(hass, "sensor.solar_power", 1000)
    async_set_power(hass, "sensor.battery_power", -300)
    async_set_power(hass, "sensor.washer_power", 200)

    entry = await async_setup_via_flow(hass)
    await async_flush_buffer(hass, entry)

    hub = entry.runtime_data
    washer_channels = hub._entity_channels.get("sensor.washer_power", ())
    assert len(washer_channels) > 0
    washer_ch = washer_channels[0]

    # Remove washer from prefs and update
    prefs_no_washer = dict(std_prefs)
    prefs_no_washer["device_consumption"] = []
    await energy_manager.async_update(prefs_no_washer)
    await hass.async_block_till_done()

    # In-flight sample arriving for removed washer
    hub.writer.record(washer_ch, 300, dt_util.utcnow().timestamp())

    await async_flush_buffer(hass, entry)
    data, _ = await async_read_buffer(hass)
    washer_rows = data[("sensor.washer_power", "metered")]

    # The last row must be the NULL removal marker, the 300 W in-flight sample was dropped
    assert washer_rows[-1][1] is None


async def test_off_loop_live_flush_no_blocking_warning(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    caplog: Any,
    freezer: Any,
) -> None:
    """Live flush through executor logs neither Detected blocking call nor BufferDB called from an event loop thread."""
    caplog.set_level(logging.WARNING)
    energy_manager.data = std_prefs
    async_set_power(hass, "sensor.grid_power", 500)
    async_set_power(hass, "sensor.solar_power", 1000)
    async_set_power(hass, "sensor.battery_power", -300)
    async_set_power(hass, "sensor.washer_power", 200)

    await async_setup_via_flow(hass)

    # Send state updates
    async_set_power(hass, "sensor.grid_power", 600)
    await hass.async_block_till_done()

    # Advance time by 61s to trigger periodic flush
    freezer.tick(timedelta(seconds=61))
    async_fire_time_changed(hass, dt_util.utcnow(), fire_all=True)
    await hass.async_block_till_done()

    # Verify no blocking call warning was logged
    for r in caplog.records:
        assert "Detected blocking call" not in r.message
        assert "BufferDB called from an event loop thread" not in r.message

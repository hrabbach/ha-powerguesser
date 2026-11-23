"""Integration tests for interval profiling and diagnostic sensors."""

import copy
from datetime import timedelta
from typing import Any

from homeassistant.components.energy.data import EnergyManager
from homeassistant.const import EntityCategory
from homeassistant.core import HomeAssistant
from homeassistant.helpers import entity_registry as er
from homeassistant.util import dt as dt_util
from pytest_homeassistant.common import async_fire_time_changed

from custom_components.powerguesser.const import (
    DOMAIN,
    PROFILE_STORE_KEY,
    PROFILE_STORE_VERSION,
)
from tests.conftest import (
    async_flush_buffer,
    async_read_buffer,
    async_set_power,
    async_setup_via_flow,
)


async def test_sample_interval_sensor_shows_median(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    freezer: Any,
) -> None:
    """Tracer: 21 unchanged re-reports of gross sources 30 s apart -> grid sensor reads 30, mode re_reporting, reports."""
    energy_manager.data = std_prefs
    async_set_power(hass, "sensor.grid_power", 500)
    async_set_power(hass, "sensor.solar_power", 1200)
    async_set_power(hass, "sensor.battery_power", -300)
    async_set_power(hass, "sensor.washer_power", 200)

    await async_setup_via_flow(hass)
    ent_reg = er.async_get(hass)

    grid_sensor_id = ent_reg.async_get_entity_id(
        "sensor", DOMAIN, "powerguesser_sample_interval_sensor.grid_power"
    )
    assert grid_sensor_id is not None
    grid_reg_entry = ent_reg.async_get(grid_sensor_id)
    assert grid_reg_entry is not None
    assert grid_reg_entry.entity_category == EntityCategory.DIAGNOSTIC
    assert not grid_reg_entry.disabled

    washer_sensor_id = ent_reg.async_get_entity_id(
        "sensor", DOMAIN, "powerguesser_sample_interval_sensor.washer_power"
    )
    assert washer_sensor_id is not None
    washer_reg_entry = ent_reg.async_get(washer_sensor_id)
    assert washer_reg_entry is not None
    assert not washer_reg_entry.disabled

    # Initially before 20 deltas exist, value is None (state unknown)
    state = hass.states.get(grid_sensor_id)
    assert state is not None
    assert state.state == "unknown"
    assert state.attributes.get("mode") == "on_change"
    assert state.attributes.get("measured_from") is None

    # 21 unchanged re-reports of every gross entity 30 s apart
    for _ in range(21):
        freezer.tick(timedelta(seconds=30))
        async_fire_time_changed(hass, dt_util.utcnow(), fire_all=True)
        async_set_power(hass, "sensor.grid_power", 500)
        async_set_power(hass, "sensor.solar_power", 1200)
        async_set_power(hass, "sensor.battery_power", -300)
        await hass.async_block_till_done()

    state = hass.states.get(grid_sensor_id)
    assert state is not None
    assert state.state == "30"
    assert state.attributes.get("mode") == "re_reporting"
    assert state.attributes.get("measured_from") == "reports"
    assert state.attributes.get("restored") is False


async def test_measured_interval_d15_staleness_and_null_row(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    freezer: Any,
) -> None:
    """D-15: Grid re-reporting every 10s for 21 samples goes stale after 26s of silence, writing NULL at last_ts + 25s."""
    energy_manager.data = std_prefs
    async_set_power(hass, "sensor.grid_power", 500)
    async_set_power(hass, "sensor.solar_power", 1200)
    async_set_power(hass, "sensor.battery_power", -300)
    async_set_power(hass, "sensor.washer_power", 200)

    entry = await async_setup_via_flow(hass)

    # 21 re-reports of grid every 10s
    last_report_dt = None
    for _ in range(21):
        freezer.tick(timedelta(seconds=10))
        async_fire_time_changed(hass, dt_util.utcnow(), fire_all=True)
        async_set_power(hass, "sensor.grid_power", 500)
        await hass.async_block_till_done()
        last_report_dt = dt_util.utcnow()

    grid_profiler = entry.runtime_data.profilers["sensor.grid_power"]
    assert grid_profiler.median_s == 10.0
    assert grid_profiler.stale_after_s() == 25.0
    assert grid_profiler.mode == "re_reporting"

    # Advance time by 24s -> not stale yet
    freezer.tick(timedelta(seconds=24))
    async_fire_time_changed(hass, dt_util.utcnow(), fire_all=True)
    await hass.async_block_till_done()
    assert hass.states.get("sensor.powerguesser_gross_power").state != "unavailable"

    # Advance by 2s more (total 26s since last report) -> gross is unavailable
    freezer.tick(timedelta(seconds=2))
    async_fire_time_changed(hass, dt_util.utcnow(), fire_all=True)
    await hass.async_block_till_done()
    assert hass.states.get("sensor.powerguesser_gross_power").state == "unavailable"

    # Flush buffer and check for NULL row at last_ts + 25 000 ms in that silence
    await async_flush_buffer(hass, entry)
    data, _ = await async_read_buffer(hass)
    grid_samples = data[("sensor.grid_power", "grid")]
    assert last_report_dt is not None
    last_report_ms = round(last_report_dt.timestamp() * 1000)
    silence_null_rows = [row for row in grid_samples if row[1] is None and row[0] > last_report_ms]
    assert len(silence_null_rows) == 1
    expected_null_ts = last_report_ms + 25_000
    assert silence_null_rows[0][0] == expected_null_ts


async def test_stray_report_keeps_on_change_cadence(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    freezer: Any,
) -> None:
    """D-24: Grid on-change at 31s for 21 changes, then 1 stray re-report -> stays on_change, gross available after 300s, no NULL row."""
    energy_manager.data = std_prefs
    async_set_power(hass, "sensor.grid_power", 500)
    async_set_power(hass, "sensor.solar_power", 1200)
    async_set_power(hass, "sensor.battery_power", -300)
    async_set_power(hass, "sensor.washer_power", 200)

    entry = await async_setup_via_flow(hass)

    # 21 value changes on grid every 31s
    for i in range(21):
        freezer.tick(timedelta(seconds=31))
        async_fire_time_changed(hass, dt_util.utcnow(), fire_all=True)
        async_set_power(hass, "sensor.grid_power", 500 + i + 1)
        await hass.async_block_till_done()

    # 1 unchanged state_reported
    freezer.tick(timedelta(seconds=31))
    async_fire_time_changed(hass, dt_util.utcnow(), fire_all=True)
    async_set_power(hass, "sensor.grid_power", 500 + 21)
    await hass.async_block_till_done()

    ent_reg = er.async_get(hass)
    grid_sensor_id = ent_reg.async_get_entity_id(
        "sensor", DOMAIN, "powerguesser_sample_interval_sensor.grid_power"
    )
    assert grid_sensor_id is not None
    state = hass.states.get(grid_sensor_id)
    assert state is not None
    assert state.state == "31"
    assert state.attributes.get("mode") == "on_change"
    assert state.attributes.get("measured_from") == "changes"

    last_report_ms = round(dt_util.utcnow().timestamp() * 1000)

    # 300s of silence
    freezer.tick(timedelta(seconds=300))
    async_fire_time_changed(hass, dt_util.utcnow(), fire_all=True)
    await hass.async_block_till_done()

    assert hass.states.get("sensor.powerguesser_gross_power").state != "unavailable"

    await async_flush_buffer(hass, entry)
    data, _ = await async_read_buffer(hass)
    grid_samples = data[("sensor.grid_power", "grid")]
    silence_null_rows = [row for row in grid_samples if row[1] is None and row[0] > last_report_ms]
    assert len(silence_null_rows) == 0


async def test_effective_sample_interval_sensor(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    freezer: Any,
) -> None:
    """Effective interval sensor shows 30 s assumed when empty, and tracks max published median."""
    energy_manager.data = std_prefs
    async_set_power(hass, "sensor.grid_power", 500)
    async_set_power(hass, "sensor.solar_power", 1200)
    async_set_power(hass, "sensor.battery_power", -300)
    async_set_power(hass, "sensor.washer_power", 200)

    await async_setup_via_flow(hass)
    ent_reg = er.async_get(hass)
    eff_sensor_id = ent_reg.async_get_entity_id(
        "sensor", DOMAIN, "powerguesser_effective_sample_interval"
    )
    assert eff_sensor_id is not None
    eff_entry = ent_reg.async_get(eff_sensor_id)
    assert eff_entry is not None
    assert eff_entry.entity_category == EntityCategory.DIAGNOSTIC

    # Initially, with no published medians among gross sources -> 30 and assumed True
    eff_state = hass.states.get(eff_sensor_id)
    assert eff_state is not None
    assert eff_state.state == "30"
    assert eff_state.attributes.get("interval_assumed") is True
    assert eff_state.attributes.get("enabled_event_types") == [
        "step",
        "cycle_shape",
        "continuous",
    ]
    assert "short_event" in eff_state.attributes.get("disabled_event_types")

    # Drive 21 re-reports on grid 31s apart
    for _ in range(21):
        freezer.tick(timedelta(seconds=31))
        async_fire_time_changed(hass, dt_util.utcnow(), fire_all=True)
        async_set_power(hass, "sensor.grid_power", 500)
        await hass.async_block_till_done()

    eff_state = hass.states.get(eff_sensor_id)
    assert eff_state is not None
    assert eff_state.state == "31"
    assert eff_state.attributes.get("interval_assumed") is False
    assert eff_state.attributes.get("enabled_event_types") == [
        "step",
        "cycle_shape",
        "continuous",
    ]
    assert (
        eff_state.attributes.get("disabled_event_types")["short_event"]
        == "short_event disabled: needs ≤5 s interval, effective is 31 s"
    )


async def test_on_change_cadence_gates(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    freezer: Any,
) -> None:
    """D-23: Grid on-change every 31 s for 21 changes gates effective interval without re-reports."""
    energy_manager.data = std_prefs
    async_set_power(hass, "sensor.grid_power", 500)
    async_set_power(hass, "sensor.solar_power", 1200)
    async_set_power(hass, "sensor.battery_power", -300)
    async_set_power(hass, "sensor.washer_power", 200)

    await async_setup_via_flow(hass)
    ent_reg = er.async_get(hass)
    grid_sensor_id = ent_reg.async_get_entity_id(
        "sensor", DOMAIN, "powerguesser_sample_interval_sensor.grid_power"
    )
    eff_sensor_id = ent_reg.async_get_entity_id(
        "sensor", DOMAIN, "powerguesser_effective_sample_interval"
    )
    assert grid_sensor_id is not None
    assert eff_sensor_id is not None

    for i in range(21):
        freezer.tick(timedelta(seconds=31))
        async_fire_time_changed(hass, dt_util.utcnow(), fire_all=True)
        async_set_power(hass, "sensor.grid_power", 500 + i + 1)
        await hass.async_block_till_done()

    grid_state = hass.states.get(grid_sensor_id)
    assert grid_state is not None
    assert grid_state.state == "31"
    assert grid_state.attributes.get("mode") == "on_change"
    assert grid_state.attributes.get("measured_from") == "changes"

    eff_state = hass.states.get(eff_sensor_id)
    assert eff_state is not None
    assert eff_state.state == "31"
    assert eff_state.attributes.get("interval_assumed") is False
    assert eff_state.attributes.get("enabled_event_types") == [
        "step",
        "cycle_shape",
        "continuous",
    ]
    assert "short_event" in eff_state.attributes.get("disabled_event_types")


async def test_on_change_solar_night_excluded(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    freezer: Any,
) -> None:
    """D-23: Solar changing at 30s non-zero, then 0 W followed by 8h silence and 40 W excludes the night delta."""
    energy_manager.data = std_prefs
    async_set_power(hass, "sensor.grid_power", 500)
    async_set_power(hass, "sensor.solar_power", 1200)
    async_set_power(hass, "sensor.battery_power", -300)
    async_set_power(hass, "sensor.washer_power", 200)

    entry = await async_setup_via_flow(hass)
    ent_reg = er.async_get(hass)
    solar_sensor_id = ent_reg.async_get_entity_id(
        "sensor", DOMAIN, "powerguesser_sample_interval_sensor.solar_power"
    )
    eff_sensor_id = ent_reg.async_get_entity_id(
        "sensor", DOMAIN, "powerguesser_effective_sample_interval"
    )
    assert solar_sensor_id is not None
    assert eff_sensor_id is not None

    # Grid changes at 31s
    for i in range(21):
        freezer.tick(timedelta(seconds=31))
        async_fire_time_changed(hass, dt_util.utcnow(), fire_all=True)
        async_set_power(hass, "sensor.grid_power", 500 + i + 1)
        await hass.async_block_till_done()

    # Solar changes 21 times 30s apart with non-zero values
    for i in range(21):
        freezer.tick(timedelta(seconds=30))
        async_fire_time_changed(hass, dt_util.utcnow(), fire_all=True)
        async_set_power(hass, "sensor.solar_power", 1000 + i + 1)
        await hass.async_block_till_done()

    solar_profiler = entry.runtime_data.profilers["sensor.solar_power"]
    assert solar_profiler.delta_count == 20
    assert solar_profiler.median_s == 30.0

    # Solar reports 0 W
    freezer.tick(timedelta(seconds=30))
    async_fire_time_changed(hass, dt_util.utcnow(), fire_all=True)
    async_set_power(hass, "sensor.solar_power", 0)
    await hass.async_block_till_done()
    assert solar_profiler.delta_count == 21

    # Stays silent for 8 hours (28800s) and changes to 40 W
    freezer.tick(timedelta(seconds=28800))
    async_fire_time_changed(hass, dt_util.utcnow(), fire_all=True)
    async_set_power(hass, "sensor.solar_power", 40)
    await hass.async_block_till_done()

    # Delta count did NOT grow by the night delta, solar median is still 30
    assert solar_profiler.delta_count == 21
    assert solar_profiler.median_s == 30.0

    solar_state = hass.states.get(solar_sensor_id)
    assert solar_state is not None
    assert solar_state.state == "30"

    eff_state = hass.states.get(eff_sensor_id)
    assert eff_state is not None
    assert eff_state.state == "31"


async def test_on_change_with_median_never_stale(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    freezer: Any,
) -> None:
    """D-23/P1 D-19 hold: After grid has 31s change-cadence median and stays silent 300s, gross stays available, no NULL row."""
    energy_manager.data = std_prefs
    async_set_power(hass, "sensor.grid_power", 500)
    async_set_power(hass, "sensor.solar_power", 1200)
    async_set_power(hass, "sensor.battery_power", -300)
    async_set_power(hass, "sensor.washer_power", 200)

    entry = await async_setup_via_flow(hass)

    # 21 changes on grid 31s apart
    for i in range(21):
        freezer.tick(timedelta(seconds=31))
        async_fire_time_changed(hass, dt_util.utcnow(), fire_all=True)
        async_set_power(hass, "sensor.grid_power", 500 + i + 1)
        await hass.async_block_till_done()

    grid_profiler = entry.runtime_data.profilers["sensor.grid_power"]
    assert grid_profiler.median_s == 31.0
    assert grid_profiler.mode == "on_change"

    last_change_ms = round(dt_util.utcnow().timestamp() * 1000)

    # Silence for 300 s
    freezer.tick(timedelta(seconds=300))
    async_fire_time_changed(hass, dt_util.utcnow(), fire_all=True)
    await hass.async_block_till_done()

    gross_state = hass.states.get("sensor.powerguesser_gross_power")
    assert gross_state is not None
    assert gross_state.state != "unavailable"

    await async_flush_buffer(hass, entry)
    data, _ = await async_read_buffer(hass)
    grid_samples = data[("sensor.grid_power", "grid")]
    silence_null_rows = [row for row in grid_samples if row[1] is None and row[0] > last_change_ms]
    assert len(silence_null_rows) == 0


async def test_store_restore_re_reporting_seeds_d24_and_staleness(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
) -> None:
    """Restored re_reporting mode seeds shared D-24 classification and D-15 staleness applies immediately."""
    hass_storage[PROFILE_STORE_KEY] = {
        "version": PROFILE_STORE_VERSION,
        "minor_version": 1,
        "key": PROFILE_STORE_KEY,
        "data": {
            "sources": {
                "sensor.grid_power": {
                    "median_s": 10.0,
                    "p90_s": 11.0,
                    "mode": "re_reporting",
                }
            }
        },
    }
    energy_manager.data = std_prefs
    async_set_power(hass, "sensor.grid_power", 500)
    async_set_power(hass, "sensor.solar_power", 1200)
    async_set_power(hass, "sensor.battery_power", -300)
    async_set_power(hass, "sensor.washer_power", 200)

    entry = await async_setup_via_flow(hass)
    ent_reg = er.async_get(hass)
    grid_sensor_id = ent_reg.async_get_entity_id(
        "sensor", DOMAIN, "powerguesser_sample_interval_sensor.grid_power"
    )
    assert grid_sensor_id is not None
    state = hass.states.get(grid_sensor_id)
    assert state is not None
    assert state.state == "10"
    assert state.attributes.get("mode") == "re_reporting"
    assert state.attributes.get("restored") is True

    # Shared D-24 classification seeded without live re-report
    assert entry.runtime_data.modes["sensor.grid_power"].re_reporting is True

    # Silence of 26s triggers staleness at 2.5 * 10 = 25s
    last_report_dt = dt_util.utcnow()
    last_report_ms = round(last_report_dt.timestamp() * 1000)
    expected_stale_ms = last_report_ms + 25_000

    freezer.tick(timedelta(seconds=26))
    async_fire_time_changed(hass, dt_util.utcnow(), fire_all=True)
    await hass.async_block_till_done()

    gross_state = hass.states.get("sensor.powerguesser_gross_power")
    assert gross_state is not None
    assert gross_state.state == "unavailable"

    await async_flush_buffer(hass, entry)
    data, _ = await async_read_buffer(hass)
    grid_samples = data[("sensor.grid_power", "grid")]
    silence_null_rows = [
        row for row in grid_samples if row[1] is None and row[0] > last_report_ms
    ]
    assert len(silence_null_rows) == 1
    assert abs(silence_null_rows[0][0] - expected_stale_ms) <= 1


async def test_store_restore_on_change_never_seeds_staleness(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
) -> None:
    """Restored on_change entry never makes a source re-reporting or stale (D-23, D-24)."""
    hass_storage[PROFILE_STORE_KEY] = {
        "version": PROFILE_STORE_VERSION,
        "minor_version": 1,
        "key": PROFILE_STORE_KEY,
        "data": {
            "sources": {
                "sensor.solar_power": {
                    "median_s": 31.0,
                    "p90_s": 33.0,
                    "mode": "on_change",
                }
            }
        },
    }
    energy_manager.data = std_prefs
    async_set_power(hass, "sensor.grid_power", 500)
    async_set_power(hass, "sensor.solar_power", 1200)
    async_set_power(hass, "sensor.battery_power", -300)
    async_set_power(hass, "sensor.washer_power", 200)

    entry = await async_setup_via_flow(hass)
    ent_reg = er.async_get(hass)
    solar_sensor_id = ent_reg.async_get_entity_id(
        "sensor", DOMAIN, "powerguesser_sample_interval_sensor.solar_power"
    )
    assert solar_sensor_id is not None
    state = hass.states.get(solar_sensor_id)
    assert state is not None
    assert state.state == "31"
    assert state.attributes.get("measured_from") == "changes"
    assert state.attributes.get("restored") is True

    # Solar stays on_change
    assert entry.runtime_data.modes["sensor.solar_power"].re_reporting is False

    # Effective sensor counts solar median
    eff_sensor_id = ent_reg.async_get_entity_id(
        "sensor", DOMAIN, "powerguesser_effective_sample_interval"
    )
    eff_state = hass.states.get(eff_sensor_id)
    assert eff_state is not None
    assert eff_state.state == "31"

    # Silence of 300s does not cause staleness or NULL row
    last_change_ms = round(dt_util.utcnow().timestamp() * 1000)
    freezer.tick(timedelta(seconds=300))
    async_fire_time_changed(hass, dt_util.utcnow(), fire_all=True)
    await hass.async_block_till_done()

    gross_state = hass.states.get("sensor.powerguesser_gross_power")
    assert gross_state is not None
    assert gross_state.state != "unavailable"

    await async_flush_buffer(hass, entry)
    data, _ = await async_read_buffer(hass)
    solar_samples = data[("sensor.solar_power", "solar")]
    silence_null_rows = [
        row for row in solar_samples if row[1] is None and row[0] > last_change_ms
    ]
    assert len(silence_null_rows) == 0


async def test_store_restore_malformed_ignored(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
) -> None:
    """Malformed stored entries (non-numeric, negative or zero median, unknown mode) are ignored."""
    hass_storage[PROFILE_STORE_KEY] = {
        "version": PROFILE_STORE_VERSION,
        "minor_version": 1,
        "key": PROFILE_STORE_KEY,
        "data": {
            "sources": {
                "sensor.grid_power": {
                    "median_s": "invalid",
                    "p90_s": 10.0,
                    "mode": "re_reporting",
                },
                "sensor.solar_power": {
                    "median_s": -5.0,
                    "p90_s": 10.0,
                    "mode": "re_reporting",
                },
                "sensor.battery_power": {
                    "median_s": 0.0,
                    "p90_s": 0.0,
                    "mode": "re_reporting",
                },
                "sensor.washer_power": {
                    "median_s": 10.0,
                    "p90_s": 10.0,
                    "mode": "not_a_mode",
                },
            }
        },
    }
    energy_manager.data = std_prefs
    async_set_power(hass, "sensor.grid_power", 500)
    async_set_power(hass, "sensor.solar_power", 1200)
    async_set_power(hass, "sensor.battery_power", -300)
    async_set_power(hass, "sensor.washer_power", 200)

    entry = await async_setup_via_flow(hass)
    ent_reg = er.async_get(hass)

    for eid in (
        "sensor.grid_power",
        "sensor.solar_power",
        "sensor.battery_power",
        "sensor.washer_power",
    ):
        sensor_id = ent_reg.async_get_entity_id(
            "sensor", DOMAIN, f"powerguesser_sample_interval_{eid}"
        )
        assert sensor_id is not None
        state = hass.states.get(sensor_id)
        assert state is not None
        assert state.state == "unknown"
        assert state.attributes.get("restored") is False
        assert entry.runtime_data.profilers[eid].median_s is None
        assert entry.runtime_data.modes[eid].re_reporting is False


async def test_store_delayed_save(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
) -> None:
    """After 21 re-reports 30s apart and 600s more, hass_storage holds powerguesser.profile with median_s 30.0."""
    energy_manager.data = std_prefs
    async_set_power(hass, "sensor.grid_power", 500)
    async_set_power(hass, "sensor.solar_power", 1200)
    async_set_power(hass, "sensor.battery_power", -300)
    async_set_power(hass, "sensor.washer_power", 200)

    entry = await async_setup_via_flow(hass)
    ent_reg = er.async_get(hass)
    grid_sensor_id = ent_reg.async_get_entity_id(
        "sensor", DOMAIN, "powerguesser_sample_interval_sensor.grid_power"
    )

    # 21 unchanged re-reports 30s apart
    for _ in range(21):
        freezer.tick(timedelta(seconds=30))
        async_fire_time_changed(hass, dt_util.utcnow(), fire_all=True)
        async_set_power(hass, "sensor.grid_power", 500)
        async_set_power(hass, "sensor.solar_power", 1200)
        async_set_power(hass, "sensor.battery_power", -300)
        await hass.async_block_till_done()

    state = hass.states.get(grid_sensor_id)
    assert state is not None
    assert state.state == "30"
    assert state.attributes.get("restored") is False

    # Store not yet written before delay
    assert (
        PROFILE_STORE_KEY not in hass_storage
        or hass_storage[PROFILE_STORE_KEY]["data"]["sources"] == {}
    )

    # Advance freezer by 600s delay
    freezer.tick(timedelta(seconds=600))
    async_fire_time_changed(hass, dt_util.utcnow(), fire_all=True)
    await hass.async_block_till_done()

    assert PROFILE_STORE_KEY in hass_storage
    stored_sources = hass_storage[PROFILE_STORE_KEY]["data"]["sources"]
    assert "sensor.grid_power" in stored_sources
    assert stored_sources["sensor.grid_power"]["median_s"] == 30.0
    assert stored_sources["sensor.grid_power"]["mode"] == "re_reporting"


async def test_dynamic_add_and_remove_sample_interval_sensor(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
) -> None:
    """An Energy-prefs update adding dryer adds sample-interval sensor; removing it removes entity from registry and hub."""
    energy_manager.data = copy.deepcopy(std_prefs)
    async_set_power(hass, "sensor.grid_power", 500)
    async_set_power(hass, "sensor.solar_power", 1200)
    async_set_power(hass, "sensor.battery_power", -300)
    async_set_power(hass, "sensor.washer_power", 200)
    async_set_power(hass, "sensor.dryer_power", 1800)

    entry = await async_setup_via_flow(hass)
    ent_reg = er.async_get(hass)

    dryer_unique_id = "powerguesser_sample_interval_sensor.dryer_power"
    assert ent_reg.async_get_entity_id("sensor", DOMAIN, dryer_unique_id) is None
    assert "sensor.dryer_power" not in entry.runtime_data.profilers
    assert "sensor.dryer_power" not in entry.runtime_data.modes

    gross_entity_id = "sensor.powerguesser_gross_power"
    remainder_entity_id = "sensor.powerguesser_remainder_power"
    assert hass.states.get(gross_entity_id) is not None
    assert hass.states.get(remainder_entity_id) is not None

    # Add dryer to device_consumption
    new_prefs = copy.deepcopy(std_prefs)
    new_prefs["device_consumption"].append({"stat_rate": "sensor.dryer_power"})
    await energy_manager.async_update(new_prefs)
    await hass.async_block_till_done()

    # Dryer sensor dynamically added
    dryer_sensor_id = ent_reg.async_get_entity_id("sensor", DOMAIN, dryer_unique_id)
    assert dryer_sensor_id is not None
    assert hass.states.get(dryer_sensor_id) is not None
    assert "sensor.dryer_power" in entry.runtime_data.profilers
    assert "sensor.dryer_power" in entry.runtime_data.modes

    # Gross and remainder untouched
    assert hass.states.get(gross_entity_id) is not None
    assert hass.states.get(remainder_entity_id) is not None

    # Remove dryer from device_consumption
    await energy_manager.async_update(copy.deepcopy(std_prefs))
    await hass.async_block_till_done()

    # Dryer sensor removed from registry
    assert ent_reg.async_get_entity_id("sensor", DOMAIN, dryer_unique_id) is None
    assert "sensor.dryer_power" not in entry.runtime_data.profilers
    assert "sensor.dryer_power" not in entry.runtime_data.modes

    # Gross and remainder still untouched
    assert hass.states.get(gross_entity_id) is not None
    assert hass.states.get(remainder_entity_id) is not None


async def test_remove_entry_removes_profile_store(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
) -> None:
    """Removing the config entry removes powerguesser.profile from hass_storage (D-19)."""
    hass_storage[PROFILE_STORE_KEY] = {
        "version": PROFILE_STORE_VERSION,
        "minor_version": 1,
        "key": PROFILE_STORE_KEY,
        "data": {
            "sources": {
                "sensor.grid_power": {
                    "median_s": 10.0,
                    "p90_s": 11.0,
                    "mode": "re_reporting",
                }
            }
        },
    }
    energy_manager.data = std_prefs
    async_set_power(hass, "sensor.grid_power", 500)
    async_set_power(hass, "sensor.solar_power", 1200)
    async_set_power(hass, "sensor.battery_power", -300)
    async_set_power(hass, "sensor.washer_power", 200)

    entry = await async_setup_via_flow(hass)
    assert PROFILE_STORE_KEY in hass_storage

    await hass.config_entries.async_remove(entry.entry_id)
    await hass.async_block_till_done()

    assert PROFILE_STORE_KEY not in hass_storage

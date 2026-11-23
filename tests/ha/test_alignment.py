"""Integration tests for temporal alignment, phantom-spike avoidance, and staleness."""

from datetime import timedelta
import logging
from typing import Any

from homeassistant.components.energy.data import EnergyManager
from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util
from pytest_homeassistant.common import async_fire_time_changed

from tests.conftest import async_set_power, async_setup_via_flow


async def test_phantom_spike_avoidance_through_events(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
) -> None:
    """SRC-04 / D-05: Step change on battery does not cause single-sample phantom spike before grid updates."""
    energy_manager.data = std_prefs
    async_set_power(hass, "sensor.grid_power", 1500)
    async_set_power(hass, "sensor.solar_power", 0)
    async_set_power(hass, "sensor.battery_power", -500)
    async_set_power(hass, "sensor.washer_power", 200)

    await async_setup_via_flow(hass)
    assert float(hass.states.get("sensor.powerguesser_gross_power").state) == 1000.0

    published_gross: list[float] = []

    def _state_listener(event: Any) -> None:
        if event.data.get("entity_id") == "sensor.powerguesser_gross_power":
            new_s = event.data.get("new_state")
            if new_s and new_s.state not in ("unavailable", "unknown"):
                published_gross.append(float(new_s.state))

    unsub = hass.bus.async_listen("state_changed", _state_listener)

    # Battery transitions from -500 W to 0 W first (load drop)
    async_set_power(hass, "sensor.battery_power", 0)
    await hass.async_block_till_done()

    # Gross must NOT have published 1500 (phantom spike)
    assert 1500.0 not in published_gross

    # Grid then settles to 1000 W
    async_set_power(hass, "sensor.grid_power", 1000)
    await hass.async_block_till_done()

    # Now settled at 1000 W
    assert float(hass.states.get("sensor.powerguesser_gross_power").state) == 1000.0
    assert 1500.0 not in published_gross
    unsub()


async def test_stale_battery_after_91s_makes_sensors_unavailable_and_recovers(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    freezer: Any,
) -> None:
    """D-06: A re-reporting entity goes stale after 90s without updates, making gross & remainder unavailable."""
    energy_manager.data = std_prefs
    async_set_power(hass, "sensor.grid_power", 500)
    async_set_power(hass, "sensor.solar_power", 1200)
    async_set_power(hass, "sensor.battery_power", -300)
    async_set_power(hass, "sensor.washer_power", 200)

    entry = await async_setup_via_flow(hass)
    assert float(hass.states.get("sensor.powerguesser_gross_power").state) == 1400.0

    # Mark battery as a periodic re-reporting source by firing state_reported 3 times
    for _ in range(3):
        freezer.tick(timedelta(seconds=10))
        async_set_power(hass, "sensor.battery_power", -300)
        await hass.async_block_till_done()

    # Assert battery profiler has no published median (D-15 90 s default applies because
    # fewer than 20 battery deltas exist and hass_storage holds no profile; any future edit
    # that drives >= 20 deltas must move the expectation to 2.5x the median)
    assert entry.runtime_data.profilers["sensor.battery_power"].median_s is None

    # Advance time by 89s -> not stale yet
    freezer.tick(timedelta(seconds=89))
    async_fire_time_changed(hass, dt_util.utcnow(), fire_all=True)
    await hass.async_block_till_done()

    gross_state = hass.states.get("sensor.powerguesser_gross_power")
    remainder_state = hass.states.get("sensor.powerguesser_remainder_power")
    assert gross_state.state == "1400.0"
    assert remainder_state.state == "1200.0"

    # Advance by 2 more seconds (total 91s since last report) -> tick triggers staleness
    freezer.tick(timedelta(seconds=2))
    async_fire_time_changed(hass, dt_util.utcnow(), fire_all=True)
    await hass.async_block_till_done()

    gross_state = hass.states.get("sensor.powerguesser_gross_power")
    remainder_state = hass.states.get("sensor.powerguesser_remainder_power")
    assert gross_state.state == "unavailable"
    assert remainder_state.state == "unavailable"

    # Battery reports again -> hysteresis recovers immediately
    async_set_power(hass, "sensor.battery_power", -300)
    await hass.async_block_till_done()

    gross_state = hass.states.get("sensor.powerguesser_gross_power")
    remainder_state = hass.states.get("sensor.powerguesser_remainder_power")
    assert gross_state.state == "1400.0"
    assert remainder_state.state == "1200.0"
    assert gross_state.attributes.get("stale_sources") == []


async def test_single_rereport_keeps_source_held(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    freezer: Any,
) -> None:
    """A single re-report keeps source held as on-change, does not go stale after silence."""
    energy_manager.data = std_prefs
    async_set_power(hass, "sensor.grid_power", 500)
    async_set_power(hass, "sensor.solar_power", 1200)
    async_set_power(hass, "sensor.battery_power", -300)
    async_set_power(hass, "sensor.washer_power", 200)

    entry = await async_setup_via_flow(hass)
    assert float(hass.states.get("sensor.powerguesser_gross_power").state) == 1400.0

    # One single re-report
    freezer.tick(timedelta(seconds=10))
    async_set_power(hass, "sensor.battery_power", -300)
    await hass.async_block_till_done()

    # 300 s of silence
    freezer.tick(timedelta(seconds=300))
    async_fire_time_changed(hass, dt_util.utcnow(), fire_all=True)
    await hass.async_block_till_done()

    gross_state = hass.states.get("sensor.powerguesser_gross_power")
    assert gross_state.state == "1400.0"
    assert "sensor.battery_power" in gross_state.attributes.get("on_change_sources", [])
    assert "sensor.battery_power" not in gross_state.attributes.get("stale_sources", [])


async def test_rereport_flip_logs_info(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    freezer: Any,
    caplog: Any,
) -> None:
    """One INFO line naming only the entity id is logged when an entity flips to re-reporting, no power value."""
    caplog.set_level(logging.INFO)
    energy_manager.data = std_prefs
    async_set_power(hass, "sensor.grid_power", 500)
    async_set_power(hass, "sensor.solar_power", 1200)
    async_set_power(hass, "sensor.battery_power", -300)
    async_set_power(hass, "sensor.washer_power", 200)

    await async_setup_via_flow(hass)

    caplog.clear()

    # First report
    freezer.tick(timedelta(seconds=10))
    async_set_power(hass, "sensor.battery_power", -300)
    await hass.async_block_till_done()

    # Second report
    freezer.tick(timedelta(seconds=10))
    async_set_power(hass, "sensor.battery_power", -300)
    await hass.async_block_till_done()

    # No flip record yet
    flip_records = [
        r for r in caplog.records
        if "powerguesser" in r.name and "sensor.battery_power" in r.message and r.levelno == logging.INFO
    ]
    assert len(flip_records) == 0

    # Third report flips
    freezer.tick(timedelta(seconds=10))
    async_set_power(hass, "sensor.battery_power", -300)
    await hass.async_block_till_done()

    flip_records = [
        r for r in caplog.records
        if "powerguesser" in r.name and "sensor.battery_power" in r.message and r.levelno == logging.INFO
    ]
    assert len(flip_records) == 1
    msg = flip_records[0].message
    assert "sensor.battery_power" in msg
    # Never log the power value
    assert "-300" not in msg and "300" not in msg

    # Fourth report does not log again (sticky)
    freezer.tick(timedelta(seconds=10))
    async_set_power(hass, "sensor.battery_power", -300)
    await hass.async_block_till_done()

    flip_records = [
        r for r in caplog.records
        if "powerguesser" in r.name and "sensor.battery_power" in r.message and r.levelno == logging.INFO
    ]
    assert len(flip_records) == 1


async def test_rereport_classification_survives_rebuild(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    freezer: Any,
) -> None:
    """A re-reporting classification in self.modes survives hub rebuilds."""
    energy_manager.data = std_prefs
    async_set_power(hass, "sensor.grid_power", 500)
    async_set_power(hass, "sensor.solar_power", 1200)
    async_set_power(hass, "sensor.battery_power", -300)
    async_set_power(hass, "sensor.washer_power", 200)

    await async_setup_via_flow(hass)

    # 3 reports to flip battery
    for _ in range(3):
        freezer.tick(timedelta(seconds=10))
        async_set_power(hass, "sensor.battery_power", -300)
        await hass.async_block_till_done()

    # Trigger rebuild via energy_manager.async_update
    await energy_manager.async_update(std_prefs)
    await hass.async_block_till_done()

    # Now a single battery silence of 91s still makes gross unavailable
    freezer.tick(timedelta(seconds=91))
    async_fire_time_changed(hass, dt_util.utcnow(), fire_all=True)
    await hass.async_block_till_done()

    gross_state = hass.states.get("sensor.powerguesser_gross_power")
    assert gross_state.state == "unavailable"


async def test_solar_unavailable_at_night_keeps_gross_available(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
) -> None:
    """D-20: Solar entity becoming unavailable at night counts as 0 W and lists in solar_unavailable."""
    energy_manager.data = std_prefs
    async_set_power(hass, "sensor.grid_power", 500)
    async_set_power(hass, "sensor.solar_power", 1200)
    async_set_power(hass, "sensor.battery_power", -300)
    async_set_power(hass, "sensor.washer_power", 200)

    entry = await async_setup_via_flow(hass)
    assert float(hass.states.get("sensor.powerguesser_gross_power").state) == 1400.0

    # Solar inverter shuts down for the night
    async_set_power(hass, "sensor.solar_power", "unavailable")
    await hass.async_block_till_done()

    # Gross power should remain available with solar counted as 0 W (500 + 0 - 300 = 200 W)
    gross_state = hass.states.get("sensor.powerguesser_gross_power")
    remainder_state = hass.states.get("sensor.powerguesser_remainder_power")
    assert gross_state.state == "200.0"
    assert remainder_state.state == "0.0"  # 200 - 200 = 0 W
    assert gross_state.attributes.get("solar_unavailable") == ["sensor.solar_power"]


async def test_never_reporting_channel_idle_for_10_minutes_keeps_gross_available(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    freezer: Any,
) -> None:
    """D-19: A never-reporting entity (no state_reported emitted) idle for 10 minutes is not timed out."""
    energy_manager.data = std_prefs
    # Initial states without state_reported
    async_set_power(hass, "sensor.grid_power", 600)
    async_set_power(hass, "sensor.solar_power", 0)
    async_set_power(hass, "sensor.battery_power", 0)
    async_set_power(hass, "sensor.washer_power", 100)

    await async_setup_via_flow(hass)
    assert float(hass.states.get("sensor.powerguesser_gross_power").state) == 600.0

    # Advance time by 10 minutes (600s)
    freezer.tick(timedelta(minutes=10))
    async_fire_time_changed(hass, dt_util.utcnow(), fire_all=True)
    await hass.async_block_till_done()

    # Sensors must stay available (not timed out by on_tick)
    gross_state = hass.states.get("sensor.powerguesser_gross_power")
    assert gross_state.state == "600.0"

    # Grid updates to 700 W; since battery is on_change (idle > 45s), grid settles immediately
    async_set_power(hass, "sensor.grid_power", 700)
    await hass.async_block_till_done()

    gross_state = hass.states.get("sensor.powerguesser_gross_power")
    assert gross_state.state == "700.0"
    assert gross_state.attributes.get("stale_sources") == []
    assert "sensor.battery_power" in gross_state.attributes.get("on_change_sources", [])

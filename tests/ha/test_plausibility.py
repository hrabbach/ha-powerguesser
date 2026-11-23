"""Integration tests for sign self-check, plausibility attributes, and Repairs issues."""

from datetime import timedelta
from typing import Any

from homeassistant.components.energy.data import EnergyManager
from homeassistant.core import HomeAssistant
from homeassistant.helpers import issue_registry as ir
from homeassistant.util import dt as dt_util
from pytest_homeassistant.common import async_fire_time_changed

from custom_components.powerguesser.const import DOMAIN
from custom_components.powerguesser.sensor import (
    GrossPowerSensor,
    RemainderPowerSensor,
)
from tests.conftest import async_set_power, async_setup_via_flow


async def _emit_all(
    hass: HomeAssistant,
    freezer: Any,
    *,
    grid: float = 0.0,
    solar: float = 0.0,
    battery: float = 0.0,
    washer: float = 0.0,
    step_s: float = 30.0,
) -> None:
    """Advance time and report new values for all channels so settle releases an emission."""
    freezer.tick(timedelta(seconds=step_s))
    async_fire_time_changed(hass, dt_util.utcnow())
    async_set_power(hass, "sensor.grid_power", grid)
    async_set_power(hass, "sensor.solar_power", solar)
    async_set_power(hass, "sensor.battery_power", battery)
    async_set_power(hass, "sensor.washer_power", washer)
    await hass.async_block_till_done()


async def test_negative_gross_raises_warning_issue_and_returns_to_ok_on_next_emission(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    freezer: Any,
) -> None:
    """A battery with flipped sign producing gross -400 W raises WARNING issue; next valid emission has plausibility ok while issue persists."""
    energy_manager.data = std_prefs
    async_set_power(hass, "sensor.grid_power", 0)
    async_set_power(hass, "sensor.solar_power", 0)
    async_set_power(hass, "sensor.battery_power", -400)
    async_set_power(hass, "sensor.washer_power", 0)

    await async_setup_via_flow(hass)
    issue_reg = ir.async_get(hass)

    # 45 minutes of samples every 30 seconds = 90 samples
    for _ in range(90):
        await _emit_all(hass, freezer, grid=0, solar=0, battery=-400, washer=0, step_s=30)

    gross_state = hass.states.get("sensor.powerguesser_gross_power")
    assert gross_state is not None
    # Gross stays unclamped (D-11)
    assert float(gross_state.state) == -400.0
    assert gross_state.attributes.get("plausibility") == "negative_gross"

    # WARNING Repairs issue is raised
    issue = issue_reg.async_get_issue(DOMAIN, "implausible_negative_gross")
    assert issue is not None
    assert issue.severity == ir.IssueSeverity.WARNING

    # Once readings turn plausible again, the very next emission shows plausibility ok
    # while the issue is still present (attribute liveness vs issue hysteresis)
    await _emit_all(hass, freezer, grid=500, solar=0, battery=0, washer=0, step_s=30)

    gross_after = hass.states.get("sensor.powerguesser_gross_power")
    assert gross_after is not None
    assert float(gross_after.state) == 500.0
    assert gross_after.attributes.get("plausibility") == "ok"

    # Issue must STILL be present due to 12-hour clear hysteresis
    issue_still = issue_reg.async_get_issue(DOMAIN, "implausible_negative_gross")
    assert issue_still is not None


async def test_energy_prefs_update_clears_selfcheck_issue_immediately(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    freezer: Any,
) -> None:
    """An Energy dashboard preferences update clears self-check issues immediately."""
    energy_manager.data = std_prefs
    async_set_power(hass, "sensor.grid_power", 0)
    async_set_power(hass, "sensor.solar_power", 0)
    async_set_power(hass, "sensor.battery_power", -400)
    async_set_power(hass, "sensor.washer_power", 0)

    await async_setup_via_flow(hass)
    issue_reg = ir.async_get(hass)

    # Trigger implausible_negative_gross via 45 minutes of negative gross
    for _ in range(90):
        await _emit_all(hass, freezer, grid=0, solar=0, battery=-400, washer=0, step_s=30)

    assert issue_reg.async_get_issue(DOMAIN, "implausible_negative_gross") is not None

    # Editing Energy dashboard preferences resets the self-check and deletes the issue immediately
    new_prefs = dict(std_prefs)
    await energy_manager.async_update(new_prefs)
    await hass.async_block_till_done()

    assert issue_reg.async_get_issue(DOMAIN, "implausible_negative_gross") is None


async def test_metered_above_gross_raises_issue_and_clears_after_12h(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    freezer: Any,
) -> None:
    """Metered 900 W above gross 300 W raises implausible_metered_above_gross and clears after 12h."""
    energy_manager.data = std_prefs
    async_set_power(hass, "sensor.grid_power", 300)
    async_set_power(hass, "sensor.solar_power", 0)
    async_set_power(hass, "sensor.battery_power", 0)
    async_set_power(hass, "sensor.washer_power", 900)

    await async_setup_via_flow(hass)
    issue_reg = ir.async_get(hass)

    # 45 minutes of metered 900 W above gross 300 W -> raises issue
    for _ in range(90):
        await _emit_all(hass, freezer, grid=300, solar=0, battery=0, washer=900, step_s=30)

    gross_state = hass.states.get("sensor.powerguesser_gross_power")
    assert gross_state is not None
    assert float(gross_state.state) == 300.0
    assert gross_state.attributes.get("plausibility") == "metered_above_gross"

    issue = issue_reg.async_get_issue(DOMAIN, "implausible_metered_above_gross")
    assert issue is not None
    assert issue.severity == ir.IssueSeverity.WARNING

    # After 11 hours of plausible emissions (gross 1000 W, metered 200 W), issue is still present
    freezer.tick(timedelta(hours=11))
    async_fire_time_changed(hass, dt_util.utcnow())
    await _emit_all(hass, freezer, grid=1000, solar=0, battery=0, washer=200, step_s=30)

    assert issue_reg.async_get_issue(DOMAIN, "implausible_metered_above_gross") is not None

    # After full 12 hours (advance 1 more hour + 30s), issue is deleted
    freezer.tick(timedelta(hours=1, seconds=30))
    async_fire_time_changed(hass, dt_util.utcnow())
    await _emit_all(hass, freezer, grid=1000, solar=0, battery=0, washer=200, step_s=30)

    assert issue_reg.async_get_issue(DOMAIN, "implausible_metered_above_gross") is None


def test_plausibility_attribute_is_recorded() -> None:
    """plausibility does not appear in either sensor's _unrecorded_attributes."""
    assert "plausibility" not in GrossPowerSensor._unrecorded_attributes
    assert "plausibility" not in RemainderPowerSensor._unrecorded_attributes

"""Tests for Repairs issue reconciliation, fault handling, and sensor transforms."""

from datetime import timedelta
import json
from pathlib import Path
from typing import Any
import pytest

from homeassistant.components.energy.data import EnergyManager
from homeassistant.config_entries import ConfigEntryState
from homeassistant.core import HomeAssistant
from homeassistant.helpers import issue_registry as ir
from homeassistant.util import dt as dt_util
from pytest_homeassistant.common import MockConfigEntry, async_fire_time_changed

from custom_components.powerguesser.const import DOMAIN
from tests.conftest import async_set_power


@pytest.fixture
def variants() -> dict:
    path = Path(__file__).parent.parent / "fixtures" / "prefs_variants.json"
    return json.loads(path.read_text(encoding="utf-8"))


async def test_unsupported_power_config_raises_issue_and_sensors_unavailable(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    variants: dict,
) -> None:
    """Entry loads, sensors stay unavailable, and Repairs error issue is raised."""
    energy_manager.data = variants["mixed_shape"]
    async_set_power(hass, "sensor.grid_power", 500)
    async_set_power(hass, "sensor.grid_export_power", 200)

    entry = MockConfigEntry(domain=DOMAIN, data={})
    entry.add_to_hass(hass)
    await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    assert entry.state == ConfigEntryState.LOADED

    gross_state = hass.states.get("sensor.powerguesser_gross_power")
    remainder_state = hass.states.get("sensor.powerguesser_remainder_power")
    assert gross_state.state == "unavailable"
    assert remainder_state.state == "unavailable"

    issue_reg = ir.async_get(hass)
    issue = issue_reg.async_get_issue(
        DOMAIN, "unsupported_power_config_sensor_grid_import_energy"
    )
    assert issue is not None
    assert issue.severity == ir.IssueSeverity.ERROR
    assert issue.is_fixable is False


async def test_inverted_grid_transform(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    variants: dict,
) -> None:
    """Inverted grid sensor with raw 500 W yields gross -500 W."""
    energy_manager.data = variants["inverted"]
    async_set_power(hass, "sensor.grid_export_power", 500)

    entry = MockConfigEntry(domain=DOMAIN, data={})
    entry.add_to_hass(hass)
    await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    gross_state = hass.states.get("sensor.powerguesser_gross_power")
    assert gross_state is not None
    assert float(gross_state.state) == -500.0


async def test_two_sensor_battery_transform_and_refresh_gate(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    variants: dict,
) -> None:
    """Two-sensor battery (700 W from, 200 W to) yields +500 W and settles only when both refresh."""
    energy_manager.data = variants["two_sensor"]
    async_set_power(hass, "sensor.battery_discharge_power", 700)
    async_set_power(hass, "sensor.battery_charge_power", 200)

    entry = MockConfigEntry(domain=DOMAIN, data={})
    entry.add_to_hass(hass)
    await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    assert float(hass.states.get("sensor.powerguesser_gross_power").state) == 500.0

    # Discharge power changes to 900 W -> aligner holds old value until charge power refreshes
    async_set_power(hass, "sensor.battery_discharge_power", 900)
    await hass.async_block_till_done()
    assert float(hass.states.get("sensor.powerguesser_gross_power").state) == 500.0

    # Charge power re-reports unchanged 200 W
    async_set_power(hass, "sensor.battery_charge_power", 200)
    await hass.async_block_till_done()
    # Now settled -> 900 - 200 = 700 W
    assert float(hass.states.get("sensor.powerguesser_gross_power").state) == 700.0


async def test_source_not_found_structural_grace(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    variants: dict,
) -> None:
    """Missing gross entity creates source_not_found issue only after 60s grace expires."""
    energy_manager.data = variants["power_config_standard"]
    # sensor.grid_power has neither state nor registry entry

    entry = MockConfigEntry(domain=DOMAIN, data={})
    entry.add_to_hass(hass)
    await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    issue_reg = ir.async_get(hass)
    issue_id = "source_not_found_sensor_grid_power"

    # Immediately after setup, structural issue is withheld during grace period
    assert issue_reg.async_get_issue(DOMAIN, issue_id) is None

    # Advance clock past 60s
    async_fire_time_changed(hass, dt_util.utcnow() + timedelta(seconds=65))
    await hass.async_block_till_done()

    # Now the issue must be raised
    issue = issue_reg.async_get_issue(DOMAIN, issue_id)
    assert issue is not None
    assert issue.severity == ir.IssueSeverity.ERROR


async def test_unload_entry_removes_issues(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    variants: dict,
) -> None:
    """Unloading the config entry clears all active Repairs issues."""
    energy_manager.data = variants["mixed_shape"]
    async_set_power(hass, "sensor.grid_power", 500)
    async_set_power(hass, "sensor.grid_export_power", 200)

    entry = MockConfigEntry(domain=DOMAIN, data={})
    entry.add_to_hass(hass)
    await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    issue_reg = ir.async_get(hass)
    issue_id = "unsupported_power_config_sensor_grid_import_energy"
    assert issue_reg.async_get_issue(DOMAIN, issue_id) is not None

    await hass.config_entries.async_unload(entry.entry_id)
    await hass.async_block_till_done()

    assert issue_reg.async_get_issue(DOMAIN, issue_id) is None

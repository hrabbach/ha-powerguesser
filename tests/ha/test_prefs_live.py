"""Tests for live Energy dashboard edits and dynamic rebuilds without reloading."""

import copy
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch
import pytest

from homeassistant.components.energy.data import EnergyManager
from homeassistant.config_entries import ConfigEntryState
from homeassistant.core import HomeAssistant
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers import issue_registry as ir
from pytest_homeassistant.common import MockConfigEntry

from custom_components.powerguesser.const import DOMAIN
from tests.conftest import async_set_power


@pytest.fixture
def variants() -> dict:
    path = Path(__file__).parent.parent / "fixtures" / "prefs_variants.json"
    return json.loads(path.read_text(encoding="utf-8"))


async def test_live_add_battery_source(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    variants: dict,
) -> None:
    """1. Adding a battery source changes gross without reload."""
    prefs = copy.deepcopy(variants["standard"])
    # Start with only grid and solar
    prefs["energy_sources"] = [
        s for s in prefs["energy_sources"] if s["type"] in ("grid", "solar")
    ]
    energy_manager.data = prefs

    async_set_power(hass, "sensor.grid_power", 500)
    async_set_power(hass, "sensor.solar_power", 1000)
    async_set_power(hass, "sensor.battery_power", -200)

    entry = MockConfigEntry(domain=DOMAIN, data={})
    entry.add_to_hass(hass)
    await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    assert entry.state == ConfigEntryState.LOADED
    # Gross is grid + solar = 1500
    assert float(hass.states.get("sensor.powerguesser_gross_power").state) == 1500.0

    # Live update prefs: add battery source back
    new_prefs = copy.deepcopy(variants["standard"])
    await energy_manager.async_update(new_prefs)
    await hass.async_block_till_done()

    # Gross now includes battery (-200 W) -> 500 + 1000 - 200 = 1300
    assert entry.state == ConfigEntryState.LOADED
    assert float(hass.states.get("sensor.powerguesser_gross_power").state) == 1300.0


async def test_live_remove_solar_source(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    variants: dict,
) -> None:
    """2. Removing solar drops it from gross."""
    energy_manager.data = copy.deepcopy(variants["standard"])
    async_set_power(hass, "sensor.grid_power", 500)
    async_set_power(hass, "sensor.solar_power", 1000)
    async_set_power(hass, "sensor.battery_power", 0)

    entry = MockConfigEntry(domain=DOMAIN, data={})
    entry.add_to_hass(hass)
    await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    assert float(hass.states.get("sensor.powerguesser_gross_power").state) == 1500.0

    # Remove solar
    new_prefs = copy.deepcopy(variants["standard"])
    new_prefs["energy_sources"] = [
        s for s in new_prefs["energy_sources"] if s["type"] != "solar"
    ]
    await energy_manager.async_update(new_prefs)
    await hass.async_block_till_done()

    # Gross drops to 500 + 0 = 500
    assert float(hass.states.get("sensor.powerguesser_gross_power").state) == 500.0


async def test_live_switch_to_unsupported_and_back(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    variants: dict,
) -> None:
    """3. Switching grid to unsupported raises ERROR issue; switching back restores values."""
    energy_manager.data = copy.deepcopy(variants["standard"])
    async_set_power(hass, "sensor.grid_power", 500)
    async_set_power(hass, "sensor.solar_power", 1000)
    async_set_power(hass, "sensor.battery_power", 0)

    entry = MockConfigEntry(domain=DOMAIN, data={})
    entry.add_to_hass(hass)
    await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    assert float(hass.states.get("sensor.powerguesser_gross_power").state) == 1500.0
    issue_reg = ir.async_get(hass)
    issue_id = "unsupported_power_config_sensor_grid_import_energy"
    assert issue_reg.async_get_issue(DOMAIN, issue_id) is None

    # Live update to unsupported mixed shape
    await energy_manager.async_update(copy.deepcopy(variants["mixed_shape"]))
    await hass.async_block_till_done()

    # Sensors must be unavailable and issue present
    assert hass.states.get("sensor.powerguesser_gross_power").state == "unavailable"
    assert issue_reg.async_get_issue(DOMAIN, issue_id) is not None

    # Switch back to standard
    await energy_manager.async_update(copy.deepcopy(variants["standard"]))
    await hass.async_block_till_done()

    # Issue deleted and values restored
    assert issue_reg.async_get_issue(DOMAIN, issue_id) is None
    assert float(hass.states.get("sensor.powerguesser_gross_power").state) == 1500.0


async def test_singleton_listener_across_reloads(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    variants: dict,
) -> None:
    """4. Reloading the entry three times leaves exactly one listener from energy_prefs."""
    energy_manager.data = copy.deepcopy(variants["standard"])
    async_set_power(hass, "sensor.grid_power", 500)
    async_set_power(hass, "sensor.solar_power", 1000)
    async_set_power(hass, "sensor.battery_power", 0)

    entry = MockConfigEntry(domain=DOMAIN, data={})
    entry.add_to_hass(hass)
    await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    for _ in range(3):
        await hass.config_entries.async_reload(entry.entry_id)
        await hass.async_block_till_done()

    powerguesser_listeners = [
        fn
        for fn in energy_manager._update_listeners
        if fn.__module__ == "custom_components.powerguesser.energy_prefs"
    ]
    assert len(powerguesser_listeners) == 1


async def test_listener_exception_does_not_break_save(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    variants: dict,
) -> None:
    """5. Patching async_dispatcher_send in energy_prefs to raise lets async_update succeed."""
    energy_manager.data = copy.deepcopy(variants["standard"])
    async_set_power(hass, "sensor.grid_power", 500)
    async_set_power(hass, "sensor.solar_power", 1000)
    async_set_power(hass, "sensor.battery_power", 0)

    entry = MockConfigEntry(domain=DOMAIN, data={})
    entry.add_to_hass(hass)
    await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    with patch(
        "custom_components.powerguesser.energy_prefs.async_dispatcher_send",
        side_effect=RuntimeError("simulated dispatcher failure"),
    ):
        # Must not raise
        await energy_manager.async_update(copy.deepcopy(variants["standard"]))
        await hass.async_block_till_done()


async def test_entity_registry_update_triggers_rebuild(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    variants: dict,
) -> None:
    """6. An entity-registry update event for a tracked entity id triggers async_rebuild."""
    energy_manager.data = copy.deepcopy(variants["standard"])
    async_set_power(hass, "sensor.grid_power", 500)
    async_set_power(hass, "sensor.solar_power", 1000)
    async_set_power(hass, "sensor.battery_power", 0)

    entry = MockConfigEntry(domain=DOMAIN, data={})
    entry.add_to_hass(hass)
    await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    hub = entry.runtime_data
    with patch.object(hub, "async_request_rebuild", wraps=hub.async_request_rebuild) as spy:
        hass.bus.async_fire(
            er.EVENT_ENTITY_REGISTRY_UPDATED,
            {"action": "update", "entity_id": "sensor.grid_power"},
        )
        await hass.async_block_till_done()
        assert spy.call_count == 1

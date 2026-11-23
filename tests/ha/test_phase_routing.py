"""Tests for phase routing invariants and gross/remainder independence."""

from typing import Any
import pytest

from homeassistant.components.energy.data import EnergyManager
from homeassistant.core import Event, HomeAssistant
from homeassistant.helpers import entity_registry as er
from pytest_homeassistant.common import MockConfigEntry

from custom_components.powerguesser.const import (
    CONF_PHASE_L1,
    CONF_PHASE_L2,
    CONF_PHASE_L3,
    DOMAIN,
)
from tests.conftest import async_set_power

# Module-level dictionary to hold results across parametrized runs
_RECORDED_RUNS: dict[bool, tuple[list[tuple[str, str]], list[tuple[str, str]]]] = {}


@pytest.mark.parametrize("phases_mapped", [False, True])
async def test_gross_and_remainder_identical_with_and_without_phases(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    phases_mapped: bool,
) -> None:
    """Invariant: gross and remainder states are identical with and without phases mapped (D-16)."""
    energy_manager.data = std_prefs

    # Set up entity registry for candidate entities
    ent_reg = er.async_get(hass)
    for eid in ("sensor.grid_power", "sensor.solar_power", "sensor.battery_power", "sensor.washer_power"):
        ent_reg.async_get_or_create("sensor", "test", eid)
    for eid in ("sensor.l1_power", "sensor.l2_power", "sensor.l3_power"):
        ent_reg.async_get_or_create("sensor", "test", eid)

    # Initial states
    async_set_power(hass, "sensor.grid_power", 500)
    async_set_power(hass, "sensor.solar_power", 1000)
    async_set_power(hass, "sensor.battery_power", -200)
    async_set_power(hass, "sensor.washer_power", 100)
    async_set_power(hass, "sensor.l1_power", 200)
    async_set_power(hass, "sensor.l2_power", 150)
    async_set_power(hass, "sensor.l3_power", 150)

    options = (
        {
            CONF_PHASE_L1: "sensor.l1_power",
            CONF_PHASE_L2: "sensor.l2_power",
            CONF_PHASE_L3: "sensor.l3_power",
        }
        if phases_mapped
        else {}
    )

    entry = MockConfigEntry(domain=DOMAIN, data={}, options=options)
    entry.add_to_hass(hass)
    await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    gross_events: list[tuple[str, str]] = []
    remainder_events: list[tuple[str, str]] = []

    def _state_listener(event: Event) -> None:
        entity_id = event.data.get("entity_id")
        new_state = event.data.get("new_state")
        if new_state is None:
            return
        entry = (new_state.state, str(new_state.attributes.get("available", True)))
        if entity_id == "sensor.powerguesser_gross_power":
            gross_events.append(entry)
        elif entity_id == "sensor.powerguesser_remainder_power":
            remainder_events.append(entry)

    unsub = hass.bus.async_listen("state_changed", _state_listener)

    # Sequence of updates: grid, solar, battery, washer, and phase changes (including unavailable phase)
    sequence = [
        ("sensor.grid_power", 800),
        ("sensor.l1_power", 400),  # Phase change only -> gross/remainder must not publish
        ("sensor.solar_power", 1500),
        ("sensor.washer_power", 300),
        ("sensor.l2_power", 250),  # Phase change only
        ("sensor.battery_power", 100),
        ("sensor.l1_power", "unavailable"),  # Phase unavailable -> gross/remainder must NOT become unavailable
        ("sensor.grid_power", 1200),
        ("sensor.l3_power", 300),  # Phase change only
        ("sensor.l1_power", 350),  # Phase recovers
        ("sensor.washer_power", 0),
        ("sensor.solar_power", 0),
        ("sensor.battery_power", -400),
    ]

    for entity_id, val in sequence:
        async_set_power(hass, entity_id, val)
        await hass.async_block_till_done()

    unsub()

    # Verify that gross and remainder published valid states
    assert len(gross_events) > 0
    assert len(remainder_events) > 0

    _RECORDED_RUNS[phases_mapped] = (list(gross_events), list(remainder_events))

    # When both parametrized runs have completed, compare their event histories
    if len(_RECORDED_RUNS) == 2:
        without_phases_gross, without_phases_remainder = _RECORDED_RUNS[False]
        with_phases_gross, with_phases_remainder = _RECORDED_RUNS[True]
        assert without_phases_gross == with_phases_gross
        assert without_phases_remainder == with_phases_remainder

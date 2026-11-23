"""Tests for remainder sensor clamping, over-attribution, and unrecorded attributes."""

from datetime import timedelta
from typing import Any
import pytest

from homeassistant.components.energy.data import EnergyManager
from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util
from pytest_homeassistant.common import MockConfigEntry, async_fire_time_changed

from custom_components.powerguesser.const import DOMAIN
from custom_components.powerguesser.sensor import (
    UNRECORDED_ATTRIBUTES,
    GrossPowerSensor,
    RemainderPowerSensor,
)
from tests.conftest import async_set_power


def test_unrecorded_attributes_configuration() -> None:
    """Both sensor classes carry UNRECORDED_ATTRIBUTES without plausibility."""
    assert "source_age_s" in UNRECORDED_ATTRIBUTES
    assert "stale_sources" in UNRECORDED_ATTRIBUTES
    assert "over_attribution" in UNRECORDED_ATTRIBUTES
    assert "clipped" in UNRECORDED_ATTRIBUTES
    assert "metered_w" in UNRECORDED_ATTRIBUTES
    assert "metered_unavailable" in UNRECORDED_ATTRIBUTES
    assert "plausibility" not in UNRECORDED_ATTRIBUTES

    assert GrossPowerSensor._unrecorded_attributes == UNRECORDED_ATTRIBUTES
    assert RemainderPowerSensor._unrecorded_attributes == UNRECORDED_ATTRIBUTES


async def test_remainder_clamping_and_over_attribution_attribute(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    freezer: Any,
) -> None:
    """Gross 1000 W with 1500 W metered -> remainder 0, clipped True, signed -500 W; 60s later over_attribution is 8.33."""
    energy_manager.data = {
        "energy_sources": [
            {
                "type": "grid",
                "stat_energy_from": "sensor.grid_import_energy",
                "stat_rate": "sensor.grid_power",
            }
        ],
        "device_consumption": [
            {
                "stat_consumption": "sensor.plug_energy",
                "stat_rate": "sensor.plug_power",
                "name": "Heavy Load",
            }
        ],
    }

    async_set_power(hass, "sensor.grid_power", 1000)
    async_set_power(hass, "sensor.plug_power", 1500)

    entry = MockConfigEntry(domain=DOMAIN, data={})
    entry.add_to_hass(hass)
    await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    remainder = hass.states.get("sensor.powerguesser_remainder_power")
    assert remainder is not None
    assert float(remainder.state) == 0.0
    assert remainder.attributes["clipped"] is True
    assert remainder.attributes["over_attribution"] == 0.0

    hub = entry.runtime_data
    assert hub.coordinator.data.signed_residual_w == -500.0

    # Advance time by 60 seconds (freezer + re-report) to trigger next emission
    freezer.tick(timedelta(seconds=60))
    async_fire_time_changed(hass, dt_util.utcnow())
    async_set_power(hass, "sensor.grid_power", 1000)
    await hass.async_block_till_done()

    remainder_after = hass.states.get("sensor.powerguesser_remainder_power")
    assert float(remainder_after.state) == 0.0
    assert remainder_after.attributes["clipped"] is True
    assert remainder_after.attributes["over_attribution"] == 8.33

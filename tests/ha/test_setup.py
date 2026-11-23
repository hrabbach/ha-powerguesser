"""Tests for PowerGuesser setup and walking skeleton end-to-end tracer."""

from typing import Any
import pytest

from homeassistant.components.energy.data import EnergyManager
from homeassistant.core import HomeAssistant

from custom_components.powerguesser.const import DOMAIN
from tests.conftest import async_set_power, async_setup_via_flow


async def test_setup_happy_path(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
) -> None:
    """(a) Flow creates entry with data {} and exposes gross and remainder W sensors."""
    energy_manager.data = std_prefs
    async_set_power(hass, "sensor.grid_power", 500)
    async_set_power(hass, "sensor.solar_power", 1200)
    async_set_power(hass, "sensor.battery_power", -300)
    async_set_power(hass, "sensor.washer_power", 200)

    entry = await async_setup_via_flow(hass)
    assert entry.data == {}

    gross_state = hass.states.get("sensor.powerguesser_gross_power")
    remainder_state = hass.states.get("sensor.powerguesser_remainder_power")
    assert gross_state is not None
    assert remainder_state is not None
    assert float(gross_state.state) == 1400.0
    assert float(remainder_state.state) == 1200.0


async def test_settle_alignment_with_state_reported(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
) -> None:
    """(b) Gross holds previous value until all gross sources refresh (via state_reported)."""
    energy_manager.data = std_prefs
    async_set_power(hass, "sensor.grid_power", 500)
    async_set_power(hass, "sensor.solar_power", 1200)
    async_set_power(hass, "sensor.battery_power", -300)
    async_set_power(hass, "sensor.washer_power", 200)

    await async_setup_via_flow(hass)
    assert float(hass.states.get("sensor.powerguesser_gross_power").state) == 1400.0

    # Grid changes to 800 W (fires state_changed)
    async_set_power(hass, "sensor.grid_power", 800)
    await hass.async_block_till_done()

    # Gross must keep 1400 until solar and battery re-report
    assert float(hass.states.get("sensor.powerguesser_gross_power").state) == 1400.0

    # Solar re-reports unchanged 1200 W (fires state_reported)
    async_set_power(hass, "sensor.solar_power", 1200)
    await hass.async_block_till_done()
    assert float(hass.states.get("sensor.powerguesser_gross_power").state) == 1400.0

    # Battery re-reports unchanged -300 W (fires state_reported)
    async_set_power(hass, "sensor.battery_power", -300)
    await hass.async_block_till_done()

    # Now all three settled -> gross reads 800 + 1200 - 300 = 1700
    assert float(hass.states.get("sensor.powerguesser_gross_power").state) == 1700.0
    assert float(hass.states.get("sensor.powerguesser_remainder_power").state) == 1500.0


async def test_no_energy_prefs_aborts(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
) -> None:
    """(c) Prefs with no sources abort with no_energy_prefs and create no entry."""
    energy_manager.data = EnergyManager.default_preferences()
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": "user"}
    )
    assert result["type"] == "abort"
    assert result["reason"] == "no_energy_prefs"
    assert not hass.config_entries.async_entries(DOMAIN)


async def test_single_instance_allowed(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
) -> None:
    """(d) A second config flow is aborted by single_instance_allowed."""
    energy_manager.data = std_prefs
    async_set_power(hass, "sensor.grid_power", 500)
    async_set_power(hass, "sensor.solar_power", 1200)
    async_set_power(hass, "sensor.battery_power", -300)
    async_set_power(hass, "sensor.washer_power", 200)
    await async_setup_via_flow(hass)
    assert len(hass.config_entries.async_entries(DOMAIN)) == 1

    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": "user"}
    )
    assert result["type"] == "abort"
    assert result["reason"] == "single_instance_allowed"


@pytest.mark.parametrize("bad_val", ["nan", "abc", "unavailable"])
async def test_unavailable_gross_fails_immediately_and_recovers_after_settle(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    bad_val: str,
) -> None:
    """(e) Bad/unavailable gross makes both sensors unavailable immediately, recovers on full settle."""
    energy_manager.data = std_prefs
    async_set_power(hass, "sensor.grid_power", 500)
    async_set_power(hass, "sensor.solar_power", 1200)
    async_set_power(hass, "sensor.battery_power", -300)
    async_set_power(hass, "sensor.washer_power", 200)

    await async_setup_via_flow(hass)
    assert float(hass.states.get("sensor.powerguesser_gross_power").state) == 1400.0

    # Grid becomes unavailable or invalid
    async_set_power(hass, "sensor.grid_power", bad_val)
    await hass.async_block_till_done()

    # Both sensors must immediately be unavailable on that same event
    gross_state = hass.states.get("sensor.powerguesser_gross_power")
    remainder_state = hass.states.get("sensor.powerguesser_remainder_power")
    assert gross_state.state == "unavailable"
    assert remainder_state.state == "unavailable"

    # Grid reports a valid number again
    async_set_power(hass, "sensor.grid_power", 500)
    await hass.async_block_till_done()

    # Still unavailable because solar and battery have not refreshed!
    assert hass.states.get("sensor.powerguesser_gross_power").state == "unavailable"

    # Solar refreshes
    async_set_power(hass, "sensor.solar_power", 1200)
    await hass.async_block_till_done()
    assert hass.states.get("sensor.powerguesser_gross_power").state == "unavailable"

    # Battery refreshes -> all settled -> values return
    async_set_power(hass, "sensor.battery_power", -300)
    await hass.async_block_till_done()
    assert float(hass.states.get("sensor.powerguesser_gross_power").state) == 1400.0
    assert float(hass.states.get("sensor.powerguesser_remainder_power").state) == 1200.0


async def test_kw_unit_conversion(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
) -> None:
    """(f) A kW grid state (0.5 kW) counts as 500 W in gross and remainder."""
    energy_manager.data = std_prefs
    async_set_power(hass, "sensor.grid_power", 0.5, unit="kW")
    async_set_power(hass, "sensor.solar_power", 0)
    async_set_power(hass, "sensor.battery_power", 0)
    async_set_power(hass, "sensor.washer_power", 0)

    await async_setup_via_flow(hass)
    assert float(hass.states.get("sensor.powerguesser_gross_power").state) == 500.0
    assert float(hass.states.get("sensor.powerguesser_remainder_power").state) == 500.0


async def test_solar_negative_clamped_to_zero(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
) -> None:
    """(g) Solar -15 W at night counts as 0 in gross."""
    energy_manager.data = std_prefs
    async_set_power(hass, "sensor.grid_power", 500)
    async_set_power(hass, "sensor.solar_power", -15)
    async_set_power(hass, "sensor.battery_power", 0)
    async_set_power(hass, "sensor.washer_power", 0)

    await async_setup_via_flow(hass)
    # Solar -15 is clamped to 0 -> gross is 500 + 0 + 0 = 500
    assert float(hass.states.get("sensor.powerguesser_gross_power").state) == 500.0

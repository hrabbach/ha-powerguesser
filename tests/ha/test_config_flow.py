"""Tests for PowerGuesser config flow and abort rules."""

import json
from pathlib import Path
from typing import Any
import pytest

from homeassistant.components.energy.data import EnergyManager
from homeassistant.core import CoreState, HomeAssistant
from homeassistant.helpers import entity_registry as er

from custom_components.powerguesser.const import DOMAIN
from custom_components.powerguesser.engine.prefs import ABORT_ORDER
from tests.conftest import async_set_power, async_setup_via_flow


@pytest.fixture
def variants() -> dict:
    path = Path(__file__).parent.parent / "fixtures" / "prefs_variants.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_abort_order_tuple() -> None:
    """ABORT_ORDER must match the exact D-09 priority tuple."""
    assert ABORT_ORDER == (
        "no_energy_prefs",
        "missing_stat_rate",
        "unsupported_power_config",
        "source_not_found",
        "invalid_unit",
    )


async def test_abort_no_energy_prefs(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    variants: dict,
) -> None:
    """Flow aborts with no_energy_prefs when no usable energy sources exist."""
    energy_manager.data = variants["no_sources"]
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": "user"}
    )
    assert result["type"] == "abort"
    assert result["reason"] == "no_energy_prefs"


async def test_abort_missing_stat_rate(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    variants: dict,
) -> None:
    """Flow aborts with missing_stat_rate and names the source."""
    energy_manager.data = variants["missing_stat_rate_grid"]
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": "user"}
    )
    assert result["type"] == "abort"
    assert result["reason"] == "missing_stat_rate"
    assert "source" in result["description_placeholders"]
    assert "Grid" in result["description_placeholders"]["source"]


async def test_abort_unsupported_power_config(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    variants: dict,
) -> None:
    """Flow aborts with unsupported_power_config on mixed power_config keys."""
    energy_manager.data = variants["mixed_shape"]
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": "user"}
    )
    assert result["type"] == "abort"
    assert result["reason"] == "unsupported_power_config"
    assert "source" in result["description_placeholders"]


async def test_abort_source_not_found(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    variants: dict,
) -> None:
    """Flow aborts with source_not_found when a referenced entity does not exist."""
    energy_manager.data = variants["power_config_standard"]
    # sensor.grid_power has neither state nor registry entry
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": "user"}
    )
    assert result["type"] == "abort"
    assert result["reason"] == "source_not_found"
    assert result["description_placeholders"]["source"] == "sensor.grid_power"


async def test_abort_invalid_unit(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    variants: dict,
) -> None:
    """Flow aborts with invalid_unit when entity reports a non-power unit."""
    energy_manager.data = variants["power_config_standard"]
    hass.states.async_set("sensor.grid_power", "100", {"unit_of_measurement": "apples"})
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": "user"}
    )
    assert result["type"] == "abort"
    assert result["reason"] == "invalid_unit"
    assert result["description_placeholders"]["source"] == "sensor.grid_power"


async def test_multi_fault_priority_order(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
) -> None:
    """With both missing_stat_rate and unsupported_power_config, missing_stat_rate wins."""
    energy_manager.data = {
        "energy_sources": [
            {
                "type": "grid",
                "stat_energy_from": "sensor.grid_a_energy",
                "power_config": {
                    "stat_rate": "sensor.grid_a_power",
                    "stat_rate_inverted": "sensor.grid_a_inverted",
                },
            },
            {
                "type": "solar",
                "stat_energy_from": "sensor.solar_energy",
            },
        ],
        "device_consumption": [],
    }
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": "user"}
    )
    assert result["type"] == "abort"
    assert result["reason"] == "missing_stat_rate"


async def test_home_source_overrides_grid_missing_stat_rate(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    variants: dict,
) -> None:
    """Home source with stat_rate bypasses grid faults and shows the setup form."""
    energy_manager.data = variants["home_source"]
    async_set_power(hass, "sensor.home_power", 1500)

    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": "user"}
    )
    assert result["type"] == "form"
    assert result["step_id"] == "user"
    assert len(result["data_schema"].schema) == 0


async def test_single_instance_abort(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
) -> None:
    """A second config flow while an entry exists aborts with single_instance_allowed."""
    energy_manager.data = std_prefs
    async_set_power(hass, "sensor.grid_power", 500)
    async_set_power(hass, "sensor.solar_power", 1200)
    async_set_power(hass, "sensor.battery_power", -300)
    async_set_power(hass, "sensor.washer_power", 200)
    await async_setup_via_flow(hass)

    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": "user"}
    )
    assert result["type"] == "abort"
    assert result["reason"] == "single_instance_allowed"


async def test_entity_in_registry_without_state_passes(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    variants: dict,
) -> None:
    """An entity registered in the entity registry with no state yet shows the form."""
    energy_manager.data = variants["power_config_standard"]
    ent_reg = er.async_get(hass)
    ent_reg.async_get_or_create(
        domain="sensor",
        platform="template",
        unique_id="test_grid_unique_1",
        suggested_object_id="grid_power",
        unit_of_measurement="W",
    )
    # Confirm hass has no state for sensor.grid_power
    assert hass.states.get("sensor.grid_power") is None

    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": "user"}
    )
    assert result["type"] == "form"
    assert result["step_id"] == "user"


async def test_source_not_found_skipped_during_startup(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    variants: dict,
) -> None:
    """During startup (CoreState.starting), source_not_found is skipped and form is shown."""
    energy_manager.data = variants["power_config_standard"]
    # In running state, missing entity aborts
    result_running = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": "user"}
    )
    assert result_running["type"] == "abort"
    assert result_running["reason"] == "source_not_found"

    # Switch state to starting
    hass.set_state(CoreState.starting)
    result_starting = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": "user"}
    )
    assert result_starting["type"] == "form"
    assert result_starting["step_id"] == "user"

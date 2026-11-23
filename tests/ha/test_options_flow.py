"""Tests for PowerGuesser options flow."""

from typing import Any
import pytest

from homeassistant.components.energy.data import EnergyManager
from homeassistant.config_entries import ConfigEntryState
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType
from homeassistant.helpers import device_registry as dr, entity_registry as er, issue_registry as ir
from pytest_homeassistant.common import MockConfigEntry

from custom_components.powerguesser.const import (
    BUFFER_DIR,
    BUFFER_FILE,
    CONF_INVERT_PHASES,
    CONF_PHASE_L1,
    CONF_PHASE_L2,
    CONF_PHASE_L3,
    DOMAIN,
)
from custom_components.powerguesser.engine.reader import load_phase_sign, open_readonly
from tests.conftest import (
    async_flush_buffer,
    async_open_options_step,
    async_read_buffer,
    async_set_power,
    async_setup_via_flow,
)


async def test_options_flow_aborts_when_not_loaded(
    recorder_mock: Any,
    hass: HomeAssistant,
) -> None:
    """Options flow aborts with not_loaded when config entry is not loaded."""
    entry = MockConfigEntry(domain=DOMAIN, data={}, options={})
    entry.add_to_hass(hass)
    assert entry.state is ConfigEntryState.NOT_LOADED

    result = await hass.config_entries.options.async_init(entry.entry_id)
    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "not_loaded"


async def test_map_l1_buffers_phase_channel(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
) -> None:
    """Tracer test: mapping L1 in options flow buffers per-phase rows live without reload."""
    energy_manager.data = std_prefs
    async_set_power(hass, "sensor.grid_power", 500)
    async_set_power(hass, "sensor.solar_power", 1200)
    async_set_power(hass, "sensor.battery_power", -300)
    async_set_power(hass, "sensor.washer_power", 200)

    # L1 power equals grid power so sum check is OK
    async_set_power(hass, "sensor.l1_power", 500)

    entry = await async_setup_via_flow(hass)
    original_hub = entry.runtime_data
    assert entry.state is ConfigEntryState.LOADED

    # Open options flow
    result = await async_open_options_step(hass, entry, "phases")
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "phases"

    # Submit mapping for L1 only
    result2 = await hass.config_entries.options.async_configure(
        result["flow_id"],
        user_input={CONF_PHASE_L1: "sensor.l1_power"},
    )
    assert result2["type"] is FlowResultType.CREATE_ENTRY
    await hass.async_block_till_done()

    # Assert options saved, entry remains loaded, and hub object was preserved (no reload)
    assert entry.options == {CONF_PHASE_L1: "sensor.l1_power"}
    assert entry.state is ConfigEntryState.LOADED
    assert entry.runtime_data is original_hub

    # Update sensor.l1_power and flush buffer
    async_set_power(hass, "sensor.l1_power", 380)
    await hass.async_block_till_done()
    await async_flush_buffer(hass, entry)

    # Verify channel kind in SQLite
    def _read_channels() -> list[tuple[str, str, str]]:
        path = hass.config.path(BUFFER_DIR, BUFFER_FILE)
        conn = open_readonly(path)
        try:
            cur = conn.execute("SELECT entity_id, role, kind FROM channels")
            return cur.fetchall()
        finally:
            conn.close()

    channels = await hass.async_add_executor_job(_read_channels)
    assert ("sensor.l1_power", "l1", "phase") in channels

    # Verify buffer row recorded
    samples_by_ch, _ = await async_read_buffer(hass)
    assert ("sensor.l1_power", "l1") in samples_by_ch
    rows = [val for _, val in samples_by_ch[("sensor.l1_power", "l1")]]
    assert 380 in rows

    # Also verify SampleIntervalSensor was dynamically registered
    interval_sensor = hass.states.get("sensor.powerguesser_l1_power_sample_interval")
    assert interval_sensor is not None


async def test_options_flow_clear_phase(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
) -> None:
    """Clearing a phase mapping removes it from entry.options."""
    energy_manager.data = std_prefs
    async_set_power(hass, "sensor.grid_power", 500)
    async_set_power(hass, "sensor.solar_power", 1200)
    async_set_power(hass, "sensor.battery_power", -300)
    async_set_power(hass, "sensor.washer_power", 200)
    async_set_power(hass, "sensor.l1_power", 500)

    entry = await async_setup_via_flow(hass)

    # Set phase_l1
    result = await async_open_options_step(hass, entry, "phases")
    await hass.config_entries.options.async_configure(
        result["flow_id"],
        user_input={CONF_PHASE_L1: "sensor.l1_power"},
    )
    await hass.async_block_till_done()
    assert entry.options == {CONF_PHASE_L1: "sensor.l1_power"}

    # Clear phase_l1 by submitting empty dict (field cleared)
    result = await async_open_options_step(hass, entry, "phases")
    await hass.config_entries.options.async_configure(
        result["flow_id"],
        user_input={},
    )
    await hass.async_block_till_done()
    assert entry.options == {}


async def test_options_flow_validation_duplicate_phase(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
) -> None:
    """Mapping the same entity to multiple phases returns duplicate_phase error."""
    energy_manager.data = std_prefs
    async_set_power(hass, "sensor.grid_power", 500)
    async_set_power(hass, "sensor.solar_power", 1200)
    async_set_power(hass, "sensor.battery_power", -300)
    async_set_power(hass, "sensor.washer_power", 200)
    async_set_power(hass, "sensor.l1_power", 250)

    entry = await async_setup_via_flow(hass)

    result = await async_open_options_step(hass, entry, "phases")
    result2 = await hass.config_entries.options.async_configure(
        result["flow_id"],
        user_input={
            CONF_PHASE_L1: "sensor.l1_power",
            CONF_PHASE_L2: "sensor.l1_power",
        },
    )
    assert result2["type"] is FlowResultType.FORM
    assert result2["errors"] == {"base": "duplicate_phase"}


async def test_options_flow_validation_self_reference(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
) -> None:
    """Mapping a PowerGuesser entity returns phase_self_reference error."""
    energy_manager.data = std_prefs
    async_set_power(hass, "sensor.grid_power", 500)
    async_set_power(hass, "sensor.solar_power", 1200)
    async_set_power(hass, "sensor.battery_power", -300)
    async_set_power(hass, "sensor.washer_power", 200)

    entry = await async_setup_via_flow(hass)

    result = await async_open_options_step(hass, entry, "phases")
    result2 = await hass.config_entries.options.async_configure(
        result["flow_id"],
        user_input={CONF_PHASE_L1: "sensor.powerguesser_gross_power"},
    )
    assert result2["type"] is FlowResultType.FORM
    assert result2["errors"] == {CONF_PHASE_L1: "phase_self_reference"}


async def test_options_flow_validation_is_source(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
) -> None:
    """Mapping a gross or metered entity returns phase_is_source error."""
    energy_manager.data = std_prefs
    async_set_power(hass, "sensor.grid_power", 500)
    async_set_power(hass, "sensor.solar_power", 1200)
    async_set_power(hass, "sensor.battery_power", -300)
    async_set_power(hass, "sensor.washer_power", 200)

    entry = await async_setup_via_flow(hass)

    result = await async_open_options_step(hass, entry, "phases")
    result2 = await hass.config_entries.options.async_configure(
        result["flow_id"],
        user_input={
            CONF_PHASE_L1: "sensor.grid_power",
            CONF_PHASE_L2: "sensor.washer_power",
        },
    )
    assert result2["type"] is FlowResultType.FORM
    assert result2["errors"] == {
        CONF_PHASE_L1: "phase_is_source",
        CONF_PHASE_L2: "phase_is_source",
    }


async def test_options_flow_phase_check_inverted(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
) -> None:
    """Phases summing to -grid show phase_check with invert_phases default True."""
    energy_manager.data = std_prefs
    async_set_power(hass, "sensor.grid_power", 1000)
    async_set_power(hass, "sensor.solar_power", 1200)
    async_set_power(hass, "sensor.battery_power", -300)
    async_set_power(hass, "sensor.washer_power", 200)

    async_set_power(hass, "sensor.l1_power", -300)
    async_set_power(hass, "sensor.l2_power", -300)
    async_set_power(hass, "sensor.l3_power", -400)

    entry = await async_setup_via_flow(hass)

    result = await async_open_options_step(hass, entry, "phases")
    result2 = await hass.config_entries.options.async_configure(
        result["flow_id"],
        user_input={
            CONF_PHASE_L1: "sensor.l1_power",
            CONF_PHASE_L2: "sensor.l2_power",
            CONF_PHASE_L3: "sensor.l3_power",
        },
    )
    assert result2["type"] is FlowResultType.FORM
    assert result2["step_id"] == "phase_check"
    assert result2["description_placeholders"]["phase_sum"] == "-1000"
    assert result2["description_placeholders"]["grid"] == "1000"
    assert result2["description_placeholders"]["result"] == "inverted"

    # Default for invert_phases is True
    marker = [k for k in result2["data_schema"].schema if k == CONF_INVERT_PHASES][0]
    assert marker.default() is True

    # Confirm with invert_phases=True
    result3 = await hass.config_entries.options.async_configure(
        result2["flow_id"],
        user_input={CONF_INVERT_PHASES: True},
    )
    assert result3["type"] is FlowResultType.CREATE_ENTRY
    await hass.async_block_till_done()

    assert entry.options[CONF_INVERT_PHASES] is True
    assert entry.options[CONF_PHASE_L1] == "sensor.l1_power"

    # Check buffer meta phase_invert
    path = hass.config.path(BUFFER_DIR, BUFFER_FILE)
    conn = open_readonly(path)
    try:
        assert load_phase_sign(conn) == -1
    finally:
        conn.close()


async def test_options_flow_phase_check_mismatch(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
) -> None:
    """Phases mismatching grid show phase_check with invert_phases default False."""
    energy_manager.data = std_prefs
    async_set_power(hass, "sensor.grid_power", 1000)
    async_set_power(hass, "sensor.solar_power", 1200)
    async_set_power(hass, "sensor.battery_power", -300)
    async_set_power(hass, "sensor.washer_power", 200)

    async_set_power(hass, "sensor.l1_power", 100)
    async_set_power(hass, "sensor.l2_power", 100)
    async_set_power(hass, "sensor.l3_power", 100)

    entry = await async_setup_via_flow(hass)

    result = await async_open_options_step(hass, entry, "phases")
    result2 = await hass.config_entries.options.async_configure(
        result["flow_id"],
        user_input={
            CONF_PHASE_L1: "sensor.l1_power",
            CONF_PHASE_L2: "sensor.l2_power",
            CONF_PHASE_L3: "sensor.l3_power",
        },
    )
    assert result2["type"] is FlowResultType.FORM
    assert result2["step_id"] == "phase_check"
    assert result2["description_placeholders"]["phase_sum"] == "300"
    assert result2["description_placeholders"]["grid"] == "1000"
    assert result2["description_placeholders"]["result"] == "mismatch"

    # Default for invert_phases is False
    marker = [k for k in result2["data_schema"].schema if k == CONF_INVERT_PHASES][0]
    assert marker.default() is False

    # Confirm with invert_phases=False
    result3 = await hass.config_entries.options.async_configure(
        result2["flow_id"],
        user_input={CONF_INVERT_PHASES: False},
    )
    assert result3["type"] is FlowResultType.CREATE_ENTRY
    await hass.async_block_till_done()

    assert entry.options.get(CONF_INVERT_PHASES, False) is False

    # No repairs issue created
    issues = ir.async_get(hass).issues
    assert len(issues) == 0


async def test_options_flow_home_only_skips_phase_check(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
) -> None:
    """Energy prefs with only a home source skips phase_check step (SKIPPED)."""
    energy_manager.data = {
        "energy_sources": [
            {
                "type": "home",
                "stat_rate": "sensor.home_power",
            }
        ],
        "device_consumption": [],
    }
    async_set_power(hass, "sensor.home_power", 500)
    async_set_power(hass, "sensor.l1_power", 250)

    entry = await async_setup_via_flow(hass)

    result = await async_open_options_step(hass, entry, "phases")
    result2 = await hass.config_entries.options.async_configure(
        result["flow_id"],
        user_input={CONF_PHASE_L1: "sensor.l1_power"},
    )
    assert result2["type"] is FlowResultType.CREATE_ENTRY
    await hass.async_block_till_done()
    assert entry.options[CONF_PHASE_L1] == "sensor.l1_power"


async def test_options_flow_grid_device_suggestions(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
) -> None:
    """Form description placeholder lists power sensors on grid meter's device."""
    ent_reg = er.async_get(hass)
    dev_reg = dr.async_get(hass)

    # Register grid meter device and link sensor.grid_power to it before flow setup
    meter_entry = MockConfigEntry(domain="test_domain", data={})
    meter_entry.add_to_hass(hass)
    grid_dev = dev_reg.async_get_or_create(
        config_entry_id=meter_entry.entry_id,
        identifiers={("test_domain", "meter_123")},
    )
    ent_reg.async_get_or_create(
        "sensor", "test_domain", "grid_power",
        suggested_object_id="grid_power",
        device_id=grid_dev.id,
        original_device_class="power",
    )
    ent_reg.async_get_or_create(
        "sensor", "test_domain", "meter_l1",
        suggested_object_id="meter_l1",
        device_id=grid_dev.id,
        original_device_class="power",
    )
    ent_reg.async_get_or_create(
        "sensor", "test_domain", "meter_l2",
        suggested_object_id="meter_l2",
        device_id=grid_dev.id,
        original_device_class="power",
    )
    ent_reg.async_get_or_create(
        "sensor", "test_domain", "meter_l3",
        suggested_object_id="meter_l3",
        device_id=grid_dev.id,
        original_device_class="power",
    )

    energy_manager.data = std_prefs
    async_set_power(hass, "sensor.grid_power", 500)
    async_set_power(hass, "sensor.solar_power", 1200)
    async_set_power(hass, "sensor.battery_power", -300)
    async_set_power(hass, "sensor.washer_power", 200)

    entry = await async_setup_via_flow(hass)

    result = await async_open_options_step(hass, entry, "phases")
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "phases"
    suggestions = result["description_placeholders"]["grid_device_sensors"]
    assert "sensor.meter_l1" in suggestions
    assert "sensor.meter_l2" in suggestions
    assert "sensor.meter_l3" in suggestions
    assert "sensor.grid_power" not in suggestions


async def test_options_flow_grid_device_suggestions_second_device(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
) -> None:
    """Suggestions continue searching when the first grid entity device has no power candidates."""
    dev_reg = dr.async_get(hass)
    ent_reg = er.async_get(hass)

    meter_entry = MockConfigEntry(domain="test_domain", data={})
    meter_entry.add_to_hass(hass)

    # First device has sensor.grid_power but no other power sensors
    dev1 = dev_reg.async_get_or_create(
        config_entry_id=meter_entry.entry_id,
        identifiers={("test_domain", "dev1")},
    )
    ent_reg.async_get_or_create(
        "sensor", "test_domain", "grid_power",
        suggested_object_id="grid_power",
        device_id=dev1.id,
        original_device_class="power",
    )

    # Second device has sensor.grid_sub and candidates
    dev2 = dev_reg.async_get_or_create(
        config_entry_id=meter_entry.entry_id,
        identifiers={("test_domain", "dev2")},
    )
    ent_reg.async_get_or_create(
        "sensor", "test_domain", "grid_sub",
        suggested_object_id="grid_sub",
        device_id=dev2.id,
        original_device_class="power",
    )
    ent_reg.async_get_or_create(
        "sensor", "test_domain", "meter_l1",
        suggested_object_id="meter_l1",
        device_id=dev2.id,
        original_device_class="power",
    )

    # Net prefs with two grid sources
    prefs = dict(std_prefs)
    prefs["energy_sources"] = [
        {
            "type": "grid",
            "stat_rate": "sensor.grid_power",
        },
        {
            "type": "grid",
            "stat_rate": "sensor.grid_sub",
        },
        {
            "type": "solar",
            "stat_rate": "sensor.solar_power",
        },
        {
            "type": "battery",
            "stat_rate": "sensor.battery_power",
        },
    ]
    energy_manager.data = prefs
    async_set_power(hass, "sensor.grid_power", 500)
    async_set_power(hass, "sensor.grid_sub", 200)
    async_set_power(hass, "sensor.solar_power", 1200)
    async_set_power(hass, "sensor.battery_power", -300)
    async_set_power(hass, "sensor.washer_power", 200)

    entry = await async_setup_via_flow(hass)
    result = await async_open_options_step(hass, entry, "phases")
    assert result["type"] is FlowResultType.FORM
    suggestions = result["description_placeholders"]["grid_device_sensors"]
    assert "sensor.meter_l1" in suggestions



async def test_clear_phases_removes_interval_sensors(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
) -> None:
    """Clearing mapped phases removes interval sensors and writes NULL rows."""
    energy_manager.data = std_prefs
    async_set_power(hass, "sensor.grid_power", 1000)
    async_set_power(hass, "sensor.solar_power", 1200)
    async_set_power(hass, "sensor.battery_power", -300)
    async_set_power(hass, "sensor.washer_power", 200)

    async_set_power(hass, "sensor.l1_power", 1000)

    entry = await async_setup_via_flow(hass)

    # Map L1
    result = await async_open_options_step(hass, entry, "phases")
    await hass.config_entries.options.async_configure(
        result["flow_id"],
        user_input={CONF_PHASE_L1: "sensor.l1_power"},
    )
    await hass.async_block_till_done()

    ent_reg = er.async_get(hass)
    interval_eid = ent_reg.async_get_entity_id(
        "sensor", DOMAIN, "powerguesser_sample_interval_sensor.l1_power"
    )
    assert interval_eid is not None

    # Now submit empty form to clear phases
    result2 = await async_open_options_step(hass, entry, "phases")
    await hass.config_entries.options.async_configure(
        result2["flow_id"],
        user_input={},
    )
    await hass.async_block_till_done()

    # The sample-interval sensor must be removed from the entity registry
    assert ent_reg.async_get_entity_id(
        "sensor", DOMAIN, "powerguesser_sample_interval_sensor.l1_power"
    ) is None

    # Verify channel got a NULL row in buffer
    await async_flush_buffer(hass, entry)
    samples_by_ch, _ = await async_read_buffer(hass)
    l1_samples = samples_by_ch.get(("sensor.l1_power", "l1"), [])
    assert len(l1_samples) > 0
    # Last sample was NULL
    assert l1_samples[-1][1] is None


async def test_phase_rename_updates_options(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
) -> None:
    """Renaming a registered phase entity updates entry.options and keeps buffer channel."""
    ent_reg = er.async_get(hass)
    ent_reg.async_get_or_create(
        "sensor", "demo", "l1_power", suggested_object_id="l1_power"
    )

    energy_manager.data = std_prefs
    async_set_power(hass, "sensor.grid_power", 500)
    async_set_power(hass, "sensor.solar_power", 1200)
    async_set_power(hass, "sensor.battery_power", -300)
    async_set_power(hass, "sensor.washer_power", 200)
    async_set_power(hass, "sensor.l1_power", 500)

    entry = await async_setup_via_flow(hass)

    result = await async_open_options_step(hass, entry, "phases")
    await hass.config_entries.options.async_configure(
        result["flow_id"],
        user_input={CONF_PHASE_L1: "sensor.l1_power"},
    )
    await hass.async_block_till_done()
    assert entry.options[CONF_PHASE_L1] == "sensor.l1_power"

    # Rename sensor.l1_power to sensor.meter_l1_power
    ent_reg.async_update_entity(
        "sensor.l1_power", new_entity_id="sensor.meter_l1_power"
    )
    await hass.async_block_till_done()

    # entry.options should now be updated to sensor.meter_l1_power
    assert entry.options[CONF_PHASE_L1] == "sensor.meter_l1_power"


async def test_phase_rename_after_unload_is_noop(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
) -> None:
    """Rename arriving after hub stopped / unloaded is a no-op."""
    energy_manager.data = std_prefs
    async_set_power(hass, "sensor.grid_power", 500)
    async_set_power(hass, "sensor.solar_power", 1200)
    async_set_power(hass, "sensor.battery_power", -300)
    async_set_power(hass, "sensor.washer_power", 200)
    async_set_power(hass, "sensor.l1_power", 500)

    entry = await async_setup_via_flow(hass)

    result = await async_open_options_step(hass, entry, "phases")
    await hass.config_entries.options.async_configure(
        result["flow_id"],
        user_input={CONF_PHASE_L1: "sensor.l1_power"},
    )
    await hass.async_block_till_done()
    assert entry.options[CONF_PHASE_L1] == "sensor.l1_power"

    hub = entry.runtime_data
    # Unload entry
    await hass.config_entries.async_unload(entry.entry_id)
    assert entry.state is ConfigEntryState.NOT_LOADED

    # Rename arriving after unload
    await hub._async_follow_rename("sensor.l1_power", "sensor.meter_l1_power")

    # Options remain unchanged
    assert entry.options[CONF_PHASE_L1] == "sensor.l1_power"

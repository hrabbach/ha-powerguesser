"""Tests for bootstrapped usable history sensor and reporting."""

from datetime import timedelta
import time
from typing import Any
from unittest.mock import patch
import pytest

from homeassistant.components.energy.data import EnergyManager
from homeassistant.core import HomeAssistant
from homeassistant.helpers import entity_registry as er
from homeassistant.util import dt as dt_util
from pytest_homeassistant.common import async_fire_time_changed
from pytest_homeassistant.components.recorder.common import async_wait_recording_done

from custom_components.powerguesser.const import (
    BOOTSTRAP_START_TIMEOUT_S,
    DOMAIN,
)
from custom_components.powerguesser.engine.buffer import BufferDB
from custom_components.powerguesser.sensor import BootstrappedUsableHistorySensor
from tests.conftest import async_set_power, async_setup_via_flow


def test_channels_attribute_unrecorded() -> None:
    """The channels attribute is unrecorded so per-channel data stays out of recorder history."""
    assert "channels" in BootstrappedUsableHistorySensor._unrecorded_attributes


@pytest.mark.bootstrap_enabled
async def test_bootstrap_sensor_shows_usable_days(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    freezer: Any,
) -> None:
    """Tracer test: sensor shows usable days, done status, ISO computed_at, and per-channel details."""
    energy_manager.data = std_prefs
    now = dt_util.utcnow()
    start_time = now - timedelta(days=3)
    freezer.move_to(start_time)

    for h in range(72):
        if 20 <= h < 22:
            async_set_power(hass, "sensor.grid_power", "unavailable")
        else:
            async_set_power(hass, "sensor.grid_power", 500 + h)
        async_set_power(hass, "sensor.solar_power", 1000 + h)
        async_set_power(hass, "sensor.battery_power", 200 + h)
        async_set_power(hass, "sensor.washer_power", 50 + h)
        freezer.tick(timedelta(hours=1))

    await async_wait_recording_done(hass)

    freezer.move_to(now)
    async_set_power(hass, "sensor.grid_power", 600)
    async_set_power(hass, "sensor.solar_power", 1500)
    async_set_power(hass, "sensor.battery_power", 100)
    async_set_power(hass, "sensor.washer_power", 75)

    entry = await async_setup_via_flow(hass)
    hub = entry.runtime_data
    assert hub.bootstrap_task is not None

    async_fire_time_changed(
        hass, dt_util.utcnow() + timedelta(seconds=BOOTSTRAP_START_TIMEOUT_S + 1)
    )
    await hub.bootstrap_task

    ent_reg = er.async_get(hass)
    sensor_entity_id = ent_reg.async_get_entity_id(
        "sensor", DOMAIN, "powerguesser_bootstrapped_usable_history"
    )
    assert sensor_entity_id is not None

    state = hass.states.get(sensor_entity_id)
    assert state is not None

    # 72 hours minus 2 hours unavailable = 70 hours = 2.9167 days (~2.92 days)
    usable_val = float(state.state)
    assert abs(usable_val - (70.0 / 24.0)) < 0.02
    assert state.attributes["bootstrap_status"] == "done"
    assert isinstance(state.attributes["computed_at"], str)

    channels = state.attributes["channels"]
    assert len(channels) == 4
    for entity_id in [
        "sensor.grid_power",
        "sensor.solar_power",
        "sensor.battery_power",
        "sensor.washer_power",
    ]:
        assert entity_id in channels
        ch_info = channels[entity_id]
        assert ch_info["excluded_by_recorder"] is False
        assert ch_info["skipped"] is None
        assert ch_info["days"] > 0
        assert ch_info["first"] is not None
        assert ch_info["last"] is not None

    # Washer plug days does not affect usable_val (plugs are excluded from usable figure)
    assert channels["sensor.washer_power"]["role"] == "metered"


@pytest.mark.bootstrap_enabled
@pytest.mark.parametrize(
    "recorder_config",
    [{"exclude": {"entities": ["sensor.washer_power"]}}],
)
async def test_excluded_by_recorder_flag(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    freezer: Any,
) -> None:
    """A recorder-excluded channel shows 0 days and excluded_by_recorder true, usable_days unaffected."""
    energy_manager.data = std_prefs
    now = dt_util.utcnow()
    start_time = now - timedelta(days=2)
    freezer.move_to(start_time)

    for h in range(48):
        async_set_power(hass, "sensor.grid_power", 500 + h)
        async_set_power(hass, "sensor.solar_power", 1000 + h)
        async_set_power(hass, "sensor.battery_power", 200 + h)
        async_set_power(hass, "sensor.washer_power", 50 + h)
        freezer.tick(timedelta(hours=1))

    await async_wait_recording_done(hass)

    freezer.move_to(now)
    async_set_power(hass, "sensor.grid_power", 600)
    async_set_power(hass, "sensor.solar_power", 1500)
    async_set_power(hass, "sensor.battery_power", 100)
    async_set_power(hass, "sensor.washer_power", 75)

    entry = await async_setup_via_flow(hass)
    hub = entry.runtime_data
    assert hub.bootstrap_task is not None

    async_fire_time_changed(
        hass, dt_util.utcnow() + timedelta(seconds=BOOTSTRAP_START_TIMEOUT_S + 1)
    )
    await hub.bootstrap_task

    ent_reg = er.async_get(hass)
    sensor_entity_id = ent_reg.async_get_entity_id(
        "sensor", DOMAIN, "powerguesser_bootstrapped_usable_history"
    )
    assert sensor_entity_id is not None
    state = hass.states.get(sensor_entity_id)
    assert state is not None

    usable_val = float(state.state)
    # 48 hours = 2.0 days
    assert abs(usable_val - 2.0) < 0.02

    channels = state.attributes["channels"]
    washer_info = channels["sensor.washer_power"]
    assert washer_info["excluded_by_recorder"] is True
    assert washer_info["days"] == 0
    assert washer_info["first"] is None
    assert washer_info["last"] is None



@pytest.mark.bootstrap_enabled
async def test_sensor_lifecycle_waiting_running_done(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
) -> None:
    """Fresh install shows waiting before gate opens, running, then done."""
    energy_manager.data = std_prefs
    async_set_power(hass, "sensor.grid_power", 500)
    async_set_power(hass, "sensor.solar_power", 1000)
    async_set_power(hass, "sensor.battery_power", 200)
    async_set_power(hass, "sensor.washer_power", 50)

    entry = await async_setup_via_flow(hass)
    hub = entry.runtime_data
    assert hub.bootstrap_task is not None

    ent_reg = er.async_get(hass)
    sensor_entity_id = ent_reg.async_get_entity_id(
        "sensor", DOMAIN, "powerguesser_bootstrapped_usable_history"
    )
    assert sensor_entity_id is not None
    state = hass.states.get(sensor_entity_id)
    assert state is not None
    # Before gate opens, sensor has no usable_days yet, status is waiting
    assert state.state in ("unknown", "unavailable", "None", "")
    assert state.attributes["bootstrap_status"] == "waiting"

    # Open gate and complete
    async_fire_time_changed(
        hass, dt_util.utcnow() + timedelta(seconds=BOOTSTRAP_START_TIMEOUT_S + 1)
    )
    await hub.bootstrap_task

    state = hass.states.get(sensor_entity_id)
    assert state is not None
    assert state.attributes["bootstrap_status"] == "done"
    assert state.state not in ("unknown", "unavailable")


@pytest.mark.bootstrap_enabled
async def test_report_survives_restart_and_shows_waiting(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    freezer: Any,
) -> None:
    """Stored report is shown with status waiting right after reload before gate opens."""
    energy_manager.data = std_prefs
    now = dt_util.utcnow()
    start_time = now - timedelta(days=2)
    freezer.move_to(start_time)

    for h in range(48):
        async_set_power(hass, "sensor.grid_power", 500 + h)
        async_set_power(hass, "sensor.solar_power", 1000 + h)
        async_set_power(hass, "sensor.battery_power", 200 + h)
        async_set_power(hass, "sensor.washer_power", 50 + h)
        freezer.tick(timedelta(hours=1))

    await async_wait_recording_done(hass)

    freezer.move_to(now)
    async_set_power(hass, "sensor.grid_power", 600)
    async_set_power(hass, "sensor.solar_power", 1500)
    async_set_power(hass, "sensor.battery_power", 100)
    async_set_power(hass, "sensor.washer_power", 75)

    entry = await async_setup_via_flow(hass)
    hub = entry.runtime_data
    async_fire_time_changed(
        hass, dt_util.utcnow() + timedelta(seconds=BOOTSTRAP_START_TIMEOUT_S + 1)
    )
    await hub.bootstrap_task

    ent_reg = er.async_get(hass)
    sensor_entity_id = ent_reg.async_get_entity_id(
        "sensor", DOMAIN, "powerguesser_bootstrapped_usable_history"
    )
    assert sensor_entity_id is not None
    state = hass.states.get(sensor_entity_id)
    assert state is not None
    orig_computed_at = state.attributes["computed_at"]
    orig_usable_val = float(state.state)
    assert orig_computed_at is not None

    # Reload entry
    await hass.config_entries.async_reload(entry.entry_id)
    await hass.async_block_till_done()

    # Before gate opens on the new run, sensor shows stored report with status waiting
    new_state = hass.states.get(sensor_entity_id)
    assert new_state is not None
    assert new_state.attributes["bootstrap_status"] == "waiting"
    assert new_state.attributes["computed_at"] == orig_computed_at
    assert float(new_state.state) == orig_usable_val
    assert len(new_state.attributes["channels"]) == 4


@pytest.mark.bootstrap_enabled
async def test_stored_report_shown_with_waiting_when_report_read_is_slow(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    freezer: Any,
) -> None:
    """A slow stored-report read cannot leave the sensor at waiting without the report.

    The bootstrap task is a background task, so async_block_till_done does not wait for
    its executor jobs. The stored report must be loaded in an awaited step of entry
    setup, so the first waiting publish already carries it.
    """
    energy_manager.data = std_prefs
    now = dt_util.utcnow()
    freezer.move_to(now - timedelta(days=2))

    for h in range(48):
        async_set_power(hass, "sensor.grid_power", 500 + h)
        async_set_power(hass, "sensor.solar_power", 1000 + h)
        async_set_power(hass, "sensor.battery_power", 200 + h)
        async_set_power(hass, "sensor.washer_power", 50 + h)
        freezer.tick(timedelta(hours=1))

    await async_wait_recording_done(hass)

    freezer.move_to(now)
    async_set_power(hass, "sensor.grid_power", 600)
    async_set_power(hass, "sensor.solar_power", 1500)
    async_set_power(hass, "sensor.battery_power", 100)
    async_set_power(hass, "sensor.washer_power", 75)

    entry = await async_setup_via_flow(hass)
    hub = entry.runtime_data
    async_fire_time_changed(
        hass, dt_util.utcnow() + timedelta(seconds=BOOTSTRAP_START_TIMEOUT_S + 1)
    )
    await hub.bootstrap_task

    ent_reg = er.async_get(hass)
    sensor_entity_id = ent_reg.async_get_entity_id(
        "sensor", DOMAIN, "powerguesser_bootstrapped_usable_history"
    )
    assert sensor_entity_id is not None
    state = hass.states.get(sensor_entity_id)
    assert state is not None
    orig_computed_at = state.attributes["computed_at"]
    orig_usable_val = float(state.state)
    assert orig_computed_at is not None

    real_get_meta = BufferDB.get_meta

    def _slow_get_meta(self: BufferDB, key: str) -> str | None:
        if key == "bootstrap_report":
            time.sleep(0.5)
        return real_get_meta(self, key)

    with patch.object(BufferDB, "get_meta", _slow_get_meta):
        await hass.config_entries.async_reload(entry.entry_id)
        await hass.async_block_till_done()
        new_state = hass.states.get(sensor_entity_id)

    assert new_state is not None
    assert new_state.attributes["bootstrap_status"] == "waiting"
    assert new_state.attributes["computed_at"] == orig_computed_at
    assert float(new_state.state) == orig_usable_val
    assert len(new_state.attributes["channels"]) == 4


@pytest.mark.bootstrap_enabled
async def test_no_op_run_republishes_stored_report_without_rewrite(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    freezer: Any,
) -> None:
    """A run with no imported channels republishes stored report and keeps computed_at."""
    energy_manager.data = std_prefs
    now = dt_util.utcnow()
    start_time = now - timedelta(days=2)
    freezer.move_to(start_time)

    for h in range(48):
        async_set_power(hass, "sensor.grid_power", 500 + h)
        async_set_power(hass, "sensor.solar_power", 1000 + h)
        async_set_power(hass, "sensor.battery_power", 200 + h)
        async_set_power(hass, "sensor.washer_power", 50 + h)
        freezer.tick(timedelta(hours=1))

    await async_wait_recording_done(hass)

    freezer.move_to(now)
    async_set_power(hass, "sensor.grid_power", 600)
    async_set_power(hass, "sensor.solar_power", 1500)
    async_set_power(hass, "sensor.battery_power", 100)
    async_set_power(hass, "sensor.washer_power", 75)

    entry = await async_setup_via_flow(hass)
    hub = entry.runtime_data
    async_fire_time_changed(
        hass, dt_util.utcnow() + timedelta(seconds=BOOTSTRAP_START_TIMEOUT_S + 1)
    )
    await hub.bootstrap_task

    ent_reg = er.async_get(hass)
    sensor_entity_id = ent_reg.async_get_entity_id(
        "sensor", DOMAIN, "powerguesser_bootstrapped_usable_history"
    )
    assert sensor_entity_id is not None
    orig_computed_at = hass.states.get(sensor_entity_id).attributes["computed_at"]

    # Reload entry and let second bootstrap run complete
    await hass.config_entries.async_reload(entry.entry_id)
    await hass.async_block_till_done()

    reloaded_hub = entry.runtime_data
    # Move time forward 1 hour so if computed_at was rewritten it would differ
    freezer.tick(timedelta(hours=1))
    async_fire_time_changed(
        hass, dt_util.utcnow() + timedelta(seconds=BOOTSTRAP_START_TIMEOUT_S + 1)
    )
    await reloaded_hub.bootstrap_task

    final_state = hass.states.get(sensor_entity_id)
    assert final_state is not None
    assert final_state.attributes["bootstrap_status"] == "done"
    # Original computed_at preserved because 0 channels imported
    assert final_state.attributes["computed_at"] == orig_computed_at


@pytest.mark.bootstrap_enabled
async def test_late_plug_adds_channel_usable_days_unchanged(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    freezer: Any,
) -> None:
    """Adding a plug later updates channels attribute while usable_days is unchanged."""
    # Start without washer_power
    prefs_no_plug = {
        "energy_sources": std_prefs["energy_sources"],
        "device_consumption": [],
    }
    energy_manager.data = prefs_no_plug

    now = dt_util.utcnow()
    start_time = now - timedelta(days=2)
    freezer.move_to(start_time)

    for h in range(48):
        async_set_power(hass, "sensor.grid_power", 500 + h)
        async_set_power(hass, "sensor.solar_power", 1000 + h)
        async_set_power(hass, "sensor.battery_power", 200 + h)
        async_set_power(hass, "sensor.washer_power", 50 + h)
        freezer.tick(timedelta(hours=1))

    await async_wait_recording_done(hass)

    freezer.move_to(now)
    async_set_power(hass, "sensor.grid_power", 600)
    async_set_power(hass, "sensor.solar_power", 1500)
    async_set_power(hass, "sensor.battery_power", 100)

    entry = await async_setup_via_flow(hass)
    hub = entry.runtime_data
    async_fire_time_changed(
        hass, dt_util.utcnow() + timedelta(seconds=BOOTSTRAP_START_TIMEOUT_S + 1)
    )
    await hub.bootstrap_task

    ent_reg = er.async_get(hass)
    sensor_entity_id = ent_reg.async_get_entity_id(
        "sensor", DOMAIN, "powerguesser_bootstrapped_usable_history"
    )
    assert sensor_entity_id is not None
    state_before = hass.states.get(sensor_entity_id)
    assert "sensor.washer_power" not in state_before.attributes["channels"]
    usable_days_before = float(state_before.state)

    # Now add washer_power as late plug
    energy_manager.data = std_prefs
    async_set_power(hass, "sensor.washer_power", 75)
    await hub.async_rebuild()
    await hass.async_block_till_done()

    # Release gate for late bootstrap and await task
    async_fire_time_changed(
        hass, dt_util.utcnow() + timedelta(seconds=BOOTSTRAP_START_TIMEOUT_S + 1)
    )
    await hub.bootstrap_task

    state_after = hass.states.get(sensor_entity_id)
    assert state_after is not None
    assert "sensor.washer_power" in state_after.attributes["channels"]
    washer_cov = state_after.attributes["channels"]["sensor.washer_power"]
    assert washer_cov["days"] > 0
    # Usable days is strictly unchanged!
    assert float(state_after.state) == usable_days_before


@pytest.mark.bootstrap_enabled
async def test_recorder_unavailable_preserves_stored_report(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    freezer: Any,
) -> None:
    """If recorder is unavailable, stored report stays visible with status recorder_unavailable."""
    energy_manager.data = std_prefs
    now = dt_util.utcnow()
    start_time = now - timedelta(days=2)
    freezer.move_to(start_time)

    for h in range(48):
        async_set_power(hass, "sensor.grid_power", 500 + h)
        async_set_power(hass, "sensor.solar_power", 1000 + h)
        async_set_power(hass, "sensor.battery_power", 200 + h)
        async_set_power(hass, "sensor.washer_power", 50 + h)
        freezer.tick(timedelta(hours=1))

    await async_wait_recording_done(hass)

    freezer.move_to(now)
    async_set_power(hass, "sensor.grid_power", 600)
    async_set_power(hass, "sensor.solar_power", 1500)
    async_set_power(hass, "sensor.battery_power", 100)
    async_set_power(hass, "sensor.washer_power", 75)

    entry = await async_setup_via_flow(hass)
    hub = entry.runtime_data
    async_fire_time_changed(
        hass, dt_util.utcnow() + timedelta(seconds=BOOTSTRAP_START_TIMEOUT_S + 1)
    )
    await hub.bootstrap_task

    ent_reg = er.async_get(hass)
    sensor_entity_id = ent_reg.async_get_entity_id(
        "sensor", DOMAIN, "powerguesser_bootstrapped_usable_history"
    )
    orig_state = hass.states.get(sensor_entity_id)
    orig_usable = float(orig_state.state)
    orig_computed_at = orig_state.attributes["computed_at"]

    # Reload entry, but mock async_db_ready to False
    import asyncio
    from unittest.mock import patch
    from homeassistant.components.recorder import get_instance

    with patch.object(
        get_instance(hass),
        "async_db_ready",
        new_callable=lambda: asyncio.Future(),
    ) as mock_ready:
        mock_ready.set_result(False)

        await hass.config_entries.async_reload(entry.entry_id)
        await hass.async_block_till_done()

        reloaded_hub = entry.runtime_data
        async_fire_time_changed(
            hass, dt_util.utcnow() + timedelta(seconds=BOOTSTRAP_START_TIMEOUT_S + 1)
        )
        await reloaded_hub.bootstrap_task

    unavail_state = hass.states.get(sensor_entity_id)
    assert unavail_state is not None
    assert unavail_state.attributes["bootstrap_status"] == "recorder_unavailable"
    assert float(unavail_state.state) == orig_usable
    assert unavail_state.attributes["computed_at"] == orig_computed_at
    assert len(unavail_state.attributes["channels"]) == 4

"""Tests for entity rename tracking, recorder exclusion buffering, and entry removal."""

import os
import sqlite3
from typing import Any
import pytest

from homeassistant.components.energy.data import EnergyManager
from homeassistant.components.recorder import is_entity_recorded
from homeassistant.config_entries import ConfigEntryState
from homeassistant.core import HomeAssistant
from homeassistant.helpers import entity_registry as er
from homeassistant.util import dt as dt_util

from custom_components.powerguesser.const import BUFFER_DIR, BUFFER_FILE, DOMAIN
from tests.conftest import (
    async_flush_buffer,
    async_read_buffer,
    async_set_power,
    async_setup_via_flow,
)


async def test_rename_registered_entity_keeps_channel_id(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
) -> None:
    """Renaming an entity in the registry keeps its channel ID and history in the buffer."""
    ent_reg = er.async_get(hass)
    ent_reg.async_get_or_create(
        "sensor", "demo", "washer_unique_id", suggested_object_id="washer_power"
    )

    energy_manager.data = std_prefs
    async_set_power(hass, "sensor.grid_power", 500)
    async_set_power(hass, "sensor.solar_power", 1000)
    async_set_power(hass, "sensor.battery_power", 0)
    async_set_power(hass, "sensor.washer_power", 150)

    entry = await async_setup_via_flow(hass)
    await async_flush_buffer(hass, entry)

    path = hass.config.path(BUFFER_DIR, BUFFER_FILE)
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    cur.execute("SELECT id, entity_id FROM channels WHERE role = 'metered'")
    initial_channel = cur.fetchone()
    conn.close()
    assert initial_channel is not None
    orig_ch_id, orig_entity_id = initial_channel
    assert orig_entity_id == "sensor.washer_power"

    # Update Energy dashboard preferences to the new entity ID
    std_prefs["device_consumption"][0]["stat_rate"] = "sensor.washer_plug_power"
    energy_manager.data = std_prefs

    # Rename entity in entity registry
    ent_reg.async_update_entity(
        "sensor.washer_power", new_entity_id="sensor.washer_plug_power"
    )
    await hass.async_block_till_done()

    # Provide new reading for renamed entity and flush
    async_set_power(hass, "sensor.washer_plug_power", 280)
    await hass.async_block_till_done()
    await async_flush_buffer(hass, entry)

    conn = sqlite3.connect(path)
    cur = conn.cursor()
    cur.execute("SELECT id, entity_id FROM channels WHERE role = 'metered'")
    metered_channels = cur.fetchall()
    conn.close()

    # Still exactly one metered channel, same ID, new entity_id
    assert len(metered_channels) == 1
    assert metered_channels[0] == (orig_ch_id, "sensor.washer_plug_power")

    # Both old and new samples belong to orig_ch_id
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    cur.execute("SELECT w FROM samples WHERE ch = ? AND w IS NOT NULL ORDER BY ts", (orig_ch_id,))
    rows = [r[0] for r in cur.fetchall()]
    conn.close()
    assert 150 in rows
    assert 280 in rows


@pytest.mark.parametrize(
    "recorder_config",
    [{"exclude": {"entities": ["sensor.washer_power"]}}],
)
async def test_recorder_exclusion_still_buffered(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
) -> None:
    """An entity excluded from recorder is still captured by the buffer writer."""
    assert not is_entity_recorded(hass, "sensor.washer_power")

    energy_manager.data = std_prefs
    async_set_power(hass, "sensor.grid_power", 500)
    async_set_power(hass, "sensor.solar_power", 1000)
    async_set_power(hass, "sensor.battery_power", 0)
    async_set_power(hass, "sensor.washer_power", 320)

    entry = await async_setup_via_flow(hass)
    await async_flush_buffer(hass, entry)

    data, _ = await async_read_buffer(hass)
    assert ("sensor.washer_power", "metered") in data
    washer_samples = [s[1] for s in data[("sensor.washer_power", "metered")] if s[1] is not None]
    assert 320 in washer_samples


async def test_rename_after_unload_is_noop(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
) -> None:
    """Awaiting _async_follow_rename after hub stopped does nothing and doesn't open buffer."""
    energy_manager.data = std_prefs
    async_set_power(hass, "sensor.grid_power", 500)
    async_set_power(hass, "sensor.solar_power", 1000)
    async_set_power(hass, "sensor.battery_power", 0)
    async_set_power(hass, "sensor.washer_power", 100)

    entry = await async_setup_via_flow(hass)
    hub = entry.runtime_data

    # Unload entry -> stops hub and writer
    assert await hass.config_entries.async_unload(entry.entry_id)
    assert hub._stopped is True

    # Rename coroutine called while stopped -> returns cleanly without error
    await hub._async_follow_rename("sensor.washer_power", "sensor.washer_plug_power")
    assert hub.writer.db._conn is None


async def test_entry_removal_deletes_buffer_files(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
) -> None:
    """Removing config entry deletes buffer files; unloading and reloading preserves them."""
    energy_manager.data = std_prefs
    async_set_power(hass, "sensor.grid_power", 500)
    async_set_power(hass, "sensor.solar_power", 1000)
    async_set_power(hass, "sensor.battery_power", 0)
    async_set_power(hass, "sensor.washer_power", 100)

    entry = await async_setup_via_flow(hass)
    await async_flush_buffer(hass, entry)

    path = hass.config.path(BUFFER_DIR, BUFFER_FILE)
    assert os.path.exists(path)

    # Unload preserves file
    assert await hass.config_entries.async_unload(entry.entry_id)
    assert os.path.exists(path)

    # Re-setup entry preserves file
    assert await hass.config_entries.async_setup(entry.entry_id)
    assert os.path.exists(path)

    # Removal deletes the buffer files
    assert await hass.config_entries.async_remove(entry.entry_id)
    await hass.async_block_till_done()
    assert not os.path.exists(path)

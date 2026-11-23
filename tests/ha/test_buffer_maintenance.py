"""Tests for buffer retention, flush-failure recovery/capping, and corruption handling."""

from datetime import timedelta
import logging
import os
import sqlite3
from typing import Any
from unittest.mock import patch
import pytest

from homeassistant.components.energy.data import EnergyManager
from homeassistant.config_entries import ConfigEntryState
from homeassistant.core import HomeAssistant
from homeassistant.helpers import issue_registry as ir
from homeassistant.util import dt as dt_util
from pytest_homeassistant.common import async_fire_time_changed

from custom_components.powerguesser.buffer_writer import MAX_PENDING_ROWS
from custom_components.powerguesser.const import BUFFER_DIR, BUFFER_FILE, DOMAIN
from tests.conftest import async_set_power, async_setup_via_flow


async def test_retention_timer_runs_after_5m_and_daily(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
) -> None:
    """Retention runs 5m after start and then daily, deleting rows older than 90 days."""
    energy_manager.data = std_prefs
    async_set_power(hass, "sensor.grid_power", 500)
    async_set_power(hass, "sensor.solar_power", 1000)
    async_set_power(hass, "sensor.battery_power", 0)
    async_set_power(hass, "sensor.washer_power", 100)

    entry = await async_setup_via_flow(hass)
    assert entry.state == ConfigEntryState.LOADED

    path = hass.config.path(BUFFER_DIR, BUFFER_FILE)
    now_ms = round(dt_util.utcnow().timestamp() * 1000)
    old_ts = now_ms - 91 * 86_400_000
    new_ts = now_ms - 10 * 86_400_000

    conn = sqlite3.connect(path)
    ch_id = conn.execute("SELECT id FROM channels LIMIT 1").fetchone()[0]
    conn.execute("INSERT INTO samples(ts, ch, w) VALUES(?, ?, ?)", (old_ts, ch_id, 450))
    conn.execute("INSERT INTO samples(ts, ch, w) VALUES(?, ?, ?)", (new_ts, ch_id, 550))
    conn.commit()
    conn.close()

    # Advance clock by 5 minutes (300 s)
    async_fire_time_changed(hass, dt_util.utcnow() + timedelta(seconds=300))
    await hass.async_block_till_done()

    conn = sqlite3.connect(path)
    timestamps = [r[0] for r in conn.execute("SELECT ts FROM samples WHERE ts IN (?, ?)", (old_ts, new_ts)).fetchall()]
    conn.close()

    assert old_ts not in timestamps
    assert new_ts in timestamps

    # Insert another old row and advance by 1 day
    older_ts2 = now_ms - 95 * 86_400_000
    conn = sqlite3.connect(path)
    conn.execute("INSERT INTO samples(ts, ch, w) VALUES(?, ?, ?)", (older_ts2, ch_id, 350))
    conn.commit()
    conn.close()

    async_fire_time_changed(hass, dt_util.utcnow() + timedelta(days=1, seconds=310))
    await hass.async_block_till_done()

    conn = sqlite3.connect(path)
    timestamps2 = [r[0] for r in conn.execute("SELECT ts FROM samples WHERE ts = ?", (older_ts2,)).fetchall()]
    conn.close()
    assert older_ts2 not in timestamps2


async def test_flush_failure_requeue_and_cap(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """write_batch failure re-queues pending rows, logs one WARNING, and caps backlog at MAX_PENDING_ROWS."""
    energy_manager.data = std_prefs
    async_set_power(hass, "sensor.grid_power", 500)
    async_set_power(hass, "sensor.solar_power", 1000)
    async_set_power(hass, "sensor.battery_power", 0)
    async_set_power(hass, "sensor.washer_power", 100)

    entry = await async_setup_via_flow(hass)
    writer = entry.runtime_data.writer

    writer.record(1, 250, dt_util.utcnow().timestamp())
    assert len(writer._pending) >= 1

    original_write = writer.db.write_batch
    failing = True

    def mock_write(rows):
        if failing:
            raise sqlite3.OperationalError("database is locked")
        return original_write(rows)

    writer.db.write_batch = mock_write
    try:
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            await writer.async_flush()

        # Did not crash, pending rows kept
        assert len(writer._pending) >= 1
        assert "Buffer write failed: database is locked" in caplog.text
        assert not any(
            "250" in r.message
            for r in caplog.records
            if "powerguesser" in r.name and r.levelno == logging.WARNING
        )

        # Calling flush again during streak does not repeat WARNING
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            await writer.async_flush()
        assert "Buffer write failed: database is locked" not in caplog.text

        # Recovery logs INFO
        failing = False
        with caplog.at_level(logging.INFO):
            await writer.async_flush()
        assert "Buffer write recovered successfully" in caplog.text
        assert len(writer._pending) == 0

        # Test overflow cap
        failing = True
        now_ms = round(dt_util.utcnow().timestamp() * 1000)
        writer._pending = [(now_ms + i, 1, 100) for i in range(MAX_PENDING_ROWS + 50)]

        caplog.clear()
        with caplog.at_level(logging.WARNING):
            await writer.async_flush()

        assert len(writer._pending) == MAX_PENDING_ROWS
        assert "Dropped 50 oldest pending buffer rows due to overflow" in caplog.text
    finally:
        writer.db.write_batch = original_write


async def test_open_corrupt_file_creates_repairs_issue(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
) -> None:
    """A corrupt buffer.sqlite is renamed to .corrupt-<epoch>, setup succeeds and creates Repairs issue."""
    energy_manager.data = std_prefs
    async_set_power(hass, "sensor.grid_power", 500)
    async_set_power(hass, "sensor.solar_power", 1000)
    async_set_power(hass, "sensor.battery_power", 0)
    async_set_power(hass, "sensor.washer_power", 100)

    db_path = hass.config.path(BUFFER_DIR, BUFFER_FILE)
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    with open(db_path, "wb") as f:
        f.write(os.urandom(256))

    entry = await async_setup_via_flow(hass)
    assert entry.state == ConfigEntryState.LOADED

    # Sensors work
    state = hass.states.get("sensor.powerguesser_gross_power")
    assert state is not None

    # Repairs issue exists
    issue_reg = ir.async_get(hass)
    issue = issue_reg.async_get_issue(DOMAIN, "buffer_reset")
    assert issue is not None
    assert issue.severity == ir.IssueSeverity.WARNING
    assert issue.translation_key == "buffer_reset"
    assert issue.translation_placeholders is not None
    backup_file = issue.translation_placeholders.get("backup")
    assert backup_file is not None
    assert backup_file.startswith("buffer.sqlite.corrupt-")


async def test_open_io_error_retries_with_reason(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
) -> None:
    """I/O error during open puts entry into setup retry with reason containing SQLite error."""
    energy_manager.data = std_prefs
    async_set_power(hass, "sensor.grid_power", 500)
    async_set_power(hass, "sensor.solar_power", 1000)
    async_set_power(hass, "sensor.battery_power", 0)
    async_set_power(hass, "sensor.washer_power", 100)

    with patch(
        "custom_components.powerguesser.engine.buffer.BufferDB.open",
        side_effect=sqlite3.OperationalError("disk I/O error"),
    ):
        result = await hass.config_entries.flow.async_init(
            DOMAIN, context={"source": "user"}
        )
        result2 = await hass.config_entries.flow.async_configure(result["flow_id"], {})
        await hass.async_block_till_done()
        entry = result2["result"]

        assert entry.state == ConfigEntryState.SETUP_RETRY
        assert "disk I/O error" in (entry.reason or "")

        issue_reg = ir.async_get(hass)
        assert issue_reg.async_get_issue(DOMAIN, "buffer_reset") is None

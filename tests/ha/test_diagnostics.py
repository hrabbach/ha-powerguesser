"""Tests for PowerGuesser diagnostics download."""

from datetime import timedelta
from typing import Any
import pytest

from homeassistant.components.energy.data import EnergyManager
from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util
from pytest_homeassistant.common import async_fire_time_changed
from pytest_homeassistant.components.recorder.common import async_wait_recording_done

from custom_components.powerguesser.const import (
    BOOTSTRAP_START_TIMEOUT_S,
    CONF_LLM_API_KEY,
    CONF_LLM_BASE_URL,
    CONF_LLM_MODEL,
    CONF_LLM_PROBE,
)
from custom_components.powerguesser.diagnostics import async_get_config_entry_diagnostics
from tests.conftest import async_set_power, async_setup_via_flow


def _walk_diagnostics(data: Any) -> None:
    """Walk diagnostics tree to verify leaf types and ensure no raw sample rows exist."""
    ts_keys = {"ts", "t", "time", "timestamp", "ts_ms"}
    val_keys = {"w", "value", "power", "watts"}

    if isinstance(data, dict):
        keys = set(data.keys())
        # No dict holds both a timestamp-like key and a power-value-like key
        assert not (keys & ts_keys and keys & val_keys), f"Forbidden dict shape found: {keys}"
        for v in data.values():
            _walk_diagnostics(v)
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                # Must not be a raw [epoch_ms, w] sample pair
                first = item[0]
                if isinstance(first, int) and first > 1_000_000_000_000:
                    pytest.fail(f"Found raw sample row in diagnostics: {item}")
            _walk_diagnostics(item)
    else:
        # Every leaf must be str, int, float, bool, or None
        assert isinstance(
            data, (str, int, float, bool, type(None))
        ), f"Disallowed leaf type {type(data)}: {data}"


@pytest.mark.bootstrap_enabled
async def test_diagnostics_top_level_and_aggregates(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    freezer: Any,
) -> None:
    """Diagnostics dict contains exactly the expected keys and aggregates only."""
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
    async_set_power(hass, "sensor.grid_power", 500)
    async_set_power(hass, "sensor.solar_power", 1500)
    async_set_power(hass, "sensor.battery_power", 100)
    async_set_power(hass, "sensor.washer_power", 75)

    entry = await async_setup_via_flow(hass)
    hub = entry.runtime_data

    # 21 on-change updates 31 s apart for grid (yields 20 deltas)
    for i in range(1, 22):
        freezer.tick(timedelta(seconds=31))
        async_fire_time_changed(hass, dt_util.utcnow())
        async_set_power(hass, "sensor.grid_power", 500 + i * 10)
        await hass.async_block_till_done()

    # Flush writer and complete bootstrap
    await hub.writer.async_flush()
    async_fire_time_changed(
        hass, dt_util.utcnow() + timedelta(seconds=BOOTSTRAP_START_TIMEOUT_S + 1)
    )
    await hub.bootstrap_task

    diag = await async_get_config_entry_diagnostics(hass, entry)

    # Exactly top-level keys: buffer, profiles, effective, bootstrap, options, data, llm_probe
    assert set(diag.keys()) == {
        "buffer",
        "profiles",
        "effective",
        "bootstrap",
        "options",
        "data",
        "llm_probe",
    }
    # No probe was configured in this entry, so the allow-list key is None
    assert diag["llm_probe"] is None

    # buffer section
    buf = diag["buffer"]
    assert "schema_version" in buf
    assert "file_bytes" in buf
    assert buf["file_bytes"] > 0
    assert "last_flush_ms" in buf
    assert "clean_stop" in buf
    assert buf["retention_days"] == 90
    assert "channels" in buf
    assert len(buf["channels"]) >= 4
    for ch_stat in buf["channels"]:
        assert {"id", "entity_id", "role", "kind", "rows", "null_rows", "first_ms", "last_ms"}.issubset(
            ch_stat.keys()
        )

    # profiles section
    prof = diag["profiles"]
    assert "sensor.grid_power" in prof
    grid_prof = prof["sensor.grid_power"]
    assert grid_prof["role"] == "grid"
    assert grid_prof["mode"] == "on_change"
    assert grid_prof["measured_from"] == "changes"
    assert grid_prof["median_s"] == 31.0
    assert grid_prof["delta_count"] >= 20
    assert 31 in grid_prof["histogram"]

    # effective section
    eff = diag["effective"]
    assert "interval_s" in eff
    assert "assumed" in eff
    assert isinstance(eff["enabled"], list)
    assert isinstance(eff["disabled"], dict)

    # bootstrap section
    boot = diag["bootstrap"]
    assert boot["status"] == "done"
    assert "usable_days" in boot
    assert "computed_at" in boot
    assert "channels" in boot

    # options section
    assert diag["options"] == dict(entry.options)


@pytest.mark.bootstrap_enabled
async def test_diagnostics_leaf_types(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    freezer: Any,
) -> None:
    """Every leaf in diagnostics is str, int, float, bool, or None, and no raw samples exist."""
    energy_manager.data = std_prefs
    async_set_power(hass, "sensor.grid_power", 500)
    async_set_power(hass, "sensor.solar_power", 1000)
    async_set_power(hass, "sensor.battery_power", 200)
    async_set_power(hass, "sensor.washer_power", 50)

    entry = await async_setup_via_flow(hass)
    hub = entry.runtime_data
    await hub.writer.async_flush()

    async_fire_time_changed(
        hass, dt_util.utcnow() + timedelta(seconds=BOOTSTRAP_START_TIMEOUT_S + 1)
    )
    await hub.bootstrap_task

    diag = await async_get_config_entry_diagnostics(hass, entry)
    _walk_diagnostics(diag)


async def test_diagnostics_redacts_api_key(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
) -> None:
    """API key is redacted in diagnostics data section and appears nowhere in diagnostics."""
    energy_manager.data = std_prefs
    async_set_power(hass, "sensor.grid_power", 500)
    async_set_power(hass, "sensor.solar_power", 1200)
    async_set_power(hass, "sensor.battery_power", -300)
    async_set_power(hass, "sensor.washer_power", 200)

    entry = await async_setup_via_flow(hass)
    secret_key = "sk-super-secret-key-12345"

    hass.config_entries.async_update_entry(
        entry,
        data={
            CONF_LLM_BASE_URL: "http://llm.test:4000/v1",
            CONF_LLM_API_KEY: secret_key,
            CONF_LLM_MODEL: "m1",
            CONF_LLM_PROBE: {
                "strategy": "json_schema",
                "token_param": "max_completion_tokens",
                "send_temperature": True,
                "model_returned": "m1-2026",
                "probed_at": "2026-11-16T12:00:00Z",
            },
        },
    )

    diag = await async_get_config_entry_diagnostics(hass, entry)
    assert diag["data"][CONF_LLM_API_KEY] == "**REDACTED**"
    assert secret_key not in str(diag)

    # CLEAN-01 (D-16): the probe result is a top-level six-key allow-list copied
    # verbatim from entry.data; a probe stored without model_mismatch reads None
    # and is never recomputed.
    assert diag["llm_probe"] == {
        "strategy": "json_schema",
        "token_param": "max_completion_tokens",
        "send_temperature": True,
        "model_returned": "m1-2026",
        "model_mismatch": None,
        "probed_at": "2026-11-16T12:00:00Z",
    }

    # CLEAN-01 (D-16): the base URL is redacted as a whole value and appears nowhere
    assert diag["data"][CONF_LLM_BASE_URL] == "**REDACTED**"
    assert "http://llm.test:4000/v1" not in str(diag)

    # Walk diagnostics asserting leaf types and structure
    _walk_diagnostics(diag)


async def test_diagnostics_llm_probe_none_without_probe(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
) -> None:
    """No probe: llm_probe is None and the base URL is still redacted (CLEAN-01).

    The redaction assertion must not pass vacuously when no probe exists, so the
    base URL carries userinfo and their secret must vanish from the dump.
    """
    energy_manager.data = std_prefs
    async_set_power(hass, "sensor.grid_power", 500)
    async_set_power(hass, "sensor.solar_power", 1200)
    async_set_power(hass, "sensor.battery_power", -300)
    async_set_power(hass, "sensor.washer_power", 200)

    entry = await async_setup_via_flow(hass)
    base_url = "http://user:secretpw@llm.test:4000/v1"

    hass.config_entries.async_update_entry(
        entry,
        data={
            CONF_LLM_BASE_URL: base_url,
            CONF_LLM_API_KEY: "sk-another-secret-key",
            CONF_LLM_MODEL: "m1",
        },
    )

    diag = await async_get_config_entry_diagnostics(hass, entry)
    assert diag["llm_probe"] is None
    assert diag["data"][CONF_LLM_BASE_URL] == "**REDACTED**"
    text = str(diag)
    assert base_url not in text
    assert "secretpw" not in text
    assert "sk-another-secret-key" not in text


async def test_diagnostics_llm_probe_drops_unknown_keys(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
) -> None:
    """An unexpected key in the stored probe is never copied into llm_probe (CLEAN-01)."""
    energy_manager.data = std_prefs
    async_set_power(hass, "sensor.grid_power", 500)
    async_set_power(hass, "sensor.solar_power", 1200)
    async_set_power(hass, "sensor.battery_power", -300)
    async_set_power(hass, "sensor.washer_power", 200)

    entry = await async_setup_via_flow(hass)

    hass.config_entries.async_update_entry(
        entry,
        data={
            CONF_LLM_BASE_URL: "http://llm.test:4000/v1",
            CONF_LLM_API_KEY: "sk-super-secret-key-12345",
            CONF_LLM_MODEL: "m1",
            CONF_LLM_PROBE: {
                "strategy": "json_schema",
                "token_param": "max_completion_tokens",
                "send_temperature": False,
                "model_returned": "m1-2026",
                "model_mismatch": True,
                "probed_at": "2026-11-16T12:00:00Z",
                "raw_headers": {"authorization": "Bearer leak"},
            },
        },
    )

    diag = await async_get_config_entry_diagnostics(hass, entry)
    assert set(diag["llm_probe"]) == {
        "strategy",
        "token_param",
        "send_temperature",
        "model_returned",
        "model_mismatch",
        "probed_at",
    }
    assert diag["llm_probe"]["model_mismatch"] is True
    assert "raw_headers" not in diag["llm_probe"]
    assert "Bearer leak" not in repr(diag["llm_probe"])
    # WR-01: the raw probe must not survive anywhere in the dump -- the data
    # section carries the same allow-listed copy, never the stored probe.
    assert "raw_headers" not in repr(diag)
    assert "Bearer leak" not in str(diag)
    assert set(diag["data"][CONF_LLM_PROBE]) == set(diag["llm_probe"])


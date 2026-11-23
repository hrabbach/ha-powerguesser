"""Tests for LLM endpoint options flow and capability probe."""

import logging
from typing import Any
from unittest.mock import patch
import pytest

from homeassistant.components.energy.data import EnergyManager
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType
from homeassistant.helpers.dispatcher import async_dispatcher_send

from custom_components.powerguesser.const import (
    CONF_INVERT_PHASES,
    CONF_LLM_API_KEY,
    CONF_LLM_BASE_URL,
    CONF_LLM_CLEAR_KEY,
    CONF_LLM_MODEL,
    CONF_LLM_PROBE,
    CONF_PHASE_L1,
    CONF_PHASE_L2,
    CONF_PHASE_L3,
    SIGNAL_PREFS_UPDATED,
)
from custom_components.powerguesser.engine.llm_wire import (
    Strategy,
    build_probe_body,
)
from custom_components.powerguesser.llm_client import normalise_base_url
from tests.conftest import (
    async_open_options_step,
    async_set_power,
    async_setup_via_flow,
    load_llm_cassette,
    mock_llm,
)


async def test_options_menu_lists_phases_and_llm_endpoint(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
) -> None:
    """Options menu lists phases, llm_endpoint, learning, review and plug_sessions."""
    energy_manager.data = std_prefs
    async_set_power(hass, "sensor.grid_power", 500)
    async_set_power(hass, "sensor.solar_power", 1200)
    async_set_power(hass, "sensor.battery_power", -300)
    async_set_power(hass, "sensor.washer_power", 200)

    entry = await async_setup_via_flow(hass)
    result = await hass.config_entries.options.async_init(entry.entry_id)
    assert result["type"] is FlowResultType.MENU
    assert result["step_id"] == "init"
    assert result["menu_options"] == [
        "phases",
        "llm_endpoint",
        "learning",
        "review",
        "plug_sessions",
    ]


async def test_llm_step_probes_and_stores_strategy(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    aioclient_mock: Any,
) -> None:
    """Tracer test: user saves LLM endpoint, probe passes on json_schema and stores probe profile in entry.data."""
    energy_manager.data = std_prefs
    async_set_power(hass, "sensor.grid_power", 500)
    async_set_power(hass, "sensor.solar_power", 1200)
    async_set_power(hass, "sensor.battery_power", -300)
    async_set_power(hass, "sensor.washer_power", 200)

    entry = await async_setup_via_flow(hass)
    initial_options = dict(entry.options)

    base_url = "http://llm.test:4000/v1"
    cassette = load_llm_cassette("probe", "json_schema_ok")
    mock_llm(aioclient_mock, base_url, lambda data: cassette)

    result = await async_open_options_step(hass, entry, "llm_endpoint")
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "llm_endpoint"

    # Submit URL with trailing slash, key, and model
    result2 = await hass.config_entries.options.async_configure(
        result["flow_id"],
        user_input={
            CONF_LLM_BASE_URL: "http://llm.test:4000/v1/",
            CONF_LLM_API_KEY: "sk-test-secret",
            CONF_LLM_MODEL: "m1",
        },
    )
    assert result2["type"] is FlowResultType.SHOW_PROGRESS
    assert result2["step_id"] == "llm_endpoint"

    await hass.async_block_till_done()

    result3 = await hass.config_entries.options.async_configure(result["flow_id"])
    assert result3["type"] is FlowResultType.CREATE_ENTRY

    # Entry data updated with normalised URL, key, model, and probe
    assert entry.data[CONF_LLM_BASE_URL] == "http://llm.test:4000/v1"
    assert entry.data[CONF_LLM_API_KEY] == "sk-test-secret"
    assert entry.data[CONF_LLM_MODEL] == "m1"
    assert entry.data[CONF_LLM_PROBE]["strategy"] == "json_schema"
    assert entry.data[CONF_LLM_PROBE]["model_returned"] == "m1-2026"
    assert entry.data[CONF_LLM_PROBE]["token_param"] == "max_completion_tokens"
    # Temperature is not sent by default (03-09 A2: LiteLLM router fallback)
    assert entry.data[CONF_LLM_PROBE]["send_temperature"] is False
    # m1 -> m1-2026 is a dated suffix, not a mismatch
    assert entry.data[CONF_LLM_PROBE]["model_mismatch"] is False

    # Options remain unchanged
    assert entry.options == initial_options

    # Verify outgoing request body and headers
    assert len(aioclient_mock.mock_calls) == 1
    method, url, body, headers = aioclient_mock.mock_calls[0]
    expected_body = build_probe_body(
        "m1",
        Strategy.JSON_SCHEMA,
        {"token_param": "max_completion_tokens", "send_temperature": False},
    )
    assert body == expected_body
    assert "temperature" not in body
    assert headers is not None
    assert headers.get("Authorization") == "Bearer sk-test-secret"


def test_normalise_base_url_rules() -> None:
    """URL normalisation accepts valid shapes and rejects invalid ones."""
    # Valid
    assert normalise_base_url("https://api.example.com") == "https://api.example.com/v1"
    assert normalise_base_url("http://h:4000/v1/") == "http://h:4000/v1"
    assert normalise_base_url("https://h/openai") == "https://h/openai/v1"
    assert normalise_base_url("https://h/V1") == "https://h/v1"

    # Invalid
    for bad in ["ftp://x", "https://u:p@x", "https://x/v1?a=1", "https://x/v1#f", "not a url"]:
        with pytest.raises(ValueError, match="invalid_url"):
            normalise_base_url(bad)


async def test_llm_step_validation_errors(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
) -> None:
    """Invalid URL or missing model produces form errors."""
    energy_manager.data = std_prefs
    async_set_power(hass, "sensor.grid_power", 500)
    async_set_power(hass, "sensor.solar_power", 1200)
    async_set_power(hass, "sensor.battery_power", -300)
    async_set_power(hass, "sensor.washer_power", 200)

    entry = await async_setup_via_flow(hass)

    # Invalid URL
    res = await async_open_options_step(hass, entry, "llm_endpoint")
    res2 = await hass.config_entries.options.async_configure(
        res["flow_id"],
        user_input={
            CONF_LLM_BASE_URL: "ftp://invalid",
            CONF_LLM_MODEL: "m1",
        },
    )
    assert res2["type"] is FlowResultType.FORM
    assert res2["errors"] == {"base": "invalid_url"}

    # Missing model
    res3 = await hass.config_entries.options.async_configure(
        res["flow_id"],
        user_input={
            CONF_LLM_BASE_URL: "https://api.example.com",
            CONF_LLM_MODEL: "",
        },
    )
    assert res3["type"] is FlowResultType.FORM
    assert res3["errors"] == {"base": "model_required"}


async def test_llm_step_ladder_fallthrough_to_tool_call(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    aioclient_mock: Any,
) -> None:
    """When json_schema is silently dropped, probe falls through to tool_call."""
    energy_manager.data = std_prefs
    async_set_power(hass, "sensor.grid_power", 500)
    async_set_power(hass, "sensor.solar_power", 1200)
    async_set_power(hass, "sensor.battery_power", -300)
    async_set_power(hass, "sensor.washer_power", 200)

    entry = await async_setup_via_flow(hass)

    base_url = "http://llm.test:4000/v1"
    calls = []

    def _responder(data: dict[str, Any]) -> dict[str, Any]:
        calls.append(data)
        # First call is json_schema -> return plain text (schema ignored)
        if "response_format" in data and data["response_format"].get("type") == "json_schema":
            return load_llm_cassette("probe", "schema_ignored")
        # Second call is tool_call -> return tool_call_ok
        if "tools" in data:
            return load_llm_cassette("probe", "tool_call_ok")
        return {"status": 500, "text": "error"}

    mock_llm(aioclient_mock, base_url, _responder)

    res = await async_open_options_step(hass, entry, "llm_endpoint")
    await hass.config_entries.options.async_configure(
        res["flow_id"],
        user_input={
            CONF_LLM_BASE_URL: base_url,
            CONF_LLM_MODEL: "m1",
        },
    )
    await hass.async_block_till_done()

    res3 = await hass.config_entries.options.async_configure(res["flow_id"])
    assert res3["type"] is FlowResultType.CREATE_ENTRY
    assert entry.data[CONF_LLM_PROBE]["strategy"] == "tool_call"
    assert entry.data[CONF_LLM_PROBE]["model_returned"] == "m1-tool"
    assert len(calls) == 2


async def test_llm_step_ladder_fallthrough_to_json_object_and_fenced(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    aioclient_mock: Any,
) -> None:
    """Probe reaches json_object or fenced when earlier rungs are unsupported."""
    energy_manager.data = std_prefs
    async_set_power(hass, "sensor.grid_power", 500)
    async_set_power(hass, "sensor.solar_power", 1200)
    async_set_power(hass, "sensor.battery_power", -300)
    async_set_power(hass, "sensor.washer_power", 200)

    entry = await async_setup_via_flow(hass)

    base_url = "http://llm.test:4000/v1"

    # Test json_object rung
    def _responder_json_object(data: dict[str, Any]) -> dict[str, Any]:
        if "response_format" in data and data["response_format"].get("type") == "json_schema":
            return load_llm_cassette("probe", "unsupported_response_format")
        if "tools" in data:
            return load_llm_cassette("probe", "unsupported_response_format")
        if "response_format" in data and data["response_format"].get("type") == "json_object":
            return load_llm_cassette("probe", "json_object_ok")
        return {"status": 500, "text": "error"}

    mock_llm(aioclient_mock, base_url, _responder_json_object)

    res = await async_open_options_step(hass, entry, "llm_endpoint")
    await hass.config_entries.options.async_configure(
        res["flow_id"],
        user_input={
            CONF_LLM_BASE_URL: base_url,
            CONF_LLM_MODEL: "m1",
        },
    )
    await hass.async_block_till_done()

    res3 = await hass.config_entries.options.async_configure(res["flow_id"])
    assert res3["type"] is FlowResultType.CREATE_ENTRY
    assert entry.data[CONF_LLM_PROBE]["strategy"] == "json_object"

    # Test fenced rung
    aioclient_mock.clear_requests()

    def _responder_fenced(data: dict[str, Any]) -> dict[str, Any]:
        if "response_format" in data or "tools" in data:
            return load_llm_cassette("probe", "unsupported_response_format")
        # No response_format -> fenced rung
        return load_llm_cassette("probe", "fenced_ok")

    mock_llm(aioclient_mock, base_url, _responder_fenced)

    res_f = await async_open_options_step(hass, entry, "llm_endpoint")
    await hass.config_entries.options.async_configure(
        res_f["flow_id"],
        user_input={
            CONF_LLM_BASE_URL: base_url,
            CONF_LLM_MODEL: "m1",
        },
    )
    await hass.async_block_till_done()

    res_f3 = await hass.config_entries.options.async_configure(res_f["flow_id"])
    assert res_f3["type"] is FlowResultType.CREATE_ENTRY
    assert entry.data[CONF_LLM_PROBE]["strategy"] == "fenced"


async def test_llm_step_param_profile_retries(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    aioclient_mock: Any,
) -> None:
    """400 naming max_completion_tokens retries with max_tokens; temperature is never sent."""
    energy_manager.data = std_prefs
    async_set_power(hass, "sensor.grid_power", 500)
    async_set_power(hass, "sensor.solar_power", 1200)
    async_set_power(hass, "sensor.battery_power", -300)
    async_set_power(hass, "sensor.washer_power", 200)

    entry = await async_setup_via_flow(hass)
    base_url = "http://llm.test:4000/v1"

    req_count = 0

    def _responder(data: dict[str, Any]) -> dict[str, Any]:
        nonlocal req_count
        req_count += 1
        if "max_completion_tokens" in data:
            return load_llm_cassette("probe", "bad_param_max_completion_tokens")
        if "temperature" in data:
            return load_llm_cassette("probe", "bad_param_temperature")
        return load_llm_cassette("probe", "json_schema_ok")

    mock_llm(aioclient_mock, base_url, _responder)

    res = await async_open_options_step(hass, entry, "llm_endpoint")
    await hass.config_entries.options.async_configure(
        res["flow_id"],
        user_input={
            CONF_LLM_BASE_URL: base_url,
            CONF_LLM_MODEL: "m1",
        },
    )
    await hass.async_block_till_done()

    res3 = await hass.config_entries.options.async_configure(res["flow_id"])
    assert res3["type"] is FlowResultType.CREATE_ENTRY
    assert entry.data[CONF_LLM_PROBE]["token_param"] == "max_tokens"
    assert entry.data[CONF_LLM_PROBE]["send_temperature"] is False
    # Only the token parameter needs a retry: the default profile sends no temperature
    assert req_count == 2


async def test_llm_step_probe_flags_model_mismatch(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    aioclient_mock: Any,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A router fallback to another model is stored as model_mismatch and logged, not failed."""
    energy_manager.data = std_prefs
    async_set_power(hass, "sensor.grid_power", 500)
    async_set_power(hass, "sensor.solar_power", 1200)
    async_set_power(hass, "sensor.battery_power", -300)
    async_set_power(hass, "sensor.washer_power", 200)

    entry = await async_setup_via_flow(hass)
    base_url = "http://llm.test:4000/v1"

    def _responder(data: dict[str, Any]) -> dict[str, Any]:
        cassette = load_llm_cassette("probe", "json_schema_ok")
        cassette["json"]["model"] = "xiaomi/mimo-v2.6-flash"
        return cassette

    mock_llm(aioclient_mock, base_url, _responder)

    res = await async_open_options_step(hass, entry, "llm_endpoint")
    with caplog.at_level(logging.WARNING, logger="custom_components.powerguesser"):
        await hass.config_entries.options.async_configure(
            res["flow_id"],
            user_input={
                CONF_LLM_BASE_URL: base_url,
                CONF_LLM_API_KEY: "sk-mismatch-secret",
                CONF_LLM_MODEL: "kimi-for-coding",
            },
        )
        await hass.async_block_till_done()
        res3 = await hass.config_entries.options.async_configure(res["flow_id"])

    assert res3["type"] is FlowResultType.CREATE_ENTRY
    probe = entry.data[CONF_LLM_PROBE]
    assert probe["strategy"] == "json_schema"
    assert probe["model_returned"] == "xiaomi/mimo-v2.6-flash"
    assert probe["model_mismatch"] is True
    warnings = [
        r for r in caplog.records
        if r.levelno == logging.WARNING and "kimi-for-coding" in r.getMessage()
    ]
    assert warnings, caplog.text
    assert "xiaomi/mimo-v2.6-flash" in warnings[0].getMessage()
    assert "sk-mismatch-secret" not in caplog.text


async def test_llm_step_probe_failures_preserve_data(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    aioclient_mock: Any,
) -> None:
    """401 (invalid_auth), connection error, and unusable model show errors and preserve entry.data."""
    energy_manager.data = std_prefs
    async_set_power(hass, "sensor.grid_power", 500)
    async_set_power(hass, "sensor.solar_power", 1200)
    async_set_power(hass, "sensor.battery_power", -300)
    async_set_power(hass, "sensor.washer_power", 200)

    entry = await async_setup_via_flow(hass)
    initial_data = dict(entry.data)
    base_url = "http://llm.test:4000/v1"

    # 1. 401 -> invalid_auth
    mock_llm(aioclient_mock, base_url, lambda data: load_llm_cassette("probe", "unauthorized"))
    res = await async_open_options_step(hass, entry, "llm_endpoint")
    await hass.config_entries.options.async_configure(
        res["flow_id"],
        user_input={CONF_LLM_BASE_URL: base_url, CONF_LLM_MODEL: "m1"},
    )
    await hass.async_block_till_done()
    res_err = await hass.config_entries.options.async_configure(res["flow_id"])
    assert res_err["type"] is FlowResultType.FORM
    assert res_err["errors"] == {"base": "invalid_auth"}
    assert entry.data == initial_data

    # 2. Connection error -> cannot_connect
    aioclient_mock.clear_requests()
    mock_llm(aioclient_mock, base_url, lambda data: {"exc": "connect"})
    res = await async_open_options_step(hass, entry, "llm_endpoint")
    await hass.config_entries.options.async_configure(
        res["flow_id"],
        user_input={CONF_LLM_BASE_URL: base_url, CONF_LLM_MODEL: "m1"},
    )
    await hass.async_block_till_done()
    res_err = await hass.config_entries.options.async_configure(res["flow_id"])
    assert res_err["type"] is FlowResultType.FORM
    assert res_err["errors"] == {"base": "cannot_connect"}
    assert entry.data == initial_data

    # 3. All rungs invalid -> model_unusable
    aioclient_mock.clear_requests()
    mock_llm(aioclient_mock, base_url, lambda data: load_llm_cassette("probe", "schema_ignored"))
    res = await async_open_options_step(hass, entry, "llm_endpoint")
    await hass.config_entries.options.async_configure(
        res["flow_id"],
        user_input={CONF_LLM_BASE_URL: base_url, CONF_LLM_MODEL: "m1"},
    )
    await hass.async_block_till_done()
    res_err = await hass.config_entries.options.async_configure(res["flow_id"])
    assert res_err["type"] is FlowResultType.FORM
    assert res_err["errors"] == {"base": "model_unusable"}
    assert entry.data == initial_data


async def test_llm_step_clearing_removes_keys(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
) -> None:
    """Submitting empty base URL removes all four llm keys from entry.data."""
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
            CONF_LLM_API_KEY: "secret",
            CONF_LLM_MODEL: "m1",
            CONF_LLM_PROBE: {"strategy": "json_schema"},
        },
    )

    res = await async_open_options_step(hass, entry, "llm_endpoint")
    res2 = await hass.config_entries.options.async_configure(
        res["flow_id"],
        user_input={CONF_LLM_BASE_URL: ""},
    )
    assert res2["type"] is FlowResultType.CREATE_ENTRY
    for key in (CONF_LLM_BASE_URL, CONF_LLM_API_KEY, CONF_LLM_MODEL, CONF_LLM_PROBE):
        assert key not in entry.data


async def test_llm_step_key_retention(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    aioclient_mock: Any,
) -> None:
    """Empty key field keeps stored key when host is unchanged, clears key when host changed."""
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
            CONF_LLM_API_KEY: "secret-key",
            CONF_LLM_MODEL: "m1",
        },
    )

    cassette = load_llm_cassette("probe", "json_schema_ok")
    mock_llm(aioclient_mock, "http://llm.test:4000/v1", lambda data: cassette)
    mock_llm(aioclient_mock, "http://other.test:4000/v1", lambda data: cassette)

    # 1. Same host, empty key -> keeps stored key
    res = await async_open_options_step(hass, entry, "llm_endpoint")
    await hass.config_entries.options.async_configure(
        res["flow_id"],
        user_input={
            CONF_LLM_BASE_URL: "http://llm.test:4000/v1",
            CONF_LLM_API_KEY: "",
            CONF_LLM_MODEL: "m2",
        },
    )
    await hass.async_block_till_done()
    await hass.config_entries.options.async_configure(res["flow_id"])
    assert entry.data[CONF_LLM_API_KEY] == "secret-key"

    # 2. Changed host, empty key -> clears key
    res2 = await async_open_options_step(hass, entry, "llm_endpoint")
    await hass.config_entries.options.async_configure(
        res2["flow_id"],
        user_input={
            CONF_LLM_BASE_URL: "http://other.test:4000/v1",
            CONF_LLM_API_KEY: "",
            CONF_LLM_MODEL: "m2",
        },
    )
    await hass.async_block_till_done()
    await hass.config_entries.options.async_configure(res2["flow_id"])
    assert entry.data[CONF_LLM_API_KEY] == ""


async def test_llm_step_key_reused_only_for_same_origin_and_clearable(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    aioclient_mock: Any,
) -> None:
    """The stored key is reused only for the same scheme, host and port, and can be cleared (CR-04)."""
    energy_manager.data = std_prefs
    async_set_power(hass, "sensor.grid_power", 500)
    async_set_power(hass, "sensor.solar_power", 1200)
    async_set_power(hass, "sensor.battery_power", -300)
    async_set_power(hass, "sensor.washer_power", 200)

    entry = await async_setup_via_flow(hass)
    cassette = load_llm_cassette("probe", "json_schema_ok")
    stored = "https://nas.lan:4000/v1"
    for url in (
        stored,
        "http://nas.lan:11434/v1",
        "https://nas.lan:5000/v1",
        "http://nas.lan:4000/v1",
        "https://nas.lan/v1",
    ):
        mock_llm(aioclient_mock, url, lambda data: cassette)

    async def _save(url: str, key: str = "", clear: bool | None = None) -> None:
        hass.config_entries.async_update_entry(
            entry,
            data={
                **entry.data,
                CONF_LLM_BASE_URL: stored,
                CONF_LLM_API_KEY: "litellm-master",
                CONF_LLM_MODEL: "m1",
            },
        )
        aioclient_mock.mock_calls.clear()
        res = await async_open_options_step(hass, entry, "llm_endpoint")
        user_input: dict[str, Any] = {
            CONF_LLM_BASE_URL: url,
            CONF_LLM_API_KEY: key,
            CONF_LLM_MODEL: "m2",
        }
        if clear is not None:
            user_input[CONF_LLM_CLEAR_KEY] = clear
        await hass.config_entries.options.async_configure(res["flow_id"], user_input=user_input)
        await hass.async_block_till_done()
        await hass.config_entries.options.async_configure(res["flow_id"])

    def _sent_auth() -> str | None:
        assert aioclient_mock.mock_calls
        return aioclient_mock.mock_calls[-1][3].get("Authorization")

    # Same scheme, host and port: the stored key is kept and sent
    await _save(stored)
    assert entry.data[CONF_LLM_API_KEY] == "litellm-master"
    assert _sent_auth() == "Bearer litellm-master"

    # Same host, other scheme and/or port: never reused, never sent
    for url in (
        "http://nas.lan:11434/v1",
        "https://nas.lan:5000/v1",
        "http://nas.lan:4000/v1",
        "https://nas.lan/v1",
    ):
        await _save(url)
        assert entry.data[CONF_LLM_API_KEY] == "", url
        assert _sent_auth() is None, url

    # Explicit clear on the same origin removes the stored key
    await _save(stored, clear=True)
    assert entry.data[CONF_LLM_API_KEY] == ""
    assert _sent_auth() is None

    # A newly typed key always wins, also together with clear
    await _save(stored, key="new-key", clear=True)
    assert entry.data[CONF_LLM_API_KEY] == "new-key"


async def test_llm_step_save_does_not_rebuild_hub(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    aioclient_mock: Any,
) -> None:
    """Saving LLM step or unchanged phases does not call hub.async_request_rebuild; changing L1 does."""
    energy_manager.data = std_prefs
    async_set_power(hass, "sensor.grid_power", 500)
    async_set_power(hass, "sensor.solar_power", 1200)
    async_set_power(hass, "sensor.battery_power", -300)
    async_set_power(hass, "sensor.washer_power", 200)

    entry = await async_setup_via_flow(hass)
    hub = entry.runtime_data

    rebuild_called = 0
    orig_rebuild = hub.async_request_rebuild

    def _rebuild_spy() -> None:
        nonlocal rebuild_called
        rebuild_called += 1
        orig_rebuild()

    hub.async_request_rebuild = _rebuild_spy

    base_url = "http://llm.test:4000/v1"
    cassette = load_llm_cassette("probe", "json_schema_ok")
    mock_llm(aioclient_mock, base_url, lambda data: cassette)

    # 1. Saving LLM step does NOT rebuild hub
    res = await async_open_options_step(hass, entry, "llm_endpoint")
    await hass.config_entries.options.async_configure(
        res["flow_id"],
        user_input={
            CONF_LLM_BASE_URL: base_url,
            CONF_LLM_API_KEY: "secret",
            CONF_LLM_MODEL: "m1",
        },
    )
    await hass.async_block_till_done()
    await hass.config_entries.options.async_configure(res["flow_id"])
    await hass.async_block_till_done()
    assert rebuild_called == 0

    # 2. Saving phases unchanged does NOT rebuild hub
    res_phase = await async_open_options_step(hass, entry, "phases")
    await hass.config_entries.options.async_configure(
        res_phase["flow_id"],
        user_input={},
    )
    await hass.async_block_till_done()
    assert rebuild_called == 0

    # 3. Changing L1 DOES rebuild hub
    async_set_power(hass, "sensor.l1_power", 500)
    res_phase2 = await async_open_options_step(hass, entry, "phases")
    await hass.config_entries.options.async_configure(
        res_phase2["flow_id"],
        user_input={CONF_PHASE_L1: "sensor.l1_power"},
    )
    await hass.async_block_till_done()
    assert rebuild_called == 1


async def test_llm_step_preserves_invert_phases_option(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    aioclient_mock: Any,
) -> None:
    """Saving LLM step preserves every existing option including invert_phases."""
    energy_manager.data = std_prefs
    async_set_power(hass, "sensor.grid_power", 500)
    async_set_power(hass, "sensor.solar_power", 1200)
    async_set_power(hass, "sensor.battery_power", -300)
    async_set_power(hass, "sensor.washer_power", 200)

    entry = await async_setup_via_flow(hass)
    hass.config_entries.async_update_entry(
        entry,
        options={
            CONF_PHASE_L1: "sensor.l1",
            CONF_PHASE_L2: "sensor.l2",
            CONF_PHASE_L3: "sensor.l3",
            CONF_INVERT_PHASES: True,
        },
    )
    expected_options = dict(entry.options)

    base_url = "http://llm.test:4000/v1"
    cassette = load_llm_cassette("probe", "json_schema_ok")
    mock_llm(aioclient_mock, base_url, lambda data: cassette)

    # Save LLM step (probe success)
    res = await async_open_options_step(hass, entry, "llm_endpoint")
    await hass.config_entries.options.async_configure(
        res["flow_id"],
        user_input={
            CONF_LLM_BASE_URL: base_url,
            CONF_LLM_API_KEY: "secret",
            CONF_LLM_MODEL: "m1",
        },
    )
    await hass.async_block_till_done()
    await hass.config_entries.options.async_configure(res["flow_id"])
    await hass.async_block_till_done()
    assert entry.options == expected_options

    # Clear LLM endpoint (empty URL)
    res2 = await async_open_options_step(hass, entry, "llm_endpoint")
    await hass.config_entries.options.async_configure(
        res2["flow_id"],
        user_input={CONF_LLM_BASE_URL: ""},
    )
    await hass.async_block_till_done()
    assert entry.options == expected_options


async def test_phases_save_keeps_other_options(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
) -> None:
    """Phases step preserves non-phase options across OK, SKIPPED, empty, and phase_check paths."""
    energy_manager.data = std_prefs
    async_set_power(hass, "sensor.grid_power", 500)
    async_set_power(hass, "sensor.solar_power", 1200)
    async_set_power(hass, "sensor.battery_power", -300)
    async_set_power(hass, "sensor.washer_power", 200)
    async_set_power(hass, "sensor.l1_power", 500)

    entry = await async_setup_via_flow(hass)
    hass.config_entries.async_update_entry(
        entry,
        options={"share_entity_names": True, CONF_INVERT_PHASES: True},
    )

    # 1. OK path (matches grid sum): keeps other keys and carries invert_phases (WR-04)
    res = await async_open_options_step(hass, entry, "phases")
    res2 = await hass.config_entries.options.async_configure(
        res["flow_id"],
        user_input={CONF_PHASE_L1: "sensor.l1_power"},
    )
    assert res2["type"] is FlowResultType.CREATE_ENTRY
    await hass.async_block_till_done()
    assert entry.options == {
        "share_entity_names": True,
        CONF_PHASE_L1: "sensor.l1_power",
        CONF_INVERT_PHASES: True,
    }

    # 2. SKIPPED path: home-only source skips phase check; keeps invert_phases if present
    hass.config_entries.async_update_entry(
        entry,
        options={"share_entity_names": True, CONF_INVERT_PHASES: True},
    )
    energy_manager.data = {
        "energy_sources": [{"type": "home", "stat_rate": "sensor.home_power"}],
        "device_consumption": [],
    }
    async_set_power(hass, "sensor.home_power", 500)
    async_dispatcher_send(hass, SIGNAL_PREFS_UPDATED)
    await hass.async_block_till_done()
    res_skip = await async_open_options_step(hass, entry, "phases")
    res_skip2 = await hass.config_entries.options.async_configure(
        res_skip["flow_id"],
        user_input={CONF_PHASE_L1: "sensor.l1_power"},
    )
    assert res_skip2["type"] is FlowResultType.CREATE_ENTRY
    await hass.async_block_till_done()
    assert entry.options == {
        "share_entity_names": True,
        CONF_INVERT_PHASES: True,
        CONF_PHASE_L1: "sensor.l1_power",
    }

    # 3. Empty mapping path: clears phases, keeps other keys
    res_empty = await async_open_options_step(hass, entry, "phases")
    res_empty2 = await hass.config_entries.options.async_configure(
        res_empty["flow_id"],
        user_input={},
    )
    assert res_empty2["type"] is FlowResultType.CREATE_ENTRY
    await hass.async_block_till_done()
    assert entry.options == {"share_entity_names": True}

    # 4. phase_check path: mismatch -> user submits invert_phases=True
    energy_manager.data = std_prefs
    async_dispatcher_send(hass, SIGNAL_PREFS_UPDATED)
    await hass.async_block_till_done()
    async_set_power(hass, "sensor.l1_power", 100)
    res_chk = await async_open_options_step(hass, entry, "phases")
    res_chk2 = await hass.config_entries.options.async_configure(
        res_chk["flow_id"],
        user_input={CONF_PHASE_L1: "sensor.l1_power"},
    )
    assert res_chk2["type"] is FlowResultType.FORM
    assert res_chk2["step_id"] == "phase_check"
    res_chk3 = await hass.config_entries.options.async_configure(
        res_chk2["flow_id"],
        user_input={CONF_INVERT_PHASES: True},
    )
    assert res_chk3["type"] is FlowResultType.CREATE_ENTRY
    await hass.async_block_till_done()
    assert entry.options == {
        "share_entity_names": True,
        CONF_PHASE_L1: "sensor.l1_power",
        CONF_INVERT_PHASES: True,
    }


async def test_probe_task_exception_retrieved(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Unexpected exception in probe task is retrieved and shows cannot_connect without 'never retrieved' warning."""
    energy_manager.data = std_prefs
    async_set_power(hass, "sensor.grid_power", 500)
    async_set_power(hass, "sensor.solar_power", 1200)
    async_set_power(hass, "sensor.battery_power", -300)
    async_set_power(hass, "sensor.washer_power", 200)

    entry = await async_setup_via_flow(hass)
    initial_data = dict(entry.data)

    async def _failing_probe(self: Any, model: str) -> dict[str, Any]:
        raise RuntimeError("Unexpected failure inside probe task")

    with patch(
        "custom_components.powerguesser.llm_client.LlmClient.async_probe",
        _failing_probe,
    ):
        res = await async_open_options_step(hass, entry, "llm_endpoint")
        await hass.config_entries.options.async_configure(
            res["flow_id"],
            user_input={
                CONF_LLM_BASE_URL: "http://llm.test:4000/v1",
                CONF_LLM_MODEL: "m1",
            },
        )
        await hass.async_block_till_done()
        res_err = await hass.config_entries.options.async_configure(res["flow_id"])
        assert res_err["type"] is FlowResultType.FORM
        assert res_err["errors"] == {"base": "cannot_connect"}
        assert entry.data == initial_data

    # Ensure no 'never retrieved' warning in logs
    assert "never retrieved" not in caplog.text

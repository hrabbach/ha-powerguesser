"""Tests for learning status sensor, learn button, and interrupted run handling."""

import asyncio
from datetime import timedelta
import json
import logging
from typing import Any
import pytest

from homeassistant.components.sensor import SensorDeviceClass
from homeassistant.const import EntityCategory, Platform
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ServiceValidationError
from homeassistant.helpers.storage import Store
import homeassistant.util.dt as dt_util

from custom_components.powerguesser.const import (
    DOMAIN,
    LEARNING_STORE_KEY,
    LEARNING_STORES_VERSION,
)
from tests.conftest import (
    async_fill_synth_buffer,
    async_seed_consent,
    async_seed_llm,
    async_set_power,
    async_setup_via_flow,
    mock_llm,
    proposal_responder,
)
from tests.fixtures.learning.synth import SynthSpec


def _set_std_powers(hass: HomeAssistant) -> None:
    async_set_power(hass, "sensor.grid_power", 500)
    async_set_power(hass, "sensor.solar_power", 1200)
    async_set_power(hass, "sensor.battery_power", -300)
    async_set_power(hass, "sensor.washer_power", 200)


async def test_learning_status_sensor_initial_state(
    hass: HomeAssistant,
    energy_manager: object,
    std_prefs: dict,
) -> None:
    """Test learning status sensor attributes before any run."""
    energy_manager.data = std_prefs
    _set_std_powers(hass)
    entry = await async_setup_via_flow(hass)

    state = hass.states.get("sensor.powerguesser_learning_status")
    assert state is not None
    # Blocked initially because no endpoint is configured
    assert state.state in ("idle", "blocked")
    attrs = state.attributes
    assert attrs.get("device_class") == SensorDeviceClass.ENUM
    assert attrs.get("options") == ["idle", "running", "blocked", "capped", "failed"]
    assert attrs.get("last_run_start") is None
    assert attrs.get("last_run_end") is None
    assert attrs.get("last_result") is None
    assert attrs.get("last_error") is None
    assert attrs.get("pending_candidates") == 0
    assert attrs.get("tokens_this_run") == 0
    assert attrs.get("tokens_this_month") == 0
    assert attrs.get("cost_this_month") is None
    assert "buffer_usable_days" in attrs
    assert "blocked_reason" in attrs


async def test_learning_status_sensor_lifecycle(
    hass: HomeAssistant,
    energy_manager: object,
    std_prefs: dict,
    aioclient_mock: object,
    inprocess_worker: None,
) -> None:
    """Test learning status sensor during run and after success."""
    energy_manager.data = std_prefs
    _set_std_powers(hass)
    entry = await async_setup_via_flow(hass)
    await async_fill_synth_buffer(hass, entry, spec=SynthSpec(days=22, washer=True))
    await async_seed_llm(hass, entry, base_url="http://llm.test/v1")
    await async_seed_consent(hass, host="llm.test", names=False)

    unblock_llm = asyncio.Event()
    base_resp = proposal_responder(2)

    async def _waiting_responder(data: object) -> dict:
        await unblock_llm.wait()
        return base_resp(data)

    mock_llm(aioclient_mock, "http://llm.test/v1", _waiting_responder)

    supervisor = entry.runtime_data.learning
    # Start run
    await supervisor.async_request_run("service")
    await asyncio.sleep(0.1)

    state = hass.states.get("sensor.powerguesser_learning_status")
    assert state is not None
    assert state.state == "running"

    unblock_llm.set()
    await supervisor.run_task

    state_after = hass.states.get("sensor.powerguesser_learning_status")
    assert state_after is not None
    assert state_after.state == "idle"
    attrs = state_after.attributes
    assert attrs["last_run_start"] is not None
    assert attrs["last_run_end"] is not None
    assert attrs["pending_candidates"] >= 1
    assert attrs["tokens_this_run"] > 0
    assert attrs["tokens_this_month"] > 0

    last_res = attrs["last_result"]
    assert last_res is not None
    assert last_res["stage"] == "compile_score"
    assert last_res["candidates"] >= 1
    assert "train_share" in last_res
    assert "val_share" in last_res
    assert "preview" in last_res
    assert "rounds" in last_res


async def test_tokens_month_rollover(
    hass: HomeAssistant,
    energy_manager: object,
    std_prefs: dict,
    aioclient_mock: object,
    inprocess_worker: None,
    freezer: object,
) -> None:
    """Test tokens_this_month resets when moving to a new calendar month."""
    freezer.move_to("2026-05-15 12:00:00")
    energy_manager.data = std_prefs
    _set_std_powers(hass)
    entry = await async_setup_via_flow(hass)
    await async_fill_synth_buffer(hass, entry, spec=SynthSpec(days=22, washer=True))
    await async_seed_llm(hass, entry, base_url="http://llm.test/v1")
    await async_seed_consent(hass, host="llm.test", names=False)
    mock_llm(aioclient_mock, "http://llm.test/v1", proposal_responder(2))

    supervisor = entry.runtime_data.learning
    await supervisor.async_request_run("service")
    await supervisor.run_task

    state = hass.states.get("sensor.powerguesser_learning_status")
    assert state is not None
    tokens_may = state.attributes["tokens_this_month"]
    assert tokens_may > 0

    # Advance to June (use noon to be well into June across all local timezones)
    freezer.move_to("2026-06-02 12:00:00")
    supervisor.invalidate_usable_days_cache()
    await supervisor.async_settings_changed()

    state_june = hass.states.get("sensor.powerguesser_learning_status")
    assert state_june is not None
    assert state_june.attributes["tokens_this_month"] == 0


async def test_learn_button_press_triggers_run_and_refusal(
    hass: HomeAssistant,
    energy_manager: object,
    std_prefs: dict,
    aioclient_mock: object,
    inprocess_worker: None,
) -> None:
    """Test pressing Learn button triggers run, or raises ServiceValidationError when blocked."""
    energy_manager.data = std_prefs
    _set_std_powers(hass)
    entry = await async_setup_via_flow(hass)

    # Initially blocked (no endpoint) -> button press raises ServiceValidationError
    with pytest.raises(ServiceValidationError) as exc:
        await hass.services.async_call(
            "button", "press", {"entity_id": "button.powerguesser_learn_devices"}, blocking=True
        )
    assert exc.value.translation_key == "no_endpoint"

    # Now configure endpoint, consent, and buffer
    await async_fill_synth_buffer(hass, entry, spec=SynthSpec(days=22, washer=True))
    await async_seed_llm(hass, entry, base_url="http://llm.test/v1")
    await async_seed_consent(hass, host="llm.test", names=False)
    mock_llm(aioclient_mock, "http://llm.test/v1", proposal_responder(2))

    # Press button
    await hass.services.async_call(
        "button", "press", {"entity_id": "button.powerguesser_learn_devices"}, blocking=True
    )
    supervisor = entry.runtime_data.learning
    assert supervisor.run_task is not None
    await supervisor.run_task
    assert supervisor.coordinator.data["status"] == "idle"


async def test_interrupted_run_loads_as_failed(
    hass: HomeAssistant,
    energy_manager: object,
    std_prefs: dict,
    aioclient_mock: object,
    inprocess_worker: None,
) -> None:
    """Test learning Store saved as running loads as failed with last_error 'interrupted'."""
    # Seed learning store with running status before setting up entry
    store = Store(hass, LEARNING_STORES_VERSION, LEARNING_STORE_KEY)
    await store.async_save(
        {
            "status": "running",
            "blocked_reason": None,
            "last_run": None,
            "last_error": None,
            "tokens_run": {"input": 100, "output": 50},
            "month": {"key": "2026-11", "input": 100, "output": 50, "cost": None},
            "usable_days": {"value": 15.0, "computed_at": None},
        }
    )

    energy_manager.data = std_prefs
    _set_std_powers(hass)
    entry = await async_setup_via_flow(hass)

    state = hass.states.get("sensor.powerguesser_learning_status")
    assert state is not None
    assert state.state == "failed"
    assert state.attributes.get("last_error") == "interrupted"

    # Verify that the next trigger starts a fresh run
    await async_fill_synth_buffer(hass, entry, spec=SynthSpec(days=22, washer=True))
    await async_seed_llm(hass, entry, base_url="http://llm.test/v1")
    await async_seed_consent(hass, host="llm.test", names=False)
    mock_llm(aioclient_mock, "http://llm.test/v1", proposal_responder(2))

    supervisor = entry.runtime_data.learning
    await supervisor.async_request_run("service")
    assert supervisor.run_task is not None
    await supervisor.run_task
    assert supervisor.coordinator.data["status"] == "idle"


@pytest.mark.parametrize(
    "stored_form",
    [
        lambda k: k,
        lambda k: json.dumps(k)[1:-1],
        lambda k: json.dumps(k, ensure_ascii=False)[1:-1],
    ],
    ids=["plain", "json-ascii", "json-unicode"],
)
async def test_persisted_last_error_with_key_is_redacted_on_load(
    hass: HomeAssistant,
    energy_manager: object,
    std_prefs: dict,
    stored_form: Any,
) -> None:
    """A last_error stored before 03-10 that carries the key is scrubbed on startup (IN-10).

    Pre-03-10 code stored raw provider text, so the key may sit there plain or
    JSON-escaped. Neither form may reach the sensor attribute or stay in the Store.
    """
    api_key = 'sk-"tëst"-secret-0123'
    energy_manager.data = std_prefs
    _set_std_powers(hass)
    entry = await async_setup_via_flow(hass)
    await async_seed_llm(hass, entry, base_url="http://llm.test/v1", api_key=api_key)
    assert await hass.config_entries.async_unload(entry.entry_id)
    await hass.async_block_till_done()

    leaked = f'{{"detail": "invalid token {stored_form(api_key)}"}}'
    store = Store(hass, LEARNING_STORES_VERSION, LEARNING_STORE_KEY)
    await store.async_save(
        {
            "status": "failed",
            "blocked_reason": None,
            "last_run": None,
            "last_error": f"http_400: {leaked}",
            "tokens_run": {"input": 0, "output": 0},
            "month": {"key": "2026-11", "input": 0, "output": 0, "cost": None},
            "usable_days": {"value": 0.0, "computed_at": None},
        }
    )

    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    state = hass.states.get("sensor.powerguesser_learning_status")
    assert state is not None
    last_error = state.attributes.get("last_error")
    assert last_error == 'http_400: {"detail": "invalid token **REDACTED**"}'
    stored = await Store(hass, LEARNING_STORES_VERSION, LEARNING_STORE_KEY).async_load()
    assert stored["last_error"] == last_error
    for form in (api_key, json.dumps(api_key)[1:-1], json.dumps(api_key, ensure_ascii=False)[1:-1]):
        assert form not in json.dumps(stored, ensure_ascii=False)


async def test_persisted_last_error_without_key_is_kept(
    hass: HomeAssistant,
    energy_manager: object,
    std_prefs: dict,
) -> None:
    """A stored last_error without the key loads unchanged (IN-10)."""
    energy_manager.data = std_prefs
    _set_std_powers(hass)
    store = Store(hass, LEARNING_STORES_VERSION, LEARNING_STORE_KEY)
    await store.async_save(
        {
            "status": "failed",
            "blocked_reason": None,
            "last_run": None,
            "last_error": "http_400: model not found",
            "tokens_run": {"input": 0, "output": 0},
            "month": {"key": "2026-11", "input": 0, "output": 0, "cost": None},
            "usable_days": {"value": 0.0, "computed_at": None},
        }
    )

    await async_setup_via_flow(hass)

    state = hass.states.get("sensor.powerguesser_learning_status")
    assert state is not None
    assert state.attributes.get("last_error") == "http_400: model not found"


async def test_phase2_entities_coexist(
    hass: HomeAssistant,
    energy_manager: object,
    std_prefs: dict,
) -> None:
    """Verify Phase 2 sensor entities still exist alongside the new status sensor."""
    energy_manager.data = std_prefs
    _set_std_powers(hass)
    entry = await async_setup_via_flow(hass)

    assert hass.states.get("sensor.powerguesser_bootstrapped_usable_history") is not None
    assert hass.states.get("sensor.powerguesser_effective_sample_interval") is not None
    assert hass.states.get("sensor.powerguesser_learning_status") is not None
    assert hass.states.get("button.powerguesser_learn_devices") is not None


async def test_probe_model_mismatch_exposed_on_status_sensor(
    hass: HomeAssistant,
    energy_manager: object,
    std_prefs: dict,
) -> None:
    """The probe's requested-vs-returned model check is a status sensor attribute (03-09 A2)."""
    energy_manager.data = std_prefs
    _set_std_powers(hass)
    entry = await async_setup_via_flow(hass)

    state = hass.states.get("sensor.powerguesser_learning_status")
    assert state.attributes.get("llm_model_mismatch") is None
    assert state.attributes.get("llm_model_returned") is None

    await async_seed_llm(
        hass,
        entry,
        model="kimi-for-coding",
        probe={
            "strategy": "json_schema",
            "token_param": "max_completion_tokens",
            "send_temperature": False,
            "model_returned": "xiaomi/mimo-v2.6-flash",
            "model_mismatch": True,
            "probed_at": dt_util.utcnow().isoformat(),
        },
    )
    state = hass.states.get("sensor.powerguesser_learning_status")
    assert state.attributes["llm_model_mismatch"] is True
    assert state.attributes["llm_model_returned"] == "xiaomi/mimo-v2.6-flash"

    # A probe stored before model_mismatch existed is checked from model_returned
    await async_seed_llm(
        hass,
        entry,
        model="gpt-4o",
        probe={
            "strategy": "json_schema",
            "token_param": "max_completion_tokens",
            "send_temperature": True,
            "model_returned": "gpt-4o-2024-08-06",
            "probed_at": dt_util.utcnow().isoformat(),
        },
    )
    state = hass.states.get("sensor.powerguesser_learning_status")
    assert state.attributes["llm_model_mismatch"] is False
    assert state.attributes["llm_model_returned"] == "gpt-4o-2024-08-06"


async def test_run_flags_model_mismatch_once(
    hass: HomeAssistant,
    energy_manager: object,
    std_prefs: dict,
    aioclient_mock: object,
    inprocess_worker: None,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Learning requests answered by another model are counted in the run and warned once."""
    energy_manager.data = std_prefs
    _set_std_powers(hass)
    entry = await async_setup_via_flow(hass)
    await async_fill_synth_buffer(hass, entry, spec=SynthSpec(days=22, washer=True))
    await async_seed_llm(hass, entry, base_url="http://llm.test/v1", model="kimi-for-coding")
    await async_seed_consent(hass, host="llm.test", names=False)

    base_resp = proposal_responder(2)

    def _fallback_responder(data: object) -> dict:
        resp = base_resp(data)
        resp["json"]["model"] = "xiaomi/mimo-v2.6-flash"
        return resp

    mock_llm(aioclient_mock, "http://llm.test/v1", _fallback_responder)

    supervisor = entry.runtime_data.learning
    with caplog.at_level(logging.WARNING, logger="custom_components.powerguesser"):
        await supervisor.async_request_run("service")
        await supervisor.run_task

    n_requests = aioclient_mock.call_count
    assert n_requests >= 1

    state = hass.states.get("sensor.powerguesser_learning_status")
    assert state.state == "idle"
    mismatch = state.attributes["model_mismatch_this_run"]
    assert mismatch == {
        "requested": "kimi-for-coding",
        "returned": "xiaomi/mimo-v2.6-flash",
        "requests": n_requests,
    }
    assert state.attributes["last_result"]["model_mismatch"] == mismatch

    warnings = [
        r for r in caplog.records
        if r.levelno == logging.WARNING and "xiaomi/mimo-v2.6-flash" in r.getMessage()
    ]
    assert len(warnings) == 1
    assert "kimi-for-coding" in warnings[0].getMessage()
    assert "sk-test-secret" not in caplog.text

    # Persisted with the run state
    stored = await Store(hass, LEARNING_STORES_VERSION, LEARNING_STORE_KEY).async_load()
    assert stored["model_mismatch_run"] == mismatch


async def test_run_without_model_mismatch_reports_none(
    hass: HomeAssistant,
    energy_manager: object,
    std_prefs: dict,
    aioclient_mock: object,
    inprocess_worker: None,
) -> None:
    """A dated snapshot of the requested model is not a mismatch."""
    energy_manager.data = std_prefs
    _set_std_powers(hass)
    entry = await async_setup_via_flow(hass)
    await async_fill_synth_buffer(hass, entry, spec=SynthSpec(days=22, washer=True))
    await async_seed_llm(hass, entry, base_url="http://llm.test/v1", model="fake")
    await async_seed_consent(hass, host="llm.test", names=False)

    # proposal_responder answers with model "fake-model": "fake" plus a "-model" suffix
    mock_llm(aioclient_mock, "http://llm.test/v1", proposal_responder(2))

    supervisor = entry.runtime_data.learning
    await supervisor.async_request_run("service")
    await supervisor.run_task

    state = hass.states.get("sensor.powerguesser_learning_status")
    assert state.state == "idle"
    assert state.attributes["model_mismatch_this_run"] is None
    assert state.attributes["last_result"]["model_mismatch"] is None

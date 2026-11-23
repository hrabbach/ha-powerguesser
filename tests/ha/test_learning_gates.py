"""Tests for learning gates: endpoint, consent, data sufficiency, and preview/validated modes."""

import asyncio
from datetime import timedelta
import logging
from unittest.mock import patch
import pytest

from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ServiceValidationError
import homeassistant.util.dt as dt_util

from custom_components.powerguesser.const import (
    CONF_LLM_BASE_URL,
    CONF_LLM_MODEL,
    CONF_SHARE_NAMES,
    DOMAIN,
    SERVICE_LEARN,
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


async def test_no_endpoint_blocks_learning(
    hass: HomeAssistant,
    energy_manager: object,
    std_prefs: dict,
    aioclient_mock: object,
) -> None:
    """Test missing endpoint configuration blocks learning and makes zero HTTP requests."""
    energy_manager.data = std_prefs
    _set_std_powers(hass)
    entry = await async_setup_via_flow(hass)
    supervisor = entry.runtime_data.learning

    with pytest.raises(ServiceValidationError) as exc_info:
        await hass.services.async_call(DOMAIN, SERVICE_LEARN, blocking=True)

    assert exc_info.value.translation_key == "no_endpoint"
    assert supervisor.coordinator.data["status"] == "blocked"
    assert supervisor.coordinator.data["blocked_reason"] == "no_endpoint"
    assert len(aioclient_mock.mock_calls) == 0


async def test_no_consent_blocks_learning(
    hass: HomeAssistant,
    energy_manager: object,
    std_prefs: dict,
    aioclient_mock: object,
) -> None:
    """Test missing or mismatched consent blocks learning and makes zero HTTP requests."""
    energy_manager.data = std_prefs
    _set_std_powers(hass)
    entry = await async_setup_via_flow(hass)
    await async_seed_llm(hass, entry, base_url="http://llm.test/v1")
    supervisor = entry.runtime_data.learning

    # Case 1: No consent record
    with pytest.raises(ServiceValidationError) as exc_info:
        await hass.services.async_call(DOMAIN, SERVICE_LEARN, blocking=True)
    assert exc_info.value.translation_key == "no_consent"
    assert supervisor.coordinator.data["status"] == "blocked"
    assert supervisor.coordinator.data["blocked_reason"] == "no_consent"
    assert len(aioclient_mock.mock_calls) == 0

    # Case 2: Consent record for different host
    await supervisor.consent_store.async_grant("http://other.host/v1", False)
    with pytest.raises(ServiceValidationError) as exc_info2:
        await hass.services.async_call(DOMAIN, SERVICE_LEARN, blocking=True)
    assert exc_info2.value.translation_key == "no_consent"
    assert len(aioclient_mock.mock_calls) == 0

    # Case 3: Consent flag mismatch (options has share_entity_names=True, consent granted with False)
    await supervisor.consent_store.async_grant("http://llm.test/v1", False)
    hass.config_entries.async_update_entry(
        entry, options={**entry.options, CONF_SHARE_NAMES: True}
    )
    await hass.async_block_till_done()

    with pytest.raises(ServiceValidationError) as exc_info3:
        await hass.services.async_call(DOMAIN, SERVICE_LEARN, blocking=True)
    assert exc_info3.value.translation_key == "no_consent"
    assert len(aioclient_mock.mock_calls) == 0


async def test_insufficient_data_blocks_learning(
    hass: HomeAssistant,
    energy_manager: object,
    std_prefs: dict,
    aioclient_mock: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test 6.99 usable days raises insufficient_data with days_remaining 1."""
    energy_manager.data = std_prefs
    _set_std_powers(hass)
    entry = await async_setup_via_flow(hass)
    await async_seed_llm(hass, entry, base_url="http://llm.test/v1")
    await async_seed_consent(hass, host="llm.test", names=False)
    supervisor = entry.runtime_data.learning

    # Mock usable days to 6.99
    supervisor.invalidate_usable_days_cache()
    monkeypatch.setattr(
        supervisor, "_async_compute_usable_days", lambda: asyncio.sleep(0, result=6.99)
    )

    with pytest.raises(ServiceValidationError) as exc_info:
        await hass.services.async_call(DOMAIN, SERVICE_LEARN, blocking=True)

    assert exc_info.value.translation_key == "insufficient_data"
    assert exc_info.value.translation_placeholders == {"days_remaining": "1"}
    assert supervisor.coordinator.data["status"] == "blocked"
    assert supervisor.coordinator.data["blocked_reason"] == "insufficient_data"
    assert len(aioclient_mock.mock_calls) == 0


async def test_gate_ordering_hierarchy(
    hass: HomeAssistant,
    energy_manager: object,
    std_prefs: dict,
) -> None:
    """Test gates fail in strict order: no_endpoint -> no_consent -> insufficient_data."""
    energy_manager.data = std_prefs
    _set_std_powers(hass)
    entry = await async_setup_via_flow(hass)
    supervisor = entry.runtime_data.learning

    # Gate 1 fails
    with pytest.raises(ServiceValidationError) as exc_info1:
        await hass.services.async_call(DOMAIN, SERVICE_LEARN, blocking=True)
    assert exc_info1.value.translation_key == "no_endpoint"

    # Add endpoint -> Gate 2 fails
    await async_seed_llm(hass, entry, base_url="http://llm.test/v1")
    with pytest.raises(ServiceValidationError) as exc_info2:
        await hass.services.async_call(DOMAIN, SERVICE_LEARN, blocking=True)
    assert exc_info2.value.translation_key == "no_consent"

    # Add consent -> Gate 3 fails
    await async_seed_consent(hass, host="llm.test", names=False)
    with pytest.raises(ServiceValidationError) as exc_info3:
        await hass.services.async_call(DOMAIN, SERVICE_LEARN, blocking=True)
    assert exc_info3.value.translation_key == "insufficient_data"


async def test_preview_mode_and_validated_mode(
    hass: HomeAssistant,
    energy_manager: object,
    std_prefs: dict,
    aioclient_mock: object,
    inprocess_worker: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test 7.0 days runs in preview mode (validated False); 21.0 runs in validated True."""
    energy_manager.data = std_prefs
    _set_std_powers(hass)
    entry = await async_setup_via_flow(hass)
    await async_fill_synth_buffer(hass, entry, spec=SynthSpec(days=22, washer=True))
    await async_seed_llm(hass, entry, base_url="http://llm.test/v1")
    await async_seed_consent(hass, host="llm.test", names=False)
    mock_llm(aioclient_mock, "http://llm.test/v1", proposal_responder(2))
    supervisor = entry.runtime_data.learning

    # 1. 7.0 days -> validated False
    supervisor.invalidate_usable_days_cache()
    monkeypatch.setattr(
        supervisor, "_async_compute_usable_days", lambda: asyncio.sleep(0, result=7.0)
    )
    await hass.services.async_call(DOMAIN, SERVICE_LEARN, blocking=True)
    await supervisor.run_task
    assert supervisor.coordinator.data["status"] == "idle"
    assert supervisor.coordinator.data["last_run"]["preview"] is True
    assert all(c.get("scores", {}).get("validated") is False for c in supervisor.candidates)

    # 2. 21.0 days -> validated True
    supervisor.invalidate_usable_days_cache()
    monkeypatch.setattr(
        supervisor, "_async_compute_usable_days", lambda: asyncio.sleep(0, result=21.0)
    )
    await hass.services.async_call(DOMAIN, SERVICE_LEARN, blocking=True)
    await supervisor.run_task
    assert supervisor.coordinator.data["status"] == "idle"
    assert supervisor.coordinator.data["last_run"]["preview"] is False
    assert all(c.get("scores", {}).get("validated") is True for c in supervisor.candidates)


async def test_endpoint_concurrency_snapshot(
    hass: HomeAssistant,
    energy_manager: object,
    std_prefs: dict,
    aioclient_mock: object,
    inprocess_worker: None,
) -> None:
    """Test changing entry.data while run waits leaves existing run on snapshot endpoint."""
    energy_manager.data = std_prefs
    _set_std_powers(hass)
    entry = await async_setup_via_flow(hass)
    await async_fill_synth_buffer(hass, entry, spec=SynthSpec(days=22, washer=True))
    await async_seed_llm(hass, entry, base_url="http://llm.old/v1", model="old-model")
    await async_seed_consent(hass, host="llm.old", names=False)
    supervisor = entry.runtime_data.learning

    first_req_received = asyncio.Event()
    unblock_first = asyncio.Event()

    resp_fn = proposal_responder(2)

    def _responder_old(data: object) -> dict:
        first_req_received.set()
        # wait before answering
        return resp_fn(data)

    async def _async_responder_old(data: object) -> dict:
        first_req_received.set()
        await unblock_first.wait()
        return resp_fn(data)

    mock_llm(aioclient_mock, "http://llm.old/v1", _async_responder_old)
    mock_llm(aioclient_mock, "http://llm.new/v1", resp_fn)

    # Start first run
    await hass.services.async_call(DOMAIN, SERVICE_LEARN, blocking=True)
    await first_req_received.wait()

    # Now change endpoint while first run is in-flight
    await async_seed_llm(hass, entry, base_url="http://llm.new/v1", model="new-model")
    await async_seed_consent(hass, host="llm.new", names=False)

    # Unblock first run
    unblock_first.set()
    await supervisor.run_task
    assert supervisor.coordinator.data["status"] == "idle"

    # Verify first run called old endpoint
    assert any("llm.old" in str(call[1]) for call in aioclient_mock.mock_calls)

    # Start second run
    aioclient_mock.mock_calls.clear()
    await hass.services.async_call(DOMAIN, SERVICE_LEARN, blocking=True)
    await supervisor.run_task
    assert supervisor.coordinator.data["status"] == "idle"

    # Second run must call new endpoint
    assert any("llm.new" in str(call[1]) for call in aioclient_mock.mock_calls)


async def test_passive_gate_reevaluation(
    hass: HomeAssistant,
    energy_manager: object,
    std_prefs: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test settings change or refresh re-evaluates gates when idle or blocked."""
    energy_manager.data = std_prefs
    _set_std_powers(hass)
    entry = await async_setup_via_flow(hass)
    supervisor = entry.runtime_data.learning

    # At start, no endpoint -> blocked
    assert supervisor.coordinator.data["status"] == "blocked"
    assert supervisor.coordinator.data["blocked_reason"] == "no_endpoint"

    # Provide endpoint
    await async_seed_llm(hass, entry, base_url="http://llm.test/v1")
    await supervisor.async_settings_changed()
    assert supervisor.coordinator.data["status"] == "blocked"
    assert supervisor.coordinator.data["blocked_reason"] == "no_consent"

    # Provide consent
    await async_seed_consent(hass, host="llm.test", names=False)
    supervisor.invalidate_usable_days_cache()
    monkeypatch.setattr(
        supervisor, "_async_compute_usable_days", lambda: asyncio.sleep(0, result=10.0)
    )
    await supervisor.async_settings_changed()
    # Now all gates pass -> idle
    assert supervisor.coordinator.data["status"] == "idle"
    assert supervisor.coordinator.data["blocked_reason"] is None


async def test_refused_trigger_keeps_last_result(
    hass: HomeAssistant,
    energy_manager: object,
    std_prefs: dict,
    aioclient_mock: object,
    inprocess_worker: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test refused trigger changes only status and blocked_reason, keeping last_run and tokens."""
    energy_manager.data = std_prefs
    _set_std_powers(hass)
    entry = await async_setup_via_flow(hass)
    await async_fill_synth_buffer(hass, entry, spec=SynthSpec(days=22, washer=True))
    await async_seed_llm(hass, entry, base_url="http://llm.test/v1")
    await async_seed_consent(hass, host="llm.test", names=False)
    mock_llm(aioclient_mock, "http://llm.test/v1", proposal_responder(2))
    supervisor = entry.runtime_data.learning

    # Run 1 succeeds
    await hass.services.async_call(DOMAIN, SERVICE_LEARN, blocking=True)
    await supervisor.run_task
    assert supervisor.coordinator.data["status"] == "idle"
    initial_last_run = supervisor.coordinator.data["last_run"]
    initial_tokens_run = supervisor.coordinator.data["tokens_run"]
    initial_month = supervisor.coordinator.data["month"]
    initial_last_error = supervisor.coordinator.data["last_error"]

    # Revoke consent (simulate invalid host)
    await supervisor.consent_store.async_grant("http://other.host/v1", False)

    with pytest.raises(ServiceValidationError) as exc:
        await hass.services.async_call(DOMAIN, SERVICE_LEARN, blocking=True)
    assert exc.value.translation_key == "no_consent"

    c_data = supervisor.coordinator.data
    assert c_data["status"] == "blocked"
    assert c_data["blocked_reason"] == "no_consent"
    assert c_data["last_run"] == initial_last_run
    assert c_data["tokens_run"] == initial_tokens_run
    assert c_data["month"] == initial_month
    assert c_data["last_error"] == initial_last_error

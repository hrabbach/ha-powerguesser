"""Tests for preview action, consent flow, and consent gates."""

from datetime import timedelta
import json
import logging
from typing import Any
import pytest

from homeassistant.auth.models import User
from homeassistant.core import Context, HomeAssistant
from homeassistant.data_entry_flow import FlowResultType
from homeassistant.exceptions import ServiceValidationError, Unauthorized
from homeassistant.helpers import issue_registry as ir
from homeassistant.setup import async_setup_component
import homeassistant.util.dt as dt_util

from pytest_homeassistant.common import async_fire_time_changed

from custom_components.powerguesser import repairs
from custom_components.powerguesser.const import (
    CONF_LLM_BASE_URL,
    CONF_LLM_MODEL,
    CONF_MAX_TOKENS,
    CONF_SHARE_NAMES,
    DEFAULT_MAX_TOKENS,
    DOMAIN,
    ISSUE_LEARNING_CONSENT,
    PREPARE_DEBOUNCE_S,
    PREPARE_STARTUP_DELAY_S,
    SERVICE_PREVIEW,
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


def _set_std_powers(hass: HomeAssistant) -> None:
    async_set_power(hass, "sensor.grid_power", 500)
    async_set_power(hass, "sensor.solar_power", 1200)
    async_set_power(hass, "sensor.battery_power", -300)
    async_set_power(hass, "sensor.washer_power", 200)


async def test_preview_returns_body_without_key(
    hass: HomeAssistant,
    energy_manager: Any,
    std_prefs: dict,
    aioclient_mock: Any,
    inprocess_worker: Any,
) -> None:
    """Test preview action returns expected payload structure with no API key or headers."""
    energy_manager.data = std_prefs
    _set_std_powers(hass)
    entry = await async_setup_via_flow(hass)
    await async_fill_synth_buffer(hass, entry)
    await async_seed_llm(hass, entry, api_key="sk-test-secret")

    response = await hass.services.async_call(
        DOMAIN,
        SERVICE_PREVIEW,
        blocking=True,
        return_response=True,
    )

    assert isinstance(response, dict)
    expected_keys = {
        "url",
        "body",
        "bytes",
        "estimated_input_tokens",
        "max_tokens",
        "entity_names_included",
    }
    assert set(response.keys()) == expected_keys
    assert response["url"] == "http://llm.test/v1/chat/completions"
    assert response["entity_names_included"] is False
    assert response["bytes"] > 0
    assert response["estimated_input_tokens"] > 0
    assert response["max_tokens"] == DEFAULT_MAX_TOKENS

    # Ensure neither API key nor Authorization header is in the serialized response
    serialized = json.dumps(response)
    assert "sk-test-secret" not in serialized
    assert "Authorization" not in serialized

    # Verify no HTTP request was sent to the endpoint
    assert len(aioclient_mock.mock_calls) == 0


async def test_preview_body_equals_first_sent_body(
    hass: HomeAssistant,
    energy_manager: Any,
    std_prefs: dict,
    aioclient_mock: Any,
    inprocess_worker: Any,
) -> None:
    """Test preview body exactly equals the body of the first request sent during a run."""
    energy_manager.data = std_prefs
    _set_std_powers(hass)
    entry = await async_setup_via_flow(hass)
    await async_fill_synth_buffer(hass, entry)
    await async_seed_llm(hass, entry)

    # 1. Obtain preview body
    preview_res = await hass.services.async_call(
        DOMAIN,
        SERVICE_PREVIEW,
        blocking=True,
        return_response=True,
    )
    preview_body = preview_res["body"]

    # 2. Grant consent and run learning
    await async_seed_consent(hass, host="llm.test", names=False)
    mock_llm(aioclient_mock, "http://llm.test/v1", proposal_responder(2))

    await hass.services.async_call(DOMAIN, "learn", blocking=True)
    supervisor = entry.runtime_data.learning
    assert supervisor.run_task is not None
    await supervisor.run_task

    # 3. Assert first sent request body matches preview body
    assert len(aioclient_mock.mock_calls) >= 1
    sent_raw = aioclient_mock.mock_calls[0][2]
    if isinstance(sent_raw, (bytes, bytearray)):
        sent_raw = sent_raw.decode("utf-8")
    if isinstance(sent_raw, str):
        sent_body = json.loads(sent_raw)
    else:
        sent_body = sent_raw

    assert sent_body == preview_body


async def test_preview_uses_configured_max_tokens(
    hass: HomeAssistant,
    energy_manager: Any,
    std_prefs: dict,
    aioclient_mock: Any,
    inprocess_worker: Any,
) -> None:
    """The preview body and the consent estimate use the max_tokens option, as the run does (WR-07)."""
    energy_manager.data = std_prefs
    _set_std_powers(hass)
    entry = await async_setup_via_flow(hass)
    await async_fill_synth_buffer(hass, entry)
    await async_seed_llm(hass, entry)
    hass.config_entries.async_update_entry(
        entry, options={**entry.options, CONF_MAX_TOKENS: 8000}
    )
    await hass.async_block_till_done()

    preview_res = await hass.services.async_call(
        DOMAIN, SERVICE_PREVIEW, blocking=True, return_response=True
    )
    assert preview_res["max_tokens"] == 8000
    assert preview_res["body"]["max_completion_tokens"] == 8000

    await async_seed_consent(hass, host="llm.test", names=False)
    mock_llm(aioclient_mock, "http://llm.test/v1", proposal_responder(2))
    await hass.services.async_call(DOMAIN, "learn", blocking=True)
    supervisor = entry.runtime_data.learning
    await supervisor.run_task

    sent_raw = aioclient_mock.mock_calls[0][2]
    if isinstance(sent_raw, (bytes, bytearray)):
        sent_raw = sent_raw.decode("utf-8")
    sent_body = json.loads(sent_raw) if isinstance(sent_raw, str) else sent_raw
    assert sent_body == preview_res["body"]


async def test_preview_with_names_includes_sanitised_plug_names(
    hass: HomeAssistant,
    energy_manager: Any,
    std_prefs: dict,
    aioclient_mock: Any,
    inprocess_worker: Any,
) -> None:
    """Test preview with entity names opted-in includes sanitized plug names in data block only."""
    energy_manager.data = std_prefs
    _set_std_powers(hass)
    entry = await async_setup_via_flow(hass)
    await async_fill_synth_buffer(hass, entry)
    await async_seed_llm(hass, entry)

    # Enable share_entity_names option and set display name with control chars and formatting symbols
    hass.config_entries.async_update_entry(entry, options={CONF_SHARE_NAMES: True})
    hass.states.async_set(
        "sensor.washer_power",
        200,
        {"friendly_name": "Smart Washer \x00*Test*"},
    )

    response = await hass.services.async_call(
        DOMAIN,
        SERVICE_PREVIEW,
        blocking=True,
        return_response=True,
    )

    assert response["entity_names_included"] is True
    body = response["body"]
    messages = body.get("messages", [])
    system_msg = next((m["content"] for m in messages if m.get("role") == "system"), "")
    user_msg = next((m["content"] for m in messages if m.get("role") == "user"), "")

    # Plug display name should appear inside the user message's <data> block
    assert "Smart Washer Test" in user_msg
    assert "p1: Smart Washer Test" in user_msg
    assert "<names>" in user_msg
    assert "</names>" in user_msg

    # Control chars / symbols stripped
    assert "\x00" not in user_msg
    assert "\t" not in user_msg
    assert "*Test*" not in user_msg

    # System message must NOT contain entity names
    assert "Smart Washer" not in system_msg


async def test_preview_requires_endpoint(
    hass: HomeAssistant,
    energy_manager: Any,
    std_prefs: dict,
) -> None:
    """Test preview without endpoint raises ServiceValidationError with translation_key no_endpoint."""
    energy_manager.data = std_prefs
    _set_std_powers(hass)
    entry = await async_setup_via_flow(hass)

    with pytest.raises(ServiceValidationError) as exc_info:
        await hass.services.async_call(
            DOMAIN,
            SERVICE_PREVIEW,
            blocking=True,
            return_response=True,
        )
    assert exc_info.value.translation_key == "no_endpoint"


async def test_preview_is_admin_only(
    hass: HomeAssistant,
    energy_manager: Any,
    std_prefs: dict,
    hass_read_only_user: User,
) -> None:
    """Test preview action rejects non-admin users with Unauthorized exception."""
    energy_manager.data = std_prefs
    _set_std_powers(hass)
    entry = await async_setup_via_flow(hass)
    await async_seed_llm(hass, entry)

    with pytest.raises(Unauthorized):
        await hass.services.async_call(
            DOMAIN,
            SERVICE_PREVIEW,
            blocking=True,
            return_response=True,
            context=Context(user_id=hass_read_only_user.id),
        )


async def test_consent_prepare_debounced_after_settings_save(
    hass: HomeAssistant,
    energy_manager: Any,
    std_prefs: dict,
    inprocess_worker: Any,
) -> None:
    """Test background prepare runs and fixable learning_consent issue appears after debounce."""
    energy_manager.data = std_prefs
    _set_std_powers(hass)
    entry = await async_setup_via_flow(hass)
    await async_fill_synth_buffer(hass, entry)
    await async_seed_llm(hass, entry, base_url="http://llm.test/v1", model="m1")
    supervisor = entry.runtime_data.learning

    # Before debounce timer fires: no issue exists
    issue_reg = ir.async_get(hass)
    assert issue_reg.async_get_issue(DOMAIN, ISSUE_LEARNING_CONSENT) is None

    # Advance time by PREPARE_DEBOUNCE_S
    async_fire_time_changed(hass, dt_util.utcnow() + timedelta(seconds=PREPARE_DEBOUNCE_S + 2))
    await hass.async_block_till_done()
    if supervisor._prepare_task is not None:
        await supervisor._prepare_task

    issue = issue_reg.async_get_issue(DOMAIN, ISSUE_LEARNING_CONSENT)
    assert issue is not None
    assert issue.is_fixable is True
    assert issue.severity == ir.IssueSeverity.WARNING
    assert issue.data["endpoint"] == "http://llm.test:80"
    assert "host" not in issue.data
    assert issue.data["location"] == "remote"
    assert issue.data["transport"] == "http, unencrypted"
    assert issue.data["model"] == "m1"
    assert issue.data["tokens"].isdigit()
    assert issue.data["bytes"].isdigit()
    assert issue.data["names"] == "not included"
    assert issue.translation_placeholders["endpoint"] == "http://llm.test:80"
    assert issue.translation_placeholders["tokens"] == issue.data["tokens"]


async def test_multiple_settings_changes_debounced_to_one_prepare(
    hass: HomeAssistant,
    energy_manager: Any,
    std_prefs: dict,
    inprocess_worker: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test 3 rapid settings changes within debounce window start exactly one prepare."""
    energy_manager.data = std_prefs
    _set_std_powers(hass)
    entry = await async_setup_via_flow(hass)
    await async_fill_synth_buffer(hass, entry)
    await async_seed_llm(hass, entry)
    supervisor = entry.runtime_data.learning

    call_count = 0
    orig_prepare = supervisor._async_prepare

    async def _counted_prepare(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return await orig_prepare(*args, **kwargs)

    monkeypatch.setattr(supervisor, "_async_prepare", _counted_prepare)

    # 3 rapid settings changes
    await supervisor.async_settings_changed()
    async_fire_time_changed(hass, dt_util.utcnow() + timedelta(seconds=5))
    await supervisor.async_settings_changed()
    async_fire_time_changed(hass, dt_util.utcnow() + timedelta(seconds=10))
    await supervisor.async_settings_changed()

    assert call_count == 0

    # Advance time past debounce timer
    async_fire_time_changed(hass, dt_util.utcnow() + timedelta(seconds=PREPARE_DEBOUNCE_S + 5))
    await hass.async_block_till_done()
    if supervisor._prepare_task is not None:
        await supervisor._prepare_task

    assert call_count == 1


async def test_startup_delay_and_no_prepare_with_valid_consent(
    hass: HomeAssistant,
    energy_manager: Any,
    std_prefs: dict,
    inprocess_worker: Any,
) -> None:
    """Test startup delay behavior and that valid consent avoids prepare at startup and settings change."""
    energy_manager.data = std_prefs
    _set_std_powers(hass)

    # Case A: with valid consent, no prepare starts at setup or on settings change
    entry = await async_setup_via_flow(hass)
    await async_seed_llm(hass, entry)
    await async_seed_consent(hass, host="llm.test", names=False)
    supervisor = entry.runtime_data.learning
    await supervisor._async_check_consent(is_startup=True)

    assert supervisor._unsub_prepare_timer is None
    assert supervisor._prepare_task is None

    async_fire_time_changed(hass, dt_util.utcnow() + timedelta(seconds=PREPARE_STARTUP_DELAY_S + 10))
    await hass.async_block_till_done()
    assert supervisor._prepare_task is None
    assert ir.async_get(hass).async_get_issue(DOMAIN, ISSUE_LEARNING_CONSENT) is None

    await supervisor.async_settings_changed()
    assert supervisor._unsub_prepare_timer is None
    assert supervisor._prepare_task is None

    # Case B: at setup with no consent, no prepare starts before PREPARE_STARTUP_DELAY_S
    # Invalidate consent for llm.test
    await supervisor.consent.async_grant("http://other.host/v1", False)
    t0 = dt_util.utcnow()
    await supervisor._async_check_consent(is_startup=True)
    assert supervisor._prepare_task is None

    # Before startup delay
    async_fire_time_changed(hass, t0 + timedelta(seconds=PREPARE_STARTUP_DELAY_S - 10))
    await hass.async_block_till_done()
    assert supervisor._prepare_task is None

    # After startup delay
    async_fire_time_changed(hass, t0 + timedelta(seconds=PREPARE_STARTUP_DELAY_S + 10))
    await hass.async_block_till_done()
    if supervisor._prepare_task is not None:
        await supervisor._prepare_task
    assert ir.async_get(hass).async_get_issue(DOMAIN, ISSUE_LEARNING_CONSENT) is not None


async def test_consent_prepare_cache_reuse_starts_no_worker(
    hass: HomeAssistant,
    energy_manager: Any,
    std_prefs: dict,
    inprocess_worker: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test second consent check on same day with unchanged key starts no worker."""
    energy_manager.data = std_prefs
    _set_std_powers(hass)
    entry = await async_setup_via_flow(hass)
    await async_fill_synth_buffer(hass, entry)
    await async_seed_llm(hass, entry)
    supervisor = entry.runtime_data.learning

    # First prepare
    await supervisor._async_check_consent(immediate=True)
    if supervisor._prepare_task is not None:
        await supervisor._prepare_task

    stage_calls = 0
    orig_stage = supervisor._async_run_stage

    async def _counted_stage(*args, **kwargs):
        nonlocal stage_calls
        stage_calls += 1
        return await orig_stage(*args, **kwargs)

    monkeypatch.setattr(supervisor, "_async_run_stage", _counted_stage)

    # Second prepare on same day with unchanged key
    await supervisor._async_check_consent(immediate=True)
    if supervisor._prepare_task is not None:
        await supervisor._prepare_task

    # No worker stage spawned on reuse
    assert stage_calls == 0


async def test_refused_learn_starts_prepare_immediately(
    hass: HomeAssistant,
    energy_manager: Any,
    std_prefs: dict,
    inprocess_worker: Any,
) -> None:
    """Test learn refused with no_consent starts prepare at once without waiting for delay."""
    energy_manager.data = std_prefs
    _set_std_powers(hass)
    entry = await async_setup_via_flow(hass)
    await async_fill_synth_buffer(hass, entry)
    await async_seed_llm(hass, entry)
    supervisor = entry.runtime_data.learning

    supervisor._cancel_prepare_timers()
    issue_reg = ir.async_get(hass)
    assert issue_reg.async_get_issue(DOMAIN, ISSUE_LEARNING_CONSENT) is None

    # Learn is refused with no_consent
    with pytest.raises(ServiceValidationError) as exc_info:
        await hass.services.async_call(DOMAIN, "learn", blocking=True)
    assert exc_info.value.translation_key == "no_consent"

    # Immediate prepare task was started
    assert supervisor._prepare_task is not None
    await supervisor._prepare_task

    # Issue appeared immediately after prepare finished
    assert issue_reg.async_get_issue(DOMAIN, ISSUE_LEARNING_CONSENT) is not None


async def test_prepare_insufficient_data_shows_unknown_tokens(
    hass: HomeAssistant,
    energy_manager: Any,
    std_prefs: dict,
    inprocess_worker: Any,
) -> None:
    """Test issue appears with unknown tokens and bytes when data is insufficient for prepare."""
    energy_manager.data = std_prefs
    _set_std_powers(hass)
    entry = await async_setup_via_flow(hass)
    # Zero history in buffer
    await async_seed_llm(hass, entry)
    supervisor = entry.runtime_data.learning

    await supervisor._async_check_consent(immediate=True)
    if supervisor._prepare_task is not None:
        await supervisor._prepare_task

    issue = ir.async_get(hass).async_get_issue(DOMAIN, ISSUE_LEARNING_CONSENT)
    assert issue is not None
    assert issue.data["tokens"] == "unknown"
    assert issue.data["bytes"] == "unknown"
    assert issue.translation_placeholders["tokens"] == "unknown"
    assert issue.translation_placeholders["bytes"] == "unknown"


async def test_open_issue_blocks_learning_zero_http_calls(
    hass: HomeAssistant,
    energy_manager: Any,
    std_prefs: dict,
    aioclient_mock: Any,
    inprocess_worker: Any,
) -> None:
    """Test that while issue is open, learn raises no_consent and sends zero HTTP requests."""
    energy_manager.data = std_prefs
    _set_std_powers(hass)
    entry = await async_setup_via_flow(hass)
    await async_fill_synth_buffer(hass, entry)
    await async_seed_llm(hass, entry)
    supervisor = entry.runtime_data.learning

    await supervisor._async_check_consent(immediate=True)
    if supervisor._prepare_task is not None:
        await supervisor._prepare_task

    assert ir.async_get(hass).async_get_issue(DOMAIN, ISSUE_LEARNING_CONSENT) is not None

    with pytest.raises(ServiceValidationError) as exc_info:
        await hass.services.async_call(DOMAIN, "learn", blocking=True)
    assert exc_info.value.translation_key == "no_consent"
    assert len(aioclient_mock.mock_calls) == 0


async def test_confirm_fix_flow_grants_consent_and_removes_issue(
    hass: HomeAssistant,
    energy_manager: Any,
    std_prefs: dict,
    aioclient_mock: Any,
    inprocess_worker: Any,
) -> None:
    """Test confirming fix flow stores consent record, deletes issue, and allows learning."""
    energy_manager.data = std_prefs
    _set_std_powers(hass)
    entry = await async_setup_via_flow(hass)
    await async_fill_synth_buffer(hass, entry)
    await async_seed_llm(hass, entry)
    supervisor = entry.runtime_data.learning

    await supervisor._async_check_consent(immediate=True)
    if supervisor._prepare_task is not None:
        await supervisor._prepare_task

    issue = ir.async_get(hass).async_get_issue(DOMAIN, ISSUE_LEARNING_CONSENT)
    assert issue is not None

    flow = await repairs.async_create_fix_flow(hass, ISSUE_LEARNING_CONSENT, issue.data)
    assert isinstance(flow, repairs.ConsentRepairFlow)
    res_init = await flow.async_step_init()
    assert res_init["type"].name == "FORM"
    assert res_init["step_id"] == "confirm"

    res_confirm = await flow.async_step_confirm({})
    assert res_confirm["type"].name == "CREATE_ENTRY"

    # Consent record stored exactly
    record = supervisor.consent.record
    assert record is not None
    assert record["scheme"] == "http"
    assert record["host"] == "llm.test"
    assert record["port"] == 80
    assert record["entity_names_flag"] is False
    assert "timestamp" in record

    # Issue deleted
    assert ir.async_get(hass).async_get_issue(DOMAIN, ISSUE_LEARNING_CONSENT) is None

    # Learn now runs
    mock_llm(aioclient_mock, "http://llm.test/v1", proposal_responder(2))
    await hass.services.async_call(DOMAIN, "learn", blocking=True)
    assert supervisor.run_task is not None
    await supervisor.run_task
    assert supervisor.coordinator.data["status"] == "idle"


@pytest.mark.parametrize(
    "start_kwargs",
    [
        # Frontend path: POST /api/repairs/issues/fix -> context only, data=None
        pytest.param({"context": {"issue_id": ISSUE_LEARNING_CONSENT}}, id="context"),
        # Legacy custom-integration path: issue_id carried in data only
        pytest.param({"data": {"issue_id": ISSUE_LEARNING_CONSENT}}, id="data"),
    ],
)
async def test_fix_flow_via_repairs_manager_shows_notice_before_granting(
    hass: HomeAssistant,
    energy_manager: Any,
    std_prefs: dict,
    inprocess_worker: Any,
    start_kwargs: dict[str, Any],
) -> None:
    """Starting the fix flow through HA's repairs flow manager shows the notice; only Submit grants.

    The manager injects {"issue_id": ...} into the init step's user_input, so the init step
    must never treat its input as the user's confirmation.
    """
    energy_manager.data = std_prefs
    _set_std_powers(hass)
    entry = await async_setup_via_flow(hass)
    await async_fill_synth_buffer(hass, entry)
    await async_seed_llm(hass, entry)
    supervisor = entry.runtime_data.learning
    assert await async_setup_component(hass, "repairs", {})

    await supervisor._async_check_consent(immediate=True)
    if supervisor._prepare_task is not None:
        await supervisor._prepare_task
    issue = ir.async_get(hass).async_get_issue(DOMAIN, ISSUE_LEARNING_CONSENT)
    assert issue is not None
    assert supervisor.consent.record is None

    flow_manager = hass.data["repairs"]["flow_manager"]
    result = await flow_manager.async_init(DOMAIN, **start_kwargs)

    # The notice is shown, nothing is granted yet, the issue stays open
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "confirm"
    expected_placeholders = {k: v for k, v in issue.data.items() if k != "entry_id"}
    assert result["description_placeholders"] == expected_placeholders
    assert result["description_placeholders"]["endpoint"] == "http://llm.test:80"
    assert "names" in result["description_placeholders"]
    assert supervisor.consent.record is None
    assert ir.async_get(hass).async_get_issue(DOMAIN, ISSUE_LEARNING_CONSENT) is not None

    # Submitting the (empty) confirm form grants exactly the shown origin
    result2 = await flow_manager.async_configure(result["flow_id"], {})
    assert result2["type"] is FlowResultType.CREATE_ENTRY
    record = supervisor.consent.record
    assert record is not None
    assert (record["scheme"], record["host"], record["port"]) == ("http", "llm.test", 80)
    assert record["entity_names_flag"] is False
    assert ir.async_get(hass).async_get_issue(DOMAIN, ISSUE_LEARNING_CONSENT) is None


async def test_fix_flow_aborts_if_endpoint_changed(
    hass: HomeAssistant,
    energy_manager: Any,
    std_prefs: dict,
    inprocess_worker: Any,
) -> None:
    """Test fix flow aborts with endpoint_changed if entry host or names changed meanwhile."""
    energy_manager.data = std_prefs
    _set_std_powers(hass)
    entry = await async_setup_via_flow(hass)
    await async_fill_synth_buffer(hass, entry)
    await async_seed_llm(hass, entry)
    supervisor = entry.runtime_data.learning

    await supervisor._async_check_consent(immediate=True)
    if supervisor._prepare_task is not None:
        await supervisor._prepare_task

    issue = ir.async_get(hass).async_get_issue(DOMAIN, ISSUE_LEARNING_CONSENT)
    assert issue is not None

    # Endpoint changes before user submits fix flow
    data = dict(entry.data)
    data[CONF_LLM_BASE_URL] = "http://different.test/v1"
    hass.config_entries.async_update_entry(entry, data=data)

    flow = await repairs.async_create_fix_flow(hass, ISSUE_LEARNING_CONSENT, issue.data)
    res_confirm = await flow.async_step_confirm({})
    assert res_confirm["type"].name == "ABORT"
    assert res_confirm["reason"] == "endpoint_changed"

    if supervisor._prepare_task is not None:
        await supervisor._prepare_task

    # Recreated issue for current endpoint
    new_issue = ir.async_get(hass).async_get_issue(DOMAIN, ISSUE_LEARNING_CONSENT)
    assert new_issue is not None
    assert new_issue.data["endpoint"] == "http://different.test:80"


async def test_host_change_recreates_issue_model_change_does_not(
    hass: HomeAssistant,
    energy_manager: Any,
    std_prefs: dict,
    inprocess_worker: Any,
) -> None:
    """Test host change requires consent again, while model change alone does not."""
    energy_manager.data = std_prefs
    _set_std_powers(hass)
    entry = await async_setup_via_flow(hass)
    await async_fill_synth_buffer(hass, entry)
    await async_seed_llm(hass, entry, base_url="http://llm.test/v1", model="m1")
    await async_seed_consent(hass, host="llm.test", names=False)
    supervisor = entry.runtime_data.learning

    await supervisor.async_settings_changed()
    assert ir.async_get(hass).async_get_issue(DOMAIN, ISSUE_LEARNING_CONSENT) is None

    # Model change alone does NOT recreate issue
    data = dict(entry.data)
    data[CONF_LLM_MODEL] = "m2"
    hass.config_entries.async_update_entry(entry, data=data)
    await supervisor.async_settings_changed()
    assert ir.async_get(hass).async_get_issue(DOMAIN, ISSUE_LEARNING_CONSENT) is None

    # Host change DOES recreate issue and blocks learning
    data[CONF_LLM_BASE_URL] = "http://newhost.test/v1"
    hass.config_entries.async_update_entry(entry, data=data)
    await supervisor.async_settings_changed()

    async_fire_time_changed(hass, dt_util.utcnow() + timedelta(seconds=PREPARE_DEBOUNCE_S + 5))
    await hass.async_block_till_done()
    if supervisor._prepare_task is not None:
        await supervisor._prepare_task

    issue = ir.async_get(hass).async_get_issue(DOMAIN, ISSUE_LEARNING_CONSENT)
    assert issue is not None
    assert issue.data["endpoint"] == "http://newhost.test:80"

    with pytest.raises(ServiceValidationError) as exc:
        await hass.services.async_call(DOMAIN, "learn", blocking=True)
    assert exc.value.translation_key == "no_consent"


async def test_share_names_option_toggles_consent_issue(
    hass: HomeAssistant,
    energy_manager: Any,
    std_prefs: dict,
    inprocess_worker: Any,
) -> None:
    """Test changing share_entity_names invalidates consent and asks again."""
    energy_manager.data = std_prefs
    _set_std_powers(hass)
    entry = await async_setup_via_flow(hass)
    await async_fill_synth_buffer(hass, entry)
    await async_seed_llm(hass, entry)
    await async_seed_consent(hass, host="llm.test", names=False)
    supervisor = entry.runtime_data.learning

    await supervisor.async_settings_changed()
    assert ir.async_get(hass).async_get_issue(DOMAIN, ISSUE_LEARNING_CONSENT) is None

    # Turning names ON recreates issue with names included
    hass.config_entries.async_update_entry(entry, options={CONF_SHARE_NAMES: True})
    await supervisor.async_settings_changed()
    async_fire_time_changed(hass, dt_util.utcnow() + timedelta(seconds=PREPARE_DEBOUNCE_S + 5))
    await hass.async_block_till_done()
    if supervisor._prepare_task is not None:
        await supervisor._prepare_task

    issue = ir.async_get(hass).async_get_issue(DOMAIN, ISSUE_LEARNING_CONSENT)
    assert issue is not None
    assert issue.data["names"] == "included"

    # Grant consent with names ON
    await supervisor.consent.async_grant("http://llm.test/v1", True)
    await supervisor.async_settings_changed()
    assert ir.async_get(hass).async_get_issue(DOMAIN, ISSUE_LEARNING_CONSENT) is None

    # Turning names OFF recreates issue with names not included
    hass.config_entries.async_update_entry(entry, options={CONF_SHARE_NAMES: False})
    await supervisor.async_settings_changed()
    async_fire_time_changed(hass, dt_util.utcnow() + timedelta(seconds=PREPARE_DEBOUNCE_S + 5))
    await hass.async_block_till_done()
    if supervisor._prepare_task is not None:
        await supervisor._prepare_task

    issue = ir.async_get(hass).async_get_issue(DOMAIN, ISSUE_LEARNING_CONSENT)
    assert issue is not None
    assert issue.data["names"] == "not included"


async def test_clearing_endpoint_deletes_issue(
    hass: HomeAssistant,
    energy_manager: Any,
    std_prefs: dict,
    inprocess_worker: Any,
) -> None:
    """Test clearing base URL deletes learning_consent issue."""
    energy_manager.data = std_prefs
    _set_std_powers(hass)
    entry = await async_setup_via_flow(hass)
    await async_fill_synth_buffer(hass, entry)
    await async_seed_llm(hass, entry)
    supervisor = entry.runtime_data.learning

    await supervisor._async_check_consent(immediate=True)
    if supervisor._prepare_task is not None:
        await supervisor._prepare_task

    assert ir.async_get(hass).async_get_issue(DOMAIN, ISSUE_LEARNING_CONSENT) is not None

    # Clear base URL
    data = dict(entry.data)
    data[CONF_LLM_BASE_URL] = ""
    hass.config_entries.async_update_entry(entry, data=data)
    await supervisor.async_settings_changed()

    assert ir.async_get(hass).async_get_issue(DOMAIN, ISSUE_LEARNING_CONSENT) is None


async def test_local_and_https_location_placeholders(
    hass: HomeAssistant,
    energy_manager: Any,
    std_prefs: dict,
    inprocess_worker: Any,
) -> None:
    """Test local network and transport detection in issue data."""
    energy_manager.data = std_prefs
    _set_std_powers(hass)
    entry = await async_setup_via_flow(hass)
    await async_fill_synth_buffer(hass, entry)
    supervisor = entry.runtime_data.learning

    # Test .local with https
    await async_seed_llm(hass, entry, base_url="https://llm-box.local/v1")
    await supervisor._async_check_consent(immediate=True)
    if supervisor._prepare_task is not None:
        await supervisor._prepare_task

    issue = ir.async_get(hass).async_get_issue(DOMAIN, ISSUE_LEARNING_CONSENT)
    assert issue is not None
    assert issue.data["location"] == "local network"
    assert issue.data["transport"] == "https"

    # Test 127.0.0.1 with http
    await async_seed_llm(hass, entry, base_url="http://127.0.0.1:8000/v1")
    await supervisor._async_check_consent(immediate=True)
    if supervisor._prepare_task is not None:
        await supervisor._prepare_task

    issue = ir.async_get(hass).async_get_issue(DOMAIN, ISSUE_LEARNING_CONSENT)
    assert issue is not None
    assert issue.data["location"] == "local network"
    assert issue.data["transport"] == "http, unencrypted"



async def test_concurrent_prepares_run_one_extract(
    hass: HomeAssistant,
    energy_manager: Any,
    std_prefs: dict,
    inprocess_worker: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two prepares at once (consent job + preview action) share one extract, never race (WR-08)."""
    import asyncio

    energy_manager.data = std_prefs
    _set_std_powers(hass)
    entry = await async_setup_via_flow(hass)
    await async_fill_synth_buffer(hass, entry)
    await async_seed_llm(hass, entry)
    supervisor = entry.runtime_data.learning

    stages: list[str] = []
    inner = supervisor._async_run_stage

    async def _counted(stage: str, job_path: Any) -> dict[str, Any]:
        stages.append(stage)
        await asyncio.sleep(0.05)
        return await inner(stage, job_path)

    monkeypatch.setattr(supervisor, "_async_run_stage", _counted)

    (d1, m1), (d2, m2) = await asyncio.gather(
        supervisor._async_prepare(), supervisor._async_prepare()
    )
    assert stages == ["extract"]
    assert d1 == d2
    assert m1 == m2


async def test_consent_keyed_on_scheme_host_port(
    hass: HomeAssistant,
    energy_manager: Any,
    std_prefs: dict,
    inprocess_worker: Any,
) -> None:
    """Consent covers one scheme + host + port (default ports normalised); the model is not part of it (WR-11)."""
    from custom_components.powerguesser.consent import consent_valid

    energy_manager.data = std_prefs
    _set_std_powers(hass)
    entry = await async_setup_via_flow(hass)
    await async_fill_synth_buffer(hass, entry)
    await async_seed_llm(hass, entry, base_url="https://llm.test/v1", model="m1")
    supervisor = entry.runtime_data.learning

    await supervisor.consent.async_grant("https://llm.test/v1", False)
    record = supervisor.consent.record
    assert record is not None
    assert (record["scheme"], record["host"], record["port"]) == ("https", "llm.test", 443)

    assert consent_valid(record, "https://llm.test/v1", False)
    assert consent_valid(record, "https://llm.test:443/v1", False)
    assert consent_valid(record, "https://LLM.test/v1", False)
    assert not consent_valid(record, "http://llm.test/v1", False)
    assert not consent_valid(record, "http://llm.test:443/v1", False)
    assert not consent_valid(record, "https://llm.test:8443/v1", False)
    assert not consent_valid(record, "https://other.test/v1", False)
    assert not consent_valid(record, "https://llm.test/v1", True)

    # http defaults to port 80 as well
    await supervisor.consent.async_grant("http://llm.test/v1", False)
    assert supervisor.consent.record["port"] == 80
    assert consent_valid(supervisor.consent.record, "http://llm.test:80/v1", False)
    await supervisor.consent.async_grant("https://llm.test/v1", False)

    # The gate follows the entry: same origin, other model -> still consented (D-11)
    assert supervisor._evaluate_endpoint_and_consent() is None
    data = dict(entry.data)
    data[CONF_LLM_MODEL] = "m2"
    hass.config_entries.async_update_entry(entry, data=data)
    assert supervisor._evaluate_endpoint_and_consent() is None

    # Transport downgrade on the same host -> asks again, and the notice names the origin
    data[CONF_LLM_BASE_URL] = "http://llm.test/v1"
    hass.config_entries.async_update_entry(entry, data=data)
    assert supervisor._evaluate_endpoint_and_consent() == "no_consent"
    await supervisor._async_check_consent(immediate=True)
    if supervisor._prepare_task is not None:
        await supervisor._prepare_task
    issue = ir.async_get(hass).async_get_issue(DOMAIN, ISSUE_LEARNING_CONSENT)
    assert issue is not None
    assert issue.translation_placeholders["endpoint"] == "http://llm.test:80"

    # Other port on the same host and scheme -> asks again
    data[CONF_LLM_BASE_URL] = "https://llm.test:8443/v1"
    hass.config_entries.async_update_entry(entry, data=data)
    assert supervisor._evaluate_endpoint_and_consent() == "no_consent"


async def test_fix_flow_grants_origin_and_aborts_on_port_change(
    hass: HomeAssistant,
    energy_manager: Any,
    std_prefs: dict,
    inprocess_worker: Any,
) -> None:
    """The fix flow grants the shown scheme://host:port and aborts if only the port changed (WR-11)."""
    energy_manager.data = std_prefs
    _set_std_powers(hass)
    entry = await async_setup_via_flow(hass)
    await async_fill_synth_buffer(hass, entry)
    await async_seed_llm(hass, entry, base_url="http://llm.test:4000/v1")
    supervisor = entry.runtime_data.learning

    await supervisor._async_check_consent(immediate=True)
    if supervisor._prepare_task is not None:
        await supervisor._prepare_task
    issue = ir.async_get(hass).async_get_issue(DOMAIN, ISSUE_LEARNING_CONSENT)
    assert issue is not None
    assert issue.data["endpoint"] == "http://llm.test:4000"

    # Port changes before the user confirms -> abort, nothing granted
    data = dict(entry.data)
    data[CONF_LLM_BASE_URL] = "http://llm.test:5000/v1"
    hass.config_entries.async_update_entry(entry, data=data)
    flow = await repairs.async_create_fix_flow(hass, ISSUE_LEARNING_CONSENT, issue.data)
    res = await flow.async_step_confirm({})
    assert res["type"].name == "ABORT"
    assert res["reason"] == "endpoint_changed"
    assert supervisor.consent.record is None
    if supervisor._prepare_task is not None:
        await supervisor._prepare_task

    issue = ir.async_get(hass).async_get_issue(DOMAIN, ISSUE_LEARNING_CONSENT)
    flow = await repairs.async_create_fix_flow(hass, ISSUE_LEARNING_CONSENT, issue.data)
    res = await flow.async_step_confirm({})
    assert res["type"].name == "CREATE_ENTRY"
    record = supervisor.consent.record
    assert (record["scheme"], record["host"], record["port"]) == ("http", "llm.test", 5000)


async def test_legacy_host_only_consent_record_is_invalid(
    hass: HomeAssistant,
    hass_storage: dict[str, Any],
) -> None:
    """A stored host-only record (pre WR-11 shape) counts as no consent; no migration (WR-11)."""
    from custom_components.powerguesser.consent import ConsentStore, consent_valid
    from custom_components.powerguesser.const import CONSENT_STORE_KEY, LEARNING_STORES_VERSION

    hass_storage[CONSENT_STORE_KEY] = {
        "version": LEARNING_STORES_VERSION,
        "minor_version": 1,
        "key": CONSENT_STORE_KEY,
        "data": {
            "host": "llm.test",
            "timestamp": "2026-11-16T10:00:00+00:00",
            "entity_names_flag": False,
        },
    }
    store = ConsentStore(hass)
    assert await store.async_load() is None
    assert store.record is None
    assert not consent_valid(
        {"host": "llm.test", "timestamp": "x", "entity_names_flag": False},
        "http://llm.test/v1",
        False,
    )

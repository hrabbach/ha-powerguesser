"""Tests for learning loop repair, salvage, failure classification, and auth repairs."""

import asyncio
import json
import logging
from typing import Any
import pytest

from homeassistant.components.repairs import FlowType
from homeassistant.core import HomeAssistant
from homeassistant.helpers import issue_registry as ir

from custom_components.powerguesser import repairs
from custom_components.powerguesser.const import (
    DOMAIN,
    ISSUE_LLM_AUTH,
)
from custom_components.powerguesser.learning import LearningStageError
from tests.conftest import (
    async_fill_synth_buffer,
    async_seed_consent,
    async_seed_llm,
    async_set_power,
    async_setup_via_flow,
    mock_llm,
    switched_cluster_ids,
)
from tests.fixtures.learning.fake_llm import chat_completion


def _set_std_powers(hass: HomeAssistant) -> None:
    async_set_power(hass, "sensor.grid_power", 500)
    async_set_power(hass, "sensor.solar_power", 1200)
    async_set_power(hass, "sensor.battery_power", -300)
    async_set_power(hass, "sensor.washer_power", 200)


async def _setup_ready_supervisor(
    hass: HomeAssistant,
    energy_manager: object,
    std_prefs: dict,
    days: int = 22,
    spec: Any = None,
) -> Any:
    from tests.fixtures.learning.synth import SynthSpec

    energy_manager.data = std_prefs
    _set_std_powers(hass)
    entry = await async_setup_via_flow(hass)
    if spec is None:
        spec = SynthSpec(days=days, oven=True)
    await async_fill_synth_buffer(hass, entry, spec=spec)
    await async_seed_llm(hass, entry)
    await async_seed_consent(hass, host="llm.test", names=False)
    return entry


def _make_proposal(devices: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "analysis": "Test proposal.",
        "schema_version": 1,
        "devices": devices,
        "rejected_clusters": [],
        "merge_suggestions": [],
    }


def _make_device(ref: str, cluster: str) -> dict[str, Any]:
    return {
        "ref": ref,
        "label": f"Device {ref}",
        "category": "unknown",
        "tracker_kind": "switched",
        "member_clusters": [cluster],
        "states": [
            {
                "name": "on",
                "cluster_ref": cluster,
                "role": "main",
            }
        ],
        "phase_hint": "single",
        "confidence": "medium",
        "rationale": f"Device for {cluster}.",
        "ambiguous_with": [],
        "reference_plug": "",
    }


async def test_invalid_reply_is_repaired(
    hass: HomeAssistant,
    energy_manager: object,
    std_prefs: dict,
    aioclient_mock: object,
    inprocess_worker: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An invalid reply triggers a repair request without digest lines, and candidate is stored."""
    monkeypatch.setattr("custom_components.powerguesser.learning.MAX_REFINE_ROUNDS", 0)
    entry = await _setup_ready_supervisor(hass, energy_manager, std_prefs)
    call_count = 0

    def _responder(request_data: Any) -> dict[str, Any]:
        nonlocal call_count
        call_count += 1
        if isinstance(request_data, (bytes, bytearray)):
            req = json.loads(request_data.decode("utf-8"))
        elif isinstance(request_data, str):
            req = json.loads(request_data)
        else:
            req = request_data
        print(f"CALL {call_count} snippet:", req["messages"][-1]["content"][:150])
        if call_count == 1:
            # Turn 1: unknown cluster c99
            prop = _make_proposal([_make_device("d1", "c99")])
            return {"status": 200, "json": chat_completion(json.dumps(prop))}
        # Turn 2: valid cluster c01
        prop = _make_proposal([_make_device("d1", "c01")])
        return {"status": 200, "json": chat_completion(json.dumps(prop))}

    mock_llm(aioclient_mock, "http://llm.test/v1", _responder)

    supervisor = entry.runtime_data.learning
    await supervisor.async_request_run("service")
    assert supervisor.run_task is not None
    await supervisor.run_task

    assert call_count == 2
    data = supervisor.coordinator.data
    assert data["status"] == "idle"
    assert data["last_error"] is None
    assert len(supervisor.candidates) >= 1

    # Check that second request body has no digest lines (e.g. no "c01: ") and lists valid ids
    second_request = aioclient_mock.mock_calls[1][2]
    if isinstance(second_request, (bytes, bytearray)):
        second_request = json.loads(second_request.decode("utf-8"))
    user_msg = second_request["messages"][1]["content"]
    assert "Valid cluster ids:" in user_msg
    assert "c01" in user_msg
    assert "c99" in user_msg  # In the error message / previous output
    assert "c01: " not in user_msg  # No digest line format


async def test_over_length_text_is_trimmed_without_repair(
    hass: HomeAssistant,
    energy_manager: object,
    std_prefs: dict,
    aioclient_mock: object,
    inprocess_worker: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Over-long free text is trimmed locally and counted; no repair request is sent."""
    monkeypatch.setattr("custom_components.powerguesser.learning.MAX_REFINE_ROUNDS", 0)
    entry = await _setup_ready_supervisor(hass, energy_manager, std_prefs)
    call_count = 0

    def _responder(request_data: Any) -> dict[str, Any]:
        nonlocal call_count
        call_count += 1
        dev = _make_device("d1", "c01")
        dev["label"] = "Kitchen oven with an unusually long label text"  # 47
        dev["rationale"] = "r" * 350
        prop = _make_proposal([dev])
        prop["analysis"] = "a" * 1300
        return {"status": 200, "json": chat_completion(json.dumps(prop))}

    mock_llm(aioclient_mock, "http://llm.test/v1", _responder)

    supervisor = entry.runtime_data.learning
    await supervisor.async_request_run("service")
    assert supervisor.run_task is not None
    await supervisor.run_task

    assert call_count == 1
    data = supervisor.coordinator.data
    assert data["status"] == "idle"
    assert data["last_error"] is None
    assert data["trimmed_run"] == 3
    assert data["last_run"]["trimmed_fields"] == 3
    assert len(supervisor.candidates) == 1
    label = supervisor.candidates[0]["label"]
    assert len(label) <= 40
    assert label.endswith("…")

    state = hass.states.get("sensor.powerguesser_learning_status")
    assert state.attributes["trimmed_this_run"] == 3
    assert state.attributes["last_result"]["trimmed_fields"] == 3


async def test_two_failed_repairs_salvage_valid_devices(
    hass: HomeAssistant,
    energy_manager: object,
    std_prefs: dict,
    aioclient_mock: object,
    inprocess_worker: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two failed repair turns salvage surviving valid devices."""
    monkeypatch.setattr("custom_components.powerguesser.learning.MAX_REFINE_ROUNDS", 0)
    entry = await _setup_ready_supervisor(hass, energy_manager, std_prefs)
    call_count = 0

    def _responder(request_data: Any) -> dict[str, Any]:
        nonlocal call_count
        call_count += 1
        # In all turns, return one valid device (c01) and one invalid device (c99)
        prop = _make_proposal([_make_device("d1", "c01"), _make_device("d2", "c99")])
        return {"status": 200, "json": chat_completion(json.dumps(prop))}

    mock_llm(aioclient_mock, "http://llm.test/v1", _responder)

    supervisor = entry.runtime_data.learning
    await supervisor.async_request_run("service")
    assert supervisor.run_task is not None
    await supervisor.run_task

    # 1 initial + 2 repairs = 3 calls
    assert call_count == 3
    data = supervisor.coordinator.data
    assert data["status"] == "idle"
    assert data["last_error"] is None
    # d1 survived salvage, d2 was dropped
    assert len(supervisor.candidates) == 1
    assert supervisor.candidates[0]["label"] == "Device d1"


async def test_salvage_uses_errors_of_the_salvaged_object(
    hass: HomeAssistant,
    energy_manager: object,
    std_prefs: dict,
    aioclient_mock: object,
    inprocess_worker: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A last turn without JSON does not hide the earlier object's device errors from salvage (WR-06)."""
    monkeypatch.setattr("custom_components.powerguesser.learning.MAX_REFINE_ROUNDS", 0)
    entry = await _setup_ready_supervisor(hass, energy_manager, std_prefs)
    call_count = 0

    def _responder(request_data: Any) -> dict[str, Any]:
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            sw = switched_cluster_ids(request_data) or ["c01"]
            prop = _make_proposal([_make_device("d1", sw[0]), _make_device("d2", "c99")])
            return {"status": 200, "json": chat_completion(json.dumps(prop))}
        return {"status": 200, "json": chat_completion("Sorry, I cannot produce JSON now.")}

    mock_llm(aioclient_mock, "http://llm.test/v1", _responder)

    supervisor = entry.runtime_data.learning
    await supervisor.async_request_run("service")
    assert supervisor.run_task is not None
    await supervisor.run_task

    assert call_count == 3
    data = supervisor.coordinator.data
    assert data["status"] == "idle", data["last_error"]
    assert len(supervisor.candidates) == 1
    assert supervisor.candidates[0]["label"] == "Device d1"


async def test_two_failed_repairs_without_valid_device_fail_malformed(
    hass: HomeAssistant,
    energy_manager: object,
    std_prefs: dict,
    aioclient_mock: object,
    inprocess_worker: object,
) -> None:
    """Two failed repair turns with no valid device fail with malformed_output."""
    entry = await _setup_ready_supervisor(hass, energy_manager, std_prefs)
    call_count = 0

    def _responder(request_data: Any) -> dict[str, Any]:
        nonlocal call_count
        call_count += 1
        prop = _make_proposal([_make_device("d1", "c99")])
        return {"status": 200, "json": chat_completion(json.dumps(prop))}

    mock_llm(aioclient_mock, "http://llm.test/v1", _responder)

    supervisor = entry.runtime_data.learning
    await supervisor.async_request_run("service")
    assert supervisor.run_task is not None
    await supervisor.run_task

    assert call_count == 3
    data = supervisor.coordinator.data
    assert data["status"] == "failed"
    assert data["last_error"] == "malformed_output"


async def test_length_fails_without_repair(
    hass: HomeAssistant,
    energy_manager: object,
    std_prefs: dict,
    aioclient_mock: object,
    inprocess_worker: object,
) -> None:
    """finish_reason length fails immediately with output_truncated."""
    entry = await _setup_ready_supervisor(hass, energy_manager, std_prefs)
    call_count = 0

    def _responder(request_data: Any) -> dict[str, Any]:
        nonlocal call_count
        call_count += 1
        return {"status": 200, "json": chat_completion('{"analysis": "truncated', finish_reason="length")}

    mock_llm(aioclient_mock, "http://llm.test/v1", _responder)

    supervisor = entry.runtime_data.learning
    await supervisor.async_request_run("service")
    assert supervisor.run_task is not None
    await supervisor.run_task

    assert call_count == 1
    data = supervisor.coordinator.data
    assert data["status"] == "failed"
    assert data["last_error"] == "output_truncated"


async def test_refusal_and_empty_output_fail_without_repair(
    hass: HomeAssistant,
    energy_manager: object,
    std_prefs: dict,
    aioclient_mock: object,
    inprocess_worker: object,
) -> None:
    """refusal and empty output both fail without sending repair requests."""
    entry = await _setup_ready_supervisor(hass, energy_manager, std_prefs)

    # 1. Refusal
    call_count = 0

    def _responder_refusal(request_data: Any) -> dict[str, Any]:
        nonlocal call_count
        call_count += 1
        return {"status": 200, "json": chat_completion(refusal="I cannot perform this task")}

    mock_llm(aioclient_mock, "http://llm.test/v1", _responder_refusal)

    supervisor = entry.runtime_data.learning
    await supervisor.async_request_run("service")
    assert supervisor.run_task is not None
    await supervisor.run_task

    assert call_count == 1
    data = supervisor.coordinator.data
    assert data["status"] == "failed"
    assert data["last_error"] == "refused"

    # 2. Empty output
    call_count = 0
    aioclient_mock.clear_requests()

    def _responder_empty(request_data: Any) -> dict[str, Any]:
        nonlocal call_count
        call_count += 1
        return {"status": 200, "json": chat_completion("")}

    mock_llm(aioclient_mock, "http://llm.test/v1", _responder_empty)

    await supervisor.async_request_run("service")
    assert supervisor.run_task is not None
    await supervisor.run_task

    assert call_count == 1
    data = supervisor.coordinator.data
    assert data["status"] == "failed"
    assert data["last_error"] == "empty_output"


async def test_empty_devices_finishes_idle_with_zero_candidates(
    hass: HomeAssistant,
    energy_manager: object,
    std_prefs: dict,
    aioclient_mock: object,
    inprocess_worker: object,
) -> None:
    """Empty devices array finishes idle with 0 candidates."""
    entry = await _setup_ready_supervisor(hass, energy_manager, std_prefs)
    call_count = 0

    def _responder(request_data: Any) -> dict[str, Any]:
        nonlocal call_count
        call_count += 1
        prop = _make_proposal([])
        return {"status": 200, "json": chat_completion(json.dumps(prop))}

    mock_llm(aioclient_mock, "http://llm.test/v1", _responder)

    supervisor = entry.runtime_data.learning
    await supervisor.async_request_run("service")
    assert supervisor.run_task is not None
    await supervisor.run_task

    assert call_count == 1
    data = supervisor.coordinator.data
    assert data["status"] == "idle"
    assert data["last_error"] is None
    assert data["pending_candidates"] == 0
    assert len(supervisor.candidates) == 0


async def test_auth_failure_raises_fixable_issue_and_keeps_power_sensors(
    hass: HomeAssistant,
    energy_manager: object,
    std_prefs: dict,
    aioclient_mock: object,
    inprocess_worker: object,
) -> None:
    """HTTP 401 raises fixable llm_auth Repairs issue and keeps power sensors running."""
    entry = await _setup_ready_supervisor(hass, energy_manager, std_prefs)

    def _responder(request_data: Any) -> dict[str, Any]:
        return {"status": 401, "text": "Unauthorized key"}

    mock_llm(aioclient_mock, "http://llm.test/v1", _responder)

    supervisor = entry.runtime_data.learning
    await supervisor.async_request_run("service")
    assert supervisor.run_task is not None
    await supervisor.run_task

    data = supervisor.coordinator.data
    assert data["status"] == "failed"
    assert "auth_failed (HTTP 401)" in data["last_error"]

    # The one failure class with a Repairs side effect carries the same
    # last_attempt guarantee (review round 1)
    assert data["last_attempt"]["outcome"] == "failed"
    assert data["last_attempt"]["started"] is not None
    assert data["last_attempt"]["ended"] is not None
    assert data["last_run"] is None

    # Gross sensor still exists and is available
    gross_state = hass.states.get("sensor.powerguesser_gross_power")
    assert gross_state is not None
    assert gross_state.state != "unavailable"

    # Repairs issue created and fixable
    issue_reg = ir.async_get(hass)
    issue = issue_reg.async_get_issue(DOMAIN, ISSUE_LLM_AUTH)
    assert issue is not None
    assert issue.is_fixable is True
    assert issue.translation_placeholders == {"host": "llm.test"}

    # Fix flow opens options
    flow = await repairs.async_create_fix_flow(hass, ISSUE_LLM_AUTH, issue.data)
    assert isinstance(flow, repairs.LlmAuthRepairFlow)
    res_init = await flow.async_step_init()
    assert res_init["type"].name == "FORM"
    assert res_init["step_id"] == "confirm"

    res_confirm = await flow.async_step_confirm({})
    assert res_confirm["type"].name == "ABORT"
    assert res_confirm["reason"] == "open_options"
    next_flow = res_confirm.get("next_flow")
    assert next_flow is not None
    assert next_flow[0] == FlowType.OPTIONS_FLOW


async def test_auth_issue_clears_after_successful_run(
    hass: HomeAssistant,
    energy_manager: object,
    std_prefs: dict,
    aioclient_mock: object,
    inprocess_worker: object,
) -> None:
    """A subsequent successful run deletes the llm_auth Repairs issue."""
    entry = await _setup_ready_supervisor(hass, energy_manager, std_prefs)
    issue_reg = ir.async_get(hass)
    ir.async_create_issue(
        hass,
        DOMAIN,
        ISSUE_LLM_AUTH,
        is_fixable=True,
        is_persistent=False,
        severity=ir.IssueSeverity.ERROR,
        translation_key=ISSUE_LLM_AUTH,
    )
    assert issue_reg.async_get_issue(DOMAIN, ISSUE_LLM_AUTH) is not None

    def _responder(request_data: Any) -> dict[str, Any]:
        prop = _make_proposal([_make_device("d1", "c01")])
        return {"status": 200, "json": chat_completion(json.dumps(prop))}

    mock_llm(aioclient_mock, "http://llm.test/v1", _responder)

    supervisor = entry.runtime_data.learning
    await supervisor.async_request_run("service")
    assert supervisor.run_task is not None
    await supervisor.run_task

    assert issue_reg.async_get_issue(DOMAIN, ISSUE_LLM_AUTH) is None


async def test_auth_issue_clears_after_successful_probe(
    hass: HomeAssistant,
    energy_manager: object,
    std_prefs: dict,
    aioclient_mock: object,
) -> None:
    """Successful probe in options flow clears the llm_auth Repairs issue."""
    entry = await _setup_ready_supervisor(hass, energy_manager, std_prefs)
    issue_reg = ir.async_get(hass)
    ir.async_create_issue(
        hass,
        DOMAIN,
        ISSUE_LLM_AUTH,
        is_fixable=True,
        is_persistent=False,
        severity=ir.IssueSeverity.ERROR,
        translation_key=ISSUE_LLM_AUTH,
    )
    assert issue_reg.async_get_issue(DOMAIN, ISSUE_LLM_AUTH) is not None

    # Mock probe response
    probe_resp = {
        "id": "probe-1",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": json.dumps({"ok": True, "word": "pong", "n": 7}),
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 50, "completion_tokens": 10},
    }
    mock_llm(aioclient_mock, "http://llm.test/v1", lambda req: {"status": 200, "json": probe_resp})

    result = await hass.config_entries.options.async_init(entry.entry_id)
    result = await hass.config_entries.options.async_configure(
        result["flow_id"], {"next_step_id": "llm_endpoint"}
    )
    result = await hass.config_entries.options.async_configure(
        result["flow_id"],
        {
            "llm_base_url": "http://llm.test/v1",
            "llm_api_key": "sk-new-valid-key",
            "llm_model": "model-1",
        },
    )
    assert result["type"].name == "SHOW_PROGRESS"
    await hass.async_block_till_done()

    result = await hass.config_entries.options.async_configure(result["flow_id"])
    assert result["type"].name == "CREATE_ENTRY"
    await hass.async_block_till_done()
    assert issue_reg.async_get_issue(DOMAIN, ISSUE_LLM_AUTH) is None


async def test_plain_failures_set_failed_without_issue(
    hass: HomeAssistant,
    energy_manager: object,
    std_prefs: dict,
    aioclient_mock: object,
    inprocess_worker: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Timeouts, connects, 500s, 429 budget, and worker errors set status failed without issue."""
    entry = await _setup_ready_supervisor(hass, energy_manager, std_prefs)
    supervisor = entry.runtime_data.learning
    issue_reg = ir.async_get(hass)

    test_cases = [
        ({"exc": "timeout"}, "timeout"),
        ({"exc": "connect"}, "cannot_connect"),
        ({"status": 500, "text": "Internal Server Error"}, "http_500: Internal Server Error"),
        (
            {
                "status": 429,
                "json": {"error": {"type": "budget_exceeded", "message": "LiteLLM: Budget exceeded"}},
            },
            "provider_budget_exceeded",
        ),
        # LlmBadResponse: a 200 body that does not parse (llm_client ~410)
        ({"status": 200, "text": "not json"}, "invalid_json"),
    ]

    for responder_spec, expected_error in test_cases:
        aioclient_mock.clear_requests()
        mock_llm(aioclient_mock, "http://llm.test/v1", lambda req, s=responder_spec: s)

        await supervisor.async_request_run("service")
        assert supervisor.run_task is not None
        await supervisor.run_task

        data = supervisor.coordinator.data
        assert data["status"] == "failed"
        assert data["last_error"] is not None
        assert data["last_error"] == expected_error
        assert len(data["last_error"]) <= 500
        assert data["last_attempt"]["outcome"] == "failed"
        assert data["last_attempt"]["started"] is not None
        assert data["last_attempt"]["ended"] is not None
        assert issue_reg.async_get_issue(DOMAIN, ISSUE_LLM_AUTH) is None

    # Worker error case
    async def _failing_worker(stage: str, job_path: Any) -> dict[str, Any]:
        raise LearningStageError("memory limit exceeded in clustering")

    monkeypatch.setattr(supervisor, "_async_run_stage", _failing_worker)
    await supervisor.async_request_run("service")
    assert supervisor.run_task is not None
    await supervisor.run_task

    data = supervisor.coordinator.data
    assert data["status"] == "failed"
    assert data["last_error"] == "worker: memory limit exceeded in clustering"
    assert len(data["last_error"]) <= 500
    assert data["last_attempt"]["outcome"] == "failed"
    assert data["last_attempt"]["ended"] is not None
    assert issue_reg.async_get_issue(DOMAIN, ISSUE_LLM_AUTH) is None


async def test_refine_accepts_better_and_rejects_worse_round3(
    hass: HomeAssistant,
    energy_manager: object,
    std_prefs: dict,
    aioclient_mock: object,
    inprocess_worker: object,
) -> None:
    """Round 2 improves S on validation and is accepted; round 3 degrades S and is rejected."""
    entry = await _setup_ready_supervisor(hass, energy_manager, std_prefs)
    call_count = 0

    def _responder(request_data: Any) -> dict[str, Any]:
        nonlocal call_count
        call_count += 1
        if isinstance(request_data, (bytes, bytearray)):
            req = json.loads(request_data.decode("utf-8"))
        elif isinstance(request_data, str):
            req = json.loads(request_data)
        else:
            req = request_data
        n_bytes = len(json.dumps(req))
        prompt_tokens = max(1000, n_bytes // 3)
        sw = switched_cluster_ids(req)
        if call_count == 1:
            # Round 1: only the first switched cluster
            prop = _make_proposal([_make_device("d1", sw[0])])
            return {"status": 200, "json": chat_completion(json.dumps(prop), usage=(prompt_tokens, 200))}
        elif call_count == 2:
            # Round 2: first + second switched cluster (higher S)
            prop = _make_proposal([_make_device("d1", sw[0]), _make_device("d2", sw[1])])
            return {"status": 200, "json": chat_completion(json.dumps(prop), usage=(prompt_tokens, 200))}
        else:
            # Round 3: only the second switched cluster (lower S)
            prop = _make_proposal([_make_device("d2", sw[1])])
            return {"status": 200, "json": chat_completion(json.dumps(prop), usage=(prompt_tokens, 200))}

    mock_llm(aioclient_mock, "http://llm.test/v1", _responder)

    supervisor = entry.runtime_data.learning
    await supervisor.async_request_run("service")
    assert supervisor.run_task is not None
    await supervisor.run_task

    assert call_count == 3
    data = supervisor.coordinator.data
    assert data["status"] == "idle"
    assert data["last_error"] is None

    last_run = data["last_run"]
    assert last_run["rounds"] == 3
    assert last_run["accepted_round"] == 2

    # Candidates Store equals library_2 (2 devices)
    assert len(supervisor.candidates) == 2
    refs = {c["provenance"]["device_ref"] for c in supervisor.candidates}
    assert refs == {"d1", "d2"}

    # Check refinement request payloads
    # Call index 1 is Round 2 request
    call_r2 = aioclient_mock.mock_calls[1][2]
    if isinstance(call_r2, (bytes, bytearray)):
        call_r2 = json.loads(call_r2.decode("utf-8"))
    r2_content = call_r2["messages"][1]["content"]
    assert "<previous>" in r2_content
    assert "<scorecard>" in r2_content
    assert '"val"' not in r2_content
    assert '"val_' not in r2_content


async def test_refine_rejects_worse_round2(
    hass: HomeAssistant,
    energy_manager: object,
    std_prefs: dict,
    aioclient_mock: object,
    inprocess_worker: object,
) -> None:
    """Round 2 is worse than round 1, rejected, keeping round 1 library."""
    entry = await _setup_ready_supervisor(hass, energy_manager, std_prefs)
    call_count = 0

    def _responder(request_data: Any) -> dict[str, Any]:
        nonlocal call_count
        call_count += 1
        if isinstance(request_data, (bytes, bytearray)):
            req = json.loads(request_data.decode("utf-8"))
        elif isinstance(request_data, str):
            req = json.loads(request_data)
        else:
            req = request_data
        sw = switched_cluster_ids(req)
        usage = (max(1000, len(json.dumps(req)) // 3), 200)
        if call_count == 1:
            # Round 1: first + second switched cluster
            prop = _make_proposal([_make_device("d1", sw[0]), _make_device("d2", sw[1])])
            return {"status": 200, "json": chat_completion(json.dumps(prop), usage=usage)}
        else:
            # Round 2 and 3: worse proposal (only the second switched cluster)
            prop = _make_proposal([_make_device("d2", sw[1])])
            return {"status": 200, "json": chat_completion(json.dumps(prop), usage=usage)}

    mock_llm(aioclient_mock, "http://llm.test/v1", _responder)

    supervisor = entry.runtime_data.learning
    await supervisor.async_request_run("service")
    assert supervisor.run_task is not None
    await supervisor.run_task

    assert call_count == 3
    data = supervisor.coordinator.data
    assert data["status"] == "idle"

    last_run = data["last_run"]
    assert last_run["rounds"] == 3
    assert last_run["accepted_round"] == 1
    assert len(supervisor.candidates) == 2


async def test_refine_in_preview_mode_14_days(
    hass: HomeAssistant,
    energy_manager: object,
    std_prefs: dict,
    aioclient_mock: object,
    inprocess_worker: object,
) -> None:
    """In preview mode (14 usable days), refinement still runs up to 2 rounds and candidates carry validated false."""
    from tests.fixtures.learning.synth import SynthSpec

    energy_manager.data = std_prefs
    _set_std_powers(hass)
    entry = await async_setup_via_flow(hass)
    await async_fill_synth_buffer(hass, entry, spec=SynthSpec(days=14, oven=True))
    await async_seed_llm(hass, entry)
    await async_seed_consent(hass, host="llm.test", names=False)

    call_count = 0

    def _responder(request_data: Any) -> dict[str, Any]:
        nonlocal call_count
        call_count += 1
        if isinstance(request_data, (bytes, bytearray)):
            req = json.loads(request_data.decode("utf-8"))
        elif isinstance(request_data, str):
            req = json.loads(request_data)
        else:
            req = request_data
        sw = switched_cluster_ids(req)
        usage = (max(1000, len(json.dumps(req)) // 3), 200)
        if call_count == 1:
            prop = _make_proposal([_make_device("d1", sw[0])])
            return {"status": 200, "json": chat_completion(json.dumps(prop), usage=usage)}
        prop = _make_proposal([_make_device("d1", sw[0]), _make_device("d2", sw[1])])
        return {"status": 200, "json": chat_completion(json.dumps(prop), usage=usage)}

    mock_llm(aioclient_mock, "http://llm.test/v1", _responder)

    supervisor = entry.runtime_data.learning
    await supervisor.async_request_run("service")
    assert supervisor.run_task is not None
    await supervisor.run_task

    assert call_count == 3
    data = supervisor.coordinator.data
    assert data["status"] == "idle"
    last_run = data["last_run"]
    assert last_run["preview"] is True
    assert last_run["rounds"] == 3

    # Candidates carry validated false and val score None
    for cand in supervisor.candidates:
        assert cand["scores"]["validated"] is False
        assert cand["scores"]["val"] is None


async def test_refine_error_keeps_incumbent_and_finishes_idle(
    hass: HomeAssistant,
    energy_manager: object,
    std_prefs: dict,
    aioclient_mock: object,
    inprocess_worker: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A round-2 timeout leaves status idle, last_error refinement stopped: timeout, and stores library_1."""
    monkeypatch.setattr("custom_components.powerguesser.const.LLM_MAX_ATTEMPTS", 1)
    entry = await _setup_ready_supervisor(hass, energy_manager, std_prefs)
    call_count = 0

    def _responder(request_data: Any) -> dict[str, Any]:
        nonlocal call_count
        call_count += 1
        if isinstance(request_data, (bytes, bytearray)):
            req = json.loads(request_data.decode("utf-8"))
        elif isinstance(request_data, str):
            req = json.loads(request_data)
        else:
            req = request_data
        print(f"CALL {call_count} snippet:", req["messages"][-1]["content"][:150])
        if call_count == 1:
            prop = _make_proposal([_make_device("d1", "c01")])
            return {"status": 200, "json": chat_completion(json.dumps(prop))}
        # Round 2: timeout
        return {"exc": "timeout"}

    mock_llm(aioclient_mock, "http://llm.test/v1", _responder)

    supervisor = entry.runtime_data.learning
    await supervisor.async_request_run("service")
    assert supervisor.run_task is not None
    await supervisor.run_task

    assert call_count == 2
    data = supervisor.coordinator.data
    assert data["status"] == "idle"
    assert data["last_error"] is not None
    assert data["last_error"].startswith("refinement stopped: timeout")

    last_run = data["last_run"]
    assert last_run["rounds"] == 2
    assert last_run["accepted_round"] == 1
    assert len(supervisor.candidates) == 1
    assert supervisor.candidates[0]["provenance"]["device_ref"] == "d1"


async def test_refine_worker_failure_keeps_round1_library(
    hass: HomeAssistant,
    energy_manager: object,
    std_prefs: dict,
    aioclient_mock: object,
    inprocess_worker: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A worker failure in a refinement round keeps the paid round-1 library (WR-03)."""
    entry = await _setup_ready_supervisor(hass, energy_manager, std_prefs)
    supervisor = entry.runtime_data.learning

    def _responder(request_data: Any) -> dict[str, Any]:
        sw = switched_cluster_ids(request_data)
        prop = _make_proposal([_make_device("d1", sw[0])])
        return {"status": 200, "json": chat_completion(json.dumps(prop))}

    mock_llm(aioclient_mock, "http://llm.test/v1", _responder)

    inner = supervisor._async_run_stage

    async def _stage(stage: str, job_path: Any) -> dict[str, Any]:
        if stage == "compile_score" and str(job_path).endswith("job_compile_2.json"):
            raise LearningStageError("compile_score timed out after 900 s")
        return await inner(stage, job_path)

    monkeypatch.setattr(supervisor, "_async_run_stage", _stage)

    await supervisor.async_request_run("service")
    assert supervisor.run_task is not None
    await supervisor.run_task

    data = supervisor.coordinator.data
    assert data["status"] == "idle"
    assert data["last_error"].startswith("refinement stopped: worker: compile_score timed out")
    assert data["last_run"]["accepted_round"] == 1
    assert len(supervisor.candidates) == 1
    assert supervisor.candidates[0]["provenance"]["device_ref"] == "d1"
    stored = await supervisor._candidates_store.async_load()
    assert stored["run_id"] == data["last_run"]["run_id"]


async def test_refine_auth_error_raises_issue_and_finishes_idle(
    hass: HomeAssistant,
    energy_manager: object,
    std_prefs: dict,
    aioclient_mock: object,
    inprocess_worker: object,
) -> None:
    """An auth error in round 2 raises llm_auth Repairs issue, stores library_1, and finishes idle."""
    entry = await _setup_ready_supervisor(hass, energy_manager, std_prefs)
    issue_reg = ir.async_get(hass)
    call_count = 0

    def _responder(request_data: Any) -> dict[str, Any]:
        nonlocal call_count
        call_count += 1
        if isinstance(request_data, (bytes, bytearray)):
            req = json.loads(request_data.decode("utf-8"))
        elif isinstance(request_data, str):
            req = json.loads(request_data)
        else:
            req = request_data
        print(f"CALL {call_count} snippet:", req["messages"][-1]["content"][:150])
        if call_count == 1:
            prop = _make_proposal([_make_device("d1", "c01")])
            return {"status": 200, "json": chat_completion(json.dumps(prop))}
        return {"status": 401, "text": "Key revoked mid-run"}

    mock_llm(aioclient_mock, "http://llm.test/v1", _responder)

    supervisor = entry.runtime_data.learning
    await supervisor.async_request_run("service")
    assert supervisor.run_task is not None
    await supervisor.run_task

    assert call_count == 2
    data = supervisor.coordinator.data
    assert data["status"] == "idle"
    assert "refinement stopped: auth_failed (HTTP 401)" in data["last_error"]

    # Repairs issue was raised
    assert issue_reg.async_get_issue(DOMAIN, ISSUE_LLM_AUTH) is not None
    assert len(supervisor.candidates) == 1


_ECHO_KEY = "sk-echo-0123456789abcdef"


@pytest.mark.parametrize(
    ("response", "prefix"),
    [
        (
            {"status": 500, "json": {"error": {"message": f"upstream failed for {_ECHO_KEY}"}}},
            "http_500:",
        ),
        ({"status": 400, "text": f"bad request: Bearer {_ECHO_KEY}"}, "http_400:"),
    ],
    ids=["500-json", "400-text"],
)
async def test_provider_error_echoing_key_is_redacted_everywhere(
    hass: HomeAssistant,
    energy_manager: object,
    std_prefs: dict,
    aioclient_mock: object,
    inprocess_worker: object,
    hass_storage: dict[str, Any],
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
    response: dict[str, Any],
    prefix: str,
) -> None:
    """A key echoed in a provider error body never reaches sensor, Store, log or stage files (LLM-01, D-10)."""
    from pathlib import Path

    from custom_components.powerguesser import const
    from custom_components.powerguesser.const import (
        BUFFER_DIR,
        LEARNING_DIR,
        LEARNING_STORE_KEY,
    )

    caplog.set_level(logging.DEBUG)
    monkeypatch.setattr(const, "RETRY_BACKOFF_S", 0.0)
    entry = await _setup_ready_supervisor(hass, energy_manager, std_prefs)
    # Same scheme, host and port as the seeded consent, so consent stays valid (D-11 as amended).
    await async_seed_llm(hass, entry, api_key=_ECHO_KEY)
    mock_llm(aioclient_mock, "http://llm.test/v1", lambda req: response)

    supervisor = entry.runtime_data.learning
    await supervisor.async_request_run("service")
    assert supervisor.run_task is not None
    await supervisor.run_task
    await hass.async_block_till_done()

    data = supervisor.coordinator.data
    assert data["status"] == "failed"
    last_error = data["last_error"]
    assert last_error.startswith(prefix), last_error
    assert "**REDACTED**" in last_error
    assert _ECHO_KEY not in last_error
    assert _ECHO_KEY not in json.dumps(data, default=str)

    state = hass.states.get("sensor.powerguesser_learning_status")
    assert state is not None
    assert state.attributes.get("last_error") == last_error
    assert _ECHO_KEY not in json.dumps(dict(state.attributes), default=str)

    assert LEARNING_STORE_KEY in hass_storage
    assert _ECHO_KEY not in json.dumps(hass_storage[LEARNING_STORE_KEY], default=str)

    assert _ECHO_KEY not in caplog.text

    root = Path(hass.config.path(BUFFER_DIR, LEARNING_DIR))

    def _files_with_key() -> tuple[int, list[str]]:
        files = [p for p in root.rglob("*") if p.is_file()]
        hits = [str(p) for p in files if _ECHO_KEY.encode("utf-8") in p.read_bytes()]
        return len(files), hits

    n_files, hits = await hass.async_add_executor_job(_files_with_key)
    assert n_files > 0
    assert hits == []


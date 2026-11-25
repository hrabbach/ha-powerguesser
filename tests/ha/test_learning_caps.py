"""Tests for learning token caps, counters, monthly rollover, and usage fallback."""

import asyncio
import json
import math
from typing import Any
import pytest

from homeassistant.core import HomeAssistant
from homeassistant.helpers import issue_registry as ir
from homeassistant.util import dt as dt_util

from custom_components.powerguesser.const import (
    CONF_MONTH_INPUT_TOKENS,
    CONF_RUN_INPUT_TOKENS,
    CONF_RUN_OUTPUT_TOKENS,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MONTH_INPUT_TOKENS,
    DOMAIN,
    ISSUE_LEARNING_CAPPED,
)
from custom_components.powerguesser.engine.learning.budget import month_key, output_estimate
from custom_components.powerguesser.engine.learning.prompts import payload_bytes
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

# Content-density model of kimi-for-coding on the owner's LiteLLM proxy (03-SPIKE.md and
# debug session truncation-guard-false-positive): the numeric digest tokenizes at about
# 0.593 tok/B, the static prefix, schema and pretty-printed JSON at about 0.24-0.29 tok/B.
DIGEST_TOK_PER_B = 0.593
OTHER_TOK_PER_B = 0.25


def _req_json(request_data: Any) -> dict[str, Any]:
    if isinstance(request_data, (bytes, bytearray)):
        return json.loads(request_data.decode("utf-8"))
    if isinstance(request_data, str):
        return json.loads(request_data)
    return request_data


def _digest_bytes(req: dict[str, Any]) -> int:
    """UTF-8 bytes of the escaped <data> digest inside the compact request body (0 if none)."""
    user = req["messages"][-1]["content"]
    if "<data>\n" not in user:
        return 0
    rendered = user.split("<data>\n", 1)[1].split("\n</data>", 1)[0]
    return len(json.dumps(rendered, ensure_ascii=False).encode("utf-8")) - 2


def _density_tokens(req: dict[str, Any]) -> int:
    n_bytes = payload_bytes(req)
    digest_b = _digest_bytes(req)
    return math.ceil(DIGEST_TOK_PER_B * digest_b + OTHER_TOK_PER_B * (n_bytes - digest_b))


def _content_floor(n_bytes: int, digest_b: int) -> float:
    """The owner-specified truncation floor: 0.8 x (digest_bytes x 0.4 + other_bytes / 4.5)."""
    return 0.8 * (digest_b * 0.4 + (n_bytes - digest_b) / 4.5)


def _set_std_powers(hass: HomeAssistant) -> None:
    async_set_power(hass, "sensor.grid_power", 500)
    async_set_power(hass, "sensor.solar_power", 1200)
    async_set_power(hass, "sensor.battery_power", -300)
    async_set_power(hass, "sensor.washer_power", 200)


async def _setup_ready_supervisor(
    hass: HomeAssistant,
    energy_manager: object,
    std_prefs: dict[str, Any],
    days: int = 22,
) -> Any:
    from tests.fixtures.learning.synth import SynthSpec

    energy_manager.data = std_prefs
    _set_std_powers(hass)
    entry = await async_setup_via_flow(hass)
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


async def test_run_cap_stops_before_sending(
    hass: HomeAssistant,
    energy_manager: object,
    std_prefs: dict[str, Any],
    aioclient_mock: object,
    inprocess_worker: object,
) -> None:
    """options run_input_tokens 1000: learn -> status capped, last_error 'cap_reached: run_input', non-fixable issue, 0 calls."""
    entry = await _setup_ready_supervisor(hass, energy_manager, std_prefs)
    hass.config_entries.async_update_entry(
        entry,
        options={**entry.options, CONF_RUN_INPUT_TOKENS: 1000},
    )

    mock_llm(
        aioclient_mock,
        "http://llm.test/v1",
        lambda _r: {"status": 200, "json": chat_completion(json.dumps(_make_proposal([_make_device("d1", "c01")])))}
    )

    supervisor = entry.runtime_data.learning
    await supervisor.async_request_run("service")
    assert supervisor.run_task is not None
    await supervisor.run_task

    assert aioclient_mock.call_count == 0
    data = supervisor.coordinator.data
    assert data["status"] == "capped"
    assert data["last_error"] == "cap_reached: run_input"
    # A capped run records its attempt as capped (CLEAN-02, D-15)
    assert data["last_attempt"]["outcome"] == "capped"
    assert data["last_attempt"]["started"] is not None
    assert data["last_attempt"]["ended"] is not None

    issue_reg = ir.async_get(hass)
    issue = issue_reg.async_get_issue(DOMAIN, ISSUE_LEARNING_CAPPED)
    assert issue is not None
    assert issue.is_fixable is False
    assert issue.is_persistent is False
    assert issue.severity == ir.IssueSeverity.WARNING
    assert issue.translation_placeholders == {"cap": "run_input", "limit": "1000"}


async def test_cap_during_refinement_keeps_incumbent(
    hass: HomeAssistant,
    energy_manager: object,
    std_prefs: dict[str, Any],
    aioclient_mock: object,
    inprocess_worker: object,
) -> None:
    """A cap hit during refinement stops further rounds, marks status capped, and keeps the incumbent library."""
    entry = await _setup_ready_supervisor(hass, energy_manager, std_prefs)
    call_count = 0

    def _responder(request_data: Any) -> dict[str, Any]:
        nonlocal call_count
        call_count += 1
        prop = _make_proposal([_make_device("d1", "c01")])
        return {
            "status": 200,
            "json": chat_completion(
                json.dumps(prop),
                usage=(2500, 200),
            ),
        }

    mock_llm(aioclient_mock, "http://llm.test/v1", _responder)

    # Preview to find exact estimated input tokens for round 1
    preview = await entry.runtime_data.learning.async_preview_payload()
    r1_tokens = preview["estimated_input_tokens"]

    # Set run_input_tokens to allow round 1 (r1_tokens) but block round 2 (2500 used + ~r1_tokens > r1_tokens + 200)
    hass.config_entries.async_update_entry(
        entry,
        options={**entry.options, CONF_RUN_INPUT_TOKENS: r1_tokens + 200},
    )

    supervisor = entry.runtime_data.learning
    await supervisor.async_request_run("service")
    assert supervisor.run_task is not None
    await supervisor.run_task

    assert call_count == 1  # Only round 1 was sent; round 2 was blocked by cap
    data = supervisor.coordinator.data
    assert data["status"] == "capped"
    assert data["last_error"] == "cap_reached: run_input"
    assert len(supervisor.candidates) == 1
    assert data["pending_candidates"] == 1

    issue_reg = ir.async_get(hass)
    assert issue_reg.async_get_issue(DOMAIN, ISSUE_LEARNING_CAPPED) is not None


async def test_month_counter_persists_and_blocks_next_run(
    hass: HomeAssistant,
    energy_manager: object,
    std_prefs: dict[str, Any],
    aioclient_mock: object,
    inprocess_worker: object,
) -> None:
    """Month input counter persists across supervisor store and blocks run when cap reached."""
    entry = await _setup_ready_supervisor(hass, energy_manager, std_prefs)
    supervisor = entry.runtime_data.learning

    curr_month = month_key(dt_util.now())
    # Pre-seed month input to near cap
    c_data = dict(supervisor.coordinator.data)
    c_data["month"] = {"key": curr_month, "input": DEFAULT_MONTH_INPUT_TOKENS - 50, "output": 0, "cost": None}
    supervisor.coordinator.data = c_data
    await supervisor._learning_store.async_save(supervisor._build_learning_store_data())

    mock_llm(
        aioclient_mock,
        "http://llm.test/v1",
        lambda _r: {"status": 200, "json": chat_completion(json.dumps(_make_proposal([_make_device("d1", "c01")])))}
    )

    await supervisor.async_request_run("service")
    assert supervisor.run_task is not None
    await supervisor.run_task

    assert aioclient_mock.call_count == 0
    assert supervisor.coordinator.data["status"] == "capped"
    assert supervisor.coordinator.data["last_error"] == "cap_reached: month_input"

    # Reload from store
    stored = await supervisor._learning_store.async_load()
    assert stored["month"]["input"] == DEFAULT_MONTH_INPUT_TOKENS - 50


async def test_month_rollover_resets_counters_and_clears_issue(
    hass: HomeAssistant,
    energy_manager: object,
    std_prefs: dict[str, Any],
    freezer: Any,
) -> None:
    """Month rollover resets monthly counters, deletes learning_capped issue, and sets status to idle."""
    freezer.move_to("2026-03-25 12:00:00")
    entry = await _setup_ready_supervisor(hass, energy_manager, std_prefs)
    supervisor = entry.runtime_data.learning

    # Set capped status and issue
    c_data = dict(supervisor.coordinator.data)
    c_data["status"] = "capped"
    c_data["last_error"] = "cap_reached: month_input"
    c_data["month"] = {"key": "2026-03", "input": 2_000_000, "output": 500_000, "cost": None}
    supervisor.coordinator.data = c_data

    ir.async_create_issue(
        hass,
        DOMAIN,
        ISSUE_LEARNING_CAPPED,
        is_fixable=False,
        is_persistent=False,
        config_entry_id=entry.entry_id,
        severity=ir.IssueSeverity.WARNING,
        translation_key=ISSUE_LEARNING_CAPPED,
        translation_placeholders={"cap": "month_input", "limit": "2000000"},
    )
    issue_reg = ir.async_get(hass)
    assert issue_reg.async_get_issue(DOMAIN, ISSUE_LEARNING_CAPPED) is not None

    # Advance time to April
    if supervisor._unsub_refresh is not None:
        supervisor._unsub_refresh()
        supervisor._unsub_refresh = None
    freezer.move_to("2026-04-02 12:00:00")
    rolled = await supervisor._async_handle_month_rollover_if_needed()
    assert rolled is True

    # Issue deleted and counters reset
    assert issue_reg.async_get_issue(DOMAIN, ISSUE_LEARNING_CAPPED) is None
    data = supervisor.coordinator.data
    assert data["month"]["key"] == "2026-04"
    assert data["month"]["input"] == 0
    assert data["month"]["output"] == 0
    assert data["status"] == "idle"


async def test_missing_usage_counts_estimates(
    hass: HomeAssistant,
    energy_manager: object,
    std_prefs: dict[str, Any],
    aioclient_mock: object,
    inprocess_worker: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When a response carries no usage, the input estimate and ceil(reply bytes / 3) are counted instead."""
    monkeypatch.setattr("custom_components.powerguesser.learning.MAX_REFINE_ROUNDS", 0)
    entry = await _setup_ready_supervisor(hass, energy_manager, std_prefs)

    prop = _make_proposal([_make_device("d1", "c01")])
    prop_json = json.dumps(prop)

    # Response with usage=None
    def _responder(request_data: Any) -> dict[str, Any]:
        reply = chat_completion(prop_json)
        reply["usage"] = None
        return {"status": 200, "json": reply}

    mock_llm(aioclient_mock, "http://llm.test/v1", _responder)

    supervisor = entry.runtime_data.learning
    await supervisor.async_request_run("service")
    assert supervisor.run_task is not None
    await supervisor.run_task

    assert supervisor.coordinator.data["status"] == "idle"
    tokens_run = supervisor.coordinator.data["tokens_run"]
    # Input tokens counted from payload estimate (> 0)
    assert tokens_run["input"] > 1000
    # Output tokens counted from content bytes / 3
    assert tokens_run["output"] == output_estimate(len(prop_json.encode("utf-8")))


async def test_equal_to_cap_is_sent(
    hass: HomeAssistant,
    energy_manager: object,
    std_prefs: dict[str, Any],
    aioclient_mock: object,
    inprocess_worker: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A request whose estimate brings a counter exactly to its cap is sent (equality allowed)."""
    monkeypatch.setattr("custom_components.powerguesser.learning.MAX_REFINE_ROUNDS", 0)
    entry = await _setup_ready_supervisor(hass, energy_manager, std_prefs)

    # Preview to find exact estimated input tokens
    preview = await entry.runtime_data.learning.async_preview_payload()
    exact_tokens = preview["estimated_input_tokens"]

    # Set run_input_tokens to exactly exact_tokens
    hass.config_entries.async_update_entry(
        entry,
        options={**entry.options, CONF_RUN_INPUT_TOKENS: exact_tokens},
    )

    prop = _make_proposal([_make_device("d1", "c01")])
    mock_llm(
        aioclient_mock,
        "http://llm.test/v1",
        lambda _r: {"status": 200, "json": chat_completion(json.dumps(prop))},
    )

    supervisor = entry.runtime_data.learning
    await supervisor.async_request_run("service")
    assert supervisor.run_task is not None
    await supervisor.run_task

    assert aioclient_mock.call_count == 1
    assert supervisor.coordinator.data["status"] == "idle"


async def test_cost_tracking_with_prices(
    hass: HomeAssistant,
    energy_manager: object,
    std_prefs: dict[str, Any],
    aioclient_mock: object,
    inprocess_worker: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With both prices set, response adds tokens / 1e6 x price to month cost and sensor shows it rounded to 4 decimals; without prices cost is null."""
    monkeypatch.setattr("custom_components.powerguesser.learning.MAX_REFINE_ROUNDS", 0)
    entry = await _setup_ready_supervisor(hass, energy_manager, std_prefs)
    supervisor = entry.runtime_data.learning

    # Without prices, sensor cost_this_month is None
    sensor_state = hass.states.get("sensor.powerguesser_learning_status")
    assert sensor_state is not None
    assert sensor_state.attributes["cost_this_month"] is None

    # Set prices: $2.00 / 1M in, $4.00 / 1M out
    from custom_components.powerguesser.const import CONF_PRICE_INPUT, CONF_PRICE_OUTPUT
    hass.config_entries.async_update_entry(
        entry,
        options={**entry.options, CONF_PRICE_INPUT: 2.0, CONF_PRICE_OUTPUT: 4.0},
    )

    prop = _make_proposal([_make_device("d1", "c01")])
    mock_llm(
        aioclient_mock,
        "http://llm.test/v1",
        lambda _r: {
            "status": 200,
            "json": chat_completion(
                json.dumps(prop),
                usage=(10_000, 2_000),
            ),
        },
    )

    await supervisor.async_request_run("service")
    assert supervisor.run_task is not None
    await supervisor.run_task

    # Cost = (10_000 / 1e6) * 2 + (2_000 / 1e6) * 4 = 0.02 + 0.008 = 0.028
    sensor_state = hass.states.get("sensor.powerguesser_learning_status")
    assert sensor_state.attributes["cost_this_month"] == 0.028


async def test_month_cost_limit_stops_request(
    hass: HomeAssistant,
    energy_manager: object,
    std_prefs: dict[str, Any],
    aioclient_mock: object,
    inprocess_worker: object,
) -> None:
    """A month cost limit stops the next request whose estimated cost would exceed it."""
    entry = await _setup_ready_supervisor(hass, energy_manager, std_prefs)
    from custom_components.powerguesser.const import (
        CONF_MONTH_COST_LIMIT,
        CONF_PRICE_INPUT,
        CONF_PRICE_OUTPUT,
    )
    # Price $10 in, $20 out; limit $0.05. Max tokens 32000 -> output cost estimate is (32000/1e6)*20 = $0.64 > $0.05
    hass.config_entries.async_update_entry(
        entry,
        options={
            **entry.options,
            CONF_PRICE_INPUT: 10.0,
            CONF_PRICE_OUTPUT: 20.0,
            CONF_MONTH_COST_LIMIT: 0.05,
        },
    )

    mock_llm(
        aioclient_mock,
        "http://llm.test/v1",
        lambda _r: {"status": 200, "json": chat_completion(json.dumps(_make_proposal([_make_device("d1", "c01")])))}
    )

    supervisor = entry.runtime_data.learning
    await supervisor.async_request_run("service")
    assert supervisor.run_task is not None
    await supervisor.run_task

    assert aioclient_mock.call_count == 0
    assert supervisor.coordinator.data["status"] == "capped"
    assert supervisor.coordinator.data["last_error"] == "cap_reached: month_cost"


async def test_truncation_guard_aborts_on_suspected_truncation(
    hass: HomeAssistant,
    energy_manager: object,
    std_prefs: dict[str, Any],
    aioclient_mock: object,
    inprocess_worker: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reporting 100 prompt tokens for a large request triggers prompt_truncated and fails without retry or repairs issue."""
    monkeypatch.setattr("custom_components.powerguesser.learning.MAX_REFINE_ROUNDS", 0)
    entry = await _setup_ready_supervisor(hass, energy_manager, std_prefs)

    prop = _make_proposal([_make_device("d1", "c01")])
    call_count = 0

    def _responder(_req: Any) -> dict[str, Any]:
        nonlocal call_count
        call_count += 1
        return {
            "status": 200,
            "json": chat_completion(
                json.dumps(prop),
                usage=(100, 200),  # far below 0.8 x (digest x 0.4 + other / 4.5)
            ),
        }

    mock_llm(aioclient_mock, "http://llm.test/v1", _responder)

    supervisor = entry.runtime_data.learning
    await supervisor.async_request_run("service")
    assert supervisor.run_task is not None
    await supervisor.run_task

    assert call_count == 1  # No retry
    assert supervisor.coordinator.data["status"] == "failed"
    last_err = supervisor.coordinator.data["last_error"]
    assert last_err.startswith("prompt_truncated")
    assert "100" in last_err
    assert "truncation check" in last_err

    # No Repairs issue exists
    issue_reg = ir.async_get(hass)
    assert issue_reg.async_get_issue(DOMAIN, ISSUE_LEARNING_CAPPED) is None


async def test_truncation_guard_disabled_allows_run_and_calibrates(
    hass: HomeAssistant,
    energy_manager: object,
    std_prefs: dict[str, Any],
    aioclient_mock: object,
    inprocess_worker: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With truncation_guard disabled, low prompt tokens are accepted, run succeeds, and calibration is updated."""
    monkeypatch.setattr("custom_components.powerguesser.learning.MAX_REFINE_ROUNDS", 0)
    entry = await _setup_ready_supervisor(hass, energy_manager, std_prefs)

    from custom_components.powerguesser.const import CONF_TRUNCATION_GUARD
    hass.config_entries.async_update_entry(
        entry,
        options={**entry.options, CONF_TRUNCATION_GUARD: False},
    )

    prop = _make_proposal([_make_device("d1", "c01")])
    mock_llm(
        aioclient_mock,
        "http://llm.test/v1",
        lambda _r: {
            "status": 200,
            "json": chat_completion(
                json.dumps(prop),
                usage=(100, 200),
            ),
        },
    )

    supervisor = entry.runtime_data.learning
    await supervisor.async_request_run("service")
    assert supervisor.run_task is not None
    await supervisor.run_task

    assert supervisor.coordinator.data["status"] == "idle"
    # Calibration is updated
    assert "llm.test|m1" in supervisor._calibration
    calib = supervisor._calibration["llm.test|m1"]
    assert calib["n"] == 1
    assert calib["ratio"] > 0


async def test_truncation_guard_passes_refine_after_digest_dense_calibration(
    hass: HomeAssistant,
    energy_manager: object,
    std_prefs: dict[str, Any],
    aioclient_mock: object,
    inprocess_worker: object,
) -> None:
    """Regression 2026-11-18: propose -> refine against a content-density endpoint is not flagged.

    The persisted calibration holds the ratio of digest-dense round-1 asks on the live
    install (10617 tok / 20571 B, n=3). The fake endpoint reports prompt_tokens from a
    density model, so each request kind has its own tokens/byte and every one of them
    is below 0.8 x bytes x stored ratio, yet none is truncated.
    """
    entry = await _setup_ready_supervisor(hass, energy_manager, std_prefs)
    supervisor = entry.runtime_data.learning
    seeded_ratio = 10_617 / 20_571
    supervisor._calibration["llm.test|m1"] = {"ratio": seeded_ratio, "n": 3}
    await supervisor._learning_store.async_save(supervisor._build_learning_store_data())

    seen: list[tuple[int, int, int]] = []

    def _responder(request_data: Any) -> dict[str, Any]:
        req = _req_json(request_data)
        tokens = _density_tokens(req)
        seen.append((payload_bytes(req), _digest_bytes(req), tokens))
        sw = switched_cluster_ids(req)
        devices = [_make_device("d1", sw[0])]
        if len(seen) >= 2:
            devices.append(_make_device("d2", sw[1]))
        return {
            "status": 200,
            "json": chat_completion(json.dumps(_make_proposal(devices)), usage=(tokens, 200)),
        }

    mock_llm(aioclient_mock, "http://llm.test/v1", _responder)

    await supervisor.async_request_run("service")
    assert supervisor.run_task is not None
    await supervisor.run_task

    data = supervisor.coordinator.data
    assert data["last_error"] is None
    assert data["status"] == "idle"
    assert len(seen) == 3  # propose + two refine rounds
    assert data["last_run"]["rounds"] == 3
    assert all(digest_b > 0 for _, digest_b, _ in seen)
    # Every request is below the old ratio-based floor: the scenario exercises the bug.
    assert all(tokens < 0.8 * n_bytes * seeded_ratio for n_bytes, _, tokens in seen)
    # Calibration still learns from every reported request (used for cap estimates).
    assert supervisor._calibration["llm.test|m1"]["n"] == 6


async def test_truncation_guard_flags_refine_below_content_floor(
    hass: HomeAssistant,
    energy_manager: object,
    std_prefs: dict[str, Any],
    aioclient_mock: object,
    inprocess_worker: object,
) -> None:
    """A refine request answered below the digest-aware floor stops refinement as prompt_truncated.

    The reported count is above the no-digest floor, so the flag proves the refine
    request's digest bytes reach the guard.
    """
    entry = await _setup_ready_supervisor(hass, energy_manager, std_prefs)
    supervisor = entry.runtime_data.learning
    seen: list[tuple[int, int, int]] = []

    def _responder(request_data: Any) -> dict[str, Any]:
        req = _req_json(request_data)
        n_bytes, digest_b = payload_bytes(req), _digest_bytes(req)
        if not seen:
            tokens = _density_tokens(req)
        else:
            tokens = math.ceil(_content_floor(n_bytes, digest_b)) - 1
        seen.append((n_bytes, digest_b, tokens))
        sw = switched_cluster_ids(req)
        prop = _make_proposal([_make_device("d1", sw[0])])
        return {"status": 200, "json": chat_completion(json.dumps(prop), usage=(tokens, 200))}

    mock_llm(aioclient_mock, "http://llm.test/v1", _responder)

    await supervisor.async_request_run("service")
    assert supervisor.run_task is not None
    await supervisor.run_task

    assert len(seen) == 2
    n_bytes, digest_b, reported = seen[1]
    assert digest_b > 0
    assert reported >= _content_floor(n_bytes, 0)  # not flagged if digest bytes were dropped

    data = supervisor.coordinator.data
    assert data["status"] == "idle"
    assert data["last_error"].startswith("refinement stopped: prompt_truncated")
    assert f"reported {reported} prompt tokens for {n_bytes} bytes" in data["last_error"]
    assert data["last_run"]["rounds"] == 2
    assert data["last_run"]["accepted_round"] == 1
    assert len(supervisor.candidates) == 1


async def test_truncation_guard_repair_turn_uses_no_digest_floor(
    hass: HomeAssistant,
    energy_manager: object,
    std_prefs: dict[str, Any],
    aioclient_mock: object,
    inprocess_worker: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A repair turn carries no digest, so its floor ignores the round's digest bytes."""
    monkeypatch.setattr("custom_components.powerguesser.learning.MAX_REFINE_ROUNDS", 0)
    entry = await _setup_ready_supervisor(hass, energy_manager, std_prefs)
    supervisor = entry.runtime_data.learning
    seen: list[tuple[int, int, int]] = []
    switched: list[str] = []

    def _responder(request_data: Any) -> dict[str, Any]:
        req = _req_json(request_data)
        n_bytes, digest_b = payload_bytes(req), _digest_bytes(req)
        if not seen:
            switched.extend(switched_cluster_ids(req))
            tokens = _density_tokens(req)
            prop = _make_proposal([_make_device("d1", "c99")])  # unknown id -> repair turn
        else:
            tokens = math.ceil(_content_floor(n_bytes, 0))
            prop = _make_proposal([_make_device("d1", switched[0])])
        seen.append((n_bytes, digest_b, tokens))
        return {"status": 200, "json": chat_completion(json.dumps(prop), usage=(tokens, 200))}

    mock_llm(aioclient_mock, "http://llm.test/v1", _responder)

    await supervisor.async_request_run("service")
    assert supervisor.run_task is not None
    await supervisor.run_task

    assert len(seen) == 2
    round_digest_b = seen[0][1]
    rep_bytes, rep_digest_b, rep_tokens = seen[1]
    assert round_digest_b > 0
    assert rep_digest_b == 0
    # Flagged if the round's digest bytes leaked into the repair turn's floor.
    assert rep_tokens < _content_floor(rep_bytes, round_digest_b)

    data = supervisor.coordinator.data
    assert data["last_error"] is None
    assert data["status"] == "idle"
    assert len(supervisor.candidates) == 1



async def test_failed_attempts_are_charged(
    hass: HomeAssistant,
    energy_manager: object,
    std_prefs: dict[str, Any],
    aioclient_mock: object,
    inprocess_worker: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A retried attempt that may have been billed counts its input estimate and max_tokens (WR-04)."""
    from custom_components.powerguesser import const

    monkeypatch.setattr("custom_components.powerguesser.learning.MAX_REFINE_ROUNDS", 0)
    monkeypatch.setattr(const, "RETRY_BACKOFF_S", 0.0)
    entry = await _setup_ready_supervisor(hass, energy_manager, std_prefs)
    calls = 0

    def _responder(request_data: Any) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        if calls == 1:
            return {"status": 502, "text": "Bad Gateway"}
        prop = _make_proposal([_make_device("d1", "c01")])
        return {"status": 200, "json": chat_completion(json.dumps(prop), usage=(5000, 200))}

    mock_llm(aioclient_mock, "http://llm.test/v1", _responder)

    supervisor = entry.runtime_data.learning
    await supervisor.async_request_run("service")
    assert supervisor.run_task is not None
    await supervisor.run_task

    assert calls == 2
    data = supervisor.coordinator.data
    assert data["status"] == "idle"
    assert data["tokens_run"]["output"] == DEFAULT_MAX_TOKENS + 200
    assert data["tokens_run"]["input"] > 5000
    assert data["month"]["input"] == data["tokens_run"]["input"]
    assert data["month"]["output"] == data["tokens_run"]["output"]


async def test_retry_refused_when_cap_would_be_exceeded(
    hass: HomeAssistant,
    energy_manager: object,
    std_prefs: dict[str, Any],
    aioclient_mock: object,
    inprocess_worker: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retries are checked against the caps too: a retry that would exceed one is not sent (WR-04)."""
    from custom_components.powerguesser import const

    monkeypatch.setattr(const, "RETRY_BACKOFF_S", 0.0)
    entry = await _setup_ready_supervisor(hass, energy_manager, std_prefs)
    hass.config_entries.async_update_entry(
        entry, options={**entry.options, CONF_RUN_OUTPUT_TOKENS: DEFAULT_MAX_TOKENS + 4000}
    )
    await hass.async_block_till_done()

    mock_llm(aioclient_mock, "http://llm.test/v1", lambda _r: {"status": 502, "text": "Bad Gateway"})

    supervisor = entry.runtime_data.learning
    await supervisor.async_request_run("service")
    assert supervisor.run_task is not None
    await supervisor.run_task

    assert aioclient_mock.call_count == 1
    data = supervisor.coordinator.data
    assert data["status"] == "capped"
    assert data["last_error"] == "cap_reached: run_output"
    assert data["tokens_run"]["output"] == DEFAULT_MAX_TOKENS


async def test_cancelled_run_persists_month_counters(
    hass: HomeAssistant,
    energy_manager: object,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    aioclient_mock: object,
    inprocess_worker: object,
) -> None:
    """Tokens of a run cancelled mid-way stay in the persisted month counter (WR-04)."""
    from custom_components.powerguesser.const import LEARNING_STORE_KEY

    entry = await _setup_ready_supervisor(hass, energy_manager, std_prefs)
    second_call = asyncio.Event()
    calls = 0

    async def _responder(request_data: Any) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        if calls == 1:
            reply = chat_completion(json.dumps(_make_proposal([_make_device("d1", "c01")])))
            reply["usage"] = None
            return {"status": 200, "json": reply}
        second_call.set()
        await asyncio.Event().wait()
        return {"status": 500}

    mock_llm(aioclient_mock, "http://llm.test/v1", _responder)

    supervisor = entry.runtime_data.learning
    await supervisor.async_request_run("service")
    assert supervisor.run_task is not None
    await second_call.wait()
    counted = supervisor.coordinator.data["month"]["input"]
    assert counted > 0

    supervisor.run_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await supervisor.run_task
    await hass.async_block_till_done()

    assert hass_storage[LEARNING_STORE_KEY]["data"]["month"]["input"] == counted
    # A cancelled (user-stopped) run is recorded as stopped (CLEAN-02, D-15)
    assert hass_storage[LEARNING_STORE_KEY]["data"]["last_attempt"]["outcome"] == "stopped"
    assert hass_storage[LEARNING_STORE_KEY]["data"]["last_attempt"]["ended"] is not None


async def test_usage_without_prompt_tokens_does_not_calibrate(
    hass: HomeAssistant,
    energy_manager: object,
    std_prefs: dict[str, Any],
    aioclient_mock: object,
    inprocess_worker: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A usage block without prompt/completion tokens never feeds the estimate into calibration (WR-13)."""
    monkeypatch.setattr("custom_components.powerguesser.learning.MAX_REFINE_ROUNDS", 0)
    entry = await _setup_ready_supervisor(hass, energy_manager, std_prefs)

    prop_json = json.dumps(_make_proposal([_make_device("d1", "c01")]))

    def _responder(_r: Any) -> dict[str, Any]:
        reply = chat_completion(prop_json)
        reply["usage"] = {"total_tokens": 4321}
        return {"status": 200, "json": reply}

    mock_llm(aioclient_mock, "http://llm.test/v1", _responder)

    supervisor = entry.runtime_data.learning
    for _ in range(3):
        await supervisor.async_request_run("service")
        assert supervisor.run_task is not None
        await supervisor.run_task
        assert supervisor.coordinator.data["status"] == "idle"

    assert "llm.test|m1" not in supervisor._calibration


def test_default_run_cap_admits_eight_full_size_calls() -> None:
    """A fresh install's run cap admits eight calls at the new per-call default.

    Folded todo (raise max tokens for reasoning models): the per-call default rose to
    32 000 and the run cap to 256 000 so a fresh install still fits eight full-size
    calls, exactly as the old 16 000 / 128 000 pair did. The month output default stays
    600 000 and still admits at least two full runs.
    """
    from custom_components.powerguesser.const import (
        DEFAULT_MONTH_OUTPUT_TOKENS,
        DEFAULT_RUN_OUTPUT_TOKENS,
    )
    from custom_components.powerguesser.engine.learning.budget import (
        Caps,
        Usage,
        check_request,
    )

    caps = Caps(
        run_output=DEFAULT_RUN_OUTPUT_TOKENS,
        month_output=DEFAULT_MONTH_OUTPUT_TOKENS,
    )

    # Seven calls reserving DEFAULT_MAX_TOKENS each stay within the default run cap.
    assert 7 * DEFAULT_MAX_TOKENS <= DEFAULT_RUN_OUTPUT_TOKENS
    usage = Usage(run_output=7 * DEFAULT_MAX_TOKENS)
    # The next (eighth) call still fits: the cap allows equality.
    assert check_request(0, DEFAULT_MAX_TOKENS, usage, caps) is None
    assert 8 * DEFAULT_MAX_TOKENS == DEFAULT_RUN_OUTPUT_TOKENS

    # The unchanged month output default still admits at least two full runs.
    assert 2 * DEFAULT_RUN_OUTPUT_TOKENS <= DEFAULT_MONTH_OUTPUT_TOKENS

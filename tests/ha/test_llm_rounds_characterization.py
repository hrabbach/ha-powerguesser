"""Characterization snapshot of the production LLM rounds (Plan 05-03, step 0).

Recorded from the UNCHANGED code and committed before any production edit
(the ``_async_llm_rounds`` extraction in this plan); it is never updated in
this plan. A difference after the extraction is a regression in the
extraction to fix, not a snapshot to refresh (cross-AI review Round 1: the
golden episode-table snapshot does not cover the LLM rounds, and the review
asked for a parity net pinning every production request body byte for byte).

Each scenario runs one learning run on a 22-day synthetic buffer with frozen
time and a fixed run id, and snapshots one JSON-safe dict: every LLM request
body in order, the sorted stage-directory file names, ``last_run`` without
its timestamp fields, ``tokens_run``, ``status``, ``last_error`` and the
persisted candidate refs.

Removed fields (named here per the plan's rule): ``last_run["started"]`` and
``last_run["ended"]`` are wall-clock timestamps frozen anyway but unstable by
definition; ``last_run["shadow"]`` is popped defensively (this plan adds the
key on purpose and asserts it in its own tests). Request bodies are never
removed.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from syrupy.extensions.amber import AmberSnapshotExtension

from homeassistant.core import HomeAssistant

from custom_components.powerguesser.const import (
    BUFFER_DIR,
    CONF_RUN_INPUT_TOKENS,
    LEARNING_DIR,
)
from tests.conftest import (
    async_fill_synth_buffer,
    async_seed_consent,
    async_seed_llm,
    async_set_power,
    async_setup_via_flow,
    mock_llm,
    switched_cluster_ids,
)
from tests.fixtures.learning.fake_llm import chat_completion, proposal_for
from tests.fixtures.learning.synth import SynthSpec

_FIXED_RUN_HEX = "0123456789abcdef0123456789abcdef"

SCENARIOS = ("three_rounds", "empty_r1", "capped_round2")


@pytest.fixture
def snapshot(snapshot: Any) -> Any:
    return snapshot.use_extension(AmberSnapshotExtension)


def _fixed_uuid4() -> Any:
    """Stand-in for uuid.uuid4() with a fixed hex so run ids repeat."""
    return SimpleNamespace(hex=_FIXED_RUN_HEX)


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
            {"name": "on", "cluster_ref": cluster, "role": "main"}
        ],
        "phase_hint": "single",
        "confidence": "medium",
        "rationale": f"Device for {cluster}.",
        "ambiguous_with": [],
        "reference_plug": "",
    }


def _request_json(request_data: Any) -> dict[str, Any]:
    if isinstance(request_data, (bytes, bytearray)):
        request_data = request_data.decode("utf-8")
    return json.loads(request_data) if isinstance(request_data, str) else request_data


def _set_std_powers(hass: HomeAssistant) -> None:
    async_set_power(hass, "sensor.grid_power", 500)
    async_set_power(hass, "sensor.solar_power", 1200)
    async_set_power(hass, "sensor.battery_power", -300)
    async_set_power(hass, "sensor.washer_power", 200)


async def _setup_scenario(
    hass: HomeAssistant, energy_manager: object, std_prefs: dict[str, Any]
) -> Any:
    """Entry + seeded LLM/consent + 22-day oven buffer at the frozen instant."""
    energy_manager.data = std_prefs
    _set_std_powers(hass)
    entry = await async_setup_via_flow(hass)
    await async_fill_synth_buffer(hass, entry, spec=SynthSpec(days=22, oven=True))
    await async_seed_llm(hass, entry)
    await async_seed_consent(hass, host="llm.test", names=False)
    return entry


def _responder_three_rounds(aioclient_mock: Any) -> None:
    """Round 2 improves S and is accepted; round 3 degrades S and is rejected."""
    call_count = 0

    def _responder(request_data: Any) -> dict[str, Any]:
        nonlocal call_count
        call_count += 1
        req = _request_json(request_data)
        n_bytes = len(json.dumps(req))
        prompt_tokens = max(1000, n_bytes // 3)
        sw = switched_cluster_ids(req)
        usage = (prompt_tokens, 200)
        if call_count == 1:
            prop = _make_proposal([_make_device("d1", sw[0])])
        elif call_count == 2:
            prop = _make_proposal([_make_device("d1", sw[0]), _make_device("d2", sw[1])])
        else:
            prop = _make_proposal([_make_device("d2", sw[1])])
        return {"status": 200, "json": chat_completion(json.dumps(prop), usage=usage)}

    mock_llm(aioclient_mock, "http://llm.test/v1", _responder)


def _responder_incumbent(aioclient_mock: Any) -> None:
    """Always the same single-device proposal (usage 2500/200 per call)."""

    def _responder(request_data: Any) -> dict[str, Any]:
        prop = _make_proposal([_make_device("d1", "c01")])
        return {
            "status": 200,
            "json": chat_completion(json.dumps(prop), usage=(2500, 200)),
        }

    mock_llm(aioclient_mock, "http://llm.test/v1", _responder)


def _responder_empty_r1(aioclient_mock: Any) -> None:
    """An R1 proposal with no devices."""
    mock_llm(
        aioclient_mock,
        "http://llm.test/v1",
        lambda _r: {
            "status": 200,
            "json": chat_completion(json.dumps(proposal_for({}, devices=[]))),
        },
    )


def _collect(hass: HomeAssistant, supervisor: Any, aioclient_mock: Any) -> dict[str, Any]:
    """One JSON-safe characterization dict for the completed run."""
    data = supervisor.coordinator.data
    raw_last_run = data.get("last_run")
    if raw_last_run:
        last_run = dict(raw_last_run)
        last_run.pop("shadow", None)
        last_run.pop("started", None)
        last_run.pop("ended", None)
    else:
        # A capped run records status/error but no last_run.
        last_run = None

    requests = [_request_json(call[2]) for call in aioclient_mock.mock_calls]

    run_id = str(last_run["run_id"]) if last_run else _FIXED_RUN_HEX
    stage_dir = Path(hass.config.path(BUFFER_DIR, LEARNING_DIR)) / run_id
    files = sorted(
        str(p.relative_to(stage_dir)) for p in stage_dir.rglob("*") if p.is_file()
    )

    candidate_refs = sorted(
        str(c["provenance"]["device_ref"]) for c in supervisor.candidates
    )

    return {
        "requests": requests,
        "files": files,
        "last_run": last_run,
        "tokens_run": dict(data["tokens_run"]),
        "status": data["status"],
        "last_error": data["last_error"],
        "candidate_refs": candidate_refs,
    }


@pytest.mark.parametrize("scenario", SCENARIOS)
async def test_llm_rounds_unchanged_by_extraction(
    hass: HomeAssistant,
    energy_manager: object,
    std_prefs: dict[str, Any],
    aioclient_mock: object,
    inprocess_worker: object,
    freezer: Any,
    monkeypatch: pytest.MonkeyPatch,
    scenario: str,
    snapshot: Any,
) -> None:
    """The rounds extraction changes no request, file, field, token count or ref."""
    freezer.move_to(datetime(2026, 11, 20, 12, 0, 0, tzinfo=timezone.utc))
    monkeypatch.setattr(
        "custom_components.powerguesser.learning.uuid.uuid4", _fixed_uuid4
    )

    entry = await _setup_scenario(hass, energy_manager, std_prefs)

    if scenario == "three_rounds":
        _responder_three_rounds(aioclient_mock)
    elif scenario == "empty_r1":
        _responder_empty_r1(aioclient_mock)
    else:
        # capped_round2: the per-run cap admits the round-1 request but refuses
        # the round-2 one (same caps as test_cap_during_refinement_keeps_incumbent).
        preview = await entry.runtime_data.learning.async_preview_payload()
        r1_tokens = preview["estimated_input_tokens"]
        hass.config_entries.async_update_entry(
            entry,
            options={**entry.options, CONF_RUN_INPUT_TOKENS: r1_tokens + 200},
        )
        _responder_incumbent(aioclient_mock)

    supervisor = entry.runtime_data.learning
    await supervisor.async_request_run("service")
    assert supervisor.run_task is not None
    await supervisor.run_task

    assert _collect(hass, supervisor, aioclient_mock) == snapshot

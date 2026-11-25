"""Tests for the hidden masked shadow pass inside learning runs (Plan 05-03).

Shared setup lives in the module-level helpers below; Task 1 covers the
end-to-end tracer (a kettle-plug learning run stores a masked shadow set that
run_validation measures) and Task 2 covers cadence, caps, failures, isolation
and the after-learning validation chain.
"""

import json
import re
from pathlib import Path
from typing import Any

import pytest

from homeassistant.core import HomeAssistant

from custom_components.powerguesser.const import (
    BUFFER_DIR,
    DOMAIN,
    LEARNING_DIR,
)
from custom_components.powerguesser.engine.validation.constants import (
    SHADOW_MAX_AGE_DAYS,
)
from tests.conftest import (
    async_fill_synth_buffer,
    async_seed_consent,
    async_seed_llm,
    async_set_power,
    async_setup_via_flow,
    mock_llm,
)
from tests.fixtures.learning.fake_llm import chat_completion, kettle_proposal, proposal_for
from tests.fixtures.learning.synth import SynthSpec

_CLUSTER_RE = re.compile(
    r"id=(?P<id>c\d+)\s+kind=(?P<kind>\w+)\s+n=(?P<n>\d+)\s+"
    r"rise=(?P<r10>-?[\d.]+)/(?P<r50>-?[\d.]+)/(?P<r90>-?[\d.]+)\s+"
    r"fall=(?P<f50>-?[\d.]+)\s+dur_min=(?P<d10>[\d.]+)-(?P<d90>[\d.]+)"
)


def _digest_from_request(request_data: Any) -> dict[str, Any]:
    """Rebuild a digest-shaped dict from the rendered digest in an LLM request.

    Same helper as tests/ha/test_device_identity.py (the rendered cluster
    lines carry dur_min as p10-p90 only; the midpoint stands in for p50).
    """
    if isinstance(request_data, (bytes, bytearray)):
        request_data = request_data.decode("utf-8")
    req = json.loads(request_data) if isinstance(request_data, str) else request_data
    clusters: list[dict[str, Any]] = []
    for m in req.get("messages", []):
        content = m.get("content", "")
        if not isinstance(content, str):
            continue
        for cm in _CLUSTER_RE.finditer(content):
            d10 = float(cm.group("d10"))
            d90 = float(cm.group("d90"))
            clusters.append(
                {
                    "id": cm.group("id"),
                    "kind": cm.group("kind"),
                    "n": int(cm.group("n")),
                    "rise_w": {
                        "p10": float(cm.group("r10")),
                        "p50": float(cm.group("r50")),
                        "p90": float(cm.group("r90")),
                    },
                    "fall_w": {"p50": float(cm.group("f50"))},
                    "dur_min": {"p10": d10, "p50": (d10 + d90) / 2.0, "p90": d90},
                }
            )
    return {"clusters": clusters}


def _request_json(request_data: Any) -> dict[str, Any]:
    if isinstance(request_data, (bytes, bytearray)):
        request_data = request_data.decode("utf-8")
    return json.loads(request_data) if isinstance(request_data, str) else request_data


def _has_plug_lines(request_data: Any) -> bool:
    """True when the rendered request carries plug reference lines (D-02).

    The digest header renders ``plugs=N`` with the plug line count; a shadow
    digest always has plugs=0 (no plug lines, whatever the share-names flag).
    """
    req = _request_json(request_data)
    for msg in req.get("messages", []):
        content = msg.get("content", "")
        if isinstance(content, str) and re.search(r"plugs=[1-9]\d*", content):
            return True
    return False


def _default_responder(aioclient_mock: Any) -> None:
    """Production proposals stay empty; shadow proposals name the kettle."""

    def _responder(request_data: Any) -> dict[str, Any]:
        digest = _digest_from_request(request_data)
        if _has_plug_lines(request_data):
            prop = proposal_for(digest, devices=[])
        else:
            prop = kettle_proposal(digest)
        return {"status": 200, "json": chat_completion(json.dumps(prop))}

    mock_llm(aioclient_mock, "http://llm.test/v1", _responder)


async def _setup_kettle_entry(
    hass: HomeAssistant,
    energy_manager: object,
    kettle_plug_prefs: dict[str, Any],
    *,
    days: int = 35,
    kettle_plug: bool = True,
) -> Any:
    """Entry with the kettle plug configured and a synth buffer filled."""
    from tests.fixtures.learning.synth import ENTITY_KETTLE

    energy_manager.data = kettle_plug_prefs
    async_set_power(hass, "sensor.grid_power", 500)
    async_set_power(hass, "sensor.solar_power", 1200)
    async_set_power(hass, "sensor.battery_power", -300)
    async_set_power(hass, "sensor.washer_power", 200)
    async_set_power(hass, ENTITY_KETTLE, 0)
    entry = await async_setup_via_flow(hass)
    await async_fill_synth_buffer(
        hass, entry, spec=SynthSpec(days=days, oven=True, kettle_plug=kettle_plug)
    )
    await async_seed_llm(hass, entry)
    await async_seed_consent(hass, host="llm.test", names=False)
    return entry


async def _run_learn(entry: Any) -> Any:
    """Request one learning run and await its task."""
    supervisor = entry.runtime_data.learning
    await supervisor.async_request_run("service")
    assert supervisor.run_task is not None
    await supervisor.run_task
    return supervisor


def _channel_of(entry: Any, entity_id: str) -> int:
    """The buffer channel id of a plug entity (executor-only DB read)."""
    rows = entry.runtime_data.writer.db.channel_rows()
    for ch, eid, _role, kind in rows:
        if eid == entity_id:
            return int(ch)
    raise AssertionError(f"channel for {entity_id} not found")


async def _async_kettle_channel(hass: HomeAssistant, entry: Any) -> int:
    """Async wrapper: the buffer DB may not be read from the event loop."""
    return await hass.async_add_executor_job(_channel_of, entry, "sensor.kettle_power")


def _run_stage_dir(hass: HomeAssistant, run_id: str) -> Path:
    return Path(hass.config.path(BUFFER_DIR, LEARNING_DIR)) / run_id


async def test_production_job_keeps_zero_holdout(
    hass: HomeAssistant,
    energy_manager: object,
    kettle_plug_prefs: dict[str, Any],
    inprocess_worker: object,
    aioclient_mock: object,
) -> None:
    """D-05: only the shadow job and its cache key carry the calendar holdout."""
    entry = await _setup_kettle_entry(hass, energy_manager, kettle_plug_prefs)
    supervisor = entry.runtime_data.learning
    kettle_ch = await _async_kettle_channel(hass, entry)
    washer_ch = await hass.async_add_executor_job(
        _channel_of, entry, "sensor.washer_power"
    )

    def _build() -> tuple[dict, dict, dict, dict]:
        job, key, _end = supervisor._build_extract_job_data(Path("stage-prod"))
        sjob, skey, _send = supervisor._build_extract_job_data(
            Path("stage-shadow"), shadow=True
        )
        return job, key, sjob, skey

    job, key, sjob, skey = await hass.async_add_executor_job(_build)

    assert job["holdout_days"] == 0
    assert job["holdout_calendar_days"] == 0
    assert key["holdout_days"] == 0
    assert key["holdout_calendar_days"] == 0
    assert kettle_ch in job["metered_channels"]

    assert sjob["holdout_calendar_days"] == 7
    assert skey["holdout_calendar_days"] == 7
    assert sjob["holdout_days"] == 0
    assert sjob["metered_channels"] == []
    assert skey["metered_channels"] == []
    assert sjob["masked_channels"] == [
        {"channel": kettle_ch, "entity_id": "sensor.kettle_power"},
        {"channel": washer_ch, "entity_id": "sensor.washer_power"},
    ]

    # After a real learning run the production job file and digest meta stay
    # production-shaped, whether or not a shadow pass runs (D-05).
    mock_llm(
        aioclient_mock,
        "http://llm.test/v1",
        lambda _r: {
            "status": 200,
            "json": chat_completion(json.dumps(proposal_for({}, devices=[]))),
        },
    )
    supervisor = await _run_learn(entry)
    last_run = supervisor.coordinator.data["last_run"]
    stage_dir = _run_stage_dir(hass, str(last_run["run_id"]))
    job_on_disk = json.loads((stage_dir / "job_extract.json").read_text(encoding="utf-8"))
    assert job_on_disk["holdout_calendar_days"] == 0
    assert kettle_ch in job_on_disk["metered_channels"]
    meta = json.loads(
        (stage_dir / "digest_meta.json").read_text(encoding="utf-8")
    )
    assert meta["days"]["holdout"] == []


async def test_learning_run_stores_shadow_set_and_validation_uses_it(
    hass: HomeAssistant,
    energy_manager: object,
    kettle_plug_prefs: dict[str, Any],
    inprocess_worker: object,
    aioclient_mock: object,
) -> None:
    """VAL-01 end to end: the run stores a masked shadow set; run_validation measures the kettle."""
    entry = await _setup_kettle_entry(hass, energy_manager, kettle_plug_prefs)
    _default_responder(aioclient_mock)

    supervisor = await _run_learn(entry)
    data = supervisor.coordinator.data
    assert data["status"] == "idle"

    last_run = data["last_run"]
    shadow = last_run["shadow"]
    assert shadow["status"] == "ran"
    assert shadow["reason"] == "first_run"  # D-04 due reason recorded
    assert shadow["links"] == 1
    # The washer plug is metered too but never runs in the synth buffer, so
    # the shadow set reports it not_discovered (D-03: never silently dropped).
    assert shadow["not_discovered"] == 1

    stored = entry.runtime_data.validation.shadow_set
    assert stored is not None
    assert stored["run_id"] == f"{last_run['run_id']}-shadow"
    assert stored["eligible"] == ["sensor.kettle_power", "sensor.washer_power"]
    assert stored["holdout_days"] == 7
    (link,) = stored["links"]
    assert link["entity_id"] == "sensor.kettle_power"
    assert link["kind"] == "switched"
    assert link["overlap"] >= 0.5
    assert stored["plugs"]["sensor.kettle_power"]["train_mean_w"] > 0.0
    (missing,) = stored["not_discovered"]
    assert missing["entity_id"] == "sensor.washer_power"

    # Isolation: no shadow signature id in the production candidates (D-01).
    shadow_sigs = {a["signature_id"] for a in stored["artefacts"]}
    assert shadow_sigs, "the shadow pass must produce the kettle artefact"
    for cand in supervisor.candidates:
        assert cand["signature_id"] not in shadow_sigs

    res = await hass.services.async_call(
        DOMAIN, "run_validation", {}, blocking=True, return_response=True
    )
    assert res["status"] == "ok"
    kettle = next(
        d for d in res["devices"] if d["entity_id"] == "sensor.kettle_power"
    )
    assert isinstance(kettle["energy_error_pct"], (int, float))


async def test_shadow_set_artefacts_carry_no_evidence(
    hass: HomeAssistant,
    energy_manager: object,
    kettle_plug_prefs: dict[str, Any],
    inprocess_worker: object,
    aioclient_mock: object,
) -> None:
    """The shadow pass stores no evidence: its compile job writes with_evidence false.

    Evidence would otherwise be attached to the shadow artefacts from the
    shadow stage dir's own residual.npz; the shadow caller's False keeps the
    hidden shadow set lean (Plan 07-01).
    """
    entry = await _setup_kettle_entry(hass, energy_manager, kettle_plug_prefs)
    _default_responder(aioclient_mock)

    supervisor = await _run_learn(entry)
    last_run = supervisor.coordinator.data["last_run"]

    stored = entry.runtime_data.validation.shadow_set
    assert stored is not None
    assert stored["artefacts"], "the shadow responder proposes the kettle"
    for artefact in stored["artefacts"]:
        assert "evidence" not in artefact

    shadow_dir = _run_stage_dir(hass, str(last_run["run_id"])) / "shadow"
    job = json.loads((shadow_dir / "job_compile_1.json").read_text(encoding="utf-8"))
    assert job["with_evidence"] is False


# ---------------------------------------------------------------------------
# Task 2: cadence, caps, failures, isolation and the after-learning chain
# ---------------------------------------------------------------------------


def _kettle_artefact() -> dict:
    from tests.fixtures.detect_traces import switched_artefact

    return switched_artefact(
        "sig-shadow-kettle",
        rise=(1800, 2000, 2200),
        fall=(-2200, -2000, -1800),
        dur=(120, 180, 300),
        plateau=(1800, 2000, 2200),
    )


def _kettle_link() -> dict:
    return {
        "entity_id": "sensor.kettle_power",
        "signature_id": "sig-shadow-kettle",
        "kind": "switched",
        "overlap": 0.8,
    }


async def _seed_kettle_shadow(hass: HomeAssistant, entry: Any, **kwargs: Any) -> dict:
    """Seed a stored shadow set (v1 contract) for cadence/cap tests."""
    from tests.conftest import async_seed_shadow_set

    return await async_seed_shadow_set(
        hass, entry, [_kettle_artefact()], [_kettle_link()], **kwargs
    )


def _n_shadow_requests(aioclient_mock: Any) -> int:
    """How many LLM requests carried a digest without plug lines (D-02)."""
    n = 0
    for call in aioclient_mock.mock_calls:
        if not _has_plug_lines(call[2]):
            n += 1
    return n


def _production_nonempty_responder(aioclient_mock: Any) -> None:
    """Production proposes real devices; shadow proposes the kettle."""

    def _responder(request_data: Any) -> dict[str, Any]:
        digest = _digest_from_request(request_data)
        if _has_plug_lines(request_data):
            prop = proposal_for(digest)
        else:
            prop = kettle_proposal(digest)
        return {"status": 200, "json": chat_completion(json.dumps(prop))}

    mock_llm(aioclient_mock, "http://llm.test/v1", _responder)


def _patch_stage_failure(
    supervisor: Any, monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    """Raise LearningStageError when one stage runs; pass everything else through."""
    from custom_components.powerguesser.learning import LearningStageError

    orig = supervisor._async_run_stage

    async def _patched(stage_name: str, job_path: Any, _orig: Any = orig) -> Any:
        if stage_name == stage:
            raise LearningStageError(f"{stage} boom")
        return await _orig(stage_name, job_path)

    monkeypatch.setattr(supervisor, "_async_run_stage", _patched)


async def test_shadow_skipped_when_not_due(
    hass: HomeAssistant,
    energy_manager: object,
    kettle_plug_prefs: dict[str, Any],
    inprocess_worker: object,
    aioclient_mock: object,
) -> None:
    """A second run within 30 days with unchanged inputs sends no shadow request."""
    entry = await _setup_kettle_entry(hass, energy_manager, kettle_plug_prefs)
    _default_responder(aioclient_mock)

    supervisor = await _run_learn(entry)
    assert supervisor.coordinator.data["last_run"]["shadow"]["status"] == "ran"
    stored_before = dict(entry.runtime_data.validation.shadow_set)
    n_shadow_before = _n_shadow_requests(aioclient_mock)

    supervisor = await _run_learn(entry)
    data = supervisor.coordinator.data
    assert data["status"] == "idle"
    shadow = data["last_run"]["shadow"]
    assert shadow["status"] == "skipped"
    assert shadow["reason"] == "not_due"
    # The LLM received only the production requests of the second run.
    assert _n_shadow_requests(aioclient_mock) == n_shadow_before
    # The stored set is unchanged, including its run_id (D-04).
    stored_after = entry.runtime_data.validation.shadow_set
    assert stored_after["created"] == stored_before["created"]
    assert stored_after["run_id"] == stored_before["run_id"]


async def test_shadow_reruns_when_stale_or_inputs_change(
    hass: HomeAssistant,
    energy_manager: object,
    kettle_plug_prefs: dict[str, Any],
    inprocess_worker: object,
    aioclient_mock: object,
) -> None:
    """shadow_due names the reason: stale, eligible/grid/event-types changed; order-free types."""
    from datetime import timedelta

    import homeassistant.util.dt as dt_util

    entry = await _setup_kettle_entry(hass, energy_manager, kettle_plug_prefs)
    await _seed_kettle_shadow(hass, entry)
    validation = entry.runtime_data.validation
    stored = validation.shadow_set
    eligible = list(stored["eligible"])
    grid_s = stored["grid_s"]
    types = list(stored["enabled_event_types"])
    now = dt_util.utcnow()

    assert validation.shadow_due(eligible, grid_s, types, now) is None
    assert (
        validation.shadow_due(eligible + ["sensor.other"], grid_s, types, now)
        == "eligible_changed"
    )
    assert validation.shadow_due(eligible, grid_s + 10, types, now) == "grid_changed"
    assert validation.shadow_due(eligible, grid_s, list(reversed(types)), now) is None
    assert (
        validation.shadow_due(eligible, grid_s, ["step"], now) == "event_types_changed"
    )
    future = now + timedelta(days=SHADOW_MAX_AGE_DAYS + 1)
    assert validation.shadow_due(eligible, grid_s, types, future) == "stale"
    assert validation.shadow_due(eligible, grid_s, types, now + timedelta(days=29)) is None

    # End to end: a set created 31 days ago is replaced on the next run.
    await _seed_kettle_shadow(
        hass,
        entry,
        created=(dt_util.utcnow() - timedelta(days=31)).isoformat(),
    )
    _default_responder(aioclient_mock)
    supervisor = await _run_learn(entry)
    shadow = supervisor.coordinator.data["last_run"]["shadow"]
    assert shadow["status"] == "ran"
    assert shadow["reason"] == "stale"
    new_set = entry.runtime_data.validation.shadow_set
    assert new_set["run_id"] == f"{supervisor.coordinator.data['last_run']['run_id']}-shadow"
    assert new_set["run_id"] != stored["run_id"]


@pytest.mark.parametrize("case", ["worker_failure", "capped_r1", "cancelled"])
async def test_shadow_pass_only_after_successful_production(
    hass: HomeAssistant,
    energy_manager: object,
    kettle_plug_prefs: dict[str, Any],
    inprocess_worker: object,
    aioclient_mock: object,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
) -> None:
    """D-04: a failed, capped or cancelled production run runs no shadow pass."""
    import asyncio

    from homeassistant.helpers import issue_registry as ir

    from custom_components.powerguesser.const import CONF_RUN_INPUT_TOKENS
    from custom_components.powerguesser.learning import LearningStageError

    entry = await _setup_kettle_entry(hass, energy_manager, kettle_plug_prefs)
    supervisor = entry.runtime_data.learning

    if case == "worker_failure":
        _production_nonempty_responder(aioclient_mock)
        _patch_stage_failure(supervisor, monkeypatch, "compile_score")
    elif case == "capped_r1":
        preview = await supervisor.async_preview_payload()
        r1_tokens = preview["estimated_input_tokens"]
        hass.config_entries.async_update_entry(
            entry,
            options={**entry.options, CONF_RUN_INPUT_TOKENS: r1_tokens - 1000},
        )

        def _forbidden(_r: Any) -> dict[str, Any]:
            raise AssertionError("no LLM request may be sent when the run is capped")

        mock_llm(aioclient_mock, "http://llm.test/v1", _forbidden)
    else:
        second_call = asyncio.Event()

        async def _responder(request_data: Any) -> dict[str, Any]:
            if _has_plug_lines(request_data):
                digest = _digest_from_request(request_data)
                prop = proposal_for(digest)
                return {"status": 200, "json": chat_completion(json.dumps(prop))}
            second_call.set()
            await asyncio.Event().wait()
            return {"status": 500}

        mock_llm(aioclient_mock, "http://llm.test/v1", _responder)

    await supervisor.async_request_run("service")
    assert supervisor.run_task is not None
    if case == "cancelled":
        await second_call.wait()
        supervisor.run_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await supervisor.run_task
        await hass.async_block_till_done()
    else:
        await supervisor.run_task

    # No shadow set is stored (the Store is absent) and last_run never
    # records a ran shadow pass. For failed/capped production runs no shadow
    # request is sent at all; a cancel can land mid-shadow-request (the
    # request went out before the cancellation), which the bullets allow.
    assert entry.runtime_data.validation.shadow_set is None
    if case in ("worker_failure", "capped_r1"):
        assert _n_shadow_requests(aioclient_mock) == 0
    data = supervisor.coordinator.data
    last_run = data.get("last_run") or {}
    assert (last_run.get("shadow") or {}).get("status") != "ran"
    if case == "worker_failure":
        assert data["status"] == "failed"
        assert data["last_error"] == "worker: compile_score boom"
    elif case == "capped_r1":
        assert data["status"] == "capped"
        assert aioclient_mock.call_count == 0
        assert ir.async_get(hass).async_get_issue(
            "powerguesser", "learning_capped"
        ) is not None


async def test_shadow_skipped_below_28_days(
    hass: HomeAssistant,
    energy_manager: object,
    kettle_plug_prefs: dict[str, Any],
    inprocess_worker: object,
    aioclient_mock: object,
) -> None:
    """A 22-day buffer: the shadow gate stops before any extract or LLM request."""
    entry = await _setup_kettle_entry(hass, energy_manager, kettle_plug_prefs, days=22)
    _default_responder(aioclient_mock)

    supervisor = await _run_learn(entry)
    data = supervisor.coordinator.data
    assert data["status"] == "idle"
    shadow = data["last_run"]["shadow"]
    assert shadow["status"] == "skipped"
    assert shadow["reason"] == "insufficient_data"
    assert _n_shadow_requests(aioclient_mock) == 0
    assert entry.runtime_data.validation.shadow_set is None


async def test_shadow_skipped_when_worker_finds_too_few_days(
    hass: HomeAssistant,
    energy_manager: object,
    kettle_plug_prefs: dict[str, Any],
    inprocess_worker: object,
    aioclient_mock: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RESEARCH Pitfall 5: the HA pre-filter passes but the worker counts 22 days.

    The shortfall is a normal extract result: status idle, production
    candidates saved, last_run.shadow {skipped, insufficient_data} (never
    failed), last_error unchanged and no shadow request.
    """
    entry = await _setup_kettle_entry(hass, energy_manager, kettle_plug_prefs, days=22)
    _production_nonempty_responder(aioclient_mock)
    supervisor = entry.runtime_data.learning

    async def _fake_usable_days() -> float:
        return 35.0

    monkeypatch.setattr(supervisor, "_async_compute_usable_days", _fake_usable_days)

    supervisor = await _run_learn(entry)
    data = supervisor.coordinator.data
    assert data["status"] == "idle"
    assert len(supervisor.candidates) >= 1
    shadow = data["last_run"]["shadow"]
    assert shadow["status"] == "skipped"
    assert shadow["reason"] == "insufficient_data"
    assert data["last_error"] is None
    assert _n_shadow_requests(aioclient_mock) == 0
    assert entry.runtime_data.validation.shadow_set is None


async def test_shadow_cancel_propagates(
    hass: HomeAssistant,
    energy_manager: object,
    kettle_plug_prefs: dict[str, Any],
    inprocess_worker: object,
    aioclient_mock: object,
) -> None:
    """Cancelling during the shadow pass propagates CancelledError; nothing is stored."""
    import asyncio

    entry = await _setup_kettle_entry(hass, energy_manager, kettle_plug_prefs)
    second_call = asyncio.Event()

    async def _responder(request_data: Any) -> dict[str, Any]:
        if _has_plug_lines(request_data):
            digest = _digest_from_request(request_data)
            prop = proposal_for(digest)
            return {"status": 200, "json": chat_completion(json.dumps(prop))}
        second_call.set()
        await asyncio.Event().wait()
        return {"status": 500}

    mock_llm(aioclient_mock, "http://llm.test/v1", _responder)

    supervisor = entry.runtime_data.learning
    await supervisor.async_request_run("service")
    assert supervisor.run_task is not None
    await second_call.wait()
    supervisor.run_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await supervisor.run_task
    await hass.async_block_till_done()

    assert entry.runtime_data.validation.shadow_set is None
    last_run = supervisor.coordinator.data.get("last_run") or {}
    assert (last_run.get("shadow") or {}).get("status") != "failed"


async def test_empty_proposal_all_not_discovered(
    hass: HomeAssistant,
    energy_manager: object,
    kettle_plug_prefs: dict[str, Any],
    inprocess_worker: object,
    aioclient_mock: object,
) -> None:
    """An empty shadow proposal still stores a set with every plug not_discovered."""
    entry = await _setup_kettle_entry(hass, energy_manager, kettle_plug_prefs)
    mock_llm(
        aioclient_mock,
        "http://llm.test/v1",
        lambda r: {
            "status": 200,
            "json": chat_completion(
                json.dumps(proposal_for(_digest_from_request(r), devices=[]))
            ),
        },
    )

    supervisor = await _run_learn(entry)
    data = supervisor.coordinator.data
    assert data["status"] == "idle"
    shadow = data["last_run"]["shadow"]
    assert shadow["status"] == "ran"
    assert shadow["links"] == 0
    assert shadow["artefacts"] == 0

    stored = entry.runtime_data.validation.shadow_set
    assert stored is not None
    assert stored["artefacts"] == []
    assert stored["links"] == []
    assert {p["entity_id"] for p in stored["not_discovered"]} == {
        "sensor.kettle_power",
        "sensor.washer_power",
    }
    assert set(stored["plugs"]) == {"sensor.kettle_power", "sensor.washer_power"}


async def test_shadow_cap_keeps_production(
    hass: HomeAssistant,
    energy_manager: object,
    kettle_plug_prefs: dict[str, Any],
    inprocess_worker: object,
    aioclient_mock: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """D-21 shadow cap: the production result survives; the previous set stays."""
    from homeassistant.helpers import issue_registry as ir

    from custom_components.powerguesser.const import (
        CONF_RUN_INPUT_TOKENS,
        ISSUE_LEARNING_CAPPED,
    )

    entry = await _setup_kettle_entry(hass, energy_manager, kettle_plug_prefs)
    # Seed a *stale* previous set: the pass is due (so it reaches the cap
    # check), and the capped pass must leave the stored set unchanged.
    from datetime import timedelta

    import homeassistant.util.dt as dt_util

    previous = await _seed_kettle_shadow(
        hass,
        entry,
        created=(dt_util.utcnow() - timedelta(days=31)).isoformat(),
    )
    supervisor = entry.runtime_data.learning

    preview = await supervisor.async_preview_payload()
    r1_tokens = preview["estimated_input_tokens"]
    hass.config_entries.async_update_entry(
        entry,
        options={**entry.options, CONF_RUN_INPUT_TOKENS: r1_tokens + 200},
    )
    monkeypatch.setattr("custom_components.powerguesser.learning.MAX_REFINE_ROUNDS", 0)
    _production_nonempty_responder(aioclient_mock)

    supervisor = await _run_learn(entry)
    data = supervisor.coordinator.data
    assert data["status"] == "idle"
    assert len(supervisor.candidates) >= 1
    assert data["last_error"] is None
    shadow = data["last_run"]["shadow"]
    assert shadow["status"] == "skipped"
    assert shadow["reason"] == "cap_reached"
    # No capped status, no learning_capped Repairs issue.
    assert ir.async_get(hass).async_get_issue("powerguesser", ISSUE_LEARNING_CAPPED) is None
    # The previous shadow set is unchanged...
    assert entry.runtime_data.validation.shadow_set["run_id"] == previous["run_id"]
    # ...and the token counter holds the production pass's spend only (the
    # shadow request was refused before being sent).
    assert aioclient_mock.call_count == 1
    assert data["tokens_run"]["input"] >= 2500


async def test_shadow_failure_keeps_production(
    hass: HomeAssistant,
    energy_manager: object,
    kettle_plug_prefs: dict[str, Any],
    inprocess_worker: object,
    aioclient_mock: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A shadow_link worker failure records {status: failed, reason} and keeps production."""
    entry = await _setup_kettle_entry(hass, energy_manager, kettle_plug_prefs)
    _default_responder(aioclient_mock)
    supervisor = entry.runtime_data.learning
    _patch_stage_failure(supervisor, monkeypatch, "shadow_link")

    supervisor = await _run_learn(entry)
    data = supervisor.coordinator.data
    assert data["status"] == "idle"
    assert data["last_error"] is None
    shadow = data["last_run"]["shadow"]
    assert shadow["status"] == "failed"
    assert "shadow_link" in shadow["reason"]
    assert len(shadow["reason"]) <= 200


async def test_shadow_isolation_no_leakage(
    hass: HomeAssistant,
    energy_manager: object,
    kettle_plug_prefs: dict[str, Any],
    inprocess_worker: object,
    aioclient_mock: object,
) -> None:
    """D-02/T-05-06: with share-names on, shadow requests leak nothing."""
    from custom_components.powerguesser.const import CONF_SHARE_NAMES

    entry = await _setup_kettle_entry(hass, energy_manager, kettle_plug_prefs)
    hass.config_entries.async_update_entry(
        entry, options={**entry.options, CONF_SHARE_NAMES: True}
    )
    await async_seed_consent(hass, host="llm.test", names=True)
    _production_nonempty_responder(aioclient_mock)

    supervisor = await _run_learn(entry)
    assert supervisor.coordinator.data["status"] == "idle"
    stored = entry.runtime_data.validation.shadow_set
    assert stored is not None
    shadow_sigs = {a["signature_id"] for a in stored["artefacts"]}
    assert shadow_sigs

    shadow_bodies = []
    production_bodies = []
    for call in aioclient_mock.mock_calls:
        req = _request_json(call[2])
        (production_bodies if _has_plug_lines(call[2]) else shadow_bodies).append(req)

    assert shadow_bodies, "the shadow pass must send at least one request"
    # The share-names flag is really on: production requests carry the
    # <names> section naming the metered plugs.
    assert any("<names>" in json.dumps(req) for req in production_bodies)
    for req in shadow_bodies:
        text = json.dumps(req)
        # The shadow request never carries the names section, a plug id or
        # an entity id (D-02). The static category catalogue mentions the
        # word "kettle" on both sides and is not leakage.
        assert "<names>" not in text
        assert "sensor.kettle_power" not in text
        assert "sensor.washer_power" not in text

    # Isolation: no shadow signature id in the candidates Store, the review
    # decisions, any device record or the visible set (D-01, Pitfalls 1-2).
    assert {c["signature_id"] for c in supervisor.candidates}.isdisjoint(shadow_sigs)
    assert set(supervisor.review.decisions).isdisjoint(shadow_sigs)
    assert supervisor.hub.devices.owned_signature_ids().isdisjoint(shadow_sigs)
    visible = supervisor.visible_candidates()
    assert {c["signature_id"] for c in visible} == {
        c["signature_id"] for c in supervisor.candidates
    }
    assert {c["signature_id"] for c in visible}.isdisjoint(shadow_sigs)


@pytest.mark.parametrize("case", ["option_on", "option_off", "failed_run"])
async def test_after_learning_chain_runs_only_on_success(
    hass: HomeAssistant,
    energy_manager: object,
    kettle_plug_prefs: dict[str, Any],
    inprocess_worker: object,
    aioclient_mock: object,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
) -> None:
    """D-13: with the option on, an idle run chains exactly one after_learning validation."""
    from datetime import datetime

    from custom_components.powerguesser.const import CONF_VALIDATE_AFTER_LEARNING

    entry = await _setup_kettle_entry(hass, energy_manager, kettle_plug_prefs)
    await _seed_kettle_shadow(hass, entry)
    validation = entry.runtime_data.validation

    if case in ("option_on", "failed_run"):
        hass.config_entries.async_update_entry(
            entry,
            options={**entry.options, CONF_VALIDATE_AFTER_LEARNING: True},
        )

    if case == "failed_run":
        _default_responder(aioclient_mock)
        supervisor = entry.runtime_data.learning
        _patch_stage_failure(supervisor, monkeypatch, "extract")
    else:
        _default_responder(aioclient_mock)

    supervisor = await _run_learn(entry)
    report = validation.latest_report

    if case == "option_on":
        assert supervisor.coordinator.data["status"] == "idle"
        assert report is not None
        assert report["trigger"] == "after_learning"
        ended = datetime.fromisoformat(
            supervisor.coordinator.data["last_run"]["ended"]
        )
        generated = datetime.fromisoformat(report["generated_at"])
        assert generated >= ended
        assert report["status"] == "ok"
    else:
        # Option off, or the run failed: no validation is chained.
        assert report is None
    if case == "failed_run":
        assert supervisor.coordinator.data["status"] == "failed"

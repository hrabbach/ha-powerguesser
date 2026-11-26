"""End-to-end reconciliation of capture previews with learned candidates (CAP-09, D-07).

Plan 08-08 Task 2: a later learning run that finds the user's captured load merges
into the preview — one candidate, the user's name, the user's decisions — for every
interleaving of capture-add and run-complete. The shadow pass carries no capture
data.
"""

import copy
import json
from collections import Counter
from pathlib import Path
import re
from typing import Any

import pytest

from homeassistant.components.energy.data import EnergyManager
from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util

from custom_components.powerguesser.const import (
    BUFFER_DIR,
    DOMAIN,
    LEARNING_DIR,
)
from custom_components.powerguesser.engine.capture.constants import CONFIRM_MIN_RUNS
from custom_components.powerguesser.engine.intervals import ProfileSnapshot, gate
from custom_components.powerguesser.review_store import resolve_session_target
from tests.conftest import (
    async_fill_synth_buffer,
    async_seed_consent,
    async_seed_llm,
    async_set_power,
    async_setup_via_flow,
    mock_llm,
)
from tests.fixtures.learning.fake_llm import chat_completion, kettle_proposal
from tests.fixtures.learning.synth import ENTITY_KETTLE, SynthSpec

GRID_S = 30
_CAPTURE_LABEL = "capture-marker-xyz"

_CLUSTER_RE = re.compile(
    r"id=(?P<id>c\d+)\s+kind=(?P<kind>\w+)\s+n=(?P<n>\d+)\s+"
    r"rise=(?P<r10>-?[\d.]+)/(?P<r50>-?[\d.]+)/(?P<r90>-?[\d.]+)\s+"
    r"fall=(?P<f50>-?[\d.]+)\s+dur_min=(?P<d10>[\d.]+)-(?P<d90>[\d.]+)"
)


def _digest_from_request(request_data: Any) -> dict[str, Any]:
    """Rebuild a digest-shaped dict from the rendered digest in an LLM request.

    Same helper as tests/ha/test_device_identity.py: the rendered cluster lines
    carry dur_min as p10-p90 only; the midpoint stands in for p50.
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


def _pin_grid(entry: Any) -> None:
    """Publish a 30 s profile snapshot so the capture snap uses a 30 s grid."""
    enabled, disabled = gate(float(GRID_S), False)
    entry.runtime_data.profile_coordinator.async_set_updated_data(
        ProfileSnapshot(
            sources={},
            effective_s=float(GRID_S),
            assumed=False,
            enabled=enabled,
            disabled=disabled,
        )
    )


def _iso_utc(ms: int) -> str:
    return dt_util.utc_from_timestamp(ms / 1000.0).isoformat()


def _set_std_powers(hass: HomeAssistant) -> None:
    async_set_power(hass, "sensor.grid_power", 500)
    async_set_power(hass, "sensor.solar_power", 0)
    async_set_power(hass, "sensor.battery_power", 0)
    async_set_power(hass, "sensor.washer_power", 0)
    async_set_power(hass, ENTITY_KETTLE, 0)


async def _setup(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    *,
    spec: SynthSpec | None = None,
) -> tuple[Any, Any]:
    """Entry with a synth buffer, a pinned grid, LLM seeded and refine off."""
    monkeypatch.setattr("custom_components.powerguesser.learning.MAX_REFINE_ROUNDS", 0)
    energy_manager.data = copy.deepcopy(std_prefs)
    _set_std_powers(hass)
    entry = await async_setup_via_flow(hass)
    _pin_grid(entry)
    await hass.async_block_till_done()
    data = await async_fill_synth_buffer(hass, entry, spec or SynthSpec(days=22))
    await async_seed_llm(hass, entry)
    await async_seed_consent(hass, host="llm.test", names=False)
    return entry, data


async def _capture(
    hass: HomeAssistant,
    *,
    start_ms: int,
    end_ms: int,
    label: str = "kettle full",
    target_type: str = "new_device",
    target: str = "Kettle",
) -> dict[str, Any]:
    return await hass.services.async_call(
        DOMAIN,
        "capture_window",
        {
            "start": _iso_utc(start_ms),
            "end": _iso_utc(end_ms),
            "label": label,
            "target_type": target_type,
            "target": target,
        },
        blocking=True,
        return_response=True,
    )


async def _capture_clean_kettle(
    hass: HomeAssistant,
    data: Any,
    *,
    label: str = "kettle full",
    target_type: str = "new_device",
    target: str = "Kettle",
    max_tries: int = 8,
) -> dict[str, Any]:
    """Capture a kettle run whose snap is clean, deleting non-clean attempts.

    A few 1 s kettle rises land between two grid samples and smear across three
    (unpairable) samples; the buffer is deterministic, so scanning from the last
    run backwards always finds a clean one within a few tries.
    """
    for on_ms, off_ms in reversed(data.kettle_runs):
        if max_tries <= 0:
            break
        max_tries -= 1
        res = await _capture(
            hass,
            start_ms=on_ms - 45_000,
            end_ms=off_ms + 50_000,
            label=label,
            target_type=target_type,
            target=target,
        )
        if res["result"]["verdict"] == "clean":
            return res
        await hass.services.async_call(
            DOMAIN,
            "capture_delete",
            {"capture_id": res["capture_id"]},
            blocking=True,
            return_response=True,
        )
    raise AssertionError("no clean kettle run in the synthetic buffer")


async def _capture_list(hass: HomeAssistant) -> dict[str, Any]:
    return await hass.services.async_call(
        DOMAIN, "capture_list", {}, blocking=True, return_response=True
    )


def _kettle_responder(aioclient_mock: Any) -> None:
    """Register an LLM responder that always proposes the kettle cluster."""

    def _responder(data: Any) -> dict[str, Any]:
        prop = kettle_proposal(_digest_from_request(data))
        return {"status": 200, "json": chat_completion(json.dumps(prop))}

    mock_llm(aioclient_mock, "http://llm.test/v1", _responder)


async def _run_learning(entry: Any) -> Any:
    supervisor = entry.runtime_data.learning
    await supervisor.async_request_run("service")
    assert supervisor.run_task is not None
    await supervisor.run_task
    return supervisor


def _visible_ids(supervisor: Any) -> set[str]:
    return {str(c.get("signature_id")) for c in supervisor.visible_candidates()}


def _assert_reconcile_invariant(hub: Any) -> None:
    """Every live clean capture-origin group has exactly one visible candidate."""
    decisions = hub.learning.review.decisions
    live_cap_ids: set[str] = set()
    for rec in hub.captures.records:
        if (rec.get("result") or {}).get("verdict") != "clean":
            continue
        resolved = resolve_session_target(rec, decisions, hub.captures._device_lookup)
        if (
            resolved is not None
            and resolved[0] == "candidate"
            and str(resolved[1]).startswith("cap_")
        ):
            live_cap_ids.add(str(resolved[1]))
    counts = Counter(str(c.get("signature_id")) for c in hub.learning.visible_candidates())
    for cid in live_cap_ids:
        assert counts[cid] == 1, (cid, dict(counts))


def _run_stage_dir(hass: HomeAssistant, run_id: str) -> Path:
    return Path(hass.config.path(BUFFER_DIR, LEARNING_DIR)) / run_id


async def test_capture_and_learning_run_reconcile_to_one_candidate(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    inprocess_worker: None,
    aioclient_mock: Any,
) -> None:
    """SC5: a later run that finds the captured kettle shows one candidate, not two."""
    entry, data = await _setup(hass, energy_manager, std_prefs, monkeypatch)
    hub = entry.runtime_data

    cap = await _capture_clean_kettle(hass, data)
    preview_id = cap["target"]["id"]
    assert preview_id.startswith("cap_")

    _kettle_responder(aioclient_mock)
    supervisor = await _run_learning(entry)
    _assert_reconcile_invariant(hub)

    visible = supervisor.visible_candidates()
    assert len(visible) == 1
    cand = visible[0]
    assert cand["signature_id"] == preview_id
    # The user's name wins: the card carries the decision name.
    assert supervisor.candidate_label(cand).startswith("Kettle")
    dec = supervisor.review.decisions[preview_id]
    assert dec["status"] == "named"
    assert dec["name"] == "Kettle"

    caps = await _capture_list(hass)
    assert caps["captures"][0]["target"]["id"] == preview_id


async def test_reconcile_invariant_over_interleavings(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    inprocess_worker: None,
    aioclient_mock: Any,
) -> None:
    """CAP-09 concurrency: run first, then a capture (two cards), then a second run."""
    entry, data = await _setup(hass, energy_manager, std_prefs, monkeypatch)
    hub = entry.runtime_data
    _kettle_responder(aioclient_mock)

    supervisor = await _run_learning(entry)
    _assert_reconcile_invariant(hub)
    learned_ids = _visible_ids(supervisor)
    assert len(learned_ids) == 1

    cap = await _capture_clean_kettle(hass, data)
    cap_id = cap["target"]["id"]
    _assert_reconcile_invariant(hub)
    # Two cards between the capture and the next run.
    assert _visible_ids(supervisor) == learned_ids | {cap_id}

    supervisor = await _run_learning(entry)
    _assert_reconcile_invariant(hub)
    assert _visible_ids(supervisor) == {cap_id}


async def test_capture_attached_to_learned_candidate_reconciles(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    inprocess_worker: None,
    aioclient_mock: Any,
) -> None:
    """CAP-09 concurrency: a capture attached to the learned candidate keeps its id."""
    entry, data = await _setup(hass, energy_manager, std_prefs, monkeypatch)
    hub = entry.runtime_data
    _kettle_responder(aioclient_mock)

    supervisor = await _run_learning(entry)
    learned_ids = _visible_ids(supervisor)
    assert len(learned_ids) == 1
    learned_id = next(iter(learned_ids))
    assert not learned_id.startswith("cap_")

    res = await _capture_clean_kettle(
        hass, data, target_type="candidate", target=learned_id
    )
    assert res["target"]["id"] == learned_id
    _assert_reconcile_invariant(hub)

    supervisor = await _run_learning(entry)
    _assert_reconcile_invariant(hub)
    assert _visible_ids(supervisor) == {learned_id}


async def test_repeat_run_is_idempotent(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    inprocess_worker: None,
    aioclient_mock: Any,
) -> None:
    """CAP-09 idempotency: a second run with no new capture leaves Review unchanged."""
    entry, data = await _setup(hass, energy_manager, std_prefs, monkeypatch)
    hub = entry.runtime_data
    await _capture_clean_kettle(hass, data)

    _kettle_responder(aioclient_mock)
    supervisor = await _run_learning(entry)
    ids_1 = _visible_ids(supervisor)
    decisions_1 = copy.deepcopy(dict(supervisor.review.decisions))
    targets_1 = [c["target"]["id"] for c in (await _capture_list(hass))["captures"]]

    supervisor = await _run_learning(entry)
    _assert_reconcile_invariant(hub)
    assert _visible_ids(supervisor) == ids_1
    assert dict(supervisor.review.decisions) == decisions_1
    assert [
        c["target"]["id"] for c in (await _capture_list(hass))["captures"]
    ] == targets_1
    cand = supervisor.visible_candidates()[0]
    assert supervisor.candidate_label(cand).startswith("Kettle")


async def test_one_capture_preview_reconciles_despite_gate(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    inprocess_worker: None,
    aioclient_mock: Any,
) -> None:
    """D-03: a one-capture preview (below CONFIRM_MIN_RUNS) still reconciles."""
    entry, data = await _setup(hass, energy_manager, std_prefs, monkeypatch)
    hub = entry.runtime_data
    cap = await _capture_clean_kettle(hass, data)
    cap_id = cap["target"]["id"]

    group = hub.captures._preview_group(cap_id)
    clean = [r for r in group if (r.get("result") or {}).get("verdict") == "clean"]
    assert len(clean) == 1 < CONFIRM_MIN_RUNS

    _kettle_responder(aioclient_mock)
    supervisor = await _run_learning(entry)
    _assert_reconcile_invariant(hub)
    assert _visible_ids(supervisor) == {cap_id}


async def test_dismissed_shape_capture_reconciles_not_suppressed(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    inprocess_worker: None,
    aioclient_mock: Any,
) -> None:
    """D-07: a capture of a shape the user dismissed reconciles instead of being
    twin-suppressed by the dismissed learned candidate."""
    entry, data = await _setup(hass, energy_manager, std_prefs, monkeypatch)
    hub = entry.runtime_data
    _kettle_responder(aioclient_mock)

    supervisor = await _run_learning(entry)
    learned = supervisor.visible_candidates()
    assert len(learned) == 1
    learned_id = learned[0]["signature_id"]
    await supervisor.review.async_dismiss(
        learned_id, learned[0].get("fingerprint"), label="Kettle"
    )

    cap = await _capture_clean_kettle(hass, data)
    cap_id = cap["target"]["id"]

    supervisor = await _run_learning(entry)
    visible = supervisor.visible_candidates()
    ids = {c["signature_id"] for c in visible}
    assert ids == {cap_id}
    cand = visible[0]
    assert (cand.get("identity") or {}).get("status") != "suppressed"


async def test_shadow_pass_carries_no_capture_data(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    inprocess_worker: None,
    aioclient_mock: Any,
) -> None:
    """T-08-25: no capture anchor or prior reaches the shadow job or an LLM request."""
    entry, data = await _setup(
        hass, energy_manager, std_prefs, monkeypatch, spec=SynthSpec(days=35)
    )
    cap = await _capture_clean_kettle(hass, data, label=_CAPTURE_LABEL)
    cap_id = cap["target"]["id"]

    _kettle_responder(aioclient_mock)
    supervisor = await _run_learning(entry)
    last_run = supervisor.coordinator.data["last_run"]
    assert last_run["shadow"]["status"] == "ran"

    shadow_dir = _run_stage_dir(hass, str(last_run["run_id"])) / "shadow"
    job = json.loads((shadow_dir / "job_compile_1.json").read_text(encoding="utf-8"))
    assert "capture_anchors" not in job
    assert "capture_prior_ids" not in job

    for call in aioclient_mock.mock_calls:
        body = call[2]
        if isinstance(body, (bytes, bytearray)):
            body = body.decode("utf-8", "ignore")
        elif not isinstance(body, str):
            body = json.dumps(body)
        assert cap_id not in body
        assert _CAPTURE_LABEL not in body

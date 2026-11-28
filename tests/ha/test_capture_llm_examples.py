"""Captures as labelled LLM examples: preview == sent, labels, trim, verdicts (CAP-10).

Plan 09-04: every done capture with a snapped pair rides the next learning run as
one example line through one round-1 builder shared by the preview action, the
consent prepare and the run's round 1 (D-06). Labels appear only under the names
opt-in (D-05); trimming is visible in the payload, the preview response and the
status sensor (D-08); the LLM's optional per-example verdicts are stored and shown
but never change a candidate, a capture preview or a device (D-07).
"""

import copy
import json
from pathlib import Path
import re
from typing import Any

import pytest

from homeassistant.components.energy.data import EnergyManager
from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util

from custom_components.powerguesser.capture import UNNAMED_CAPTURE_NAME
from custom_components.powerguesser.const import (
    BUFFER_DIR,
    CONF_MAX_TOKENS,
    CONF_RUN_INPUT_TOKENS,
    CONF_SHARE_NAMES,
    CONF_TRUNCATION_GUARD,
    DEFAULT_MAX_TOKENS,
    DOMAIN,
    LEARNING_DIR,
    SERVICE_PREVIEW,
)
from custom_components.powerguesser.engine.intervals import ProfileSnapshot, gate
from custom_components.powerguesser.engine.learning import prompts
from custom_components.powerguesser.engine.learning.examples import select_examples
from custom_components.powerguesser.engine.learning.prompts import payload_bytes
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
_CLUSTER_RE = re.compile(
    r"id=(?P<id>c\d+)\s+kind=(?P<kind>\w+)\s+n=(?P<n>\d+)\s+"
    r"rise=(?P<r10>-?[\d.]+)/(?P<r50>-?[\d.]+)/(?P<r90>-?[\d.]+)\s+"
    r"fall=(?P<f50>-?[\d.]+)\s+dur_min=(?P<d10>[\d.]+)-(?P<d90>[\d.]+)"
)
_EXAMPLE_LINE_RE = re.compile(r"(?m)^(e\d+) ")


def _req_json(request_data: Any) -> dict[str, Any]:
    if isinstance(request_data, (bytes, bytearray)):
        request_data = request_data.decode("utf-8")
    return json.loads(request_data) if isinstance(request_data, str) else request_data


def _user_content(body: dict[str, Any]) -> str:
    return str(body["messages"][-1]["content"])


def _canonical(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _digest_from_request(request_data: Any) -> dict[str, Any]:
    """Rebuild a digest-shaped dict from the rendered digest in an LLM request.

    Same helper as tests/ha/test_capture_reconcile.py: the rendered cluster lines
    carry dur_min as p10-p90 only; the midpoint stands in for p50.
    """
    request_data = _req_json(request_data)
    clusters: list[dict[str, Any]] = []
    for m in request_data.get("messages", []):
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
    names: bool = False,
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
    if names:
        hass.config_entries.async_update_entry(
            entry, options={**entry.options, CONF_SHARE_NAMES: True}
        )
        await hass.async_block_till_done()
    await async_seed_consent(hass, host="llm.test", names=names)
    return entry, data


async def _capture(
    hass: HomeAssistant,
    *,
    start_ms: int,
    end_ms: int,
    label: str = "kettle run",
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
    label: str = "kettle run",
    target_type: str = "new_device",
    target: str = "Kettle",
    max_tries: int = 8,
) -> dict[str, Any]:
    """Capture a kettle run whose snap is clean, deleting non-clean attempts."""
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


async def _preview(hass: HomeAssistant) -> dict[str, Any]:
    return await hass.services.async_call(
        DOMAIN, SERVICE_PREVIEW, blocking=True, return_response=True
    )


async def _run_learning(entry: Any) -> Any:
    supervisor = entry.runtime_data.learning
    await supervisor.async_request_run("service")
    assert supervisor.run_task is not None
    await supervisor.run_task
    return supervisor


def _stage_dir(hass: HomeAssistant, run_id: str) -> Path:
    return Path(hass.config.path(BUFFER_DIR, LEARNING_DIR)) / run_id


def _sent_body(hass: HomeAssistant, supervisor: Any) -> dict[str, Any]:
    run_id = str(supervisor.coordinator.data["last_run"]["run_id"])
    return json.loads((_stage_dir(hass, run_id) / "request_1.json").read_text("utf-8"))


def _kettle_responder(aioclient_mock: Any) -> None:
    """Register an LLM responder that always proposes the kettle cluster."""

    def _responder(data: Any) -> dict[str, Any]:
        prop = kettle_proposal(_digest_from_request(data))
        return {"status": 200, "json": chat_completion(json.dumps(prop))}

    mock_llm(aioclient_mock, "http://llm.test/v1", _responder)


def _example_records(
    capture_id: str,
    state: str,
    *,
    target_kind: str = "candidate",
    target_id: str = "sig_ghost",
    label: str = "",
    new_device_name: str | None = None,
    rise_ms: int | None = 1000,
    verdict: str = "none",
) -> dict[str, Any]:
    """A minimal capture record for example_rows() unit assertions."""
    if rise_ms is None:
        result: dict[str, Any] | None = None
        features: dict[str, Any] | None = None
    else:
        result = {
            "rise_ms": rise_ms,
            "fall_ms": rise_ms + 100_000,
            "step_w": 2000.0,
            "fall_w": -1995.0,
            "dur_s": 180.0,
            "energy_wh": 100.0,
            "verdict": verdict,
        }
        features = {
            "rise_w": 2000.0,
            "fall_w": -1995.0,
            "dur_s": 180.0,
            "plateau_w": 2000.0,
            "energy_wh": 100.0,
            "hour": 7,
            "dow": 2,
            "l1_share": 1.0,
            "l2_share": 0.0,
            "l3_share": 0.0,
        }
    return {
        "capture_id": capture_id,
        "mode": "window",
        "state": state,
        "target_kind": target_kind,
        "target_id": target_id,
        "new_device_name": new_device_name,
        "label": label,
        "press_start_ts": None,
        "press_stop_ts": None,
        "window_start_ms": None,
        "window_end_ms": None,
        "created_at": "2026-11-26T06:00:00+00:00",
        "resolved_at": "2026-11-26T07:00:00+00:00",
        "reason_code": None,
        "result": result,
        "features": features,
        "trace": None,
        "holdout_excluded": True,
        "holdout_window_ms": None,
        "replay": None,
        "earlier_runs": None,
    }


# --- Task 1: one round-1 builder, labels, trim, status ---------------------------


@pytest.mark.parametrize("names", [False, True])
async def test_preview_body_equals_sent_request(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    inprocess_worker: None,
    aioclient_mock: Any,
    names: bool,
) -> None:
    """The preview body is byte-identical to the run's request_1.json (D-06)."""
    entry, data = await _setup(hass, energy_manager, std_prefs, monkeypatch, names=names)
    await _capture_clean_kettle(hass, data, label="kettle alpha")
    await _capture_clean_kettle(hass, data, label="kettle beta")

    preview = await _preview(hass)
    assert preview["captures_left_out"] == 0
    assert "<examples>" in _user_content(preview["body"])

    _kettle_responder(aioclient_mock)
    supervisor = await _run_learning(entry)
    sent = _sent_body(hass, supervisor)

    assert preview["body"] == sent
    assert _canonical(preview["body"]) == _canonical(sent)
    # The consent notice's byte estimate is the same body's payload bytes.
    assert preview["bytes"] == payload_bytes(preview["body"])


@pytest.mark.parametrize("names", [False, True])
async def test_no_label_without_names_opt_in(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    inprocess_worker: None,
    aioclient_mock: Any,
    names: bool,
) -> None:
    """A label reaches the LLM only inside <example_names>, only with the opt-in (D-05)."""
    entry, data = await _setup(hass, energy_manager, std_prefs, monkeypatch, names=names)
    marker = "kettle-marker-secret"
    await _capture_clean_kettle(hass, data, label=marker)

    preview = await _preview(hass)
    _kettle_responder(aioclient_mock)
    supervisor = await _run_learning(entry)
    sent = _sent_body(hass, supervisor)

    for body in (preview["body"], sent):
        content = _user_content(body)
        if not names:
            assert marker not in content
            assert "<example_names>" not in content
        else:
            assert "<example_names>" in content
            assert marker in content
            block = re.search(r"<example_names>.*?</example_names>", content, re.S)
            assert block is not None
            outside = content[: block.start()] + content[block.end() :]
            assert marker not in outside


async def test_payload_has_no_timestamps_or_capture_ids(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    inprocess_worker: None,
    aioclient_mock: Any,
) -> None:
    """No capture id, ISO date or epoch ms from a record appears in the body (T-09-25)."""
    entry, data = await _setup(hass, energy_manager, std_prefs, monkeypatch)
    await _capture_clean_kettle(hass, data, label="kettle run")

    preview = await _preview(hass)
    _kettle_responder(aioclient_mock)
    supervisor = await _run_learning(entry)
    sent = _sent_body(hass, supervisor)

    records = entry.runtime_data.captures.records
    assert records
    for body in (preview["body"], sent):
        serialized = json.dumps(body)
        for rec in records:
            assert rec["capture_id"] not in serialized
            resolved_at = rec["resolved_at"]
            assert resolved_at not in serialized
            ms = int(dt_util.parse_datetime(resolved_at).timestamp() * 1000)
            assert str(ms) not in serialized


async def test_capture_free_run_body_unchanged(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    inprocess_worker: None,
    aioclient_mock: Any,
) -> None:
    """With no captures the body equals prompts.build_request and left-out is 0."""
    entry, data = await _setup(hass, energy_manager, std_prefs, monkeypatch)

    preview = await _preview(hass)
    assert preview["captures_left_out"] == 0

    _kettle_responder(aioclient_mock)
    supervisor = await _run_learning(entry)
    sent = _sent_body(hass, supervisor)

    run_id = str(supervisor.coordinator.data["last_run"]["run_id"])
    digest = json.loads((_stage_dir(hass, run_id) / "digest.json").read_text("utf-8"))
    expected = prompts.build_request(
        digest,
        model=str(entry.data["llm_model"]),
        profile=entry.data["llm_probe"],
        max_tokens=int(entry.options.get(CONF_MAX_TOKENS, DEFAULT_MAX_TOKENS)),
        names=None,
    )
    assert sent == expected
    assert "<examples>" not in _user_content(sent)


async def test_tiny_run_cap_trims_and_reports_left_out(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    inprocess_worker: None,
    aioclient_mock: Any,
) -> None:
    """A tight run cap leaves captures out and says so in payload, preview and status (D-08)."""
    entry, data = await _setup(hass, energy_manager, std_prefs, monkeypatch)
    hass.config_entries.async_update_entry(
        entry, options={**entry.options, CONF_TRUNCATION_GUARD: False}
    )
    await hass.async_block_till_done()

    for i in range(3):
        await _capture_clean_kettle(hass, data, label=f"kettle {i}")
    hub = entry.runtime_data
    assert len(hub.captures.example_rows()) == 3

    chosen_cap: int | None = None
    preview: dict[str, Any] | None = None
    for cap in range(2400, 5201, 20):
        hass.config_entries.async_update_entry(
            entry, options={**entry.options, CONF_RUN_INPUT_TOKENS: cap}
        )
        candidate = await _preview(hass)
        if candidate["captures_left_out"] == 2:
            chosen_cap = cap
            preview = candidate
    assert chosen_cap is not None, "no run-input cap leaves exactly two of three captures out"
    assert preview is not None
    hass.config_entries.async_update_entry(
        entry, options={**entry.options, CONF_RUN_INPUT_TOKENS: chosen_cap}
    )
    await hass.async_block_till_done()

    _kettle_responder(aioclient_mock)
    supervisor = await _run_learning(entry)
    assert supervisor.coordinator.data["status"] == "idle"
    sent = _sent_body(hass, supervisor)
    content = _user_content(sent)
    assert "examples_left_out n=2" in content
    assert len(_EXAMPLE_LINE_RE.findall(content)) == 1
    assert preview["captures_left_out"] == 2

    state = hass.states.get("sensor.powerguesser_learning_status")
    assert state is not None
    assert state.attributes["captures_left_out_this_run"] == 2

    # A cap below the capture-free body still ends capped, not failed.
    hass.config_entries.async_update_entry(
        entry, options={**entry.options, CONF_RUN_INPUT_TOKENS: 50}
    )
    await hass.async_block_till_done()
    await _run_learning(entry)
    assert entry.runtime_data.learning.coordinator.data["status"] == "capped"


async def test_shadow_pass_sends_no_examples(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    inprocess_worker: None,
    aioclient_mock: Any,
) -> None:
    """The masked validation pass carries no examples; production round 1 does."""
    entry, data = await _setup(
        hass, energy_manager, std_prefs, monkeypatch, spec=SynthSpec(days=35)
    )
    await _capture_clean_kettle(hass, data, label="kettle run")

    _kettle_responder(aioclient_mock)
    supervisor = await _run_learning(entry)
    last_run = supervisor.coordinator.data["last_run"]
    assert last_run["shadow"]["status"] == "ran"

    stage_dir = _stage_dir(hass, str(last_run["run_id"]))
    production = json.loads((stage_dir / "request_1.json").read_text("utf-8"))
    assert "<examples>" in _user_content(production)

    shadow = json.loads((stage_dir / "shadow" / "request_1.json").read_text("utf-8"))
    assert "<examples>" not in _user_content(shadow)


async def test_example_label_is_user_supplied_only(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    inprocess_worker: None,
    aioclient_mock: Any,
) -> None:
    """A placeholder or id typed to pick a target is dropped, never sent as a label (D-05)."""
    entry, data = await _setup(hass, energy_manager, std_prefs, monkeypatch, names=True)
    hub = entry.runtime_data
    sig = "sig_kettle_candidate"
    device_uuid = "device-uuid-1234"
    preview_id = "cap_deadbeefcafe"
    monkeypatch.setattr(
        hub.captures,
        "_device_lookup",
        lambda uuid: (
            {"uuid": uuid, "name": "Kettle Device", "status": "confirmed"}
            if uuid == device_uuid
            else None
        ),
    )
    hub.captures._records.extend(
        [
            _example_records(
                "cap_a", "done", target_kind="candidate", target_id=preview_id, label=""
            ),
            _example_records(
                "cap_b", "done", target_kind="candidate", target_id=sig, label=sig
            ),
            _example_records(
                "cap_c", "done", target_kind="device", target_id=device_uuid, label=""
            ),
            _example_records(
                "cap_d",
                "done",
                target_kind="candidate",
                target_id="cap_other",
                label=UNNAMED_CAPTURE_NAME,
            ),
        ]
    )

    rows = {row["capture_id"]: row for row in hub.captures.example_rows()}
    assert rows["cap_a"]["label"] is None
    assert rows["cap_a"]["target_name"] is None
    assert rows["cap_b"]["label"] is None
    assert rows["cap_b"]["target_name"] is None
    assert rows["cap_c"]["label"] is None
    assert rows["cap_c"]["target_name"] == "Kettle Device"
    assert rows["cap_d"]["label"] is None

    # A Review rename of the candidate names example (b) without reviving the id label.
    hub.learning.review.decisions[sig] = {"status": "named", "name": "Kettle"}
    rows2 = {row["capture_id"]: row for row in hub.captures.example_rows()}
    assert rows2["cap_b"]["label"] is None
    assert rows2["cap_b"]["target_name"] == "Kettle"

    rendered = select_examples(
        hub.captures.example_rows(),
        digest={},
        names_flag=True,
        max_bytes=12288,
        max_count=40,
    ).text
    for forbidden in (UNNAMED_CAPTURE_NAME, "Unknown", preview_id, sig, device_uuid):
        assert forbidden not in rendered
    assert "Kettle Device" in rendered
    assert "Kettle" in rendered

    # And the same through the real preview body.
    preview = await _preview(hass)
    content = _user_content(preview["body"])
    for forbidden in (UNNAMED_CAPTURE_NAME, "Unknown", preview_id, sig, device_uuid):
        assert forbidden not in content
    assert "Kettle Device" in content


async def test_only_done_paired_captures_are_examples(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    inprocess_worker: None,
    aioclient_mock: Any,
) -> None:
    """Only done captures with a snapped pair are examples; nothing else counts (CAP-10)."""
    entry, data = await _setup(hass, energy_manager, std_prefs, monkeypatch)
    hub = entry.runtime_data
    hub.captures._records.extend(
        [
            _example_records("c_failed", "failed", rise_ms=None),
            _example_records("c_cancelled", "cancelled", rise_ms=1000),
            _example_records("c_interrupted", "interrupted", rise_ms=1000),
            _example_records("c_done_nopair", "done", rise_ms=None),
        ]
    )

    assert hub.captures.example_rows() == []
    preview = await _preview(hass)
    assert preview["captures_left_out"] == 0
    assert "<examples>" not in _user_content(preview["body"])

    cap = await _capture_clean_kettle(hass, data, label="kettle run")
    rows = hub.captures.example_rows()
    assert [row["capture_id"] for row in rows] == [cap["capture_id"]]

    _kettle_responder(aioclient_mock)
    supervisor = await _run_learning(entry)
    content = _user_content(_sent_body(hass, supervisor))
    assert len(_EXAMPLE_LINE_RE.findall(content)) == 1
    assert "c_failed" not in content
    assert "c_cancelled" not in content
    assert "c_interrupted" not in content
    assert "c_done_nopair" not in content


# --- Task 2: stored, shown, harmless per-example verdicts (D-07) ------------------


async def _capture_list(hass: HomeAssistant) -> dict[str, Any]:
    return await hass.services.async_call(
        DOMAIN, "capture_list", {}, blocking=True, return_response=True
    )


def _verdict_responder(aioclient_mock: Any, verdicts: tuple[str, ...]) -> None:
    """A kettle proposal whose reply answers the first len(verdicts) example ids."""

    def _responder(request_data: Any) -> dict[str, Any]:
        req = _req_json(request_data)
        content = "".join(
            m.get("content", "")
            for m in req.get("messages", [])
            if isinstance(m.get("content"), str)
        )
        eids = _EXAMPLE_LINE_RE.findall(content)
        prop = kettle_proposal(_digest_from_request(req))
        entries = [
            {"example": eids[i], "verdict": verdict}
            for i, verdict in enumerate(verdicts)
            if i < len(eids)
        ]
        if entries:
            prop["example_verdicts"] = entries
        return {"status": 200, "json": chat_completion(json.dumps(prop))}

    mock_llm(aioclient_mock, "http://llm.test/v1", _responder)


def _candidate_signature(supervisor: Any) -> list[tuple[str, str, float]]:
    return sorted(
        (
            str(c.get("signature_id")),
            json.dumps(c.get("fingerprint"), sort_keys=True),
            float((c.get("scores") or {}).get("train", {}).get("share") or 0.0),
        )
        for c in supervisor.candidates
    )


async def test_verdicts_mapped_to_captures(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    inprocess_worker: None,
    aioclient_mock: Any,
) -> None:
    """The reply's verdicts land on the right captures and never reach the compile job (D-07)."""
    entry, data = await _setup(hass, energy_manager, std_prefs, monkeypatch)
    captures = []
    for label in ("kettle one", "kettle two", "kettle three"):
        captures.append(await _capture_clean_kettle(hass, data, label=label))
    hub = entry.runtime_data
    assert len(hub.captures.example_rows()) == 3

    _verdict_responder(aioclient_mock, ("consistent", "inconsistent"))
    supervisor = await _run_learning(entry)
    assert supervisor.coordinator.data["status"] == "idle"

    recs = {r["capture_id"]: r for r in hub.captures.records}
    ordered = sorted(
        (c["capture_id"] for c in captures),
        key=lambda cid: recs[cid]["resolved_at"],
        reverse=True,
    )
    by_id = {
        c["capture_id"]: c for c in (await _capture_list(hass))["captures"]
    }
    # e1 (newest) -> consistent, e2 -> inconsistent, e3 unanswered -> null.
    assert by_id[ordered[0]]["llm_verdict"] == "consistent"
    assert by_id[ordered[1]]["llm_verdict"] == "inconsistent"
    assert by_id[ordered[2]]["llm_verdict"] is None

    run_id = str(supervisor.coordinator.data["last_run"]["run_id"])
    proposal = json.loads(
        (_stage_dir(hass, run_id) / "proposal_1.json").read_text("utf-8")
    )
    assert "example_verdicts" not in proposal


async def test_rejected_refinement_round_keeps_accepted_verdicts(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    inprocess_worker: None,
    aioclient_mock: Any,
) -> None:
    """A rejected refinement round cannot replace the accepted round's verdicts (WR-01).

    Round 1 answers every example ``consistent``; the refinement round answers the
    same example ids ``inconsistent`` but proposes the same library, so it is
    rejected and its verdicts must never be persisted or shown on ``capture_list``.
    """
    entry, data = await _setup(hass, energy_manager, std_prefs, monkeypatch)
    monkeypatch.setattr("custom_components.powerguesser.learning.MAX_REFINE_ROUNDS", 1)
    captures = []
    for label in ("kettle one", "kettle two", "kettle three"):
        captures.append(await _capture_clean_kettle(hass, data, label=label))
    hub = entry.runtime_data

    calls: list[int] = []

    def _responder(request_data: Any) -> dict[str, Any]:
        req = _req_json(request_data)
        prop = kettle_proposal(_digest_from_request(req))
        content = "".join(
            m.get("content", "")
            for m in req.get("messages", [])
            if isinstance(m.get("content"), str)
        )
        eids = _EXAMPLE_LINE_RE.findall(content)
        calls.append(len(calls) + 1)
        # Round 1: every example consistent; refinement round: every example
        # inconsistent. The refinement proposes the identical kettle library, so
        # round acceptance rejects it.
        verdict = "consistent" if len(calls) == 1 else "inconsistent"
        entries = [{"example": eid, "verdict": verdict} for eid in eids]
        if entries:
            prop["example_verdicts"] = entries
        return {"status": 200, "json": chat_completion(json.dumps(prop))}

    mock_llm(aioclient_mock, "http://llm.test/v1", _responder)
    supervisor = await _run_learning(entry)
    assert supervisor.coordinator.data["status"] == "idle"
    # Round 1 plus exactly one refinement round, which the gate rejects.
    assert len(calls) == 2
    last_run = supervisor.coordinator.data["last_run"]
    assert last_run["rounds"] == 2
    assert last_run["accepted_round"] == 1

    by_id = {
        c["capture_id"]: c for c in (await _capture_list(hass))["captures"]
    }
    # Every capture keeps round 1's adopted verdicts, never the rejected round's.
    for capture in captures:
        assert by_id[capture["capture_id"]]["llm_verdict"] == "consistent"


async def test_bad_verdict_entry_does_not_fail_run(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    inprocess_worker: None,
    aioclient_mock: Any,
) -> None:
    """A reply whose verdict entry names an unknown example is salvaged, not fatal (D-07)."""
    entry, data = await _setup(hass, energy_manager, std_prefs, monkeypatch)
    await _capture_clean_kettle(hass, data, label="kettle run")

    cache: dict[str, Any] = {}

    def _bad_responder(request_data: Any) -> dict[str, Any]:
        # A repair turn carries no digest lines, so build the proposal once from
        # the round-1 body and answer every turn with the same bad verdict entry.
        if "prop" not in cache:
            cache["prop"] = kettle_proposal(_digest_from_request(request_data))
        prop = dict(cache["prop"])
        prop["example_verdicts"] = [{"example": "eX", "verdict": "consistent"}]
        return {"status": 200, "json": chat_completion(json.dumps(prop))}

    mock_llm(aioclient_mock, "http://llm.test/v1", _bad_responder)
    supervisor = await _run_learning(entry)
    assert supervisor.coordinator.data["status"] == "idle"
    assert supervisor.coordinator.data["last_error"] is None
    assert len(supervisor.candidates) >= 1


async def test_verdicts_never_change_candidates(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    inprocess_worker: None,
    aioclient_mock: Any,
) -> None:
    """An inconsistent verdict leaves candidates, previews and devices untouched (T-09-24)."""
    entry, data = await _setup(hass, energy_manager, std_prefs, monkeypatch)
    await _capture_clean_kettle(hass, data, label="kettle one")
    await _capture_clean_kettle(hass, data, label="kettle two")
    hub = entry.runtime_data

    state: dict[str, Any] = {"verdicts": ()}

    def _responder(request_data: Any) -> dict[str, Any]:
        req = _req_json(request_data)
        prop = kettle_proposal(_digest_from_request(req))
        if state["verdicts"]:
            content = "".join(
                m.get("content", "")
                for m in req.get("messages", [])
                if isinstance(m.get("content"), str)
            )
            eids = _EXAMPLE_LINE_RE.findall(content)
            prop["example_verdicts"] = [
                {"example": eid, "verdict": "inconsistent"} for eid in eids
            ]
        return {"status": 200, "json": chat_completion(json.dumps(prop))}

    mock_llm(aioclient_mock, "http://llm.test/v1", _responder)

    supervisor = await _run_learning(entry)
    candidates_before = _candidate_signature(supervisor)
    previews_before = sorted(
        str(c.get("signature_id")) for c in hub.captures.preview_candidates()
    )
    assert candidates_before  # the kettle candidate exists to compare

    state["verdicts"] = ("inconsistent", "inconsistent")
    supervisor = await _run_learning(entry)
    assert supervisor.coordinator.data["status"] == "idle"
    assert _candidate_signature(supervisor) == candidates_before
    assert sorted(
        str(c.get("signature_id")) for c in hub.captures.preview_candidates()
    ) == previews_before
    assert hub.devices.confirmed_entries() == []
    assert hub.devices.owned_signature_ids() == set()


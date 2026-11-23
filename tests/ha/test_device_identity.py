"""Tests for confirmed-device identity across learning runs (OUT-05, D-11, D-27)."""

import copy
import json
import re
from typing import Any

import pytest

from homeassistant.components.energy.data import EnergyManager
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType

from custom_components.powerguesser.const import REVIEW_STORE_KEY
from tests.conftest import (
    async_confirm_candidate,
    async_fill_synth_buffer,
    async_open_options_step,
    async_seed_consent,
    async_seed_llm,
    async_set_power,
    async_setup_via_flow,
    mock_llm,
    switched_cluster_ids,
)
from tests.fixtures.learning.fake_llm import chat_completion, kettle_proposal
from tests.fixtures.learning.synth import SynthSpec

_CLUSTER_RE = re.compile(
    r"id=(?P<id>c\d+)\s+kind=(?P<kind>\w+)\s+n=(?P<n>\d+)\s+"
    r"rise=(?P<r10>-?[\d.]+)/(?P<r50>-?[\d.]+)/(?P<r90>-?[\d.]+)\s+"
    r"fall=(?P<f50>-?[\d.]+)\s+dur_min=(?P<d10>[\d.]+)-(?P<d90>[\d.]+)"
)


def _digest_from_request(request_data: Any) -> dict[str, Any]:
    """Rebuild a digest-shaped dict from the rendered digest in an LLM request.

    Same helper as tests/ha/test_switched_loads_learning.py: the rendered
    cluster lines carry dur_min as p10-p90 only; the midpoint stands in for
    p50 in the dur-range selection (rise p50 is exact in the text).
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


def _set_std_powers(hass: HomeAssistant) -> None:
    async_set_power(hass, "sensor.grid_power", 500)
    async_set_power(hass, "sensor.solar_power", 1200)
    async_set_power(hass, "sensor.battery_power", -300)
    async_set_power(hass, "sensor.washer_power", 200)


async def _setup_learning_env(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> Any:
    """Entry with a 22-day synth buffer (kettle + oven), LLM seeded, refine off."""
    monkeypatch.setattr("custom_components.powerguesser.learning.MAX_REFINE_ROUNDS", 0)
    energy_manager.data = std_prefs
    _set_std_powers(hass)
    entry = await async_setup_via_flow(hass)
    spec = SynthSpec(days=22, oven=True)
    await async_fill_synth_buffer(hass, entry, spec=spec)
    await async_seed_llm(hass, entry)
    await async_seed_consent(hass, host="llm.test", names=False)
    return entry


async def _run_learning(sup: Any) -> None:
    """Request a learning run and wait for it."""
    await sup.async_request_run("service")
    assert sup.run_task is not None
    await sup.run_task


def _kettle_responder(label: str = "Kettle") -> Any:
    """Responder proposing the kettle cluster, relabelled as requested."""

    def _responder(data: Any) -> dict[str, Any]:
        prop = kettle_proposal(_digest_from_request(data))
        prop["devices"][0]["label"] = label
        return {"status": 200, "json": chat_completion(json.dumps(prop))}

    return _responder


def _session_only_responder() -> Any:
    """Responder proposing one program device on the session cluster only."""

    def _responder(data: Any) -> dict[str, Any]:
        digest = _digest_from_request(data)
        session_ids = [c["id"] for c in digest["clusters"] if c["kind"] == "session"]
        all_ids = [c["id"] for c in digest["clusters"]]
        assert session_ids, f"digest holds no session cluster: {digest}"
        cid = session_ids[0]
        prop = {
            "analysis": "Only the session cluster looks like a program appliance.",
            "schema_version": 1,
            "devices": [
                {
                    "ref": "d1",
                    "label": "Washer program",
                    "category": "unknown",
                    "tracker_kind": "program",
                    "member_clusters": [cid],
                    "states": [
                        {"name": "on", "cluster_ref": cid, "role": "main"}
                    ],
                    "phase_hint": "single",
                    "confidence": "medium",
                    "rationale": "Program cluster only.",
                    "ambiguous_with": [],
                    "reference_plug": "",
                }
            ],
            "rejected_clusters": [
                {"cluster_ref": c, "reason": "unexplainable"} for c in all_ids if c != cid
            ],
            "merge_suggestions": [],
        }
        return {"status": 200, "json": chat_completion(json.dumps(prop))}

    return _responder


def _rise_p50s(candidate: dict[str, Any]) -> list[float]:
    """All rise p50 values of a candidate's states."""
    out = []
    for st in candidate.get("states") or []:
        rise = st.get("rise_w") or {}
        if isinstance(rise.get("p50"), (int, float)):
            out.append(float(rise["p50"]))
    return out


def _assert_kettle_still_owned(sup: Any, sig_kettle: str, n_devices: int = 1) -> None:
    """The run's kettle artefact matched the device and nothing re-proposes it."""
    devices = sup.hub.devices
    matched = [c for c in sup.candidates if c["signature_id"] == sig_kettle]
    assert len(matched) == 1, [
        (c["signature_id"], (c.get("identity") or {}).get("status"))
        for c in sup.candidates
    ]
    assert matched[0]["identity"]["status"] == "matched"
    assert matched[0]["identity"]["matched"] == sig_kettle
    # No visible candidate is the kettle load: nothing near 2000 W is offered.
    visible = sup.visible_candidates()
    for cand in visible:
        assert not any(1500.0 <= p50 <= 2500.0 for p50 in _rise_p50s(cand)), (
            cand.get("signature_id"),
            _rise_p50s(cand),
        )
    assert len(devices._devices) == n_devices


async def test_drifted_library_entry_keeps_device_identity(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    aioclient_mock: Any,
    inprocess_worker: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stored library artefact drifted (or relabelled) under the device's signature
    does not break identity: the run's kettle artefact still carries the device's
    signature id as a matched identity and the load is not offered again (OUT-05, D-11).
    """
    entry = await _setup_learning_env(hass, energy_manager, std_prefs, monkeypatch)

    call_count = 0

    def _responder(data: Any) -> dict[str, Any]:
        nonlocal call_count
        call_count += 1
        label = "Kettle" if call_count == 1 else "Water boiler"
        return _kettle_responder(label)(data)

    mock_llm(aioclient_mock, "http://llm.test/v1", _responder)
    sup = entry.runtime_data.learning

    # --- Run 1: propose and confirm the kettle ---
    await _run_learning(sup)
    cand_run1 = sup.visible_candidates()[0]
    sig_kettle = cand_run1["signature_id"]
    await async_confirm_candidate(hass, entry, sig_kettle, "Kettle")
    assert len(entry.runtime_data.devices._devices) == 1

    # --- The stored library entry under the device's signature drifts: the
    # next unaccepted run's artefact (rise box 600-800 W) replaces it. ---
    stored = next(c for c in sup.candidates if c["signature_id"] == sig_kettle)
    drifted = copy.deepcopy(stored)
    for st in drifted["states"]:
        for key in ("rise_w", "plateau_w"):
            st[key] = {"lo": 600.0, "p50": 700.0, "hi": 800.0}
    for st in (drifted.get("fingerprint") or {}).get("states") or []:
        for key in ("rise_w", "plateau_w"):
            st[key] = {"lo": 600.0, "p50": 700.0, "hi": 800.0}
    sup.candidates = [
        drifted if c["signature_id"] == sig_kettle else c for c in sup.candidates
    ]
    sup._candidates_record["candidates"] = sup.candidates
    await sup._candidates_store.async_save(sup._candidates_record)

    # --- Run 2: the same load proposed again under a new label ---
    await _run_learning(sup)

    _assert_kettle_still_owned(sup, sig_kettle)
    assert sup.visible_candidates() == []

    # The review list has no option for the load (C1 across runs); the only
    # entry is the confirmed device itself (Review contract C2, Plan 04-10).
    res_review = await async_open_options_step(hass, entry, "review")
    assert res_review["type"] is FlowResultType.FORM
    opts = res_review["data_schema"].schema["candidate"].config["options"]
    assert [o["value"] for o in opts] == [
        f"device_{entry.runtime_data.devices.device_for_signature(sig_kettle)}"
    ]


async def test_device_identity_survives_run_without_it(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    aioclient_mock: Any,
    inprocess_worker: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A run whose library lacks the kettle (only a session proposal) does not
    unbind the device; the next run that proposes the kettle again matches it (OUT-05).
    """
    entry = await _setup_learning_env(hass, energy_manager, std_prefs, monkeypatch)

    call_count = 0

    def _responder(data: Any) -> dict[str, Any]:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _kettle_responder("Kettle")(data)
        if call_count == 2:
            return _session_only_responder()(data)
        return _kettle_responder("Water boiler")(data)

    mock_llm(aioclient_mock, "http://llm.test/v1", _responder)
    sup = entry.runtime_data.learning

    # --- Run 1: propose and confirm the kettle ---
    await _run_learning(sup)
    cand_run1 = sup.visible_candidates()[0]
    sig_kettle = cand_run1["signature_id"]
    await async_confirm_candidate(hass, entry, sig_kettle, "Kettle")

    # --- Run 2: the library holds only a session-cluster candidate ---
    await _run_learning(sup)
    assert all(
        c["signature_id"] != sig_kettle for c in sup.candidates
    ), "run 2 must not re-propose the kettle"
    visible_run2 = sup.visible_candidates()
    assert all(c["signature_id"] != sig_kettle for c in visible_run2)
    assert len(visible_run2) == 1 and visible_run2[0]["kind"] == "program"
    assert len(entry.runtime_data.devices._devices) == 1

    # --- Run 3: the kettle is proposed again ---
    await _run_learning(sup)
    _assert_kettle_still_owned(sup, sig_kettle)
    assert sup.visible_candidates() == []


async def test_learning_runs_still_never_write_review_store_after_confirm(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    aioclient_mock: Any,
    inprocess_worker: Any,
    monkeypatch: pytest.MonkeyPatch,
    hass_storage: dict[str, Any],
) -> None:
    """Learning runs after a confirm leave the review Store content unchanged (REV-01)."""
    entry = await _setup_learning_env(hass, energy_manager, std_prefs, monkeypatch)

    call_count = 0

    def _responder(data: Any) -> dict[str, Any]:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _kettle_responder("Kettle")(data)
        if call_count == 2:
            return _session_only_responder()(data)
        return _kettle_responder("Water boiler")(data)

    mock_llm(aioclient_mock, "http://llm.test/v1", _responder)
    sup = entry.runtime_data.learning

    # --- Run 1: propose and confirm the kettle (writes the review link) ---
    await _run_learning(sup)
    sig_kettle = sup.visible_candidates()[0]["signature_id"]
    await async_confirm_candidate(hass, entry, sig_kettle, "Kettle")
    assert REVIEW_STORE_KEY in hass_storage

    # --- Run 2: a library without the kettle ---
    await _run_learning(sup)
    snapshot = copy.deepcopy(hass_storage[REVIEW_STORE_KEY]["data"])

    # --- Run 3: the kettle proposed again ---
    await _run_learning(sup)

    after = hass_storage[REVIEW_STORE_KEY]["data"]
    assert json.dumps(after, sort_keys=True) == json.dumps(snapshot, sort_keys=True)


# --- D-27: the overlaps note in Review ---


def _make_proposal(devices: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "analysis": "Test proposal.",
        "schema_version": 1,
        "devices": devices,
        "rejected_clusters": [],
        "merge_suggestions": [],
    }


def _make_device(ref: str, cluster: str, label: str | None = None) -> dict[str, Any]:
    return {
        "ref": ref,
        "label": label or f"Device {ref}",
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
        "confidence": "high",
        "rationale": f"Device for {cluster}.",
        "ambiguous_with": [],
        "reference_plug": "",
    }


async def test_superset_of_confirmed_device_offered_with_overlap_note(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    aioclient_mock: Any,
    inprocess_worker: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """D-27: a candidate that strictly contains a confirmed device's episodes
    (more claimed episodes outside the device than inside, Jaccard below
    JACCARD_MIN) is NOT hidden: it stays in Review with Confirm and the review
    label ends with " — overlaps <device name>"."""
    from custom_components.powerguesser.config_flow import _escape_markdown

    entry = await _setup_learning_env(hass, energy_manager, std_prefs, monkeypatch)

    call_count = 0

    def _responder(data: Any) -> dict[str, Any]:
        nonlocal call_count
        call_count += 1
        sw = switched_cluster_ids(data)
        if call_count == 1:
            # Run 1: a candidate on the smaller switched cluster (oven pulses).
            prop = _make_proposal([_make_device("d1", sw[1], label="Oven")])
        else:
            # Run 2: one device over ALL switched clusters — its claims strictly
            # contain the device's episodes (overlap coefficient 1.0) while the
            # one-to-one pass sees only Jaccard 38/129 < 0.5 and does not match.
            # Each state's boxes are refit from its own cluster_ref, so the union
            # needs one state per cluster to claim both episode sets.
            dev = _make_device("d2", sw[0])
            dev["member_clusters"] = list(sw)
            state_b = dict(dev["states"][0])
            state_b["cluster_ref"] = sw[1]
            dev["states"] = [dev["states"][0], state_b]
            prop = _make_proposal([dev])
        return {"status": 200, "json": chat_completion(json.dumps(prop))}

    mock_llm(aioclient_mock, "http://llm.test/v1", _responder)
    sup = entry.runtime_data.learning

    # --- Run 1: propose and confirm the oven ---
    await _run_learning(sup)
    cand_run1 = sup.visible_candidates()[0]
    sig_oven = cand_run1["signature_id"]
    await async_confirm_candidate(hass, entry, sig_oven, "Oven")

    # --- Run 2: a super-cluster proposal containing the device's episodes ---
    await _run_learning(sup)

    visible = sup.visible_candidates()
    assert len(visible) == 1
    cand = visible[0]
    ident = cand["identity"]
    assert ident["status"] == "new"
    assert ident["overlaps"] == sig_oven
    # The device's own signature is never offered as a candidate (C1).
    assert cand["signature_id"] != sig_oven

    label = sup.candidate_label(cand)
    assert label.endswith(" — overlaps Oven")

    # The review list offers the candidate but no option for the device signature.
    res_review = await async_open_options_step(hass, entry, "review")
    opts = res_review["data_schema"].schema["candidate"].config["options"]
    values = [opt["value"] for opt in opts]
    assert cand["signature_id"] in values
    assert sig_oven not in values

    # The candidate menu escapes the whole label (note included) and, as a
    # switched candidate, offers Confirm.
    res_cand = await hass.config_entries.options.async_configure(
        res_review["flow_id"], {"candidate": cand["signature_id"]}
    )
    assert res_cand["type"] is FlowResultType.MENU
    assert "confirm" in res_cand["menu_options"]
    assert res_cand["description_placeholders"]["candidate"] == _escape_markdown(label)

    # No second device was created.
    assert len(entry.runtime_data.devices._devices) == 1


async def test_unconfirmed_device_candidate_reappears_with_confirm(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    aioclient_mock: Any,
    inprocess_worker: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """D-05: an unconfirmed device record (deleted in Settings) is not owned —
    a run proposing its load again lists the candidate under its old signature
    with Confirm, no overlaps marker, and re-confirming reuses the uuid."""
    entry = await _setup_learning_env(hass, energy_manager, std_prefs, monkeypatch)
    mock_llm(aioclient_mock, "http://llm.test/v1", _kettle_responder("Kettle"))
    sup = entry.runtime_data.learning
    devices = entry.runtime_data.devices

    # --- Run 1: propose and confirm the kettle ---
    await _run_learning(sup)
    sig_kettle = sup.visible_candidates()[0]["signature_id"]
    await async_confirm_candidate(hass, entry, sig_kettle, "Kettle")
    sub = next(s for s in entry.get_subentries_of_type("device") if s.title == "Kettle")
    dev_uuid = sub.data["device_uuid"]

    # Delete the device in Settings: un-confirm (D-05).
    hass.config_entries.async_remove_subentry(entry, sub.subentry_id)
    await hass.async_block_till_done()
    assert devices.record(dev_uuid)["status"] == "unconfirmed"

    # --- Run 2: the kettle proposed again lists it with its old signature ---
    await _run_learning(sup)
    visible = sup.visible_candidates()
    back = next((c for c in visible if c.get("signature_id") == sig_kettle), None)
    assert back is not None
    ident = back["identity"]
    assert ident["status"] == "matched"
    assert ident["matched"] == sig_kettle
    # Unconfirmed records are not owned: no twin suppression, no overlaps marker.
    assert "overlaps" not in ident
    assert sup.candidate_label(back).startswith("Kettle")

    # --- Re-confirm reuses the uuid ---
    await async_confirm_candidate(hass, entry, sig_kettle, "Kettle")
    new_sub = next(
        s for s in entry.get_subentries_of_type("device") if s.title == "Kettle"
    )
    assert new_sub.data["device_uuid"] == dev_uuid
    assert devices.record(dev_uuid)["status"] == "confirmed"
    assert len(devices._devices) == 1

"""Tests for confirmed-device update proposals, acceptance, rollback and rename (D-11, D-25, OUT-05, REV-02).

Plan 04-10 Task 1: a later learning run whose artefact matches a confirmed
device only proposes an update — the device's parameters never change without
the user's explicit Accept, which stores the proposal as parameter version
n+1 under the same device uuid and activates it without a reload. Confirmed
devices appear in Review as device entries (Review contract C2) with the
device menu (Accept update, Rename; Task 2 adds Restore).

Plan 04-10 Task 2: the device menu offers Restore an earlier parameter
version (only with two or more stored versions), re-activating any stored
version through the same path as Accept without touching sensors, history,
energy totals or a pending proposal; renaming a device from Review updates
the subentry so the reconcile path renames everything; the menu never offers
Confirm or Dismiss.
"""

import copy
import json
import math
import re
from datetime import timedelta
from typing import Any

import pytest

import custom_components.powerguesser as pg
from homeassistant.components.energy.data import EnergyManager
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers import entity_registry as er
from homeassistant.util import dt as dt_util
from pytest_homeassistant.common import async_fire_time_changed

from tests.conftest import (
    async_confirm_candidate,
    async_fill_synth_buffer,
    async_open_options_step,
    async_seed_consent,
    async_seed_llm,
    async_set_power,
    async_setup_via_flow,
    mock_llm,
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

    Same helper as tests/ha/test_device_identity.py.
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
) -> ConfigEntry:
    """Entry with a 22-day synth buffer (kettle + oven + water heater), refine off."""
    monkeypatch.setattr("custom_components.powerguesser.learning.MAX_REFINE_ROUNDS", 0)
    energy_manager.data = std_prefs
    _set_std_powers(hass)
    entry = await async_setup_via_flow(hass)
    spec = SynthSpec(days=22, oven=True, water_heater=True)
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


def _kettle_wh_responder(
    label: str = "Kettle", state_names: tuple[str, str] = ("heat", "tank")
) -> Any:
    """Responder proposing the kettle device over the kettle AND water-heater clusters.

    The compiled artefact carries two states (kettle boxes + water-heater
    boxes): its episode claims overlap the confirmed kettle device enough to
    match its signature, while its states differ from the one-state active
    version — the D-11 proposal shape.
    """

    def _responder(data: Any) -> dict[str, Any]:
        digest = _digest_from_request(data)
        clusters = digest["clusters"]
        ck = min(clusters, key=lambda c: abs(c["rise_w"]["p50"] - 2000))["id"]
        cw = min(clusters, key=lambda c: abs(c["rise_w"]["p50"] - 3000))["id"]
        prop = {
            "analysis": "Kettle plus water heater state.",
            "schema_version": 1,
            "devices": [
                {
                    "ref": "d1",
                    "label": label,
                    "category": "unknown",
                    "tracker_kind": "switched",
                    "member_clusters": [ck, cw],
                    "states": [
                        {"name": state_names[0], "cluster_ref": ck, "role": "main"},
                        {"name": state_names[1], "cluster_ref": cw, "role": "main"},
                    ],
                    "phase_hint": "single",
                    "confidence": "high",
                    "rationale": "Two switched states refit from real episodes.",
                    "ambiguous_with": [],
                    "reference_plug": "",
                }
            ],
            "rejected_clusters": [],
            "merge_suggestions": [],
        }
        return {"status": 200, "json": chat_completion(json.dumps(prop))}

    return _responder


def _device_ids(hass: HomeAssistant, entry: ConfigEntry, dev_uuid: str) -> dict[str, Any]:
    """Collect the entity ids and unique ids of a confirmed device's sensors."""
    ent_reg = er.async_get(hass)
    power_unique_id = f"{entry.entry_id}_{dev_uuid}_power"
    energy_unique_id = f"{entry.entry_id}_{dev_uuid}_energy"
    return {
        "power_eid": ent_reg.async_get_entity_id("sensor", "powerguesser", power_unique_id),
        "energy_eid": ent_reg.async_get_entity_id("sensor", "powerguesser", energy_unique_id),
        "power_unique_id": power_unique_id,
        "energy_unique_id": energy_unique_id,
    }


def _count_setups(monkeypatch: pytest.MonkeyPatch) -> list[None]:
    """Monkeypatch async_setup_entry to count reload-driven setup calls."""
    setup_calls: list[None] = []
    orig_setup = pg.async_setup_entry

    async def _counting_setup(hass_: HomeAssistant, entry_: ConfigEntry) -> bool:
        setup_calls.append(None)
        return await orig_setup(hass_, entry_)

    monkeypatch.setattr(pg, "async_setup_entry", _counting_setup)
    return setup_calls


async def test_learning_run_proposes_and_accept_creates_version(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    aioclient_mock: Any,
    inprocess_worker: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A later run proposes an update for the confirmed kettle; Accept stores it
    as version 2 under the same uuid and activates it without touching identity."""
    entry = await _setup_learning_env(hass, energy_manager, std_prefs, monkeypatch)

    call_count = 0

    def _responder(data: Any) -> dict[str, Any]:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _kettle_responder("Kettle")(data)
        return _kettle_wh_responder("Kettle")(data)

    mock_llm(aioclient_mock, "http://llm.test/v1", _responder)
    sup = entry.runtime_data.learning
    devices = entry.runtime_data.devices

    # --- Run 1: propose and confirm the kettle ---
    await _run_learning(sup)
    sig_kettle = sup.visible_candidates()[0]["signature_id"]
    await async_confirm_candidate(hass, entry, sig_kettle, "Kettle")
    sub = next(s for s in entry.get_subentries_of_type("device") if s.title == "Kettle")
    dev_uuid = sub.data["device_uuid"]
    ids = _device_ids(hass, entry, dev_uuid)
    rec = devices.record(dev_uuid)
    v1_blob = json.dumps(rec["versions"][0], sort_keys=True)
    assert rec["active_v"] == 1

    setup_calls = _count_setups(monkeypatch)

    # --- Run 2: the matched artefact's states differ -> a proposal, no param change ---
    await _run_learning(sup)
    rec = devices.record(dev_uuid)
    assert rec["active_v"] == 1
    assert len(rec["versions"]) == 1
    assert json.dumps(rec["versions"][0], sort_keys=True) == v1_blob
    proposal = rec["pending_proposal"]
    assert set(proposal.keys()) == {"artefact", "run_id", "proposed_at", "trained_at", "scores"}
    assert [s["name"] for s in proposal["artefact"]["states"]] == ["heat", "tank"]
    assert devices.update_available(dev_uuid) is True

    # The load is not offered as a candidate (C1); Review shows the device entry
    # with the update note; pending_candidates counts candidates only.
    assert sup.visible_candidates() == []
    assert sup.coordinator.data["pending_candidates"] == 0
    res_review = await async_open_options_step(hass, entry, "review")
    assert res_review["type"] is FlowResultType.FORM
    opts = res_review["data_schema"].schema["candidate"].config["options"]
    assert [o["value"] for o in opts] == [f"device_{dev_uuid}"]
    assert opts[0]["label"] == "Kettle — confirmed device — update available"

    # The power sensor carries update_available true.
    state = hass.states.get(ids["power_eid"])
    assert state is not None
    assert state.attributes["update_available"] is True

    # --- Accept the update in Review ---
    res_dev = await hass.config_entries.options.async_configure(
        res_review["flow_id"], {"candidate": f"device_{dev_uuid}"}
    )
    assert res_dev["type"] is FlowResultType.MENU
    assert res_dev["step_id"] == "device"
    assert "accept_update" in res_dev["menu_options"]
    assert "confirm" not in res_dev["menu_options"]
    assert "dismiss" not in res_dev["menu_options"]
    res_after = await hass.config_entries.options.async_configure(
        res_dev["flow_id"], {"next_step_id": "accept_update"}
    )
    assert res_after["type"] is FlowResultType.FORM
    assert res_after["step_id"] == "review"

    # Version 2 stored under the same uuid, activated, proposal cleared.
    rec = devices.record(dev_uuid)
    assert [v["v"] for v in rec["versions"]] == [1, 2]
    assert rec["active_v"] == 2
    assert rec["pending_proposal"] is None
    assert rec["versions"][1]["source"] == "proposal"
    assert json.dumps(rec["versions"][0], sort_keys=True) == v1_blob

    # Identity: unique_ids, entity_ids and no reload.
    ids_after = _device_ids(hass, entry, dev_uuid)
    assert ids_after == ids
    assert len(setup_calls) == 0
    assert hass.states.get(ids["power_eid"]).attributes["update_available"] is False

    # The review label no longer carries the update note.
    opts = res_after["data_schema"].schema["candidate"].config["options"]
    assert opts[0]["label"] == "Kettle — confirmed device"


async def test_identical_states_make_no_proposal(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    aioclient_mock: Any,
    inprocess_worker: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A run whose matched artefact has the active version's states stores nothing."""
    entry = await _setup_learning_env(hass, energy_manager, std_prefs, monkeypatch)
    mock_llm(aioclient_mock, "http://llm.test/v1", _kettle_responder("Kettle"))
    sup = entry.runtime_data.learning
    devices = entry.runtime_data.devices

    await _run_learning(sup)
    sig_kettle = sup.visible_candidates()[0]["signature_id"]
    await async_confirm_candidate(hass, entry, sig_kettle, "Kettle")
    sub = next(s for s in entry.get_subentries_of_type("device") if s.title == "Kettle")
    dev_uuid = sub.data["device_uuid"]

    # Run 2 proposes exactly the same device: compiled states are identical.
    await _run_learning(sup)
    rec = devices.record(dev_uuid)
    assert rec["pending_proposal"] is None
    assert len(rec["versions"]) == 1
    assert devices.update_available(dev_uuid) is False

    res_review = await async_open_options_step(hass, entry, "review")
    opts = res_review["data_schema"].schema["candidate"].config["options"]
    assert opts[0]["label"] == "Kettle — confirmed device"
    assert "update available" not in opts[0]["label"]
    state = hass.states.get(
        er.async_get(hass).async_get_entity_id(
            "sensor", "powerguesser", f"{entry.entry_id}_{dev_uuid}_power"
        )
    )
    assert state.attributes["update_available"] is False


async def test_run_never_changes_confirmed_params(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    aioclient_mock: Any,
    inprocess_worker: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Learning runs without an Accept never touch the device's parameters:
    versions and the active artefact stay byte-identical (T-04-20, VAL-05)."""
    entry = await _setup_learning_env(hass, energy_manager, std_prefs, monkeypatch)

    call_count = 0

    def _responder(data: Any) -> dict[str, Any]:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _kettle_responder("Kettle")(data)
        return _kettle_wh_responder("Kettle")(data)

    mock_llm(aioclient_mock, "http://llm.test/v1", _responder)
    sup = entry.runtime_data.learning
    devices = entry.runtime_data.devices

    await _run_learning(sup)
    sig_kettle = sup.visible_candidates()[0]["signature_id"]
    await async_confirm_candidate(hass, entry, sig_kettle, "Kettle")
    sub = next(s for s in entry.get_subentries_of_type("device") if s.title == "Kettle")
    dev_uuid = sub.data["device_uuid"]
    blob = json.dumps(devices.record(dev_uuid), sort_keys=True, default=str)

    # Two runs whose artefact differs from the active version: both only replace
    # the pending proposal; the record's parameters never move.
    await _run_learning(sup)
    await _run_learning(sup)
    rec = devices.record(dev_uuid)
    assert [v["v"] for v in rec["versions"]] == [1]
    assert rec["active_v"] == 1
    unchanged = copy.deepcopy(json.loads(blob))
    assert rec["pending_proposal"] is not None
    cleared = {**rec, "pending_proposal": unchanged["pending_proposal"]}
    assert json.dumps(cleared, sort_keys=True, default=str) == blob


async def test_confirmed_device_listed_as_device_entry(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    aioclient_mock: Any,
    inprocess_worker: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Review contract C2: the confirmed device is one option with value
    device_<uuid> and the '— confirmed device' label; its signature id is
    never an option; pending_candidates counts candidates only."""
    entry = await _setup_learning_env(hass, energy_manager, std_prefs, monkeypatch)
    mock_llm(aioclient_mock, "http://llm.test/v1", _kettle_responder("Kettle"))
    sup = entry.runtime_data.learning

    await _run_learning(sup)
    sig_kettle = sup.visible_candidates()[0]["signature_id"]
    await async_confirm_candidate(hass, entry, sig_kettle, "Kettle")
    sub = next(s for s in entry.get_subentries_of_type("device") if s.title == "Kettle")
    dev_uuid = sub.data["device_uuid"]

    assert sup.coordinator.data["pending_candidates"] == 0
    res_review = await async_open_options_step(hass, entry, "review")
    assert res_review["type"] is FlowResultType.FORM
    opts = res_review["data_schema"].schema["candidate"].config["options"]
    values = [o["value"] for o in opts]
    assert values == [f"device_{dev_uuid}"]
    assert sig_kettle not in values
    assert opts[0]["label"].startswith("Kettle — confirmed device")

    # The device entry opens the device menu, never the candidate menu.
    res_dev = await hass.config_entries.options.async_configure(
        res_review["flow_id"], {"candidate": f"device_{dev_uuid}"}
    )
    assert res_dev["type"] is FlowResultType.MENU
    assert res_dev["step_id"] == "device"
    assert "confirm" not in res_dev["menu_options"]
    assert "dismiss" not in res_dev["menu_options"]


# --- Task 2: rollback and rename from Review (D-25) -------------------------


async def _open_device_menu(
    hass: HomeAssistant, entry: ConfigEntry, dev_uuid: str
) -> dict[str, Any]:
    """Open Review and pick the device entry; returns the device menu result."""
    res_review = await async_open_options_step(hass, entry, "review")
    assert res_review["type"] is FlowResultType.FORM
    return await hass.config_entries.options.async_configure(
        res_review["flow_id"], {"candidate": f"device_{dev_uuid}"}
    )


async def _pick_menu(
    hass: HomeAssistant, res: dict[str, Any], next_step_id: str
) -> dict[str, Any]:
    return await hass.config_entries.options.async_configure(
        res["flow_id"], {"next_step_id": next_step_id}
    )


async def _aligned_move(freezer: Any, grid_s: int) -> None:
    """Move the freezer to the next epoch multiple of the detector grid."""
    now = dt_util.utcnow().timestamp()
    aligned = math.ceil(now / grid_s) * grid_s
    freezer.move_to(dt_util.utc_from_timestamp(aligned))


async def _pin_grid(entry: ConfigEntry, grid_s: int) -> None:
    """Publish a profile snapshot pinning the D-26 grid at the artefact grid."""
    from custom_components.powerguesser.engine.intervals import (
        ProfileSnapshot,
        gate,
    )

    enabled, disabled = gate(float(grid_s), False)
    entry.runtime_data.profile_coordinator.async_set_updated_data(
        ProfileSnapshot(
            sources={},
            effective_s=float(grid_s),
            assumed=False,
            enabled=enabled,
            disabled=disabled,
        )
    )


async def _tick(hass: HomeAssistant, freezer: Any, seconds: int) -> None:
    """Advance the freezer and fire the time-changed event."""
    freezer.tick(timedelta(seconds=seconds))
    async_fire_time_changed(hass, dt_util.utcnow())
    await hass.async_block_till_done()


async def _drive_grid_steps(hass: HomeAssistant, freezer: Any, n: int, grid_s: int) -> None:
    """Advance n full grid steps via the 10 s tick path."""
    ticks_per_step = max(3, math.ceil(grid_s / 10) + 1)
    for _ in range(n * ticks_per_step):
        await _tick(hass, freezer, 10)


async def _wh_on(hass: HomeAssistant, freezer: Any, grid_s: int) -> None:
    """Switch a 3000 W water-heater load on and settle three grid points."""
    async_set_power(hass, "sensor.grid_power", 3500)
    await hass.async_block_till_done()
    await _drive_grid_steps(hass, freezer, 3, grid_s)


async def _wh_off(hass: HomeAssistant, freezer: Any, grid_s: int) -> None:
    """Switch the water-heater load off and settle three grid points."""
    async_set_power(hass, "sensor.grid_power", 500)
    await hass.async_block_till_done()
    await _drive_grid_steps(hass, freezer, 3, grid_s)


async def _rollback_to(
    hass: HomeAssistant, entry: ConfigEntry, dev_uuid: str, version: int
) -> dict[str, Any]:
    """Open the device menu, pick Restore and submit the given version."""
    res_dev = await _open_device_menu(hass, entry, dev_uuid)
    assert res_dev["type"] is FlowResultType.MENU
    assert "rollback" in res_dev["menu_options"]
    res_rb = await _pick_menu(hass, res_dev, "rollback")
    assert res_rb["type"] is FlowResultType.FORM
    assert res_rb["step_id"] == "rollback"
    return await hass.config_entries.options.async_configure(
        res_rb["flow_id"], {"version": str(version)}
    )


async def test_rollback_reactivates_earlier_version(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    aioclient_mock: Any,
    inprocess_worker: Any,
    monkeypatch: pytest.MonkeyPatch,
    freezer: Any,
) -> None:
    """D-25: Restore re-activates an earlier stored version through the same
    path as Accept — no version added or removed, identity and energy totals
    untouched, no reload — and the detector uses the re-activated boxes from
    the next edge on, in both directions."""
    entry = await _setup_learning_env(hass, energy_manager, std_prefs, monkeypatch)

    call_count = 0

    def _responder(data: Any) -> dict[str, Any]:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _kettle_responder("Kettle")(data)
        return _kettle_wh_responder("Kettle")(data)

    mock_llm(aioclient_mock, "http://llm.test/v1", _responder)
    sup = entry.runtime_data.learning
    devices = entry.runtime_data.devices

    # --- Run 1: confirm the kettle; run 2: accept the two-state proposal as v2 ---
    await _run_learning(sup)
    sig_kettle = sup.visible_candidates()[0]["signature_id"]
    await async_confirm_candidate(hass, entry, sig_kettle, "Kettle")
    sub = next(s for s in entry.get_subentries_of_type("device") if s.title == "Kettle")
    dev_uuid = sub.data["device_uuid"]
    ids = _device_ids(hass, entry, dev_uuid)
    await _run_learning(sup)
    res_dev = await _open_device_menu(hass, entry, dev_uuid)
    await _pick_menu(hass, res_dev, "accept_update")
    rec = devices.record(dev_uuid)
    assert [v["v"] for v in rec["versions"]] == [1, 2]
    assert rec["active_v"] == 2
    v1_blob = json.dumps(rec["versions"][0], sort_keys=True)
    v2_blob = json.dumps(rec["versions"][1], sort_keys=True)
    energy_before = devices.energy_wh(dev_uuid)
    grid_s = int(rec["versions"][1]["artefact"]["extraction"]["grid_s"])

    setup_calls = _count_setups(monkeypatch)

    # --- Roll back to v1 through the device menu ---
    res_done = await _rollback_to(hass, entry, dev_uuid, 1)
    assert res_done["type"] is FlowResultType.FORM
    assert res_done["step_id"] == "review"
    rec = devices.record(dev_uuid)
    assert rec["active_v"] == 1
    assert [v["v"] for v in rec["versions"]] == [1, 2]
    assert json.dumps(rec["versions"][0], sort_keys=True) == v1_blob
    assert json.dumps(rec["versions"][1], sort_keys=True) == v2_blob
    assert devices.energy_wh(dev_uuid) == energy_before
    assert _device_ids(hass, entry, dev_uuid) == ids
    assert len(setup_calls) == 0

    # The version selector listed every stored version except the active one.
    res_dev = await _open_device_menu(hass, entry, dev_uuid)
    res_rb = await _pick_menu(hass, res_dev, "rollback")
    opts = res_rb["data_schema"].schema["version"].config["options"]
    assert [o["value"] for o in opts] == ["2"]
    assert opts[0]["label"].startswith("v2 — proposal, accepted ")

    # --- The detector uses the re-activated boxes from the next edge on ---
    _aligned_move(freezer, grid_s)
    await _pin_grid(entry, grid_s)

    def _power_w() -> float:
        st = hass.states.get(ids["power_eid"])
        if st is None:
            return 0.0
        try:
            return float(st.state)
        except (TypeError, ValueError):
            return 0.0

    # A 3000 W water-heater edge: v1 (kettle only) does not attribute it.
    await _wh_on(hass, freezer, grid_s)
    assert _power_w() < 2500.0
    await _wh_off(hass, freezer, grid_s)

    # Rolling forward to v2: the same edge shape is attributed again.
    await _rollback_to(hass, entry, dev_uuid, 2)
    assert devices.record(dev_uuid)["active_v"] == 2
    await _wh_on(hass, freezer, grid_s)
    assert _power_w() >= 2500.0
    await _wh_off(hass, freezer, grid_s)

    # And back to v1: not attributed.
    await _rollback_to(hass, entry, dev_uuid, 1)
    assert devices.record(dev_uuid)["active_v"] == 1
    await _wh_on(hass, freezer, grid_s)
    assert _power_w() < 2500.0
    await _wh_off(hass, freezer, grid_s)


async def test_rollback_hidden_with_single_version(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    aioclient_mock: Any,
    inprocess_worker: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A device with one stored version has no Restore option."""
    entry = await _setup_learning_env(hass, energy_manager, std_prefs, monkeypatch)
    mock_llm(aioclient_mock, "http://llm.test/v1", _kettle_responder("Kettle"))
    sup = entry.runtime_data.learning

    await _run_learning(sup)
    sig_kettle = sup.visible_candidates()[0]["signature_id"]
    await async_confirm_candidate(hass, entry, sig_kettle, "Kettle")
    sub = next(s for s in entry.get_subentries_of_type("device") if s.title == "Kettle")
    dev_uuid = sub.data["device_uuid"]
    assert len(entry.runtime_data.devices.record(dev_uuid)["versions"]) == 1

    res_dev = await _open_device_menu(hass, entry, dev_uuid)
    assert res_dev["type"] is FlowResultType.MENU
    assert "rollback" not in res_dev["menu_options"]


async def test_rollback_keeps_pending_proposal(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    aioclient_mock: Any,
    inprocess_worker: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rollback with a pending proposal: the proposal survives, update_available
    stays true and Accept afterwards appends the proposal as v3 over the
    rolled-back state (review concern L264@4e11826)."""
    entry = await _setup_learning_env(hass, energy_manager, std_prefs, monkeypatch)

    call_count = 0

    def _responder(data: Any) -> dict[str, Any]:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _kettle_responder("Kettle")(data)
        if call_count == 2:
            return _kettle_wh_responder("Kettle")(data)
        # Run 3: the same two-cluster device under different state names —
        # its states differ from the ACTIVE v2, so it becomes a new proposal.
        return _kettle_wh_responder("Kettle", state_names=("boil", "cylinder"))(data)

    mock_llm(aioclient_mock, "http://llm.test/v1", _responder)
    sup = entry.runtime_data.learning
    devices = entry.runtime_data.devices

    await _run_learning(sup)
    sig_kettle = sup.visible_candidates()[0]["signature_id"]
    await async_confirm_candidate(hass, entry, sig_kettle, "Kettle")
    sub = next(s for s in entry.get_subentries_of_type("device") if s.title == "Kettle")
    dev_uuid = sub.data["device_uuid"]

    # Accept the first proposal -> v2 active; a third run stores a second proposal.
    await _run_learning(sup)
    res_dev = await _open_device_menu(hass, entry, dev_uuid)
    await _pick_menu(hass, res_dev, "accept_update")
    assert devices.record(dev_uuid)["active_v"] == 2

    await _run_learning(sup)
    rec = devices.record(dev_uuid)
    assert rec["pending_proposal"] is not None
    assert [s["name"] for s in rec["pending_proposal"]["artefact"]["states"]] == [
        "boil",
        "cylinder",
    ]
    proposal_blob = json.dumps(rec["pending_proposal"], sort_keys=True)

    # Roll back to v1: active_v 1, proposal untouched, update still offered.
    await _rollback_to(hass, entry, dev_uuid, 1)
    rec = devices.record(dev_uuid)
    assert rec["active_v"] == 1
    assert [v["v"] for v in rec["versions"]] == [1, 2]
    assert json.dumps(rec["pending_proposal"], sort_keys=True) == proposal_blob
    assert devices.update_available(dev_uuid) is True

    res_review = await async_open_options_step(hass, entry, "review")
    opts = res_review["data_schema"].schema["candidate"].config["options"]
    assert opts[0]["label"] == "Kettle — confirmed device — update available"
    state = hass.states.get(
        er.async_get(hass).async_get_entity_id(
            "sensor", "powerguesser", f"{entry.entry_id}_{dev_uuid}_power"
        )
    )
    assert state.attributes["update_available"] is True

    # Accept appends the pending proposal as v3 over the rolled-back state.
    res_dev = await _open_device_menu(hass, entry, dev_uuid)
    await _pick_menu(hass, res_dev, "accept_update")
    rec = devices.record(dev_uuid)
    assert [v["v"] for v in rec["versions"]] == [1, 2, 3]
    assert rec["active_v"] == 3
    assert rec["pending_proposal"] is None
    assert [s["name"] for s in rec["versions"][2]["artefact"]["states"]] == [
        "boil",
        "cylinder",
    ]


async def test_rename_confirmed_device_from_review_renames_everything(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    aioclient_mock: Any,
    inprocess_worker: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Renaming a confirmed device from Review updates the subentry title and
    data name; the Plan 04-05 reconcile renames the cold record, the review
    record and the device registry; entity ids never change (OUT-05)."""
    entry = await _setup_learning_env(hass, energy_manager, std_prefs, monkeypatch)
    mock_llm(aioclient_mock, "http://llm.test/v1", _kettle_responder("Kettle"))
    sup = entry.runtime_data.learning
    devices = entry.runtime_data.devices

    await _run_learning(sup)
    sig_kettle = sup.visible_candidates()[0]["signature_id"]
    await async_confirm_candidate(hass, entry, sig_kettle, "Kettle")
    sub = next(s for s in entry.get_subentries_of_type("device") if s.title == "Kettle")
    dev_uuid = sub.data["device_uuid"]
    ids = _device_ids(hass, entry, dev_uuid)
    dev_reg = dr.async_get(hass)
    device_entry = dev_reg.async_get_device_by_identifier(
        ("powerguesser", dev_uuid), entry.entry_id
    )
    assert device_entry is not None

    # Device menu -> Rename -> submit the new name.
    res_dev = await _open_device_menu(hass, entry, dev_uuid)
    res_form = await _pick_menu(hass, res_dev, "rename")
    assert res_form["type"] is FlowResultType.FORM
    assert res_form["step_id"] == "rename"
    res_done = await hass.config_entries.options.async_configure(
        res_form["flow_id"], {"name": "  Water   boiler "}
    )
    assert res_done["type"] is FlowResultType.FORM
    assert res_done["step_id"] == "review"
    await hass.async_block_till_done()

    # The subentry title and data name are updated.
    sub = next(s for s in entry.get_subentries_of_type("device") if s.subentry_id == sub.subentry_id)
    assert sub.title == "Water boiler"
    assert sub.data["name"] == "Water boiler"

    # The reconcile mirrors the name into the cold record and the review record.
    rec = devices.record(dev_uuid)
    assert rec["name"] == "Water boiler"
    dec = sup.review.decisions[sig_kettle]
    assert dec["name"] == "Water boiler"
    assert dec["status"] == "named"

    # The device registry follows; entity ids and unique ids never move.
    device_entry = dev_reg.async_get_device_by_identifier(
        ("powerguesser", dev_uuid), entry.entry_id
    )
    assert device_entry is not None
    assert device_entry.name == "Water boiler"
    assert _device_ids(hass, entry, dev_uuid) == ids

    # The review label shows the new name.
    opts = res_done["data_schema"].schema["candidate"].config["options"]
    assert opts[0]["label"] == "Water boiler — confirmed device"


async def test_device_menu_never_offers_confirm_or_dismiss(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    aioclient_mock: Any,
    inprocess_worker: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The device menu only ever offers Accept update, Restore and Rename."""
    entry = await _setup_learning_env(hass, energy_manager, std_prefs, monkeypatch)
    mock_llm(aioclient_mock, "http://llm.test/v1", _kettle_responder("Kettle"))
    sup = entry.runtime_data.learning

    await _run_learning(sup)
    sig_kettle = sup.visible_candidates()[0]["signature_id"]
    await async_confirm_candidate(hass, entry, sig_kettle, "Kettle")
    sub = next(s for s in entry.get_subentries_of_type("device") if s.title == "Kettle")
    dev_uuid = sub.data["device_uuid"]

    res_dev = await _open_device_menu(hass, entry, dev_uuid)
    assert set(res_dev["menu_options"]) == {"rename"}

    devices = entry.runtime_data.devices
    await devices.async_set_proposal(
        dev_uuid,
        copy.deepcopy(devices.record(dev_uuid)["versions"][0]["artefact"]),
        run_id="run-x",
        proposed_at="2026-11-19T00:00:00Z",
        trained_at="2026-11-19T00:00:00Z",
    )
    res_dev = await _open_device_menu(hass, entry, dev_uuid)
    assert set(res_dev["menu_options"]) == {"accept_update", "rename"}

    await devices.async_accept_proposal(dev_uuid)
    res_dev = await _open_device_menu(hass, entry, dev_uuid)
    assert set(res_dev["menu_options"]) == {"rollback", "rename"}


# --- D-17: the proposal gate (Plan 05-06) ---


async def _confirm_kettle(
    hass: HomeAssistant, entry: ConfigEntry, sup: Any
) -> tuple[str, str, dict[str, Any]]:
    """Run one learning round, confirm the kettle and return (sig, uuid, v1 artefact)."""
    devices = entry.runtime_data.devices
    await _run_learning(sup)
    sig_kettle = sup.visible_candidates()[0]["signature_id"]
    await async_confirm_candidate(hass, entry, sig_kettle, "Kettle")
    sub = next(s for s in entry.get_subentries_of_type("device") if s.title == "Kettle")
    dev_uuid = sub.data["device_uuid"]
    v1_artefact = copy.deepcopy(devices.record(dev_uuid)["versions"][0]["artefact"])
    return sig_kettle, dev_uuid, v1_artefact


def _differing_entry(v1_artefact: dict[str, Any], sig_kettle: str) -> dict[str, Any]:
    """A matched library entry whose states differ from the active version."""
    entry = copy.deepcopy(v1_artefact)
    entry["signature_id"] = sig_kettle
    for st in entry["states"]:
        st["rise_w"] = {"lo": 600.0, "p50": 700.0, "hi": 800.0}
        st["plateau_w"] = {"lo": 600.0, "p50": 700.0, "hi": 800.0}
    return entry


async def test_compile_job_carries_actives(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    aioclient_mock: Any,
    inprocess_worker: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """D-17: after a confirm, the run's compile job carries the device's frozen
    active artefact under its signature id; the shadow pass sends an empty
    actives map, so shadow artefacts never produce a gate entry."""
    from pathlib import Path

    from custom_components.powerguesser.const import BUFFER_DIR, LEARNING_DIR

    # 35 usable days so the masked shadow pass actually runs on the first run.
    monkeypatch.setattr("custom_components.powerguesser.learning.MAX_REFINE_ROUNDS", 0)
    energy_manager.data = std_prefs
    _set_std_powers(hass)
    entry = await async_setup_via_flow(hass)
    spec = SynthSpec(days=35, oven=True, water_heater=True)
    await async_fill_synth_buffer(hass, entry, spec=spec)
    await async_seed_llm(hass, entry)
    await async_seed_consent(hass, host="llm.test", names=False)

    call_count = 0

    def _responder(data: Any) -> dict[str, Any]:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _kettle_responder("Kettle")(data)
        return _kettle_wh_responder("Kettle")(data)

    mock_llm(aioclient_mock, "http://llm.test/v1", _responder)
    sup = entry.runtime_data.learning
    devices = entry.runtime_data.devices

    # --- Run 1 (no confirmed device yet): both jobs carry empty actives ---
    await _run_learning(sup)
    sig_kettle = sup.visible_candidates()[0]["signature_id"]
    run1_dir = (
        Path(hass.config.path(BUFFER_DIR, LEARNING_DIR))
        / str(sup.coordinator.data["last_run"]["run_id"])
    )
    run1_job = json.loads((run1_dir / "job_compile_1.json").read_text(encoding="utf-8"))
    assert run1_job["actives"] == {}
    assert (run1_dir / "shadow" / "job_compile_1.json").is_file()
    shadow_job = json.loads(
        (run1_dir / "shadow" / "job_compile_1.json").read_text(encoding="utf-8")
    )
    assert shadow_job["actives"] == {}
    # The shadow library carries no gate entries.
    shadow_lib = json.loads(
        (run1_dir / "shadow" / "library_1.json").read_text(encoding="utf-8")
    )
    assert all("proposal_gate" not in a for a in shadow_lib)

    # --- Confirm, run 2 (a two-state proposal): the production job carries
    # the frozen v1 artefact and the gate passes the no-worse proposal ---
    await async_confirm_candidate(hass, entry, sig_kettle, "Kettle")
    sub = next(s for s in entry.get_subentries_of_type("device") if s.title == "Kettle")
    dev_uuid = sub.data["device_uuid"]
    v1_artefact = devices.record(dev_uuid)["versions"][0]["artefact"]

    await _run_learning(sup)
    run2_dir = (
        Path(hass.config.path(BUFFER_DIR, LEARNING_DIR))
        / str(sup.coordinator.data["last_run"]["run_id"])
    )
    run2_job = json.loads((run2_dir / "job_compile_1.json").read_text(encoding="utf-8"))
    assert set(run2_job["actives"].keys()) == {sig_kettle}
    assert run2_job["actives"][sig_kettle] == v1_artefact
    run2_lib = json.loads(
        (run2_dir / "library_1.json").read_text(encoding="utf-8")
    )
    kettle_entry = next(c for c in run2_lib if c["signature_id"] == sig_kettle)
    gate = kettle_entry["proposal_gate"]
    assert gate["passed"] is True
    assert gate["proposed_S"] >= gate["active_S"]
    assert gate["basis"] == "learning_split_val"
    assert devices.record(dev_uuid)["pending_proposal"] is not None


async def test_worse_proposal_not_offered_and_recorded(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    aioclient_mock: Any,
    inprocess_worker: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """D-17: a differing matched artefact whose proposal_gate failed is not
    offered; validation.last_rejected_proposal records both scores, the basis
    and the reason, and nothing else changes."""
    entry = await _setup_learning_env(hass, energy_manager, std_prefs, monkeypatch)
    mock_llm(aioclient_mock, "http://llm.test/v1", _kettle_responder("Kettle"))
    sup = entry.runtime_data.learning
    devices = entry.runtime_data.devices
    sig_kettle, dev_uuid, v1_artefact = await _confirm_kettle(hass, entry, sup)

    blob = json.dumps(devices.record(dev_uuid), sort_keys=True, default=str)
    art = _differing_entry(v1_artefact, sig_kettle)
    art["proposal_gate"] = {
        "passed": False,
        "proposed_S": 0.01234,
        "active_S": 0.05678,
        "basis": "learning_split_val",
    }
    await sup._route_device_matches([art], "run-reject", "2026-11-21T10:00:00Z")

    rec = devices.record(dev_uuid)
    assert rec["pending_proposal"] is None
    assert [v["v"] for v in rec["versions"]] == [1]
    assert rec["active_v"] == 1
    rejected = rec["validation"]["last_rejected_proposal"]
    assert rejected["run_id"] == "run-reject"
    assert rejected["reason"] == "worse_than_active"
    assert rejected["proposed_S"] == 0.01234
    assert rejected["active_S"] == 0.05678
    assert rejected["basis"] == "learning_split_val"
    assert "rejected_at" in rejected
    # The record changed only in the diagnostics key.
    cleared = copy.deepcopy(rec)
    cleared["validation"] = copy.deepcopy(json.loads(blob)["validation"])
    assert json.dumps(cleared, sort_keys=True, default=str) == json.dumps(
        json.loads(blob), sort_keys=True, default=str
    )


async def test_ungated_proposal_not_offered_and_recorded(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    aioclient_mock: Any,
    inprocess_worker: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """D-17: a differing matched artefact with no proposal_gate (a device
    confirmed while the run was in flight) is not offered and records
    reason not_gated."""
    entry = await _setup_learning_env(hass, energy_manager, std_prefs, monkeypatch)
    mock_llm(aioclient_mock, "http://llm.test/v1", _kettle_responder("Kettle"))
    sup = entry.runtime_data.learning
    devices = entry.runtime_data.devices
    sig_kettle, dev_uuid, v1_artefact = await _confirm_kettle(hass, entry, sup)

    art = _differing_entry(v1_artefact, sig_kettle)
    assert "proposal_gate" not in art
    await sup._route_device_matches([art], "run-ungated", "2026-11-21T10:05:00Z")

    rec = devices.record(dev_uuid)
    assert rec["pending_proposal"] is None
    rejected = rec["validation"]["last_rejected_proposal"]
    assert rejected["run_id"] == "run-ungated"
    assert rejected["reason"] == "not_gated"
    assert rejected["proposed_S"] is None
    assert rejected["active_S"] is None
    assert rejected["basis"] is None


async def test_gated_proposal_offered(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    aioclient_mock: Any,
    inprocess_worker: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """D-17: a differing matched artefact whose proposal_gate passed is offered
    as the device's pending proposal (the run scores it no worse than the
    frozen active version on the same validation block)."""
    entry = await _setup_learning_env(hass, energy_manager, std_prefs, monkeypatch)
    mock_llm(aioclient_mock, "http://llm.test/v1", _kettle_responder("Kettle"))
    sup = entry.runtime_data.learning
    devices = entry.runtime_data.devices
    sig_kettle, dev_uuid, v1_artefact = await _confirm_kettle(hass, entry, sup)

    art = _differing_entry(v1_artefact, sig_kettle)
    art["proposal_gate"] = {
        "passed": True,
        "proposed_S": 0.05678,
        "active_S": 0.05678,
        "basis": "learning_split_val",
    }
    await sup._route_device_matches([art], "run-pass", "2026-11-21T10:10:00Z")

    rec = devices.record(dev_uuid)
    assert rec["pending_proposal"] is not None
    assert rec["pending_proposal"]["run_id"] == "run-pass"
    assert "last_rejected_proposal" not in rec["validation"]

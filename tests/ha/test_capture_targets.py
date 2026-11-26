"""HA tests for capture target kinds: preview refit, learned candidate, device (CAP-03, CAP-09, D-04).

Plan 08-05 Task 1: a capture can target a new device, an existing preview, a
learned candidate (attach-only) or a confirmed device (a read-only replay block);
device targets resolve through the device registry or, without a picked device,
by the confirmed record's name or uuid.
"""

import copy
from datetime import datetime, timezone
import json
from typing import Any

import pytest

from homeassistant.components.energy.data import EnergyManager
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType
from homeassistant.exceptions import ServiceValidationError
from homeassistant.helpers import device_registry as dr, issue_registry as ir
from homeassistant.util import dt as dt_util
from pytest_homeassistant.common import async_fire_time_changed

from custom_components.powerguesser.const import (
    CAPTURES_STORE_KEY,
    DOMAIN,
    ISSUE_CAPTURE_ORPHANS,
)
from custom_components.powerguesser.engine.intervals import ProfileSnapshot, gate
from tests.conftest import (
    async_confirm_candidate,
    async_fill_synth_buffer,
    async_open_options_step,
    async_set_power,
    async_setup_via_flow,
    make_switched_candidate,
    seed_candidates,
)
from tests.fixtures.learning.synth import SynthSpec, generate

GRID_S = 30
BASIS = "active version alone on the capture window"


def _pin_grid(entry: Any) -> None:
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


def _stored_captures(hass_storage: dict[str, Any]) -> list[dict[str, Any]]:
    return hass_storage.get(CAPTURES_STORE_KEY, {}).get("data", {}).get("captures", [])


async def _setup(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    *,
    candidates: list[dict[str, Any]] | None = None,
    spec: SynthSpec | None = None,
) -> tuple[Any, Any]:
    if candidates:
        seed_candidates(hass_storage, candidates)
    energy_manager.data = copy.deepcopy(std_prefs)
    async_set_power(hass, "sensor.grid_power", 500)
    async_set_power(hass, "sensor.solar_power", 0)
    async_set_power(hass, "sensor.battery_power", 0)
    async_set_power(hass, "sensor.washer_power", 0)
    entry = await async_setup_via_flow(hass)
    _pin_grid(entry)
    await hass.async_block_till_done()
    data = await async_fill_synth_buffer(hass, entry, spec or SynthSpec(days=3))
    return entry, data


async def _capture(
    hass: HomeAssistant,
    *,
    start_ms: int,
    end_ms: int,
    label: str = "run",
    target_type: str = "new_device",
    target: str = "Kettle",
    device_id: str | None = None,
) -> dict[str, Any]:
    data: dict[str, Any] = {
        "start": _iso_utc(start_ms),
        "end": _iso_utc(end_ms),
        "label": label,
        "target_type": target_type,
        "target": target,
    }
    if device_id is not None:
        data["device_id"] = device_id
    return await hass.services.async_call(
        DOMAIN, "capture_window", data, blocking=True, return_response=True
    )


async def _capture_list(hass: HomeAssistant, capture_id: str | None = None) -> dict[str, Any]:
    data: dict[str, Any] = {} if capture_id is None else {"capture_id": capture_id}
    return await hass.services.async_call(
        DOMAIN, "capture_list", data, blocking=True, return_response=True
    )


async def _delete(hass: HomeAssistant, capture_id: str) -> dict[str, Any]:
    return await hass.services.async_call(
        DOMAIN, "capture_delete", {"capture_id": capture_id}, blocking=True, return_response=True
    )


async def _reassign(
    hass: HomeAssistant,
    capture_id: str,
    *,
    target_type: str,
    target: str,
    device_id: str | None = None,
) -> dict[str, Any]:
    data: dict[str, Any] = {
        "capture_id": capture_id,
        "target_type": target_type,
        "target": target,
    }
    if device_id is not None:
        data["device_id"] = device_id
    return await hass.services.async_call(
        DOMAIN, "capture_reassign", data, blocking=True, return_response=True
    )


async def _confirm(hass: HomeAssistant, entry: Any, sig: str, name: str) -> tuple[str, str]:
    """Confirm a seeded candidate and return (device record uuid, registry device id)."""
    await async_confirm_candidate(hass, entry, sig, name)
    sub = next(s for s in entry.get_subentries_of_type("device") if s.title == name)
    uuid = sub.data["device_uuid"]
    device = dr.async_get(hass).async_get_device_by_identifier(
        (DOMAIN, uuid), config_entry_id=entry.entry_id
    )
    assert device is not None
    return uuid, device.id


def _device_snapshot(devices: Any, uuid: str) -> dict[str, Any]:
    rec = devices.record(uuid)
    return {
        "versions": json.dumps(rec["versions"], sort_keys=True),
        "active_v": rec["active_v"],
        "pending": rec["pending_proposal"],
        "energy": devices._energy_wh.get(uuid, 0.0),
    }


# ---------------------------------------------------------------------------
# Preview refit, learned candidate, ambiguity
# ---------------------------------------------------------------------------


async def test_capture_into_existing_preview_refits(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
) -> None:
    """A new-device capture then two candidate captures join one preview and refit."""
    entry, data = await _setup(hass, energy_manager, std_prefs, hass_storage)
    hub = entry.runtime_data
    on_ms, off_ms = data.kettle_runs[-1]

    first = await _capture(hass, start_ms=on_ms - 45_000, end_ms=off_ms + 50_000)
    sig = first["target"]["id"]
    assert sig.startswith("cap_")

    for _ in range(2):
        joined = await _capture(
            hass,
            start_ms=on_ms - 45_000,
            end_ms=off_ms + 50_000,
            target_type="candidate",
            target="kettle",
        )
        assert joined["target"]["id"] == sig

    visible = hub.learning.visible_candidates()
    assert len(visible) == 1
    cand = visible[0]
    assert cand["signature_id"] == sig
    assert cand["scores"]["n_examples"] == 3
    assert cand["provenance"]["fit"] == "percentile"
    assert "from 3 captured examples" in hub.learning.candidate_label(cand)


async def test_capture_on_learned_candidate_attaches_only(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
) -> None:
    """A capture aimed at a learned candidate attaches as an example, no new card."""
    cand = make_switched_candidate("sig-kettle", "Kettle")
    entry, data = await _setup(
        hass, energy_manager, std_prefs, hass_storage, candidates=[cand]
    )
    hub = entry.runtime_data
    on_ms, off_ms = data.kettle_runs[-1]
    before = len(hub.learning.visible_candidates())

    res = await _capture(
        hass,
        start_ms=on_ms - 45_000,
        end_ms=off_ms + 50_000,
        target_type="candidate",
        target="sig-kettle",
    )
    assert res["target"]["kind"] == "candidate"
    assert res["target"]["id"] == "sig-kettle"
    assert res["attached_as"] == "candidate_example"
    assert len(hub.learning.visible_candidates()) == before


async def test_unknown_and_ambiguous_candidate_targets(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
) -> None:
    """An unknown target lists choices; two same-name candidates are ambiguous; no record."""
    candidates = [
        make_switched_candidate("sig-a", "Kettle"),
        make_switched_candidate("sig-b", "kettle "),
    ]
    entry, data = await _setup(
        hass, energy_manager, std_prefs, hass_storage, candidates=candidates
    )
    on_ms, off_ms = data.kettle_runs[-1]

    with pytest.raises(ServiceValidationError) as unknown:
        await _capture(
            hass,
            start_ms=on_ms - 45_000,
            end_ms=off_ms + 50_000,
            target_type="candidate",
            target="nope",
        )
    assert unknown.value.translation_key == "capture_target_unknown"
    assert "Kettle" in unknown.value.translation_placeholders["choices"]

    with pytest.raises(ServiceValidationError) as ambiguous:
        await _capture(
            hass,
            start_ms=on_ms - 45_000,
            end_ms=off_ms + 50_000,
            target_type="candidate",
            target="Kettle",
        )
    assert ambiguous.value.translation_key == "capture_target_ambiguous"

    assert _stored_captures(hass_storage) == []


# ---------------------------------------------------------------------------
# Device targets (D-04)
# ---------------------------------------------------------------------------


async def test_device_target_replay_is_read_only(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
) -> None:
    """A device target reports a read-only replay block and changes no device state."""
    cand = make_switched_candidate("sig-kettle", "Kettle")
    entry, data = await _setup(
        hass, energy_manager, std_prefs, hass_storage, candidates=[cand]
    )
    hub = entry.runtime_data
    uuid, dev_id = await _confirm(hass, entry, "sig-kettle", "Kettle")
    before = _device_snapshot(hub.devices, uuid)
    on_ms, off_ms = data.kettle_runs[-1]

    res = await _capture(
        hass,
        start_ms=on_ms - 45_000,
        end_ms=off_ms + 50_000,
        target_type="device",
        target="Kettle",
        device_id=dev_id,
    )
    assert res["target"]["kind"] == "device"
    assert res["target"]["id"] == uuid
    assert res["replay"]["would_claim"] is True
    assert res["replay"]["basis"] == BASIS
    assert res["attached_as"] == "device_example"

    listed = await _capture_list(hass, res["capture_id"])
    assert listed["captures"][0]["replay"] == res["replay"]

    assert _device_snapshot(hub.devices, uuid) == before

    foreign = dr.async_get(hass).async_get_or_create(
        identifiers={("other", "x")}, config_entry_id=entry.entry_id
    )
    with pytest.raises(ServiceValidationError) as unknown:
        await _capture(
            hass,
            start_ms=on_ms - 45_000,
            end_ms=off_ms + 50_000,
            target_type="device",
            target="Foreign",
            device_id=foreign.id,
        )
    assert unknown.value.translation_key == "capture_target_unknown"


async def test_device_registry_missing_targets_by_name(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
) -> None:
    """A confirmed device without a registry entry stays targetable by name or uuid."""
    cand = make_switched_candidate("sig-kettle", "Kettle")
    entry, data = await _setup(
        hass, energy_manager, std_prefs, hass_storage, candidates=[cand]
    )
    hub = entry.runtime_data
    uuid, dev_id = await _confirm(hass, entry, "sig-kettle", "Kettle")
    before = _device_snapshot(hub.devices, uuid)

    dr.async_get(hass).async_remove_device(dev_id)
    await hass.async_block_till_done()

    on_ms, off_ms = data.kettle_runs[-1]
    with pytest.raises(ServiceValidationError) as unknown:
        await _capture(
            hass,
            start_ms=on_ms - 45_000,
            end_ms=off_ms + 50_000,
            target_type="device",
            target="Kettle",
            device_id=dev_id,
        )
    assert unknown.value.translation_key == "capture_target_unknown"
    assert "Kettle" in unknown.value.translation_placeholders["choices"]

    by_name = await _capture(
        hass,
        start_ms=on_ms - 45_000,
        end_ms=off_ms + 50_000,
        target_type="device",
        target="kettle",
    )
    assert by_name["target"]["kind"] == "device"
    assert by_name["target"]["id"] == uuid
    assert by_name["replay"]["would_claim"] is True

    by_uuid = await _capture(
        hass,
        start_ms=on_ms - 45_000,
        end_ms=off_ms + 50_000,
        target_type="device",
        target=uuid,
    )
    assert by_uuid["target"]["id"] == uuid
    assert by_uuid["replay"]["would_claim"] is True

    assert _device_snapshot(hub.devices, uuid) == before


# ---------------------------------------------------------------------------
# Orphaned captures (D-06)
# ---------------------------------------------------------------------------


async def test_dismissed_preview_keeps_examples_as_orphans(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
) -> None:
    """A dismissed preview keeps its captures as orphans; the issue clears when they move."""
    entry, data = await _setup(hass, energy_manager, std_prefs, hass_storage)
    hub = entry.runtime_data
    on_ms, off_ms = data.kettle_runs[-1]

    first = await _capture(hass, start_ms=on_ms - 45_000, end_ms=off_ms + 50_000)
    sig = first["target"]["id"]
    second = await _capture(
        hass,
        start_ms=on_ms - 45_000,
        end_ms=off_ms + 50_000,
        target_type="candidate",
        target="Kettle",
    )
    assert second["target"]["id"] == sig
    assert len(hub.learning.visible_candidates()) == 1

    # Dismiss the preview through the Review flow.
    res_review = await async_open_options_step(hass, entry, "review")
    res_cand = await hass.config_entries.options.async_configure(
        res_review["flow_id"], {"candidate": sig}
    )
    assert res_cand["type"] is FlowResultType.MENU
    await hass.config_entries.options.async_configure(
        res_cand["flow_id"], {"next_step_id": "dismiss"}
    )

    assert len(_stored_captures(hass_storage)) == 2
    listed = await _capture_list(hass)
    assert all(c["target_state"] == "target_gone" for c in listed["captures"])
    assert hub.learning.visible_candidates() == []

    issue_key = (DOMAIN, ISSUE_CAPTURE_ORPHANS)
    assert issue_key in ir.async_get(hass).issues
    assert ir.async_get(hass).issues[issue_key].translation_placeholders["count"] == "2"

    # Re-assign one capture out of the orphan set and delete the other.
    await _reassign(hass, first["capture_id"], target_type="new_device", target="Kettle")
    assert ir.async_get(hass).issues[issue_key].translation_placeholders["count"] == "1"

    out = await _delete(hass, second["capture_id"])
    assert out["deleted"] is True
    assert issue_key not in ir.async_get(hass).issues


# ---------------------------------------------------------------------------
# Live captures accept every target kind
# ---------------------------------------------------------------------------

_SPEC = SynthSpec(days=3, interval_s=(30.0, 30.0))
_END = int(datetime(2026, 11, 21, 23, 0, tzinfo=timezone.utc).timestamp() * 1000)
_TRUTH = generate(_SPEC, _END)


async def _setup_live(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
    monkeypatch: pytest.MonkeyPatch,
    *,
    now_ms: int,
    candidates: list[dict[str, Any]] | None = None,
) -> Any:
    if candidates:
        seed_candidates(hass_storage, candidates)
    freezer.move_to(dt_util.utc_from_timestamp(now_ms / 1000))
    energy_manager.data = copy.deepcopy(std_prefs)
    async_set_power(hass, "sensor.grid_power", 250)
    async_set_power(hass, "sensor.solar_power", 0)
    async_set_power(hass, "sensor.battery_power", 0)
    async_set_power(hass, "sensor.washer_power", 0)
    entry = await async_setup_via_flow(hass)
    _pin_grid(entry)
    await hass.async_block_till_done()
    await async_fill_synth_buffer(hass, entry, _SPEC, end_ms=_END)
    writer = entry.runtime_data.writer
    monkeypatch.setattr(writer, "mark_stale", lambda *_a, **_k: None)

    async def _noop(*_a: Any, **_k: Any) -> None:
        return None

    monkeypatch.setattr(writer, "async_flush", _noop)
    return entry


async def _advance(hass: HomeAssistant, freezer: Any, seconds: float) -> None:
    from datetime import timedelta

    freezer.tick(timedelta(seconds=seconds))
    async_fire_time_changed(hass, dt_util.utcnow())
    await hass.async_block_till_done()


async def _start(
    hass: HomeAssistant,
    *,
    target_type: str = "new_device",
    target: str = "Kettle",
    device_id: str | None = None,
) -> dict[str, Any]:
    data: dict[str, Any] = {"label": "run", "target_type": target_type, "target": target}
    if device_id is not None:
        data["device_id"] = device_id
    return await hass.services.async_call(
        DOMAIN, "capture_start", data, blocking=True, return_response=True
    )


async def _stop(hass: HomeAssistant, capture_id: str) -> dict[str, Any]:
    return await hass.services.async_call(
        DOMAIN, "capture_stop", {"capture_id": capture_id}, blocking=True, return_response=True
    )


async def test_live_capture_accepts_all_target_types(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """capture_start resolves candidate and device targets and stores the same kinds."""
    run_a = _TRUTH.kettle_runs[-1]
    run_b = _TRUTH.kettle_runs[-2]
    candidates = [
        make_switched_candidate("sig-kettle", "Kettle"),
        make_switched_candidate("sig-cand", "Toaster"),
    ]
    entry = await _setup_live(
        hass,
        energy_manager,
        std_prefs,
        hass_storage,
        freezer,
        monkeypatch,
        now_ms=run_a[0] - 60_000,
        candidates=candidates,
    )
    uuid, dev_id = await _confirm(hass, entry, "sig-kettle", "Kettle")

    # Device target live at run_a.
    await _advance(hass, freezer, 80)
    r1 = await _start(hass, target_type="device", target="Kettle", device_id=dev_id)
    await _advance(hass, freezer, (run_a[1] - run_a[0]) / 1000)
    await _stop(hass, r1["capture_id"])
    await _advance(hass, freezer, 90)

    e1 = (await _capture_list(hass, r1["capture_id"]))["captures"][0]
    assert e1["target_kind"] == "device"
    assert e1["target"]["id"] == uuid
    assert e1["replay"] is not None
    assert e1["replay"]["would_claim"] is True
    assert e1["replay"]["basis"] == BASIS

    # Candidate target live at run_b (move the frozen clock back to that run).
    freezer.move_to(dt_util.utc_from_timestamp((run_b[0] - 60_000) / 1000))
    async_fire_time_changed(hass, dt_util.utcnow())
    await hass.async_block_till_done()
    await _advance(hass, freezer, 80)
    r2 = await _start(hass, target_type="candidate", target="sig-cand")
    await _advance(hass, freezer, (run_b[1] - run_b[0]) / 1000)
    await _stop(hass, r2["capture_id"])
    await _advance(hass, freezer, 90)

    e2 = (await _capture_list(hass, r2["capture_id"]))["captures"][0]
    assert e2["target_kind"] == "candidate"
    assert e2["target"]["id"] == "sig-cand"
    assert e2["attached_as"] == "candidate_example"

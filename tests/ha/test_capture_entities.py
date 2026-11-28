"""HA tests for the hub capture entities and the tracer path (CAP-07, D-01..D-03).

Plan 09-01: the dashboard surface -- a status enum sensor, Start/Stop buttons, a
target-kind select and a label text entity -- present one CaptureManager through
one dispatcher signal. A user types a label, presses Start, and an automation
gets one label-free ``capture_finished`` event when the capture settles.
"""

import copy
import json
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from homeassistant.components.energy.data import EnergyManager
from homeassistant.const import STATE_UNAVAILABLE, STATE_UNKNOWN
from homeassistant.core import HomeAssistant, State
from homeassistant.exceptions import ServiceValidationError
from homeassistant.helpers import entity_registry as er
from homeassistant.util import dt as dt_util
from pytest_homeassistant.common import (
    async_capture_events,
    async_fire_time_changed,
    mock_restore_cache,
)

from custom_components.powerguesser.const import CAPTURES_STORE_KEY, DOMAIN
from custom_components.powerguesser.engine.intervals import ProfileSnapshot, gate
from custom_components.powerguesser.sensor import CaptureStatusSensor
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
_SPEC = SynthSpec(days=3, interval_s=(30.0, 30.0))
_END = int(datetime(2026, 11, 21, 23, 0, tzinfo=timezone.utc).timestamp() * 1000)
_TRUTH = generate(_SPEC, _END)


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


def _stored_captures(hass_storage: dict[str, Any]) -> list[dict[str, Any]]:
    return hass_storage.get(CAPTURES_STORE_KEY, {}).get("data", {}).get("captures", [])


async def _setup(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    freezer: Any,
    monkeypatch: pytest.MonkeyPatch,
    now_ms: int,
) -> Any:
    """Set up an entry at a frozen time, fill the buffer, silence the live writer."""
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
    freezer.tick(timedelta(seconds=seconds))
    async_fire_time_changed(hass, dt_util.utcnow())
    await hass.async_block_till_done()


async def _set_label(hass: HomeAssistant, value: str) -> None:
    await hass.services.async_call(
        "text",
        "set_value",
        {"entity_id": "text.powerguesser_capture_label", "value": value},
        blocking=True,
    )


async def _press(hass: HomeAssistant, entity_id: str) -> None:
    await hass.services.async_call(
        "button", "press", {"entity_id": entity_id}, blocking=True
    )


async def test_tracer_buttons_drive_a_capture_and_fire_the_event(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Type a label, Start then Stop; the sensor follows the capture; one event fires."""
    on_ms, off_ms = _TRUTH.kettle_runs[-1]
    await _setup(hass, energy_manager, std_prefs, freezer, monkeypatch, on_ms - 60_000)
    events = async_capture_events(hass, DOMAIN)

    await _advance(hass, freezer, 80)

    await _set_label(hass, "Kettle")
    assert hass.states.get("text.powerguesser_capture_label").state == "Kettle"

    await _press(hass, "button.powerguesser_start_capture")
    await hass.async_block_till_done()

    status = hass.states.get("sensor.powerguesser_capture_status")
    assert status is not None
    assert status.state == "recording"
    assert status.attributes["target_kind"] == "candidate"
    assert status.attributes["target_name"] == "Kettle"
    assert status.attributes["capture_id"]
    assert status.attributes["started_at"]
    # The label cleared only after the successful start (D-02).
    assert hass.states.get("text.powerguesser_capture_label").state == ""

    await _advance(hass, freezer, (off_ms - on_ms) / 1000)
    await _press(hass, "button.powerguesser_stop_capture")
    await hass.async_block_till_done()
    assert hass.states.get("sensor.powerguesser_capture_status").state == "resolving"

    await _advance(hass, freezer, 90)

    status = hass.states.get("sensor.powerguesser_capture_status")
    assert status.state == "idle"
    assert status.attributes["last_result"] is not None
    assert status.attributes["last_result"]["quality"] is not None

    record = _stored_captures(hass_storage)[0]
    assert len(events) == 1
    assert events[0].data["type"] == "capture_finished"
    assert events[0].data["state"] == record["state"]


# ---------------------------------------------------------------------------
# Task 2: target select, restore, unlabelled captures and label refusals
# ---------------------------------------------------------------------------


async def _setup_light(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
) -> Any:
    """Set up an entry with no buffer (entity/restore tests)."""
    energy_manager.data = copy.deepcopy(std_prefs)
    async_set_power(hass, "sensor.grid_power", 250)
    async_set_power(hass, "sensor.solar_power", 0)
    async_set_power(hass, "sensor.battery_power", 0)
    async_set_power(hass, "sensor.washer_power", 0)
    entry = await async_setup_via_flow(hass)
    await hass.async_block_till_done()
    return entry


async def _set_select(hass: HomeAssistant, option: str) -> None:
    await hass.services.async_call(
        "select",
        "select_option",
        {"entity_id": "select.powerguesser_capture_target", "option": option},
        blocking=True,
    )


async def _cancel(hass: HomeAssistant) -> dict[str, Any]:
    return await hass.services.async_call(
        DOMAIN, "capture_cancel", {}, blocking=True, return_response=True
    )


async def _capture_list(hass: HomeAssistant) -> dict[str, Any]:
    return await hass.services.async_call(
        DOMAIN, "capture_list", {}, blocking=True, return_response=True
    )


async def test_entity_ids_and_options_are_pinned(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
) -> None:
    """The five capture entities carry the fixed ids and the select's options."""
    await _setup_light(hass, energy_manager, std_prefs)
    registry = er.async_get(hass)
    expected = {
        "sensor.powerguesser_capture_status": "powerguesser_capture_status",
        "button.powerguesser_start_capture": "powerguesser_capture_start",
        "button.powerguesser_stop_capture": "powerguesser_capture_stop",
        "select.powerguesser_capture_target": "powerguesser_capture_target",
        "text.powerguesser_capture_label": "powerguesser_capture_label",
    }
    for entity_id, unique_id in expected.items():
        entry_reg = registry.async_get(entity_id)
        assert entry_reg is not None, f"{entity_id} missing from the registry"
        assert entry_reg.unique_id == unique_id

    select = hass.states.get("select.powerguesser_capture_target")
    assert select.attributes["options"] == ["new_device", "candidate", "device"]
    assert select.state == "new_device"


async def test_select_and_label_restore(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
) -> None:
    """A restored select and label value survive setup (D-01)."""
    mock_restore_cache(
        hass,
        [
            State("select.powerguesser_capture_target", "device"),
            State("text.powerguesser_capture_label", "Kettle"),
        ],
    )
    await _setup_light(hass, energy_manager, std_prefs)
    assert hass.states.get("select.powerguesser_capture_target").state == "device"
    assert hass.states.get("text.powerguesser_capture_label").state == "Kettle"


async def test_restored_select_bogus_falls_back(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
) -> None:
    """A restored select value outside the options falls back to new_device."""
    mock_restore_cache(hass, [State("select.powerguesser_capture_target", "bogus")])
    await _setup_light(hass, energy_manager, std_prefs)
    assert hass.states.get("select.powerguesser_capture_target").state == "new_device"


async def test_unlabelled_new_device_capture_names_later(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty label with new_device starts an unlabelled capture, named in Review."""
    on_ms, off_ms = _TRUTH.kettle_runs[-1]
    entry = await _setup(
        hass, energy_manager, std_prefs, freezer, monkeypatch, on_ms - 60_000
    )
    hub = entry.runtime_data

    await _advance(hass, freezer, 80)
    # The select defaults to new_device and the label is empty.
    await _press(hass, "button.powerguesser_start_capture")
    await hass.async_block_till_done()
    assert hass.states.get("sensor.powerguesser_capture_status").state == "recording"
    record = _stored_captures(hass_storage)[0]
    assert record["label"] == ""
    assert record["new_device_name"] is None

    await _advance(hass, freezer, (off_ms - on_ms) / 1000)
    await _press(hass, "button.powerguesser_stop_capture")
    await _advance(hass, freezer, 90)

    visible = hub.learning.visible_candidates()
    assert len(visible) == 1
    assert hub.learning.candidate_label(visible[0]).startswith("Unnamed capture")
    sig = visible[0]["signature_id"]

    # Rename the preview through the options-flow rename step.
    res_review = await async_open_options_step(hass, entry, "review")
    res_cand = await hass.config_entries.options.async_configure(
        res_review["flow_id"], {"candidate": sig}
    )
    res_rename = await hass.config_entries.options.async_configure(
        res_cand["flow_id"], {"next_step_id": "rename"}
    )
    await hass.config_entries.options.async_configure(
        res_rename["flow_id"], {"name": "Kettle"}
    )

    listed = await _capture_list(hass)
    entry_list = next(
        c for c in listed["captures"] if c["capture_id"] == record["capture_id"]
    )
    assert entry_list["target"]["name"] == "Kettle"


async def test_empty_label_refused_for_candidate_and_device(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
) -> None:
    """An empty label on candidate or device is refused and creates no record."""
    await _setup_light(hass, energy_manager, std_prefs)

    for kind in ("candidate", "device"):
        await _set_select(hass, kind)
        with pytest.raises(ServiceValidationError) as excinfo:
            await _press(hass, "button.powerguesser_start_capture")
        assert excinfo.value.translation_key == "capture_label_required"
        assert excinfo.value.translation_placeholders["target_kind"] == kind

    assert _stored_captures(hass_storage) == []
    assert hass.states.get("text.powerguesser_capture_label").state == ""


async def test_label_targets_candidate_and_device(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The label names a candidate or a confirmed device; an unknown name is refused."""
    on_ms, _off_ms = _TRUTH.kettle_runs[-1]
    seed_candidates(hass_storage, [make_switched_candidate("sig-kettle", "Kettle")])
    entry = await _setup(
        hass, energy_manager, std_prefs, freezer, monkeypatch, on_ms - 60_000
    )

    # Candidate target by display name (while the candidate is still unconfirmed).
    await _advance(hass, freezer, 5)
    await _set_select(hass, "candidate")
    await _set_label(hass, "Kettle")
    await _press(hass, "button.powerguesser_start_capture")
    await hass.async_block_till_done()
    record = _stored_captures(hass_storage)[0]
    assert record["target_kind"] == "candidate"
    assert record["target_id"] == "sig-kettle"
    await _cancel(hass)
    await hass.async_block_till_done()

    # Confirm it -> a confirmed device; a device target resolves by name.
    await async_confirm_candidate(hass, entry, "sig-kettle", "Kettle")
    sub = next(s for s in entry.get_subentries_of_type("device") if s.title == "Kettle")
    uuid = sub.data["device_uuid"]

    await _set_select(hass, "device")
    await _set_label(hass, "Kettle")
    await _press(hass, "button.powerguesser_start_capture")
    await hass.async_block_till_done()
    record = _stored_captures(hass_storage)[1]
    assert record["target_kind"] == "device"
    assert record["target_id"] == uuid
    await _cancel(hass)
    await hass.async_block_till_done()

    # Unknown device name lists the choices.
    await _set_select(hass, "device")
    await _set_label(hass, "Nope")
    with pytest.raises(ServiceValidationError) as excinfo:
        await _press(hass, "button.powerguesser_start_capture")
    assert excinfo.value.translation_key == "capture_target_unknown"
    assert "Kettle" in excinfo.value.translation_placeholders["choices"]


async def test_invalid_label_refused(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
) -> None:
    """An over-long or control-character label is refused and the stored label stays.

    The 65-character value is refused by the entity's ``native_max`` (64) at the
    service layer; a value that passes the length but fails ``normalise_name``
    (a control character) is refused with ``capture_invalid_name``.
    """
    await _setup_light(hass, energy_manager, std_prefs)
    text_state = hass.states.get("text.powerguesser_capture_label")
    assert text_state.attributes["max"] == 64

    with pytest.raises(ServiceValidationError) as too_long:
        await _set_label(hass, "x" * 65)
    assert too_long.value.translation_key == "value_too_long"
    assert hass.states.get("text.powerguesser_capture_label").state == ""

    with pytest.raises(ServiceValidationError) as control:
        await _set_label(hass, "Ket\x00tle")
    assert control.value.translation_key == "capture_invalid_name"
    assert hass.states.get("text.powerguesser_capture_label").state == ""


async def test_refused_start_keeps_label(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A refused start (already open) leaves the typed label in place (D-02)."""
    on_ms, _off_ms = _TRUTH.kettle_runs[-1]
    await _setup(hass, energy_manager, std_prefs, freezer, monkeypatch, on_ms - 60_000)

    await _advance(hass, freezer, 80)
    await _set_label(hass, "Kettle")
    await _press(hass, "button.powerguesser_start_capture")
    await hass.async_block_till_done()
    assert hass.states.get("text.powerguesser_capture_label").state == ""

    await _set_label(hass, "Toaster")
    with pytest.raises(ServiceValidationError) as excinfo:
        await _press(hass, "button.powerguesser_start_capture")
    assert excinfo.value.translation_key == "capture_already_open"
    assert hass.states.get("text.powerguesser_capture_label").state == "Toaster"
    assert len(_stored_captures(hass_storage)) == 1


# ---------------------------------------------------------------------------
# CAP-07 D-03 (no ticking / unrecorded volatile attributes) and the CAP-07
# prohibition pinned behaviourally: the entities never write a confirmed device.
# ---------------------------------------------------------------------------


def test_status_sensor_volatile_attributes_are_unrecorded() -> None:
    """capture_id, target_name, started_at and last_result are recorder-unrecorded (D-03)."""
    assert CaptureStatusSensor._unrecorded_attributes == frozenset(
        {"capture_id", "target_name", "started_at", "last_result"}
    )


async def test_status_sensor_reports_no_ticking_attribute(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The status sensor is an enum that changes only on capture transitions (D-03).

    A websocket subscriber sees exactly the open capture's four volatile
    attributes plus target_kind, and nothing ticks: reading the same open capture
    thirty seconds later yields an identical attribute set (no elapsed time, no
    sample counter, no derived duration).
    """
    on_ms, _off_ms = _TRUTH.kettle_runs[-1]
    await _setup(hass, energy_manager, std_prefs, freezer, monkeypatch, on_ms - 60_000)
    await _advance(hass, freezer, 80)

    await _set_label(hass, "Ticker")
    await _press(hass, "button.powerguesser_start_capture")
    await hass.async_block_till_done()

    status = hass.states.get("sensor.powerguesser_capture_status")
    assert status.state == "recording"
    assert status.attributes["device_class"] == "enum"
    assert status.attributes["options"] == ["idle", "recording", "resolving"]

    # Only the four volatile attributes plus target_kind exist beyond HA's own keys.
    standard = {"device_class", "options", "friendly_name", "unit_of_measurement"}
    assert set(status.attributes) - standard == {
        "capture_id",
        "target_kind",
        "target_name",
        "started_at",
        "last_result",
    }
    assert status.attributes["capture_id"]
    assert status.attributes["target_name"] == "Ticker"
    assert status.attributes["started_at"]

    first = dict(status.attributes)
    await _advance(hass, freezer, 30)
    later = hass.states.get("sensor.powerguesser_capture_status")
    assert later.state == "recording"
    assert dict(later.attributes) == first


async def _setup_with_confirmed_device(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
    monkeypatch: pytest.MonkeyPatch,
    now_ms: int,
) -> tuple[Any, str]:
    """Entry with a synthetic buffer and one confirmed kettle device."""
    seed_candidates(
        hass_storage,
        [make_switched_candidate("sig-kettle", "Kettle", rise_w=2000.0)],
    )
    entry = await _setup(hass, energy_manager, std_prefs, freezer, monkeypatch, now_ms)
    await async_confirm_candidate(hass, entry, "sig-kettle", "Kettle")
    dev_uuid = entry.runtime_data.devices.device_for_signature("sig-kettle")
    assert dev_uuid is not None
    return entry, dev_uuid


async def test_capture_entities_never_write_a_confirmed_device(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rolling a full capture through the hub entities leaves the device untouched.

    The hub capture entities only start, stop and label captures (CAP-07
    prohibition, T-09-01): a start -> stop -> settle cycle aimed at a confirmed
    device must leave the device record byte-identical, with no new version and
    no pending proposal.
    """
    on_ms, off_ms = _TRUTH.kettle_runs[-1]
    entry, dev_uuid = await _setup_with_confirmed_device(
        hass, energy_manager, std_prefs, hass_storage, freezer, monkeypatch,
        on_ms - 60_000,
    )
    devices = entry.runtime_data.devices
    before = json.dumps(devices.record(dev_uuid), sort_keys=True)

    await _set_select(hass, "device")
    await _set_label(hass, "Kettle")
    await _press(hass, "button.powerguesser_start_capture")
    await hass.async_block_till_done()
    assert hass.states.get("sensor.powerguesser_capture_status").state == "recording"
    assert hass.states.get("sensor.powerguesser_capture_status").attributes[
        "target_kind"
    ] == "device"

    await _advance(hass, freezer, 80)
    await _advance(hass, freezer, (off_ms - on_ms) / 1000)
    await _press(hass, "button.powerguesser_stop_capture")
    await hass.async_block_till_done()
    assert hass.states.get("sensor.powerguesser_capture_status").state == "resolving"
    await _advance(hass, freezer, 150)
    assert hass.states.get("sensor.powerguesser_capture_status").state == "idle"

    after = json.dumps(devices.record(dev_uuid), sort_keys=True)
    assert after == before
    assert devices.record(dev_uuid)["active_v"] == 1
    assert devices.record(dev_uuid).get("pending_proposal") is None
    assert [v["v"] for v in devices.record(dev_uuid)["versions"]] == [1]


@pytest.mark.parametrize("literal", [STATE_UNKNOWN, STATE_UNAVAILABLE])
async def test_restored_unknown_or_unavailable_label_is_ignored(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    literal: str,
) -> None:
    """A restored 'unknown'/'unavailable' literal is never taken as the label (D-02).

    A recorder flush can persist those literals; ``normalise_name`` would accept
    them and the label entity would restore as the literal text, which a later
    new-device capture start would use as the device name.
    """
    mock_restore_cache(hass, [State("text.powerguesser_capture_label", literal)])
    entry = await _setup_light(hass, energy_manager, std_prefs)
    assert entry.runtime_data.captures.surface_label == ""

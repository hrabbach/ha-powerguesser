"""Tests for the options-flow Capture menu (CAP-08, CAP-11, CAP-12, D-04, D-10..D-12).

The options flow grows a Capture menu whose steps present the same
CaptureManager the hub entities use: Start and Capture a past run are forms
whose Submit commits and returns at once (never a waiting or progress step),
Stop is an action step and Recent captures is read-only. Every step renders the
capture's stored insight (most similar, near twin, earlier runs) and the D-10
notes, reusing the same messages the services raise.
"""

import copy
from datetime import datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

import pytest

from homeassistant.components.energy.data import EnergyManager
from homeassistant.config_entries import ConfigEntryState
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType
from homeassistant.exceptions import ServiceValidationError
from homeassistant.util import dt as dt_util
from pytest_homeassistant.common import async_capture_events, async_fire_time_changed

import custom_components.powerguesser.config_flow as config_flow
from custom_components.powerguesser.const import (
    CAPTURES_STORE_KEY,
    CAPTURES_STORE_VERSION,
    DOMAIN,
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
_SPEC = SynthSpec(days=3, interval_s=(30.0, 30.0))
_END = int(datetime(2026, 11, 21, 23, 0, tzinfo=timezone.utc).timestamp() * 1000)
_TRUTH = generate(_SPEC, _END)

# The step result types that never wait for a capture to resolve (D-04).
_NON_WAITING = (FlowResultType.MENU, FlowResultType.FORM, FlowResultType.ABORT)


def _pin_grid(entry: Any, effective_s: float = float(GRID_S)) -> None:
    """Publish a profile snapshot so the grid is pinned at ``effective_s``."""
    enabled, disabled = gate(float(effective_s), False)
    entry.runtime_data.profile_coordinator.async_set_updated_data(
        ProfileSnapshot(
            sources={},
            effective_s=float(effective_s),
            assumed=False,
            enabled=enabled,
            disabled=disabled,
        )
    )


def _stored_captures(hass_storage: dict[str, Any]) -> list[dict[str, Any]]:
    return hass_storage.get(CAPTURES_STORE_KEY, {}).get("data", {}).get("captures", [])


def _plain(text: str) -> str:
    """Undo the menu's markdown escaping so assertions read the plain sentence."""
    return text.replace("\\", "")


def _capture_record(
    capture_id: str,
    target_kind: str,
    target_id: str,
    *,
    verdict: str | None = "clean",
    replay: dict[str, Any] | None = None,
    new_device_name: str | None = "Kettle",
    created_at: str = "2026-11-20T00:00:00+00:00",
    state: str = "done",
) -> dict[str, Any]:
    """One stored capture record of the D-05 shape (23 keys)."""
    result = None
    if verdict is not None:
        result = {
            "verdict": verdict,
            "rise_ms": 10**12,
            "fall_ms": 10**12 + 180_000,
            "step_w": 2000.0,
            "fall_w": -2000.0,
            "plateau_w": 2000.0,
            "dur_s": 180.0,
            "energy_wh": 100.0,
            "grid_s": GRID_S,
        }
    return {
        "capture_id": capture_id,
        "mode": "window",
        "state": state,
        "target_kind": target_kind,
        "target_id": target_id,
        "new_device_name": new_device_name,
        "label": "Kettle capture",
        "press_start_ts": None,
        "press_stop_ts": None,
        "window_start_ms": 10**12,
        "window_end_ms": 10**12 + 180_000,
        "created_at": created_at,
        "resolved_at": created_at,
        "reassigned_at": None,
        "reason_code": None,
        "gap_flagged": False,
        "result": result,
        "features": None,
        "trace": None,
        "holdout_excluded": True,
        "holdout_window_ms": [10**12, 10**12 + 180_000],
        "replay": replay,
        "earlier_runs": None,
    }


async def _write_captures(
    hass: HomeAssistant, entry: Any, hass_storage: dict[str, Any], records: list[dict]
) -> None:
    """Write the given capture records into the captures Store and reload."""
    hass_storage[CAPTURES_STORE_KEY] = {
        "version": CAPTURES_STORE_VERSION,
        "minor_version": 1,
        "key": CAPTURES_STORE_KEY,
        "data": {"captures": list(records)},
    }
    await entry.runtime_data.captures.async_load()
    await hass.async_block_till_done()


async def _setup_live(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
    monkeypatch: pytest.MonkeyPatch,
    now_ms: int,
    *,
    candidates: list[dict[str, Any]] | None = None,
) -> Any:
    """Set up an entry at a frozen time, fill the buffer, silence the live writer."""
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
    freezer.tick(timedelta(seconds=seconds))
    async_fire_time_changed(hass, dt_util.utcnow())
    await hass.async_block_till_done()


async def _open_capture(hass: HomeAssistant, entry: Any) -> dict[str, Any]:
    """Open the options flow and land on the Capture menu."""
    result = await async_open_options_step(hass, entry, "capture")
    assert result["type"] is FlowResultType.MENU
    assert result["step_id"] == "capture"
    return result


async def _open_start_form(hass: HomeAssistant, entry: Any) -> dict[str, Any]:
    """Open the Capture menu and advance to the Start form."""
    menu = await _open_capture(hass, entry)
    form = await hass.config_entries.options.async_configure(
        menu["flow_id"], {"next_step_id": "capture_start"}
    )
    assert form["type"] is FlowResultType.FORM
    assert form["step_id"] == "capture_start"
    return form


async def _submit_start(
    hass: HomeAssistant,
    entry: Any,
    target: str,
    name: str | None = None,
) -> dict[str, Any]:
    """Open the Start form and submit one target (with an optional new name)."""
    form = await _open_start_form(hass, entry)
    data: dict[str, Any] = {"target": target}
    if name is not None:
        data["name"] = name
    return await hass.config_entries.options.async_configure(form["flow_id"], data)


async def _cancel(hass: HomeAssistant) -> dict[str, Any]:
    return await hass.services.async_call(
        DOMAIN, "capture_cancel", {}, blocking=True, return_response=True
    )


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


def _status(hass: HomeAssistant) -> str:
    state = hass.states.get("sensor.powerguesser_capture_status")
    assert state is not None
    return state.state


async def _recent_text(hass: HomeAssistant, entry: Any) -> str:
    """Run the Recent captures step and return its {captures} placeholder, unescaped."""
    menu = await _open_capture(hass, entry)
    result = await hass.config_entries.options.async_configure(
        menu["flow_id"], {"next_step_id": "capture_recent"}
    )
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "capture_recent"
    return _plain(str(result["description_placeholders"]["captures"]))


# ---------------------------------------------------------------------------
# Task 1: the Capture menu, its three steps and the Recent list
# ---------------------------------------------------------------------------


async def test_capture_menu_start_stop_recent(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Start, Stop and Recent captures round-trip through the menu without waiting."""
    on_ms, off_ms = _TRUTH.kettle_runs[-1]
    entry = await _setup_live(
        hass,
        energy_manager,
        std_prefs,
        hass_storage,
        freezer,
        monkeypatch,
        on_ms - 60_000,
        candidates=[make_switched_candidate("sig-kettle", "Kettle")],
    )
    await _advance(hass, freezer, 80)

    menu = await _open_capture(hass, entry)
    assert _plain(menu["description_placeholders"]["status"]) == "No capture is running."
    assert "capture_stop" not in menu["menu_options"]

    form = await hass.config_entries.options.async_configure(
        menu["flow_id"], {"next_step_id": "capture_start"}
    )
    assert form["type"] is FlowResultType.FORM
    after = await hass.config_entries.options.async_configure(
        form["flow_id"], {"target": "new_device", "name": "Kettle"}
    )
    # Start returns to the menu at once: the capture is recording.
    assert after["type"] is FlowResultType.MENU
    assert _status(hass) == "recording"
    status_text = _plain(after["description_placeholders"]["status"])
    assert "Recording" in status_text and "Kettle" in status_text
    assert "capture_stop" in after["menu_options"]

    await _advance(hass, freezer, (off_ms - on_ms) / 1000)
    stopped = await hass.config_entries.options.async_configure(
        after["flow_id"], {"next_step_id": "capture_stop"}
    )
    assert stopped["type"] is FlowResultType.MENU
    assert _status(hass) == "resolving"
    stop_text = _plain(stopped["description_placeholders"]["status"])
    assert "about 2-3 minutes" in stop_text

    await _advance(hass, freezer, 90)
    assert _status(hass) == "idle"

    text = await _recent_text(hass, entry)
    assert "Kettle" in text
    assert "clean" in text
    assert "2000 W" in text
    assert "3 min" in text
    assert "Wh" in text
    assert "teaches its preview" in text
    assert "Most similar" in text
    assert "Earlier runs matched" in text


async def test_capture_steps_never_wait(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every capture step returns a menu, form or abort — never a progress step."""
    on_ms, off_ms = _TRUTH.kettle_runs[-1]
    entry = await _setup_live(
        hass, energy_manager, std_prefs, hass_storage, freezer, monkeypatch, on_ms - 60_000
    )
    await _advance(hass, freezer, 80)

    seen: list[Any] = []
    menu = await _open_capture(hass, entry)
    seen.append(menu)
    form = await hass.config_entries.options.async_configure(
        menu["flow_id"], {"next_step_id": "capture_start"}
    )
    seen.append(form)
    after = await hass.config_entries.options.async_configure(
        form["flow_id"], {"target": "new_device", "name": "Kettle"}
    )
    seen.append(after)

    await _advance(hass, freezer, (off_ms - on_ms) / 1000)
    stop = await hass.config_entries.options.async_configure(
        after["flow_id"], {"next_step_id": "capture_stop"}
    )
    seen.append(stop)
    recent = await hass.config_entries.options.async_configure(
        stop["flow_id"], {"next_step_id": "capture_recent"}
    )
    seen.append(recent)

    for result in seen:
        assert result["type"] in _NON_WAITING
        assert result["type"] is not FlowResultType.SHOW_PROGRESS


async def test_start_targets_candidate_by_id(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A candidate/device option stores the target's user name (or the id), never the option text."""
    on_ms, _off_ms = _TRUTH.kettle_runs[-1]
    entry = await _setup_live(
        hass,
        energy_manager,
        std_prefs,
        hass_storage,
        freezer,
        monkeypatch,
        on_ms - 60_000,
        candidates=[
            make_switched_candidate("sig-kettle", "Kettle"),
            make_switched_candidate("sig-toaster", "Toaster", rise_w=800.0),
        ],
    )
    hub = entry.runtime_data
    await _advance(hass, freezer, 80)

    # A labelled candidate with no decision name stores the signature id.
    await _submit_start(hass, entry, "candidate_sig-kettle")
    rec = _stored_captures(hass_storage)[0]
    assert rec["target_kind"] == "candidate"
    assert rec["target_id"] == "sig-kettle"
    assert rec["label"] == "sig-kettle"
    await _cancel(hass)

    # A target that vanished between render and submit is target_gone.
    form = await _open_start_form(hass, entry)
    await hub.learning.async_record_decision("dismiss", "sig-toaster")
    await hass.async_block_till_done()
    gone = await hass.config_entries.options.async_configure(
        form["flow_id"], {"target": "candidate_sig-toaster"}
    )
    assert gone["type"] is FlowResultType.FORM
    assert "target_gone" in gone["errors"].values()
    assert len(_stored_captures(hass_storage)) == 1

    # A renamed candidate stores the Review decision name.
    await hub.learning.async_record_decision(
        "rename", "sig-kettle", name="Kettle renamed"
    )
    await _submit_start(hass, entry, "candidate_sig-kettle")
    rec = _stored_captures(hass_storage)[1]
    assert rec["label"] == "Kettle renamed"
    await _cancel(hass)

    # A device option stores the device's record name.
    await async_confirm_candidate(hass, entry, "sig-kettle", "Kettle device")
    dev_uuid = hub.devices.device_for_signature("sig-kettle")
    assert dev_uuid is not None
    await _submit_start(hass, entry, f"device_{dev_uuid}")
    rec = _stored_captures(hass_storage)[2]
    assert rec["target_kind"] == "device"
    assert rec["target_id"] == dev_uuid
    assert rec["label"] == "Kettle device"
    await _cancel(hass)

    # An "Unnamed capture" preview also stores its signature id, never the option text.
    preview = _capture_record("cap-unnamed", "candidate", "cap_unnamed", new_device_name=None)
    await _write_captures(
        hass, entry, hass_storage, [*_stored_captures(hass_storage), preview]
    )
    visible = {c["signature_id"] for c in hub.learning.visible_candidates()}
    assert "cap_unnamed" in visible

    await _submit_start(hass, entry, "candidate_cap_unnamed")
    rec = _stored_captures(hass_storage)[-1]
    assert rec["target_id"] == "cap_unnamed"
    assert rec["label"] == "cap_unnamed"


async def test_start_while_open_and_stop_when_idle(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Start while open is refused and Stop when nothing is open states it (CAP-08 edges)."""
    on_ms, off_ms = _TRUTH.kettle_runs[-1]
    entry = await _setup_live(
        hass, energy_manager, std_prefs, hass_storage, freezer, monkeypatch, on_ms - 60_000
    )
    await _advance(hass, freezer, 80)

    menu = await _open_capture(hass, entry)
    form = await hass.config_entries.options.async_configure(
        menu["flow_id"], {"next_step_id": "capture_start"}
    )
    started = await hass.config_entries.options.async_configure(
        form["flow_id"], {"target": "new_device", "name": "Kettle"}
    )
    assert started["type"] is FlowResultType.MENU
    assert len(_stored_captures(hass_storage)) == 1

    # A second Start while one is open is refused with the service's message.
    again = await _submit_start(hass, entry, "new_device", "Toaster")
    assert again["type"] is FlowResultType.FORM
    assert "capture_refused" in again["errors"].values()
    refusal = str(again["description_placeholders"]["refusal"])
    assert "A capture is already open" in refusal
    assert len(_stored_captures(hass_storage)) == 1

    # The Stop button already stopped it; the still-rendered recording menu then
    # offers capture_stop, which finds nothing open and says so.
    await _advance(hass, freezer, (off_ms - on_ms) / 1000)
    await _press(hass, "button.powerguesser_stop_capture")
    await hass.async_block_till_done()
    assert _status(hass) == "resolving"
    await _advance(hass, freezer, 90)
    assert _status(hass) == "idle"

    idle = await hass.config_entries.options.async_configure(
        started["flow_id"], {"next_step_id": "capture_stop"}
    )
    assert idle["type"] is FlowResultType.MENU
    assert "No capture is open" in _plain(idle["description_placeholders"]["status"])
    assert len(_stored_captures(hass_storage)) == 1


async def test_refusal_placeholder_is_markdown_escaped(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Capture form's refusal placeholder escapes markdown (WR-03).

    A refusal message embeds candidate/device display names in ``{choices}``;
    those names can come from the LLM, and HA renders flow descriptions as
    markdown, so an unescaped value would render as a live link or heading in the
    admin dialog. The form must escape it like every sibling text path.
    """
    on_ms, _off_ms = _TRUTH.kettle_runs[-1]
    entry = await _setup_live(
        hass,
        energy_manager,
        std_prefs,
        hass_storage,
        freezer,
        monkeypatch,
        on_ms - 60_000,
    )
    await _advance(hass, freezer, 80)

    async def _raise(*_a: Any, **_k: Any) -> dict[str, Any]:
        raise ServiceValidationError(
            translation_domain=DOMAIN,
            translation_key="capture_target_unknown",
            translation_placeholders={
                "target": "Kettle",
                "choices": "[Support](https://example.invalid)",
            },
        )

    monkeypatch.setattr(entry.runtime_data.captures, "async_start", _raise)

    result = await _submit_start(hass, entry, "new_device", "Kettle")
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "capture_start"
    assert "capture_refused" in result["errors"].values()
    refusal = str(result["description_placeholders"]["refusal"])
    # The link syntax survives only in its escaped form.
    assert "[Support](https://example.invalid)" not in refusal
    assert "\\[Support\\]\\(https://example\\.invalid\\)" in refusal


async def test_button_and_menu_share_the_capture(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A button-started capture can be stopped from the menu and vice versa (CAP-08)."""
    on_ms, off_ms = _TRUTH.kettle_runs[-1]
    entry = await _setup_live(
        hass, energy_manager, std_prefs, hass_storage, freezer, monkeypatch, on_ms - 60_000
    )
    await _advance(hass, freezer, 80)

    # Start with the button, stop from the menu.
    await _set_label(hass, "Kettle")
    await _press(hass, "button.powerguesser_start_capture")
    await hass.async_block_till_done()
    assert _status(hass) == "recording"

    menu = await _open_capture(hass, entry)
    assert "capture_stop" in menu["menu_options"]
    await _advance(hass, freezer, (off_ms - on_ms) / 1000)
    stopped = await hass.config_entries.options.async_configure(
        menu["flow_id"], {"next_step_id": "capture_stop"}
    )
    assert stopped["type"] is FlowResultType.MENU
    assert _status(hass) == "resolving"
    await _advance(hass, freezer, 90)
    assert _status(hass) == "idle"

    # Start from the menu, stop with the button.
    started = await _submit_start(hass, entry, "new_device", "Kettle")
    assert started["type"] is FlowResultType.MENU
    assert _status(hass) == "recording"
    await _press(hass, "button.powerguesser_stop_capture")
    await hass.async_block_till_done()
    assert _status(hass) == "resolving"


async def test_recent_shows_d10_note_and_llm_verdict(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
) -> None:
    """Recent captures carries the D-10 note, the replay flag and the LLM verdict."""
    seed_candidates(hass_storage, [make_switched_candidate("sig01", "Kettle")])
    energy_manager.data = copy.deepcopy(std_prefs)
    async_set_power(hass, "sensor.grid_power", 250)
    async_set_power(hass, "sensor.solar_power", 0)
    async_set_power(hass, "sensor.battery_power", 0)
    async_set_power(hass, "sensor.washer_power", 0)
    entry = await async_setup_via_flow(hass)
    _pin_grid(entry)
    await hass.async_block_till_done()
    hub = entry.runtime_data
    await async_confirm_candidate(hass, entry, "sig01", "Kettle")
    dev_uuid = hub.devices.device_for_signature("sig01")
    assert dev_uuid is not None

    records = [
        _capture_record("cap-dirty", "device", dev_uuid, verdict="contaminated"),
        _capture_record(
            "cap-claim",
            "device",
            dev_uuid,
            replay={"would_claim": True, "n_episodes": 4, "basis": "x"},
        ),
        _capture_record(
            "cap-noclaim",
            "device",
            dev_uuid,
            replay={"would_claim": False, "n_episodes": 1, "basis": "x"},
        ),
        _capture_record("cap-plain", "device", dev_uuid),
    ]
    for i, record in enumerate(records):
        record["created_at"] = f"2026-11-20T00:0{i}:00+00:00"
        record["resolved_at"] = f"2026-11-20T00:0{i}:05+00:00"
    await _write_captures(hass, entry, hass_storage, records)
    hub.learning._capture_verdicts["cap-claim"] = "consistent"

    text = await _recent_text(hass, entry)
    assert "Not used to retrain the device (only clean captures train)" in text
    assert "LLM: consistent" in text
    assert "would have claimed this run" in text
    assert "would not have claimed this run" in text
    # Only the two captures that carry a replay result show the evidence line.
    assert text.count("Replay check (evidence only)") == 2


async def test_capture_menu_not_loaded_abort(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
) -> None:
    """With the entry unloaded the capture step aborts not_loaded (CAP-08)."""
    energy_manager.data = copy.deepcopy(std_prefs)
    async_set_power(hass, "sensor.grid_power", 250)
    async_set_power(hass, "sensor.solar_power", 0)
    async_set_power(hass, "sensor.battery_power", 0)
    async_set_power(hass, "sensor.washer_power", 0)
    entry = await async_setup_via_flow(hass)
    await hass.async_block_till_done()

    result = await hass.config_entries.options.async_init(entry.entry_id)
    assert result["type"] is FlowResultType.MENU

    # Unload the entry; HA drops its runtime data.
    assert await hass.config_entries.async_unload(entry.entry_id)
    assert entry.state is ConfigEntryState.NOT_LOADED

    aborted = await hass.config_entries.options.async_configure(
        result["flow_id"], {"next_step_id": "capture"}
    )
    assert aborted["type"] is FlowResultType.ABORT
    assert aborted["reason"] == "not_loaded"


# ---------------------------------------------------------------------------
# Task 2: capturing a past run from the menu with stated refusals
# ---------------------------------------------------------------------------


async def _setup_retro(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    *,
    candidates: list[dict[str, Any]] | None = None,
    timezone_name: str = "Europe/Berlin",
) -> tuple[Any, Any, ZoneInfo]:
    """Set up an entry with a 3-day synthetic buffer ending at today's local midnight."""
    if candidates:
        seed_candidates(hass_storage, candidates)
    await hass.config.async_set_time_zone(timezone_name)
    energy_manager.data = copy.deepcopy(std_prefs)
    async_set_power(hass, "sensor.grid_power", 500)
    async_set_power(hass, "sensor.solar_power", 0)
    async_set_power(hass, "sensor.battery_power", 0)
    async_set_power(hass, "sensor.washer_power", 0)
    entry = await async_setup_via_flow(hass)
    _pin_grid(entry)
    await hass.async_block_till_done()
    data = await async_fill_synth_buffer(hass, entry, SynthSpec(days=3))
    return entry, data, ZoneInfo(timezone_name)


def _local_naive(ms: int, tz: ZoneInfo) -> str:
    """A UTC millisecond epoch as a naive local datetime string (as the selector sends)."""
    return (
        dt_util.utc_from_timestamp(ms / 1000.0)
        .astimezone(tz)
        .replace(tzinfo=None)
        .isoformat()
    )


async def _open_retro_form(hass: HomeAssistant, entry: Any) -> dict[str, Any]:
    menu = await _open_capture(hass, entry)
    form = await hass.config_entries.options.async_configure(
        menu["flow_id"], {"next_step_id": "capture_retro"}
    )
    assert form["type"] is FlowResultType.FORM
    assert form["step_id"] == "capture_retro"
    return form


async def _submit_retro(
    hass: HomeAssistant,
    entry: Any,
    start: str,
    end: str,
    target: str = "new_device",
    name: str | None = "Kettle",
) -> dict[str, Any]:
    form = await _open_retro_form(hass, entry)
    data: dict[str, Any] = {"start": start, "end": end, "target": target}
    if name is not None:
        data["name"] = name
    return await hass.config_entries.options.async_configure(form["flow_id"], data)


async def _insert_grid_anomaly(hass: HomeAssistant, entry: Any, ts_ms: int) -> None:
    """Write a NULL gross row at ts_ms so the window scan sees a gap."""
    import sqlite3

    db = entry.runtime_data.writer.db

    def _do() -> None:
        grid_ch = next(
            ch for ch, eid, _role, _kind in db.channel_rows() if eid == "sensor.grid_power"
        )
        conn = sqlite3.connect(db.path, timeout=30)
        try:
            conn.execute(
                "INSERT OR REPLACE INTO samples(ts, ch, w) VALUES(?, ?, NULL)",
                (ts_ms, grid_ch),
            )
            conn.commit()
        finally:
            conn.close()

    await hass.async_add_executor_job(_do)
    await hass.async_block_till_done()


async def test_retro_form_captures_a_past_run(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
) -> None:
    """A past run is captured from the form and summarised in the menu text."""
    entry, data, tz = await _setup_retro(
        hass,
        energy_manager,
        std_prefs,
        hass_storage,
        candidates=[make_switched_candidate("sig01", "Kettle", rise_w=2000.0)],
    )
    hub = entry.runtime_data
    on_ms, off_ms = data.kettle_runs[-1]

    events = async_capture_events(hass, DOMAIN)
    result = await _submit_retro(
        hass, entry, _local_naive(on_ms - 45_000, tz), _local_naive(off_ms + 50_000, tz)
    )
    assert result["type"] is FlowResultType.MENU
    assert result["step_id"] == "capture"
    text = _plain(result["description_placeholders"]["status"])
    assert "clean:" in text
    assert " W," in text
    assert " min," in text
    assert " Wh" in text
    assert "near twin" in text
    assert "Earlier runs matched" in text

    records = _stored_captures(hass_storage)
    assert len(records) == 1
    assert records[0]["mode"] == "window"
    assert records[0]["state"] == "done"
    assert records[0]["target_id"] == hub.captures.records[0]["target_id"]

    listed = await hass.services.async_call(
        DOMAIN, "capture_list", {}, blocking=True, return_response=True
    )
    assert len(listed["captures"]) == 1
    assert listed["captures"][0]["result"]["similar"] is not None

    finished = [e for e in events if e.data.get("type") == "capture_finished"]
    assert len(finished) == 1


async def test_retro_refusals_are_shown_on_the_form(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
) -> None:
    """Every engine refusal re-shows the form with capture_refused and no record."""
    entry, data, tz = await _setup_retro(hass, energy_manager, std_prefs, hass_storage)
    on_ms, _off_ms = data.kettle_runs[-1]
    grid_rows = data.rows["sensor.grid_power"]
    buf_first = grid_rows[0][0]
    now_ms = int(dt_util.utcnow().timestamp() * 1000)

    cases = [
        (
            _local_naive(buf_first - 3_600_000, tz),
            _local_naive(buf_first - 3_000_000, tz),
            "outside the buffered history",
        ),
        (
            _local_naive(now_ms - 1_800_000, tz),
            _local_naive(now_ms + 3_600_000, tz),
            "ends in the future",
        ),
    ]
    for start, end, message in cases:
        result = await _submit_retro(hass, entry, start, end)
        assert result["type"] is FlowResultType.FORM
        assert result["step_id"] == "capture_retro"
        assert "capture_refused" in result["errors"].values()
        refusal = str(result["description_placeholders"]["refusal"])
        assert message in refusal
    assert _stored_captures(hass_storage) == []

    # across_gap: a NULL gross row inside the window.
    slot = on_ms + 1_800_000
    await _insert_grid_anomaly(hass, entry, slot)
    result = await _submit_retro(
        hass, entry, _local_naive(slot - 300_000, tz), _local_naive(slot + 300_000, tz)
    )
    assert result["type"] is FlowResultType.FORM
    assert "capture_refused" in result["errors"].values()
    assert "crosses a gap" in str(result["description_placeholders"]["refusal"])
    assert _stored_captures(hass_storage) == []


async def test_retro_target_candidate_and_device(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
) -> None:
    """The retro form targets a visible candidate and a confirmed device by entry."""
    entry, data, tz = await _setup_retro(
        hass,
        energy_manager,
        std_prefs,
        hass_storage,
        candidates=[make_switched_candidate("sig01", "Kettle", rise_w=2000.0)],
    )
    hub = entry.runtime_data
    on_ms, off_ms = data.kettle_runs[-1]
    start, end = _local_naive(on_ms - 45_000, tz), _local_naive(off_ms + 50_000, tz)

    first = await _submit_retro(hass, entry, start, end, target="candidate_sig01")
    assert first["type"] is FlowResultType.MENU
    rec = _stored_captures(hass_storage)[0]
    assert rec["target_kind"] == "candidate"
    assert rec["target_id"] == "sig01"
    assert rec["label"] == "sig01"

    await async_confirm_candidate(hass, entry, "sig01", "Kettle device")
    dev_uuid = hub.devices.device_for_signature("sig01")
    assert dev_uuid is not None

    second = await _submit_retro(
        hass, entry, start, end, target=f"device_{dev_uuid}", name=None
    )
    assert second["type"] is FlowResultType.MENU
    rec = _stored_captures(hass_storage)[1]
    assert rec["target_kind"] == "device"
    assert rec["target_id"] == dev_uuid
    assert rec["label"] == "Kettle device"


# ---------------------------------------------------------------------------
# OpenCodeReview MR !21 fixes
# ---------------------------------------------------------------------------


async def test_capture_result_shows_once_and_is_cleared(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The last action line shows on the following menu return only, then is cleared (D-04).

    ``self._capture_result`` was only ever assigned, so "Recording started for X."
    kept rendering on every later Capture visit; it must be consumed once.
    """
    on_ms, _off_ms = _TRUTH.kettle_runs[-1]
    entry = await _setup_live(
        hass, energy_manager, std_prefs, hass_storage, freezer, monkeypatch, on_ms - 60_000
    )
    await _advance(hass, freezer, 80)

    form = await _open_start_form(hass, entry)
    menu = await hass.config_entries.options.async_configure(
        form["flow_id"], {"target": "new_device", "name": "Kettle"}
    )
    assert menu["type"] is FlowResultType.MENU
    assert "Recording started for Kettle." in _plain(
        menu["description_placeholders"]["status"]
    )

    # Leave the menu (Recent captures) and return to it: the line is gone.
    recent = await hass.config_entries.options.async_configure(
        menu["flow_id"], {"next_step_id": "capture_recent"}
    )
    assert recent["type"] is FlowResultType.FORM
    back = await hass.config_entries.options.async_configure(recent["flow_id"], {})
    assert back["type"] is FlowResultType.MENU
    assert "Recording started" not in _plain(back["description_placeholders"]["status"])


async def test_refusal_text_uses_the_configured_language(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The refusal text is read in the configured locale, not always English.

    HA's ``async_get_exception_message`` is English-only, so the cache warm-up must
    no longer pin the user to English.
    """
    entry, _data, tz = await _setup_retro(hass, energy_manager, std_prefs, hass_storage)
    now_ms = int(dt_util.utcnow().timestamp() * 1000)
    key = "component.powerguesser.exceptions.capture_future_end.message"

    async def fake_translations(
        hass_: HomeAssistant,
        language: str,
        category: str,
        integrations: Any = None,
        config_flow_kw: Any = None,
    ) -> dict[str, str]:
        return {key: f"[{language}] capture window ends in the future"}

    monkeypatch.setattr(config_flow, "async_get_translations", fake_translations)
    hass.config.language = "de"

    result = await _submit_retro(
        hass,
        entry,
        _local_naive(now_ms - 1_800_000, tz),
        _local_naive(now_ms + 3_600_000, tz),
    )
    assert result["type"] is FlowResultType.FORM
    assert "capture_refused" in result["errors"].values()
    assert "[de]" in _plain(str(result["description_placeholders"]["refusal"]))


async def test_recent_shows_a_placeholder_for_a_missing_verdict(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
) -> None:
    """A non-done capture (no result) never renders a literal None verdict (D-11)."""
    energy_manager.data = copy.deepcopy(std_prefs)
    async_set_power(hass, "sensor.grid_power", 250)
    async_set_power(hass, "sensor.solar_power", 0)
    async_set_power(hass, "sensor.battery_power", 0)
    async_set_power(hass, "sensor.washer_power", 0)
    entry = await async_setup_via_flow(hass)
    _pin_grid(entry)
    await hass.async_block_till_done()

    record = _capture_record(
        "cap-interrupted", "candidate", "cap_missing", verdict=None, state="interrupted"
    )
    await _write_captures(hass, entry, hass_storage, [record])

    text = await _recent_text(hass, entry)
    assert "no verdict" in text
    assert ", None" not in text


async def test_retro_bad_datetime_is_not_reported_as_name_invalid(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
) -> None:
    """An unparseable start/end reports invalid_datetime, not name_invalid (Plan 09-07).

    The ``DateTimeSelector`` rejects a malformed value before the handler runs, so
    the step is invoked directly here (a value submitted outside the selector UI).
    """
    entry, _data, _tz = await _setup_retro(hass, energy_manager, std_prefs, hass_storage)
    form = await _open_retro_form(hass, entry)
    flow = hass.config_entries.options._progress[form["flow_id"]]

    result = await flow.async_step_capture_retro(
        {
            "start": "not-a-date",
            "end": "also-not-a-date",
            "target": "new_device",
            "name": "Kettle",
        }
    )

    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "capture_retro"
    assert result["errors"].get("start") == "invalid_datetime"
    assert result["errors"].get("end") == "invalid_datetime"
    assert "name_invalid" not in result["errors"].values()
    assert _stored_captures(hass_storage) == []

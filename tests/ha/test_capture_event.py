"""HA tests for the capture_finished bus event (CAP-14, D-13).

Plan 09-01: every terminal capture fires exactly one ``powerguesser`` event with
``type: capture_finished`` after its record is saved, carrying a fixed, label-free
payload. This covers the settle (done/failed), expiry, settle-error, cancel,
retroactive and restart-sweep paths and pins the payload key set.
"""

import copy
from datetime import datetime, timedelta, timezone
import json
from typing import Any

import pytest

from homeassistant.components.energy.data import EnergyManager
from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util
from pytest_homeassistant.common import async_capture_events, async_fire_time_changed

from custom_components.powerguesser.const import (
    CAPTURES_STORE_KEY,
    CAPTURES_STORE_VERSION,
    DOMAIN,
)
from custom_components.powerguesser.engine.intervals import ProfileSnapshot, gate
from tests.conftest import (
    async_fill_synth_buffer,
    async_set_power,
    async_setup_via_flow,
)
from tests.fixtures.learning.synth import SynthSpec, generate

GRID_S = 30
_SPEC = SynthSpec(days=3, interval_s=(30.0, 30.0))
_END = int(datetime(2026, 11, 21, 23, 0, tzinfo=timezone.utc).timestamp() * 1000)
_TRUTH = generate(_SPEC, _END)

LABEL = "mysecretlabel"
TARGET = "FullDisplayName"

EXPECTED_KEYS = {
    "type",
    "capture_id",
    "state",
    "quality",
    "reason_code",
    "target_kind",
    "target_id",
    "step_w",
    "duration_s",
    "energy_wh",
    "near_twin",
    "earlier_runs",
}

_FAILED_SNAP_RESULT: dict[str, Any] = {
    "result": {
        "rise_ms": None,
        "fall_ms": None,
        "step_w": None,
        "fall_w": None,
        "dur_s": None,
        "plateau_w": None,
        "energy_wh": None,
        "phase_share": None,
        "grid_s": GRID_S,
        "threshold_w": 120.0,
        "verdict": "none",
        "reason_code": "no_edge_pair",
        "candidates": [],
        "edges_inside": [],
        "press_to_rise_s": None,
        "press_to_fall_s": None,
        "quality_flags": [],
        "similar_dismissed": None,
    },
    "features": None,
    "trace": None,
}


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


def _seed_record(
    capture_id: str, state: str, start: str, stop: str | None, resolved: str | None
) -> dict[str, Any]:
    return {
        "capture_id": capture_id,
        "mode": "live",
        "state": state,
        "target_kind": "candidate",
        "target_id": "cap_" + capture_id,
        "new_device_name": "Kettle",
        "label": "kettle",
        "press_start_ts": start,
        "press_stop_ts": stop,
        "window_start_ms": None,
        "window_end_ms": None,
        "created_at": start,
        "resolved_at": resolved,
        "reassigned_at": None,
        "reason_code": None,
        "gap_flagged": False,
        "result": None,
        "features": None,
        "trace": None,
        "holdout_excluded": True,
        "holdout_window_ms": None,
        "replay": None,
    }


async def _setup(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    freezer: Any,
    monkeypatch: pytest.MonkeyPatch,
    now_ms: int,
) -> Any:
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


def _set_powers(hass: HomeAssistant) -> None:
    async_set_power(hass, "sensor.grid_power", 250)
    async_set_power(hass, "sensor.solar_power", 0)
    async_set_power(hass, "sensor.battery_power", 0)
    async_set_power(hass, "sensor.washer_power", 0)


async def _advance(hass: HomeAssistant, freezer: Any, seconds: float) -> None:
    freezer.tick(timedelta(seconds=seconds))
    async_fire_time_changed(hass, dt_util.utcnow())
    await hass.async_block_till_done()


async def _start(hass: HomeAssistant, target: str = TARGET) -> dict[str, Any]:
    return await hass.services.async_call(
        DOMAIN,
        "capture_start",
        {"label": LABEL, "target_type": "new_device", "target": target},
        blocking=True,
        return_response=True,
    )


async def _stop(hass: HomeAssistant, capture_id: str) -> dict[str, Any]:
    return await hass.services.async_call(
        DOMAIN, "capture_stop", {"capture_id": capture_id}, blocking=True, return_response=True
    )


async def _cancel(hass: HomeAssistant, capture_id: str) -> dict[str, Any]:
    return await hass.services.async_call(
        DOMAIN, "capture_cancel", {"capture_id": capture_id}, blocking=True, return_response=True
    )


async def _delete(hass: HomeAssistant, capture_id: str) -> dict[str, Any]:
    return await hass.services.async_call(
        DOMAIN, "capture_delete", {"capture_id": capture_id}, blocking=True, return_response=True
    )


async def _reassign(hass: HomeAssistant, capture_id: str, target: str) -> dict[str, Any]:
    return await hass.services.async_call(
        DOMAIN,
        "capture_reassign",
        {"capture_id": capture_id, "target_type": "new_device", "target": target},
        blocking=True,
        return_response=True,
    )


async def _capture_window(
    hass: HomeAssistant, *, start_ms: int, end_ms: int
) -> dict[str, Any]:
    def _iso(ms: int) -> str:
        return dt_util.utc_from_timestamp(ms / 1000.0).isoformat()

    return await hass.services.async_call(
        DOMAIN,
        "capture_window",
        {
            "start": _iso(start_ms),
            "end": _iso(end_ms),
            "label": LABEL,
            "target_type": "new_device",
            "target": TARGET,
        },
        blocking=True,
        return_response=True,
    )


async def _assert_pinned(hass: HomeAssistant, event: Any) -> None:
    """The payload has exactly the twelve keys, no free text, and follows the save."""
    assert set(event.data) == EXPECTED_KEYS
    blob = json.dumps(event.data)
    assert LABEL not in blob
    assert TARGET not in blob
    listed = await hass.services.async_call(
        DOMAIN,
        "capture_list",
        {"capture_id": event.data["capture_id"]},
        blocking=True,
        return_response=True,
    )
    rows = listed["captures"]
    assert len(rows) == 1
    assert rows[0]["state"] == event.data["state"]


async def test_event_on_settle_done_and_failed(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A clean live capture fires done/clean; a no-pair capture fires failed/none."""
    on_ms, off_ms = _TRUTH.kettle_runs[-1]
    await _setup(hass, energy_manager, std_prefs, freezer, monkeypatch, on_ms - 60_000)
    events = async_capture_events(hass, DOMAIN)

    await _advance(hass, freezer, 80)
    res = await _start(hass)
    await _advance(hass, freezer, (off_ms - on_ms) / 1000)
    await _stop(hass, res["capture_id"])
    await _advance(hass, freezer, 90)

    assert _stored_captures(hass_storage)[0]["state"] == "done"
    assert len(events) == 1
    assert events[0].data["type"] == "capture_finished"
    assert events[0].data["state"] == "done"
    assert events[0].data["quality"] == "clean"
    await _assert_pinned(hass, events[0])

    # A no-pair capture fails.
    monkeypatch.setattr(
        "custom_components.powerguesser.engine.capture.snap.snap_run",
        lambda *_a, **_k: copy.deepcopy(_FAILED_SNAP_RESULT),
    )
    await _advance(hass, freezer, 5)
    res = await _start(hass)
    await _advance(hass, freezer, 30)
    await _stop(hass, res["capture_id"])
    await _advance(hass, freezer, 90)

    assert _stored_captures(hass_storage)[1]["state"] == "failed"
    assert len(events) == 2
    assert events[1].data["state"] == "failed"
    assert events[1].data["quality"] == "none"
    await _assert_pinned(hass, events[1])


async def test_event_on_expiry_is_interrupted_timeout(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An open capture past live_max_open_s fires interrupted with reason_code timeout."""
    on_ms, _off_ms = _TRUTH.kettle_runs[-1]
    await _setup(hass, energy_manager, std_prefs, freezer, monkeypatch, on_ms - 60_000)
    events = async_capture_events(hass, DOMAIN)

    await _advance(hass, freezer, 80)
    await _start(hass)
    await _advance(hass, freezer, 3700)

    record = _stored_captures(hass_storage)[0]
    assert record["state"] == "interrupted"
    assert record["reason_code"] == "timeout"
    assert len(events) == 1
    assert events[0].data["state"] == "interrupted"
    assert events[0].data["reason_code"] == "timeout"
    await _assert_pinned(hass, events[0])


async def test_event_on_settle_error(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A settle job that raises fires interrupted with reason_code settle_error."""
    on_ms, _off_ms = _TRUTH.kettle_runs[-1]
    await _setup(hass, energy_manager, std_prefs, freezer, monkeypatch, on_ms - 60_000)
    events = async_capture_events(hass, DOMAIN)

    def _boom(*_a: Any, **_k: Any) -> Any:
        raise RuntimeError("snap failed")

    monkeypatch.setattr(
        "custom_components.powerguesser.engine.capture.snap.snap_run", _boom
    )

    await _advance(hass, freezer, 80)
    res = await _start(hass)
    await _advance(hass, freezer, 30)
    await _stop(hass, res["capture_id"])
    await _advance(hass, freezer, 90)

    record = _stored_captures(hass_storage)[0]
    assert record["state"] == "interrupted"
    assert record["reason_code"] == "settle_error"
    assert len(events) == 1
    assert events[0].data["state"] == "interrupted"
    assert events[0].data["reason_code"] == "settle_error"


async def test_event_on_cancel(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """capture_cancel fires once; cancelling a terminal capture fires nothing."""
    on_ms, _off_ms = _TRUTH.kettle_runs[-1]
    await _setup(hass, energy_manager, std_prefs, freezer, monkeypatch, on_ms - 60_000)
    events = async_capture_events(hass, DOMAIN)

    await _advance(hass, freezer, 80)
    res = await _start(hass)
    await _cancel(hass, res["capture_id"])

    assert _stored_captures(hass_storage)[0]["state"] == "cancelled"
    assert len(events) == 1
    assert events[0].data["state"] == "cancelled"

    await _cancel(hass, res["capture_id"])
    assert len(events) == 1


async def test_event_on_retro_capture(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A retroactive capture_window fires once with state done."""
    on_ms, _off_ms = _TRUTH.kettle_runs[-1]
    await _setup(hass, energy_manager, std_prefs, freezer, monkeypatch, on_ms - 60_000)
    events = async_capture_events(hass, DOMAIN)

    run_on, run_off = _TRUTH.kettle_runs[-2]
    await _capture_window(hass, start_ms=run_on - 45_000, end_ms=run_off + 50_000)

    assert _stored_captures(hass_storage)[0]["state"] == "done"
    assert len(events) == 1
    assert events[0].data["state"] == "done"
    assert events[0].data["type"] == "capture_finished"
    await _assert_pinned(hass, events[0])


async def test_event_on_restart_sweep(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
) -> None:
    """The load-time sweep fires one interrupted/restart event per open capture."""
    recording = _seed_record(
        "rec", "recording", "2026-11-20T10:00:00+00:00", None, None
    )
    resolving = _seed_record(
        "res",
        "resolving",
        "2026-11-20T10:00:00+00:00",
        "2026-11-20T10:05:00+00:00",
        None,
    )
    done = _seed_record(
        "done1",
        "done",
        "2026-11-19T10:00:00+00:00",
        "2026-11-19T10:05:00+00:00",
        "2026-11-19T10:06:00+00:00",
    )
    hass_storage[CAPTURES_STORE_KEY] = {
        "version": CAPTURES_STORE_VERSION,
        "minor_version": 1,
        "key": CAPTURES_STORE_KEY,
        "data": {"captures": [recording, resolving, done]},
    }
    energy_manager.data = copy.deepcopy(std_prefs)
    _set_powers(hass)
    events = async_capture_events(hass, DOMAIN)
    await async_setup_via_flow(hass)
    await hass.async_block_till_done()

    restart_events = [e for e in events if e.data.get("reason_code") == "restart"]
    assert len(restart_events) == 2
    assert {e.data["state"] for e in restart_events} == {"interrupted"}
    assert {e.data["capture_id"] for e in restart_events} == {"rec", "res"}
    for event in restart_events:
        assert set(event.data) == EXPECTED_KEYS


async def test_event_payload_keys_are_pinned(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every terminal path fires one event with the pinned, label-free payload."""
    on_ms, off_ms = _TRUTH.kettle_runs[-1]
    await _setup(hass, energy_manager, std_prefs, freezer, monkeypatch, on_ms - 60_000)
    events = async_capture_events(hass, DOMAIN)

    # 1. settle done
    await _advance(hass, freezer, 80)
    res = await _start(hass)
    await _advance(hass, freezer, (off_ms - on_ms) / 1000)
    await _stop(hass, res["capture_id"])
    await _advance(hass, freezer, 90)

    # 2. retroactive done
    run_on, run_off = _TRUTH.kettle_runs[-2]
    await _capture_window(hass, start_ms=run_on - 45_000, end_ms=run_off + 50_000)

    # 3. cancel
    await _advance(hass, freezer, 5)
    res = await _start(hass)
    await _cancel(hass, res["capture_id"])

    # 4. expiry
    await _advance(hass, freezer, 5)
    await _start(hass)
    await _advance(hass, freezer, 3700)

    assert len(events) == 4
    assert [e.data["state"] for e in events] == ["done", "done", "cancelled", "interrupted"]
    for event in events:
        await _assert_pinned(hass, event)


async def test_delete_and_reassign_fire_nothing(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
) -> None:
    """capture_delete and capture_reassign never fire capture_finished."""
    first = _seed_record(
        "done1", "done", "2026-11-19T10:00:00+00:00", "2026-11-19T10:05:00+00:00",
        "2026-11-19T10:06:00+00:00",
    )
    second = _seed_record(
        "done2", "done", "2026-11-19T11:00:00+00:00", "2026-11-19T11:05:00+00:00",
        "2026-11-19T11:06:00+00:00",
    )
    hass_storage[CAPTURES_STORE_KEY] = {
        "version": CAPTURES_STORE_VERSION,
        "minor_version": 1,
        "key": CAPTURES_STORE_KEY,
        "data": {"captures": [first, second]},
    }
    energy_manager.data = copy.deepcopy(std_prefs)
    _set_powers(hass)
    events = async_capture_events(hass, DOMAIN)
    await async_setup_via_flow(hass)
    await hass.async_block_till_done()

    await _delete(hass, "done1")
    await _reassign(hass, "done2", "Toaster")
    assert events == []

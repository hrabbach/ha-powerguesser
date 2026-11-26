"""HA tests for live capture: start/stop settle, cancel, expiry, idempotency.

Plan 08-03 Task 3 (CAP-02, CAP-05 cancel, CAP-06). Timers are driven with
``freezer`` + ``async_fire_time_changed``; ``snap_run`` is patched where a
blocking or repeating resolution is needed.
"""

import asyncio
import copy
from datetime import datetime, timedelta, timezone
import threading
from typing import Any

import pytest

from homeassistant.components.energy.data import EnergyManager
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ServiceValidationError
from homeassistant.util import dt as dt_util
from pytest_homeassistant.common import async_fire_time_changed

from custom_components.powerguesser.const import CAPTURES_STORE_KEY, DOMAIN
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

_NO_OP_SNAP_RESULT: dict[str, Any] = {
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
        "reason_code": "no_off_edge",
        "candidates": [],
        "edges_inside": [],
        "press_to_rise_s": None,
        "press_to_fall_s": None,
        "quality_flags": ["still_on"],
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


async def _noop_async(*_a: Any, **_k: Any) -> None:
    return None


async def _setup(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    freezer: Any,
    monkeypatch: pytest.MonkeyPatch,
    now_ms: int,
) -> tuple[Any, Any]:
    """Set up an entry at a frozen time, fill the buffer, silence the live writer.

    ``mark_stale``/``async_flush`` are no-ops so the frozen-time hub tick cannot
    inject NULL stale markers into the synthetic buffer.
    """
    freezer.move_to(dt_util.utc_from_timestamp(now_ms / 1000))
    energy_manager.data = copy.deepcopy(std_prefs)
    async_set_power(hass, "sensor.grid_power", 250)
    async_set_power(hass, "sensor.solar_power", 0)
    async_set_power(hass, "sensor.battery_power", 0)
    async_set_power(hass, "sensor.washer_power", 0)
    entry = await async_setup_via_flow(hass)
    _pin_grid(entry)
    await hass.async_block_till_done()
    data = await async_fill_synth_buffer(hass, entry, _SPEC, end_ms=_END)
    writer = entry.runtime_data.writer
    monkeypatch.setattr(writer, "mark_stale", lambda *_a, **_k: None)
    monkeypatch.setattr(writer, "async_flush", _noop_async)
    return entry, data


async def _advance(hass: HomeAssistant, freezer: Any, seconds: float) -> None:
    freezer.tick(timedelta(seconds=seconds))
    async_fire_time_changed(hass, dt_util.utcnow())
    await hass.async_block_till_done()


def _stored_captures(hass_storage: dict[str, Any]) -> list[dict[str, Any]]:
    return hass_storage.get(CAPTURES_STORE_KEY, {}).get("data", {}).get("captures", [])


async def _start(hass: HomeAssistant, *, label: str = "kettle", target: str = "Kettle") -> dict[str, Any]:
    return await hass.services.async_call(
        DOMAIN,
        "capture_start",
        {"label": label, "target_type": "new_device", "target": target},
        blocking=True,
        return_response=True,
    )


async def _stop(hass: HomeAssistant, capture_id: str | None = None) -> dict[str, Any]:
    data: dict[str, Any] = {} if capture_id is None else {"capture_id": capture_id}
    return await hass.services.async_call(
        DOMAIN, "capture_stop", data, blocking=True, return_response=True
    )


async def _cancel(hass: HomeAssistant, capture_id: str | None = None) -> dict[str, Any]:
    data: dict[str, Any] = {} if capture_id is None else {"capture_id": capture_id}
    return await hass.services.async_call(
        DOMAIN, "capture_cancel", data, blocking=True, return_response=True
    )


async def test_live_capture_resolves_nearest_off_edge(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Start on, stop after the run, settle -> done/clean with a Review preview."""
    on_ms, off_ms = _TRUTH.kettle_runs[-1]
    entry, _data = await _setup(
        hass, energy_manager, std_prefs, freezer, monkeypatch, on_ms - 60_000
    )
    hub = entry.runtime_data

    await _advance(hass, freezer, 80)  # now 20 s after the run start
    res = await _start(hass)
    cap = res["capture_id"]
    assert res["state"] == "recording"

    await _advance(hass, freezer, (off_ms - on_ms) / 1000)  # now 20 s after the run end
    assert await _stop(hass, cap) == {"capture_id": cap, "state": "resolving"}

    await _advance(hass, freezer, 90)  # settle wait = clamp(3*30, 90, 180)

    rec = _stored_captures(hass_storage)[0]
    assert rec["state"] == "done"
    assert rec["result"]["verdict"] == "clean"
    assert abs(rec["result"]["rise_ms"] - on_ms) <= 33_000
    assert abs(rec["result"]["fall_ms"] - off_ms) <= 33_000
    assert rec["result"]["press_to_rise_s"] is not None
    assert rec["result"]["press_to_fall_s"] is not None

    visible = hub.learning.visible_candidates()
    assert len(visible) == 1
    assert visible[0]["signature_id"].startswith("cap_")


async def test_live_capture_resolves_while_learning_holds_slot(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A live capture resolves while the learning slot is held (never raises already_running)."""
    on_ms, off_ms = _TRUTH.kettle_runs[-1]
    entry, _data = await _setup(
        hass, energy_manager, std_prefs, freezer, monkeypatch, on_ms - 60_000
    )
    entry.runtime_data.learning._start_pending = True
    try:
        await _advance(hass, freezer, 80)
        res = await _start(hass)
        await _advance(hass, freezer, (off_ms - on_ms) / 1000)
        await _stop(hass, res["capture_id"])
        await _advance(hass, freezer, 90)
    finally:
        entry.runtime_data.learning._start_pending = False

    rec = _stored_captures(hass_storage)[0]
    assert rec["state"] == "done"
    assert rec["result"]["verdict"] == "clean"


async def test_live_capture_waits_once_then_no_off_edge(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Still-on at the stop press: one extension, then failed/no_off_edge; exactly two snaps."""
    on_ms, _off_ms = _TRUTH.kettle_runs[-1]
    entry, _data = await _setup(
        hass, energy_manager, std_prefs, freezer, monkeypatch, on_ms - 60_000
    )

    calls: list[int] = []

    def _fake_snap(*_a: Any, **_k: Any) -> dict[str, Any]:
        calls.append(1)
        return copy.deepcopy(_NO_OP_SNAP_RESULT)

    monkeypatch.setattr(
        "custom_components.powerguesser.engine.capture.snap.snap_run", _fake_snap
    )

    await _advance(hass, freezer, 80)
    res = await _start(hass)
    assert await _stop(hass, res["capture_id"]) == {
        "capture_id": res["capture_id"],
        "state": "resolving",
    }
    await _advance(hass, freezer, 90)  # first settle: no fall, still on -> extend
    await _advance(hass, freezer, 400)  # past the one extension -> finish failed

    assert len(calls) == 2, f"expected two snap executions, got {len(calls)}"
    rec = _stored_captures(hass_storage)[0]
    assert rec["state"] == "failed"
    assert rec["reason_code"] == "no_off_edge"


async def test_second_start_and_stop_without_open(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A second start is refused; stop with nothing open and an unknown cancel id are refused."""
    on_ms, _off_ms = _TRUTH.kettle_runs[-1]
    entry, _data = await _setup(
        hass, energy_manager, std_prefs, freezer, monkeypatch, on_ms - 60_000
    )
    await _advance(hass, freezer, 80)

    res = await _start(hass)
    assert len(_stored_captures(hass_storage)) == 1

    with pytest.raises(ServiceValidationError) as open_err:
        await _start(hass, target="Other")
    assert open_err.value.translation_key == "capture_already_open"
    assert open_err.value.translation_placeholders["capture_id"] == res["capture_id"]
    assert len(_stored_captures(hass_storage)) == 1  # the refused start added nothing

    await _cancel(hass, res["capture_id"])
    with pytest.raises(ServiceValidationError) as not_open:
        await _stop(hass)
    assert not_open.value.translation_key == "capture_not_open"

    with pytest.raises(ServiceValidationError) as not_found:
        await _cancel(hass, "nope")
    assert not_found.value.translation_key == "capture_not_found"


async def test_second_stop_while_resolving_is_idempotent(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A double stop press runs exactly one resolution; a terminal stop writes nothing."""
    on_ms, off_ms = _TRUTH.kettle_runs[-1]
    entry, _data = await _setup(
        hass, energy_manager, std_prefs, freezer, monkeypatch, on_ms - 60_000
    )
    captures = entry.runtime_data.captures

    real_snap = __import__(
        "custom_components.powerguesser.engine.capture.snap", fromlist=["snap_run"]
    ).snap_run
    calls: list[int] = []

    def _counting_snap(*a: Any, **k: Any) -> dict[str, Any]:
        calls.append(1)
        return real_snap(*a, **k)

    monkeypatch.setattr(
        "custom_components.powerguesser.engine.capture.snap.snap_run", _counting_snap
    )

    await _advance(hass, freezer, 80)
    res = await _start(hass)
    cap = res["capture_id"]
    await _advance(hass, freezer, (off_ms - on_ms) / 1000)

    first = await _stop(hass)
    assert first == {"capture_id": cap, "state": "resolving"}
    after_first = copy.deepcopy(_stored_captures(hass_storage))

    second = await _stop(hass, cap)
    assert second == {"capture_id": cap, "state": "resolving"}
    assert _stored_captures(hass_storage) == after_first  # no write on the second stop
    assert captures._unsub_settle is not None  # still exactly one settle timer

    await _advance(hass, freezer, 90)
    assert len(calls) == 1

    terminal = await _stop(hass, cap)
    assert terminal == {"capture_id": cap, "state": "done"}
    assert len(_stored_captures(hass_storage)) == 1


async def test_cancel_is_idempotent_and_discards_late_result(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancel is idempotent; a cancel during a resolution discards its late result."""
    on_ms, _off_ms = _TRUTH.kettle_runs[-1]
    entry, _data = await _setup(
        hass, energy_manager, std_prefs, freezer, monkeypatch, on_ms - 60_000
    )

    # Cancel while recording.
    await _advance(hass, freezer, 80)
    res = await _start(hass)
    cap = res["capture_id"]
    assert await _cancel(hass, cap) == {"capture_id": cap, "state": "cancelled"}
    before = copy.deepcopy(_stored_captures(hass_storage))
    assert await _cancel(hass, cap) == {"capture_id": cap, "state": "cancelled"}
    assert _stored_captures(hass_storage) == before

    # Cancel while resolving: the in-flight resolution writes nothing.
    release = threading.Event()
    started = threading.Event()
    real_snap = __import__(
        "custom_components.powerguesser.engine.capture.snap", fromlist=["snap_run"]
    ).snap_run

    def _blocking_snap(*a: Any, **k: Any) -> dict[str, Any]:
        started.set()
        release.wait(timeout=5)
        return real_snap(*a, **k)

    monkeypatch.setattr(
        "custom_components.powerguesser.engine.capture.snap.snap_run", _blocking_snap
    )
    res2 = await _start(hass, target="Kettle2")
    cap2 = res2["capture_id"]
    await _stop(hass, cap2)

    freezer.tick(timedelta(seconds=90))
    async_fire_time_changed(hass, dt_util.utcnow())
    for _ in range(200):
        if started.is_set():
            break
        await asyncio.sleep(0.01)
    assert started.is_set()

    assert await _cancel(hass, cap2) == {"capture_id": cap2, "state": "cancelled"}
    release.set()
    await hass.async_block_till_done()

    rec = next(r for r in _stored_captures(hass_storage) if r["capture_id"] == cap2)
    assert rec["state"] == "cancelled"
    assert rec["result"] is None


async def test_open_capture_expires_after_max_open(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An open capture auto-expires as interrupted/timeout with a holdout window."""
    on_ms, _off_ms = _TRUTH.kettle_runs[-1]
    entry, _data = await _setup(
        hass, energy_manager, std_prefs, freezer, monkeypatch, on_ms - 60_000
    )
    await _advance(hass, freezer, 80)
    res = await _start(hass)
    press_start_ms = int(
        dt_util.parse_datetime(_stored_captures(hass_storage)[0]["press_start_ts"]).timestamp()
        * 1000
    )

    await _advance(hass, freezer, 3600)

    rec = _stored_captures(hass_storage)[0]
    assert rec["state"] == "interrupted"
    assert rec["reason_code"] == "timeout"
    assert rec["holdout_window_ms"] == [press_start_ms, press_start_ms + 3600 * 1000]
    assert res["capture_id"] == rec["capture_id"]


async def test_no_timer_without_open_capture(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    freezer: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A manager with no open capture holds no timer handle."""
    on_ms, _off_ms = _TRUTH.kettle_runs[-1]
    entry, _data = await _setup(
        hass, energy_manager, std_prefs, freezer, monkeypatch, on_ms - 60_000
    )
    captures = entry.runtime_data.captures
    assert captures._unsub_settle is None
    assert captures._unsub_expiry is None


async def test_settle_error_marks_record_interrupted(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A raising settle job ends the still-resolving record as interrupted/settle_error.

    An unhandled exception in the settle job must not strand the record in
    ``resolving`` with no save and no re-armed timer.
    """
    on_ms, off_ms = _TRUTH.kettle_runs[-1]
    entry, _data = await _setup(
        hass, energy_manager, std_prefs, freezer, monkeypatch, on_ms - 60_000
    )
    assert entry is not None

    def _boom(*_a: Any, **_k: Any) -> dict[str, Any]:
        raise RuntimeError("snap blew up")

    monkeypatch.setattr(
        "custom_components.powerguesser.engine.capture.snap.snap_run", _boom
    )

    await _advance(hass, freezer, 80)
    res = await _start(hass)
    await _advance(hass, freezer, (off_ms - on_ms) / 1000)
    await _stop(hass, res["capture_id"])
    await _advance(hass, freezer, 90)  # fire the settle timer

    rec = _stored_captures(hass_storage)[0]
    assert rec["state"] == "interrupted"
    assert rec["reason_code"] == "settle_error"


async def test_recording_capture_window_echo_has_no_end(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A still-recording live capture echoes a null end, never a 1970 timestamp."""
    on_ms, _off_ms = _TRUTH.kettle_runs[-1]
    entry, _data = await _setup(
        hass, energy_manager, std_prefs, freezer, monkeypatch, on_ms - 60_000
    )
    assert entry is not None
    await _advance(hass, freezer, 80)
    res = await _start(hass)

    listed = await hass.services.async_call(
        DOMAIN,
        "capture_list",
        {"capture_id": res["capture_id"]},
        blocking=True,
        return_response=True,
    )
    window = listed["captures"][0]["window"]
    assert window["start_utc"] is not None
    assert window["end_local"] is None
    assert window["end_utc"] is None

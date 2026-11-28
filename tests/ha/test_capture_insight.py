"""HA tests for the settle-time similarity and earlier-runs insight (CAP-11, CAP-12, D-11, D-12).

Plan 09-02: every capture with a snapped pair carries ``result.similar`` and
``earlier_runs`` from settle on, computed once in one bounded executor job
(off the event loop), advisory and isolated (a failure never interrupts the
capture), and visible through ``capture_list``, the status sensor and the
``capture_finished`` event.
"""

import ast
import copy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import threading
from typing import Any

import pytest

from homeassistant.components.energy.data import EnergyManager
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType
from homeassistant.util import dt as dt_util
from pytest_homeassistant.common import async_capture_events, async_fire_time_changed

import custom_components.powerguesser.capture as capture_mod
from custom_components.powerguesser.const import (
    CAPTURES_STORE_KEY,
    CAPTURES_STORE_VERSION,
    DOMAIN,
)
from custom_components.powerguesser.engine.capture.config import CaptureConfig
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
_RETRO_SPEC = SynthSpec(days=3, interval_s=(30.0, 30.0))

_LIVE_SPEC = SynthSpec(days=3, interval_s=(30.0, 30.0))
_LIVE_END = int(datetime(2026, 11, 21, 23, 0, tzinfo=timezone.utc).timestamp() * 1000)
_LIVE_TRUTH = generate(_LIVE_SPEC, _LIVE_END)

_REAL_SIMILAR = capture_mod.similar
_REAL_EARLIER_RUNS = capture_mod.earlier_runs
_REAL_INSIGHT_JOB = capture_mod._insight_job


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
    data = await async_fill_synth_buffer(hass, entry, _RETRO_SPEC)
    return entry, data


async def _setup_live(
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
    await async_fill_synth_buffer(hass, entry, _LIVE_SPEC, end_ms=_LIVE_END)
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


async def _capture_window(
    hass: HomeAssistant,
    *,
    start_ms: int,
    end_ms: int,
    target_type: str = "new_device",
    target: str = "Kettle",
) -> dict[str, Any]:
    return await hass.services.async_call(
        DOMAIN,
        "capture_window",
        {
            "start": _iso_utc(start_ms),
            "end": _iso_utc(end_ms),
            "label": "kettle",
            "target_type": target_type,
            "target": target,
        },
        blocking=True,
        return_response=True,
    )


async def _capture_list(hass: HomeAssistant, capture_id: str | None = None) -> dict[str, Any]:
    data: dict[str, Any] = {} if capture_id is None else {"capture_id": capture_id}
    return await hass.services.async_call(
        DOMAIN, "capture_list", data, blocking=True, return_response=True
    )


async def test_settle_records_similar_and_excludes_own_target(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
) -> None:
    """A retro capture lists the most similar candidates; its own target is never listed."""
    kettle = make_switched_candidate("sig-kettle", "Kettle")
    heater = make_switched_candidate("sig-heater", "Heater", rise_w=9000.0)
    entry, data = await _setup(
        hass, energy_manager, std_prefs, hass_storage, candidates=[kettle, heater]
    )
    on_ms, off_ms = data.kettle_runs[-1]

    res = await _capture_window(hass, start_ms=on_ms - 45_000, end_ms=off_ms + 50_000)
    similar = res["result"]["similar"]
    assert similar["top"][0]["id"] == "sig-kettle"
    assert similar["top"][0]["kind"] == "candidate"
    assert similar["top"][0]["claims"] is True
    assert similar["near_twin"] is True
    assert similar["twin"] == {"id": "sig-kettle", "kind": "candidate", "name": "Kettle"}
    # The display top is bounded by similar_top_k.
    assert len(similar["top"]) <= 3

    # A capture aimed at the kettle candidate itself does not list it.
    res2 = await _capture_window(
        hass,
        start_ms=on_ms - 45_000,
        end_ms=off_ms + 50_000,
        target_type="candidate",
        target="sig-kettle",
    )
    ids = [e["id"] for e in res2["result"]["similar"]["top"]]
    assert "sig-kettle" not in ids


async def test_confirmed_device_appears_in_similar(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
) -> None:
    """A confirmed device is listed with kind device and its record name."""
    kettle = make_switched_candidate("sig-kettle", "Kettle")
    entry, data = await _setup(
        hass, energy_manager, std_prefs, hass_storage, candidates=[kettle]
    )
    hub = entry.runtime_data
    await async_confirm_candidate(hass, entry, "sig-kettle", "Kettle")
    on_ms, off_ms = data.kettle_runs[-1]

    res = await _capture_window(hass, start_ms=on_ms - 45_000, end_ms=off_ms + 50_000)
    top = res["result"]["similar"]["top"]
    devices = [e for e in top if e["kind"] == "device"]
    assert devices, top
    assert devices[0]["name"] == "Kettle"
    assert "sig-kettle" not in [e["id"] for e in top]


async def test_similar_runs_off_the_event_loop(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """similar() runs in an executor thread on plain-data deep copies of the pool."""
    kettle = make_switched_candidate("sig-kettle", "Kettle")
    entry, data = await _setup(
        hass, energy_manager, std_prefs, hass_storage, candidates=[kettle]
    )
    hub = entry.runtime_data

    seen: dict[str, Any] = {}

    def wrapper(capture: Any, pool: Any, *, config: Any = CaptureConfig()) -> Any:
        seen["ident"] = threading.get_ident()
        seen["pool"] = pool
        return _REAL_SIMILAR(capture, pool, config=config)

    monkeypatch.setattr(capture_mod, "similar", wrapper)
    on_ms, off_ms = data.kettle_runs[-1]
    main_ident = threading.get_ident()

    res = await _capture_window(hass, start_ms=on_ms - 45_000, end_ms=off_ms + 50_000)

    assert seen["ident"] != main_ident  # not the event loop thread
    # Plain data only: the pool survives a JSON round trip.
    json.dumps(seen["pool"])
    # Deep copies: no pool artefact aliases a live candidate dict.
    live = hub.learning.visible_candidates()
    assert all(
        entry["artefact"] is not cand for entry in seen["pool"] for cand in live
    )

    # Mutating a live candidate after settle does not change the stored result.
    stored_before = json.dumps(res["result"]["similar"], sort_keys=True)
    for cand in live:
        cand["label"] = "MUTATED"
    stored_after = json.dumps(
        (await _capture_list(hass, res["capture_id"]))["captures"][0]["result"]["similar"],
        sort_keys=True,
    )
    assert stored_before == stored_after


async def test_insight_failure_keeps_capture_done(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A similarity or earlier-runs failure leaves the capture done, never interrupted."""
    kettle = make_switched_candidate("sig-kettle", "Kettle")
    entry, data = await _setup(
        hass, energy_manager, std_prefs, hass_storage, candidates=[kettle]
    )
    hub = entry.runtime_data
    on_ms, off_ms = data.kettle_runs[-1]

    def _boom(*_a: Any, **_k: Any) -> Any:
        raise RuntimeError("boom")

    # A direct call with no context keeps similar and reports earlier runs unavailable.
    normal = await _capture_window(hass, start_ms=on_ms - 45_000, end_ms=off_ms + 50_000)
    record = hub.captures.records[0]
    pool = hub.captures._similar_pool(record)
    sim, earlier, _errors = capture_mod._insight_job(record, pool, None)
    assert sim is not None
    assert earlier == {"n": None, "basis": "unavailable"}

    # similar and earlier_runs both raising.
    monkeypatch.setattr(capture_mod, "similar", _boom)
    monkeypatch.setattr(capture_mod, "earlier_runs", _boom)
    events = async_capture_events(hass, DOMAIN)
    res = await _capture_window(hass, start_ms=on_ms - 45_000, end_ms=off_ms + 50_000)
    assert res["state"] == "done"
    assert res["result"]["similar"] is None
    assert res["earlier_runs"] == {"n": None, "basis": "unavailable"}
    assert len(events) == 1
    assert events[0].data["state"] == "done"

    # similar alone raising: earlier_runs is still computed.
    monkeypatch.setattr(capture_mod, "similar", _boom)
    monkeypatch.setattr(capture_mod, "earlier_runs", _REAL_EARLIER_RUNS)
    res = await _capture_window(hass, start_ms=on_ms - 45_000, end_ms=off_ms + 50_000)
    assert res["state"] == "done"
    assert res["result"]["similar"] is None
    assert res["earlier_runs"]["n"] is not None and res["earlier_runs"]["n"] > 0

    # earlier_runs alone raising: similar is kept.
    monkeypatch.setattr(capture_mod, "similar", _REAL_SIMILAR)
    monkeypatch.setattr(capture_mod, "earlier_runs", _boom)
    res = await _capture_window(hass, start_ms=on_ms - 45_000, end_ms=off_ms + 50_000)
    assert res["state"] == "done"
    assert res["result"]["similar"] is not None
    assert res["earlier_runs"] == {"n": None, "basis": "unavailable"}

    # The job itself raising: both null/unavailable, capture done.
    monkeypatch.setattr(capture_mod, "earlier_runs", _REAL_EARLIER_RUNS)
    monkeypatch.setattr(capture_mod, "_insight_job", _boom)
    res = await _capture_window(hass, start_ms=on_ms - 45_000, end_ms=off_ms + 50_000)
    assert res["state"] == "done"
    assert res["result"]["similar"] is None
    assert res["earlier_runs"] == {"n": None, "basis": "unavailable"}

    # Restore the real job for the normal call above / other tests.
    monkeypatch.setattr(capture_mod, "_insight_job", _REAL_INSIGHT_JOB)
    assert normal["state"] == "done"


async def test_insight_respects_delete_during_job(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Deleting the capture while the earlier-runs job runs leaves no record and no event."""
    on_ms, off_ms = _LIVE_TRUTH.kettle_runs[-1]
    entry = await _setup_live(
        hass, energy_manager, std_prefs, freezer, monkeypatch, on_ms - 60_000
    )
    hub = entry.runtime_data
    events = async_capture_events(hass, DOMAIN)

    await _advance(hass, freezer, 80)
    started = await hass.services.async_call(
        DOMAIN,
        "capture_start",
        {"label": "kettle", "target_type": "new_device", "target": "Kettle"},
        blocking=True,
        return_response=True,
    )
    cap = started["capture_id"]
    await _advance(hass, freezer, (off_ms - on_ms) / 1000)
    await hass.services.async_call(
        DOMAIN, "capture_stop", {"capture_id": cap}, blocking=True, return_response=True
    )

    done = threading.Event()

    def slow_insight(*_a: Any, **_k: Any) -> Any:
        async def _delete() -> None:
            await hub.captures.async_delete(cap)
            done.set()

        hass.loop.call_soon_threadsafe(lambda: hass.async_create_task(_delete()))
        done.wait(5.0)
        return None, None, ["deleted"]

    monkeypatch.setattr(capture_mod, "_insight_job", slow_insight)
    await _advance(hass, freezer, 90)

    assert _stored_captures(hass_storage) == []
    assert hub.captures.records == []
    assert [e for e in events if e.data.get("capture_id") == cap] == []


async def test_earlier_runs_on_record_and_surfaces(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
) -> None:
    """A retro capture carries earlier_runs on the record, the sensor view and the event."""
    entry, data = await _setup(hass, energy_manager, std_prefs, hass_storage)
    hub = entry.runtime_data
    events = async_capture_events(hass, DOMAIN)
    on_ms, off_ms = data.kettle_runs[-1]

    res = await _capture_window(hass, start_ms=on_ms - 45_000, end_ms=off_ms + 50_000)
    n = res["earlier_runs"]["n"]
    assert isinstance(n, int) and n > 0
    assert "days" in res["earlier_runs"]["basis"]

    # capture_list carries the stored value.
    listed = await _capture_list(hass, res["capture_id"])
    assert listed["captures"][0]["earlier_runs"] == res["earlier_runs"]

    # The status sensor's last_result carries it.
    last = hub.captures.status_view()["last_result"]
    assert last["earlier_runs"] == n

    # The capture_finished event carries it.
    assert len(events) == 1
    assert events[0].data["earlier_runs"] == n


async def test_live_settle_carries_insight(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    freezer: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A live settled capture carries similar and earlier_runs too."""
    on_ms, off_ms = _LIVE_TRUTH.kettle_runs[-1]
    entry = await _setup_live(
        hass, energy_manager, std_prefs, freezer, monkeypatch, on_ms - 60_000
    )
    hub = entry.runtime_data

    await _advance(hass, freezer, 80)
    started = await hass.services.async_call(
        DOMAIN,
        "capture_start",
        {"label": "kettle", "target_type": "new_device", "target": "Kettle"},
        blocking=True,
        return_response=True,
    )
    await _advance(hass, freezer, (off_ms - on_ms) / 1000)
    await hass.services.async_call(
        DOMAIN,
        "capture_stop",
        {"capture_id": started["capture_id"]},
        blocking=True,
        return_response=True,
    )
    await _advance(hass, freezer, 90)

    record = hub.captures.records[-1]
    assert record["state"] == "done"
    assert record["result"]["similar"] is not None
    assert isinstance(record["earlier_runs"]["n"], int)
    assert record["earlier_runs"]["n"] > 0


async def test_insight_context_built_off_the_event_loop(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """build_context_job runs in an executor thread and the record's earlier_runs.n is an int."""
    entry, data = await _setup(hass, energy_manager, std_prefs, hass_storage)
    hub = entry.runtime_data
    idents: list[int] = []
    real = hub.learning.build_context_job

    def wrapper() -> dict[str, Any]:
        idents.append(threading.get_ident())
        return real()

    monkeypatch.setattr(hub.learning, "build_context_job", wrapper)
    main_ident = threading.get_ident()

    on_ms, off_ms = data.kettle_runs[-1]
    await _capture_window(hass, start_ms=on_ms - 45_000, end_ms=off_ms + 50_000)

    record = hub.captures.records[-1]
    assert isinstance(record["earlier_runs"]["n"], int)
    assert record["earlier_runs"]["n"] > 0
    assert idents, "build_context_job was never called"
    assert all(ident != main_ident for ident in idents)


def test_build_context_job_only_in_executor_closures() -> None:
    """Every build_context_job call in capture.py sits in a sync nested function."""
    source = Path(capture_mod.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    parents: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node

    def innermost_function(node: ast.AST) -> ast.AST | None:
        cur = parents.get(node)
        while cur is not None:
            if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return cur
            cur = parents.get(cur)
        return None

    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "build_context_job"
    ]
    assert len(calls) >= 4
    nested_in_insight = 0
    for call in calls:
        fn = innermost_function(call)
        assert isinstance(fn, ast.FunctionDef), (
            ast.dump(fn) if fn is not None else "no enclosing function"
        )
        if fn.name == "_insight_callable":
            nested_in_insight += 1
    assert nested_in_insight >= 1


# ---------------------------------------------------------------------------
# Plan 09-07 Task 3: the Review insight before Confirm, advisory only
# ---------------------------------------------------------------------------


def _insight_record(
    capture_id: str,
    target_kind: str,
    target_id: str,
    *,
    verdict: str = "clean",
    top: list[dict[str, Any]] | None = None,
    near_twin: bool = False,
    twin: dict[str, Any] | None = None,
    earlier_runs: dict[str, Any] | None = None,
    created_at: str = "2026-11-20T00:00:00+00:00",
) -> dict[str, Any]:
    """A stored done capture that carries the D-11 insight (23 keys)."""
    return {
        "capture_id": capture_id,
        "mode": "window",
        "state": "done",
        "target_kind": target_kind,
        "target_id": target_id,
        "new_device_name": "Kettle",
        "label": "kettle",
        "press_start_ts": None,
        "press_stop_ts": None,
        "window_start_ms": 10**12,
        "window_end_ms": 10**12 + 180_000,
        "created_at": created_at,
        "resolved_at": created_at,
        "reassigned_at": None,
        "reason_code": None,
        "gap_flagged": False,
        "result": {
            "verdict": verdict,
            "rise_ms": 10**12,
            "fall_ms": 10**12 + 180_000,
            "step_w": 2000.0,
            "fall_w": -2000.0,
            "plateau_w": 2000.0,
            "dur_s": 180.0,
            "energy_wh": 100.0,
            "grid_s": GRID_S,
            "similar": {"top": top or [], "near_twin": near_twin, "twin": twin},
        },
        "features": None,
        "trace": None,
        "holdout_excluded": True,
        "holdout_window_ms": [10**12, 10**12 + 180_000],
        "replay": None,
        "earlier_runs": earlier_runs,
    }


async def _seed_insight_records(
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


def _plain(text: str) -> str:
    """Undo the insight's markdown escaping so assertions read the plain sentence."""
    return text.replace("\\", "")


def _entry(kind: Any, ident: str, name: str, *, score: float = 0.9) -> dict[str, Any]:
    return {
        "id": ident,
        "kind": kind,
        "name": name,
        "score": score,
        "step_w": 2000.0,
        "dur_s": 180.0,
        "claims": True,
    }


async def _candidate_menu(hass: HomeAssistant, entry: Any, sig: str) -> dict[str, Any]:
    res_review = await async_open_options_step(hass, entry, "review")
    assert res_review["type"] is FlowResultType.FORM
    menu = await hass.config_entries.options.async_configure(
        res_review["flow_id"], {"candidate": sig}
    )
    assert menu["type"] is FlowResultType.MENU
    assert menu["step_id"] == "candidate"
    return menu


async def _device_menu(hass: HomeAssistant, entry: Any, dev_uuid: str) -> dict[str, Any]:
    res_review = await async_open_options_step(hass, entry, "review")
    assert res_review["type"] is FlowResultType.FORM
    menu = await hass.config_entries.options.async_configure(
        res_review["flow_id"], {"candidate": f"device_{dev_uuid}"}
    )
    assert menu["type"] is FlowResultType.MENU
    assert menu["step_id"] == "device"
    return menu


async def test_review_shows_similar_and_confirm_is_not_blocked(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
) -> None:
    """The candidate menu and the Confirm form show the insight; Confirm still works."""
    kettle = make_switched_candidate("sig-kettle", "Kettle")
    entry, data = await _setup(
        hass, energy_manager, std_prefs, hass_storage, candidates=[kettle]
    )
    on_ms, off_ms = data.kettle_runs[-1]
    res = await _capture_window(hass, start_ms=on_ms - 45_000, end_ms=off_ms + 50_000)
    sig = res["target"]["id"]
    assert sig.startswith("cap_")
    assert res["result"]["similar"]["near_twin"] is True

    menu = await _candidate_menu(hass, entry, sig)
    evidence = _plain(menu["description_placeholders"]["evidence"])
    assert "Most similar" in evidence
    assert "near twin" in evidence
    assert "Advisory only; it never blocks Confirm" in evidence

    form = await hass.config_entries.options.async_configure(
        menu["flow_id"], {"next_step_id": "confirm"}
    )
    assert form["type"] is FlowResultType.FORM
    warning = _plain(form["description_placeholders"]["warning"])
    assert "Most similar" in warning
    assert "near twin" in warning

    confirmed = await hass.config_entries.options.async_configure(
        form["flow_id"], {"name": "Kettle", "confirm_anyway": True}
    )
    assert confirmed["type"] is FlowResultType.FORM  # back at the review list
    subs = entry.get_subentries_of_type("device")
    assert len(subs) == 1
    uuid = subs[0].data["device_uuid"]
    assert entry.runtime_data.devices.record(uuid)["status"] == "confirmed"


async def test_review_names_twin_outside_top_list(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
) -> None:
    """A twin outside the top list gets its own line and no entry note."""
    kettle = make_switched_candidate("sig-kettle", "Kettle")
    entry, _data = await _setup(
        hass, energy_manager, std_prefs, hass_storage, candidates=[kettle]
    )
    record = _insight_record(
        "cap-twin",
        "candidate",
        "sig-kettle",
        top=[_entry("candidate", "sig-kettle", "Kettle")],
        near_twin=True,
        twin={"id": "sig-ghost", "kind": "candidate", "name": "Toaster"},
        earlier_runs={"n": 4, "basis": "this capture's tolerance box on the last 3 days"},
    )
    await _seed_insight_records(hass, entry, hass_storage, [record])

    evidence = _plain(
        (await _candidate_menu(hass, entry, "sig-kettle"))[
            "description_placeholders"
        ]["evidence"]
    )
    assert "Near twin: Toaster" in evidence
    assert "— near twin" not in evidence
    assert "Earlier runs matched: 4 (this capture's tolerance box on the last 3 days)" in evidence


async def test_device_menu_shows_insight_before_proposal_text(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
) -> None:
    """A device capture's insight joins the evidence; the {proposal} text still follows."""
    kettle = make_switched_candidate("sig01", "Kettle")
    entry, _data = await _setup(
        hass, energy_manager, std_prefs, hass_storage, candidates=[kettle]
    )
    hub = entry.runtime_data
    await async_confirm_candidate(hass, entry, "sig01", "Kettle")
    dev_uuid = hub.devices.device_for_signature("sig01")
    assert dev_uuid is not None

    record = _insight_record(
        "cap-dev",
        "device",
        dev_uuid,
        top=[_entry("device", dev_uuid, "Kettle", score=0.88)],
        near_twin=False,
        twin=None,
        earlier_runs={"n": 2, "basis": "this capture's tolerance box on the last 3 days"},
    )
    await _seed_insight_records(hass, entry, hass_storage, [record])

    menu = await _device_menu(hass, entry, dev_uuid)
    evidence = _plain(menu["description_placeholders"]["evidence"])
    assert "Most similar: Kettle (0.88, 2000 W, 3 min)" in evidence
    assert "Earlier runs matched: 2" in evidence
    assert "Advisory only; it never blocks Confirm" in evidence

    # 09-06's {proposal} text still follows the evidence, unchanged.
    devices = hub.devices
    base = copy.deepcopy(devices.record(dev_uuid)["versions"][0]["artefact"])
    base["proposal_gate"] = {
        "passed": True,
        "proposed_S": 0.0601,
        "active_S": 0.055,
        "basis": "learning_split_val",
    }
    await devices.async_set_proposal(
        dev_uuid,
        base,
        run_id="run-capture",
        proposed_at="2026-11-20T00:00:00Z",
        trained_at="2026-11-20T00:00:00Z",
        source="capture",
        capture_ids=["cap-dev"],
        captures_snapshot_at="2026-11-20T00:00:00Z",
    )
    await hass.async_block_till_done()
    menu = await _device_menu(hass, entry, dev_uuid)
    proposal = menu["description_placeholders"]["proposal"].replace("\\", "")
    assert "Update available from your captures (1 captured run)" in proposal
    assert "Most similar: Kettle (0.88, 2000 W, 3 min)" in _plain(
        menu["description_placeholders"]["evidence"]
    )


async def test_review_insight_reresolves_and_escapes_names(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
) -> None:
    """Names are read from the current candidate and markdown-escaped; a target without captures shows nothing."""
    kettle = make_switched_candidate("sig-kettle", "Kettle")
    other = make_switched_candidate("sig-other", "Toaster", rise_w=800.0)
    entry, _data = await _setup(
        hass, energy_manager, std_prefs, hass_storage, candidates=[kettle, other]
    )
    hub = entry.runtime_data
    record = _insight_record(
        "cap-old",
        "candidate",
        "sig-kettle",
        top=[_entry("candidate", "sig-kettle", "Old stored name")],
        earlier_runs={"n": 3, "basis": "this capture's tolerance box on the last 2 days"},
    )
    await _seed_insight_records(hass, entry, hass_storage, [record])

    # The stored name is re-resolved to the current candidate name.
    evidence = _plain(
        (await _candidate_menu(hass, entry, "sig-kettle"))[
            "description_placeholders"
        ]["evidence"]
    )
    assert "Most similar: Kettle (0.9, 2000 W, 3 min)" in evidence
    assert "Old stored name" not in evidence

    # A rename is shown under its new name, markdown-escaped.
    await hub.learning.async_record_decision("rename", "sig-kettle", name="Ket_tle*")
    raw = (await _candidate_menu(hass, entry, "sig-kettle"))[
        "description_placeholders"
    ]["evidence"]
    assert "Ket\\_tle\\*" in raw
    assert "Ket_tle*" not in raw

    # A candidate with no captures carries no insight text.
    empty = (await _candidate_menu(hass, entry, "sig-other"))[
        "description_placeholders"
    ]["evidence"]
    assert "Advisory only" not in empty
    assert "Most similar" not in empty


# ---------------------------------------------------------------------------
# OpenCodeReview MR !21: the insight pool build is advisory too
# ---------------------------------------------------------------------------


def _pool_boom(*_a: Any, **_k: Any) -> Any:
    raise RuntimeError("pool boom")


async def test_insight_pool_build_failure_keeps_window_capture_done(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pool-build failure is advisory: the retro capture is still stored and done.

    ``_similar_pool`` runs on the loop before the guarded executor job; without its
    own guard the exception escaped ``async_capture_window`` (which has no catch),
    so the snapped capture was never saved and the service call failed.
    """
    kettle = make_switched_candidate("sig-kettle", "Kettle")
    entry, data = await _setup(
        hass, energy_manager, std_prefs, hass_storage, candidates=[kettle]
    )
    hub = entry.runtime_data
    on_ms, off_ms = data.kettle_runs[-1]

    monkeypatch.setattr(hub.captures, "_similar_pool", _pool_boom)

    res = await _capture_window(hass, start_ms=on_ms - 45_000, end_ms=off_ms + 50_000)

    assert res["state"] == "done"
    assert res["result"]["similar"] is None
    assert res["earlier_runs"] == {"n": None, "basis": "unavailable"}
    stored = _stored_captures(hass_storage)
    assert len(stored) == 1
    assert stored[0]["state"] == "done"


async def test_insight_pool_build_failure_keeps_live_capture_done(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pool-build failure on the live-settle path never marks a clean capture interrupted."""
    on_ms, off_ms = _LIVE_TRUTH.kettle_runs[-1]
    entry = await _setup_live(
        hass, energy_manager, std_prefs, freezer, monkeypatch, on_ms - 60_000
    )
    hub = entry.runtime_data

    await _advance(hass, freezer, 80)
    started = await hass.services.async_call(
        DOMAIN,
        "capture_start",
        {"label": "kettle", "target_type": "new_device", "target": "Kettle"},
        blocking=True,
        return_response=True,
    )
    await _advance(hass, freezer, (off_ms - on_ms) / 1000)
    await hass.services.async_call(
        DOMAIN,
        "capture_stop",
        {"capture_id": started["capture_id"]},
        blocking=True,
        return_response=True,
    )

    monkeypatch.setattr(hub.captures, "_similar_pool", _pool_boom)
    await _advance(hass, freezer, 90)

    record = hub.captures.records[-1]
    assert record["state"] == "done"
    assert record["result"]["similar"] is None
    assert record["earlier_runs"] == {"n": None, "basis": "unavailable"}


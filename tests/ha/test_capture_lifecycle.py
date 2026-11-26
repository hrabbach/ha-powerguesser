"""HA tests for capture lifecycle: delete, re-assign and the record cap (CAP-05, D-06).

Plan 08-05 Task 2: ``capture_delete`` removes a record idempotently, cancels an
open one's timers and reports a removed preview; ``capture_reassign`` moves an
example between preview groups and is the only target rewrite; the Store stays
bounded without ever dropping a taught example.
"""

import copy
from datetime import datetime, timedelta, timezone
import json
from typing import Any

import pytest

from homeassistant.components.energy.data import EnergyManager
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ServiceValidationError
from homeassistant.util import dt as dt_util
from pytest_homeassistant.common import async_fire_time_changed

from custom_components.powerguesser.const import (
    CAPTURES_STORE_KEY,
    CAPTURES_STORE_VERSION,
    DOMAIN,
)
from custom_components.powerguesser.engine.intervals import ProfileSnapshot, gate
from tests.conftest import (
    async_confirm_candidate,
    async_fill_synth_buffer,
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


def _iso_utc(ms: int) -> str:
    return dt_util.utc_from_timestamp(ms / 1000.0).isoformat()


def _stored_captures(hass_storage: dict[str, Any]) -> list[dict[str, Any]]:
    return hass_storage.get(CAPTURES_STORE_KEY, {}).get("data", {}).get("captures", [])


async def _setup(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
    monkeypatch: pytest.MonkeyPatch,
    *,
    now_ms: int = _END,
    candidates: list[dict[str, Any]] | None = None,
    seeded: list[dict[str, Any]] | None = None,
) -> Any:
    if candidates:
        seed_candidates(hass_storage, candidates)
    if seeded is not None:
        hass_storage[CAPTURES_STORE_KEY] = {
            "version": CAPTURES_STORE_VERSION,
            "minor_version": 1,
            "key": CAPTURES_STORE_KEY,
            "data": {"captures": seeded},
        }
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


async def _capture(
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
            "label": "run",
            "target_type": target_type,
            "target": target,
        },
        blocking=True,
        return_response=True,
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


async def _start(hass: HomeAssistant, *, target: str = "Kettle") -> dict[str, Any]:
    return await hass.services.async_call(
        DOMAIN,
        "capture_start",
        {"label": "run", "target_type": "new_device", "target": target},
        blocking=True,
        return_response=True,
    )


async def _stop(hass: HomeAssistant, capture_id: str) -> dict[str, Any]:
    return await hass.services.async_call(
        DOMAIN, "capture_stop", {"capture_id": capture_id}, blocking=True, return_response=True
    )


def _record(capture_id: str, state: str, created: str) -> dict[str, Any]:
    return {
        "capture_id": capture_id,
        "mode": "window",
        "state": state,
        "target_kind": "device",
        "target_id": f"dev-{capture_id}",
        "new_device_name": None,
        "label": "l",
        "press_start_ts": None,
        "press_stop_ts": None,
        "window_start_ms": 0,
        "window_end_ms": 1000,
        "created_at": created,
        "resolved_at": created,
        "reassigned_at": None,
        "reason_code": None,
        "gap_flagged": False,
        "result": {"verdict": "clean", "grid_s": 30},
        "features": None,
        "trace": None,
        "holdout_excluded": True,
        "holdout_window_ms": [0, 1000],
        "replay": None,
    }


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------


async def test_delete_is_idempotent(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Deleting a done capture removes its preview; deleting the id again writes nothing."""
    entry = await _setup(hass, energy_manager, std_prefs, hass_storage, freezer, monkeypatch)
    hub = entry.runtime_data
    on_ms, off_ms = _TRUTH.kettle_runs[-1]
    res = await _capture(hass, start_ms=on_ms - 45_000, end_ms=off_ms + 50_000)
    assert len(hub.learning.visible_candidates()) == 1

    out = await _delete(hass, res["capture_id"])
    assert out == {"deleted": True, "preview_removed": True}
    assert hub.learning.visible_candidates() == []

    mid = json.dumps(_stored_captures(hass_storage), sort_keys=True)
    again = await _delete(hass, res["capture_id"])
    assert again == {"deleted": False}
    assert json.dumps(_stored_captures(hass_storage), sort_keys=True) == mid


async def test_delete_while_resolving_discards_result(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Deleting a resolving capture cancels its settle timer; the late result writes nothing."""
    on_ms, off_ms = _TRUTH.kettle_runs[-1]
    entry = await _setup(
        hass, energy_manager, std_prefs, hass_storage, freezer, monkeypatch, now_ms=on_ms - 60_000
    )
    hub = entry.runtime_data

    await _advance(hass, freezer, 80)
    res = await _start(hass)
    await _advance(hass, freezer, (off_ms - on_ms) / 1000)
    assert await _stop(hass, res["capture_id"]) == {
        "capture_id": res["capture_id"],
        "state": "resolving",
    }

    out = await _delete(hass, res["capture_id"])
    assert out["deleted"] is True
    after_delete = json.dumps(_stored_captures(hass_storage), sort_keys=True)

    await _advance(hass, freezer, 90)  # fire the settle timer
    assert json.dumps(_stored_captures(hass_storage), sort_keys=True) == after_delete
    assert _stored_captures(hass_storage) == []
    assert hub.learning.visible_candidates() == []


# ---------------------------------------------------------------------------
# re-assign
# ---------------------------------------------------------------------------


async def test_reassign_moves_example_between_groups(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reassigning moves a clean example A -> B (both refit); a busy capture is refused."""
    entry = await _setup(hass, energy_manager, std_prefs, hass_storage, freezer, monkeypatch)
    hub = entry.runtime_data
    run_a = _TRUTH.kettle_runs[-1]
    run_b = _TRUTH.kettle_runs[-2]

    a1 = await _capture(hass, start_ms=run_a[0] - 45_000, end_ms=run_a[1] + 50_000, target="GroupA")
    a2 = await _capture(
        hass,
        start_ms=run_a[0] - 45_000,
        end_ms=run_a[1] + 50_000,
        target_type="candidate",
        target="GroupA",
    )
    b1 = await _capture(hass, start_ms=run_b[0] - 45_000, end_ms=run_b[1] + 50_000, target="GroupB")
    sig_a = a1["target"]["id"]
    sig_b = b1["target"]["id"]
    assert a2["target"]["id"] == sig_a

    previews = {c["signature_id"]: c for c in hub.learning.visible_candidates()}
    assert previews[sig_a]["scores"]["n_examples"] == 2
    assert previews[sig_b]["scores"]["n_examples"] == 1

    out = await _reassign(hass, a2["capture_id"], target_type="candidate", target="GroupB")
    assert out["target"]["id"] == sig_b
    assert out["reassigned_at"] is not None

    previews = {c["signature_id"]: c for c in hub.learning.visible_candidates()}
    assert previews[sig_a]["scores"]["n_examples"] == 1
    assert previews[sig_b]["scores"]["n_examples"] == 2

    mid = json.dumps(_stored_captures(hass_storage), sort_keys=True)
    again = await _reassign(hass, a2["capture_id"], target_type="candidate", target="GroupB")
    assert again["target"]["id"] == sig_b
    assert json.dumps(_stored_captures(hass_storage), sort_keys=True) == mid

    await _advance(hass, freezer, 80)
    open_res = await _start(hass)
    with pytest.raises(ServiceValidationError) as busy:
        await _reassign(hass, open_res["capture_id"], target_type="new_device", target="X")
    assert busy.value.translation_key == "capture_busy"


async def test_reassign_to_device_recomputes_replay(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reassigning to a confirmed device adds the replay block; a candidate clears it."""
    candidates = [
        make_switched_candidate("sig-kettle", "Kettle"),
        make_switched_candidate("sig-spare", "Toaster"),
    ]
    entry = await _setup(
        hass, energy_manager, std_prefs, hass_storage, freezer, monkeypatch, candidates=candidates
    )
    hub = entry.runtime_data
    uuid, dev_id = await async_confirm_candidate_and_id(hass, entry, "sig-kettle", "Kettle")
    run_a = _TRUTH.kettle_runs[-1]

    res = await _capture(hass, start_ms=run_a[0] - 45_000, end_ms=run_a[1] + 50_000)
    to_device = await _reassign(
        hass, res["capture_id"], target_type="device", target="Kettle", device_id=dev_id
    )
    assert to_device["target"]["kind"] == "device"
    assert to_device["target"]["id"] == uuid
    assert to_device["replay"]["would_claim"] is True

    back = await _reassign(
        hass, res["capture_id"], target_type="candidate", target="sig-spare"
    )
    assert back["target"]["id"] == "sig-spare"
    assert back["replay"] is None


async def async_confirm_candidate_and_id(
    hass: HomeAssistant, entry: Any, sig: str, name: str
) -> tuple[str, str]:
    """Confirm a candidate and return (device record uuid, registry device id)."""
    from homeassistant.helpers import device_registry as dr

    await async_confirm_candidate(hass, entry, sig, name)
    sub = next(s for s in entry.get_subentries_of_type("device") if s.title == name)
    uuid = sub.data["device_uuid"]
    device = dr.async_get(hass).async_get_device_by_identifier(
        (DOMAIN, uuid), config_entry_id=entry.entry_id
    )
    assert device is not None
    return uuid, device.id


# ---------------------------------------------------------------------------
# record cap
# ---------------------------------------------------------------------------


async def test_record_cap_prunes_non_teaching(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A new capture prunes the oldest failed record and succeeds; done records stay."""
    failed = [_record(f"f{i}", "failed", "2026-11-19T00:00:00+00:00") for i in range(10)]
    done = [_record(f"d{i}", "done", "2026-11-20T00:00:00+00:00") for i in range(190)]
    await _setup(
        hass, energy_manager, std_prefs, hass_storage, freezer, monkeypatch, seeded=[*failed, *done]
    )
    on_ms, off_ms = _TRUTH.kettle_runs[-1]

    res = await _capture(hass, start_ms=on_ms - 45_000, end_ms=off_ms + 50_000)
    records = _stored_captures(hass_storage)
    assert len(records) == 200
    ids = {r["capture_id"] for r in records}
    assert "f0" not in ids  # the oldest failed record was pruned
    assert "f1" in ids
    assert res["capture_id"] in ids
    # No done (teaching) record was dropped.
    assert all(f"d{i}" in ids for i in range(190))


async def test_record_cap_refuses_when_only_done(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """At the cap with only done records, a new capture raises and nothing is removed."""
    only_done = [_record(f"done{i}", "done", "2026-11-20T00:00:00+00:00") for i in range(200)]
    await _setup(
        hass, energy_manager, std_prefs, hass_storage, freezer, monkeypatch, seeded=only_done
    )
    on_ms, off_ms = _TRUTH.kettle_runs[-1]
    before = json.dumps(_stored_captures(hass_storage), sort_keys=True)

    with pytest.raises(ServiceValidationError) as full:
        await _capture(hass, start_ms=on_ms - 45_000, end_ms=off_ms + 50_000)
    assert full.value.translation_key == "capture_store_full"
    assert json.dumps(_stored_captures(hass_storage), sort_keys=True) == before


async def test_record_cap_enforced_on_live_start(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The live path enforces the cap too: a start prunes the oldest failed record (WR-01)."""
    failed = [_record(f"f{i}", "failed", "2026-11-19T00:00:00+00:00") for i in range(10)]
    done = [_record(f"d{i}", "done", "2026-11-20T00:00:00+00:00") for i in range(190)]
    await _setup(
        hass, energy_manager, std_prefs, hass_storage, freezer, monkeypatch, seeded=[*failed, *done]
    )

    res = await _start(hass)
    records = _stored_captures(hass_storage)
    assert len(records) == 200
    ids = {r["capture_id"] for r in records}
    assert "f0" not in ids  # the oldest failed record was pruned by the live path
    assert "f1" in ids
    assert res["capture_id"] in ids
    assert all(f"d{i}" in ids for i in range(190))  # no taught example dropped


async def test_record_cap_refuses_live_start_when_only_done(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """At the cap with only done records, capture_start raises capture_store_full (WR-01)."""
    only_done = [_record(f"done{i}", "done", "2026-11-20T00:00:00+00:00") for i in range(200)]
    await _setup(
        hass, energy_manager, std_prefs, hass_storage, freezer, monkeypatch, seeded=only_done
    )
    before = json.dumps(_stored_captures(hass_storage), sort_keys=True)

    with pytest.raises(ServiceValidationError) as full:
        await _start(hass)
    assert full.value.translation_key == "capture_store_full"
    assert json.dumps(_stored_captures(hass_storage), sort_keys=True) == before


async def test_refused_capture_does_not_prune_memory_or_store(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A refused capture leaves memory and the Store identical — no prune divergence (WR-02)."""
    failed = [_record(f"f{i}", "failed", "2026-11-19T00:00:00+00:00") for i in range(10)]
    done = [_record(f"d{i}", "done", "2026-11-20T00:00:00+00:00") for i in range(190)]
    entry = await _setup(
        hass, energy_manager, std_prefs, hass_storage, freezer, monkeypatch, seeded=[*failed, *done]
    )
    captures = entry.runtime_data.captures
    on_ms, off_ms = _TRUTH.kettle_runs[-1]
    before_mem = json.dumps(captures._records, sort_keys=True)
    before_store = json.dumps(_stored_captures(hass_storage), sort_keys=True)

    # Refused before the executor job: an unknown candidate target.
    with pytest.raises(ServiceValidationError) as err:
        await _capture(
            hass,
            start_ms=on_ms - 45_000,
            end_ms=off_ms + 50_000,
            target_type="candidate",
            target="NoSuchCandidate",
        )
    assert err.value.translation_key == "capture_target_unknown"

    # A refused capture must not have pruned the memory (which would diverge from
    # the unsaved Store) — the oldest failed record is still present in both.
    assert json.dumps(captures._records, sort_keys=True) == before_mem
    assert json.dumps(_stored_captures(hass_storage), sort_keys=True) == before_store
    assert "f0" in {r["capture_id"] for r in captures._records}
    assert len(captures._records) == len(_stored_captures(hass_storage)) == 200


async def test_settle_after_delete_does_not_overwrite_another_record(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A delete during the clean-preview await must not write the result onto another record (WR-03)."""
    on_ms, off_ms = _TRUTH.kettle_runs[-1]
    run_b = _TRUTH.kettle_runs[-2]
    seed0 = _record("seed0", "done", "2026-11-19T00:00:00+00:00")
    entry = await _setup(
        hass,
        energy_manager,
        std_prefs,
        hass_storage,
        freezer,
        monkeypatch,
        now_ms=on_ms - 60_000,
        seeded=[seed0],
    )
    captures = entry.runtime_data.captures

    await _advance(hass, freezer, 80)
    live = await _start(hass)
    live_id = live["capture_id"]
    await _advance(hass, freezer, (off_ms - on_ms) / 1000)
    await _stop(hass, live_id)

    # A window capture appended after the resolving live record: after the seed is
    # deleted a stale index for the live record would point at this window record.
    window = await _capture(hass, start_ms=run_b[0] - 45_000, end_ms=run_b[1] + 50_000)
    window_id = window["capture_id"]
    window_before = json.dumps(
        next(r for r in _stored_captures(hass_storage) if r["capture_id"] == window_id),
        sort_keys=True,
    )

    # Simulate a concurrent delete of the seed record landing during the preview
    # await: the live record shifts from index 1 to 0, so a stale index would
    # write its result onto the window record that moves into index 1.
    orig = type(captures)._register_clean_preview

    async def _deleting(record: dict[str, Any]) -> None:
        captures._records = [r for r in captures._records if r["capture_id"] != "seed0"]
        await orig(captures, record)

    monkeypatch.setattr(captures, "_register_clean_preview", _deleting)

    await _advance(hass, freezer, 90)  # fire the settle timer

    by_id = {r["capture_id"]: r for r in captures._records}
    assert set(by_id) == {live_id, window_id}  # no duplicate live record, window intact
    assert by_id[live_id]["state"] == "done"
    assert by_id[live_id]["result"]["verdict"] == "clean"
    assert json.dumps(by_id[window_id], sort_keys=True) == window_before
    stored_ids = {r["capture_id"] for r in _stored_captures(hass_storage)}
    assert stored_ids == {live_id, window_id}


async def test_settle_after_cancel_discards_late_result(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cancel during the clean-preview await keeps the record cancelled (WR-03)."""
    on_ms, off_ms = _TRUTH.kettle_runs[-1]
    entry = await _setup(
        hass, energy_manager, std_prefs, hass_storage, freezer, monkeypatch, now_ms=on_ms - 60_000
    )
    captures = entry.runtime_data.captures

    await _advance(hass, freezer, 80)
    live = await _start(hass)
    live_id = live["capture_id"]
    await _advance(hass, freezer, (off_ms - on_ms) / 1000)
    await _stop(hass, live_id)

    orig = type(captures)._register_clean_preview

    async def _cancelling(record: dict[str, Any]) -> None:
        captures._records = [
            {**r, "state": "cancelled", "result": None}
            if r["capture_id"] == live_id
            else r
            for r in captures._records
        ]
        await captures._store.async_save({"captures": captures._records})
        await orig(captures, record)

    monkeypatch.setattr(captures, "_register_clean_preview", _cancelling)

    await _advance(hass, freezer, 90)  # fire the settle timer

    rec = next(r for r in captures._records if r["capture_id"] == live_id)
    assert rec["state"] == "cancelled"
    assert rec["result"] is None
    stored = next(r for r in _stored_captures(hass_storage) if r["capture_id"] == live_id)
    assert stored["state"] == "cancelled"
    assert stored["result"] is None


async def test_preview_registration_failure_does_not_prune(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A raise inside preview registration must not prune memory without a save.

    The cap prune runs after preview registration, so a failure there leaves
    memory and the Store identical (no WR-02 divergence).
    """
    failed = [_record(f"f{i}", "failed", "2026-11-19T00:00:00+00:00") for i in range(10)]
    done = [_record(f"d{i}", "done", "2026-11-20T00:00:00+00:00") for i in range(190)]
    entry = await _setup(
        hass, energy_manager, std_prefs, hass_storage, freezer, monkeypatch, seeded=[*failed, *done]
    )
    captures = entry.runtime_data.captures
    on_ms, off_ms = _TRUTH.kettle_runs[-1]
    before_mem = json.dumps(captures._records, sort_keys=True)
    before_store = json.dumps(_stored_captures(hass_storage), sort_keys=True)

    async def _boom(record: dict[str, Any]) -> None:
        raise RuntimeError("preview refit failed")

    monkeypatch.setattr(captures, "_register_clean_preview", _boom)

    with pytest.raises(Exception):
        await _capture(hass, start_ms=on_ms - 45_000, end_ms=off_ms + 50_000)

    assert json.dumps(captures._records, sort_keys=True) == before_mem
    assert json.dumps(_stored_captures(hass_storage), sort_keys=True) == before_store
    assert len(captures._records) == 200


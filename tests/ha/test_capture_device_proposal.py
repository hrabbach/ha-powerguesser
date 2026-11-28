"""Device-target captures become gated, accept-only proposals (CAP-13, D-09/D-10).

Plan 09-06 Task 1 and Task 2. A clean, paired capture aimed at a confirmed
device is batched into the next learning run: after the run's candidates are
saved, the supervisor runs 09-05's capture_refit stage on the eligible captures
and routes the gated result into the existing proposal path with source, capture
ids and a supersession record. Every production compile round of a run whose
captures carry a holdout window is judged on the capture-excluded table, and the
device menu explains the proposal, the captures that do not train and the
advisory replay flag.

The device and its captures are seeded deterministically (a seeded candidate is
confirmed and capture records are written straight into the captures Store), and
where a routing outcome is asserted the capture_refit stage is stubbed; the two
run-wide invariant tests drive the real compile_score and capture_refit stages
through the same in-process worker fixture every HA stage test uses.
"""

import asyncio
import copy
import json
import logging
from pathlib import Path
from typing import Any

import pytest

from homeassistant.components.energy.data import EnergyManager
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType
from homeassistant.util import dt as dt_util

from custom_components.powerguesser.const import (
    BUFFER_DIR,
    CAPTURES_STORE_KEY,
    CAPTURES_STORE_VERSION,
    DOMAIN,
    LEARNING_DIR,
)
from custom_components.powerguesser.engine.capture.constants import HOLDOUT_PAD_STEPS
from custom_components.powerguesser.engine.learning import worker as engine_worker
from custom_components.powerguesser.engine.learning.artefact import SignatureArtefact
from custom_components.powerguesser.engine.intervals import ProfileSnapshot, gate
from custom_components.powerguesser.capture import CaptureManager
from tests.conftest import (
    async_confirm_candidate,
    async_fill_synth_buffer,
    async_open_options_step,
    async_seed_consent,
    async_seed_llm,
    async_set_power,
    async_setup_via_flow,
    make_switched_candidate,
    mock_llm,
    seed_candidates,
)
from tests.fixtures.learning.fake_llm import chat_completion, kettle_proposal
from tests.fixtures.learning.synth import SynthSpec, generate

GRID_S = 30
DAY_MS = 86_400_000
HUGE_WINDOW = [[0, 10**15]]

import re

_CLUSTER_RE = re.compile(
    r"id=(?P<id>c\d+)\s+kind=(?P<kind>\w+)\s+n=(?P<n>\d+)\s+"
    r"rise=(?P<r10>-?[\d.]+)/(?P<r50>-?[\d.]+)/(?P<r90>-?[\d.]+)\s+"
    r"fall=(?P<f50>-?[\d.]+)\s+dur_min=(?P<d10>[\d.]+)-(?P<d90>[\d.]+)"
)


def _digest_from_request(request_data: Any) -> dict[str, Any]:
    """Rebuild a digest-shaped dict from the rendered digest in an LLM request."""
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


def _kettle_responder(label: str = "Kettle") -> Any:
    """Responder proposing the kettle cluster, relabelled as requested."""

    def _responder(data: Any) -> dict[str, Any]:
        prop = kettle_proposal(_digest_from_request(data))
        prop["devices"][0]["label"] = label
        return {"status": 200, "json": chat_completion(json.dumps(prop))}

    return _responder


def _pin_grid(entry: Any, grid_s: int = GRID_S) -> None:
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


def _set_std_powers(hass: HomeAssistant) -> None:
    async_set_power(hass, "sensor.grid_power", 500)
    async_set_power(hass, "sensor.solar_power", 1200)
    async_set_power(hass, "sensor.battery_power", -300)
    async_set_power(hass, "sensor.washer_power", 200)


def _capture_record(
    capture_id: str,
    dev_uuid: str,
    *,
    rise_ms: int = 10**12,
    fall_ms: int = 10**12 + 180_000,
    verdict: str | None = "clean",
    replay: dict[str, Any] | None = None,
    resolved_at: str = "2026-11-20T00:05:00+00:00",
    state: str = "done",
) -> dict[str, Any]:
    """One stored capture record aimed at a confirmed device (D-09/D-10)."""
    result = None
    if verdict is not None:
        result = {
            "verdict": verdict,
            "rise_ms": int(rise_ms),
            "fall_ms": int(fall_ms),
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
        "target_kind": "device",
        "target_id": dev_uuid,
        "new_device_name": None,
        "label": "Kettle capture",
        "press_start_ts": None,
        "press_stop_ts": None,
        "window_start_ms": int(rise_ms),
        "window_end_ms": int(fall_ms),
        "created_at": "2026-11-20T00:00:00+00:00",
        "resolved_at": resolved_at,
        "reassigned_at": None,
        "reason_code": None,
        "gap_flagged": False,
        "result": result,
        "features": {
            "rise_w": 2000.0,
            "fall_w": -2000.0,
            "dur_s": 180.0,
            "plateau_w": 2000.0,
            "hour": 8,
            "dow": 0,
        },
        "trace": None,
        "holdout_excluded": True,
        "holdout_window_ms": [int(rise_ms), int(fall_ms) + GRID_S * 1000],
        "replay": replay,
        "earlier_runs": None,
    }


async def _seed_captures(
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


async def _setup_device_entry(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    *,
    days: int = 22,
    kettle_plug: bool = True,
    confirm: bool = True,
    fill: bool = True,
) -> tuple[Any, str | None]:
    """Entry with a synth buffer and (optionally) a confirmed kettle device."""
    energy_manager.data = copy.deepcopy(std_prefs)
    _set_std_powers(hass)
    seed_candidates(
        hass_storage, [make_switched_candidate("sig01", "Kettle", rise_w=2000.0)]
    )
    entry = await async_setup_via_flow(hass)
    _pin_grid(entry)
    await hass.async_block_till_done()
    if fill:
        await async_fill_synth_buffer(
            hass, entry, SynthSpec(days=days, kettle_plug=kettle_plug)
        )
    await async_seed_llm(hass, entry)
    await async_seed_consent(hass, host="llm.test", names=False)
    dev_uuid: str | None = None
    if confirm:
        await async_confirm_candidate(hass, entry, "sig01", "Kettle")
        dev_uuid = entry.runtime_data.devices.device_for_signature("sig01")
        assert dev_uuid is not None
    return entry, dev_uuid


async def _run_learning(sup: Any) -> None:
    await sup.async_request_run("service")
    assert sup.run_task is not None
    await sup.run_task


def _run_dir(hass: HomeAssistant, sup: Any) -> Path:
    run_id = str(sup.coordinator.data["last_run"]["run_id"])
    return Path(hass.config.path(BUFFER_DIR, LEARNING_DIR)) / run_id


async def _open_device_menu(hass: HomeAssistant, entry: Any, dev_uuid: str) -> Any:
    res_review = await async_open_options_step(hass, entry, "review")
    assert res_review["type"] is FlowResultType.FORM
    return await hass.config_entries.options.async_configure(
        res_review["flow_id"], {"candidate": f"device_{dev_uuid}"}
    )


def _gated_artefact(current: dict[str, Any], capture_ids: list[str]) -> dict[str, Any]:
    """A gated capture-refit artefact derived from the device's active version."""
    art = copy.deepcopy(current)
    art["provenance"] = {
        "source": "capture",
        "capture_ids": sorted(capture_ids),
        "captures_snapshot_at": "2026-11-20T00:05:00+00:00",
    }
    art["proposal_gate"] = {
        "passed": True,
        "proposed_S": 0.0601,
        "active_S": 0.055,
        "basis": "learning_split_val",
    }
    return art


def _capture_refit_stub(devices_result: list[dict[str, Any]], calls: list[int]) -> Any:
    """A capture_refit stage stub that writes `devices_result` and counts calls."""

    def _stub(job: dict[str, Any]) -> dict[str, Any]:
        calls.append(1)
        stage_dir = Path(job["stage_dir"])
        path = stage_dir / "capture_refit.json"
        path.write_text(json.dumps({"devices": devices_result}), encoding="utf-8")
        return {
            "stage_dir": str(stage_dir),
            "result": str(path),
            "n_devices": len(devices_result),
        }

    return _stub


def _forced_diff_refit_from_rows() -> Any:
    """Wrap worker._refit_from_rows so a refit always differs from the active version."""
    original = engine_worker._refit_from_rows

    def _spy(current: Any, table: Any, union_idx: Any, threshold_w: Any, *, provenance: Any) -> Any:
        art = original(current, table, union_idx, threshold_w, provenance=provenance)
        data = art.to_dict()
        if data["states"]:
            dur = data["states"][0]["dur_s"]
            dur["hi"] = float(dur["hi"]) + 1.0
        return SignatureArtefact.from_dict(data)

    return _spy


def _overlaps(row_start: float, row_end: float, windows: list, pad_ms: int) -> bool:
    for window in windows:
        if row_start < int(window[1]) + pad_ms and row_end >= int(window[0]) - pad_ms:
            return True
    return False


def _plain(text: str) -> str:
    """Undo the menu's markdown escaping so assertions read the plain sentence."""
    return text.replace("\\", "")


def _device_id(entry: Any, dev_uuid: str) -> str | None:
    """The confirmed device's energy sensor entity id, if registered."""
    from homeassistant.helpers import entity_registry as er

    return er.async_get(entry.runtime_data.hass).async_get_entity_id(
        "sensor", "powerguesser", f"{entry.entry_id}_{dev_uuid}_energy"
    )


# --------------------------------------------------------------------------- #
# Task 1: scheduling, routing, eligibility, supersession and the run invariants
# --------------------------------------------------------------------------- #


async def test_capture_proposal_accept_and_rollback(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    aioclient_mock: Any,
    inprocess_worker: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Three clean kettle captures become a gated capture proposal after the next
    run; nothing changes until Accept, which stores version 2 with source capture
    and can be rolled back."""
    entry, dev_uuid = await _setup_device_entry(
        hass, energy_manager, std_prefs, hass_storage
    )
    assert dev_uuid is not None
    devices = entry.runtime_data.devices
    v1 = copy.deepcopy(devices.record(dev_uuid)["versions"][0]["artefact"])
    capture_ids = ["cap0", "cap1", "cap2"]
    await _seed_captures(
        hass,
        entry,
        hass_storage,
        [
            _capture_record(cid, dev_uuid, rise_ms=10**12 + i * 10**6)
            for i, cid in enumerate(capture_ids)
        ],
    )
    calls: list[int] = []
    monkeypatch.setattr(
        engine_worker,
        "capture_refit",
        _capture_refit_stub(
            [
                {
                    "device_uuid": dev_uuid,
                    "signature_id": "sig01",
                    "reason": "gated",
                    "passed": True,
                    "proposed_S": 0.0601,
                    "active_S": 0.055,
                    "basis": "learning_split_val",
                    "artefact": _gated_artefact(v1, capture_ids),
                    "capture_ids": capture_ids,
                    "n_unmapped": 0,
                }
            ],
            calls,
        ),
    )
    mock_llm(aioclient_mock, "http://llm.test/v1", _kettle_responder("Kettle"))
    sup = entry.runtime_data.learning

    energy_eid = _device_id(entry, dev_uuid)
    energy_before = hass.states.get(energy_eid).state if energy_eid else None
    v1_blob = json.dumps(devices.record(dev_uuid)["versions"][0], sort_keys=True)

    await _run_learning(sup)

    rec = devices.record(dev_uuid)
    proposal = rec["pending_proposal"]
    assert proposal is not None
    assert proposal["source"] == "capture"
    assert proposal["capture_ids"] == capture_ids
    assert isinstance(proposal["captures_snapshot_at"], str)
    # Before Accept nothing moves: still version 1, same artefact, same energy.
    assert rec["active_v"] == 1
    assert [v["v"] for v in rec["versions"]] == [1]
    assert json.dumps(rec["versions"][0], sort_keys=True) == v1_blob
    if energy_eid:
        assert hass.states.get(energy_eid).state == energy_before

    next_v = await devices.async_accept_proposal(dev_uuid)
    assert next_v == 2
    rec = devices.record(dev_uuid)
    assert rec["versions"][1]["source"] == "capture"
    assert rec["active_v"] == 2
    assert rec["pending_proposal"] is None

    await devices.async_activate_version(dev_uuid, 1)
    assert devices.record(dev_uuid)["active_v"] == 1


async def test_contaminated_capture_never_trains(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    inprocess_worker: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Only contaminated/ambiguous device captures: no capture_refit job, no
    proposal and no rejection record."""
    entry, dev_uuid = await _setup_device_entry(
        hass, energy_manager, std_prefs, hass_storage
    )
    assert dev_uuid is not None
    devices = entry.runtime_data.devices
    await _seed_captures(
        hass,
        entry,
        hass_storage,
        [
            _capture_record("cap-c", dev_uuid, verdict="contaminated"),
            _capture_record("cap-a", dev_uuid, verdict="ambiguous"),
        ],
    )
    calls: list[int] = []
    monkeypatch.setattr(engine_worker, "capture_refit", _capture_refit_stub([], calls))
    sup = entry.runtime_data.learning

    await sup._async_capture_proposals(tmp_path, "run-x", "2026-11-20T00:00:00Z", True)

    assert calls == []
    assert not (tmp_path / "capture_refit_job.json").exists()
    rec = devices.record(dev_uuid)
    assert rec["pending_proposal"] is None
    assert "last_rejected_proposal" not in rec.get("validation", {})


async def test_failing_gate_records_rejection(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    inprocess_worker: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A capture_refit result that passed False is recorded as
    validation.last_rejected_proposal with reason worse_than_active and source
    capture; no pending proposal is stored."""
    entry, dev_uuid = await _setup_device_entry(
        hass, energy_manager, std_prefs, hass_storage
    )
    assert dev_uuid is not None
    devices = entry.runtime_data.devices
    await _seed_captures(hass, entry, hass_storage, [_capture_record("cap0", dev_uuid)])
    monkeypatch.setattr(
        engine_worker,
        "capture_refit",
        _capture_refit_stub(
            [
                {
                    "device_uuid": dev_uuid,
                    "signature_id": "sig01",
                    "reason": "worse_than_active",
                    "passed": False,
                    "proposed_S": 0.01,
                    "active_S": 0.05,
                    "basis": "learning_split_val",
                    "artefact": None,
                    "capture_ids": ["cap0"],
                    "n_unmapped": 0,
                }
            ],
            [],
        ),
    )
    sup = entry.runtime_data.learning

    await sup._async_capture_proposals(tmp_path, "run-r", "2026-11-20T00:00:00Z", True)

    rec = devices.record(dev_uuid)
    assert rec["pending_proposal"] is None
    rejected = rec["validation"]["last_rejected_proposal"]
    assert rejected["reason"] == "worse_than_active"
    assert rejected["source"] == "capture"
    assert rejected["proposed_S"] == 0.01
    assert rejected["active_S"] == 0.05


async def test_newer_proposal_supersedes_and_logs(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    inprocess_worker: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: Any,
) -> None:
    """A gated capture proposal supersedes a pending LLM one and the supersession
    is recorded with an INFO log."""
    entry, dev_uuid = await _setup_device_entry(
        hass, energy_manager, std_prefs, hass_storage
    )
    assert dev_uuid is not None
    devices = entry.runtime_data.devices
    v1 = copy.deepcopy(devices.record(dev_uuid)["versions"][0]["artefact"])
    await devices.async_set_proposal(
        dev_uuid,
        copy.deepcopy(v1),
        run_id="run-llm",
        proposed_at="2026-11-19T00:00:00Z",
        trained_at="2026-11-19T00:00:00Z",
    )
    await _seed_captures(hass, entry, hass_storage, [_capture_record("cap0", dev_uuid)])
    monkeypatch.setattr(
        engine_worker,
        "capture_refit",
        _capture_refit_stub(
            [
                {
                    "device_uuid": dev_uuid,
                    "signature_id": "sig01",
                    "reason": "gated",
                    "passed": True,
                    "proposed_S": 0.06,
                    "active_S": 0.055,
                    "basis": "learning_split_val",
                    "artefact": _gated_artefact(v1, ["cap0"]),
                    "capture_ids": ["cap0"],
                    "n_unmapped": 0,
                }
            ],
            [],
        ),
    )
    sup = entry.runtime_data.learning

    with caplog.at_level(logging.INFO, logger="custom_components.powerguesser.devices"):
        await sup._async_capture_proposals(
            tmp_path, "run-capture", "2026-11-20T00:00:00Z", True
        )

    rec = devices.record(dev_uuid)
    assert rec["pending_proposal"]["source"] == "capture"
    assert rec["pending_proposal"]["run_id"] == "run-capture"
    superseded = rec["validation"]["last_superseded_proposal"]
    assert superseded["run_id"] == "run-llm"
    assert superseded["source"] == "proposal"
    assert superseded["superseded_by_run_id"] == "run-capture"
    assert "supersedes" in caplog.text


async def test_second_run_without_new_captures_is_noop(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    inprocess_worker: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A second run over the same captures leaves the pending proposal and the
    validation untouched (no supersession record)."""
    entry, dev_uuid = await _setup_device_entry(
        hass, energy_manager, std_prefs, hass_storage
    )
    assert dev_uuid is not None
    devices = entry.runtime_data.devices
    v1 = copy.deepcopy(devices.record(dev_uuid)["versions"][0]["artefact"])
    await _seed_captures(hass, entry, hass_storage, [_capture_record("cap0", dev_uuid)])
    calls: list[int] = []
    monkeypatch.setattr(
        engine_worker,
        "capture_refit",
        _capture_refit_stub(
            [
                {
                    "device_uuid": dev_uuid,
                    "signature_id": "sig01",
                    "reason": "gated",
                    "passed": True,
                    "proposed_S": 0.06,
                    "active_S": 0.055,
                    "basis": "learning_split_val",
                    "artefact": _gated_artefact(v1, ["cap0"]),
                    "capture_ids": ["cap0"],
                    "n_unmapped": 0,
                }
            ],
            calls,
        ),
    )
    sup = entry.runtime_data.learning

    await sup._async_capture_proposals(tmp_path, "run-1", "2026-11-20T00:00:00Z", True)
    assert calls == [1]
    rec = devices.record(dev_uuid)
    proposal_before = json.dumps(rec["pending_proposal"], sort_keys=True)
    validation_before = json.dumps(rec.get("validation"), sort_keys=True)

    await sup._async_capture_proposals(tmp_path, "run-2", "2026-11-20T01:00:00Z", True)

    assert calls == [1]  # no second job was written
    rec = devices.record(dev_uuid)
    assert json.dumps(rec["pending_proposal"], sort_keys=True) == proposal_before
    assert json.dumps(rec.get("validation"), sort_keys=True) == validation_before
    assert "last_superseded_proposal" not in rec.get("validation", {})


async def test_capture_after_snapshot_stays_eligible(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    inprocess_worker: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """An accepted capture version consumes only its own capture ids and its
    snapshot; a capture resolved later stays eligible for the next run."""
    entry, dev_uuid = await _setup_device_entry(
        hass, energy_manager, std_prefs, hass_storage
    )
    assert dev_uuid is not None
    devices = entry.runtime_data.devices
    v1 = copy.deepcopy(devices.record(dev_uuid)["versions"][0]["artefact"])
    await _seed_captures(
        hass,
        entry,
        hass_storage,
        [_capture_record("cap0", dev_uuid, resolved_at="2026-11-20T00:05:00+00:00")],
    )
    monkeypatch.setattr(
        engine_worker,
        "capture_refit",
        _capture_refit_stub(
            [
                {
                    "device_uuid": dev_uuid,
                    "signature_id": "sig01",
                    "reason": "gated",
                    "passed": True,
                    "proposed_S": 0.06,
                    "active_S": 0.055,
                    "basis": "learning_split_val",
                    "artefact": _gated_artefact(v1, ["cap0"]),
                    "capture_ids": ["cap0"],
                    "n_unmapped": 0,
                }
            ],
            [],
        ),
    )
    sup = entry.runtime_data.learning
    snapshot_at = "2026-11-21T00:00:00+00:00"
    await sup._async_capture_proposals(tmp_path, "run-1", "2026-11-21T00:00:00Z", True)
    # Simulate the user Accept: version 2 carries cap0 and the snapshot.
    rec = devices.record(dev_uuid)
    rec["pending_proposal"]["captures_snapshot_at"] = snapshot_at
    await devices.async_accept_proposal(dev_uuid)
    assert devices.record(dev_uuid)["versions"][1]["source"] == "capture"

    # A capture that resolved after the accepted snapshot.
    records = entry.runtime_data.captures.records
    records.append(
        _capture_record("cap1", dev_uuid, resolved_at="2026-11-22T00:00:00+00:00")
    )
    await _seed_captures(hass, entry, hass_storage, records)

    entries, _snap = sup._capture_refit_devices()
    assert len(entries) == 1
    ids = [c["capture_id"] for c in entries[0]["captures"]]
    assert ids == ["cap1"]


async def test_capture_settling_during_job_build_stays_eligible(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A capture that settles after the job snapshot is absent from that job but
    carries a resolved_at not earlier than the snapshot, so the next run sees it.

    The settling capture's insight step is held on an event: the job snapshot is
    taken while the capture is still ``resolving`` (so it is not in the job), then
    the clock moves on and the capture commits with the commit-time re-stamp.
    Without that re-stamp its stored resolved_at would predate the snapshot and
    the capture would be lost for good.
    """
    entry, dev_uuid = await _setup_device_entry(
        hass, energy_manager, std_prefs, hass_storage
    )
    assert dev_uuid is not None
    hub = entry.runtime_data
    captures = hub.captures
    end_ms = int(
        round(dt_util.start_of_local_day(dt_util.now()).timestamp() * 1000)
    )
    on_ms, off_ms = generate(SynthSpec(days=22), end_ms).kettle_runs[-3]

    freezer.move_to(dt_util.utcnow())

    release = asyncio.Event()
    entered = asyncio.Event()

    async def _blocking_insight(self: CaptureManager, record: dict[str, Any]) -> None:
        entered.set()
        await release.wait()

    monkeypatch.setattr(CaptureManager, "_async_insight", _blocking_insight)

    def _iso(ms: int) -> str:
        return dt_util.utc_from_timestamp(ms / 1000.0).isoformat()

    task = asyncio.ensure_future(
        hass.services.async_call(
            DOMAIN,
            "capture_window",
            {
                "start": _iso(on_ms - 45_000),
                "end": _iso(off_ms + 50_000),
                "label": "Kettle",
                "target_type": "device",
                "target": dev_uuid,
            },
            blocking=True,
            return_response=True,
        )
    )
    await asyncio.wait_for(entered.wait(), timeout=10)

    freezer.tick(60)
    entries, snapshot_at = hub.learning._capture_refit_devices()
    assert entries == []  # the capture is still resolving: not committed yet

    release.set()
    await asyncio.wait_for(task, timeout=10)

    stored = next(r for r in captures.records if r.get("state") == "done")
    assert stored["resolved_at"] >= snapshot_at

    freezer.tick(60)
    entries2, _ = hub.learning._capture_refit_devices()
    assert len(entries2) == 1
    assert [c["capture_id"] for c in entries2[0]["captures"]] == [stored["capture_id"]]


async def test_replay_flag_does_not_gate_proposal(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
) -> None:
    """The Phase 8 replay flag is never an eligibility input: flipping every
    would_claim leaves the capture_refit job devices unchanged."""
    entry, dev_uuid = await _setup_device_entry(
        hass, energy_manager, std_prefs, hass_storage
    )
    assert dev_uuid is not None
    caps = [
        _capture_record("cap-a", dev_uuid, replay={"would_claim": True, "n_episodes": 3, "basis": "x"}),
        _capture_record("cap-b", dev_uuid, replay={"would_claim": False, "n_episodes": 2, "basis": "x"}),
        _capture_record("cap-c", dev_uuid, replay=None),
    ]
    await _seed_captures(hass, entry, hass_storage, caps)
    sup = entry.runtime_data.learning

    entries, _ = sup._capture_refit_devices()
    assert len(entries) == 1
    ids = sorted(c["capture_id"] for c in entries[0]["captures"])
    assert ids == ["cap-a", "cap-b", "cap-c"]

    for cap in caps:
        if isinstance(cap["replay"], dict):
            cap["replay"]["would_claim"] = not cap["replay"]["would_claim"]
    await _seed_captures(hass, entry, hass_storage, caps)

    entries2, _ = sup._capture_refit_devices()
    assert json.dumps(entries, sort_keys=True) == json.dumps(entries2, sort_keys=True)


async def test_device_capture_counts_reports_replay(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
) -> None:
    """device_capture_counts reports clean, not_clean, replay_checked and
    replay_claimed over the device's done captures."""
    entry, dev_uuid = await _setup_device_entry(
        hass, energy_manager, std_prefs, hass_storage
    )
    assert dev_uuid is not None
    await _seed_captures(
        hass,
        entry,
        hass_storage,
        [
            _capture_record("cap-a", dev_uuid, replay={"would_claim": True, "n_episodes": 4, "basis": "x"}),
            _capture_record("cap-b", dev_uuid, replay={"would_claim": False, "n_episodes": 1, "basis": "x"}),
            _capture_record("cap-c", dev_uuid, verdict="contaminated", replay=None),
        ],
    )
    counts = entry.runtime_data.captures.device_capture_counts(dev_uuid)
    assert counts == {
        "clean": 2,
        "not_clean": 1,
        "replay_checked": 2,
        "replay_claimed": 1,
    }


async def test_stage_failure_does_not_fail_run(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    aioclient_mock: Any,
    inprocess_worker: None,
    monkeypatch: pytest.MonkeyPatch,
    caplog: Any,
) -> None:
    """A raising capture_refit logs a warning and never fails the run."""
    entry, dev_uuid = await _setup_device_entry(
        hass, energy_manager, std_prefs, hass_storage
    )
    assert dev_uuid is not None
    await _seed_captures(hass, entry, hass_storage, [_capture_record("cap0", dev_uuid)])

    def _boom(job: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("capture_refit blew up")

    monkeypatch.setattr(engine_worker, "capture_refit", _boom)
    mock_llm(aioclient_mock, "http://llm.test/v1", _kettle_responder("Kettle"))
    sup = entry.runtime_data.learning

    with caplog.at_level(logging.WARNING):
        await _run_learning(sup)

    assert sup.coordinator.data["status"] == "idle"
    assert sup.coordinator.data["last_error"] is None
    # The run's candidates were saved; the kettle load is owned by the confirmed
    # device, so it is hidden from Review but present in the supervisor.
    assert sup.candidates != []
    assert "capture_refit stage failed" in caplog.text


async def test_llm_proposal_keys_unchanged(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
) -> None:
    """An LLM-path proposal keeps exactly its five pending-proposal keys."""
    entry, dev_uuid = await _setup_device_entry(
        hass, energy_manager, std_prefs, hass_storage
    )
    assert dev_uuid is not None
    devices = entry.runtime_data.devices
    v1 = copy.deepcopy(devices.record(dev_uuid)["versions"][0]["artefact"])
    await devices.async_set_proposal(
        dev_uuid,
        copy.deepcopy(v1),
        run_id="run-llm",
        proposed_at="2026-11-19T00:00:00Z",
        trained_at="2026-11-19T00:00:00Z",
    )
    proposal = devices.record(dev_uuid)["pending_proposal"]
    assert set(proposal.keys()) == {
        "artefact",
        "run_id",
        "proposed_at",
        "trained_at",
        "scores",
    }


# --------------------------------------------------------------------------- #
# Run-wide invariants: the gate table, every production round, the empty block
# --------------------------------------------------------------------------- #


async def test_no_gate_call_scores_a_captured_row(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    aioclient_mock: Any,
    inprocess_worker: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every gate call of a run with capture records reads a table with no block-1
    row overlapping a padded capture window, and the compile job carries them."""
    import numpy as np

    entry, dev_uuid = await _setup_device_entry(
        hass, energy_manager, std_prefs, hass_storage
    )
    assert dev_uuid is not None
    await _seed_captures(
        hass,
        entry,
        hass_storage,
        [
            _capture_record("cap0", dev_uuid, rise_ms=10**12),
            _capture_record("cap1", dev_uuid, rise_ms=10**12 + 10**6),
        ],
    )
    # Force the capture refit to differ from the active version so the stage gates.
    monkeypatch.setattr(
        engine_worker, "_refit_from_rows", _forced_diff_refit_from_rows()
    )
    tables: list[Any] = []
    original_gate = engine_worker.gate_proposal

    def _gate_spy(proposed: Any, active: Any, table: Any, residual_wh: Any) -> Any:
        tables.append(table)
        return original_gate(proposed, active, table, residual_wh)

    monkeypatch.setattr(engine_worker, "gate_proposal", _gate_spy)
    mock_llm(aioclient_mock, "http://llm.test/v1", _kettle_responder("Kettle"))
    sup = entry.runtime_data.learning

    await _run_learning(sup)

    windows = entry.runtime_data.captures.holdout_windows()
    assert windows
    run_dir = _run_dir(hass, sup)
    with np.load(str(run_dir / "residual.npz"), allow_pickle=False) as res:
        start_ms = int(res["start_ms"][0])
        step_ms = int(res["step_ms"][0])
    pad_ms = HOLDOUT_PAD_STEPS * step_ms
    # compile_score's gate and capture_refit's gate both ran.
    assert len(tables) >= 2, [list(t.keys()) for t in tables]
    for table in tables:
        for i in np.where(table["block"] == 1)[0]:
            assert not _overlaps(
                int(start_ms + table["start_i"][i] * step_ms),
                int(start_ms + table["end_i"][i] * step_ms),
                windows,
                pad_ms,
            )
    job = json.loads((run_dir / "job_compile_1.json").read_text(encoding="utf-8"))
    assert job["capture_windows"] == windows


async def test_every_production_round_carries_capture_windows(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    aioclient_mock: Any,
    inprocess_worker: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With capture records and one refinement round, round 1 and round 2 compile
    jobs both carry the capture windows; the shadow jobs carry no key."""
    monkeypatch.setattr("custom_components.powerguesser.learning.MAX_REFINE_ROUNDS", 1)
    entry, dev_uuid = await _setup_device_entry(
        hass, energy_manager, std_prefs, hass_storage, days=35
    )
    assert dev_uuid is not None
    await _seed_captures(hass, entry, hass_storage, [_capture_record("cap0", dev_uuid)])
    mock_llm(aioclient_mock, "http://llm.test/v1", _kettle_responder("Kettle"))
    sup = entry.runtime_data.learning

    await _run_learning(sup)

    windows = entry.runtime_data.captures.holdout_windows()
    assert windows
    run_dir = _run_dir(hass, sup)
    for round_no in (1, 2):
        job = json.loads(
            (run_dir / f"job_compile_{round_no}.json").read_text(encoding="utf-8")
        )
        assert job["capture_windows"] == windows
    shadow_job = json.loads(
        (run_dir / "shadow" / "job_compile_1.json").read_text(encoding="utf-8")
    )
    assert "capture_windows" not in shadow_job


async def test_capture_free_compile_job_unchanged(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    aioclient_mock: Any,
    inprocess_worker: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no capture records the compile job has no capture_windows key and the
    gate result on the library artefact keeps exactly its four keys."""
    entry, dev_uuid = await _setup_device_entry(
        hass, energy_manager, std_prefs, hass_storage
    )
    assert dev_uuid is not None
    mock_llm(aioclient_mock, "http://llm.test/v1", _kettle_responder("Kettle"))
    sup = entry.runtime_data.learning

    await _run_learning(sup)

    run_dir = _run_dir(hass, sup)
    job = json.loads((run_dir / "job_compile_1.json").read_text(encoding="utf-8"))
    assert "capture_windows" not in job
    library = json.loads((run_dir / "library_1.json").read_text(encoding="utf-8"))
    gated = [a for a in library if "proposal_gate" in a]
    assert gated, "the confirmed device's matched artefact carries no gate"
    for art in gated:
        assert set(art["proposal_gate"].keys()) == {
            "passed",
            "proposed_S",
            "active_S",
            "basis",
        }


async def test_insufficient_validation_recorded(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    aioclient_mock: Any,
    inprocess_worker: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An emptied validation block is never a pass: the capture path records
    insufficient_validation with null scores and source capture, and the
    LLM-proposal path records it too, with no pending proposal either way."""
    entry, dev_uuid = await _setup_device_entry(
        hass, energy_manager, std_prefs, hass_storage
    )
    assert dev_uuid is not None
    devices = entry.runtime_data.devices
    captures = entry.runtime_data.captures
    await _seed_captures(hass, entry, hass_storage, [_capture_record("cap0", dev_uuid)])
    monkeypatch.setattr(captures, "holdout_windows", lambda: HUGE_WINDOW)
    monkeypatch.setattr(
        engine_worker, "_refit_from_rows", _forced_diff_refit_from_rows()
    )
    mock_llm(aioclient_mock, "http://llm.test/v1", _kettle_responder("Kettle"))
    sup = entry.runtime_data.learning

    await _run_learning(sup)

    rec = devices.record(dev_uuid)
    assert rec["pending_proposal"] is None
    rejected = rec["validation"]["last_rejected_proposal"]
    assert rejected["reason"] == "insufficient_validation"
    assert rejected["source"] == "capture"
    assert rejected["proposed_S"] is None
    assert rejected["active_S"] is None

    # The LLM-proposal path through _route_device_matches records the same reason
    # without a capture source.
    v1 = copy.deepcopy(rec["versions"][0]["artefact"])
    art = copy.deepcopy(v1)
    art["signature_id"] = "sig01"
    art["states"][0]["rise_w"] = {"lo": 600.0, "p50": 700.0, "hi": 800.0}
    art["proposal_gate"] = {
        "passed": False,
        "proposed_S": None,
        "active_S": None,
        "basis": "learning_split_val",
        "reason": "insufficient_validation",
    }
    await sup._route_device_matches([art], "run-insuff", "2026-11-22T00:00:00Z")
    rejected = devices.record(dev_uuid)["validation"]["last_rejected_proposal"]
    assert rejected["reason"] == "insufficient_validation"
    assert rejected["proposed_S"] is None
    assert "source" not in rejected


async def test_round1_insufficient_validation_skips_refinement(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    aioclient_mock: Any,
    inprocess_worker: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Round 1 on an emptied validation block keeps its candidates, requests no
    refinement round and reports the reason; the capture-free run still refines."""
    monkeypatch.setattr("custom_components.powerguesser.learning.MAX_REFINE_ROUNDS", 1)
    entry, _dev = await _setup_device_entry(
        hass, energy_manager, std_prefs, hass_storage, confirm=False
    )
    captures = entry.runtime_data.captures
    mock_llm(aioclient_mock, "http://llm.test/v1", _kettle_responder("Kettle"))
    sup = entry.runtime_data.learning

    monkeypatch.setattr(captures, "holdout_windows", lambda: HUGE_WINDOW)
    await _run_learning(sup)

    run_a = _run_dir(hass, sup)
    assert sup.coordinator.data["status"] == "idle"
    assert sup.coordinator.data["last_run"]["accepted_round"] == 1
    assert sup.coordinator.data["last_error"] == "refinement stopped: insufficient_validation"
    assert not (run_a / "request_2.json").exists()
    assert sup.visible_candidates() != []

    # Capture-free: no window, so round 2 is requested.
    monkeypatch.setattr(captures, "holdout_windows", lambda: [])
    await _run_learning(sup)

    run_b = _run_dir(hass, sup)
    assert (run_b / "request_2.json").is_file()
    assert sup.coordinator.data["last_error"] is None


# --------------------------------------------------------------------------- #
# Task 2: the device menu and the capture-free holdout
# --------------------------------------------------------------------------- #


async def test_device_menu_shows_capture_proposal(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
) -> None:
    """The device menu names the proposal's origin and the gate's scores; the
    {proposal} text is empty without a proposal."""
    entry, dev_uuid = await _setup_device_entry(
        hass, energy_manager, std_prefs, hass_storage
    )
    assert dev_uuid is not None
    devices = entry.runtime_data.devices
    v1 = copy.deepcopy(devices.record(dev_uuid)["versions"][0]["artefact"])

    menu = await _open_device_menu(hass, entry, dev_uuid)
    assert menu["description_placeholders"]["proposal"] == ""

    await devices.async_set_proposal(
        dev_uuid,
        _gated_artefact(v1, ["cap0", "cap1"]),
        run_id="run-capture",
        proposed_at="2026-11-20T00:00:00Z",
        trained_at="2026-11-20T00:00:00Z",
        source="capture",
        capture_ids=["cap0", "cap1"],
        captures_snapshot_at="2026-11-20T00:00:00Z",
    )
    await hass.async_block_till_done()
    menu = await _open_device_menu(hass, entry, dev_uuid)
    text = _plain(menu["description_placeholders"]["proposal"])
    assert "from your captures" in text
    assert "2 captured runs" in text
    assert "Validation-day score" in text
    assert "0.0601" in text and "0.0550" in text
    assert "learning_split_val" in text

    # An LLM proposal reads differently.
    await devices.async_set_proposal(
        dev_uuid,
        copy.deepcopy(v1),
        run_id="run-llm",
        proposed_at="2026-11-21T00:00:00Z",
        trained_at="2026-11-21T00:00:00Z",
    )
    await hass.async_block_till_done()
    menu = await _open_device_menu(hass, entry, dev_uuid)
    assert "from a learning run" in _plain(menu["description_placeholders"]["proposal"])


async def test_device_menu_names_non_teaching_captures(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
) -> None:
    """Two contaminated device captures: the menu says they are not used."""
    entry, dev_uuid = await _setup_device_entry(
        hass, energy_manager, std_prefs, hass_storage
    )
    assert dev_uuid is not None
    await _seed_captures(
        hass,
        entry,
        hass_storage,
        [
            _capture_record("cap-c1", dev_uuid, verdict="contaminated"),
            _capture_record("cap-c2", dev_uuid, verdict="ambiguous"),
        ],
    )
    menu = await _open_device_menu(hass, entry, dev_uuid)
    text = _plain(menu["description_placeholders"]["proposal"])
    assert (
        "2 captured runs on this device are not used for retraining "
        "(only clean captures train)" in text
    )


async def test_device_menu_shows_replay_flag(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
) -> None:
    """The replay check is shown as an evidence-only line, with and without a
    pending proposal, and omitted when no capture carries a replay result."""
    entry, dev_uuid = await _setup_device_entry(
        hass, energy_manager, std_prefs, hass_storage
    )
    assert dev_uuid is not None
    devices = entry.runtime_data.devices
    v1 = copy.deepcopy(devices.record(dev_uuid)["versions"][0]["artefact"])
    await _seed_captures(
        hass,
        entry,
        hass_storage,
        [
            _capture_record("cap-a", dev_uuid, replay={"would_claim": True, "n_episodes": 4, "basis": "x"}),
            _capture_record("cap-b", dev_uuid, replay={"would_claim": True, "n_episodes": 1, "basis": "x"}),
            _capture_record("cap-c", dev_uuid, replay={"would_claim": False, "n_episodes": 2, "basis": "x"}),
        ],
    )
    menu = await _open_device_menu(hass, entry, dev_uuid)
    text = _plain(menu["description_placeholders"]["proposal"])
    assert (
        "Replay check (evidence only): the active version alone would have "
        "claimed 2 of 3 checked captured runs; it does not decide whether an "
        "update is proposed." in text
    )

    # The same line appears beside a pending proposal.
    await devices.async_set_proposal(
        dev_uuid,
        _gated_artefact(v1, ["cap-a"]),
        run_id="run-capture",
        proposed_at="2026-11-20T00:00:00Z",
        trained_at="2026-11-20T00:00:00Z",
        source="capture",
        capture_ids=["cap-a"],
        captures_snapshot_at="2026-11-20T00:00:00Z",
    )
    await hass.async_block_till_done()
    menu = await _open_device_menu(hass, entry, dev_uuid)
    assert "Replay check (evidence only)" in _plain(
        menu["description_placeholders"]["proposal"]
    )

    # A device with no replay result shows no replay line.
    await _seed_captures(
        hass,
        entry,
        hass_storage,
        [_capture_record("cap-a", dev_uuid, replay=None)],
    )
    menu = await _open_device_menu(hass, entry, dev_uuid)
    assert "Replay check" not in _plain(menu["description_placeholders"]["proposal"])


async def test_rollback_labels_show_capture_source(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
) -> None:
    """After accepting a capture proposal the rollback options read the capture
    source for version 2."""
    entry, dev_uuid = await _setup_device_entry(
        hass, energy_manager, std_prefs, hass_storage
    )
    assert dev_uuid is not None
    devices = entry.runtime_data.devices
    v1 = copy.deepcopy(devices.record(dev_uuid)["versions"][0]["artefact"])
    await devices.async_set_proposal(
        dev_uuid,
        _gated_artefact(v1, ["cap0"]),
        run_id="run-capture",
        proposed_at="2026-11-20T00:00:00Z",
        trained_at="2026-11-20T00:00:00Z",
        source="capture",
        capture_ids=["cap0"],
        captures_snapshot_at="2026-11-20T00:00:00Z",
    )
    await devices.async_accept_proposal(dev_uuid)
    # Roll back to version 1 so version 2 (the capture version) is offered again.
    await devices.async_activate_version(dev_uuid, 1)

    menu = await _open_device_menu(hass, entry, dev_uuid)
    assert "rollback" in menu["menu_options"]
    res_rollback = await hass.config_entries.options.async_configure(
        menu["flow_id"], {"next_step_id": "rollback"}
    )
    assert res_rollback["type"] is FlowResultType.FORM
    options = res_rollback["data_schema"].schema["version"].config["options"]
    labels = {o["value"]: o["label"] for o in options}
    assert "capture" in labels["2"]


async def test_accepted_capture_version_keeps_holdout_exclusion(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    kettle_plug_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    inprocess_worker: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After a capture-trained version is accepted, run_validation still reports
    the capture windows in holdout.excluded_capture."""
    from tests.conftest import async_seed_shadow_set
    from tests.fixtures.detect_traces import switched_artefact

    entry, dev_uuid = await _setup_device_entry(
        hass, energy_manager, kettle_plug_prefs, hass_storage, days=35
    )
    assert dev_uuid is not None
    async_set_power(hass, "sensor.kettle_power", 0)
    devices = entry.runtime_data.devices
    v1 = copy.deepcopy(devices.record(dev_uuid)["versions"][0]["artefact"])
    await devices.async_set_proposal(
        dev_uuid,
        _gated_artefact(v1, ["cap0"]),
        run_id="run-capture",
        proposed_at="2026-11-20T00:00:00Z",
        trained_at="2026-11-20T00:00:00Z",
        source="capture",
        capture_ids=["cap0"],
        captures_snapshot_at="2026-11-20T00:00:00Z",
    )
    await devices.async_accept_proposal(dev_uuid)
    assert devices.record(dev_uuid)["versions"][1]["source"] == "capture"

    end_ms = int(
        round(dt_util.start_of_local_day(dt_util.now()).timestamp() * 1000)
    )
    await _seed_captures(
        hass,
        entry,
        hass_storage,
        [
            _capture_record(
                "cap0",
                dev_uuid,
                rise_ms=end_ms - 3 * DAY_MS,
                fall_ms=end_ms - 2 * DAY_MS,
            )
        ],
    )
    artefact = switched_artefact(
        "sig-shadow-kettle",
        rise=(1800, 2000, 2200),
        fall=(-2200, -2000, -1800),
        dur=(120, 180, 300),
        plateau=(1800, 2000, 2200),
    )
    await async_seed_shadow_set(
        hass,
        entry,
        [artefact],
        [
            {
                "entity_id": "sensor.kettle_power",
                "signature_id": "sig-shadow-kettle",
                "kind": "switched",
                "overlap": 0.8,
            }
        ],
    )

    res = await hass.services.async_call(
        DOMAIN, "run_validation", {}, blocking=True, return_response=True
    )
    assert res["status"] == "ok"
    assert res["holdout"]["excluded_capture"]["windows"] == 1




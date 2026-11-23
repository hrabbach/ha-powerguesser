"""Tests for the live drift guard on the device surface (Plan 05-05; D-15, D-16).

The detector's ClosedEpisode stream feeds each confirmed device's LiveDrift
tracker inside DeviceRuntime._drain_step; a changed verdict writes the D-16
flags (drift, drift_reason, drift_since) into the record's validation dict,
both device sensors publish them after unvalidated and before retired, the
diagnostic sensor shows drift_evidence, and the Review list appends
" — drift detected". Drift only sets flags: it never changes versions,
active_v, uuid, name or entity ids, never creates a Repairs issue, clears
after one good evaluation, survives restarts, and resets on version
activation (D-22).
"""

from typing import Any

import pytest

from homeassistant.components.energy.data import EnergyManager
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers import issue_registry as ir
from homeassistant.util import dt as dt_util

from custom_components.powerguesser.const import DOMAIN
from custom_components.powerguesser.const import (
    DEVICES_STORE_KEY,
    DEVICE_STATE_STORE_KEY,
    LEARNING_STORES_VERSION,
    REVIEW_STORE_KEY,
    VALIDATION_STORE_KEY,
    VALIDATION_STORES_VERSION,
)
from custom_components.powerguesser.engine.validation.drift import LiveDrift

from tests.conftest import (
    async_fill_synth_buffer,
    async_open_options_step,
    async_seed_shadow_set,
    async_set_power,
    async_setup_via_flow,
    make_switched_candidate,
    seed_candidates,
)
from tests.fixtures.learning.synth import SynthSpec
from tests.ha.test_device_detection import (
    _device_uuid,
    _drive_ticks,
    _setup_confirmed,
)
from tests.ha.test_device_validation_sensor import TWELVE_KEYS, _extra

# The kettle candidate's rise box is 1800/2000/2200 (make_switched_candidate
# builds lo=0.9*rise_w, p50=rise_w, hi=1.1*rise_w around rise_w=2000).
KETTLE_RISE_W = 2000.0
EDGE_RISE_W = 2180.0  # |2180 - 2000| / 200 = 0.9 -> match_score 0.1
DAY_S = 86400.0

# Plan 05-06 (D-23): the class-holdout drift tests.
KETTLE_CLASS = "switched_1.5_3kw_1_level"
WINDOW_A = "2026-11-10"
WINDOW_B = "2026-11-17"
WINDOW_C = "2026-11-24"
WINDOW_D = "2026-12-01"


async def _rise_on(hass: HomeAssistant, freezer: Any, watts: float) -> None:
    """Turn a load on with an exact rise and let the detector attribute it.

    Three grid steps (90 s ON) still verify against the candidate's 60-300 s
    duration box, and the power is re-reported at every grid step: a real
    source publishes at its sample interval, and the hub consumes re-reports
    (EVENT_STATE_REPORTED) too, so the interval profiler keeps a 30 s median
    instead of drifting to the 90 s episode cadence and disabling STEP
    (D-18 gate) mid-test.
    """
    async_set_power(hass, "sensor.grid_power", 500 + watts)
    await hass.async_block_till_done()
    for _ in range(2):
        await _drive_ticks(hass, freezer, 1)
        async_set_power(hass, "sensor.grid_power", 500 + watts)
        await hass.async_block_till_done()
    await _drive_ticks(hass, freezer, 1)


async def _rise_off(hass: HomeAssistant, freezer: Any) -> None:
    """Turn the load off and let the episode close (re-reporting per grid
    step, same profiler rationale as `_rise_on`)."""
    async_set_power(hass, "sensor.grid_power", 500)
    await hass.async_block_till_done()
    for _ in range(2):
        await _drive_ticks(hass, freezer, 1)
        async_set_power(hass, "sensor.grid_power", 500)
        await hass.async_block_till_done()
    await _drive_ticks(hass, freezer, 1)


def _seeded_tracker(now: float, *, score: float = 0.1, baseline_n: int = 10) -> LiveDrift:
    """A tracker with a fresh baseline plus exactly one over evaluation.

    The next over evaluation (a second consecutive one) therefore sets drift.
    """
    tracker = LiveDrift()
    for i in range(baseline_n):
        tracker.record(now - 3600.0 + i, score, 0)
    verdict = tracker.evaluate()
    assert verdict.drift is False  # one over evaluation: counter at 1
    return tracker


def _seeded_one_over_tracker(now: float, *, baseline_score: float = 1.0) -> LiveDrift:
    """A tracker with a baseline (20 days old) and exactly ONE over evaluation
    an hour ago: the first 7 evaluations there are insufficient (the aged
    baseline leaves the window short of 8), the 8th is the first over
    evaluation - so the NEXT over evaluation sets drift (the restart path,
    D-15 hysteresis)."""
    tracker = LiveDrift()
    for i in range(10):
        tracker.record(now - 20 * DAY_S + i, baseline_score, 0)
    verdict = None
    for i in range(8):
        tracker.record(now - 3600.0 + i, 0.1, 0)
        verdict = tracker.evaluate()
    assert verdict is not None
    assert verdict.drift is False
    assert verdict.insufficient is False
    return tracker


def _drifted_tracker(now: float, *, baseline_score: float = 1.0) -> LiveDrift:
    """A tracker with drift already active: baseline 20 days old, 9 bad
    episodes an hour ago; the 9th bad evaluation is the second consecutive
    over evaluation, so drift is set and the bad episodes still sit in the
    14-day window."""
    tracker = LiveDrift()
    for i in range(10):
        tracker.record(now - 20 * DAY_S + i, baseline_score, 0)
    verdict = None
    for i in range(9):
        tracker.record(now - 3600.0 + i, 0.1, 0)
        verdict = tracker.evaluate()
    assert verdict is not None and verdict.drift is True
    return tracker


def _validation_eid(hass: HomeAssistant, entry: ConfigEntry, dev_uuid: str) -> str:
    """Return the diagnostic validation entity id for a confirmed device."""
    eid = er.async_get(hass).async_get_entity_id(
        "sensor", DOMAIN, f"{entry.entry_id}_{dev_uuid}_validation"
    )
    assert eid is not None
    return eid


async def test_low_match_scores_set_drift_on_both_sensors_and_review(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
) -> None:
    """A confirmed kettle whose live rises keep landing at the box edge gets
    drift true with its reason and start time on both sensors and in Review,
    while the device itself stays exactly as confirmed (D-15, D-16)."""
    entry, entities = await _setup_confirmed(
        hass,
        energy_manager,
        std_prefs,
        hass_storage,
        freezer,
        [make_switched_candidate("sig-kettle", "Kettle", rise_w=KETTLE_RISE_W)],
    )
    power_eid, energy_eid = entities["Kettle"]
    dev_uuid = _device_uuid(entry, "Kettle")
    devices = entry.runtime_data.devices
    ent_reg = er.async_get(hass)
    registry_before = set(ent_reg.entities)

    rec = devices.record(dev_uuid)
    assert rec is not None
    versions_before = [dict(v) for v in rec["versions"]]
    active_v_before = rec["active_v"]
    name_before = rec["name"]
    issues_before = set(ir.async_get(hass).issues)

    # Both sensors read drift false / null by default (twelve-key order).
    for eid in (power_eid, energy_eid):
        attrs = hass.states.get(eid).attributes
        assert attrs["drift"] is False
        assert attrs["drift_reason"] is None
        assert attrs["drift_since"] is None

    # Seed the device's tracker: 10 baseline episodes of score 0.1 (the
    # baseline describes the CURRENT parameters) plus one over evaluation, so
    # one more over evaluation sets drift after the D-15 hysteresis.
    tracker = _seeded_tracker(dt_util.utcnow().timestamp(), score=0.1)
    devices._drift[dev_uuid] = tracker

    # One more kettle episode whose rise is 2180 W: match_score 0.1, a second
    # consecutive over evaluation -> drift true, reason match_score_low.
    await _rise_on(hass, freezer, EDGE_RISE_W)
    await _rise_off(hass, freezer)

    stored = devices.record(dev_uuid)["validation"]["drift"]
    assert stored["drift"] is True
    assert stored["reason"] == "match_score_low"
    assert isinstance(stored["since"], str)
    since_dt = dt_util.parse_datetime(stored["since"])
    assert since_dt is not None

    # Both device sensors show the flags with an ISO drift_since.
    for eid in (power_eid, energy_eid):
        attrs = hass.states.get(eid).attributes
        assert attrs["drift"] is True
        assert attrs["drift_reason"] == "match_score_low"
        assert attrs["drift_since"] == stored["since"]
        assert dt_util.parse_datetime(attrs["drift_since"]) == since_dt

    # The Review list labels the device with the D-16 drift note.
    res = await async_open_options_step(hass, entry, "review")
    assert res["type"] == "form"
    opts = res["data_schema"].schema["candidate"].config["options"]
    assert len(opts) == 1
    assert opts[0]["label"].endswith(" — drift detected")

    # Drift only sets flags: versions, active_v, uuid, name and entity ids
    # are unchanged, and no Repairs issue was created.
    rec = devices.record(dev_uuid)
    assert rec["versions"] == versions_before
    assert rec["active_v"] == active_v_before
    assert rec["name"] == name_before
    assert set(ent_reg.entities) == registry_before
    assert set(ir.async_get(hass).issues) == issues_before


async def test_drift_clears_after_one_good_evaluation(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
) -> None:
    """One evaluation with the window median well clear of both rules clears
    drift on both sensors and in Review (D-15 clear-after-1)."""
    entry, entities = await _setup_confirmed(
        hass,
        energy_manager,
        std_prefs,
        hass_storage,
        freezer,
        [make_switched_candidate("sig-kettle", "Kettle", rise_w=KETTLE_RISE_W)],
    )
    power_eid, energy_eid = entities["Kettle"]
    dev_uuid = _device_uuid(entry, "Kettle")
    devices = entry.runtime_data.devices

    # Drift active on match_score_low; the 9 bad episodes stay inside the
    # 14-day window so recovery evaluations are never insufficient.
    devices._drift[dev_uuid] = _drifted_tracker(dt_util.utcnow().timestamp())
    devices.record(dev_uuid)["validation"]["drift"] = {
        "drift": True,
        "reason": "match_score_low",
        "since": dt_util.utcnow().isoformat(),
    }

    # Eight good episodes: the 9 bad ones still dominate the window median,
    # drift stays set.
    for _ in range(8):
        await _rise_on(hass, freezer, KETTLE_RISE_W)
        await _rise_off(hass, freezer)
    assert hass.states.get(power_eid).attributes["drift"] is True
    assert devices.record(dev_uuid)["validation"]["drift"]["drift"] is True

    # Two more good episodes lift the window median (18 entries -> 0.55,
    # still over; 19 entries -> 1.0), clear of both the absolute and the
    # baseline-relative bounds: drift clears everywhere.
    for _ in range(2):
        await _rise_on(hass, freezer, KETTLE_RISE_W)
        await _rise_off(hass, freezer)
    for eid in (power_eid, energy_eid):
        attrs = hass.states.get(eid).attributes
        assert attrs["drift"] is False
        assert attrs["drift_reason"] is None
        assert attrs["drift_since"] is None
    assert devices.record(dev_uuid)["validation"]["drift"]["drift"] is False

    res = await async_open_options_step(hass, entry, "review")
    opts = res["data_schema"].schema["candidate"].config["options"]
    assert opts[0]["label"].endswith(" — confirmed device")
    assert "drift detected" not in opts[0]["label"]


async def test_insufficient_evidence_visible(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
) -> None:
    """A device with fewer than 10 episodes shows drift_evidence insufficient
    on its diagnostic sensor, sufficient once a baseline exists (D-15)."""
    entry, entities = await _setup_confirmed(
        hass,
        energy_manager,
        std_prefs,
        hass_storage,
        freezer,
        [make_switched_candidate("sig-kettle", "Kettle", rise_w=KETTLE_RISE_W)],
    )
    power_eid, _energy_eid = entities["Kettle"]
    dev_uuid = _device_uuid(entry, "Kettle")
    devices = entry.runtime_data.devices
    validation_eid = _validation_eid(hass, entry, dev_uuid)

    now = dt_util.utcnow().timestamp()
    tracker = LiveDrift()
    for i in range(5):
        tracker.record(now - 3600.0 + i, 1.0, 0)
    devices._drift[dev_uuid] = tracker

    attrs = hass.states.get(validation_eid).attributes
    assert attrs["drift_evidence"] == "insufficient"
    assert hass.states.get(power_eid).attributes["drift"] is False

    # Five more good episodes complete the baseline: the window holds 10
    # episodes -> sufficient, and the good scores never set drift.
    for _ in range(5):
        await _rise_on(hass, freezer, KETTLE_RISE_W)
        await _rise_off(hass, freezer)
    attrs = hass.states.get(validation_eid).attributes
    assert attrs["drift_evidence"] == "sufficient"
    assert hass.states.get(power_eid).attributes["drift"] is False


async def test_drift_state_survives_reload(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
) -> None:
    """A reload restores evidence, baseline and the hysteresis counter: the
    next over evaluation after a restart that had one over evaluation sets
    drift (D-15)."""
    entry, entities = await _setup_confirmed(
        hass,
        energy_manager,
        std_prefs,
        hass_storage,
        freezer,
        [make_switched_candidate("sig-kettle", "Kettle", rise_w=KETTLE_RISE_W)],
    )
    power_eid, _energy_eid = entities["Kettle"]
    dev_uuid = _device_uuid(entry, "Kettle")
    devices = entry.runtime_data.devices

    devices._drift[dev_uuid] = _seeded_one_over_tracker(dt_util.utcnow().timestamp())
    await devices._async_save_hot()
    await hass.config_entries.async_reload(entry.entry_id)
    await hass.async_block_till_done()

    devices2 = entry.runtime_data.devices
    tracker = devices2._drift.get(dev_uuid)
    assert tracker is not None
    assert tracker.evidence_state() == "sufficient"
    assert devices2.record(dev_uuid)["validation"].get("drift") is None
    assert hass.states.get(power_eid).attributes["drift"] is False

    # The restored tracker still sits at one over evaluation: this one over
    # evaluation (second consecutive) sets drift without a fresh pair.
    await _rise_on(hass, freezer, EDGE_RISE_W)
    await _rise_off(hass, freezer)
    assert hass.states.get(power_eid).attributes["drift"] is True
    assert (
        devices2.record(dev_uuid)["validation"]["drift"]["reason"] == "match_score_low"
    )


async def test_activation_resets_drift_evidence(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
) -> None:
    """Accept and rollback share async_activate_version (D-25); activating a
    version resets that device's evidence, baseline and drift flags while
    another device's drift state is untouched (D-22)."""
    entry, entities = await _setup_confirmed(
        hass,
        energy_manager,
        std_prefs,
        hass_storage,
        freezer,
        [
            make_switched_candidate("sig-kettle", "Kettle", rise_w=KETTLE_RISE_W),
            make_switched_candidate("sig-oven", "Oven", rise_w=1800.0),
        ],
    )
    kettle_power, _kettle_energy = entities["Kettle"]
    oven_power, _oven_energy = entities["Oven"]
    devices = entry.runtime_data.devices
    kettle_uuid = _device_uuid(entry, "Kettle")
    oven_uuid = _device_uuid(entry, "Oven")

    # The kettle drifts through the real path; the oven carries an injected
    # drifted tracker with flags.
    devices._drift[kettle_uuid] = _seeded_one_over_tracker(
        dt_util.utcnow().timestamp(), baseline_score=0.1
    )
    await _rise_on(hass, freezer, EDGE_RISE_W)
    await _rise_off(hass, freezer)
    kettle_flags = devices.record(kettle_uuid)["validation"]["drift"]
    assert kettle_flags["drift"] is True

    oven_tracker = _drifted_tracker(dt_util.utcnow().timestamp(), baseline_score=0.1)
    devices._drift[oven_uuid] = oven_tracker
    devices.record(oven_uuid)["validation"]["drift"] = {
        "drift": True,
        "reason": "match_score_low",
        "since": dt_util.utcnow().isoformat(),
    }
    oven_tracker_dict = oven_tracker.to_dict()

    # Rollback/accept path: activating a stored version resets the device.
    await devices.async_activate_version(kettle_uuid, 1)
    await hass.async_block_till_done()

    kettle_tracker = devices._drift[kettle_uuid]
    assert kettle_tracker.to_dict()["evidence"] == []
    assert kettle_tracker.to_dict()["baseline"] is None
    assert "drift" not in devices.record(kettle_uuid)["validation"]
    assert hass.states.get(kettle_power).attributes["drift"] is False

    # The other device's evidence and flags are untouched.
    assert devices._drift[oven_uuid].to_dict() == oven_tracker_dict
    assert devices.record(oven_uuid)["validation"]["drift"]["drift"] is True
    assert hass.states.get(oven_power).attributes["drift"] is True


async def test_detector_logs_drained_in_live_use(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
) -> None:
    """Both detector logs stay empty after every batch in live use
    (RESEARCH Pitfall 7): the verified log is drained and discarded, the
    closed log feeds the drift trackers."""
    entry, _entities = await _setup_confirmed(
        hass,
        energy_manager,
        std_prefs,
        hass_storage,
        freezer,
        [make_switched_candidate("sig-kettle", "Kettle", rise_w=KETTLE_RISE_W)],
    )
    devices = entry.runtime_data.devices
    for _ in range(3):
        await _rise_on(hass, freezer, KETTLE_RISE_W)
        await _rise_off(hass, freezer)
        detector = devices._detector
        assert detector is not None
        assert len(detector._closed_log) == 0
        assert detector._episode_log == {}


async def test_drift_never_creates_repairs_or_changes_device(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
) -> None:
    """Drift set or cleared never creates a Repairs issue and never changes
    versions, active_v, name, uuid or entity ids (D-16, T-05-15)."""
    entry, entities = await _setup_confirmed(
        hass,
        energy_manager,
        std_prefs,
        hass_storage,
        freezer,
        [make_switched_candidate("sig-kettle", "Kettle", rise_w=KETTLE_RISE_W)],
    )
    power_eid, _energy_eid = entities["Kettle"]
    dev_uuid = _device_uuid(entry, "Kettle")
    devices = entry.runtime_data.devices
    ent_reg = er.async_get(hass)

    rec = devices.record(dev_uuid)
    versions_before = [dict(v) for v in rec["versions"]]
    active_v_before = rec["active_v"]
    name_before = rec["name"]
    entities_before = set(ent_reg.entities)
    issues_before = set(ir.async_get(hass).issues)
    sub_before = [s.subentry_id for s in entry.get_subentries_of_type("device")]

    devices._drift[dev_uuid] = _seeded_one_over_tracker(
        dt_util.utcnow().timestamp(), baseline_score=0.1
    )
    await _rise_on(hass, freezer, EDGE_RISE_W)
    await _rise_off(hass, freezer)
    assert hass.states.get(power_eid).attributes["drift"] is True

    # Recovery clears it again (10 good episodes lift the 18-entry window
    # median from 0.55 to 1.0).
    devices._drift[dev_uuid] = _drifted_tracker(dt_util.utcnow().timestamp())
    devices.record(dev_uuid)["validation"]["drift"] = {
        "drift": True,
        "reason": "match_score_low",
        "since": dt_util.utcnow().isoformat(),
    }
    for _ in range(10):
        await _rise_on(hass, freezer, KETTLE_RISE_W)
        await _rise_off(hass, freezer)
    assert hass.states.get(power_eid).attributes["drift"] is False

    rec = devices.record(dev_uuid)
    assert rec["versions"] == versions_before
    assert rec["active_v"] == active_v_before
    assert rec["name"] == name_before
    assert set(ent_reg.entities) == entities_before
    assert set(ir.async_get(hass).issues) == issues_before
    assert [s.subentry_id for s in entry.get_subentries_of_type("device")] == sub_before


# --- Class-holdout drift on confirmed devices (Plan 05-06; D-23, D-16, D-26) ---


def _class_history_entry(end: str, err: float, run_id: str = "test-shadow") -> dict[str, Any]:
    return {"shadow_run_id": run_id, "holdout_end": end, "energy_error_pct": err}


def _over_counted_class_state(end: str, run_id: str = "test-shadow") -> dict[str, Any]:
    """A stored holdout_drift state whose hysteresis already counted `end` over."""
    return {
        "shadow_run_id": run_id,
        "drift": False,
        "reason": None,
        "since": None,
        "last_end": end,
        "hysteresis": {"active": False, "over_count": 1},
    }


def _seed_class_history(
    hass_storage: dict[str, Any],
    windows: list[tuple[str, float]],
    state_end: str,
) -> None:
    """Seed the validation Store with the kettle class's holdout history."""
    hass_storage[VALIDATION_STORE_KEY] = {
        "version": VALIDATION_STORES_VERSION,
        "minor_version": 1,
        "key": VALIDATION_STORE_KEY,
        "data": {
            "report": None,
            "plug_classes": {},
            "history": {
                f"class:{KETTLE_CLASS}": [
                    _class_history_entry(end, err) for end, err in windows
                ]
            },
            "holdout_drift": {
                f"class:{KETTLE_CLASS}": _over_counted_class_state(state_end)
            },
        },
    }


def _class_entry(
    median: float | None = 30.0,
    *,
    passes: bool = True,
    klass: str = KETTLE_CLASS,
) -> dict[str, Any]:
    """One classes entry as Plan 05-02's report builder writes it."""
    return {
        "class": klass,
        "peers": 2,
        "passing": 2,
        "median_error_pct": median,
        "passes": passes,
    }


def _date_ms(day: str) -> int:
    """Midday UTC of an ISO date: the local holdout_end stays that date in any tz."""
    from datetime import datetime, timezone

    year, month, dom = (int(part) for part in day.split("-"))
    return int(datetime(year, month, dom, 12, 0, tzinfo=timezone.utc).timestamp() * 1000)


def _class_stub(
    classes: list[dict[str, Any]],
    *,
    end_day: str = WINDOW_C,
    status: str = "ok",
    reason: str | None = None,
) -> Any:
    """A validate-stage stub reporting the given classes for the window."""

    async def _stub(self: Any, stage: str, job_path: Any) -> dict:
        assert stage == "validate"
        import json
        from pathlib import Path

        job = json.loads(Path(str(job_path)).read_text(encoding="utf-8"))
        stage_dir = Path(job["stage_dir"])
        report = {
            "status": status,
            "reason": reason,
            "train_cutoff_ms": job["shadow"]["train_cutoff_ms"],
            "holdout": {"start_ms": 0, "end_ms": _date_ms(end_day), "days": []},
            "devices": [],
            "classes": classes,
        }
        (stage_dir / "report.json").write_text(json.dumps(report), encoding="utf-8")
        return {"stats": {"wall_s": 0.0, "max_rss_kb": 0}}

    return _stub


async def _run_validation(hass: HomeAssistant) -> dict[str, Any]:
    res = await hass.services.async_call(
        DOMAIN, "run_validation", {}, blocking=True, return_response=True
    )
    await hass.async_block_till_done()
    return res


async def _setup_class_drift_env(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    candidates: list[dict[str, Any]],
    *,
    class_windows: list[tuple[str, float]] | None = None,
    class_state_end: str | None = None,
) -> Any:
    """Confirmed device(s) plus a stored shadow set and seeded class history."""
    if class_windows is not None and class_state_end is not None:
        _seed_class_history(hass_storage, class_windows, class_state_end)
    entry, entities = await _setup_confirmed(
        hass, energy_manager, std_prefs, hass_storage, None, candidates
    )
    await async_fill_synth_buffer(hass, entry, SynthSpec(days=35))
    await async_seed_shadow_set(hass, entry, [], [])
    return entry, entities


def _device_drift_attrs(devices: Any, entry: ConfigEntry, label: str) -> dict[str, Any]:
    """The drift view of one confirmed device, whatever its sensor availability."""
    dev_uuid = _device_uuid(entry, label)
    return devices.attributes(dev_uuid)


async def test_class_holdout_drift_flags_devices_showing_the_class(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    inprocess_worker: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """D-23: a confirmed kettle whose class passes in the latest ok report and
    whose class median rose on 2 consecutive windows of the same shadow set
    shows drift true, reason validation_error_rising and the class's
    drift_since on both device sensors and in Review; the displayed figure
    and basis are the class's own (validation_error_pct stays the class
    median, basis class_holdout, unvalidated false)."""
    from custom_components.powerguesser.learning import LearningSupervisor

    entry, entities = await _setup_class_drift_env(
        hass,
        energy_manager,
        std_prefs,
        hass_storage,
        [make_switched_candidate("sig-kettle", "Kettle", rise_w=KETTLE_RISE_W)],
        class_windows=[(WINDOW_A, 8.0), (WINDOW_B, 20.0)],
        class_state_end=WINDOW_B,
    )
    power_eid, energy_eid = entities["Kettle"]
    dev_uuid = _device_uuid(entry, "Kettle")

    monkeypatch.setattr(
        LearningSupervisor,
        "_async_run_stage",
        _class_stub([_class_entry(30.0)]),
    )
    res = await _run_validation(hass)
    klass = next(c for c in res["classes"] if c["class"] == KETTLE_CLASS)
    assert klass["drift"] is True
    assert klass["drift_reason"] == "validation_error_rising"
    assert klass["drift_since"] == WINDOW_C

    # The twelve-key D-17 order is unchanged; the power sensor keeps its
    # Phase 4 live keys after the static set (05-04 pin).
    power_attrs = _extra(hass, power_eid)
    assert list(power_attrs.keys())[: len(TWELVE_KEYS)] == TWELVE_KEYS
    assert list(power_attrs.keys())[len(TWELVE_KEYS) :] == [
        "match_score",
        "unattributed_edges",
        "update_available",
        "overlap_refused",
    ]
    energy_attrs = _extra(hass, energy_eid)
    assert list(energy_attrs.keys()) == TWELVE_KEYS
    for attrs in (power_attrs, energy_attrs):
        assert attrs["drift"] is True
        assert attrs["drift_reason"] == "validation_error_rising"
        assert attrs["drift_since"] == WINDOW_C
        assert attrs["validation_basis"] == "class_holdout"
        assert attrs["validation_error_pct"] == 30.0
        assert attrs["unvalidated"] is False

    # The Review list labels the device with the D-16 drift note.
    res_review = await async_open_options_step(hass, entry, "review")
    opts = res_review["data_schema"].schema["candidate"].config["options"]
    assert len(opts) == 1
    assert opts[0]["label"].endswith(" — drift detected")

    # The record itself carries no class-holdout drift: the flags are derived
    # from the validation Store's class entry at read time (D-26 (3)).
    rec = entry.runtime_data.devices.record(dev_uuid)
    assert "drift" not in rec["validation"]
    assert [v["v"] for v in rec["versions"]] == [1]


async def test_class_holdout_drift_only_where_the_class_figure_is_shown(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    inprocess_worker: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """D-23/D-16: class-holdout drift reaches only a device that shows that
    class's figure: a device in another class, a retired device of the
    drifting class, a device whose class entry reads drift true but passes
    false, and every device after a refused latest report all read drift
    false from the class path."""
    from custom_components.powerguesser.learning import LearningSupervisor

    entry, entities = await _setup_class_drift_env(
        hass,
        energy_manager,
        std_prefs,
        hass_storage,
        [
            make_switched_candidate("sig-kettle-a", "Kettle A", rise_w=KETTLE_RISE_W),
            make_switched_candidate("sig-kettle-b", "Kettle B", rise_w=KETTLE_RISE_W),
            make_switched_candidate("sig-other", "Other", rise_w=4000.0),
        ],
        class_windows=[(WINDOW_A, 8.0), (WINDOW_B, 20.0)],
        class_state_end=WINDOW_B,
    )
    devices = entry.runtime_data.devices
    uuid_a = _device_uuid(entry, "Kettle A")
    uuid_b = _device_uuid(entry, "Kettle B")

    # Retire Kettle B into Kettle A: its signature becomes an alias and it
    # keeps its class, but it is no longer confirmed (D-12).
    await devices.async_retire(uuid_b, into=uuid_a)

    monkeypatch.setattr(
        LearningSupervisor,
        "_async_run_stage",
        _class_stub([_class_entry(30.0)]),
    )
    await _run_validation(hass)

    # Only the confirmed device showing the drifting passing class is flagged.
    assert _device_drift_attrs(devices, entry, "Kettle A")["drift"] is True
    assert _device_drift_attrs(devices, entry, "Kettle B")["drift"] is False
    assert _device_drift_attrs(devices, entry, "Other")["drift"] is False
    assert _device_drift_attrs(devices, entry, "Kettle B")["drift_reason"] is None
    assert _device_drift_attrs(devices, entry, "Other")["drift_reason"] is None

    # The class entry now fails D-08 (passes false) while still drifting:
    # the figure is not shown, so no device inherits the flag.
    monkeypatch.setattr(
        LearningSupervisor,
        "_async_run_stage",
        _class_stub([_class_entry(30.0, passes=False)]),
    )
    res = await _run_validation(hass)
    klass = next(c for c in res["classes"] if c["class"] == KETTLE_CLASS)
    assert klass["drift"] is True  # same window re-measured: still set
    assert klass["passes"] is False
    assert _device_drift_attrs(devices, entry, "Kettle A")["drift"] is False
    assert _device_drift_attrs(devices, entry, "Kettle A")["unvalidated"] is True

    # A refused latest report clears every class result: nothing is flagged.
    monkeypatch.setattr(
        LearningSupervisor,
        "_async_run_stage",
        _class_stub([_class_entry(30.0)], status="refused", reason="insufficient_data"),
    )
    await _run_validation(hass)
    assert _device_drift_attrs(devices, entry, "Kettle A")["drift"] is False
    assert _device_drift_attrs(devices, entry, "Kettle A")["drift_reason"] is None
    assert _device_drift_attrs(devices, entry, "Kettle A")["drift_since"] is None


async def test_live_and_class_drift_combine(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    inprocess_worker: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """D-26 (2): with live and class-holdout drift both set, the class reason
    (validation_error_rising, measured against real plugs) and the class
    since win; once the class drift clears, the live reason and since show
    again; with neither, the device reads drift false everywhere."""
    from custom_components.powerguesser.learning import LearningSupervisor

    entry, entities = await _setup_class_drift_env(
        hass,
        energy_manager,
        std_prefs,
        hass_storage,
        [make_switched_candidate("sig-kettle", "Kettle", rise_w=KETTLE_RISE_W)],
        class_windows=[(WINDOW_A, 8.0), (WINDOW_B, 20.0)],
        class_state_end=WINDOW_B,
    )
    power_eid, _energy_eid = entities["Kettle"]
    dev_uuid = _device_uuid(entry, "Kettle")
    devices = entry.runtime_data.devices

    # Seed live drift on the record (a written match_score_low verdict).
    live_since = "2026-11-20T08:00:00+00:00"
    devices.record(dev_uuid)["validation"]["drift"] = {
        "drift": True,
        "reason": "match_score_low",
        "since": live_since,
    }

    monkeypatch.setattr(
        LearningSupervisor,
        "_async_run_stage",
        _class_stub([_class_entry(30.0)]),
    )
    await _run_validation(hass)
    attrs = hass.states.get(power_eid).attributes
    assert attrs["drift"] is True
    assert attrs["drift_reason"] == "validation_error_rising"
    assert attrs["drift_since"] == WINDOW_C

    # The class drift clears on a window 5 pp under the threshold (best 8.0):
    # the live reason and ISO-timestamp since show again (D-26 (4)).
    monkeypatch.setattr(
        LearningSupervisor,
        "_async_run_stage",
        _class_stub([_class_entry(7.0)], end_day=WINDOW_D),
    )
    await _run_validation(hass)
    attrs = hass.states.get(power_eid).attributes
    assert attrs["drift"] is True
    assert attrs["drift_reason"] == "match_score_low"
    assert attrs["drift_since"] == live_since

    # Neither: a version activation resets the live drift (D-22) and the
    # class entry reads drift false -> drift false, null reason and since.
    await devices.async_activate_version(dev_uuid, 1)
    attrs = hass.states.get(power_eid).attributes
    assert attrs["drift"] is False
    assert attrs["drift_reason"] is None
    assert attrs["drift_since"] is None
    assert "drift" not in devices.record(dev_uuid)["validation"]


async def test_activation_keeps_class_holdout_drift(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    inprocess_worker: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """D-22 boundary: Accept or Rollback resets the live drift only. A device
    whose re-derived class is still the drifting passing class keeps showing
    validation_error_rising; one whose class moved stops showing it."""
    from custom_components.powerguesser.learning import LearningSupervisor

    entry, entities = await _setup_class_drift_env(
        hass,
        energy_manager,
        std_prefs,
        hass_storage,
        [make_switched_candidate("sig-kettle", "Kettle", rise_w=KETTLE_RISE_W)],
        class_windows=[(WINDOW_A, 8.0), (WINDOW_B, 20.0)],
        class_state_end=WINDOW_B,
    )
    power_eid, _energy_eid = entities["Kettle"]
    dev_uuid = _device_uuid(entry, "Kettle")
    devices = entry.runtime_data.devices

    monkeypatch.setattr(
        LearningSupervisor,
        "_async_run_stage",
        _class_stub([_class_entry(30.0)]),
    )
    await _run_validation(hass)
    assert hass.states.get(power_eid).attributes["drift"] is True

    # Live drift also set; activation (rollback to v1) resets the live drift
    # only — the class-holdout drift belongs to the displayed class figure
    # and survives.
    devices.record(dev_uuid)["validation"]["drift"] = {
        "drift": True,
        "reason": "match_score_low",
        "since": "2026-11-20T08:00:00+00:00",
    }
    await devices.async_activate_version(dev_uuid, 1)
    attrs = hass.states.get(power_eid).attributes
    assert attrs["drift"] is True
    assert attrs["drift_reason"] == "validation_error_rising"
    assert attrs["drift_since"] == WINDOW_C
    assert "drift" not in devices.record(dev_uuid)["validation"]

    # Accept a proposal whose artefact moved to another band: the device's
    # re-derived class no longer matches the drifting class entry, so the
    # class-holdout drift stops showing (live drift was reset by the
    # activation above).
    other = make_switched_candidate("sig-other-band", "Other band", rise_w=4000.0)
    await devices.async_set_proposal(
        dev_uuid,
        other,
        run_id="run-other",
        proposed_at="2026-11-21T10:00:00Z",
        trained_at="2026-11-21T10:00:00Z",
    )
    v2 = await devices.async_accept_proposal(dev_uuid)
    assert v2 == 2
    attrs = hass.states.get(power_eid).attributes
    assert attrs["drift"] is False
    assert attrs["drift_reason"] is None
    assert attrs["drift_since"] is None
    assert attrs["validation_class"] == "switched_over_3kw_1_level"


async def test_unconfirm_clears_live_drift_flags(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
) -> None:
    """D-05 reconcile: un-confirm drops the cold D-16 flags with the tracker.

    A confirmed record whose subentry is gone returns to Review; the live
    drift evidence described the confirmed parameters, so the record's
    validation.drift flags leave the cold record too (a fresh post-re-confirm
    tracker only rewrites them when a verdict *changes*, so a stale flag
    would otherwise reappear on the re-confirmed device's sensors). The
    record keeps uuid, versions, name and energy; only the drift flags go.
    """
    candidate = make_switched_candidate("sig-alpha", "Alpha", rise_w=2000.0)
    artefact = {
        k: v
        for k, v in candidate.items()
        if k not in ("identity", "fingerprint", "first_label")
    }
    hass_storage[DEVICES_STORE_KEY] = {
        "version": 1,
        "minor_version": 1,
        "key": DEVICES_STORE_KEY,
        "data": {
            "devices": {
                "uuid-alpha": {
                    "name": "Alpha",
                    "signature_id": "sig-alpha",
                    "status": "confirmed",
                    "subentry_id": "sub-missing",
                    "versions": [
                        {
                            "v": 1,
                            "artefact": artefact,
                            "trained_at": "2026-11-16T12:00:00Z",
                            "scores": {},
                            "accepted_at": "2026-11-16T12:00:00Z",
                            "source": "confirm",
                        }
                    ],
                    "active_v": 1,
                    "validation": {
                        "drift": {
                            "drift": True,
                            "reason": "match_score_low",
                            "since": "2026-11-20T08:00:00+00:00",
                        }
                    },
                }
            }
        },
    }
    hass_storage[DEVICE_STATE_STORE_KEY] = {
        "version": 1,
        "minor_version": 1,
        "key": DEVICE_STATE_STORE_KEY,
        "data": {"energy_wh": {"uuid-alpha": 777.0}, "detector": None},
    }
    hass_storage[REVIEW_STORE_KEY] = {
        "version": LEARNING_STORES_VERSION,
        "minor_version": 1,
        "key": REVIEW_STORE_KEY,
        "data": {
            "decisions": {
                "sig-alpha": {
                    "status": "named",
                    "name": "Alpha",
                    "device_uuid": "uuid-alpha",
                    "subentry_id": "sub-missing",
                    "pre_confirm_status": "kept",
                }
            }
        },
    }
    seed_candidates(hass_storage, [candidate])
    energy_manager.data = std_prefs
    async_set_power(hass, "sensor.grid_power", 500)
    async_set_power(hass, "sensor.solar_power", 0)
    async_set_power(hass, "sensor.battery_power", 0)
    async_set_power(hass, "sensor.washer_power", 200)

    entry = await async_setup_via_flow(hass)
    await hass.async_block_till_done()

    devices = entry.runtime_data.devices
    rec = devices.record("uuid-alpha")
    # Un-confirm kept uuid/versions/name/energy (D-05) but cleared the drift
    # flags from the cold record: no phantom drift after a re-confirm.
    assert rec["status"] == "unconfirmed"
    assert rec["signature_id"] == "sig-alpha"
    assert rec["name"] == "Alpha"
    assert len(rec["versions"]) == 1
    assert devices.energy_wh("uuid-alpha") == pytest.approx(777.0, abs=1e-6)
    assert "drift" not in rec["validation"]
    assert devices._drift_view("uuid-alpha") == {
        "drift": False,
        "reason": None,
        "since": None,
    }


# --- Plan 06-09 (REV-04, D-14/D-38): drift revokes a plug_session validation ---


async def test_drift_revokes_a_session_validation_for_good(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
) -> None:
    """A validated-via-session device loses the status when live drift fires,
    keeps the session period and never regains it when drift clears (D-14, D-38)."""
    entry, entities = await _setup_confirmed(
        hass,
        energy_manager,
        std_prefs,
        hass_storage,
        freezer,
        [make_switched_candidate("sig-kettle", "Kettle", rise_w=KETTLE_RISE_W)],
    )
    power_eid, energy_eid = entities["Kettle"]
    dev_uuid = _device_uuid(entry, "Kettle")
    devices = entry.runtime_data.devices
    rec = devices.record(dev_uuid)
    assert rec is not None

    # A stored passing session result whose version is the active one.
    session = {
        "session_id": "sess-kettle",
        "artefact_v": rec["active_v"],
        "passes": True,
        "energy_error_pct": 4.0,
        "window_start": "2026-11-03T00:00:00+00:00",
        "window_end": "2026-11-18T00:00:00+00:00",
        "holdout_start": "2026-11-11T00:00:00+00:00",
        "holdout_end": "2026-11-18T00:00:00+00:00",
        "validated_at": dt_util.utcnow().isoformat(),
        "revoked": None,
    }
    rec["validation"]["session"] = dict(session)
    devices._republish()

    for eid in (power_eid, energy_eid):
        attrs = hass.states.get(eid).attributes
        assert attrs["status"] == "validated"
        assert attrs["validation_basis"] == "plug_session"
        assert attrs["unvalidated"] is False
        assert attrs["validation_error_pct"] == 4.0
        assert attrs["validation_session_start"] == session["window_start"]
        assert attrs["validation_session_end"] == session["window_end"]

    # One kettle episode at the box edge: a second consecutive over evaluation
    # sets drift (D-15).
    devices._drift[dev_uuid] = _seeded_tracker(dt_util.utcnow().timestamp(), score=0.1)
    await _rise_on(hass, freezer, EDGE_RISE_W)
    await _rise_off(hass, freezer)

    validation = devices.record(dev_uuid)["validation"]
    assert validation["drift"]["drift"] is True
    assert validation["session"]["revoked"]["reason"] == "drift"
    assert isinstance(validation["session"]["revoked"]["at"], str)
    for eid in (power_eid, energy_eid):
        attrs = hass.states.get(eid).attributes
        assert attrs["status"] == "provisional"
        assert attrs["unvalidated"] is True
        assert attrs["validation_basis"] != "plug_session"
        # The session period stays visible after the revocation (D-14).
        assert attrs["validation_session_start"] == session["window_start"]
        assert attrs["validation_session_end"] == session["window_end"]

    # Drift clearing does not restore the session validation: the revocation
    # stands and the device stays provisional.
    rec = devices.record(dev_uuid)
    assert rec is not None
    rec["validation"]["drift"] = {"drift": False, "reason": None, "since": None}
    devices._republish()
    for eid in (power_eid, energy_eid):
        attrs = hass.states.get(eid).attributes
        assert attrs["status"] == "provisional"
        assert attrs["unvalidated"] is True
        assert attrs["validation_session_start"] == session["window_start"]


async def test_drift_revocation_survives_a_later_validation_run(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
) -> None:
    """A re-judged device-target session never un-revokes a drift revocation.

    A still-open session is re-emitted by every validation run with the run time
    as its window_end, so the newer-entry gate alone would replace the revoked
    stored result with ``revoked: None`` and the device would read validated
    again as soon as live drift is false. The revocation is permanent for that
    session; a genuinely new session still replaces it (WR-01, D-14/D-38)."""
    entry, entities = await _setup_confirmed(
        hass,
        energy_manager,
        std_prefs,
        hass_storage,
        freezer,
        [make_switched_candidate("sig-kettle", "Kettle", rise_w=KETTLE_RISE_W)],
    )
    power_eid, _energy_eid = entities["Kettle"]
    dev_uuid = _device_uuid(entry, "Kettle")
    devices = entry.runtime_data.devices
    rec = devices.record(dev_uuid)
    assert rec is not None
    active_v = rec["active_v"]

    # The stored passing result as the drift guard leaves it (devices.py
    # _drain_step): revoked {"reason": "drift"} with the original window_end.
    window_start = "2026-11-03T00:00:00+00:00"
    window_end = "2026-11-18T00:00:00+00:00"
    rec["validation"]["session"] = {
        "session_id": "sess-kettle",
        "artefact_v": active_v,
        "passes": True,
        "energy_error_pct": 4.0,
        "window_start": window_start,
        "window_end": window_end,
        "holdout_start": "2026-11-11T00:00:00+00:00",
        "holdout_end": window_end,
        "validated_at": dt_util.utcnow().isoformat(),
        "revoked": {"reason": "drift", "at": dt_util.utcnow().isoformat()},
    }
    devices._republish()
    assert hass.states.get(power_eid).attributes["status"] == "provisional"

    def _session_entry(session_id: str, entry_end: str) -> dict[str, Any]:
        return {
            "session_id": session_id,
            "device_uuid": dev_uuid,
            "artefact_v": active_v,
            "passes": True,
            "energy_error_pct": 4.0,
            "window_start": window_start,
            "window_end": entry_end,
            "holdout_start": "2026-11-11T00:00:00+00:00",
            "holdout_end": entry_end,
        }

    # Live drift has cleared, the validated version is still active and the
    # result passes, so only the permanent revocation keeps it provisional.
    assert devices._drift_view(dev_uuid)["drift"] is False
    devices.apply_validation(
        {
            "status": "ok",
            "sessions": [_session_entry("sess-kettle", "2026-11-20T00:00:00+00:00")],
        }
    )
    stored = devices.record(dev_uuid)["validation"]["session"]
    assert stored["revoked"] is not None
    assert stored["revoked"]["reason"] == "drift"
    # The re-judged entry never replaced the stored one.
    assert stored["window_end"] == window_end
    attrs = hass.states.get(power_eid).attributes
    assert attrs["status"] == "provisional"
    assert attrs["unvalidated"] is True
    assert attrs["validation_basis"] != "plug_session"
    assert attrs["validation_session_end"] == window_end

    # A different (newer) session is a new result and still replaces the
    # revoked one (06-09-PLAN.md:202).
    devices.apply_validation(
        {
            "status": "ok",
            "sessions": [_session_entry("sess-kettle-2", "2026-11-27T00:00:00+00:00")],
        }
    )
    stored = devices.record(dev_uuid)["validation"]["session"]
    assert stored["session_id"] == "sess-kettle-2"
    assert stored["revoked"] is None
    attrs = hass.states.get(power_eid).attributes
    assert attrs["status"] == "validated"
    assert attrs["validation_basis"] == "plug_session"
    assert attrs["validation_session_end"] == "2026-11-27T00:00:00+00:00"


async def test_version_change_revokes_a_session_validation(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
) -> None:
    """Accepting or rolling back a version revokes the session validation (D-38)."""
    entry, entities = await _setup_confirmed(
        hass,
        energy_manager,
        std_prefs,
        hass_storage,
        freezer,
        [make_switched_candidate("sig-kettle", "Kettle", rise_w=KETTLE_RISE_W)],
    )
    power_eid, _energy_eid = entities["Kettle"]
    dev_uuid = _device_uuid(entry, "Kettle")
    devices = entry.runtime_data.devices
    rec = devices.record(dev_uuid)
    assert rec is not None

    # A second stored version to roll back to.
    big = make_switched_candidate("sig-kettle", "Kettle", rise_w=KETTLE_RISE_W)
    artefact2 = {
        k: v for k, v in big.items() if k not in ("identity", "fingerprint", "first_label")
    }
    rec["versions"].append(
        {
            "v": 2,
            "artefact": artefact2,
            "trained_at": "2026-11-18T12:00:00Z",
            "scores": dict(big["scores"]),
            "accepted_at": "2026-11-18T12:00:00Z",
            "source": "proposal",
        }
    )
    rec["validation"]["session"] = {
        "session_id": "sess-kettle",
        "artefact_v": 1,
        "passes": True,
        "energy_error_pct": 4.0,
        "window_start": "2026-11-03T00:00:00+00:00",
        "window_end": "2026-11-18T00:00:00+00:00",
        "holdout_start": "2026-11-11T00:00:00+00:00",
        "holdout_end": "2026-11-18T00:00:00+00:00",
        "validated_at": dt_util.utcnow().isoformat(),
        "revoked": None,
    }
    devices._republish()
    assert hass.states.get(power_eid).attributes["status"] == "validated"

    await devices.async_activate_version(dev_uuid, 2)

    validation = devices.record(dev_uuid)["validation"]
    assert validation["session"]["revoked"]["reason"] == "version_changed"
    attrs = hass.states.get(power_eid).attributes
    assert attrs["status"] == "provisional"
    assert attrs["unvalidated"] is True
    assert attrs["validation_session_start"] == "2026-11-03T00:00:00+00:00"

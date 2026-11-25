"""Tests for the background evidence refresh stage (Plan 07-04).

The whole module opts into the refresh hook: the autouse
``disable_evidence_refresh`` fixture in ``tests/conftest.py`` turns
``LearningSupervisor._async_maybe_evidence`` into a no-op unless a test carries
the ``evidence_enabled`` marker, so every other test in the net stays isolated.
"""

import asyncio
import copy
from datetime import timedelta
import json
from pathlib import Path
import threading
from typing import Any

import pytest

from homeassistant.components.energy.data import EnergyManager
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType
from homeassistant.helpers.storage import Store
import homeassistant.util.dt as dt_util

from custom_components.powerguesser.const import (
    CANDIDATES_STORE_KEY,
    LEARNING_STORE_KEY,
    LEARNING_STORES_VERSION,
)
from custom_components.powerguesser.learning import (
    LearningStageError,
    LearningSupervisor,
)
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
    proposal_responder,
    seed_candidates,
)
from tests.fixtures.learning.synth import SynthSpec

pytestmark = pytest.mark.evidence_enabled


def _set_std_powers(hass: HomeAssistant) -> None:
    async_set_power(hass, "sensor.grid_power", 500)
    async_set_power(hass, "sensor.solar_power", 1200)
    async_set_power(hass, "sensor.battery_power", -300)
    async_set_power(hass, "sensor.washer_power", 200)


def _evidence_map(
    sig_ids: list[str],
    *,
    observed_days: int = 20,
    source: str = "refresh",
    computed_at: str | None = None,
) -> dict[str, Any]:
    """Build a stage-shaped evidence.json mapping for the given ids."""
    ts = computed_at or dt_util.utcnow().isoformat()
    return {
        str(sig): {
            "v": 1,
            "computed_at": ts,
            "source": source,
            "window": {
                "start_ms": 0,
                "end_ms": 1,
                "observed_days": observed_days,
                "n_days": 30,
            },
            "n_runs": 5,
            "first_ms": 1000,
            "last_ms": 2000,
            "occurrences": [],
            "rhythm": None,
            "profile": {},
            "hints": [],
        }
        for sig in sig_ids
    }


def _write_evidence(job: dict[str, Any], sig_ids: list[str]) -> str:
    path = Path(job["stage_dir"]) / "evidence.json"
    path.write_text(json.dumps(_evidence_map(sig_ids)), encoding="utf-8")
    return str(path)


def _sig_ids_of(job_path: Any) -> list[str]:
    job = json.loads(Path(job_path).read_text(encoding="utf-8"))
    return [str(c["signature_id"]) for c in job.get("candidates", [])]


def _single_evidence(
    *,
    source: str = "refresh",
    observed_days: int = 20,
    computed_at: str | None = None,
) -> dict[str, Any]:
    """One evidence dict in the stage-shaped contract."""
    return _evidence_map(
        ["x"], observed_days=observed_days, source=source, computed_at=computed_at
    )["x"]


def _delegating_stage(
    hass: HomeAssistant,
    jobs: list[tuple[str, dict[str, Any]]] | None = None,
    fail_stages: tuple[str, ...] = (),
) -> Any:
    """A fake _async_run_stage that records jobs and runs the real worker stages.

    ``evidence`` in ``fail_stages`` raises LearningStageError instead, so one
    fake covers both the real evidence stage and the failing path.
    """
    from custom_components.powerguesser.engine.learning.worker import (
        run_stage_from_file,
        stage_outputs,
    )

    async def _stage(self: Any, stage: str, job_path: Any) -> dict[str, Any]:
        await asyncio.sleep(0)
        if jobs is not None:
            jobs.append(
                (stage, json.loads(Path(job_path).read_text(encoding="utf-8")))
            )
        if stage in fail_stages:
            raise LearningStageError(f"{stage} worker crash")
        outputs = await hass.async_add_executor_job(
            lambda: stage_outputs(stage, run_stage_from_file(stage, job_path))
        )
        return {**outputs, "stats": {"wall_s": 0.0, "max_rss_kb": 0}}

    return _stage


async def _seed_record(
    supervisor: LearningSupervisor,
    candidates: list[dict[str, Any]],
    *,
    validated: bool = True,
) -> dict[str, Any]:
    """Save a candidates record and make the supervisor hold it."""
    record = {
        "run_id": "run-evidence",
        "created": "2026-11-16T12:00:00Z",
        "window_end_ms": 0,
        "validated": validated,
        "candidates": candidates,
    }
    await supervisor._candidates_store.async_save(record)
    supervisor._candidates_record = record
    supervisor.candidates = candidates
    return record


async def _setup_env(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    *,
    days: int = 22,
    washer: bool = False,
) -> ConfigEntry:
    energy_manager.data = std_prefs
    _set_std_powers(hass)
    entry = await async_setup_via_flow(hass)
    await async_fill_synth_buffer(hass, entry, spec=SynthSpec(days=days, washer=washer))
    await async_seed_llm(hass, entry)
    await async_seed_consent(hass, host="llm.test", names=False)
    return entry


async def _seed_evidence_less(
    supervisor: LearningSupervisor, sigs: tuple[str, ...] = ("sig-kettle",)
) -> list[dict[str, Any]]:
    """Put evidence-less candidates into the supervisor and its Store."""
    candidates = [make_switched_candidate(s, s) for s in sigs]
    record = {
        "run_id": "run-evidence",
        "created": "2026-11-16T12:00:00Z",
        "window_end_ms": 0,
        "validated": True,
        "candidates": candidates,
    }
    await supervisor._candidates_store.async_save(record)
    supervisor._candidates_record = record
    supervisor.candidates = candidates
    return candidates


async def _drain(supervisor: LearningSupervisor, limit: int = 20) -> None:
    """Await the current run_task and any follow-up task it queued."""
    for _ in range(limit):
        task = supervisor.run_task
        if task is None:
            return
        await task
    raise AssertionError("run_task never cleared")


# --- upgrade path, running_job, restart, deferral -------------------------


async def test_upgrade_candidates_gain_evidence_without_llm(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    aioclient_mock: Any,
    inprocess_worker: Any,
) -> None:
    """A restart fills evidence for every visible stored candidate, no LLM call."""
    entry = await _setup_env(hass, energy_manager, std_prefs)
    supervisor = entry.runtime_data.learning

    seed_candidates(
        hass_storage, [make_switched_candidate("sig-kettle", "Kettle")]
    )

    new_supervisor = type(supervisor)(hass, entry, supervisor.hub)
    try:
        await new_supervisor.async_start()
        status_before = new_supervisor.coordinator.data["status"]
        assert new_supervisor.run_task is not None
        await new_supervisor.run_task

        visible = new_supervisor.visible_candidates()
        assert visible, "the stored candidate must still be visible"
        for cand in visible:
            evidence = cand.get("evidence")
            assert isinstance(evidence, dict)
            assert evidence["v"] == 1
            assert evidence["source"] == "refresh"

        stored = await new_supervisor._candidates_store.async_load()
        assert stored["candidates"][0]["evidence"]["source"] == "refresh"

        assert len(aioclient_mock.mock_calls) == 0
        data = new_supervisor.coordinator.data
        assert data["tokens_run"] == {"input": 0, "output": 0}
        assert int(data["month"]["input"]) == 0
        assert data["status"] == status_before
        assert data["running_job"] is None
    finally:
        await new_supervisor.async_stop()


async def test_running_job_is_evidence_while_stage_runs(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """running_job reads 'evidence' and status is unchanged; the job is the production extract job."""
    entry = await _setup_env(hass, energy_manager, std_prefs)
    supervisor = entry.runtime_data.learning
    await _seed_evidence_less(supervisor)
    status_before = supervisor.coordinator.data["status"]

    entered = asyncio.Event()
    release = asyncio.Event()
    captured: dict[str, Any] = {}

    async def _fake_stage(self: Any, stage: str, job_path: Any) -> dict[str, Any]:
        await asyncio.sleep(0)
        captured["stage"] = stage
        captured["job"] = json.loads(Path(job_path).read_text(encoding="utf-8"))
        entered.set()
        await release.wait()
        return {"result": _write_evidence(captured["job"], _sig_ids_of(job_path))}

    monkeypatch.setattr(LearningSupervisor, "_async_run_stage", _fake_stage)

    await supervisor._async_maybe_evidence()
    assert supervisor.run_task is not None
    try:
        await entered.wait()

        assert captured["stage"] == "evidence"
        job = captured["job"]
        assert job["holdout_days"] == 0
        assert job["holdout_calendar_days"] == 0
        assert job["masked_channels"] == []
        assert supervisor.coordinator.data["running_job"] == "evidence"
        assert supervisor.coordinator.data["status"] == status_before

        stored = await Store(
            hass, LEARNING_STORES_VERSION, LEARNING_STORE_KEY
        ).async_load()
        assert (stored or {}).get("status") != "running"
    finally:
        release.set()
        await supervisor.run_task

    stored = await Store(hass, LEARNING_STORES_VERSION, LEARNING_STORE_KEY).async_load()
    assert stored["status"] == status_before
    assert supervisor.coordinator.data["running_job"] is None


async def test_restart_during_evidence_stage_keeps_status_and_last_attempt(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A restart during the stage never reads failed/interrupted: status and last_attempt survive (D-18)."""
    seeded_started = "2026-11-24T05:00:00+00:00"
    seeded_ended = "2026-11-24T05:10:00+00:00"
    seeded_attempt = {
        "started": seeded_started,
        "ended": seeded_ended,
        "outcome": "success",
    }
    store = Store(hass, LEARNING_STORES_VERSION, LEARNING_STORE_KEY)
    await store.async_save(
        {
            "status": "idle",
            "blocked_reason": None,
            "last_run": None,
            "last_error": None,
            "last_attempt": dict(seeded_attempt),
            "tokens_run": {"input": 0, "output": 0},
            "month": {"key": "2026-11", "input": 0, "output": 0, "cost": None},
            "usable_days": {"value": 22.0, "computed_at": None},
        }
    )

    entry = await _setup_env(hass, energy_manager, std_prefs)
    supervisor = entry.runtime_data.learning
    assert supervisor.coordinator.data["last_attempt"] == seeded_attempt
    await _seed_evidence_less(supervisor)

    calls = {"n": 0}
    entered = asyncio.Event()
    release = asyncio.Event()

    async def _fake_stage(self: Any, stage: str, job_path: Any) -> dict[str, Any]:
        await asyncio.sleep(0)
        if stage == "evidence":
            calls["n"] += 1
            if calls["n"] == 1:
                entered.set()
                await release.wait()
        return {"result": _write_evidence(
            json.loads(Path(job_path).read_text(encoding="utf-8")),
            _sig_ids_of(job_path),
        )}

    monkeypatch.setattr(LearningSupervisor, "_async_run_stage", _fake_stage)

    await supervisor._async_maybe_evidence()
    assert supervisor.run_task is not None
    await entered.wait()

    new_supervisor = type(supervisor)(hass, entry, supervisor.hub)
    try:
        await new_supervisor.async_start()
        assert new_supervisor.coordinator.data["status"] == "idle"
        assert new_supervisor.coordinator.data["last_error"] is None
        assert new_supervisor.coordinator.data["last_attempt"] == seeded_attempt

        reloaded = await Store(
            hass, LEARNING_STORES_VERSION, LEARNING_STORE_KEY
        ).async_load()
        assert reloaded["status"] == "idle"
        assert reloaded["last_attempt"] == seeded_attempt
    finally:
        release.set()
        await supervisor.run_task
        if new_supervisor.run_task is not None:
            await new_supervisor.run_task
        await new_supervisor.async_stop()


async def test_hook_defers_when_slot_busy(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
) -> None:
    """The hook returns without a task and without raising while the slot is busy."""
    entry = await _setup_env(hass, energy_manager, std_prefs)
    supervisor = entry.runtime_data.learning
    await _seed_evidence_less(supervisor)

    supervisor._start_pending = True
    await supervisor._async_maybe_evidence()
    assert supervisor.run_task is None
    supervisor._start_pending = False

    release = asyncio.Event()

    async def _hold() -> None:
        await release.wait()

    task = entry.async_create_background_task(hass, _hold(), "test_hold")
    supervisor.run_task = task
    try:
        await supervisor._async_maybe_evidence()
        assert supervisor.run_task is task
    finally:
        release.set()
        await task
        supervisor.run_task = None


# --- failure cap and no-loop ----------------------------------------------


async def test_failure_cap_stops_retries(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Three failures cap the stage; a failed attempt never retries from its own finally (D-12b)."""
    entry = await _setup_env(hass, energy_manager, std_prefs)
    supervisor = entry.runtime_data.learning
    await _seed_evidence_less(supervisor)
    status_before = supervisor.coordinator.data["status"]
    last_error_before = supervisor.coordinator.data["last_error"]

    calls = {"evidence": 0}

    async def _failing_stage(self: Any, stage: str, job_path: Any) -> dict[str, Any]:
        await asyncio.sleep(0)
        calls["evidence"] += 1
        raise LearningStageError("evidence worker crash")

    monkeypatch.setattr(LearningSupervisor, "_async_run_stage", _failing_stage)

    for n in range(1, 4):
        await supervisor._async_maybe_evidence()
        assert supervisor.run_task is not None
        await supervisor.run_task
        assert calls["evidence"] == n
        assert supervisor.run_task is None

    assert supervisor.coordinator.data["evidence_failures"] == 3
    stored = await Store(hass, LEARNING_STORES_VERSION, LEARNING_STORE_KEY).async_load()
    assert stored["evidence_failures"] == 3
    assert supervisor.coordinator.data["status"] == status_before
    assert supervisor.coordinator.data["last_error"] == last_error_before

    # The cap stops the fourth call.
    await supervisor._async_maybe_evidence()
    assert supervisor.run_task is None
    assert calls["evidence"] == 3

    # A restart restores the counter.
    new_supervisor = type(supervisor)(hass, entry, supervisor.hub)
    try:
        await new_supervisor.async_start()
        assert new_supervisor.coordinator.data["evidence_failures"] == 3
    finally:
        await new_supervisor.async_stop()


async def test_failed_stage_waits_for_next_trigger(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed stage starts nothing from its own finally; the next trigger retries (D-12b)."""
    entry = await _setup_env(hass, energy_manager, std_prefs)
    supervisor = entry.runtime_data.learning
    await _seed_evidence_less(supervisor)

    calls = {"evidence": 0}

    async def _fake_stage(self: Any, stage: str, job_path: Any) -> dict[str, Any]:
        await asyncio.sleep(0)
        if stage == "evidence":
            calls["evidence"] += 1
            raise LearningStageError("evidence worker crash")
        return {}

    monkeypatch.setattr(LearningSupervisor, "_async_run_stage", _fake_stage)

    await supervisor._async_maybe_evidence()
    assert supervisor.run_task is not None
    await supervisor.run_task
    assert calls["evidence"] == 1
    assert supervisor.coordinator.data["evidence_failures"] == 1
    assert supervisor.run_task is None

    # A natural trigger (another stage's finally) launches the second attempt.
    job_path = tmp_path / "job_validate.json"
    job_path.write_text("{}", encoding="utf-8")
    await supervisor.async_run_exclusive_stage("validate", job_path)
    assert supervisor.run_task is not None
    await supervisor.run_task
    assert calls["evidence"] == 2
    assert supervisor.coordinator.data["evidence_failures"] == 2


async def test_no_refresh_loop_after_success(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A successful stage does not start a second evidence task from its own finally."""
    entry = await _setup_env(hass, energy_manager, std_prefs)
    supervisor = entry.runtime_data.learning
    await _seed_evidence_less(supervisor)

    calls = {"evidence": 0}

    async def _fake_stage(self: Any, stage: str, job_path: Any) -> dict[str, Any]:
        await asyncio.sleep(0)
        calls["evidence"] += 1
        return {"result": _write_evidence(
            json.loads(Path(job_path).read_text(encoding="utf-8")),
            _sig_ids_of(job_path),
        )}

    monkeypatch.setattr(LearningSupervisor, "_async_run_stage", _fake_stage)

    await supervisor._async_maybe_evidence()
    assert supervisor.run_task is not None
    await supervisor.run_task
    assert calls["evidence"] == 1
    assert supervisor.run_task is None


# --- rescore priority (D-10) ----------------------------------------------


async def test_due_rescore_wins_slot_over_pending_evidence(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A due rescore keeps priority over evidence at every hook site (D-10)."""
    from tests.ha.test_learning_rescore import _setup_rescore_env

    entry, _record = await _setup_rescore_env(hass, energy_manager, std_prefs)
    supervisor = entry.runtime_data.learning
    monkeypatch.setattr(
        supervisor, "_async_get_usable_days", lambda: asyncio.sleep(0, result=22.0)
    )

    order: list[str] = []

    async def _record_stage(self: Any, stage: str, job_path: Any) -> dict[str, Any]:
        await asyncio.sleep(0)
        order.append(stage)
        if stage == "evidence":
            return {"result": _write_evidence(
                json.loads(Path(job_path).read_text(encoding="utf-8")),
                _sig_ids_of(job_path),
            )}
        if stage == "rescore":
            return {
                "S": 0.1,
                "conflict_share": 0.0,
                "per_signature": {"sig-kettle": {"share": 0.1, "n": 10}},
                "days": 8,
            }
        return {}

    monkeypatch.setattr(LearningSupervisor, "_async_run_stage", _record_stage)

    job_path = tmp_path / "job_validate.json"
    job_path.write_text("{}", encoding="utf-8")

    # (1) The exclusive stage's finally must not take the slot while a rescore is due.
    await supervisor.async_run_exclusive_stage("validate", job_path)
    assert order == ["validate"]
    assert supervisor.run_task is None

    # (2) A direct hook call starts nothing either.
    await supervisor._async_maybe_evidence()
    assert order == ["validate"]
    assert supervisor.run_task is None

    # (3) The rescore takes the slot, and evidence follows from its finally.
    await supervisor._async_maybe_rescore()
    await _drain(supervisor)
    assert order == ["validate", "rescore", "evidence"]

    cand = supervisor.candidates[0]
    assert cand["evidence"]["source"] == "refresh"
    assert supervisor._candidates_record["validated"] is True


async def test_failed_rescore_does_not_starve_evidence(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed rescore's own finally lets evidence follow (D-10)."""
    from tests.ha.test_learning_rescore import _setup_rescore_env

    entry, _record = await _setup_rescore_env(hass, energy_manager, std_prefs)
    supervisor = entry.runtime_data.learning
    monkeypatch.setattr(
        supervisor, "_async_get_usable_days", lambda: asyncio.sleep(0, result=22.0)
    )

    order: list[str] = []

    async def _fake_stage(self: Any, stage: str, job_path: Any) -> dict[str, Any]:
        await asyncio.sleep(0)
        order.append(stage)
        if stage == "rescore":
            raise LearningStageError("rescore worker crash")
        if stage == "evidence":
            return {"result": _write_evidence(
                json.loads(Path(job_path).read_text(encoding="utf-8")),
                _sig_ids_of(job_path),
            )}
        return {}

    monkeypatch.setattr(LearningSupervisor, "_async_run_stage", _fake_stage)

    await supervisor._async_maybe_rescore()
    await _drain(supervisor)

    assert order == ["rescore", "evidence"]
    assert supervisor.coordinator.data["rescore_failures"] == 1
    assert supervisor.candidates[0]["evidence"]["source"] == "refresh"


# --- merge guard, unconfirmed rows, read-only Review, off-loop I/O --------


async def test_merge_skips_candidates_gone_mid_stage(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A candidate removed mid-stage is not resurrected; the other gets its evidence."""
    entry = await _setup_env(hass, energy_manager, std_prefs)
    supervisor = entry.runtime_data.learning
    await _seed_evidence_less(supervisor, sigs=("sig-kettle", "sig-oven"))

    removed = {"done": False}

    async def _fake_stage(self: Any, stage: str, job_path: Any) -> dict[str, Any]:
        await asyncio.sleep(0)
        # The stage drops a candidate before answering for both.
        if not removed["done"]:
            supervisor.candidates = [
                c
                for c in supervisor.candidates
                if c.get("signature_id") != "sig-oven"
            ]
            removed["done"] = True
        return {"result": _write_evidence(
            json.loads(Path(job_path).read_text(encoding="utf-8")),
            _sig_ids_of(job_path),
        )}

    monkeypatch.setattr(LearningSupervisor, "_async_run_stage", _fake_stage)

    await supervisor._async_maybe_evidence()
    await supervisor.run_task

    sigs = {c.get("signature_id"): c for c in supervisor.candidates}
    assert "sig-oven" not in sigs
    assert sigs["sig-kettle"]["evidence"]["source"] == "refresh"


async def test_merge_does_not_clobber_good_candidate_evidence(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
) -> None:
    """An n_runs-0 (unparsable-artefact) entry never overwrites stored good evidence (MR !17 note 4917)."""
    entry = await _setup_env(hass, energy_manager, std_prefs)
    supervisor = entry.runtime_data.learning
    cands = await _seed_evidence_less(supervisor)
    good = {"v": 1, "source": "refresh", "n_runs": 4}
    cands[0]["evidence"] = good

    await supervisor._merge_evidence(
        {"sig-kettle": {"v": 1, "source": "refresh", "n_runs": 0}}, ()
    )
    assert supervisor.candidates[0]["evidence"] is good
    assert supervisor.candidates[0]["evidence"]["n_runs"] == 4

    fresh = {"v": 1, "source": "refresh", "n_runs": 7}
    await supervisor._merge_evidence({"sig-kettle": fresh}, ())
    assert supervisor.candidates[0]["evidence"]["n_runs"] == 7

    # A stale stored entry must be replaced even by an empty one, or
    # _evidence_needed stays true and the success path re-runs the stage
    # without bound (MR !17 notes 4917/4922).
    supervisor.candidates[0]["evidence"] = {"v": 0, "source": "refresh", "n_runs": 4}
    await supervisor._merge_evidence(
        {"sig-kettle": {"v": 1, "source": "refresh", "n_runs": 0}}, ()
    )
    assert supervisor.candidates[0]["evidence"]["v"] == 1
    assert supervisor.candidates[0]["evidence"]["n_runs"] == 0

    supervisor.candidates[0]["evidence"] = {"v": 1, "source": "rescore", "n_runs": 4}
    await supervisor._merge_evidence(
        {"sig-kettle": {"v": 1, "source": "refresh", "n_runs": 0}}, ()
    )
    assert supervisor.candidates[0]["evidence"]["source"] == "refresh"


async def test_merge_does_not_clobber_good_device_evidence(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
) -> None:
    """The device path keeps stored good evidence when the stage returns n_runs-0 (MR !17 note 4917)."""
    entry = await _setup_env(hass, energy_manager, std_prefs)
    supervisor = entry.runtime_data.learning
    await _seed_evidence_less(supervisor)
    await async_confirm_candidate(hass, entry, "sig-kettle", "Kettle")
    sub = next(
        s for s in entry.get_subentries_of_type("device") if s.title == "Kettle"
    )
    device_uuid = str(sub.data["device_uuid"])
    good = {"v": 1, "source": "refresh", "n_runs": 3, "active_v": 9}
    supervisor._device_evidence[device_uuid] = good

    await supervisor._merge_evidence(
        {f"device:{device_uuid}": {"v": 1, "source": "refresh", "n_runs": 0}},
        ({"uuid": device_uuid, "active_v": 9},),
    )
    assert supervisor._device_evidence[device_uuid] is good
    assert supervisor._device_evidence[device_uuid]["n_runs"] == 3

    fresh = {"v": 1, "source": "refresh", "n_runs": 6}
    await supervisor._merge_evidence(
        {f"device:{device_uuid}": fresh},
        ({"uuid": device_uuid, "active_v": 9},),
    )
    assert supervisor._device_evidence[device_uuid]["n_runs"] == 6

    # A stale stored entry (active_v mismatch, or old EVIDENCE_VERSION) must be
    # replaced even by an empty one, or _device_evidence_targets stays true and
    # the success path re-runs the stage without bound (MR !17 notes 4917/4921).
    supervisor._device_evidence[device_uuid] = {
        "v": 1,
        "source": "refresh",
        "n_runs": 3,
        "active_v": 8,
    }
    await supervisor._merge_evidence(
        {f"device:{device_uuid}": {"v": 1, "source": "refresh", "n_runs": 0}},
        ({"uuid": device_uuid, "active_v": 9},),
    )
    assert supervisor._device_evidence[device_uuid]["active_v"] == 9
    assert supervisor._device_evidence[device_uuid]["n_runs"] == 0

    supervisor._device_evidence[device_uuid] = {
        "v": 0,
        "source": "refresh",
        "n_runs": 3,
        "active_v": 9,
    }
    await supervisor._merge_evidence(
        {f"device:{device_uuid}": {"v": 1, "source": "refresh", "n_runs": 0}},
        ({"uuid": device_uuid, "active_v": 9},),
    )
    assert supervisor._device_evidence[device_uuid]["v"] == 1


async def test_unconfirmed_device_rows_are_not_candidate_targets(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A deleted device's row is a device target, never a candidate target (Pitfall 3)."""
    entry = await _setup_env(hass, energy_manager, std_prefs)
    supervisor = entry.runtime_data.learning
    await _seed_evidence_less(supervisor)

    await async_confirm_candidate(hass, entry, "sig-kettle", "Kettle")
    sub = next(
        s for s in entry.get_subentries_of_type("device") if s.title == "Kettle"
    )
    device_uuid = sub.data["device_uuid"]
    hass.config_entries.async_remove_subentry(entry, sub.subentry_id)
    await hass.async_block_till_done()

    # No stored candidates: only the unconfirmed-device row is visible.
    await _seed_record(supervisor, [])
    visible = supervisor.visible_candidates()
    assert any(c.get("signature_id") == "sig-kettle" for c in visible)

    # It is never a candidate target (D-07a: it is a device target).
    assert supervisor._evidence_targets(include_aged=False) == []

    jobs: list[tuple[str, dict[str, Any]]] = []
    monkeypatch.setattr(
        LearningSupervisor, "_async_run_stage", _delegating_stage(hass, jobs=jobs)
    )

    await supervisor._async_maybe_evidence()
    assert supervisor.run_task is not None
    await supervisor.run_task

    evidence_jobs = [job for stage, job in jobs if stage == "evidence"]
    assert len(evidence_jobs) == 1
    sent = [str(c["signature_id"]) for c in evidence_jobs[0]["candidates"]]
    assert sent == [f"device:{device_uuid}"]
    assert supervisor.candidates == []
    assert supervisor.device_evidence(device_uuid) is not None


async def test_review_menus_are_read_only(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Opening Review and a candidate/device menu starts no job and writes nothing (D-12)."""
    entry = await _setup_env(hass, energy_manager, std_prefs)
    supervisor = entry.runtime_data.learning
    await _seed_evidence_less(supervisor, sigs=("sig-kettle", "sig-oven"))

    await async_confirm_candidate(hass, entry, "sig-oven", "Oven")
    sub = next(
        s for s in entry.get_subentries_of_type("device") if s.title == "Oven"
    )
    device_uuid = sub.data["device_uuid"]

    calls = {"evidence": 0}

    async def _fake_stage(self: Any, stage: str, job_path: Any) -> dict[str, Any]:
        await asyncio.sleep(0)
        calls["evidence"] += 1
        return {"result": _write_evidence(
            json.loads(Path(job_path).read_text(encoding="utf-8")),
            _sig_ids_of(job_path),
        )}

    monkeypatch.setattr(LearningSupervisor, "_async_run_stage", _fake_stage)

    await supervisor._async_maybe_evidence()
    assert supervisor.run_task is not None
    await supervisor.run_task
    assert calls["evidence"] == 1

    store_before = json.dumps(await supervisor._candidates_store.async_load(), sort_keys=True)

    res_review = await async_open_options_step(hass, entry, "review")
    assert res_review["type"] is FlowResultType.FORM
    res_cand = await hass.config_entries.options.async_configure(
        res_review["flow_id"], {"candidate": "sig-kettle"}
    )
    assert res_cand["step_id"] == "candidate"

    res_review2 = await async_open_options_step(hass, entry, "review")
    res_dev = await hass.config_entries.options.async_configure(
        res_review2["flow_id"], {"candidate": f"device_{device_uuid}"}
    )
    assert res_dev["step_id"] == "device"
    await hass.async_block_till_done()

    assert calls["evidence"] == 1
    assert supervisor.run_task is None
    store_after = json.dumps(await supervisor._candidates_store.async_load(), sort_keys=True)
    assert store_after == store_before


async def test_evidence_stage_dir_io_runs_off_loop(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The stage directory is created and removed in the executor, never on the event loop.

    The primary runtime proof of the no-blocking-I/O rule: only the ``tempfile``
    and ``shutil`` names inside learning.py's namespace are replaced, so no other
    module's I/O is recorded or altered.
    """
    loop_ident = threading.get_ident()

    import shutil as shutil_mod
    import tempfile as tempfile_mod

    import custom_components.powerguesser.learning as learning_mod

    mk_records: list[tuple[int, str, str]] = []
    rm_records: list[tuple[int, str]] = []

    class _Recorder:
        def __init__(self, real: Any) -> None:
            self._real = real

        def __getattr__(self, name: str) -> Any:
            real_attr = getattr(self._real, name)
            if name == "mkdtemp":
                def _mkdtemp(*args: Any, **kwargs: Any) -> str:
                    prefix = str(kwargs.get("prefix"))
                    result = real_attr(*args, **kwargs)
                    mk_records.append((threading.get_ident(), prefix, str(result)))
                    return result

                return _mkdtemp
            if name == "rmtree":
                def _rmtree(path: Any, *args: Any, **kwargs: Any) -> Any:
                    result = real_attr(path, *args, **kwargs)
                    rm_records.append((threading.get_ident(), str(path)))
                    return result

                return _rmtree
            return real_attr

    monkeypatch.setattr(learning_mod, "tempfile", _Recorder(tempfile_mod))
    monkeypatch.setattr(learning_mod, "shutil", _Recorder(shutil_mod))

    entry = await _setup_env(hass, energy_manager, std_prefs)
    supervisor = entry.runtime_data.learning
    await _seed_evidence_less(supervisor)

    calls = {"n": 0}

    async def _fake_stage(self: Any, stage: str, job_path: Any) -> dict[str, Any]:
        await asyncio.sleep(0)
        calls["n"] += 1
        if calls["n"] == 2:
            raise LearningStageError("evidence worker crash")
        return {"result": _write_evidence(
            json.loads(Path(job_path).read_text(encoding="utf-8")),
            _sig_ids_of(job_path),
        )}

    monkeypatch.setattr(LearningSupervisor, "_async_run_stage", _fake_stage)

    # One success, one failure.
    await supervisor._async_maybe_evidence()
    await supervisor.run_task
    supervisor.candidates[0].pop("evidence", None)
    await supervisor._async_maybe_evidence()
    await supervisor.run_task

    ev_mk = [r for r in mk_records if r[1] == "powerguesser-evidence-"]
    assert len(ev_mk) == 2, mk_records
    assert all(ident != loop_ident for ident, _prefix, _path in ev_mk)
    mk_dirs = [path for _ident, _prefix, path in ev_mk]

    assert sorted(path for _ident, path in rm_records) == sorted(mk_dirs)
    assert all(ident != loop_ident for ident, _path in rm_records)
    assert all(not Path(path).exists() for path in mk_dirs)


# --- Task 2: age refresh, rescore merge, device evidence (D-07a, D-11a) ----


async def test_periodic_refresh_recomputes_aged_evidence(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The 6-hourly refresh recomputes evidence older than 7 days (and missing/unparsable)."""
    entry = await _setup_env(hass, energy_manager, std_prefs)
    supervisor = entry.runtime_data.learning
    old_ts = (dt_util.utcnow() - timedelta(days=8)).isoformat()
    cand = make_switched_candidate("sig-kettle", "Kettle")
    cand["evidence"] = _single_evidence(computed_at=old_ts)
    await _seed_record(supervisor, [cand])

    calls = {"evidence": 0}

    async def _fake_stage(self: Any, stage: str, job_path: Any) -> dict[str, Any]:
        await asyncio.sleep(0)
        calls["evidence"] += 1
        return {"result": _write_evidence(
            json.loads(Path(job_path).read_text(encoding="utf-8")),
            _sig_ids_of(job_path),
        )}

    monkeypatch.setattr(LearningSupervisor, "_async_run_stage", _fake_stage)

    await supervisor._async_periodic_refresh(None)
    assert supervisor.run_task is not None
    await supervisor.run_task
    assert calls["evidence"] == 1
    fresh_ts = supervisor.candidates[0]["evidence"]["computed_at"]
    assert fresh_ts != old_ts

    # Six days old: not aged, no stage.
    six_ts = (dt_util.utcnow() - timedelta(days=6)).isoformat()
    supervisor.candidates[0]["evidence"] = _single_evidence(computed_at=six_ts)
    await supervisor._async_periodic_refresh(None)
    assert supervisor.run_task is None
    assert calls["evidence"] == 1

    # Unparsable computed_at counts as aged.
    supervisor.candidates[0]["evidence"] = _single_evidence(computed_at="not-a-date")
    await supervisor._async_periodic_refresh(None)
    assert supervisor.run_task is not None
    await supervisor.run_task
    assert calls["evidence"] == 2

    # A missing computed_at counts as aged.
    missing = _single_evidence()
    missing.pop("computed_at")
    supervisor.candidates[0]["evidence"] = missing
    await supervisor._async_periodic_refresh(None)
    assert supervisor.run_task is not None
    await supervisor.run_task
    assert calls["evidence"] == 3


async def test_age_is_not_checked_outside_periodic_refresh(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An 8-day-old evidence never starts a stage from async_start or a job finally."""
    entry = await _setup_env(hass, energy_manager, std_prefs)
    supervisor = entry.runtime_data.learning
    old_ts = (dt_util.utcnow() - timedelta(days=8)).isoformat()
    cand = make_switched_candidate("sig-kettle", "Kettle")
    cand["evidence"] = _single_evidence(computed_at=old_ts)
    await _seed_record(supervisor, [cand])

    stages: list[str] = []

    async def _fake_stage(self: Any, stage: str, job_path: Any) -> dict[str, Any]:
        await asyncio.sleep(0)
        stages.append(stage)
        return {}

    monkeypatch.setattr(LearningSupervisor, "_async_run_stage", _fake_stage)

    await supervisor._async_maybe_evidence()
    assert supervisor.run_task is None

    job_path = tmp_path / "job_validate.json"
    job_path.write_text("{}", encoding="utf-8")
    await supervisor.async_run_exclusive_stage("validate", job_path)
    assert supervisor.run_task is None

    new_supervisor = type(supervisor)(hass, entry, supervisor.hub)
    try:
        await new_supervisor.async_start()
        assert new_supervisor.run_task is None
    finally:
        await new_supervisor.async_stop()

    assert "evidence" not in stages


async def test_rescore_evidence_fills_missing_then_stage_refreshes(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The rescore job opts in and fills missing evidence; the stage then refreshes it (D-11)."""
    from tests.ha.test_learning_rescore import _setup_rescore_env

    entry, _record = await _setup_rescore_env(hass, energy_manager, std_prefs)
    supervisor = entry.runtime_data.learning
    monkeypatch.setattr(
        supervisor, "_async_get_usable_days", lambda: asyncio.sleep(0, result=22.0)
    )

    jobs: list[tuple[str, dict[str, Any]]] = []
    intermediate: dict[str, Any] = {}

    async def _fake_stage(self: Any, stage: str, job_path: Any) -> dict[str, Any]:
        await asyncio.sleep(0)
        job = json.loads(Path(job_path).read_text(encoding="utf-8"))
        jobs.append((stage, job))
        if stage == "rescore":
            return {
                "S": 0.1,
                "conflict_share": 0.0,
                "per_signature": {
                    "sig-kettle": {
                        "share": 0.1,
                        "n": 10,
                        "evidence": _single_evidence(
                            source="rescore", observed_days=8
                        ),
                    }
                },
                "days": 8,
            }
        if stage == "evidence":
            intermediate["evidence"] = copy.deepcopy(
                supervisor.candidates[0].get("evidence")
            )
            return {"result": _write_evidence(job, _sig_ids_of(job_path))}
        return {}

    monkeypatch.setattr(LearningSupervisor, "_async_run_stage", _fake_stage)

    await supervisor._async_maybe_rescore()
    await _drain(supervisor)

    rescore_job = next(job for stage, job in jobs if stage == "rescore")
    assert rescore_job["with_evidence"] is True
    assert isinstance(rescore_job.get("now_iso"), str) and rescore_job["now_iso"]

    assert intermediate["evidence"]["source"] == "rescore"
    assert supervisor.candidates[0]["evidence"]["source"] == "refresh"


async def test_rescore_evidence_never_replaces_wider_window(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stored 20-day window is kept when the rescore evidence covers 5 days (D-11)."""
    from tests.ha.test_learning_rescore import _setup_rescore_env

    entry, record = await _setup_rescore_env(hass, energy_manager, std_prefs)
    supervisor = entry.runtime_data.learning
    monkeypatch.setattr(
        supervisor, "_async_get_usable_days", lambda: asyncio.sleep(0, result=22.0)
    )
    supervisor.candidates[0]["evidence"] = _single_evidence(observed_days=20)

    async def _fake_stage(self: Any, stage: str, job_path: Any) -> dict[str, Any]:
        await asyncio.sleep(0)
        if stage == "rescore":
            return {
                "S": 0.1,
                "conflict_share": 0.0,
                "per_signature": {
                    "sig-kettle": {
                        "share": 0.1,
                        "n": 10,
                        "evidence": _single_evidence(
                            source="rescore", observed_days=5
                        ),
                    }
                },
                "days": 8,
            }
        if stage == "evidence":
            return {"result": _write_evidence(
                json.loads(Path(job_path).read_text(encoding="utf-8")),
                _sig_ids_of(job_path),
            )}
        return {}

    monkeypatch.setattr(LearningSupervisor, "_async_run_stage", _fake_stage)

    await supervisor._async_maybe_rescore()
    await _drain(supervisor)

    evidence = supervisor.candidates[0]["evidence"]
    assert evidence["window"]["observed_days"] == 20
    assert evidence["source"] == "refresh"
    assert record["validated"] is True


async def test_learning_run_resets_evidence_failures(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    aioclient_mock: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A learning run resets the evidence failure counter (D-12a)."""
    entry = await _setup_env(hass, energy_manager, std_prefs, washer=True)
    supervisor = entry.runtime_data.learning
    await _seed_evidence_less(supervisor)

    monkeypatch.setattr(
        LearningSupervisor, "_async_run_stage",
        _delegating_stage(hass, fail_stages=("evidence",)),
    )

    for _ in range(3):
        await supervisor._async_maybe_evidence()
        assert supervisor.run_task is not None
        await supervisor.run_task
    assert supervisor.coordinator.data["evidence_failures"] == 3

    mock_llm(aioclient_mock, "http://llm.test/v1", proposal_responder(1))
    await supervisor.async_request_run("service")
    assert supervisor.run_task is not None
    await supervisor.run_task

    assert supervisor.coordinator.data["evidence_failures"] == 0
    stored = await Store(hass, LEARNING_STORES_VERSION, LEARNING_STORE_KEY).async_load()
    assert stored["evidence_failures"] == 0

    # The cap no longer blocks: an evidence-less target starts a task again.
    for cand in supervisor.candidates:
        cand.pop("evidence", None)
    await supervisor._async_maybe_evidence()
    assert supervisor.run_task is not None
    await supervisor.run_task


async def test_device_evidence_for_confirmed_and_unconfirmed_devices(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    aioclient_mock: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Confirmed and deleted devices get evidence from one stage, outside every record (D-07a)."""
    entry = await _setup_env(hass, energy_manager, std_prefs)
    supervisor = entry.runtime_data.learning
    await _seed_evidence_less(supervisor, sigs=("sig-kettle", "sig-oven"))
    await async_confirm_candidate(hass, entry, "sig-kettle", "Kettle")
    await async_confirm_candidate(hass, entry, "sig-oven", "Oven")

    devices = entry.runtime_data.devices
    sub_oven = next(
        s for s in entry.get_subentries_of_type("device") if s.title == "Oven"
    )
    oven_uuid = sub_oven.data["device_uuid"]
    kettle_uuid = next(
        s for s in entry.get_subentries_of_type("device") if s.title == "Kettle"
    ).data["device_uuid"]
    hass.config_entries.async_remove_subentry(entry, sub_oven.subentry_id)
    await hass.async_block_till_done()

    # No stored candidates: only the two device records are evidence targets.
    await _seed_record(supervisor, [])

    jobs: list[tuple[str, dict[str, Any]]] = []
    monkeypatch.setattr(
        LearningSupervisor, "_async_run_stage", _delegating_stage(hass, jobs=jobs)
    )

    new_supervisor = type(supervisor)(hass, entry, supervisor.hub)
    try:
        await new_supervisor.async_start()
        assert new_supervisor.run_task is not None
        await new_supervisor.run_task
    finally:
        await new_supervisor.async_stop()

    evidence_jobs = [job for stage, job in jobs if stage == "evidence"]
    assert len(evidence_jobs) == 1
    sent = {str(c["signature_id"]) for c in evidence_jobs[0]["candidates"]}
    assert f"device:{kettle_uuid}" in sent
    assert f"device:{oven_uuid}" in sent

    for uuid in (kettle_uuid, oven_uuid):
        evidence = new_supervisor.device_evidence(uuid)
        assert isinstance(evidence, dict)
        assert evidence["v"] == 1
        assert evidence["source"] == "refresh"
        assert set(evidence.keys()) == {
            "v", "computed_at", "source", "window", "n_runs", "first_ms",
            "last_ms", "occurrences", "rhythm", "profile", "hints", "active_v",
        }
        assert evidence["active_v"] == devices.record(uuid)["active_v"]

    stored = await Store(hass, LEARNING_STORES_VERSION, LEARNING_STORE_KEY).async_load()
    assert set(stored["device_evidence"].keys()) == {kettle_uuid, oven_uuid}

    assert len(aioclient_mock.mock_calls) == 0
    for uuid in (kettle_uuid, oven_uuid):
        rec = devices.record(uuid)
        assert "evidence" not in rec
        for version in rec["versions"]:
            assert "evidence" not in version["artefact"]
        assert rec["pending_proposal"] is None
        assert all("evidence" not in a for a in rec["alias_artefacts"].values())

    # A second restart restores the device evidence and starts no stage.
    jobs2: list[tuple[str, dict[str, Any]]] = []
    monkeypatch.setattr(
        LearningSupervisor, "_async_run_stage", _delegating_stage(hass, jobs=jobs2)
    )
    third = type(supervisor)(hass, entry, supervisor.hub)
    try:
        await third.async_start()
        assert third.run_task is None
        assert third.device_evidence(kettle_uuid)["source"] == "refresh"
    finally:
        await third.async_stop()
    assert not [job for stage, job in jobs2 if stage == "evidence"]


async def test_device_evidence_targets_version_change_and_age(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A version change re-targets device evidence; age only the 6-hourly refresh."""
    entry = await _setup_env(hass, energy_manager, std_prefs)
    supervisor = entry.runtime_data.learning
    await _seed_evidence_less(supervisor)
    await async_confirm_candidate(hass, entry, "sig-kettle", "Kettle")
    devices = entry.runtime_data.devices
    uuid = next(
        s for s in entry.get_subentries_of_type("device") if s.title == "Kettle"
    ).data["device_uuid"]
    await _seed_record(supervisor, [])

    jobs: list[tuple[str, dict[str, Any]]] = []
    monkeypatch.setattr(
        LearningSupervisor, "_async_run_stage", _delegating_stage(hass, jobs=jobs)
    )

    # A stored entry for active_v 1 while the record is at 2 is a target.
    supervisor._device_evidence[uuid] = {**_single_evidence(), "active_v": 1}
    rec = devices.record(uuid)
    rec["versions"].append(
        {
            "v": 2,
            "artefact": dict(rec["versions"][0]["artefact"]),
            "trained_at": None,
            "scores": {},
            "accepted_at": None,
            "source": "test",
        }
    )
    rec["active_v"] = 2
    await supervisor._async_maybe_evidence()
    assert supervisor.run_task is not None
    await supervisor.run_task
    assert [stage for stage, _job in jobs].count("evidence") == 1
    assert supervisor._device_evidence[uuid]["active_v"] == 2

    # An orphaned entry is dropped by the next successful stage.
    supervisor._device_evidence["orphan-uuid"] = {**_single_evidence(), "active_v": 1}
    supervisor._device_evidence[uuid]["active_v"] = 1
    await supervisor._async_maybe_evidence()
    assert supervisor.run_task is not None
    await supervisor.run_task
    assert "orphan-uuid" not in supervisor._device_evidence

    active_v = devices.record(uuid)["active_v"]

    # Matching active_v and fresh: no target anywhere.
    supervisor._device_evidence[uuid] = {**_single_evidence(), "active_v": active_v}
    await supervisor._async_maybe_evidence()
    assert supervisor.run_task is None

    # Matching active_v but 8 days old: only the periodic refresh picks it up.
    aged_ts = (dt_util.utcnow() - timedelta(days=8)).isoformat()
    supervisor._device_evidence[uuid] = {
        **_single_evidence(computed_at=aged_ts),
        "active_v": active_v,
    }
    await supervisor._async_maybe_evidence()
    assert supervisor.run_task is None
    await supervisor._async_periodic_refresh(None)
    assert supervisor.run_task is not None
    await supervisor.run_task


async def test_retired_device_evidence_pruned_without_stage(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An orphaned device_evidence entry is dropped at the next hook check, no stage."""
    entry = await _setup_env(hass, energy_manager, std_prefs)
    supervisor = entry.runtime_data.learning
    await _seed_evidence_less(supervisor)
    await async_confirm_candidate(hass, entry, "sig-kettle", "Kettle")
    devices = entry.runtime_data.devices
    uuid = next(
        s for s in entry.get_subentries_of_type("device") if s.title == "Kettle"
    ).data["device_uuid"]
    await _seed_record(supervisor, [])

    supervisor._device_evidence[uuid] = {
        **_single_evidence(),
        "active_v": devices.record(uuid)["active_v"],
    }
    supervisor._device_evidence["orphan-uuid"] = {**_single_evidence(), "active_v": 1}

    calls = {"n": 0}

    async def _fake_stage(self: Any, stage: str, job_path: Any) -> dict[str, Any]:
        await asyncio.sleep(0)
        calls["n"] += 1
        return {}

    monkeypatch.setattr(LearningSupervisor, "_async_run_stage", _fake_stage)

    await supervisor._async_maybe_evidence()
    assert calls["n"] == 0
    assert supervisor.run_task is None
    assert "orphan-uuid" not in supervisor._device_evidence
    assert uuid in supervisor._device_evidence

    await supervisor._learning_store.async_save(supervisor._build_learning_store_data())
    stored = await Store(hass, LEARNING_STORES_VERSION, LEARNING_STORE_KEY).async_load()
    assert "orphan-uuid" not in stored.get("device_evidence", {})


async def test_device_evidence_read_time_fallback(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
) -> None:
    """device_evidence falls back to the confirmed-from candidate until a stage stores an entry."""
    entry = await _setup_env(hass, energy_manager, std_prefs)
    supervisor = entry.runtime_data.learning
    cand = make_switched_candidate("sig-kettle", "Kettle")
    cand["evidence"] = _single_evidence(observed_days=13)
    await _seed_record(supervisor, [cand])
    await async_confirm_candidate(hass, entry, "sig-kettle", "Kettle")
    devices = entry.runtime_data.devices
    uuid = next(
        s for s in entry.get_subentries_of_type("device") if s.title == "Kettle"
    ).data["device_uuid"]

    assert "evidence" not in devices.record(uuid)
    assert supervisor.device_evidence(uuid) == cand["evidence"]

    cand.pop("evidence")
    assert supervisor.device_evidence(uuid) is None

    stored_entry = {
        **_single_evidence(observed_days=3),
        "active_v": devices.record(uuid)["active_v"],
    }
    supervisor._device_evidence[uuid] = stored_entry
    assert supervisor.device_evidence(uuid) == stored_entry

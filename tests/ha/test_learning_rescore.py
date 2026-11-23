"""Tests for automatic local preview candidate re-scoring at 21 days."""

import asyncio
from datetime import datetime, time, timedelta
import json
from pathlib import Path
from typing import Any
import pytest

from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ServiceValidationError
from homeassistant.helpers.storage import Store
import homeassistant.util.dt as dt_util

from custom_components.powerguesser.const import (
    CANDIDATES_STORE_KEY,
    DOMAIN,
    LEARNING_STORE_KEY,
    LEARNING_STORES_VERSION,
)
from custom_components.powerguesser.learning import LearningStageError
from tests.conftest import (
    async_fill_synth_buffer,
    async_seed_consent,
    async_seed_llm,
    async_set_power,
    async_setup_via_flow,
    mock_llm,
)
from tests.fixtures.learning.synth import SynthSpec


def _set_std_powers(hass: HomeAssistant) -> None:
    async_set_power(hass, "sensor.grid_power", 500)
    async_set_power(hass, "sensor.solar_power", 1200)
    async_set_power(hass, "sensor.battery_power", -300)
    async_set_power(hass, "sensor.washer_power", 200)


def _make_candidate(sig_id: str = "sig-kettle", ref: str = "d1") -> dict[str, Any]:
    return {
        "schema_version": 1,
        "signature_id": sig_id,
        "kind": "switched",
        "label": "Electric kettle",
        "category": "kettle",
        "extraction": {"grid_s": 30, "edge_threshold_w": 120.0},
        "states": [
            {
                "name": "pulse",
                "rise_w": {"lo": 1800.0, "p50": 2000.0, "hi": 2200.0},
                "fall_w": {"lo": -2200.0, "p50": -2000.0, "hi": -1800.0},
                "dur_s": {"lo": 120.0, "p50": 180.0, "hi": 240.0},
                "plateau_w": {"lo": 1800.0, "p50": 2000.0, "hi": 2200.0},
                "phase_share": None,
            }
        ],
        "session": None,
        "time_prior": {},
        "constraints": {},
        "provenance": {"device_ref": ref, "cluster_ids": ["c01"]},
        "scores": {
            "train": {"share": 0.15, "n": 20},
            "val": None,
            "validated": False,
            "conflicts": 0,
        },
    }


async def _setup_rescore_env(
    hass: HomeAssistant,
    energy_manager: object,
    std_prefs: dict,
    days: int = 22,
    window_days_ago: int = 8,
    validated: bool = False,
) -> tuple[Any, dict[str, Any]]:
    """Set up supervisor with synthetic data and a preview candidate record."""
    energy_manager.data = std_prefs
    _set_std_powers(hass)
    entry = await async_setup_via_flow(hass)
    await async_fill_synth_buffer(hass, entry, spec=SynthSpec(days=days))
    await async_seed_llm(hass, entry)
    await async_seed_consent(hass, host="llm.test", names=False)

    now = dt_util.now()
    today_midnight = dt_util.start_of_local_day(now)
    window_end_dt = today_midnight - timedelta(days=window_days_ago)
    window_end_ms = int(round(window_end_dt.timestamp() * 1000))

    cand = _make_candidate()
    record = {
        "run_id": "preview-run-1",
        "created": window_end_dt.isoformat(),
        "window_end_ms": window_end_ms,
        "validated": validated,
        "candidates": [cand],
    }
    cand_store = Store(hass, LEARNING_STORES_VERSION, CANDIDATES_STORE_KEY)
    await cand_store.async_save(record)

    supervisor = entry.runtime_data.learning
    supervisor._candidates_record = record
    supervisor.candidates = [cand]
    c_data = dict(supervisor.coordinator.data)
    c_data["pending_candidates"] = 1
    supervisor.coordinator.data = c_data

    return entry, record


async def test_rescore_automatic_on_usable_days_refresh(
    hass: HomeAssistant,
    energy_manager: object,
    std_prefs: dict,
    aioclient_mock: object,
    inprocess_worker: object,
) -> None:
    """Candidates Store with validated=False and usable days >= 21 gets rescored automatically."""
    entry, _ = await _setup_rescore_env(hass, energy_manager, std_prefs)
    mock_llm(aioclient_mock, "http://llm.test/v1", lambda req: {"status": 200, "json": {}})

    supervisor = entry.runtime_data.learning
    await supervisor._async_periodic_refresh(None)

    assert supervisor.run_task is not None
    await supervisor.run_task

    # No LLM calls and no token changes
    assert len(aioclient_mock.mock_calls) == 0
    state = hass.states.get("sensor.powerguesser_learning_status")
    assert state is not None
    attrs = state.attributes
    assert attrs["tokens_this_run"] == 0
    assert attrs["tokens_this_month"] == 0
    assert attrs["last_result"] is None
    assert attrs["running_job"] is None

    # Candidate and Store were updated with validation scores
    cand = supervisor.candidates[0]
    assert cand["scores"]["validated"] is True
    val_scores = cand["scores"]["val"]
    assert val_scores is not None
    assert val_scores["share"] > 0
    assert val_scores["n"] > 0

    saved_store = await supervisor._candidates_store.async_load()
    assert saved_store["validated"] is True


async def test_first_label_survives_rescore(
    hass: HomeAssistant,
    energy_manager: object,
    std_prefs: dict,
    aioclient_mock: object,
    inprocess_worker: object,
) -> None:
    """The rescore merge replaces only cand['scores']; first_label set by the run survives (D-18)."""
    entry, _ = await _setup_rescore_env(hass, energy_manager, std_prefs)

    supervisor = entry.runtime_data.learning
    supervisor.candidates[0]["first_label"] = "Kettle A"
    await supervisor._candidates_store.async_save(supervisor._candidates_record)

    mock_llm(aioclient_mock, "http://llm.test/v1", lambda req: {"status": 200, "json": {}})
    await supervisor._async_periodic_refresh(None)

    assert supervisor.run_task is not None
    await supervisor.run_task

    # Validation scores were merged in, first_label untouched
    cand = supervisor.candidates[0]
    assert cand["scores"]["validated"] is True
    assert cand["first_label"] == "Kettle A"

    saved_store = await supervisor._candidates_store.async_load()
    assert saved_store["candidates"][0]["first_label"] == "Kettle A"


async def test_no_rescore_when_conditions_not_met(
    hass: HomeAssistant,
    energy_manager: object,
    std_prefs: dict,
    aioclient_mock: object,
    inprocess_worker: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rescore does not run if already validated, usable days < 21, or no days post window."""
    # 1. Already validated
    entry, _ = await _setup_rescore_env(hass, energy_manager, std_prefs, validated=True)
    supervisor = entry.runtime_data.learning
    await supervisor._async_maybe_rescore()
    assert supervisor.run_task is None

    # 2. Usable days < 21
    supervisor._candidates_record["validated"] = False
    monkeypatch.setattr(supervisor, "_async_get_usable_days", lambda: asyncio.sleep(0, result=18.0))
    await supervisor._async_maybe_rescore()
    assert supervisor.run_task is None

    # 3. No whole day after window_end_ms
    monkeypatch.setattr(supervisor, "_async_get_usable_days", lambda: asyncio.sleep(0, result=22.0))
    today_midnight = dt_util.start_of_local_day(dt_util.now())
    supervisor._candidates_record["window_end_ms"] = int(round(today_midnight.timestamp() * 1000))
    await supervisor._async_maybe_rescore()
    assert supervisor.run_task is None

    # 4. Trigger learning during rescore raises already_running
    # Set valid window_end 3 days ago
    supervisor._candidates_record["window_end_ms"] = int(
        round((today_midnight - timedelta(days=3)).timestamp() * 1000)
    )

    stage_held = asyncio.Event()
    orig_run_stage = supervisor._async_run_stage

    async def _holding_run_stage(stage: str, job_path: Any) -> dict[str, Any]:
        stage_held.set()
        await asyncio.sleep(0.05)
        return await orig_run_stage(stage, job_path)

    monkeypatch.setattr(supervisor, "_async_run_stage", _holding_run_stage)

    await supervisor._async_maybe_rescore()
    assert supervisor.run_task is not None
    await stage_held.wait()

    with pytest.raises(ServiceValidationError) as exc_info:
        await supervisor.async_request_run("service")
    assert exc_info.value.translation_key == "already_running"

    await supervisor.run_task


async def test_rescore_shows_running_job(
    hass: HomeAssistant,
    energy_manager: object,
    std_prefs: dict,
    aioclient_mock: object,
    inprocess_worker: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sensor state is 'running' with running_job 'rescore', then restores previous status."""
    entry, _ = await _setup_rescore_env(hass, energy_manager, std_prefs)
    supervisor = entry.runtime_data.learning

    worker_event = asyncio.Event()
    worker_released = asyncio.Event()

    async def _held_worker(stage: str, job_path: Any) -> dict[str, Any]:
        worker_event.set()
        await worker_released.wait()
        return {
            "S": 0.12,
            "conflict_share": 0.0,
            "per_signature": {"sig-kettle": {"share": 0.12, "n": 15}},
            "days": 8,
        }

    monkeypatch.setattr(supervisor, "_async_run_stage", _held_worker)

    await supervisor._async_maybe_rescore()
    assert supervisor.run_task is not None
    await worker_event.wait()

    # During rescore
    state = hass.states.get("sensor.powerguesser_learning_status")
    assert state is not None
    assert state.state == "running"
    assert state.attributes.get("running_job") == "rescore"

    # Finish rescore
    worker_released.set()
    await supervisor.run_task

    # After rescore
    state = hass.states.get("sensor.powerguesser_learning_status")
    assert state is not None
    assert state.state == "idle"
    assert state.attributes.get("running_job") is None

    # Now verify learning run shows running_job "learning"
    mock_llm(aioclient_mock, "http://llm.test/v1", lambda req: {"status": 200, "json": {}})
    learn_event = asyncio.Event()
    learn_released = asyncio.Event()

    async def _held_learning_worker(stage: str, job_path: Any) -> dict[str, Any]:
        learn_event.set()
        await learn_released.wait()
        raise LearningStageError("interrupted test")

    monkeypatch.setattr(supervisor, "_async_run_stage", _held_learning_worker)

    await supervisor.async_request_run("service")
    assert supervisor.run_task is not None
    await learn_event.wait()

    state = hass.states.get("sensor.powerguesser_learning_status")
    assert state is not None
    assert state.state == "running"
    assert state.attributes.get("running_job") == "learning"

    learn_released.set()
    await supervisor.run_task
    state = hass.states.get("sensor.powerguesser_learning_status")
    assert state.attributes.get("running_job") is None


@pytest.mark.parametrize("fail", [False, True])
async def test_rescore_never_persists_running(
    hass: HomeAssistant,
    energy_manager: object,
    std_prefs: dict,
    monkeypatch: pytest.MonkeyPatch,
    fail: bool,
) -> None:
    """The learning Store holds the restored status after a rescore, never "running" (WR-02)."""
    entry, _ = await _setup_rescore_env(hass, energy_manager, std_prefs)
    supervisor = entry.runtime_data.learning
    status_before = supervisor.coordinator.data["status"]
    assert status_before != "running"

    async def _worker(stage: str, job_path: Any) -> dict[str, Any]:
        if fail:
            raise LearningStageError("rescore worker crash")
        return {
            "S": 0.1,
            "conflict_share": 0.0,
            "per_signature": {"sig-kettle": {"share": 0.1, "n": 10}},
            "days": 8,
            "stats": {},
        }

    monkeypatch.setattr(supervisor, "_async_run_stage", _worker)
    await supervisor._async_maybe_rescore()
    assert supervisor.run_task is not None
    await supervisor.run_task

    stored = await Store(hass, LEARNING_STORES_VERSION, LEARNING_STORE_KEY).async_load()
    assert stored is not None
    assert stored["status"] == status_before
    assert stored["rescore_failures"] == (1 if fail else 0)

    # A restart does not turn the finished rescore into "failed / interrupted"
    new_supervisor = type(supervisor)(hass, entry, supervisor.hub)
    await new_supervisor.async_start()
    try:
        assert new_supervisor.coordinator.data["status"] != "failed"
        assert new_supervisor.coordinator.data["last_error"] != "interrupted"
    finally:
        await new_supervisor.async_stop()


async def test_rescore_without_per_signature_does_not_validate(
    hass: HomeAssistant,
    energy_manager: object,
    std_prefs: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rescore result without per_signature is a failure, never a 0 % validation (CR-02)."""
    entry, _ = await _setup_rescore_env(hass, energy_manager, std_prefs)
    supervisor = entry.runtime_data.learning

    async def _old_contract_worker(stage: str, job_path: Any) -> dict[str, Any]:
        return {
            "digest": "digest.json",
            "meta": "digest_meta.json",
            "episodes": "episodes.npz",
            "stats": {"wall_s": 1.0, "max_rss_kb": 1},
        }

    monkeypatch.setattr(supervisor, "_async_run_stage", _old_contract_worker)

    await supervisor._async_maybe_rescore()
    assert supervisor.run_task is not None
    await supervisor.run_task

    cand = supervisor.candidates[0]
    assert cand["scores"]["validated"] is False
    assert cand["scores"]["val"] is None
    assert supervisor._candidates_record["validated"] is False
    saved = await supervisor._candidates_store.async_load()
    assert saved["validated"] is False
    assert supervisor.coordinator.data["rescore_failures"] == 1


async def test_rescore_failures_counter(
    hass: HomeAssistant,
    energy_manager: object,
    std_prefs: dict,
    aioclient_mock: object,
    inprocess_worker: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Failing rescores increment rescore_failures, survives store reload, resets on success."""
    entry, _ = await _setup_rescore_env(hass, energy_manager, std_prefs)
    supervisor = entry.runtime_data.learning

    async def _failing_worker(stage: str, job_path: Any) -> dict[str, Any]:
        raise LearningStageError("rescore worker crash")

    monkeypatch.setattr(supervisor, "_async_run_stage", _failing_worker)

    # 1. First failure
    await supervisor._async_maybe_rescore()
    assert supervisor.run_task is not None
    await supervisor.run_task

    state = hass.states.get("sensor.powerguesser_learning_status")
    assert state is not None
    assert state.attributes.get("rescore_failures") == 1
    assert supervisor.candidates[0]["scores"]["validated"] is False

    # 2. Second failure
    supervisor.run_task = None
    await supervisor._async_maybe_rescore()
    assert supervisor.run_task is not None
    await supervisor.run_task

    state = hass.states.get("sensor.powerguesser_learning_status")
    assert state.attributes.get("rescore_failures") == 2

    # 3. Survives reload of learning Store
    new_supervisor = type(supervisor)(hass, entry, supervisor.hub)
    await new_supervisor.async_start()
    try:
        assert new_supervisor.coordinator.data.get("rescore_failures") == 2

        # 4. Success resets counter to 0
        monkeypatch.setattr(
            new_supervisor,
            "_async_run_stage",
            lambda stage, job_path: asyncio.sleep(
                0,
                result={
                    "S": 0.1,
                    "conflict_share": 0.0,
                    "per_signature": {"sig-kettle": {"share": 0.1, "n": 10}},
                    "days": 8,
                },
            ),
        )
        new_supervisor.run_task = None
        await new_supervisor._async_maybe_rescore()
        assert new_supervisor.run_task is not None
        await new_supervisor.run_task

        c_data = new_supervisor.coordinator.data
        assert c_data.get("rescore_failures") == 0
        assert new_supervisor.candidates[0]["scores"]["validated"] is True
    finally:
        await new_supervisor.async_stop()


@pytest.mark.parametrize("vep", [7.5, None])
async def test_rescore_merges_validation_error_pct(
    hass: HomeAssistant,
    energy_manager: object,
    std_prefs: dict,
    monkeypatch: pytest.MonkeyPatch,
    vep: float | None,
) -> None:
    """The merge reads the rescore key "validation_error_pct" (never "claimed_share").

    A numeric value lands in scores with basis "learning_split"; None nulls
    both (D-16, review concern L197@4e11826).
    """
    entry, _ = await _setup_rescore_env(hass, energy_manager, std_prefs)
    supervisor = entry.runtime_data.learning

    async def _worker(stage: str, job_path: Any) -> dict[str, Any]:
        return {
            "S": 0.1,
            "conflict_share": 0.0,
            "per_signature": {
                "sig-kettle": {"share": 0.04, "n": 5, "validation_error_pct": vep}
            },
            "days": 8,
        }

    monkeypatch.setattr(supervisor, "_async_run_stage", _worker)
    await supervisor._async_maybe_rescore()
    assert supervisor.run_task is not None
    await supervisor.run_task

    scores = supervisor.candidates[0]["scores"]
    assert scores["val"]["share"] == pytest.approx(0.04)
    assert scores["val"]["n"] == 5
    assert scores["validation_error_pct"] == vep
    assert scores["validation_basis"] == ("learning_split" if vep is not None else None)

    saved = await supervisor._candidates_store.async_load()
    saved_scores = saved["candidates"][0]["scores"]
    assert saved_scores["validation_error_pct"] == vep
    assert saved_scores["validation_basis"] == (
        "learning_split" if vep is not None else None
    )


async def test_concurrent_starts_claim_one_slot(
    hass: HomeAssistant,
    energy_manager: object,
    std_prefs: dict,
    aioclient_mock: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two learn requests, or a learn request and a rescore, racing over an await start one job (WR-09)."""
    entry, _ = await _setup_rescore_env(hass, energy_manager, std_prefs)
    supervisor = entry.runtime_data.learning
    release = asyncio.Event()
    started: list[str] = []

    async def _held_worker(stage: str, job_path: Any) -> dict[str, Any]:
        started.append(stage)
        await release.wait()
        raise LearningStageError("stopped by test")

    monkeypatch.setattr(supervisor, "_async_run_stage", _held_worker)

    async def _slow_usable_days() -> float:
        await asyncio.sleep(0.01)  # the cache refresh hops to the executor in production
        return 22.0

    monkeypatch.setattr(supervisor, "_async_get_usable_days", _slow_usable_days)

    results = await asyncio.gather(
        supervisor.async_request_run("service"),
        supervisor.async_request_run("schedule"),
        supervisor._async_maybe_rescore(),
        return_exceptions=True,
    )
    errors = [r for r in results if isinstance(r, ServiceValidationError)]
    assert len(errors) == 1
    assert errors[0].translation_key == "already_running"
    task = supervisor.run_task
    assert task is not None

    for _ in range(100):
        if started:
            break
        await asyncio.sleep(0.05)
    await asyncio.sleep(0.1)
    assert len(started) == 1
    assert supervisor.run_task is task

    release.set()
    await task
    assert supervisor.coordinator.data["running_job"] is None

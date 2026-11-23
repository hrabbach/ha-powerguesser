"""Learning jobs fall back to the 30 s gate when no interval profile snapshot exists yet."""

import json
from pathlib import Path
from typing import Any

import pytest

from homeassistant.core import HomeAssistant

from custom_components.powerguesser.engine.intervals import gate
from custom_components.powerguesser.learning import LearningSupervisor
from tests.ha.test_learning_rescore import _setup_rescore_env

_FALLBACK_TYPES = [t.value for t in gate(30.0, True)[0]]


@pytest.fixture
def captured_jobs(
    monkeypatch: pytest.MonkeyPatch, hass: HomeAssistant, inprocess_worker: object
) -> dict[str, dict[str, Any]]:
    """Record each stage's job file, then run the stage in process."""
    jobs: dict[str, dict[str, Any]] = {}
    inner = LearningSupervisor._async_run_stage

    async def _capture(self: LearningSupervisor, stage: str, job_path: Any) -> dict[str, Any]:
        jobs[stage] = await hass.async_add_executor_job(
            lambda: json.loads(Path(job_path).read_text(encoding="utf-8"))
        )
        return await inner(self, stage, job_path)

    monkeypatch.setattr(LearningSupervisor, "_async_run_stage", _capture)
    return jobs


async def test_prepare_without_profile_uses_fallback_gate(
    hass: HomeAssistant,
    energy_manager: object,
    std_prefs: dict,
    captured_jobs: dict[str, dict[str, Any]],
) -> None:
    """Prepare (extract job) with no profile snapshot uses grid 30 s and gate(30, assumed) types."""
    entry, _ = await _setup_rescore_env(hass, energy_manager, std_prefs)
    supervisor = entry.runtime_data.learning
    supervisor.hub.profile_coordinator.data = None

    digest, meta = await supervisor._async_prepare()

    assert isinstance(digest, dict)
    assert isinstance(meta, dict)
    job = captured_jobs["extract"]
    assert job["grid_s"] == 30
    assert job["enabled_event_types"] == _FALLBACK_TYPES


async def test_rescore_without_profile_uses_fallback_gate(
    hass: HomeAssistant,
    energy_manager: object,
    std_prefs: dict,
    captured_jobs: dict[str, dict[str, Any]],
) -> None:
    """Rescore with no profile snapshot uses grid 30 s and gate(30, assumed) types and validates."""
    entry, record = await _setup_rescore_env(hass, energy_manager, std_prefs)
    supervisor = entry.runtime_data.learning
    supervisor.hub.profile_coordinator.data = None

    await supervisor._async_periodic_refresh(None)
    assert supervisor.run_task is not None
    await supervisor.run_task

    job = captured_jobs["rescore"]
    assert job["grid_s"] == 30
    assert job["enabled_event_types"] == _FALLBACK_TYPES
    assert supervisor._candidates_record["validated"] is True

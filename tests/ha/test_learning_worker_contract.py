"""Contract tests: what the real worker subprocess returns to the supervisor (CR-01, CR-02, WR-15).

These tests run the real `_async_run_stage` (real subprocess, real stdout `done` event) so a
mismatch between the worker output contract and what the supervisor reads cannot hide behind
the `inprocess_worker` fixture.
"""

import json
import logging
import os
from pathlib import Path
import re
from typing import Any

import numpy as np
import pytest

from homeassistant.core import HomeAssistant

from custom_components.powerguesser.const import BUFFER_DIR, LEARNING_DIR
from tests.conftest import (
    async_fill_synth_buffer,
    async_set_power,
    async_setup_via_flow,
)
from tests.fixtures.learning.synth import SynthSpec


def _set_std_powers(hass: HomeAssistant) -> None:
    async_set_power(hass, "sensor.grid_power", 500)
    async_set_power(hass, "sensor.solar_power", 1200)
    async_set_power(hass, "sensor.battery_power", -300)
    async_set_power(hass, "sensor.washer_power", 200)


def _device(ref: str, cluster: str) -> dict[str, Any]:
    return {
        "ref": ref,
        "label": f"Device {ref}",
        "category": "unknown",
        "tracker_kind": "switched",
        "member_clusters": [cluster],
        "states": [{"name": "on", "cluster_ref": cluster, "role": "main"}],
        "phase_hint": "single",
        "confidence": "medium",
        "rationale": f"Device for {cluster}.",
        "ambiguous_with": [],
        "reference_plug": "",
    }


async def _extract(hass: HomeAssistant, supervisor: Any, stage_dir: Path) -> tuple[dict, dict]:
    def _write_job() -> tuple[Path, dict[str, Any]]:
        stage_dir.mkdir(parents=True, exist_ok=True)
        job, _, _ = supervisor._build_extract_job_data(stage_dir)
        path = stage_dir / "job_extract.json"
        path.write_text(json.dumps(job), encoding="utf-8")
        return path, job

    job_path, job = await hass.async_add_executor_job(_write_job)
    await supervisor._async_run_stage("extract", job_path)

    def _read() -> dict[str, Any]:
        return json.loads((stage_dir / "digest.json").read_text(encoding="utf-8"))

    return await hass.async_add_executor_job(_read), job


async def _compile(
    hass: HomeAssistant,
    supervisor: Any,
    stage_dir: Path,
    round_no: int,
    proposal: dict[str, Any],
    incumbent: dict[str, Any] | None,
) -> dict[str, Any]:
    def _write() -> Path:
        (stage_dir / f"proposal_{round_no}.json").write_text(
            json.dumps(proposal), encoding="utf-8"
        )
        job = {
            "stage_dir": str(stage_dir),
            "episodes_path": str(stage_dir / "episodes.npz"),
            "proposal_path": str(stage_dir / f"proposal_{round_no}.json"),
            "run_id": "contract",
            "round": round_no,
            "model": "m1",
            "incumbent": incumbent,
            "validated": True,
            "prior": [],
            "decisions": {},
        }
        path = stage_dir / f"job_compile_{round_no}.json"
        path.write_text(json.dumps(job), encoding="utf-8")
        return path

    job_path = await hass.async_add_executor_job(_write)
    return await supervisor._async_run_stage("compile_score", job_path)


@pytest.mark.slow
async def test_real_worker_returns_compile_and_rescore_results(
    hass: HomeAssistant,
    energy_manager: object,
    std_prefs: dict,
) -> None:
    """The supervisor sees S, conflict_share, accepted and per_signature from the real worker."""
    energy_manager.data = std_prefs
    _set_std_powers(hass)
    entry = await async_setup_via_flow(hass)
    await async_fill_synth_buffer(hass, entry, spec=SynthSpec(days=22, washer=True))
    supervisor = entry.runtime_data.learning

    stage_dir = Path(hass.config.path(BUFFER_DIR, LEARNING_DIR, "contract"))
    digest, extract_job = await _extract(hass, supervisor, stage_dir)
    switched = [c["id"] for c in digest["clusters"] if c.get("kind") == "switched"]
    assert switched, digest
    proposal = {
        "analysis": "Contract test.",
        "schema_version": 1,
        "devices": [_device("d1", switched[0])],
        "rejected_clusters": [],
        "merge_suggestions": [],
    }

    # Round 1: no incumbent, always accepted; the score must be the one in val_1.json.
    out1 = await _compile(hass, supervisor, stage_dir, 1, proposal, None)
    val_1 = json.loads((stage_dir / "val_1.json").read_text(encoding="utf-8"))
    assert out1["accepted"] is True
    assert out1["S"] == pytest.approx(val_1["S"], abs=1e-4)
    assert out1["conflict_share"] == pytest.approx(val_1["conflict_share"], abs=1e-4)
    assert out1["S"] > 0.0

    # Round 2: the same proposal against its own score is not an improvement.
    out2 = await _compile(
        hass,
        supervisor,
        stage_dir,
        2,
        proposal,
        {"round": 1, "S": out1["S"], "conflict_share": out1["conflict_share"]},
    )
    assert out2["accepted"] is False

    # Round 3: against a much worse incumbent it is accepted.
    out3 = await _compile(
        hass,
        supervisor,
        stage_dir,
        3,
        proposal,
        {"round": 1, "S": -1.0, "conflict_share": 1.0},
    )
    assert out3["accepted"] is True

    library_1 = json.loads((stage_dir / "library_1.json").read_text(encoding="utf-8"))
    assert library_1

    # Rescore the round-1 library over the last 8 days.
    def _write_rescore_job() -> Path:
        job = {
            "buffer_path": extract_job["buffer_path"],
            "stage_dir": str(stage_dir / "rescore"),
            "grid_s": extract_job["grid_s"],
            "day_bounds_ms": extract_job["day_bounds_ms"][-9:],
            "first_weekday": 0,
            "enabled_event_types": extract_job["enabled_event_types"],
            "metered_channels": extract_job["metered_channels"],
            "phase_channels": extract_job["phase_channels"],
            "candidates": library_1,
        }
        path = stage_dir / "job_rescore.json"
        path.write_text(json.dumps(job), encoding="utf-8")
        return path

    rescore_job = await hass.async_add_executor_job(_write_rescore_job)
    res = await supervisor._async_run_stage("rescore", rescore_job)
    assert "per_signature" in res
    assert set(res["per_signature"]) == {c["signature_id"] for c in library_1}
    sig = library_1[0]["signature_id"]
    assert res["per_signature"][sig]["share"] > 0.0
    assert res["per_signature"][sig]["n"] > 0


# 300 MiB in kB: the parent's ballast. Under the ru_maxrss bug the child reports at least this.
_BALLAST_BYTES = 300 * 1024 * 1024
_BALLAST_KB = _BALLAST_BYTES // 1024


@pytest.mark.slow
@pytest.mark.skipif(
    not os.path.exists("/proc/self/status"), reason="needs /proc/self/status (Linux)"
)
async def test_real_worker_reports_its_own_peak_memory(
    hass: HomeAssistant,
    energy_manager: object,
    std_prefs: dict,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A stage spawned under a 300 MiB parent reports and logs the worker's own peak memory.

    The test process is the worker's real parent (as Home Assistant is in production). On Linux
    ru_maxrss carries the parent's RSS high-water mark across fork and exec, so a worker that
    reports ru_maxrss shows at least the ballast; VmHWM is the worker's own (03-VERIFICATION gap).
    """
    energy_manager.data = std_prefs
    _set_std_powers(hass)
    entry = await async_setup_via_flow(hass)
    await async_fill_synth_buffer(hass, entry, spec=SynthSpec(days=22, washer=True))
    supervisor = entry.runtime_data.learning

    stage_dir = Path(hass.config.path(BUFFER_DIR, LEARNING_DIR, "peak_memory"))

    def _write_job() -> Path:
        stage_dir.mkdir(parents=True, exist_ok=True)
        job, _, _ = supervisor._build_extract_job_data(stage_dir)
        path = stage_dir / "job_extract.json"
        path.write_text(json.dumps(job), encoding="utf-8")
        return path

    job_path = await hass.async_add_executor_job(_write_job)

    # Allocate and touch the ballast so it is resident in the parent while the worker spawns.
    ballast = np.ones(_BALLAST_BYTES, dtype=np.uint8)
    caplog.set_level(logging.INFO)
    result = await supervisor._async_run_stage("extract", job_path)
    assert int(ballast[-1]) == 1  # keep the ballast alive across the spawn

    max_rss_kb = result["stats"]["max_rss_kb"]
    assert 0 < max_rss_kb < _BALLAST_KB, max_rss_kb

    lines = [
        r.getMessage()
        for r in caplog.records
        if r.getMessage().startswith("Learning stage extract finished in")
    ]
    assert len(lines) == 1, lines
    match = re.search(r"peak memory (\d+) MB", lines[0])
    assert match is not None, lines[0]
    assert 0 < int(match.group(1)) < 300, lines[0]

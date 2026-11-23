"""Tests for the powerguesser.run_validation service (Plan 05-01)."""

import asyncio
import contextlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError, ServiceValidationError

from custom_components.powerguesser.const import (
    BUFFER_DIR,
    DOMAIN,
    LEARNING_DIR,
    SHADOW_STORE_KEY,
    VALIDATION_DIR,
    VALIDATION_STORE_KEY,
    VALIDATION_STORES_VERSION,
)
from tests.conftest import (
    async_fill_synth_buffer,
    async_seed_shadow_set,
    async_set_power,
    async_setup_via_flow,
)


def _kettle_artefact() -> dict:
    from tests.fixtures.detect_traces import switched_artefact

    return switched_artefact(
        "sig-shadow-kettle",
        rise=(1800, 2000, 2200),
        fall=(-2200, -2000, -1800),
        dur=(120, 180, 300),
        plateau=(1800, 2000, 2200),
    )


def _kettle_link(overlap: float = 0.8) -> dict:
    return {
        "entity_id": "sensor.kettle_power",
        "signature_id": "sig-shadow-kettle",
        "kind": "switched",
        "overlap": overlap,
    }


async def _seed_kettle_shadow(hass: HomeAssistant, entry: Any, **kwargs: Any) -> dict:
    """Seed a shadow set with the switched kettle artefact linked to the kettle plug."""
    return await async_seed_shadow_set(
        hass, entry, [_kettle_artefact()], [_kettle_link()], **kwargs
    )


async def _run_validation(hass: HomeAssistant) -> dict:
    return await hass.services.async_call(
        DOMAIN, "run_validation", {}, blocking=True, return_response=True
    )


async def test_run_validation_reports_kettle_plug_figures(
    hass: HomeAssistant,
    energy_manager: object,
    kettle_plug_prefs: dict,
    inprocess_worker: None,
    aioclient_mock: object,
) -> None:
    """run_validation replays a stored shadow set and reports the kettle plug's figures."""
    from tests.fixtures.learning.synth import SynthSpec

    energy_manager.data = kettle_plug_prefs
    async_set_power(hass, "sensor.grid_power", 500)
    async_set_power(hass, "sensor.solar_power", 1200)
    async_set_power(hass, "sensor.battery_power", -300)
    async_set_power(hass, "sensor.washer_power", 200)
    async_set_power(hass, "sensor.kettle_power", 0)

    entry = await async_setup_via_flow(hass)
    await async_fill_synth_buffer(hass, entry, SynthSpec(days=35, kettle_plug=True))
    shadow = await _seed_kettle_shadow(hass, entry)

    res = await _run_validation(hass)

    assert res["status"] == "ok"
    assert res["reason"] is None
    assert "train_cutoff" in res
    holdout = res["holdout"]
    assert len(holdout["days"]) == 7
    assert all(day["coverage"] >= 0.5 for day in holdout["days"])
    assert all("date" in day for day in holdout["days"])
    assert holdout["start_ms"] >= shadow["train_cutoff_ms"]
    assert holdout["end_ms"] > holdout["start_ms"]

    kettle = next(
        d for d in res["devices"] if d["entity_id"] == "sensor.kettle_power"
    )
    assert kettle["signature_id"] == "sig-shadow-kettle"
    assert kettle["link_overlap"] == 0.8
    assert kettle["energy_true_wh"] > 0
    assert kettle["energy_est_wh"] > 0
    assert isinstance(kettle["energy_error_pct"], (int, float))
    assert isinstance(kettle["baseline"]["energy_error_pct"], (int, float))
    assert kettle["n_events"] >= 8

    # The washer plug has no shadow link: no figure, and it is not discovered
    assert all(d["entity_id"] != "sensor.washer_power" for d in res["devices"])

    # Local and deterministic: no HTTP request was made
    assert len(aioclient_mock.mock_calls) == 0


async def test_run_validation_reports_nde_and_mae(
    hass: HomeAssistant,
    energy_manager: object,
    kettle_plug_prefs: dict,
    inprocess_worker: None,
    aioclient_mock: object,
) -> None:
    """VAL-01: the kettle entry carries NDE, MAE and the baseline's NDE on the matched samples."""
    from tests.fixtures.learning.synth import SynthSpec

    energy_manager.data = kettle_plug_prefs
    async_set_power(hass, "sensor.grid_power", 500)
    async_set_power(hass, "sensor.solar_power", 1200)
    async_set_power(hass, "sensor.battery_power", -300)
    async_set_power(hass, "sensor.washer_power", 200)
    async_set_power(hass, "sensor.kettle_power", 0)

    entry = await async_setup_via_flow(hass)
    await async_fill_synth_buffer(hass, entry, SynthSpec(days=35, kettle_plug=True))
    await _seed_kettle_shadow(hass, entry)

    res = await _run_validation(hass)

    assert res["status"] == "ok"
    kettle = next(
        d for d in res["devices"] if d["entity_id"] == "sensor.kettle_power"
    )
    assert isinstance(kettle["nde"], (int, float))
    assert isinstance(kettle["mae_w"], (int, float))
    assert isinstance(kettle["baseline"]["nde"], (int, float))
    assert kettle["energy_est_wh"] > 0

    # The report states how both figures were computed (VAL-01).
    metrics = res["metrics"]
    assert "nde" in metrics and "sum((y - y_hat)^2)" in metrics["nde"]
    assert "mae_w" in metrics and "mean |y - y_hat|" in metrics["mae_w"]

    # Local: no HTTP request was made
    assert len(aioclient_mock.mock_calls) == 0


async def test_run_validation_not_loaded(hass: HomeAssistant) -> None:
    """Test calling powerguesser.run_validation raises when no entry is loaded."""
    from custom_components.powerguesser import async_setup
    from custom_components.powerguesser.const import SERVICE_RUN_VALIDATION

    assert await async_setup(hass, {})

    with pytest.raises(ServiceValidationError) as exc_info:
        await hass.services.async_call(
            DOMAIN, SERVICE_RUN_VALIDATION, {}, blocking=True, return_response=True
        )
    assert exc_info.value.translation_key == "not_loaded"


async def test_run_validation_already_running(
    hass: HomeAssistant,
    energy_manager: object,
    kettle_plug_prefs: dict,
    aioclient_mock: object,
) -> None:
    """A validation refuses to start while another PowerGuesser job holds the slot."""
    energy_manager.data = kettle_plug_prefs
    async_set_power(hass, "sensor.grid_power", 500)
    async_set_power(hass, "sensor.solar_power", 1200)
    async_set_power(hass, "sensor.battery_power", -300)
    async_set_power(hass, "sensor.washer_power", 200)
    async_set_power(hass, "sensor.kettle_power", 0)

    entry = await async_setup_via_flow(hass)
    await _seed_kettle_shadow(hass, entry)
    supervisor = entry.runtime_data.learning

    async def _hang() -> None:
        await asyncio.Event().wait()

    supervisor.run_task = entry.async_create_background_task(hass, _hang(), "hang")
    try:
        with pytest.raises(ServiceValidationError) as exc_info:
            await hass.services.async_call(
                DOMAIN, "run_validation", {}, blocking=True, return_response=True
            )
        assert exc_info.value.translation_key == "already_running"
        assert entry.runtime_data.validation.latest_report is None
    finally:
        supervisor.run_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await supervisor.run_task
        supervisor.run_task = None


async def test_run_validation_without_shadow_set_refuses(
    hass: HomeAssistant,
    energy_manager: object,
    kettle_plug_prefs: dict,
    aioclient_mock: object,
) -> None:
    """Without a stored shadow set the report is refused with no number in it."""
    energy_manager.data = kettle_plug_prefs
    async_set_power(hass, "sensor.grid_power", 500)
    async_set_power(hass, "sensor.solar_power", 1200)
    async_set_power(hass, "sensor.battery_power", -300)
    async_set_power(hass, "sensor.washer_power", 200)
    async_set_power(hass, "sensor.kettle_power", 0)

    entry = await async_setup_via_flow(hass)

    res = await _run_validation(hass)

    assert res["status"] == "refused"
    assert res["reason"] == "no_shadow_set"
    assert res["schema_version"] == 1
    assert "generated_at" in res
    assert "trigger" in res
    assert "devices" not in res
    assert entry.runtime_data.validation.latest_report == res


async def test_run_validation_worker_failure_keeps_report(
    hass: HomeAssistant,
    energy_manager: object,
    kettle_plug_prefs: dict,
    inprocess_worker: None,
    aioclient_mock: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A worker failure raises validation_failed and keeps the previous report."""
    from custom_components.powerguesser.learning import (
        LearningStageError,
        LearningSupervisor,
    )
    from tests.fixtures.learning.synth import SynthSpec

    energy_manager.data = kettle_plug_prefs
    async_set_power(hass, "sensor.grid_power", 500)
    async_set_power(hass, "sensor.solar_power", 1200)
    async_set_power(hass, "sensor.battery_power", -300)
    async_set_power(hass, "sensor.washer_power", 200)
    async_set_power(hass, "sensor.kettle_power", 0)

    entry = await async_setup_via_flow(hass)
    await async_fill_synth_buffer(hass, entry, SynthSpec(days=35, kettle_plug=True))
    await _seed_kettle_shadow(hass, entry)

    res_ok = await _run_validation(hass)
    assert res_ok["status"] == "ok"

    async def _boom(self: Any, stage: str, job_path: Any) -> dict:
        raise LearningStageError(f"{stage}: boom")

    monkeypatch.setattr(LearningSupervisor, "_async_run_stage", _boom)

    with pytest.raises(HomeAssistantError) as exc_info:
        await _run_validation(hass)
    assert exc_info.value.translation_key == "validation_failed"
    assert entry.runtime_data.validation.latest_report == res_ok


async def test_latest_report_survives_stage_dir_cleanup_and_reload(
    hass: HomeAssistant,
    energy_manager: object,
    kettle_plug_prefs: dict,
    inprocess_worker: None,
    aioclient_mock: object,
) -> None:
    """The latest report lives in the Store, not the transient stage directory."""
    import shutil

    from tests.fixtures.learning.synth import SynthSpec

    energy_manager.data = kettle_plug_prefs
    async_set_power(hass, "sensor.grid_power", 500)
    async_set_power(hass, "sensor.solar_power", 1200)
    async_set_power(hass, "sensor.battery_power", -300)
    async_set_power(hass, "sensor.washer_power", 200)
    async_set_power(hass, "sensor.kettle_power", 0)

    entry = await async_setup_via_flow(hass)
    await async_fill_synth_buffer(hass, entry, SynthSpec(days=35, kettle_plug=True))
    await _seed_kettle_shadow(hass, entry)

    res = await _run_validation(hass)
    assert res["status"] == "ok"
    report = dict(entry.runtime_data.validation.latest_report)

    # The next learning run's sibling cleanup deletes the validation stage dir
    validation_dir = hass.config.path(BUFFER_DIR, LEARNING_DIR, VALIDATION_DIR)
    assert Path(validation_dir, "report.json").exists()
    await hass.async_add_executor_job(shutil.rmtree, validation_dir, True)

    assert await hass.config_entries.async_reload(entry.entry_id)
    reloaded = hass.config_entries.async_entries(DOMAIN)[0]
    assert reloaded.runtime_data.validation.latest_report == report


async def test_plug_classes_round_trip_through_job(
    hass: HomeAssistant,
    energy_manager: object,
    kettle_plug_prefs: dict,
    aioclient_mock: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The stored class map rides in the job and an ok report's map is merged."""
    from homeassistant.helpers.storage import Store

    from custom_components.powerguesser.learning import LearningSupervisor

    energy_manager.data = kettle_plug_prefs
    async_set_power(hass, "sensor.grid_power", 500)
    async_set_power(hass, "sensor.solar_power", 1200)
    async_set_power(hass, "sensor.battery_power", -300)
    async_set_power(hass, "sensor.washer_power", 200)
    async_set_power(hass, "sensor.kettle_power", 0)

    # Seed the class map before the entry loads
    await Store(hass, VALIDATION_STORES_VERSION, VALIDATION_STORE_KEY).async_save(
        {"report": None, "plug_classes": {"sensor.kettle_power": "x"}}
    )

    entry = await async_setup_via_flow(hass)
    await _seed_kettle_shadow(hass, entry)
    assert entry.runtime_data.validation.latest_report is None

    calls = {"n": 0}

    async def _write_stub_report(
        stage: str, job_path: Any, plug_classes: dict | None
    ) -> None:
        job = json.loads(Path(str(job_path)).read_text(encoding="utf-8"))
        # The stored map travels in the job (first run: the seeded map)
        if calls["n"] == 0:
            assert job["plug_classes"] == {"sensor.kettle_power": "x"}
        calls["n"] += 1
        stage_dir = Path(job["stage_dir"])
        report: dict[str, Any] = {
            "status": "ok",
            "reason": None,
            "train_cutoff_ms": job["shadow"]["train_cutoff_ms"],
            "holdout": {"start_ms": 0, "end_ms": 0, "days": []},
        }
        if plug_classes is not None:
            report["plug_classes"] = plug_classes
        (stage_dir / "report.json").write_text(json.dumps(report), encoding="utf-8")

    async def _run_with_stub(plug_classes: dict | None) -> dict:
        async def _stub(self: Any, stage: str, job_path: Any) -> dict:
            await _write_stub_report(stage, job_path, plug_classes)
            return {"stats": {"wall_s": 0.0, "max_rss_kb": 0}}

        monkeypatch.setattr(LearningSupervisor, "_async_run_stage", _stub)
        return await _run_validation(hass)

    res = await _run_with_stub(
        {"sensor.kettle_power": "z", "sensor.other_power": "y"}
    )
    assert res["status"] == "ok"
    stored = await Store(
        hass, VALIDATION_STORES_VERSION, VALIDATION_STORE_KEY
    ).async_load()
    # kettle overwritten, other added
    assert stored["plug_classes"] == {
        "sensor.kettle_power": "z",
        "sensor.other_power": "y",
    }

    # An ok report without the key keeps the map
    res = await _run_with_stub(None)
    assert res["status"] == "ok"
    stored = await Store(
        hass, VALIDATION_STORES_VERSION, VALIDATION_STORE_KEY
    ).async_load()
    assert stored["plug_classes"] == {
        "sensor.kettle_power": "z",
        "sensor.other_power": "y",
    }


async def test_refused_report_keeps_plug_classes(
    hass: HomeAssistant,
    energy_manager: object,
    kettle_plug_prefs: dict,
    aioclient_mock: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A refused report, whatever it carries, never erases the class map (D-06)."""
    from homeassistant.helpers.storage import Store

    from custom_components.powerguesser.learning import LearningSupervisor

    energy_manager.data = kettle_plug_prefs
    async_set_power(hass, "sensor.grid_power", 500)
    async_set_power(hass, "sensor.solar_power", 1200)
    async_set_power(hass, "sensor.battery_power", -300)
    async_set_power(hass, "sensor.washer_power", 200)
    async_set_power(hass, "sensor.kettle_power", 0)

    await Store(hass, VALIDATION_STORES_VERSION, VALIDATION_STORE_KEY).async_save(
        {"report": None, "plug_classes": {"sensor.kettle_power": "x"}}
    )

    entry = await async_setup_via_flow(hass)
    await _seed_kettle_shadow(hass, entry)

    async def _run_with_stub(report: dict) -> dict:
        async def _stub(self: Any, stage: str, job_path: Any) -> dict:
            job = json.loads(Path(str(job_path)).read_text(encoding="utf-8"))
            stage_dir = Path(job["stage_dir"])
            full = {
                "train_cutoff_ms": job["shadow"]["train_cutoff_ms"],
                "holdout": {"start_ms": 0, "end_ms": 0, "days": []},
                **report,
            }
            (stage_dir / "report.json").write_text(json.dumps(full), encoding="utf-8")
            return {"stats": {"wall_s": 0.0, "max_rss_kb": 0}}

        monkeypatch.setattr(LearningSupervisor, "_async_run_stage", _stub)
        return await _run_validation(hass)

    base = {"status": "refused", "reason": "insufficient_data"}

    # Refusal without a plug_classes key
    res = await _run_with_stub(dict(base))
    assert res["status"] == "refused"
    stored = await Store(
        hass, VALIDATION_STORES_VERSION, VALIDATION_STORE_KEY
    ).async_load()
    assert stored["plug_classes"] == {"sensor.kettle_power": "x"}

    # Refusal carrying an empty plug_classes map
    res = await _run_with_stub({**base, "plug_classes": {}})
    assert res["status"] == "refused"
    stored = await Store(
        hass, VALIDATION_STORES_VERSION, VALIDATION_STORE_KEY
    ).async_load()
    assert stored["plug_classes"] == {"sensor.kettle_power": "x"}

    # The no_shadow_set refusal leaves the map unchanged too
    entry.runtime_data.validation._shadow = None
    res = await _run_validation(hass)
    assert res["status"] == "refused"
    assert res["reason"] == "no_shadow_set"
    stored = await Store(
        hass, VALIDATION_STORES_VERSION, VALIDATION_STORE_KEY
    ).async_load()
    assert stored["plug_classes"] == {"sensor.kettle_power": "x"}


async def test_report_entries_carry_names(
    hass: HomeAssistant,
    energy_manager: object,
    kettle_plug_prefs: dict,
    aioclient_mock: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """devices/refusals/not_discovered entries carry the entity's friendly name."""
    from custom_components.powerguesser.learning import LearningSupervisor

    energy_manager.data = kettle_plug_prefs
    async_set_power(hass, "sensor.grid_power", 500)
    async_set_power(hass, "sensor.solar_power", 1200)
    async_set_power(hass, "sensor.battery_power", -300)
    async_set_power(hass, "sensor.washer_power", 200)
    hass.states.async_set(
        "sensor.kettle_power",
        "0",
        {
            "device_class": "power",
            "unit_of_measurement": "W",
            "friendly_name": "Kitchen Kettle",
        },
    )

    entry = await async_setup_via_flow(hass)
    await _seed_kettle_shadow(hass, entry)

    async def _stub(self: Any, stage: str, job_path: Any) -> dict:
        job = json.loads(Path(str(job_path)).read_text(encoding="utf-8"))
        stage_dir = Path(job["stage_dir"])
        report = {
            "status": "ok",
            "reason": None,
            "train_cutoff_ms": job["shadow"]["train_cutoff_ms"],
            "holdout": {"start_ms": 0, "end_ms": 0, "days": []},
            "devices": [
                {"entity_id": "sensor.kettle_power", "signature_id": "s"},
                {"entity_id": "sensor.ghost_power", "signature_id": "s2"},
            ],
            "refusals": [{"entity_id": "sensor.kettle_power", "reason": "x"}],
            "not_discovered": [{"entity_id": "sensor.ghost_power"}],
        }
        (stage_dir / "report.json").write_text(json.dumps(report), encoding="utf-8")
        return {"stats": {"wall_s": 0.0, "max_rss_kb": 0}}

    monkeypatch.setattr(LearningSupervisor, "_async_run_stage", _stub)
    res = await _run_validation(hass)

    assert res["devices"][0]["name"] == "Kitchen Kettle"
    # Missing state falls back to the entity_id
    assert res["devices"][1]["name"] == "sensor.ghost_power"
    assert res["refusals"][0]["name"] == "Kitchen Kettle"
    assert res["not_discovered"][0]["name"] == "sensor.ghost_power"


async def test_remove_entry_removes_validation_stores(
    hass: HomeAssistant,
    energy_manager: object,
    kettle_plug_prefs: dict,
    inprocess_worker: None,
    aioclient_mock: object,
) -> None:
    """Entry removal deletes the powerguesser.shadow and powerguesser.validation Stores."""
    from homeassistant.helpers.storage import Store

    from custom_components.powerguesser import async_remove_entry
    from tests.fixtures.learning.synth import SynthSpec

    energy_manager.data = kettle_plug_prefs
    async_set_power(hass, "sensor.grid_power", 500)
    async_set_power(hass, "sensor.solar_power", 1200)
    async_set_power(hass, "sensor.battery_power", -300)
    async_set_power(hass, "sensor.washer_power", 200)
    async_set_power(hass, "sensor.kettle_power", 0)

    entry = await async_setup_via_flow(hass)
    await async_fill_synth_buffer(hass, entry, SynthSpec(days=35, kettle_plug=True))
    await _seed_kettle_shadow(hass, entry)
    res = await _run_validation(hass)
    assert res["status"] == "ok"

    await async_remove_entry(hass, entry)

    assert (
        await Store(hass, VALIDATION_STORES_VERSION, SHADOW_STORE_KEY).async_load()
        is None
    )
    assert (
        await Store(hass, VALIDATION_STORES_VERSION, VALIDATION_STORE_KEY).async_load()
        is None
    )


async def test_run_validation_is_local_and_deterministic(
    hass: HomeAssistant,
    energy_manager: object,
    kettle_plug_prefs: dict,
    inprocess_worker: None,
    aioclient_mock: object,
) -> None:
    """Two validations on the same seeded data return identical device entries."""
    from tests.fixtures.learning.synth import SynthSpec

    energy_manager.data = kettle_plug_prefs
    async_set_power(hass, "sensor.grid_power", 500)
    async_set_power(hass, "sensor.solar_power", 1200)
    async_set_power(hass, "sensor.battery_power", -300)
    async_set_power(hass, "sensor.washer_power", 200)
    async_set_power(hass, "sensor.kettle_power", 0)

    entry = await async_setup_via_flow(hass)
    await async_fill_synth_buffer(hass, entry, SynthSpec(days=35, kettle_plug=True))
    await _seed_kettle_shadow(hass, entry)

    res1 = await _run_validation(hass)
    res2 = await _run_validation(hass)

    assert res1["status"] == "ok"
    assert res2["status"] == "ok"
    assert res1["devices"] == res2["devices"]
    assert len(aioclient_mock.mock_calls) == 0


async def test_run_validation_reports_stale_shadow_set_without_refusing(
    hass: HomeAssistant,
    energy_manager: object,
    kettle_plug_prefs: dict,
    inprocess_worker: None,
    aioclient_mock: object,
) -> None:
    """D-27: the report says what it measured; a stale set is flagged, never refused."""
    from datetime import timedelta

    import homeassistant.util.dt as dt_util

    from tests.fixtures.learning.synth import SynthSpec

    energy_manager.data = kettle_plug_prefs
    async_set_power(hass, "sensor.grid_power", 500)
    async_set_power(hass, "sensor.solar_power", 1200)
    async_set_power(hass, "sensor.battery_power", -300)
    async_set_power(hass, "sensor.washer_power", 200)
    async_set_power(hass, "sensor.kettle_power", 0)

    entry = await async_setup_via_flow(hass)
    await async_fill_synth_buffer(hass, entry, SynthSpec(days=35, kettle_plug=True))

    created_fresh = (dt_util.utcnow() - timedelta(days=29)).isoformat()
    shadow = await _seed_kettle_shadow(hass, entry, created=created_fresh)
    res = await _run_validation(hass)

    assert res["status"] == "ok"
    assert res["shadow_created"] == created_fresh
    assert abs(res["shadow_age_days"] - 29.0) <= 0.1
    assert res["shadow_stale"] is False

    created_stale = (dt_util.utcnow() - timedelta(days=31)).isoformat()
    await _seed_kettle_shadow(hass, entry, created=created_stale)
    res2 = await _run_validation(hass)

    assert res2["status"] == "ok"
    assert res2["shadow_created"] == created_stale
    assert abs(res2["shadow_age_days"] - 31.0) <= 0.1
    assert res2["shadow_stale"] is True
    # The stale set is still measured: the same kettle figures
    assert res2["devices"] == res["devices"]
    assert shadow["schema_version"] == 1


async def test_run_validation_refuses_short_history(
    hass: HomeAssistant,
    energy_manager: object,
    kettle_plug_prefs: dict,
    inprocess_worker: None,
    aioclient_mock: object,
) -> None:
    """Fewer than 28 usable days: the report refuses with a reason, no figures (D-12)."""
    from tests.fixtures.learning.synth import SynthSpec

    energy_manager.data = kettle_plug_prefs
    async_set_power(hass, "sensor.grid_power", 500)
    async_set_power(hass, "sensor.solar_power", 1200)
    async_set_power(hass, "sensor.battery_power", -300)
    async_set_power(hass, "sensor.washer_power", 200)
    async_set_power(hass, "sensor.kettle_power", 0)

    entry = await async_setup_via_flow(hass)
    await async_fill_synth_buffer(hass, entry, SynthSpec(days=20, kettle_plug=True))
    await _seed_kettle_shadow(hass, entry)

    res = await _run_validation(hass)

    assert res["status"] == "refused"
    assert res["reason"] == "insufficient_data"
    assert res["devices"] == []
    assert res["classes"] == []
    assert "plug_classes" not in res


async def test_run_validation_classes_round_trip(
    hass: HomeAssistant,
    energy_manager: object,
    kettle_plug_prefs: dict,
    inprocess_worker: None,
    aioclient_mock: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An ok report's class map merges into the Store and rides the next job.

    A later run against the same shadow set re-stored with a different grid_s
    refuses with grid_mismatch and the stored class map survives (D-06, D-12).
    """
    from homeassistant.helpers.storage import Store

    from custom_components.powerguesser.learning import LearningSupervisor
    from tests.fixtures.learning.synth import SynthSpec

    energy_manager.data = kettle_plug_prefs
    async_set_power(hass, "sensor.grid_power", 500)
    async_set_power(hass, "sensor.solar_power", 1200)
    async_set_power(hass, "sensor.battery_power", -300)
    async_set_power(hass, "sensor.washer_power", 200)
    async_set_power(hass, "sensor.kettle_power", 0)

    entry = await async_setup_via_flow(hass)
    await async_fill_synth_buffer(hass, entry, SynthSpec(days=35, kettle_plug=True))
    shadow = await _seed_kettle_shadow(hass, entry)

    res = await _run_validation(hass)
    assert res["status"] == "ok"
    kettle = next(d for d in res["devices"] if d["entity_id"] == "sensor.kettle_power")
    assert kettle["class"] == "switched_1.5_3kw_1_level"
    assert isinstance(kettle["passes"], bool)
    assert isinstance(kettle["reasons"], list)
    assert kettle["unvalidated"] == (not kettle["passes"])
    assert res["plug_classes"] == {"sensor.kettle_power": "switched_1.5_3kw_1_level"}

    # The stored map rides into the next validate job.
    captured: dict[str, Any] = {}

    async def _capture(self: Any, stage: str, job_path: Any) -> dict:
        job = json.loads(Path(str(job_path)).read_text(encoding="utf-8"))
        captured["plug_classes"] = job.get("plug_classes")
        return await orig_run_stage(self, stage, job_path)

    orig_run_stage = LearningSupervisor._async_run_stage
    monkeypatch.setattr(LearningSupervisor, "_async_run_stage", _capture)
    await _run_validation(hass)
    assert captured["plug_classes"] == {"sensor.kettle_power": "switched_1.5_3kw_1_level"}
    monkeypatch.undo()

    # Re-store the same set with a different grid_s: refused, map survives.
    shadow_bad_grid = {**shadow, "grid_s": 60}
    await entry.runtime_data.validation.async_store_shadow(shadow_bad_grid)
    res2 = await _run_validation(hass)
    assert res2["status"] == "refused"
    assert res2["reason"] == "grid_mismatch"
    stored = await Store(
        hass, VALIDATION_STORES_VERSION, VALIDATION_STORE_KEY
    ).async_load()
    assert stored["plug_classes"] == {"sensor.kettle_power": "switched_1.5_3kw_1_level"}


# --- Holdout-basis drift in the report (Plan 05-06; D-15, D-17, D-23) ---

_KETTLE_CLASS = "switched_1.5_3kw_1_level"


def _holdout_entry(end: str, err: float, run_id: str) -> dict[str, Any]:
    """One plug/class history entry under a shadow set."""
    return {"shadow_run_id": run_id, "holdout_end": end, "energy_error_pct": err}


def _over_counted_state(end: str, run_id: str = "test-shadow") -> dict[str, Any]:
    """A stored holdout_drift state whose hysteresis already counted `end` over."""
    return {
        "shadow_run_id": run_id,
        "drift": False,
        "reason": None,
        "since": None,
        "last_end": end,
        "hysteresis": {"active": False, "over_count": 1},
    }


def _seed_holdout_store(
    hass_storage: dict[str, Any],
    *,
    run_id: str = "test-shadow",
    plug_windows: list[tuple[str, float]] | None = None,
    plug_state: dict[str, Any] | None = None,
    class_windows: list[tuple[str, float]] | None = None,
    class_state: dict[str, Any] | None = None,
    corrupt_class: list[Any] | None = None,
) -> None:
    """Seed the powerguesser.validation Store with history/holdout_drift."""
    history: dict[str, Any] = {}
    if plug_windows is not None:
        history["sensor.kettle_power"] = [
            _holdout_entry(end, err, run_id) for end, err in plug_windows
        ]
    if class_windows is not None:
        history[f"class:{_KETTLE_CLASS}"] = [
            _holdout_entry(end, err, run_id) for end, err in class_windows
        ]
    if corrupt_class is not None:
        history["class:corrupt"] = corrupt_class
    drift: dict[str, Any] = {}
    if plug_state is not None:
        drift["sensor.kettle_power"] = plug_state
    if class_state is not None:
        drift[f"class:{_KETTLE_CLASS}"] = class_state
    hass_storage[VALIDATION_STORE_KEY] = {
        "version": VALIDATION_STORES_VERSION,
        "minor_version": 1,
        "key": VALIDATION_STORE_KEY,
        "data": {
            "report": None,
            "plug_classes": {},
            "history": history,
            "holdout_drift": drift,
        },
    }


def _date_ms(day: str) -> int:
    """Midday UTC of an ISO date: the local holdout_end stays that date in any tz."""
    year, month, dom = (int(part) for part in day.split("-"))
    return int(datetime(year, month, dom, 12, 0, tzinfo=timezone.utc).timestamp() * 1000)


def _stub_validate(
    *,
    devices: list[dict[str, Any]] | None = None,
    classes: list[dict[str, Any]] | None = None,
    end_day: str = "2026-11-24",
    status: str = "ok",
    reason: str | None = None,
) -> Any:
    """A validate-stage stub writing a report for the window ending on end_day."""

    async def _stub(self: Any, stage: str, job_path: Any) -> dict:
        assert stage == "validate"
        job = json.loads(Path(str(job_path)).read_text(encoding="utf-8"))
        stage_dir = Path(job["stage_dir"])
        report: dict[str, Any] = {
            "status": status,
            "reason": reason,
            "train_cutoff_ms": job["shadow"]["train_cutoff_ms"],
            "holdout": {"start_ms": 0, "end_ms": _date_ms(end_day), "days": []},
            "devices": devices or [],
            "classes": classes or [],
        }
        (stage_dir / "report.json").write_text(json.dumps(report), encoding="utf-8")
        return {"stats": {"wall_s": 0.0, "max_rss_kb": 0}}

    return _stub


def _kettle_device_entry(error: float = 30.0) -> dict[str, Any]:
    """A figure-bearing kettle devices entry as assemble_report writes it."""
    return {
        "entity_id": "sensor.kettle_power",
        "signature_id": "sig-shadow-kettle",
        "energy_error_pct": error,
        "passes": True,
        "unvalidated": False,
    }


def _kettle_class_entry(median: float | None = 30.0) -> dict[str, Any]:
    """A classes entry for the kettle class as assemble_report writes it."""
    return {
        "class": _KETTLE_CLASS,
        "peers": 2,
        "passing": 2,
        "median_error_pct": median,
        "passes": median is not None,
    }


async def _setup_holdout_env(
    hass: HomeAssistant,
    energy_manager: object,
    kettle_plug_prefs: dict,
) -> Any:
    """Kettle-plug entry with a 35-day buffer and a seeded kettle shadow set."""
    from tests.fixtures.learning.synth import SynthSpec

    energy_manager.data = kettle_plug_prefs
    async_set_power(hass, "sensor.grid_power", 500)
    async_set_power(hass, "sensor.solar_power", 1200)
    async_set_power(hass, "sensor.battery_power", -300)
    async_set_power(hass, "sensor.washer_power", 200)
    async_set_power(hass, "sensor.kettle_power", 0)

    entry = await async_setup_via_flow(hass)
    await async_fill_synth_buffer(hass, entry, SynthSpec(days=35, kettle_plug=True))
    shadow = await _seed_kettle_shadow(hass, entry)
    return entry, shadow


async def _load_validation_store(hass: HomeAssistant) -> dict[str, Any]:
    from homeassistant.helpers.storage import Store

    stored = await Store(
        hass, VALIDATION_STORES_VERSION, VALIDATION_STORE_KEY
    ).async_load()
    assert isinstance(stored, dict)
    return stored


async def test_plug_holdout_drift_in_report(
    hass: HomeAssistant,
    energy_manager: object,
    kettle_plug_prefs: dict,
    inprocess_worker: None,
    aioclient_mock: object,
    monkeypatch: pytest.MonkeyPatch,
    hass_storage: dict[str, Any],
) -> None:
    """D-15/D-16: an ok report's kettle entry reads drift true with reason,
    since and best_error_pct when the third window of the same shadow set
    confirms the rise; every other field (incl. unvalidated) is unchanged;
    a refused report changes no history or state."""
    from custom_components.powerguesser.learning import LearningSupervisor

    _seed_holdout_store(
        hass_storage,
        plug_windows=[("2026-11-10", 8.0), ("2026-11-17", 20.0)],
        plug_state=_over_counted_state("2026-11-17"),
    )
    entry, _shadow = await _setup_holdout_env(hass, energy_manager, kettle_plug_prefs)

    monkeypatch.setattr(
        LearningSupervisor,
        "_async_run_stage",
        _stub_validate(devices=[_kettle_device_entry(30.0)]),
    )
    res = await _run_validation(hass)
    kettle = next(d for d in res["devices"] if d["entity_id"] == "sensor.kettle_power")
    assert kettle["drift"] is True
    assert kettle["drift_reason"] == "validation_error_rising"
    assert kettle["drift_since"] == "2026-11-24"
    assert kettle["best_error_pct"] == 8.0
    # Every non-drift field is returned and stored unchanged.
    assert kettle["energy_error_pct"] == 30.0
    assert kettle["passes"] is True
    assert kettle["unvalidated"] is False

    stored = await _load_validation_store(hass)
    assert stored["history"]["sensor.kettle_power"] == [
        _holdout_entry("2026-11-10", 8.0, "test-shadow"),
        _holdout_entry("2026-11-17", 20.0, "test-shadow"),
        _holdout_entry("2026-11-24", 30.0, "test-shadow"),
    ]
    state = stored["holdout_drift"]["sensor.kettle_power"]
    assert state["drift"] is True
    assert state["reason"] == "validation_error_rising"
    assert state["since"] == "2026-11-24"
    assert state["shadow_run_id"] == "test-shadow"
    # The stored report carries the enriched entry too.
    stored_kettle = next(
        d for d in stored["report"]["devices"] if d["entity_id"] == "sensor.kettle_power"
    )
    assert stored_kettle["drift"] is True

    # A refused report changes no history and no state.
    monkeypatch.setattr(
        LearningSupervisor,
        "_async_run_stage",
        _stub_validate(status="refused", reason="insufficient_data"),
    )
    res2 = await _run_validation(hass)
    assert res2["status"] == "refused"
    stored2 = await _load_validation_store(hass)
    assert stored2["history"]["sensor.kettle_power"] == stored["history"]["sensor.kettle_power"]
    assert stored2["holdout_drift"]["sensor.kettle_power"] == state


async def test_same_window_rerun_does_not_count(
    hass: HomeAssistant,
    energy_manager: object,
    kettle_plug_prefs: dict,
    inprocess_worker: None,
    aioclient_mock: object,
    monkeypatch: pytest.MonkeyPatch,
    hass_storage: dict[str, Any],
) -> None:
    """RESEARCH Pitfall 6: a re-run on the same window of the same shadow set
    replaces the history entry and does not advance the consecutive-over
    counter; the next distinct window is the second consecutive over."""
    from custom_components.powerguesser.learning import LearningSupervisor

    _seed_holdout_store(
        hass_storage,
        plug_windows=[("2026-11-10", 8.0), ("2026-11-17", 20.0)],
        plug_state=_over_counted_state("2026-11-17"),
    )
    entry, _shadow = await _setup_holdout_env(hass, energy_manager, kettle_plug_prefs)

    # Run 1: window B measured again -> replaces, no evaluation, drift false.
    monkeypatch.setattr(
        LearningSupervisor,
        "_async_run_stage",
        _stub_validate(devices=[_kettle_device_entry(20.0)], end_day="2026-11-17"),
    )
    res = await _run_validation(hass)
    kettle = res["devices"][0]
    assert kettle["drift"] is False
    assert kettle["drift_reason"] is None

    # Run 2: same window once more -> still no drift (the counter would have
    # fired at two if reruns counted).
    res = await _run_validation(hass)
    assert res["devices"][0]["drift"] is False

    # Run 3: a new window over the threshold -> the second consecutive over.
    monkeypatch.setattr(
        LearningSupervisor,
        "_async_run_stage",
        _stub_validate(devices=[_kettle_device_entry(30.0)], end_day="2026-11-24"),
    )
    res = await _run_validation(hass)
    kettle = res["devices"][0]
    assert kettle["drift"] is True
    assert kettle["drift_since"] == "2026-11-24"

    stored = await _load_validation_store(hass)
    entries = stored["history"]["sensor.kettle_power"]
    # Window B was replaced, not duplicated.
    assert [e["holdout_end"] for e in entries] == [
        "2026-11-10",
        "2026-11-17",
        "2026-11-24",
    ]


async def test_new_shadow_set_does_not_set_error_rising(
    hass: HomeAssistant,
    energy_manager: object,
    kettle_plug_prefs: dict,
    inprocess_worker: None,
    aioclient_mock: object,
    monkeypatch: pytest.MonkeyPatch,
    hass_storage: dict[str, Any],
) -> None:
    """D-15/D-17: a new shadow set starts a fresh history window and a fresh
    hysteresis state, so its errors are never compared with the old set's
    best; the old set's entries stay in the history for display only."""
    from custom_components.powerguesser.learning import LearningSupervisor

    _seed_holdout_store(
        hass_storage,
        plug_windows=[("2026-11-10", 8.0), ("2026-11-17", 20.0)],
        plug_state=_over_counted_state("2026-11-17"),
    )
    entry, shadow = await _setup_holdout_env(hass, energy_manager, kettle_plug_prefs)

    # Re-store the same shadow set under a new run_id.
    shadow2 = {**shadow, "run_id": "test-shadow-2"}
    await entry.runtime_data.validation.async_store_shadow(shadow2)

    monkeypatch.setattr(
        LearningSupervisor,
        "_async_run_stage",
        _stub_validate(devices=[_kettle_device_entry(30.0)], end_day="2026-11-24"),
    )
    res = await _run_validation(hass)
    kettle = res["devices"][0]
    assert kettle["drift"] is False
    assert kettle["drift_reason"] is None
    assert kettle["best_error_pct"] is None  # no earlier error of the new set

    monkeypatch.setattr(
        LearningSupervisor,
        "_async_run_stage",
        _stub_validate(devices=[_kettle_device_entry(31.0)], end_day="2026-12-01"),
    )
    res = await _run_validation(hass)
    kettle = res["devices"][0]
    # Against the old set's best (8.0) both 30 and 31 would set drift; scoped
    # to the new set the best earlier error is 30.0 and 31 stays far under.
    assert kettle["drift"] is False
    assert kettle["drift_reason"] is None
    assert kettle["best_error_pct"] == 30.0

    stored = await _load_validation_store(hass)
    entries = stored["history"]["sensor.kettle_power"]
    assert [e["holdout_end"] for e in entries if e["shadow_run_id"] == "test-shadow"] == [
        "2026-11-10",
        "2026-11-17",
    ]
    assert [e["holdout_end"] for e in entries if e["shadow_run_id"] == "test-shadow-2"] == [
        "2026-11-24",
        "2026-12-01",
    ]
    state = stored["holdout_drift"]["sensor.kettle_power"]
    assert state["shadow_run_id"] == "test-shadow-2"
    assert state["drift"] is False


async def test_class_median_holdout_drift_in_report(
    hass: HomeAssistant,
    energy_manager: object,
    kettle_plug_prefs: dict,
    inprocess_worker: None,
    aioclient_mock: object,
    monkeypatch: pytest.MonkeyPatch,
    hass_storage: dict[str, Any],
) -> None:
    """D-23: a class median with the D-15 rule sets drift on the classes
    entry; a class without a numeric median gets no history entry; a corrupt
    class history entry is dropped on load; a refused report changes nothing;
    a re-stored shadow set starts a fresh class window."""
    from custom_components.powerguesser.learning import LearningSupervisor

    _seed_holdout_store(
        hass_storage,
        class_windows=[("2026-11-10", 8.0), ("2026-11-17", 20.0)],
        class_state=_over_counted_state("2026-11-17"),
        corrupt_class=[{"bad": 1}],
    )
    entry, shadow = await _setup_holdout_env(hass, energy_manager, kettle_plug_prefs)

    median_less = {
        "class": "switched_over_3kw_1_level",
        "peers": 1,
        "passing": 0,
        "median_error_pct": None,
        "passes": False,
    }
    monkeypatch.setattr(
        LearningSupervisor,
        "_async_run_stage",
        _stub_validate(
            classes=[_kettle_class_entry(30.0), median_less], end_day="2026-11-24"
        ),
    )
    res = await _run_validation(hass)
    klass = next(c for c in res["classes"] if c["class"] == _KETTLE_CLASS)
    assert klass["drift"] is True
    assert klass["drift_reason"] == "validation_error_rising"
    assert klass["drift_since"] == "2026-11-24"
    assert klass["best_error_pct"] == 8.0
    # The class entry's own fields are unchanged.
    assert klass["peers"] == 2
    assert klass["passing"] == 2
    assert klass["median_error_pct"] == 30.0
    assert klass["passes"] is True
    # The median-less class carries no drift fields and no history entry.
    bare = next(c for c in res["classes"] if c["class"] == "switched_over_3kw_1_level")
    assert "drift" not in bare

    stored = await _load_validation_store(hass)
    assert stored["history"][f"class:{_KETTLE_CLASS}"] == [
        _holdout_entry("2026-11-10", 8.0, "test-shadow"),
        _holdout_entry("2026-11-17", 20.0, "test-shadow"),
        _holdout_entry("2026-11-24", 30.0, "test-shadow"),
    ]
    assert "switched_over_3kw_1_level" not in "".join(stored["history"].keys())
    # The corrupt class entry was dropped on load, not persisted again.
    assert "class:corrupt" not in stored["history"]
    state = stored["holdout_drift"][f"class:{_KETTLE_CLASS}"]
    assert state["drift"] is True
    assert state["since"] == "2026-11-24"

    # A refused report changes no class history or state.
    monkeypatch.setattr(
        LearningSupervisor,
        "_async_run_stage",
        _stub_validate(status="refused", reason="insufficient_data"),
    )
    res2 = await _run_validation(hass)
    assert res2["status"] == "refused"
    stored2 = await _load_validation_store(hass)
    assert stored2["history"][f"class:{_KETTLE_CLASS}"] == stored["history"][
        f"class:{_KETTLE_CLASS}"
    ]
    assert stored2["holdout_drift"][f"class:{_KETTLE_CLASS}"] == state

    # A re-stored shadow set starts a fresh class window: one new-set entry
    # cannot set drift and the old set's state does not carry over.
    shadow2 = {**shadow, "run_id": "test-shadow-2"}
    await entry.runtime_data.validation.async_store_shadow(shadow2)
    monkeypatch.setattr(
        LearningSupervisor,
        "_async_run_stage",
        _stub_validate(classes=[_kettle_class_entry(31.0)], end_day="2026-12-01"),
    )
    res3 = await _run_validation(hass)
    klass = next(c for c in res3["classes"] if c["class"] == _KETTLE_CLASS)
    assert klass["drift"] is False
    assert klass["drift_reason"] is None
    assert klass["median_error_pct"] == 31.0


async def test_holdout_history_trim_and_corrupt_entries_dropped(
    hass: HomeAssistant,
    energy_manager: object,
    kettle_plug_prefs: dict,
    inprocess_worker: None,
    aioclient_mock: object,
    monkeypatch: pytest.MonkeyPatch,
    hass_storage: dict[str, Any],
) -> None:
    """T-05-21/T-05-22: at most HISTORY_MAX entries are kept per key (oldest
    dropped) and corrupt entries are dropped on load."""
    from custom_components.powerguesser.learning import LearningSupervisor

    windows = [(f"2026-10-{day:02d}", float(day)) for day in range(1, 26)]
    _seed_holdout_store(
        hass_storage,
        plug_windows=windows,
        corrupt_class=[{"bad": 1}, {"shadow_run_id": "test-shadow"}],
    )
    entry, _shadow = await _setup_holdout_env(hass, energy_manager, kettle_plug_prefs)

    monkeypatch.setattr(
        LearningSupervisor,
        "_async_run_stage",
        _stub_validate(devices=[_kettle_device_entry(4.0)], end_day="2026-11-24"),
    )
    await _run_validation(hass)

    stored = await _load_validation_store(hass)
    entries = stored["history"]["sensor.kettle_power"]
    assert len(entries) == 20  # HISTORY_MAX, oldest holdout_end dropped
    assert all(e["holdout_end"] >= "2026-10-07" for e in entries)
    assert entries[-1] == _holdout_entry("2026-11-24", 4.0, "test-shadow")
    # The all-corrupt class key did not survive the load.
    assert "class:corrupt" not in stored["history"]


async def test_run_validation_synthetic_overlap_flag(
    hass: HomeAssistant,
    energy_manager: object,
    kettle_plug_prefs: dict,
    inprocess_worker: None,
    aioclient_mock: object,
) -> None:
    """D-28: the run_validation flag returns a labelled synthetic-overlap block,
    a call without the field behaves exactly as before, and a non-boolean value
    is rejected by the service schema."""
    import voluptuous as vol

    from tests.fixtures.learning.synth import SynthSpec
    from tests.fixtures.programs import program_artefact

    energy_manager.data = kettle_plug_prefs
    async_set_power(hass, "sensor.grid_power", 500)
    async_set_power(hass, "sensor.solar_power", 1200)
    async_set_power(hass, "sensor.battery_power", -300)
    async_set_power(hass, "sensor.washer_power", 200)
    async_set_power(hass, "sensor.kettle_power", 0)

    entry = await async_setup_via_flow(hass)
    await async_fill_synth_buffer(
        hass,
        entry,
        SynthSpec(
            days=35,
            washer=True,
            washer_daily=True,
            kettle_plug=True,
            kettle_during_washer=True,
        ),
    )
    washer_artefact = program_artefact(
        "sig-shadow-washer", [(2000, 30), (200, 80), (500, 10)]
    )
    await async_seed_shadow_set(
        hass,
        entry,
        [_kettle_artefact(), washer_artefact],
        [
            _kettle_link(),
            {
                "entity_id": "sensor.washer_power",
                "signature_id": "sig-shadow-washer",
                "kind": "program",
                "overlap": 0.9,
            },
        ],
    )

    res = await hass.services.async_call(
        DOMAIN,
        "run_validation",
        {"synthetic_overlap": True},
        blocking=True,
        return_response=True,
    )
    assert res["status"] == "ok"
    block = res["overlap_synthetic"]
    assert block["synthetic"] is True
    assert block["label"].startswith("SYNTHETIC:")
    assert block["injected"]["shift_s"] != 0
    assert block["injected"]["b"] in block["label"]

    # Without the field the run behaves exactly as before: no synthetic block and
    # the same real overlap figures.
    plain = await _run_validation(hass)
    assert "overlap_synthetic" not in plain
    assert plain["overlap"] == res["overlap"]

    # A non-boolean value is rejected by the service schema.
    with pytest.raises(vol.Invalid):
        await hass.services.async_call(
            DOMAIN,
            "run_validation",
            {"synthetic_overlap": ["yes"]},
            blocking=True,
            return_response=True,
        )

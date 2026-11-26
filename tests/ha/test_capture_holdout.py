"""run_validation passes capture windows to both validate stages (D-05, Plan 08-04).

Every capture record's holdout window crosses from the captures Store into the
validate and validate_sessions job JSON, so a taught run can never sit in a
validation holdout. Without capture records the job carries no key and the
report carries no ``excluded_capture``.
"""

import copy
from datetime import timedelta
import json
import math
from pathlib import Path
from typing import Any

from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util
from pytest_homeassistant.common import async_fire_time_changed

from custom_components.powerguesser.const import (
    BUFFER_DIR,
    CAPTURES_STORE_KEY,
    CAPTURES_STORE_VERSION,
    DOMAIN,
    LEARNING_DIR,
    VALIDATION_DIR,
)
from custom_components.powerguesser.engine.intervals import ProfileSnapshot, gate
from tests.conftest import (
    async_confirm_candidate,
    async_fill_synth_buffer,
    async_seed_shadow_set,
    async_set_power,
    async_setup_via_flow,
    make_switched_candidate,
    seed_candidates,
)
from tests.fixtures.learning.synth import ENTITY_KETTLE_SESSION, SynthSpec

GRID_S = 30
DAY_MS = 86_400_000


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


async def _seed_kettle_shadow(hass: HomeAssistant, entry: Any) -> dict:
    return await async_seed_shadow_set(
        hass, entry, [_kettle_artefact()], [_kettle_link()]
    )


def _validation_dir(hass: HomeAssistant) -> Path:
    return Path(hass.config.path(BUFFER_DIR, LEARNING_DIR, VALIDATION_DIR))


def _today_midnight_ms() -> int:
    return int(round(dt_util.start_of_local_day(dt_util.now()).timestamp() * 1000))


def _seed_capture_record(hass_storage: dict[str, Any], window_ms: list[int]) -> dict:
    """Seed the captures Store with one done window record carrying the window."""
    record = {
        "capture_id": "cap01",
        "mode": "window",
        "state": "done",
        "target_kind": "candidate",
        "target_id": "cap_0123456789abcdef",
        "new_device_name": "Kettle capture",
        "label": "Kettle capture",
        "press_start_ts": None,
        "press_stop_ts": None,
        "window_start_ms": int(window_ms[0]),
        "window_end_ms": int(window_ms[1]),
        "created_at": "2026-11-20T00:00:00+00:00",
        "resolved_at": "2026-11-20T00:05:00+00:00",
        "reassigned_at": None,
        "reason_code": None,
        "gap_flagged": False,
        "result": None,
        "features": None,
        "trace": None,
        "holdout_excluded": True,
        "holdout_window_ms": [int(window_ms[0]), int(window_ms[1])],
        "replay": None,
    }
    hass_storage[CAPTURES_STORE_KEY] = {
        "version": CAPTURES_STORE_VERSION,
        "minor_version": 1,
        "key": CAPTURES_STORE_KEY,
        "data": {"captures": [record]},
    }
    return record


async def test_run_validation_excludes_capture_windows(
    hass: HomeAssistant,
    energy_manager: object,
    kettle_plug_prefs: dict,
    inprocess_worker: None,
    aioclient_mock: object,
    hass_storage: dict[str, Any],
) -> None:
    """A seeded capture window rides in the validate job and appears in the report."""
    energy_manager.data = kettle_plug_prefs
    async_set_power(hass, "sensor.grid_power", 500)
    async_set_power(hass, "sensor.solar_power", 1200)
    async_set_power(hass, "sensor.battery_power", -300)
    async_set_power(hass, "sensor.washer_power", 200)
    async_set_power(hass, "sensor.kettle_power", 0)

    end_ms = _today_midnight_ms()
    window = [end_ms - 3 * DAY_MS, end_ms - 2 * DAY_MS]
    _seed_capture_record(hass_storage, window)

    entry = await async_setup_via_flow(hass)
    await async_fill_synth_buffer(hass, entry, SynthSpec(days=35, kettle_plug=True))
    await _seed_kettle_shadow(hass, entry)

    res = await hass.services.async_call(
        DOMAIN, "run_validation", {}, blocking=True, return_response=True
    )

    job = json.loads((_validation_dir(hass) / "job_validate.json").read_text())
    assert job["capture_windows"] == [[window[0], window[1]]]
    assert res["status"] == "ok"
    excluded = res["holdout"]["excluded_capture"]
    assert excluded["windows"] == 1
    assert excluded["samples"] > 0


async def test_run_validation_without_captures_passes_no_key(
    hass: HomeAssistant,
    energy_manager: object,
    kettle_plug_prefs: dict,
    inprocess_worker: None,
    aioclient_mock: object,
) -> None:
    """With no capture records the job carries no key and the report none either."""
    energy_manager.data = kettle_plug_prefs
    async_set_power(hass, "sensor.grid_power", 500)
    async_set_power(hass, "sensor.solar_power", 1200)
    async_set_power(hass, "sensor.battery_power", -300)
    async_set_power(hass, "sensor.washer_power", 200)
    async_set_power(hass, "sensor.kettle_power", 0)

    entry = await async_setup_via_flow(hass)
    await async_fill_synth_buffer(hass, entry, SynthSpec(days=35, kettle_plug=True))
    await _seed_kettle_shadow(hass, entry)

    res = await hass.services.async_call(
        DOMAIN, "run_validation", {}, blocking=True, return_response=True
    )

    job = json.loads((_validation_dir(hass) / "job_validate.json").read_text())
    assert "capture_windows" not in job
    assert res["status"] == "ok"
    assert "excluded_capture" not in res["holdout"]


# --- the validate_sessions job also carries the windows ----------------------


def _aligned_move(freezer: Any) -> None:
    now = dt_util.utcnow().timestamp()
    aligned = math.ceil(now / GRID_S) * GRID_S
    freezer.move_to(dt_util.utc_from_timestamp(aligned))


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


async def _seed_device_session(
    hass: HomeAssistant, entry: Any, *, device_uuid: str, window_days: int
) -> None:
    hub = entry.runtime_data
    review = hub.learning.review
    now = dt_util.utcnow()
    review._sessions.append(
        {
            "session_id": "sess-kettle",
            "plug_entity_id": ENTITY_KETTLE_SESSION,
            "target_kind": "device",
            "target_id": device_uuid,
            "start_ts": (now - timedelta(days=window_days)).isoformat(),
            "end_ts": None,
            "planned_end_ts": None,
            "end_reason": None,
        }
    )
    await review._sessions_store.async_save({"sessions": review._sessions})
    await hub.async_rebuild()
    await hass.async_block_till_done()


async def test_sessions_job_carries_capture_windows(
    hass: HomeAssistant,
    energy_manager: object,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
    inprocess_worker: None,
) -> None:
    """A device-target session job carries the same capture_windows as the validate job."""
    seed_candidates(
        hass_storage, [make_switched_candidate("sig01", "Kettle", rise_w=2000.0)]
    )
    energy_manager.data = copy.deepcopy(std_prefs)
    async_set_power(hass, "sensor.grid_power", 1500)
    async_set_power(hass, "sensor.solar_power", 0)
    async_set_power(hass, "sensor.battery_power", 0)
    async_set_power(hass, "sensor.washer_power", 0)

    end_ms = _today_midnight_ms()
    window = [end_ms - 3 * DAY_MS, end_ms - 2 * DAY_MS]
    _seed_capture_record(hass_storage, window)

    _aligned_move(freezer)
    entry = await async_setup_via_flow(hass)
    _pin_grid(entry)
    await hass.async_block_till_done()
    hub = entry.runtime_data

    await async_confirm_candidate(hass, entry, "sig01", "Kettle")
    device_uuid = hub.devices.device_for_signature("sig01")
    assert device_uuid is not None

    await _seed_device_session(hass, entry, device_uuid=device_uuid, window_days=15)
    await async_fill_synth_buffer(
        hass, entry, SynthSpec(days=20, kettle_session=True)
    )
    await hub.writer.async_flush()

    await hass.services.async_call(
        DOMAIN, "run_validation", {}, blocking=True, return_response=True
    )

    job = json.loads(
        (_validation_dir(hass) / "job_validate_sessions.json").read_text()
    )
    assert job["capture_windows"] == [[window[0], window[1]]]

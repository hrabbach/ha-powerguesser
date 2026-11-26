"""HA tests for the D-08 restart sweep and unload timer cleanup.

Plan 08-03 Task 3. On HA restart or entry reload every ``recording`` or
``resolving`` capture ends ``interrupted`` with reason ``restart``; no
``recording`` state survives and no timer callback of the old manager fires
afterwards.
"""

import copy
from datetime import timedelta
from typing import Any

from homeassistant.components.energy.data import EnergyManager
from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util
from pytest_homeassistant.common import async_fire_time_changed

from custom_components.powerguesser.const import (
    CAPTURES_STORE_KEY,
    CAPTURES_STORE_VERSION,
    DOMAIN,
)
from tests.conftest import async_set_power, async_setup_via_flow


def _seed_record(
    capture_id: str, state: str, press_start: str, press_stop: str | None
) -> dict[str, Any]:
    """Build a D-05-complete capture record for seeding the Store."""
    return {
        "capture_id": capture_id,
        "mode": "live",
        "state": state,
        "target_kind": "candidate",
        "target_id": "cap_" + capture_id,
        "new_device_name": "Kettle",
        "label": "kettle",
        "press_start_ts": press_start,
        "press_stop_ts": press_stop,
        "window_start_ms": None,
        "window_end_ms": None,
        "created_at": press_start,
        "resolved_at": None,
        "reassigned_at": None,
        "reason_code": None,
        "gap_flagged": False,
        "result": None,
        "features": None,
        "trace": None,
        "holdout_excluded": True,
        "holdout_window_ms": None,
        "replay": None,
    }


def _stored_captures(hass_storage: dict[str, Any]) -> list[dict[str, Any]]:
    return hass_storage.get(CAPTURES_STORE_KEY, {}).get("data", {}).get("captures", [])


async def _light_setup(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
) -> Any:
    energy_manager.data = copy.deepcopy(std_prefs)
    async_set_power(hass, "sensor.grid_power", 250)
    async_set_power(hass, "sensor.solar_power", 0)
    async_set_power(hass, "sensor.battery_power", 0)
    async_set_power(hass, "sensor.washer_power", 0)
    entry = await async_setup_via_flow(hass)
    await hass.async_block_till_done()
    return entry


async def test_restart_closes_open_captures_as_interrupted(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
) -> None:
    """Seeded recording/resolving records become interrupted/restart at setup."""
    recording = _seed_record(
        "rec", "recording", "2026-11-20T10:00:00+00:00", None
    )
    resolving = _seed_record(
        "res", "resolving", "2026-11-20T10:00:00+00:00", "2026-11-20T10:05:00+00:00"
    )
    hass_storage[CAPTURES_STORE_KEY] = {
        "version": CAPTURES_STORE_VERSION,
        "minor_version": 1,
        "key": CAPTURES_STORE_KEY,
        "data": {"captures": [recording, resolving]},
    }

    entry = await _light_setup(hass, energy_manager, std_prefs)
    records = {r["capture_id"]: r for r in entry.runtime_data.captures.records}

    assert records["rec"]["state"] == "interrupted"
    assert records["rec"]["reason_code"] == "restart"
    assert records["rec"]["gap_flagged"] is True
    assert records["res"]["state"] == "interrupted"
    assert records["res"]["reason_code"] == "restart"
    assert records["res"]["gap_flagged"] is True

    rec_start = int(
        dt_util.parse_datetime(recording["press_start_ts"]).timestamp() * 1000
    )
    assert records["rec"]["holdout_window_ms"] == [
        rec_start,
        rec_start + 3600 * 1000,
    ]
    res_start = int(
        dt_util.parse_datetime(resolving["press_start_ts"]).timestamp() * 1000
    )
    res_stop = int(dt_util.parse_datetime(resolving["press_stop_ts"]).timestamp() * 1000)
    assert records["res"]["holdout_window_ms"] == [res_start, res_stop]

    # capture_list shows no recording.
    listed = await hass.services.async_call(
        DOMAIN, "capture_list", {}, blocking=True, return_response=True
    )
    assert all(c["state"] != "recording" for c in listed["captures"])


async def test_reload_during_recording_interrupts(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
) -> None:
    """A live recording capture is interrupted at reload and no late timer fires."""
    entry = await _light_setup(hass, energy_manager, std_prefs)
    res = await hass.services.async_call(
        DOMAIN,
        "capture_start",
        {"label": "kettle", "target_type": "new_device", "target": "Kettle"},
        blocking=True,
        return_response=True,
    )
    cap = res["capture_id"]
    assert _stored_captures(hass_storage)[0]["state"] == "recording"

    old_captures = entry.runtime_data.captures
    await hass.config_entries.async_reload(entry.entry_id)
    await hass.async_block_till_done()

    reloaded = hass.config_entries.async_entries(DOMAIN)[0]
    assert reloaded.runtime_data.captures is not old_captures
    assert reloaded.runtime_data.captures._unsub_expiry is None
    assert reloaded.runtime_data.captures._unsub_settle is None

    rec = _stored_captures(hass_storage)[0]
    assert rec["capture_id"] == cap
    assert rec["state"] == "interrupted"
    assert rec["reason_code"] == "restart"
    assert rec["gap_flagged"] is True

    # Advancing well past the old expiry timer leaves the record interrupted.
    freezer.tick(timedelta(seconds=3700))
    async_fire_time_changed(hass, dt_util.utcnow())
    await hass.async_block_till_done()
    assert _stored_captures(hass_storage)[0]["state"] == "interrupted"

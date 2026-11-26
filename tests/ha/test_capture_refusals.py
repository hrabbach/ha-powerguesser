"""HA tests for capture_window refusals and DST time handling (CAP-01).

Plan 08-03 Task 1: each SC1 refusal through the admin ``capture_window`` service
raises ``ServiceValidationError`` with translation_key ``capture_<code>`` and
leaves the captures Store untouched; a naive local time is interpreted in the HA
time zone and the result echoes local and UTC times.
"""

import copy
import json
from pathlib import Path
import sqlite3
from typing import Any
from zoneinfo import ZoneInfo

import pytest

from homeassistant.components.energy.data import EnergyManager
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ServiceValidationError
from homeassistant.util import dt as dt_util

from custom_components.powerguesser.const import CAPTURES_STORE_KEY
from custom_components.powerguesser.engine.capture.window import REFUSAL_CODES
from custom_components.powerguesser.engine.intervals import ProfileSnapshot, gate
from tests.conftest import (
    async_fill_synth_buffer,
    async_set_power,
    async_setup_via_flow,
)
from tests.fixtures.learning.synth import ENTITY_GRID, SynthSpec

GRID_S = 30
REPO_ROOT = Path(__file__).resolve().parents[2]


def _pin_grid(entry: Any, effective_s: float = float(GRID_S)) -> None:
    """Publish a profile snapshot so the capture snap uses the given grid/events."""
    enabled, disabled = gate(effective_s, False)
    entry.runtime_data.profile_coordinator.async_set_updated_data(
        ProfileSnapshot(
            sources={},
            effective_s=effective_s,
            assumed=False,
            enabled=enabled,
            disabled=disabled,
        )
    )


async def _setup(
    hass: HomeAssistant, energy_manager: EnergyManager, std_prefs: dict[str, Any]
) -> tuple[Any, Any]:
    energy_manager.data = copy.deepcopy(std_prefs)
    async_set_power(hass, "sensor.grid_power", 500)
    async_set_power(hass, "sensor.solar_power", 0)
    async_set_power(hass, "sensor.battery_power", 0)
    async_set_power(hass, "sensor.washer_power", 0)
    entry = await async_setup_via_flow(hass)
    _pin_grid(entry)
    await hass.async_block_till_done()
    data = await async_fill_synth_buffer(hass, entry, SynthSpec(days=3))
    return entry, data


def _iso_utc(ms: int) -> str:
    return dt_util.utc_from_timestamp(ms / 1000.0).isoformat()


def _stored_captures(hass_storage: dict[str, Any]) -> list[dict[str, Any]]:
    return hass_storage.get(CAPTURES_STORE_KEY, {}).get("data", {}).get("captures", [])


async def _capture(hass: HomeAssistant, start: Any, end: Any) -> dict[str, Any]:
    return await hass.services.async_call(
        DOMAIN_CALL(),
        "capture_window",
        {"start": start, "end": end, "label": "k", "target_type": "new_device", "target": "K"},
        blocking=True,
        return_response=True,
    )


def DOMAIN_CALL() -> str:
    from custom_components.powerguesser.const import DOMAIN

    return DOMAIN


async def _insert_grid_anomaly(
    hass: HomeAssistant, entry: Any, ts_ms: int, *, delete: bool
) -> None:
    def _do() -> None:
        db = entry.runtime_data.writer.db
        rows = db.channel_rows()
        grid_ch = next(ch for ch, eid, _role, _kind in rows if eid == ENTITY_GRID)
        conn = sqlite3.connect(db.path)
        try:
            if delete:
                conn.execute(
                    "DELETE FROM samples WHERE ch = ? AND ts >= ? AND ts <= ?",
                    (grid_ch, ts_ms, ts_ms + 400_000),
                )
            else:
                conn.execute(
                    "INSERT OR REPLACE INTO samples(ts, ch, w) VALUES(?, ?, NULL)",
                    (ts_ms, grid_ch),
                )
            conn.commit()
        finally:
            conn.close()

    await hass.async_add_executor_job(_do)
    await hass.async_block_till_done()


async def test_refused_window_creates_no_record(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
) -> None:
    """Every refusal raises capture_<code> and leaves the Store untouched."""
    entry, data = await _setup(hass, energy_manager, std_prefs)
    on_ms, off_ms = data.kettle_runs[-1]
    now_ms = int(round(dt_util.utcnow().timestamp() * 1000))
    grid_rows = data.rows[ENTITY_GRID]
    buf_first = grid_rows[0][0]
    buf_last = grid_rows[-1][0]

    cases: list[tuple[str, Any, Any]] = [
        ("invalid_window", _iso_utc(on_ms), _iso_utc(on_ms)),
        ("future_end", _iso_utc(on_ms), _iso_utc(now_ms + 60_000)),
        ("window_too_long", _iso_utc(on_ms), _iso_utc(on_ms + 6 * 3_600_000 + 1000)),
        ("too_short_for_rate", _iso_utc(on_ms), _iso_utc(on_ms + GRID_S * 1000)),
        ("outside_buffer", _iso_utc(buf_first - 3_600_000), _iso_utc(buf_first - 3_000_000)),
    ]

    for code, start, end in cases:
        before = json.dumps(_stored_captures(hass_storage), sort_keys=True)
        with pytest.raises(ServiceValidationError) as err:
            await _capture(hass, start, end)
        assert err.value.translation_key == f"capture_{code}"
        assert json.dumps(_stored_captures(hass_storage), sort_keys=True) == before

    # interval_too_coarse: republish a 60 s grid so STEP is disabled.
    _pin_grid(entry, effective_s=60.0)
    await hass.async_block_till_done()
    before = json.dumps(_stored_captures(hass_storage), sort_keys=True)
    with pytest.raises(ServiceValidationError) as coarse:
        await _capture(hass, _iso_utc(on_ms), _iso_utc(off_ms))
    assert coarse.value.translation_key == "capture_interval_too_coarse"
    assert json.dumps(_stored_captures(hass_storage), sort_keys=True) == before
    _pin_grid(entry)

    # across_gap: a NULL gross row inside the window.
    slot = off_ms + 1_800_000
    await _insert_grid_anomaly(hass, entry, slot, delete=False)
    before = json.dumps(_stored_captures(hass_storage), sort_keys=True)
    with pytest.raises(ServiceValidationError) as gap:
        await _capture(hass, _iso_utc(slot - 300_000), _iso_utc(slot + 300_000))
    assert gap.value.translation_key == "capture_across_gap"
    assert json.dumps(_stored_captures(hass_storage), sort_keys=True) == before


async def test_every_refusal_code_has_a_message() -> None:
    """strings.json holds capture_<code> for every code and equals translations/en.json."""
    strings = (REPO_ROOT / "custom_components/powerguesser/strings.json").read_bytes()
    translations = (
        REPO_ROOT / "custom_components/powerguesser/translations/en.json"
    ).read_bytes()
    assert strings == translations

    data = json.loads(strings)
    exceptions = data.get("exceptions", {})
    for code in REFUSAL_CODES:
        key = f"capture_{code}"
        assert key in exceptions, f"missing exception {key}"
        assert exceptions[key].get("message")


async def test_naive_window_uses_ha_time_zone(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
) -> None:
    """A naive local window snaps the run; the echo shows local and UTC an hour apart."""
    await hass.config.async_set_time_zone("Europe/Berlin")
    entry, data = await _setup(hass, energy_manager, std_prefs)
    on_ms, off_ms = data.kettle_runs[-1]
    tz = ZoneInfo("Europe/Berlin")

    def _local_naive(ms: int) -> str:
        return (
            dt_util.utc_from_timestamp(ms / 1000.0)
            .astimezone(tz)
            .replace(tzinfo=None)
            .isoformat()
        )

    res = await _capture(hass, _local_naive(on_ms - 45_000), _local_naive(off_ms + 50_000))
    result = res["result"]
    assert result["verdict"] == "clean"
    assert abs(result["rise_ms"] - on_ms) <= 33_000
    assert abs(result["fall_ms"] - off_ms) <= 33_000
    # The echo carries the interpreted local and UTC window (winter: +01:00 apart).
    assert res["window"]["tz"] == "Europe/Berlin"
    assert res["window"]["start_local"].endswith("+01:00")
    assert res["window"]["start_utc"].endswith("+00:00")

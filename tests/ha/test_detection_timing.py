"""Tests for the Detection time diagnostic sensor and the D-34 Repairs issue.

Plan 04-04 Task 3 (DET-04, D-34): the per-tick detector cost and state are
visible on one diagnostic sensor, and a too-coarse sample interval raises a
non-fixable WARNING Repairs issue naming the interval detection needs.
"""

from typing import Any

import pytest

from homeassistant.components.energy.data import EnergyManager
from homeassistant.core import HomeAssistant
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers import issue_registry as ir

from custom_components.powerguesser.const import DOMAIN
from custom_components.powerguesser.engine.intervals import (
    EVENT_TYPE_MIN_S,
    EventType,
)
from custom_components.powerguesser.sensor import DetectionTimingSensor

from tests.ha.test_device_detection import (
    _device_uuid,
    _drive_ticks,
    _kettle_on,
    _publish_profile,
    _setup_confirmed,
    _state,
    make_switched_candidate,
)

TIMING_UNIQUE_ID = "powerguesser_detection_timing"
ISSUE_ID = "detection_interval_too_slow"

TIMING_ATTRIBUTES = {
    "p50_ms",
    "max_ms",
    "ticks",
    "devices",
    "detector",
    "unattributed_total",
    "unavailable_devices",
}


def _timing_eid(hass: HomeAssistant) -> str:
    entity_id = er.async_get(hass).async_get_entity_id(
        "sensor", DOMAIN, TIMING_UNIQUE_ID
    )
    assert entity_id is not None
    return entity_id


def _issue(hass: HomeAssistant) -> Any:
    return ir.async_get(hass).async_get_issue(DOMAIN, ISSUE_ID)


async def test_timing_sensor_reports_p95_after_ticks(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
) -> None:
    """Before the first tick the state is unknown; after ticks it is the p95 in ms."""
    entry, entities = await _setup_confirmed(
        hass,
        energy_manager,
        std_prefs,
        hass_storage,
        freezer,
        [make_switched_candidate("sig-kettle", "Kettle", rise_w=2000.0)],
    )
    timing_eid = _timing_eid(hass)
    assert _state(hass, timing_eid) == "unknown"

    power_eid, _energy_eid = entities["Kettle"]
    await _kettle_on(hass, freezer)
    assert _state(hass, power_eid) == "2000.0"
    summary = entry.runtime_data.devices.timing_summary()
    assert summary["ticks"] > 0
    assert float(_state(hass, timing_eid)) == summary["p95_ms"]


async def test_timing_attributes_unrecorded(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
) -> None:
    """The timing attributes are exactly the summary set and all unrecorded."""
    await _setup_confirmed(
        hass,
        energy_manager,
        std_prefs,
        hass_storage,
        freezer,
        [make_switched_candidate("sig-kettle", "Kettle", rise_w=2000.0)],
    )
    await _kettle_on(hass, freezer)
    state = hass.states.get(_timing_eid(hass))
    for key in TIMING_ATTRIBUTES:
        assert key in state.attributes
        assert key in DetectionTimingSensor._unrecorded_attributes


async def test_timing_reports_detector_state_and_unavailable_devices(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
) -> None:
    """detector reads running/no_devices/disabled_interval; unavailable_devices names reasons."""
    c_kettle = make_switched_candidate("sig-kettle", "Kettle", rise_w=2000.0)
    c_slow = make_switched_candidate("sig-slow", "Slow heater", rise_w=2000.0)
    c_slow["extraction"]["grid_s"] = 60
    entry, _entities = await _setup_confirmed(
        hass,
        energy_manager,
        std_prefs,
        hass_storage,
        freezer,
        [c_kettle, c_slow],
    )
    devices = entry.runtime_data.devices

    # Both devices confirmed at the pinned 30 s grid: the mismatched one is
    # named with its reason, the kettle is not listed.
    summary = devices.timing_summary()
    assert summary["detector"] == "running"
    assert summary["devices"] == 1
    assert summary["unavailable_devices"] == {"Slow heater": "grid_mismatch"}

    # A too-coarse interval disables the step type for every confirmed device.
    _publish_profile(entry, 60.0)
    await hass.async_block_till_done()
    summary = devices.timing_summary()
    assert summary["detector"] == "disabled_interval"
    assert summary["unavailable_devices"] == {
        "Kettle": "disabled_interval",
        "Slow heater": "disabled_interval",
    }

    # Back to 30 s: running again, nothing unavailable.
    _publish_profile(entry, 30.0)
    await hass.async_block_till_done()
    summary = devices.timing_summary()
    assert summary["detector"] == "running"
    assert summary["unavailable_devices"] == {"Slow heater": "grid_mismatch"}

    # no_devices: retire nothing here — a fresh runtime without confirmed
    # devices reports no_devices through detector_status.
    assert devices.detector_status == "running"


async def test_no_devices_status_before_confirm(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
) -> None:
    """Without a confirmed device the detector status reads no_devices."""
    entry, _entities = await _setup_confirmed(
        hass,
        energy_manager,
        std_prefs,
        hass_storage,
        freezer,
        [],
    )
    assert entry.runtime_data.devices.detector_status == "no_devices"


async def test_slow_sampling_raises_and_clears_repairs_issue(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
) -> None:
    """D-34: the detection_interval_too_slow issue tracks disabled confirmed devices."""
    entry, _entities = await _setup_confirmed(
        hass,
        energy_manager,
        std_prefs,
        hass_storage,
        freezer,
        [make_switched_candidate("sig-kettle", "Kettle", rise_w=2000.0)],
        skip_confirm=True,
    )

    # Too-coarse sampling with no confirmed device: no issue.
    _publish_profile(entry, 60.0)
    await hass.async_block_till_done()
    assert _issue(hass) is None

    # Confirm the kettle at 30 s, then publish the 60 s snapshot: the issue
    # appears with the derived limit and the affected device names.
    from tests.conftest import async_confirm_candidate

    await async_confirm_candidate(hass, entry, "sig-kettle", "Kettle")
    dev_uuid = _device_uuid(entry, "Kettle")
    ent_reg = er.async_get(hass)
    power_eid = ent_reg.async_get_entity_id(
        "sensor", DOMAIN, f"{entry.entry_id}_{dev_uuid}_power"
    )
    energy_eid = ent_reg.async_get_entity_id(
        "sensor", DOMAIN, f"{entry.entry_id}_{dev_uuid}_energy"
    )

    _publish_profile(entry, 30.0)
    await hass.async_block_till_done()
    assert _issue(hass) is None

    _publish_profile(entry, 60.0)
    await hass.async_block_till_done()
    issue = _issue(hass)
    assert issue is not None
    assert issue.severity == ir.IssueSeverity.WARNING
    assert issue.is_fixable is False
    assert issue.translation_key == ISSUE_ID
    placeholders = issue.translation_placeholders
    assert placeholders["interval"] == "60"
    expected_max = f"{EVENT_TYPE_MIN_S[EventType.STEP] / 2:g}"
    assert placeholders["max_interval"] == expected_max
    assert "Kettle" in placeholders["devices"]

    # The kettle's power sensor is unavailable; its energy sensor stays
    # available with its total (D-33).
    devices = entry.runtime_data.devices
    assert devices.unavailable_reason(_device_uuid(entry, "Kettle")) == "disabled_interval"
    assert _state(hass, power_eid) == "unavailable"
    assert _state(hass, energy_eid) == "0.0"

    # Publishing a 30 s profile deletes the issue and detection resumes.
    _publish_profile(entry, 30.0)
    await hass.async_block_till_done()
    assert _issue(hass) is None
    await _kettle_on(hass, freezer)
    assert _state(hass, power_eid) == "2000.0"

    # Publishing 60 s again re-creates the issue...
    _publish_profile(entry, 60.0)
    await hass.async_block_till_done()
    assert _issue(hass) is not None

    # ...and unloading the entry clears it.
    assert await hass.config_entries.async_unload(entry.entry_id)
    await hass.async_block_till_done()
    assert _issue(hass) is None

"""Tests for PowerGuesser opt-in weekly learning schedule."""

from datetime import datetime, timedelta
import json
from typing import Any

import pytest
from homeassistant.core import HomeAssistant
import homeassistant.util.dt as dt_util
from pytest_homeassistant.common import async_fire_time_changed

from custom_components.powerguesser.const import (
    CONF_SCHEDULE,
    CONF_SCHEDULE_TIME,
    CONF_SCHEDULE_WEEKDAY,
    DEFAULT_SCHEDULE,
)
from tests.conftest import (
    async_fill_synth_buffer,
    async_seed_consent,
    async_seed_llm,
    async_set_power,
    async_setup_via_flow,
    mock_llm,
)
from tests.fixtures.learning.fake_llm import chat_completion


def _set_std_powers(hass: HomeAssistant) -> None:
    async_set_power(hass, "sensor.grid_power", 500)
    async_set_power(hass, "sensor.solar_power", 1200)
    async_set_power(hass, "sensor.battery_power", -300)
    async_set_power(hass, "sensor.washer_power", 200)


async def _setup_ready_supervisor(
    hass: HomeAssistant,
    energy_manager: object,
    std_prefs: dict[str, Any],
    days: int = 22,
) -> Any:
    from tests.fixtures.learning.synth import SynthSpec

    energy_manager.data = std_prefs
    _set_std_powers(hass)
    entry = await async_setup_via_flow(hass)
    spec = SynthSpec(days=days, oven=True)
    await async_fill_synth_buffer(hass, entry, spec=spec)
    await async_seed_llm(hass, entry)
    await async_seed_consent(hass, host="llm.test", names=False)
    return entry


def _make_proposal(devices: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "analysis": "Test proposal.",
        "schema_version": 1,
        "devices": devices,
        "rejected_clusters": [],
        "merge_suggestions": [],
    }


def _make_device(ref: str, cluster: str) -> dict[str, Any]:
    return {
        "ref": ref,
        "label": f"Device {ref}",
        "category": "unknown",
        "tracker_kind": "switched",
        "member_clusters": [cluster],
        "states": [
            {
                "name": "on",
                "cluster_ref": cluster,
                "role": "main",
            }
        ],
        "phase_hint": "single",
        "confidence": "medium",
        "rationale": f"Device for {cluster}.",
        "ambiguous_with": [],
        "reference_plug": "",
    }


async def test_schedule_off_by_default(
    hass: HomeAssistant,
    energy_manager: object,
    std_prefs: dict[str, Any],
    aioclient_mock: object,
    inprocess_worker: object,
    freezer: Any,
) -> None:
    """With default options, firing time changes over a full week never starts a run."""
    entry = await _setup_ready_supervisor(hass, energy_manager, std_prefs)
    supervisor = entry.runtime_data.learning
    if supervisor._unsub_refresh:
        supervisor._unsub_refresh()
        supervisor._unsub_refresh = None

    assert entry.options.get(CONF_SCHEDULE, DEFAULT_SCHEDULE) == "off"

    prop = _make_proposal([_make_device("d1", "c01")])
    mock_llm(
        aioclient_mock,
        "http://llm.test/v1",
        lambda _r: {"status": 200, "json": chat_completion(json.dumps(prop))},
    )

    now = dt_util.now()
    for day in range(7):
        for hour in (2, 8, 14, 20):
            target = (now + timedelta(days=day)).replace(hour=hour, minute=30, second=0)
            freezer.move_to(target)
            async_fire_time_changed(hass, target)
            await hass.async_block_till_done()

    assert aioclient_mock.call_count == 0
    assert supervisor.run_task is None


async def test_weekly_schedule_triggers_at_configured_weekday_and_time(
    hass: HomeAssistant,
    energy_manager: object,
    std_prefs: dict[str, Any],
    aioclient_mock: object,
    inprocess_worker: object,
    freezer: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Schedule weekly on Tuesday 02:30 triggers on Tuesday 02:30 and not on Monday or Wednesday."""
    monkeypatch.setattr("custom_components.powerguesser.learning.MAX_REFINE_ROUNDS", 0)
    entry = await _setup_ready_supervisor(hass, energy_manager, std_prefs)
    supervisor = entry.runtime_data.learning
    if supervisor._unsub_refresh:
        supervisor._unsub_refresh()
        supervisor._unsub_refresh = None

    hass.config_entries.async_update_entry(
        entry,
        options={
            **entry.options,
            CONF_SCHEDULE: "weekly",
            CONF_SCHEDULE_WEEKDAY: "tue",
            CONF_SCHEDULE_TIME: "02:30:00",
        },
    )
    await hass.async_block_till_done()

    prop = _make_proposal([_make_device("d1", "c01")])
    mock_llm(
        aioclient_mock,
        "http://llm.test/v1",
        lambda _r: {"status": 200, "json": chat_completion(json.dumps(prop))},
    )

    now = dt_util.now()
    days_to_monday = (0 - now.weekday()) % 7 or 7  # never today: the slot may already be past
    monday = (now + timedelta(days=days_to_monday)).replace(hour=0, minute=0, second=0, microsecond=0)
    freezer.move_to(monday)
    async_fire_time_changed(hass, monday)
    await hass.async_block_till_done()

    # 1. Monday 02:30:00 -> no run
    mon_230 = monday.replace(hour=2, minute=30, second=0)
    freezer.move_to(mon_230)
    async_fire_time_changed(hass, mon_230)
    await hass.async_block_till_done()
    assert supervisor.run_task is None
    assert aioclient_mock.call_count == 0

    # 2. Tuesday 02:30:00 -> run triggers
    tue_230 = monday + timedelta(days=1, hours=2, minutes=30)
    freezer.move_to(tue_230)
    async_fire_time_changed(hass, tue_230)
    await hass.async_block_till_done(wait_background_tasks=True)

    # Assert the trigger, not an exact HTTP total: since Plan 05 a run may make a
    # second call for the masked shadow pass (present or skipped depending on
    # usable days), so the count is 1 or 2 here.
    after_tue = aioclient_mock.call_count
    assert after_tue >= 1
    assert supervisor.coordinator.data["status"] == "idle"

    # 3. Wednesday 02:30:00 -> no run
    wed_230 = monday + timedelta(days=2, hours=2, minutes=30)
    freezer.move_to(wed_230)
    async_fire_time_changed(hass, wed_230)
    await hass.async_block_till_done()
    # Still no new call
    assert aioclient_mock.call_count == after_tue


async def test_schedule_rearms_on_time_change_and_removes_on_off(
    hass: HomeAssistant,
    energy_manager: object,
    std_prefs: dict[str, Any],
    aioclient_mock: object,
    inprocess_worker: object,
    freezer: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Changing time re-arms the schedule; switching to off removes the schedule."""
    monkeypatch.setattr("custom_components.powerguesser.learning.MAX_REFINE_ROUNDS", 0)
    entry = await _setup_ready_supervisor(hass, energy_manager, std_prefs)
    supervisor = entry.runtime_data.learning
    if supervisor._unsub_refresh:
        supervisor._unsub_refresh()
        supervisor._unsub_refresh = None

    hass.config_entries.async_update_entry(
        entry,
        options={
            **entry.options,
            CONF_SCHEDULE: "weekly",
            CONF_SCHEDULE_WEEKDAY: "tue",
            CONF_SCHEDULE_TIME: "02:30:00",
        },
    )
    await hass.async_block_till_done()

    prop = _make_proposal([_make_device("d1", "c01")])
    mock_llm(
        aioclient_mock,
        "http://llm.test/v1",
        lambda _r: {"status": 200, "json": chat_completion(json.dumps(prop))},
    )

    # Change time to 04:15:00
    hass.config_entries.async_update_entry(
        entry,
        options={
            **entry.options,
            CONF_SCHEDULE: "weekly",
            CONF_SCHEDULE_WEEKDAY: "tue",
            CONF_SCHEDULE_TIME: "04:15:00",
        },
    )
    await hass.async_block_till_done()

    now = dt_util.now()
    days_to_tue = (1 - now.weekday()) % 7 or 7  # never today: the slot may already be past
    tue = (now + timedelta(days=days_to_tue)).replace(hour=0, minute=0, second=0, microsecond=0)

    # Old time 02:30:00 does not trigger
    tue_230 = tue.replace(hour=2, minute=30)
    freezer.move_to(tue_230)
    async_fire_time_changed(hass, tue_230)
    await hass.async_block_till_done()
    assert supervisor.run_task is None

    # New time 04:15:00 triggers
    tue_415 = tue.replace(hour=4, minute=15)
    freezer.move_to(tue_415)
    async_fire_time_changed(hass, tue_415)
    await hass.async_block_till_done(wait_background_tasks=True)
    # Assert the trigger, not an exact HTTP total: a run may also make a masked
    # shadow-pass call (Plan 05), so the count is 1 or 2 here.
    after = aioclient_mock.call_count
    assert after >= 1
    assert supervisor.coordinator.data["status"] == "idle"

    # Switch to off
    hass.config_entries.async_update_entry(
        entry,
        options={
            **entry.options,
            CONF_SCHEDULE: "off",
        },
    )
    await hass.async_block_till_done()

    # Next Tuesday at 04:15 does not trigger
    next_tue_415 = tue_415 + timedelta(days=7)
    freezer.move_to(next_tue_415)
    async_fire_time_changed(hass, next_tue_415)
    await hass.async_block_till_done()
    # No new call
    assert aioclient_mock.call_count == after


async def test_schedule_skips_when_blocked_or_capped(
    hass: HomeAssistant,
    energy_manager: object,
    std_prefs: dict[str, Any],
    aioclient_mock: object,
    freezer: Any,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A scheduled trigger while learning is blocked or capped sends nothing and leaves status unchanged."""
    entry = await _setup_ready_supervisor(hass, energy_manager, std_prefs)
    supervisor = entry.runtime_data.learning
    if supervisor._unsub_refresh:
        supervisor._unsub_refresh()
        supervisor._unsub_refresh = None

    hass.config_entries.async_update_entry(
        entry,
        options={
            **entry.options,
            CONF_SCHEDULE: "weekly",
            CONF_SCHEDULE_WEEKDAY: "tue",
            CONF_SCHEDULE_TIME: "02:30:00",
        },
    )
    await hass.async_block_till_done()

    prop = _make_proposal([_make_device("d1", "c01")])
    mock_llm(
        aioclient_mock,
        "http://llm.test/v1",
        lambda _r: {"status": 200, "json": chat_completion(json.dumps(prop))},
    )

    now = dt_util.now()
    days_to_tue = (1 - now.weekday()) % 7 or 7  # never today: the slot may already be past
    tue_230 = (now + timedelta(days=days_to_tue)).replace(hour=2, minute=30, second=0, microsecond=0)

    # Set status to capped
    c_data = dict(supervisor.coordinator.data)
    c_data["status"] = "capped"
    c_data["last_error"] = "cap_reached: month_input"
    supervisor.coordinator.data = c_data

    freezer.move_to(tue_230)
    async_fire_time_changed(hass, tue_230)
    await hass.async_block_till_done()

    # Did not send request
    assert aioclient_mock.call_count == 0
    assert supervisor.coordinator.data["status"] == "capped"
    assert supervisor.run_task is None
    # No error logged
    assert not any(r.levelname == "ERROR" for r in caplog.records)


async def test_run_cap_does_not_suppress_schedule_or_settings(
    hass: HomeAssistant,
    energy_manager: object,
    std_prefs: dict[str, Any],
    aioclient_mock: object,
    inprocess_worker: object,
    freezer: Any,
) -> None:
    """A per-run cap is not sticky: the next weekly slot runs, and a settings change clears it (WR-01)."""
    entry = await _setup_ready_supervisor(hass, energy_manager, std_prefs)
    supervisor = entry.runtime_data.learning
    if supervisor._unsub_refresh:
        supervisor._unsub_refresh()
        supervisor._unsub_refresh = None

    # A settings change clears a run-cap "capped" and re-evaluates the gates
    c_data = dict(supervisor.coordinator.data)
    c_data["status"] = "capped"
    c_data["last_error"] = "cap_reached: run_input"
    supervisor.coordinator.data = c_data
    await supervisor.async_settings_changed()
    assert supervisor.coordinator.data["status"] == "idle"
    assert supervisor.coordinator.data["last_error"] is None

    # A month cap stays capped across a settings change
    c_data = dict(supervisor.coordinator.data)
    c_data["status"] = "capped"
    c_data["last_error"] = "cap_reached: month_input"
    supervisor.coordinator.data = c_data
    await supervisor.async_settings_changed()
    assert supervisor.coordinator.data["status"] == "capped"

    hass.config_entries.async_update_entry(
        entry,
        options={
            **entry.options,
            CONF_SCHEDULE: "weekly",
            CONF_SCHEDULE_WEEKDAY: "tue",
            CONF_SCHEDULE_TIME: "02:30:00",
        },
    )
    await hass.async_block_till_done()

    prop = _make_proposal([_make_device("d1", "c01")])
    mock_llm(
        aioclient_mock,
        "http://llm.test/v1",
        lambda _r: {"status": 200, "json": chat_completion(json.dumps(prop))},
    )

    # The options update above re-ran the gates; force the run-cap state again
    c_data = dict(supervisor.coordinator.data)
    c_data["status"] = "capped"
    c_data["last_error"] = "cap_reached: run_output"
    supervisor.coordinator.data = c_data

    now = dt_util.now()
    days_to_monday = (0 - now.weekday()) % 7 or 7  # never today: the slot may already be past
    monday = (now + timedelta(days=days_to_monday)).replace(hour=0, minute=0, second=0, microsecond=0)
    freezer.move_to(monday)
    async_fire_time_changed(hass, monday)
    await hass.async_block_till_done()
    assert supervisor.run_task is None

    calls_before = aioclient_mock.call_count
    tue_230 = monday + timedelta(days=1, hours=2, minutes=30)
    freezer.move_to(tue_230)
    async_fire_time_changed(hass, tue_230)
    await hass.async_block_till_done(wait_background_tasks=True)

    # 04-12 (T-04-25) releases the background slot in the run finale, so
    # run_task is None again by the time the run finishes; the WR-01 contract
    # — the weekly slot starts a run despite the per-run cap — is proven by
    # the fresh LLM request.
    assert aioclient_mock.call_count > calls_before


async def test_schedule_skips_when_run_is_active(
    hass: HomeAssistant,
    energy_manager: object,
    std_prefs: dict[str, Any],
    aioclient_mock: object,
    freezer: Any,
) -> None:
    """A scheduled trigger while a run is active does not start another run."""
    entry = await _setup_ready_supervisor(hass, energy_manager, std_prefs)
    supervisor = entry.runtime_data.learning
    if supervisor._unsub_refresh:
        supervisor._unsub_refresh()
        supervisor._unsub_refresh = None

    hass.config_entries.async_update_entry(
        entry,
        options={
            **entry.options,
            CONF_SCHEDULE: "weekly",
            CONF_SCHEDULE_WEEKDAY: "tue",
            CONF_SCHEDULE_TIME: "02:30:00",
        },
    )
    await hass.async_block_till_done()

    import asyncio
    hang_future: asyncio.Future[None] = asyncio.Future()

    async def _hanging_run(_trigger: str) -> None:
        await hang_future

    fake_task = entry.async_create_background_task(hass, _hanging_run("active"), "fake_task")
    supervisor.run_task = fake_task

    now = dt_util.now()
    days_to_tue = (1 - now.weekday()) % 7 or 7  # never today: the slot may already be past
    tue_230 = (now + timedelta(days=days_to_tue)).replace(hour=2, minute=30, second=0, microsecond=0)

    freezer.move_to(tue_230)
    async_fire_time_changed(hass, tue_230)
    await hass.async_block_till_done()

    assert supervisor.run_task is fake_task
    hang_future.set_result(None)
    await fake_task


async def test_options_save_with_phase_change_still_rearms_schedule(
    hass: HomeAssistant,
    energy_manager: object,
    std_prefs: dict[str, Any],
) -> None:
    """An options save that changes phases and the schedule rebuilds and re-arms (WR-10)."""
    from unittest.mock import patch

    from custom_components.powerguesser.const import CONF_PHASE_L1

    entry = await _setup_ready_supervisor(hass, energy_manager, std_prefs)
    supervisor = entry.runtime_data.learning
    assert supervisor._unsub_schedule is None
    async_set_power(hass, "sensor.l1_power", 500)

    with patch.object(
        entry.runtime_data, "async_request_rebuild", wraps=entry.runtime_data.async_request_rebuild
    ) as rebuild:
        hass.config_entries.async_update_entry(
            entry,
            options={
                **entry.options,
                CONF_PHASE_L1: "sensor.l1_power",
                CONF_SCHEDULE: "weekly",
                CONF_SCHEDULE_WEEKDAY: "tue",
                CONF_SCHEDULE_TIME: "02:30:00",
            },
        )
        await hass.async_block_till_done()

    assert rebuild.call_count == 1
    assert supervisor._unsub_schedule is not None

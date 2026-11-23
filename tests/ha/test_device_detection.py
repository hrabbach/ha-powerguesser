"""Tests for confirmed-device detection driving power, energy and remainder.

Plan 04-04: a confirmed switching load drives its own power and energy
sensors and lowers the remainder through the real hub path (DET-01, OUT-01,
OUT-02, OUT-03, D-06, D-08, D-09). Every test pins the D-26 grid explicitly
by publishing a 30 s ProfileSnapshot through hub.profile_coordinator.
"""

from datetime import timedelta
import math
from typing import Any

import pytest

from homeassistant.components.energy.data import EnergyManager
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers import entity_registry as er
from homeassistant.util import dt as dt_util
from pytest_homeassistant.common import async_fire_time_changed

from custom_components.powerguesser.const import DOMAIN
from custom_components.powerguesser.engine.intervals import ProfileSnapshot, gate

from tests.conftest import (
    async_confirm_candidate,
    async_set_power,
    async_setup_via_flow,
    make_switched_candidate,
    seed_candidates,
)

GRID_S = 30


def _aligned_move(freezer: Any) -> None:
    """Move the freezer to the next epoch multiple of the 30 s grid."""
    now = dt_util.utcnow().timestamp()
    aligned = math.ceil(now / GRID_S) * GRID_S
    freezer.move_to(dt_util.utc_from_timestamp(aligned))


def _pin_grid(entry: ConfigEntry) -> None:
    """Publish a 30 s profile snapshot so the D-26 grid is pinned at 30."""
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


async def _tick(hass: HomeAssistant, freezer: Any, seconds: int = GRID_S) -> None:
    """Advance the freezer and fire the time-changed event.

    fire_all=False: only listeners whose interval matches the new time fire,
    so the aligned 30 s ticks drive the 10 s hub tick without also firing the
    6-hourly learning refresh or daily retention timers.
    """
    freezer.tick(timedelta(seconds=seconds))
    async_fire_time_changed(hass, dt_util.utcnow())
    await hass.async_block_till_done()


async def _setup_confirmed(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any | None,
    candidates: list[dict[str, Any]],
    skip_confirm: bool = False,
) -> tuple[ConfigEntry, dict[str, tuple[str, str]]]:
    """Set up the entry, pin the grid at 30 s and confirm every candidate.

    Returns (entry, {label: (power_entity_id, energy_entity_id)}); the entity
    map is empty when skip_confirm is set (confirm later in the test). Pass
    freezer=None to skip the clock alignment (real-clock tests).
    """
    seed_candidates(hass_storage, candidates)
    energy_manager.data = std_prefs
    async_set_power(hass, "sensor.grid_power", 500)
    async_set_power(hass, "sensor.solar_power", 0)
    async_set_power(hass, "sensor.battery_power", 0)
    async_set_power(hass, "sensor.washer_power", 200)
    if freezer is not None:
        _aligned_move(freezer)
    entry = await async_setup_via_flow(hass)
    _pin_grid(entry)
    await hass.async_block_till_done()

    ent_reg = er.async_get(hass)
    entities: dict[str, tuple[str, str]] = {}
    if skip_confirm:
        return entry, entities
    for cand in candidates:
        label = str(cand["label"])
        await async_confirm_candidate(hass, entry, str(cand["signature_id"]), label)
        sub = next(s for s in entry.get_subentries_of_type("device") if s.title == label)
        dev_uuid = sub.data["device_uuid"]
        power_eid = ent_reg.async_get_entity_id(
            "sensor", DOMAIN, f"{entry.entry_id}_{dev_uuid}_power"
        )
        energy_eid = ent_reg.async_get_entity_id(
            "sensor", DOMAIN, f"{entry.entry_id}_{dev_uuid}_energy"
        )
        assert power_eid is not None and energy_eid is not None
        entities[label] = (power_eid, energy_eid)
    return entry, entities


async def _drive_ticks(hass: HomeAssistant, freezer: Any, n: int) -> None:
    """Advance n full grid steps via the 10 s tick path."""
    for _ in range(n * 3):
        await _tick(hass, freezer, 10)


async def test_confirmed_kettle_drives_power_energy_and_remainder(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
) -> None:
    """A confirmed kettle drives its power/energy sensors and lowers the remainder."""
    _entry, entities = await _setup_confirmed(
        hass,
        energy_manager,
        std_prefs,
        hass_storage,
        freezer,
        [make_switched_candidate("sig-kettle", "Kettle", rise_w=2000.0)],
    )
    power_eid, energy_eid = entities["Kettle"]
    remainder_eid = "sensor.powerguesser_remainder_power"
    assert hass.states.get(power_eid).state == "0.0"

    remainder_before = float(hass.states.get(remainder_eid).state)

    # Kettle on: the step is attributed one grid sample after the edge.
    async_set_power(hass, "sensor.grid_power", 2500)
    await hass.async_block_till_done()
    await _drive_ticks(hass, freezer, 5)
    power_state = hass.states.get(power_eid)
    assert power_state.state == "2000.0"
    remainder = hass.states.get(remainder_eid)
    # The remainder keeps carrying the base load; the kettle moved out of it.
    assert float(remainder.state) == pytest.approx(remainder_before, abs=1.0)
    assert remainder.attributes["estimated_w"] == pytest.approx(2000.0, abs=0.5)

    # Kettle off: power returns to 0.0 and the energy total grew by
    # roughly 2000 W x run time.
    async_set_power(hass, "sensor.grid_power", 500)
    await hass.async_block_till_done()
    await _drive_ticks(hass, freezer, 5)
    assert hass.states.get(power_eid).state == "0.0"
    energy_kwh = float(hass.states.get(energy_eid).state)
    # ~90-150 s at 2000 W is 0.05-0.09 kWh; allow the grid-alignment window.
    assert 0.02 < energy_kwh < 0.2


async def test_unmatched_rise_stays_in_remainder(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
) -> None:
    """A 600 W rise matches no confirmed device: power stays 0.0, remainder carries it."""
    _entry, entities = await _setup_confirmed(
        hass,
        energy_manager,
        std_prefs,
        hass_storage,
        freezer,
        [make_switched_candidate("sig-kettle", "Kettle", rise_w=2000.0)],
    )
    power_eid, _energy_eid = entities["Kettle"]
    remainder_eid = "sensor.powerguesser_remainder_power"
    remainder_before = float(hass.states.get(remainder_eid).state)

    # 600 W rise: above the 120 W edge threshold, outside the kettle's box.
    async_set_power(hass, "sensor.grid_power", 1100)
    await hass.async_block_till_done()
    await _drive_ticks(hass, freezer, 5)
    assert hass.states.get(power_eid).state == "0.0"
    remainder = hass.states.get(remainder_eid)
    assert float(remainder.state) == pytest.approx(remainder_before + 600.0, abs=1.0)
    assert remainder.attributes["estimated_w"] == 0.0


async def test_remainder_never_negative_and_exposes_estimated_w(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
) -> None:
    """Estimated devices are clamped to the residual; the remainder never goes below 0."""
    _entry, entities = await _setup_confirmed(
        hass,
        energy_manager,
        std_prefs,
        hass_storage,
        freezer,
        [make_switched_candidate("sig-kettle", "Kettle", rise_w=2000.0)],
    )
    power_eid, _energy_eid = entities["Kettle"]
    remainder_eid = "sensor.powerguesser_remainder_power"

    published: list[float] = []

    def _collect(event: Any) -> None:
        if event.data.get("entity_id") == remainder_eid:
            new_s = event.data.get("new_state")
            if new_s and new_s.state not in ("unavailable", "unknown"):
                published.append(float(new_s.state))

    unsub = hass.bus.async_listen("state_changed", _collect)

    # Kettle on: the estimate moves out of the remainder, which keeps
    # carrying the base load and never goes negative.
    async_set_power(hass, "sensor.grid_power", 2500)
    await hass.async_block_till_done()
    await _drive_ticks(hass, freezer, 5)
    assert hass.states.get(power_eid).state == "2000.0"
    remainder = hass.states.get(remainder_eid)
    assert float(remainder.state) >= 0.0
    assert remainder.attributes["estimated_w"] == pytest.approx(2000.0, abs=0.5)

    # Metered exceeds gross: the published remainder floors at 0 and the
    # estimate clamps at the (negative) residual instead of pushing the
    # remainder below zero (D-06).
    async_set_power(hass, "sensor.washer_power", 5000)
    await hass.async_block_till_done()
    async_set_power(hass, "sensor.grid_power", 2600)
    await hass.async_block_till_done()
    await _drive_ticks(hass, freezer, 4)
    remainder = hass.states.get(remainder_eid)
    assert float(remainder.state) == 0.0
    assert remainder.attributes["estimated_w"] == 0.0
    assert all(value >= 0.0 for value in published)
    unsub()


# -- Task 2: D-26 grid, availability reasons, gaps, stuck guard (TDD RED) ---


def _device_uuid(entry: ConfigEntry, label: str) -> str:
    """Return the device uuid for a confirmed device title."""
    sub = next(s for s in entry.get_subentries_of_type("device") if s.title == label)
    return sub.data["device_uuid"]


def _publish_profile(
    entry: ConfigEntry,
    effective_s: float,
    enabled: tuple | None = None,
    assumed: bool = False,
) -> None:
    """Publish a ProfileSnapshot through the hub profile coordinator."""
    if enabled is None:
        enabled, disabled = gate(effective_s, assumed)
    else:
        disabled = {}
    entry.runtime_data.profile_coordinator.async_set_updated_data(
        ProfileSnapshot(
            sources={},
            effective_s=float(effective_s),
            assumed=assumed,
            enabled=tuple(enabled),
            disabled=disabled,
        )
    )


async def _kettle_on(hass: HomeAssistant, freezer: Any) -> None:
    """Turn the kettle on and let the detector attribute it."""
    async_set_power(hass, "sensor.grid_power", 2500)
    await hass.async_block_till_done()
    await _drive_ticks(hass, freezer, 5)


async def _kettle_off(hass: HomeAssistant, freezer: Any) -> None:
    """Turn the kettle off and let the detector finish the episode."""
    async_set_power(hass, "sensor.grid_power", 500)
    await hass.async_block_till_done()
    await _drive_ticks(hass, freezer, 5)


def _state(hass: HomeAssistant, entity_id: str) -> str:
    """Return the state string of an entity."""
    state = hass.states.get(entity_id)
    assert state is not None
    return state.state


async def test_grid_mismatch_device_unavailable(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
) -> None:
    """D-26/D-32: an artefact learned on another grid is not modelled; only its power is unavailable."""
    c_kettle = make_switched_candidate("sig-kettle", "Kettle", rise_w=2000.0)
    c_slow = make_switched_candidate("sig-slow", "Slow heater", rise_w=2000.0)
    c_slow["extraction"]["grid_s"] = 60
    _entry, entities = await _setup_confirmed(
        hass,
        energy_manager,
        std_prefs,
        hass_storage,
        freezer,
        [c_kettle, c_slow],
    )
    kettle_power, kettle_energy = entities["Kettle"]
    slow_power, slow_energy = entities["Slow heater"]
    entry = hass.config_entries.async_entries("powerguesser")[0]
    devices = entry.runtime_data.devices

    assert devices.unavailable_reason(_device_uuid(entry, "Slow heater")) == "grid_mismatch"
    assert _state(hass, slow_power) == "unavailable"
    assert _state(hass, slow_energy) == "0.0"

    # The kettle still detects; the mismatched device's energy stays frozen.
    await _kettle_on(hass, freezer)
    assert _state(hass, kettle_power) == "2000.0"
    assert devices.unavailable_reason(_device_uuid(entry, "Slow heater")) == "grid_mismatch"
    assert _state(hass, slow_power) == "unavailable"
    assert _state(hass, slow_energy) == "0.0"
    await _kettle_off(hass, freezer)
    assert _state(hass, kettle_power) == "0.0"
    assert float(_state(hass, kettle_energy)) > 0.0
    assert _state(hass, slow_energy) == "0.0"


async def test_effective_interval_change_rebuilds_grid(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
) -> None:
    """D-26/D-29a/D-32: an interval change rebuilds the detector; ON devices keep the unknown flag."""
    entry, entities = await _setup_confirmed(
        hass,
        energy_manager,
        std_prefs,
        hass_storage,
        freezer,
        [make_switched_candidate("sig-kettle", "Kettle", rise_w=2000.0)],
    )
    power_eid, energy_eid = entities["Kettle"]
    devices = entry.runtime_data.devices
    kettle_uuid = _device_uuid(entry, "Kettle")

    await _kettle_on(hass, freezer)
    assert _state(hass, power_eid) == "2000.0"
    energy_before = _state(hass, energy_eid)

    # 30 s -> 15 s: the kettle's 30 s artefact no longer matches the grid.
    _publish_profile(entry, 15.0)
    await hass.async_block_till_done()
    assert devices.unavailable_reason(kettle_uuid) == "grid_mismatch"
    assert _state(hass, power_eid) == "unavailable"
    assert _state(hass, energy_eid) == energy_before

    # Kettle keeps running while unmodelled: no energy, flat total (D-32).
    await _drive_ticks(hass, freezer, 4)
    assert _state(hass, power_eid) == "unavailable"
    assert _state(hass, energy_eid) == energy_before

    # 15 s -> 30 s again while the kettle is still on: the flag survived the
    # unmodelled period, so the power sensor reads state_unknown, never 0 W.
    _publish_profile(entry, 30.0)
    await hass.async_block_till_done()
    assert devices.unavailable_reason(kettle_uuid) == "state_unknown"
    assert _state(hass, power_eid) == "unavailable"
    assert _state(hass, energy_eid) == energy_before

    # The kettle's fall ends the unknown state at 0 W (D-29c).
    await _kettle_off(hass, freezer)
    assert _state(hass, power_eid) == "0.0"
    assert _state(hass, energy_eid) == energy_before

    # A new run is detected again.
    await _kettle_on(hass, freezer)
    assert _state(hass, power_eid) == "2000.0"

    # Same-grid rebuild with SHORT_EVENT toggled while ON: the device enters
    # the unknown state (a rebuild is no evidence the device went off).
    from custom_components.powerguesser.engine.intervals import EventType

    _publish_profile(
        entry,
        30.0,
        enabled=(
            EventType.SHORT_EVENT,
            EventType.STEP,
            EventType.CYCLE_SHAPE,
            EventType.CONTINUOUS,
        ),
    )
    await hass.async_block_till_done()
    assert devices.unavailable_reason(kettle_uuid) == "state_unknown"
    assert _state(hass, power_eid) == "unavailable"
    energy_during_unknown = _state(hass, energy_eid)
    await _drive_ticks(hass, freezer, 3)
    assert _state(hass, energy_eid) == energy_during_unknown

    # The run's fall makes the power sensor 0.0 (D-29c).
    await _kettle_off(hass, freezer)
    assert _state(hass, power_eid) == "0.0"


async def test_model_error_marks_only_that_device(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
) -> None:
    """D-33: a corrupt artefact marks only that device model_error; the other keeps detecting."""
    c_kettle = make_switched_candidate("sig-kettle", "Kettle", rise_w=2000.0)
    c_corrupt = make_switched_candidate("sig-corrupt", "Broken", rise_w=2000.0)
    c_corrupt["states"][0]["rise_w"] = {"lo": 3000.0, "p50": 2000.0, "hi": 2200.0}
    entry, entities = await _setup_confirmed(
        hass,
        energy_manager,
        std_prefs,
        hass_storage,
        freezer,
        [c_kettle, c_corrupt],
    )
    kettle_power, _kettle_energy = entities["Kettle"]
    broken_power, broken_energy = entities["Broken"]
    devices = entry.runtime_data.devices

    assert devices.unavailable_reason(_device_uuid(entry, "Broken")) == "model_error"
    assert _state(hass, broken_power) == "unavailable"
    assert _state(hass, broken_energy) == "0.0"

    await _kettle_on(hass, freezer)
    assert _state(hass, kettle_power) == "2000.0"
    assert _state(hass, broken_power) == "unavailable"
    assert _state(hass, broken_energy) == "0.0"
    assert devices.unavailable_reason(_device_uuid(entry, "Kettle")) is None


async def test_step_disabled_power_unavailable_energy_kept(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
) -> None:
    """D-33: a too-coarse interval disables the step type: power unavailable, energy frozen."""
    entry, entities = await _setup_confirmed(
        hass,
        energy_manager,
        std_prefs,
        hass_storage,
        freezer,
        [make_switched_candidate("sig-kettle", "Kettle", rise_w=2000.0)],
    )
    power_eid, energy_eid = entities["Kettle"]
    devices = entry.runtime_data.devices

    await _kettle_on(hass, freezer)
    await _kettle_off(hass, freezer)
    energy_before = _state(hass, energy_eid)
    assert float(energy_before) > 0.0

    # 60 s hub interval: the event-type gate disables STEP (D-34 gate).
    _publish_profile(entry, 60.0)
    await hass.async_block_till_done()
    assert devices.unavailable_reason(_device_uuid(entry, "Kettle")) == "disabled_interval"
    assert _state(hass, power_eid) == "unavailable"
    assert _state(hass, energy_eid) == energy_before

    # A further kettle run is not estimated: the remainder carries the whole
    # step and the energy total stays frozen.
    await _kettle_on(hass, freezer)
    assert _state(hass, power_eid) == "unavailable"
    assert _state(hass, energy_eid) == energy_before
    remainder = hass.states.get("sensor.powerguesser_remainder_power")
    assert float(remainder.state) == pytest.approx(2300.0, abs=1.0)
    assert remainder.attributes["estimated_w"] == 0.0
    assert devices.detector_status == "disabled_interval"


async def test_detection_without_llm(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
) -> None:
    """LLM-06: with no LLM configured, detection works exactly the same."""
    entry, entities = await _setup_confirmed(
        hass,
        energy_manager,
        std_prefs,
        hass_storage,
        freezer,
        [make_switched_candidate("sig-kettle", "Kettle", rise_w=2000.0)],
    )
    power_eid, energy_eid = entities["Kettle"]
    assert not any(
        key.startswith("llm_") for key in entry.data
    ), "test expects no LLM configuration"

    await _kettle_on(hass, freezer)
    assert _state(hass, power_eid) == "2000.0"
    await _kettle_off(hass, freezer)
    assert _state(hass, power_eid) == "0.0"
    assert float(_state(hass, energy_eid)) > 0.0


@pytest.mark.slow
async def test_detection_survives_failing_learning_run(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    aioclient_mock: Any,
    inprocess_worker: Any,
) -> None:
    """LLM-06/D-10: a failing learning run never blocks or disturbs detection."""
    from freezegun import freeze_time

    from tests.conftest import (
        async_fill_synth_buffer,
        async_seed_consent,
        async_seed_llm,
        mock_llm,
    )

    # The learning run recomputes usable days through an executor job inside
    # its background task; that await deadlocks while freezegun is active
    # (lost wakeup). Keep this test on the real clock and freeze time only
    # for the detection phase at the end (no pytest-freezer fixture).
    entry, entities = await _setup_confirmed(
        hass,
        energy_manager,
        std_prefs,
        hass_storage,
        None,
        [make_switched_candidate("sig-kettle", "Kettle", rise_w=2000.0)],
    )
    power_eid, energy_eid = entities["Kettle"]

    await async_fill_synth_buffer(hass, entry)
    await async_seed_llm(hass, entry)
    await async_seed_consent(hass, host="llm.test", names=False)
    mock_llm(aioclient_mock, "http://llm.test/v1", lambda _data: {"exc": "connect"})

    await hass.services.async_call(DOMAIN, "learn", blocking=True)
    supervisor = entry.runtime_data.learning
    if supervisor.run_task is not None:
        await supervisor.run_task
    assert supervisor.coordinator.data["status"] == "failed"

    # The kettle still drives its sensors while and after the run fails.
    now = dt_util.utcnow().timestamp()
    aligned = math.ceil(now / GRID_S) * GRID_S
    with freeze_time(dt_util.utc_from_timestamp(aligned)) as frozen:
        await _kettle_on(hass, frozen)
        assert _state(hass, power_eid) == "2000.0"
        await _kettle_off(hass, frozen)
        assert _state(hass, power_eid) == "0.0"
        assert float(_state(hass, energy_eid)) > 0.0
        # Drain the timer burst the real->frozen monotonic jump queued
        # (flush, retention) so nothing touches the buffer after teardown.
        # The learning phase is done: stop its periodic usable-days refresh
        # so a fire-all tick cannot schedule a coverage read that lands in
        # the executor after the buffer closes.
        entry.runtime_data.learning._unsub_refresh()
        from tests.conftest import async_flush_buffer

        await async_flush_buffer(hass, entry)
        await hass.async_block_till_done()
        await hass.async_block_till_done()
    # Leaving the frozen phase jumps the monotonic clock back, making the
    # frozen-phase timers due again: drain that burst while the buffer is
    # still open.
    await hass.async_block_till_done()
    await hass.async_block_till_done()


async def test_unavailable_sources_feed_gap_rule(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
) -> None:
    """D-29a/OUT-01: a source gap longer than the hold puts an ON device into state_unknown."""
    entry, entities = await _setup_confirmed(
        hass,
        energy_manager,
        std_prefs,
        hass_storage,
        freezer,
        [make_switched_candidate("sig-kettle", "Kettle", rise_w=2000.0)],
    )
    power_eid, energy_eid = entities["Kettle"]
    devices = entry.runtime_data.devices
    kettle_uuid = _device_uuid(entry, "Kettle")

    await _kettle_on(hass, freezer)
    assert _state(hass, power_eid) == "2000.0"
    energy_before = _state(hass, energy_eid)

    # Gross source goes away for longer than GAP_HOLD_STEPS grid steps.
    async_set_power(hass, "sensor.grid_power", "unavailable")
    await hass.async_block_till_done()
    await _drive_ticks(hass, freezer, 4)
    assert devices.unavailable_reason(kettle_uuid) == "state_unknown"
    assert _state(hass, power_eid) == "unavailable"
    assert _state(hass, energy_eid) == energy_before
    # The blind publishes carry no estimate (signed residual is None).
    assert entry.runtime_data.coordinator.data.estimated_w == 0.0

    # Source returns with the kettle still on: no rise edge is seen (it
    # happened inside the blind gap), so the device stays unknown and the
    # remainder carries it.
    async_set_power(hass, "sensor.grid_power", 2500)
    await hass.async_block_till_done()
    await _drive_ticks(hass, freezer, 4)
    assert _state(hass, power_eid) == "unavailable"
    remainder = hass.states.get("sensor.powerguesser_remainder_power")
    assert float(remainder.state) == pytest.approx(2300.0, abs=1.0)
    assert remainder.attributes["estimated_w"] == 0.0
    assert _state(hass, energy_eid) == energy_before

    # The kettle's fall ends the unknown state at 0 W (D-29c).
    await _kettle_off(hass, freezer)
    assert _state(hass, power_eid) == "0.0"

    # A new run is detected again.
    await _kettle_on(hass, freezer)
    assert _state(hass, power_eid) == "2000.0"


async def test_unknown_after_gap_ends_on_sustained_low_remainder(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
) -> None:
    """D-30a: a device that switched off during a gap returns to 0 W on a sustained low remainder."""
    from custom_components.powerguesser.engine.detect.online import LOW_RESIDUAL_STEPS

    entry, entities = await _setup_confirmed(
        hass,
        energy_manager,
        std_prefs,
        hass_storage,
        freezer,
        [make_switched_candidate("sig-kettle", "Kettle", rise_w=2000.0)],
    )
    power_eid, energy_eid = entities["Kettle"]
    devices = entry.runtime_data.devices

    await _kettle_on(hass, freezer)
    assert _state(hass, power_eid) == "2000.0"
    energy_before = _state(hass, energy_eid)

    # Gap; the kettle switches off inside it (grid is away).
    async_set_power(hass, "sensor.grid_power", "unavailable")
    await hass.async_block_till_done()
    await _drive_ticks(hass, freezer, 4)
    assert _state(hass, power_eid) == "unavailable"
    assert devices.unavailable_reason(_device_uuid(entry, "Kettle")) == "state_unknown"

    # Source returns at the base load (below half the kettle's step): the
    # power sensor stays unavailable at first, then reads 0.0 once the low
    # remainder is sustained (D-30a), with no energy and no unattributed edge.
    async_set_power(hass, "sensor.grid_power", 500)
    await hass.async_block_till_done()
    unseen_zero = None
    for step in range(1, LOW_RESIDUAL_STEPS + 2):
        await _drive_ticks(hass, freezer, 1)
        state_now = _state(hass, power_eid)
        if step < LOW_RESIDUAL_STEPS and state_now != "unavailable":
            unseen_zero = (step, state_now)
        if state_now == "0.0":
            break
    assert unseen_zero is None, f"power claimed {unseen_zero} while unknown"
    assert _state(hass, power_eid) == "0.0"
    assert _state(hass, energy_eid) == energy_before
    summary = devices.timing_summary()
    assert summary["unattributed_total"] == 0

    # The next kettle run is detected one grid sample after its edge.
    await _kettle_on(hass, freezer)
    assert _state(hass, power_eid) == "2000.0"


@pytest.mark.parametrize("return_w,expect_unknown", [(500, False), (2500, True)])
async def test_off_device_after_source_gap_unknown_only_if_plausible(
    return_w: int,
    expect_unknown: bool,
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
) -> None:
    """D-31: after a long gap an OFF device turns unknown only when the first valid remainder says it could be running."""
    entry, entities = await _setup_confirmed(
        hass,
        energy_manager,
        std_prefs,
        hass_storage,
        freezer,
        [make_switched_candidate("sig-kettle", "Kettle", rise_w=2000.0)],
    )
    power_eid, energy_eid = entities["Kettle"]
    devices = entry.runtime_data.devices
    kettle_uuid = _device_uuid(entry, "Kettle")

    # Kettle stays OFF through the gap.
    async_set_power(hass, "sensor.grid_power", "unavailable")
    await hass.async_block_till_done()
    await _drive_ticks(hass, freezer, 4)
    energy_before = _state(hass, energy_eid)

    # Source returns at base load (low) or base + step (high).
    async_set_power(hass, "sensor.grid_power", return_w)
    await hass.async_block_till_done()
    await _drive_ticks(hass, freezer, 3)
    if expect_unknown:
        assert devices.unavailable_reason(kettle_uuid) == "state_unknown"
        assert _state(hass, power_eid) == "unavailable"
        assert _state(hass, energy_eid) == energy_before
        remainder = hass.states.get("sensor.powerguesser_remainder_power")
        assert float(remainder.state) == pytest.approx(2300.0, abs=1.0)
        assert remainder.attributes["estimated_w"] == 0.0
        # The kettle's fall (inside its fall box) ends the unknown state at
        # 0 W with the energy total unchanged (D-29c).
        await _kettle_off(hass, freezer)
        assert _state(hass, power_eid) == "0.0"
        assert _state(hass, energy_eid) == energy_before
    else:
        assert devices.unavailable_reason(kettle_uuid) is None
        assert _state(hass, power_eid) == "0.0"
        assert _state(hass, energy_eid) == energy_before
        # The next run is detected one grid sample after its edge.
        await _kettle_on(hass, freezer)
        assert _state(hass, power_eid) == "2000.0"


async def test_energy_advances_on_tick_while_on(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
) -> None:
    """A steady ON period advances the energy sensor at grid cadence via on_tick republishes."""
    _entry, entities = await _setup_confirmed(
        hass,
        energy_manager,
        std_prefs,
        hass_storage,
        freezer,
        [make_switched_candidate("sig-kettle", "Kettle", rise_w=2000.0)],
    )
    power_eid, energy_eid = entities["Kettle"]

    await _kettle_on(hass, freezer)
    assert _state(hass, power_eid) == "2000.0"

    # No aligner publish between grid points: only the 10 s tick drains the
    # grid, and each energy delta republishes the snapshot.
    previous = float(_state(hass, energy_eid))
    for _ in range(6):
        await _drive_ticks(hass, freezer, 1)
        current = float(_state(hass, energy_eid))
        assert current > previous
        previous = current


async def test_stuck_guard_power_unavailable_energy_kept(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
) -> None:
    """D-28/D-29: the stuck-ON guard forces unknown (never 0 W) with the energy total kept."""
    from custom_components.powerguesser.engine.detect.online import DeviceModel
    from custom_components.powerguesser.engine.intervals import EventType

    c_kettle = make_switched_candidate("sig-kettle", "Kettle", rise_w=2000.0)
    c_kettle["states"][0]["dur_s"] = {"lo": 30.0, "p50": 60.0, "hi": 90.0}
    entry, entities = await _setup_confirmed(
        hass,
        energy_manager,
        std_prefs,
        hass_storage,
        freezer,
        [c_kettle],
    )
    power_eid, energy_eid = entities["Kettle"]
    devices = entry.runtime_data.devices
    kettle_uuid = _device_uuid(entry, "Kettle")
    rec = devices.record(kettle_uuid)
    model = DeviceModel.from_artefact(
        kettle_uuid, rec["versions"][rec["active_v"] - 1]["artefact"]
    )
    max_on_s = model.max_on_s

    async_set_power(hass, "sensor.grid_power", 2500)
    await hass.async_block_till_done()
    remainder_before = float(_state(hass, "sensor.powerguesser_remainder_power"))

    # Hold the plateau past max_on_s without any fall edge.
    forced = False
    for _ in range(int(max_on_s // 30) + 6):
        await _drive_ticks(hass, freezer, 1)
        if _state(hass, power_eid) == "unavailable":
            forced = True
            break
    assert forced, "stuck-ON duration trigger never fired"
    assert devices.unavailable_reason(kettle_uuid) == "state_unknown"
    energy_at_forcing = _state(hass, energy_eid)

    # No energy accrues while unknown; the remainder rises by the kettle step.
    await _drive_ticks(hass, freezer, 3)
    assert _state(hass, power_eid) == "unavailable"
    assert _state(hass, energy_eid) == energy_at_forcing
    remainder = hass.states.get("sensor.powerguesser_remainder_power")
    assert float(remainder.state) == pytest.approx(remainder_before + 2000.0, abs=1.0)
    assert remainder.attributes["estimated_w"] == 0.0

    # A same-grid detector rebuild (SHORT_EVENT toggled) keeps state_unknown.
    _publish_profile(
        entry,
        30.0,
        enabled=(
            EventType.SHORT_EVENT,
            EventType.STEP,
            EventType.CYCLE_SHAPE,
            EventType.CONTINUOUS,
        ),
    )
    await hass.async_block_till_done()
    assert _state(hass, power_eid) == "unavailable"
    assert devices.unavailable_reason(kettle_uuid) == "state_unknown"

    # The plateau's fall (inside the kettle's fall box) makes the sensor 0.0 (D-29c).
    await _kettle_off(hass, freezer)
    assert _state(hass, power_eid) == "0.0"
    # The next run is detected again. max_on_s is narrowed to 135 s (~4 grid
    # steps) above, so assert inside the run: the step is attributed one grid
    # sample after the edge and the fall returns to 0.0 (a 5-tick _kettle_on
    # would outlive the narrowed stuck-guard window).
    async_set_power(hass, "sensor.grid_power", 2500)
    await hass.async_block_till_done()
    await _drive_ticks(hass, freezer, 2)
    assert _state(hass, power_eid) == "2000.0"
    await _kettle_off(hass, freezer)
    assert _state(hass, power_eid) == "0.0"

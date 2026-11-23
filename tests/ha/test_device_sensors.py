"""Tests for device energy and identity across reloads, restarts and crashes.

Plan 04-05 Task 3 (OUT-02, OUT-05, D-26, D-29a, D-30a/b, D-31, D-32, D-34):
the energy counter survives reloads and restarts without jumps or resets
(hot Store flush on unload, restore-cache floor raise, monotonic totals),
the detector restores on the first published grid (never the provisional
30 s default), a device restored after a gap starts unknown when it could
plausibly be running (never a false 0 W, D-29a/D-30b/D-31), a changed
interval during downtime turns stored-ON devices unknown (D-26/D-32), and
every Store/subentry mismatch after a crash or backup restore ends in a
clear, recoverable state (Repairs issue + unavailable sensors).
"""

from datetime import timedelta
import math
from types import MappingProxyType
from typing import Any

import pytest

from homeassistant.components.energy.data import EnergyManager
from homeassistant.config_entries import ConfigEntry, ConfigSubentry
from homeassistant.core import HomeAssistant, State
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers import issue_registry as ir
from homeassistant.util import dt as dt_util
from pytest_homeassistant.common import (
    async_fire_time_changed,
    mock_restore_cache_with_extra_data,
)

from custom_components.powerguesser.const import (
    DEVICES_STORE_KEY,
    DEVICE_STATE_STORE_KEY,
    DOMAIN,
    LEARNING_STORES_VERSION,
    PROFILE_STORE_KEY,
    PROFILE_STORE_VERSION,
    REVIEW_STORE_KEY,
)
from custom_components.powerguesser.engine.detect.online import (
    DeviceModel,
    OnlineDetector,
)
from custom_components.powerguesser.engine.intervals import ProfileSnapshot, gate

from tests.conftest import (
    async_confirm_candidate,
    async_set_power,
    async_setup_via_flow,
    make_switched_candidate,
    seed_candidates,
)
from tests.fixtures.learning.synth import SynthSpec
from tests.ha.test_switched_loads_learning import (
    _confirm_and_entities,
    _pin_grid,
    _run_learning_to_candidate,
)

GRID_S = 30


def _aligned_move(freezer: Any, grid_s: int = GRID_S) -> None:
    """Move the freezer to the next epoch multiple of the grid."""
    now = dt_util.utcnow().timestamp()
    aligned = math.ceil(now / grid_s) * grid_s
    freezer.move_to(dt_util.utc_from_timestamp(aligned))


def _pin_grid(entry: ConfigEntry, grid_s: int = GRID_S) -> None:
    """Publish a profile snapshot pinning the D-26 grid."""
    enabled, disabled = gate(float(grid_s), False)
    entry.runtime_data.profile_coordinator.async_set_updated_data(
        ProfileSnapshot(
            sources={},
            effective_s=float(grid_s),
            assumed=False,
            enabled=enabled,
            disabled=disabled,
        )
    )


def _seed_profile(hass_storage: dict[str, Any], median_s: float) -> None:
    """Seed the profile Store the way tests/ha/test_interval_sensors.py does."""
    hass_storage[PROFILE_STORE_KEY] = {
        "version": PROFILE_STORE_VERSION,
        "minor_version": 1,
        "key": PROFILE_STORE_KEY,
        "data": {
            "sources": {
                "sensor.grid_power": {
                    "median_s": float(median_s),
                    "p90_s": float(median_s) + 1.0,
                    "mode": "re_reporting",
                }
            }
        },
    }


async def _tick(hass: HomeAssistant, freezer: Any, seconds: int) -> None:
    """Advance the freezer and fire the time-changed event."""
    freezer.tick(timedelta(seconds=seconds))
    async_fire_time_changed(hass, dt_util.utcnow())
    await hass.async_block_till_done()


async def _drive_grid(hass: HomeAssistant, freezer: Any, n: int, grid_s: int = GRID_S) -> None:
    """Advance n full grid steps via the 10 s tick path."""
    for _ in range(n * 3):
        await _tick(hass, freezer, 10)


async def _drive_reported(
    hass: HomeAssistant, freezer: Any, n: int, step_s: int, grid_w: float
) -> None:
    """Advance n steps, re-reporting the grid power each step.

    A source profiled as re_reporting (restored from the seeded profile
    Store) goes stale after 2.5 x median seconds of silence, so the 15 s
    scenarios must feed reports at the source cadence
    (tests/ha/test_interval_sensors.py pattern).
    """
    for _ in range(n):
        freezer.tick(timedelta(seconds=step_s))
        async_fire_time_changed(hass, dt_util.utcnow(), fire_all=True)
        async_set_power(hass, "sensor.grid_power", grid_w)
        await hass.async_block_till_done()


async def _setup_confirmed(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
    candidates: list[dict[str, Any]],
    grid_s: int = GRID_S,
) -> tuple[ConfigEntry, dict[str, tuple[str, str, str]]]:
    """Set up the entry, pin the grid and confirm every candidate.

    Returns (entry, {label: (power_eid, energy_eid, device_uuid)}).
    """
    seed_candidates(hass_storage, candidates)
    energy_manager.data = std_prefs
    async_set_power(hass, "sensor.grid_power", 500)
    async_set_power(hass, "sensor.solar_power", 0)
    async_set_power(hass, "sensor.battery_power", 0)
    async_set_power(hass, "sensor.washer_power", 200)
    _aligned_move(freezer, grid_s)
    entry = await async_setup_via_flow(hass)
    _pin_grid(entry, grid_s)
    await hass.async_block_till_done()

    ent_reg = er.async_get(hass)
    entities: dict[str, tuple[str, str, str]] = {}
    for cand in candidates:
        label = str(cand["label"])
        await async_confirm_candidate(hass, entry, str(cand["signature_id"]), label)
        sub = next(
            s for s in entry.get_subentries_of_type("device") if s.title == label
        )
        dev_uuid = sub.data["device_uuid"]
        power_eid = ent_reg.async_get_entity_id(
            "sensor", DOMAIN, f"{entry.entry_id}_{dev_uuid}_power"
        )
        energy_eid = ent_reg.async_get_entity_id(
            "sensor", DOMAIN, f"{entry.entry_id}_{dev_uuid}_energy"
        )
        assert power_eid is not None and energy_eid is not None
        entities[label] = (power_eid, energy_eid, dev_uuid)
    return entry, entities


def _kettle_on(hass: HomeAssistant) -> None:
    """Switch the 2000 W kettle on (base load 500 W)."""
    async_set_power(hass, "sensor.grid_power", 2500)


def _kettle_off(hass: HomeAssistant) -> None:
    """Switch the kettle off (base load only)."""
    async_set_power(hass, "sensor.grid_power", 500)


def _seed_energy_restore_cache(hass: HomeAssistant, energy_eid: str, kwh: float) -> None:
    """Seed the restore cache with one energy-sensor reading (kWh)."""
    mock_restore_cache_with_extra_data(
        hass,
        (
            (
                State(
                    energy_eid,
                    str(kwh),
                    {
                        "device_class": "energy",
                        "unit_of_measurement": "kWh",
                        "state_class": "total_increasing",
                    },
                ),
                {
                    "native_value": kwh,
                    "native_unit_of_measurement": "kWh",
                },
            ),
        ),
    )


async def test_energy_survives_reload_and_never_decreases(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
) -> None:
    """The hot Store holds the same total right after unload; the state continues."""
    entry, entities = await _setup_confirmed(
        hass,
        energy_manager,
        std_prefs,
        hass_storage,
        freezer,
        [make_switched_candidate("sig-kettle", "Kettle", rise_w=2000.0)],
    )
    power_eid, energy_eid, dev_uuid = entities["Kettle"]

    _kettle_on(hass)
    await hass.async_block_till_done()
    await _drive_grid(hass, freezer, 5)
    _kettle_off(hass)
    await hass.async_block_till_done()
    await _drive_grid(hass, freezer, 5)
    energy_before = float(hass.states.get(energy_eid).state)
    assert energy_before > 0.0

    await hass.config_entries.async_reload(entry.entry_id)
    await hass.async_block_till_done()

    # The flush on unload wrote the same total into the hot Store.
    hot = hass_storage[DEVICE_STATE_STORE_KEY]["data"]
    assert hot["energy_wh"][dev_uuid] == pytest.approx(energy_before * 1000.0, abs=1.0)
    # After the reload the energy state equals the pre-reload value.
    assert float(hass.states.get(energy_eid).state) == pytest.approx(
        energy_before, abs=1e-6
    )
    assert hass.states.get(power_eid).state == "0.0"


async def test_energy_restored_from_restore_cache_takes_max(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
) -> None:
    """Hot Store lost: the restore cache seeds the floor (D-30b); max wins."""
    entry, entities = await _setup_confirmed(
        hass,
        energy_manager,
        std_prefs,
        hass_storage,
        freezer,
        [make_switched_candidate("sig-kettle", "Kettle", rise_w=2000.0)],
    )
    power_eid, energy_eid, dev_uuid = entities["Kettle"]

    # Hot Store lost (crash before the first save), restore cache holds 1.234 kWh.
    await hass.config_entries.async_unload(entry.entry_id)
    await hass.async_block_till_done()
    hass_storage.pop(DEVICE_STATE_STORE_KEY)
    _seed_energy_restore_cache(hass, energy_eid, 1.234)
    await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    devices = entry.runtime_data.devices
    # D-30b: the lost device starts unknown (power unavailable, never 0.0)...
    assert hass.states.get(power_eid).state == "unavailable"
    assert devices.unavailable_reason(dev_uuid) == "state_unknown"
    # ...while the energy sensor is available with the restored total.
    energy_state = hass.states.get(energy_eid)
    assert energy_state.state != "unavailable"
    assert float(energy_state.state) == pytest.approx(1.234, abs=1e-6)

    # Later increments start from the restored value.
    _kettle_on(hass)
    await hass.async_block_till_done()
    await _drive_grid(hass, freezer, 5)
    assert float(hass.states.get(energy_eid).state) > 1.234

    # Store 2.0 kWh beats restore 1.5 kWh: the max wins, no reset.
    await hass.config_entries.async_unload(entry.entry_id)
    await hass.async_block_till_done()
    hass_storage[DEVICE_STATE_STORE_KEY]["data"]["energy_wh"][dev_uuid] = 2000.0
    _seed_energy_restore_cache(hass, energy_eid, 1.5)
    await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    assert float(hass.states.get(energy_eid).state) == pytest.approx(2.0, abs=1e-6)


async def test_energy_states_monotonic_over_runs_and_reloads(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
) -> None:
    """Over runs and two reloads the recorded energy states never decrease."""
    entry, entities = await _setup_confirmed(
        hass,
        energy_manager,
        std_prefs,
        hass_storage,
        freezer,
        [make_switched_candidate("sig-kettle", "Kettle", rise_w=2000.0)],
    )
    _power_eid, energy_eid, _uuid = entities["Kettle"]

    readings: list[float] = []

    async def _run_and_record() -> None:
        _kettle_on(hass)
        await hass.async_block_till_done()
        await _drive_grid(hass, freezer, 5)
        _kettle_off(hass)
        await hass.async_block_till_done()
        await _drive_grid(hass, freezer, 3)
        readings.append(float(hass.states.get(energy_eid).state))

    await _run_and_record()
    await hass.config_entries.async_reload(entry.entry_id)
    await hass.async_block_till_done()
    readings.append(float(hass.states.get(energy_eid).state))
    await _run_and_record()
    await hass.config_entries.async_reload(entry.entry_id)
    await hass.async_block_till_done()
    readings.append(float(hass.states.get(energy_eid).state))
    await _run_and_record()

    assert readings == sorted(readings)
    assert readings[-1] > readings[0]


async def test_on_state_across_quick_and_slow_reload(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
) -> None:
    """ON across a quick reload stays ON; across a gap it starts unknown (D-29a)."""
    entry, entities = await _setup_confirmed(
        hass,
        energy_manager,
        std_prefs,
        hass_storage,
        freezer,
        [make_switched_candidate("sig-kettle", "Kettle", rise_w=2000.0)],
    )
    power_eid, energy_eid, dev_uuid = entities["Kettle"]

    # Quick reload (no clock advance: within GAP_HOLD_STEPS grid steps).
    _kettle_on(hass)
    await hass.async_block_till_done()
    await _drive_grid(hass, freezer, 5)
    assert hass.states.get(power_eid).state == "2000.0"
    energy_before = float(hass.states.get(energy_eid).state)

    await hass.config_entries.async_reload(entry.entry_id)
    await hass.async_block_till_done()
    devices = entry.runtime_data.devices
    # Still ON right after the reload: the step, never unavailable.
    assert hass.states.get(power_eid).state == "2000.0"
    assert devices.unavailable_reason(dev_uuid) is None

    # Slow reload: advance past GAP_HOLD_STEPS (2 x 30 s) grid steps, unload
    # and set up again while the kettle is still on.
    await hass.config_entries.async_unload(entry.entry_id)
    await hass.async_block_till_done()
    freezer.tick(timedelta(seconds=120))
    await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    devices = entry.runtime_data.devices

    # D-29a: unknown, not 0.0; the energy sensor keeps its pre-reload total.
    power_state = hass.states.get(power_eid)
    assert power_state.state == "unavailable"
    assert devices.unavailable_reason(dev_uuid) == "state_unknown"
    assert float(hass.states.get(energy_eid).state) == pytest.approx(
        energy_before, abs=1e-6
    )

    # The kettle is still on: a fresh report keeps the grid draining with no
    # fall and no energy for the gap.
    _kettle_on(hass)
    await hass.async_block_till_done()
    await _drive_grid(hass, freezer, 3)
    assert hass.states.get(power_eid).state == "unavailable"
    assert float(hass.states.get(energy_eid).state) == pytest.approx(
        energy_before, abs=1e-6
    )

    # Its fall (inside the fall box) makes the power sensor 0.0 (D-29c).
    _kettle_off(hass)
    await hass.async_block_till_done()
    await _drive_grid(hass, freezer, 5)
    assert hass.states.get(power_eid).state == "0.0"
    assert float(hass.states.get(energy_eid).state) == pytest.approx(
        energy_before, abs=1e-6
    )


@pytest.mark.parametrize("scenario", ["low", "high"])
async def test_off_device_after_slow_reload_unknown_only_if_plausible(
    scenario: str,
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
) -> None:
    """D-31: an OFF device goes unknown after a gap only when the first valid
    remainder is at least half its smallest step; otherwise it stays at 0 W."""
    entry, entities = await _setup_confirmed(
        hass,
        energy_manager,
        std_prefs,
        hass_storage,
        freezer,
        [make_switched_candidate("sig-kettle", "Kettle", rise_w=2000.0)],
    )
    power_eid, energy_eid, dev_uuid = entities["Kettle"]

    # One completed run: the hot Store holds a stepped detector state with
    # last_grid_ts set; the kettle is OFF afterwards.
    _kettle_on(hass)
    await hass.async_block_till_done()
    await _drive_grid(hass, freezer, 5)
    _kettle_off(hass)
    await hass.async_block_till_done()
    await _drive_grid(hass, freezer, 5)
    energy_before = float(hass.states.get(energy_eid).state)

    # Restart with a gap: more than GAP_HOLD_STEPS grid steps of downtime.
    await hass.config_entries.async_unload(entry.entry_id)
    await hass.async_block_till_done()
    freezer.tick(timedelta(seconds=120))
    if scenario == "high":
        # The kettle was switched on during the downtime.
        _kettle_on(hass)
    await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    devices = entry.runtime_data.devices
    # A fresh report wakes the aligner so grid points drain again (a real
    # meter re-reports within one poll interval after a restart).
    if scenario == "high":
        _kettle_on(hass)
    else:
        _kettle_off(hass)
    await hass.async_block_till_done()

    # 0.0 right after setup in both scenarios (nothing stepped yet).
    assert hass.states.get(power_eid).state == "0.0"

    if scenario == "low":
        # Samples at the base load (below half the step): the post-gap check
        # sees a low remainder and the device stays OFF at 0.0, never
        # unavailable; the energy total is unchanged.
        await _drive_grid(hass, freezer, 4)
        assert hass.states.get(power_eid).state == "0.0"
        assert devices.unavailable_reason(dev_uuid) is None
        assert float(hass.states.get(energy_eid).state) == pytest.approx(
            energy_before, abs=1e-6
        )
        # The next run is detected normally one grid sample after its edge.
        _kettle_on(hass)
        await hass.async_block_till_done()
        await _drive_grid(hass, freezer, 3)
        assert hass.states.get(power_eid).state == "2000.0"
    else:
        # Samples at base + step: within two grid points the device is
        # unknown (state_unknown), never reading its step; the energy sensor
        # stays available at the pre-reload total and does not grow.
        await _drive_grid(hass, freezer, 2)
        assert hass.states.get(power_eid).state == "unavailable"
        assert devices.unavailable_reason(dev_uuid) == "state_unknown"
        assert float(hass.states.get(energy_eid).state) == pytest.approx(
            energy_before, abs=1e-6
        )
        await _drive_grid(hass, freezer, 3)
        assert hass.states.get(power_eid).state == "unavailable"
        # The kettle's fall (inside its fall box) makes the power 0.0 (D-29c).
        _kettle_off(hass)
        await hass.async_block_till_done()
        await _drive_grid(hass, freezer, 5)
        assert hass.states.get(power_eid).state == "0.0"
        assert float(hass.states.get(energy_eid).state) == pytest.approx(
            energy_before, abs=1e-6
        )


@pytest.mark.parametrize(
    "case", ["store_present", "cache_present", "increment_before_restore"]
)
async def test_restore_floor_order_independent(
    case: str,
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
) -> None:
    """max(current, restored + delta_since_load) is independent of ordering."""
    entry, entities = await _setup_confirmed(
        hass,
        energy_manager,
        std_prefs,
        hass_storage,
        freezer,
        [make_switched_candidate("sig-kettle", "Kettle", rise_w=2000.0)],
    )
    _power_eid, energy_eid, dev_uuid = entities["Kettle"]
    devices = entry.runtime_data.devices

    _kettle_on(hass)
    await hass.async_block_till_done()
    await _drive_grid(hass, freezer, 5)
    _kettle_off(hass)
    await hass.async_block_till_done()
    await _drive_grid(hass, freezer, 3)

    if case == "store_present":
        # (a) Hot Store present, restore cache absent (the common restart):
        # the detector restores ON/OFF/unknown from the Store and the energy
        # state continues from the Store total.
        energy_before = float(hass.states.get(energy_eid).state)
        assert energy_before > 0.0
        await hass.config_entries.async_reload(entry.entry_id)
        await hass.async_block_till_done()
        assert float(hass.states.get(energy_eid).state) == pytest.approx(
            energy_before, abs=1e-6
        )
        _kettle_on(hass)
        await hass.async_block_till_done()
        await _drive_grid(hass, freezer, 5)
        assert float(hass.states.get(energy_eid).state) > energy_before
        return

    if case == "cache_present":
        # (b) Hot Store absent, restore cache 1.234 kWh: the kettle's rise and
        # plateau grid points processed after setup add increments that start
        # from the restored floor, never from the increments alone.
        await hass.config_entries.async_unload(entry.entry_id)
        await hass.async_block_till_done()
        hass_storage.pop(DEVICE_STATE_STORE_KEY)
        _seed_energy_restore_cache(hass, energy_eid, 1.234)
        await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()
        assert float(hass.states.get(energy_eid).state) == pytest.approx(
            1.234, abs=1e-6
        )
        _kettle_on(hass)
        await hass.async_block_till_done()
        await _drive_grid(hass, freezer, 5)
        assert float(hass.states.get(energy_eid).state) > 1.234
        return

    # (c) An increment before the restore is counted exactly once: raising the
    # floor after the detector stepped (the sensor's restore arriving late)
    # gives restored + delta, not restored + 2 x delta, and never clips the
    # increment away.
    total_wh = devices.energy_wh(dev_uuid)
    delta_wh = devices._delta_since_load.get(dev_uuid, 0.0)  # noqa: SLF001
    assert delta_wh > 0.0
    restored_wh = total_wh - delta_wh + 1500.0
    devices.raise_energy_floor(dev_uuid, restored_wh)
    assert devices.energy_wh(dev_uuid) == pytest.approx(total_wh + 1500.0, abs=1e-6)
    assert float(hass.states.get(energy_eid).state) == pytest.approx(
        (total_wh + 1500.0) / 1000.0, abs=1e-6
    )


@pytest.mark.parametrize("case", ["hot_store_missing", "partial_state", "fresh_confirm"])
async def test_lost_runtime_state_starts_unknown(
    case: str,
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
) -> None:
    """D-30b: lost saved runtime state starts a confirmed device unknown."""
    candidates = [
        make_switched_candidate("sig-a", "Device A", rise_w=2000.0),
        make_switched_candidate("sig-b", "Device B", rise_w=1800.0),
    ]
    if case == "fresh_confirm":
        # (iii) A device confirmed while the entry runs starts OFF at 0.0,
        # not unknown: the D-30b seed exists only in the load-time build.
        # Both candidates are seeded; only A is confirmed during setup.
        seed_candidates(hass_storage, candidates)
        energy_manager.data = std_prefs
        async_set_power(hass, "sensor.grid_power", 500)
        async_set_power(hass, "sensor.solar_power", 0)
        async_set_power(hass, "sensor.battery_power", 0)
        async_set_power(hass, "sensor.washer_power", 200)
        _aligned_move(freezer, GRID_S)
        entry = await async_setup_via_flow(hass)
        _pin_grid(entry)
        await hass.async_block_till_done()
        await async_confirm_candidate(hass, entry, "sig-a", "Device A")
        await async_confirm_candidate(hass, entry, "sig-b", "Device B")
        await hass.async_block_till_done()
        devices = entry.runtime_data.devices
        sub = next(
            s
            for s in entry.get_subentries_of_type("device")
            if s.title == "Device B"
        )
        uuid_b = sub.data["device_uuid"]
        power_b = er.async_get(hass).async_get_entity_id(
            "sensor", DOMAIN, f"{entry.entry_id}_{uuid_b}_power"
        )
        assert hass.states.get(power_b).state == "0.0"
        assert devices.unavailable_reason(uuid_b) is None
        return

    entry, entities = await _setup_confirmed(
        hass,
        energy_manager,
        std_prefs,
        hass_storage,
        freezer,
        candidates,
    )
    power_a, energy_a, uuid_a = entities["Device A"]
    power_b, energy_b, uuid_b = entities["Device B"]

    if case == "hot_store_missing":
        # (i) The hot Store is gone before setup; the restore cache seeds the
        # energy total for the lost device.
        await hass.config_entries.async_unload(entry.entry_id)
        await hass.async_block_till_done()
        hass_storage.pop(DEVICE_STATE_STORE_KEY)
        _seed_energy_restore_cache(hass, energy_a, 1.234)
        await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()
        devices = entry.runtime_data.devices
        # A fresh report wakes the aligner so grid points drain again.
        async_set_power(hass, "sensor.grid_power", 500)
        await hass.async_block_till_done()

        assert hass.states.get(power_a).state == "unavailable"
        assert devices.unavailable_reason(uuid_a) == "state_unknown"
        assert float(hass.states.get(energy_a).state) == pytest.approx(1.234, abs=1e-6)
        # Device B is covered by nothing either: it starts unknown too.
        assert hass.states.get(power_b).state == "unavailable"
        assert devices.unavailable_reason(uuid_b) == "state_unknown"
        assert float(hass.states.get(energy_b).state) == pytest.approx(0.0, abs=1e-6)

        # A sustained low remainder returns the really-off device to 0.0
        # within LOW_RESIDUAL_STEPS + 1 grid points (D-30a), energy unchanged.
        await _drive_grid(hass, freezer, 5)
        assert hass.states.get(power_a).state == "0.0"
        assert devices.unavailable_reason(uuid_a) is None
        assert float(hass.states.get(energy_a).state) == pytest.approx(1.234, abs=1e-6)
        return

    # (ii) The stored detector state covers only device A (built through the
    # public engine API, never stepped: no last_grid_ts, no D-31 arm); the
    # hot energy total holds both devices.
    artefact_a = {
        k: v
        for k, v in candidates[0].items()
        if k not in ("identity", "fingerprint", "first_label")
    }
    detector = OnlineDetector(
        [DeviceModel.from_artefact(uuid_a, artefact_a)],
        grid_s=GRID_S,
        threshold_w=120.0,
    )
    await hass.config_entries.async_unload(entry.entry_id)
    await hass.async_block_till_done()
    hass_storage[DEVICE_STATE_STORE_KEY] = {
        "version": 1,
        "minor_version": 1,
        "key": DEVICE_STATE_STORE_KEY,
        "data": {
            "energy_wh": {uuid_a: 0.0, uuid_b: 1500.0},
            "detector": detector.state(),
        },
    }
    await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    devices = entry.runtime_data.devices

    # A is covered by the stored state: OFF at 0.0. B is not covered: unknown.
    assert hass.states.get(power_a).state == "0.0"
    assert devices.unavailable_reason(uuid_a) is None
    assert hass.states.get(power_b).state == "unavailable"
    assert devices.unavailable_reason(uuid_b) == "state_unknown"
    # Both energy sensors are available with their stored totals.
    assert float(hass.states.get(energy_a).state) == pytest.approx(0.0, abs=1e-6)
    assert float(hass.states.get(energy_b).state) == pytest.approx(1.5, abs=1e-6)


@pytest.mark.parametrize("scenario", ["same", "changed"])
async def test_restart_with_changed_interval_enters_unknown(
    scenario: str,
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
) -> None:
    """D-26 with D-29a/D-32: restore happens on the first published grid.

    A quick restart at an unchanged interval keeps an ON device ON; an
    interval change during downtime turns stored-ON devices unknown
    (grid_mismatch, energy kept), then state_unknown once rematched on the
    old grid, and 0.0 on the matching fall.
    """
    grid_s = 15 if scenario == "same" else GRID_S
    candidate = make_switched_candidate("sig-kettle", "Kettle", rise_w=2000.0)
    candidate["extraction"]["grid_s"] = grid_s

    if scenario == "same":
        _seed_profile(hass_storage, 15.0)

    entry, entities = await _setup_confirmed(
        hass,
        energy_manager,
        std_prefs,
        hass_storage,
        freezer,
        [candidate],
        grid_s=grid_s,
    )
    power_eid, energy_eid, dev_uuid = entities["Kettle"]
    devices = entry.runtime_data.devices

    # The kettle is ON on the hub's grid. The 15 s scenario's seeded
    # re_reporting profile needs reports at the source cadence; the 30 s
    # scenario uses the plain tick path.
    _kettle_on(hass)
    await hass.async_block_till_done()
    if scenario == "same":
        await _drive_reported(hass, freezer, 4, 15, 2500)
    else:
        await _drive_grid(hass, freezer, 4)
    assert hass.states.get(power_eid).state == "2000.0"
    energy_before = float(hass.states.get(energy_eid).state)
    assert energy_before > 0.0

    # Quick restart (no clock advance): within GAP_HOLD_STEPS grid steps.
    await hass.config_entries.async_unload(entry.entry_id)
    await hass.async_block_till_done()
    if scenario == "same":
        _seed_profile(hass_storage, 15.0)
    else:
        # The interval changed to 15 s during the downtime.
        _seed_profile(hass_storage, 15.0)
    await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    devices = entry.runtime_data.devices

    if scenario == "same":
        # Restored on the published 15 s grid (never the provisional 30 s
        # default): the kettle is still ON and never unavailable.
        assert hass.states.get(power_eid).state == "2000.0"
        assert devices.unavailable_reason(dev_uuid) is None
        assert float(hass.states.get(energy_eid).state) == pytest.approx(
            energy_before, abs=1e-6
        )
        await _drive_reported(hass, freezer, 3, 15, 2500)
        assert hass.states.get(power_eid).state == "2000.0"
        return

    # Changed interval: the stored 30 s state is restored on a 15 s grid, so
    # every stored-ON device enters the unknown state without energy (D-29a);
    # the artefact no longer matches the detector grid either (D-32).
    assert hass.states.get(power_eid).state == "unavailable"
    assert devices.unavailable_reason(dev_uuid) == "grid_mismatch"
    assert float(hass.states.get(energy_eid).state) == pytest.approx(
        energy_before, abs=1e-6
    )
    # Never the kettle's step and never 0.0 while unknown.
    await _drive_reported(hass, freezer, 3, 15, 2500)
    assert hass.states.get(power_eid).state == "unavailable"
    assert devices.unavailable_reason(dev_uuid) == "grid_mismatch"
    assert float(hass.states.get(energy_eid).state) == pytest.approx(
        energy_before, abs=1e-6
    )

    # Publishing a 30 s profile snapshot again makes the reason state_unknown
    # (the device is modelled again but still unknown); energy stays flat.
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
    await hass.async_block_till_done()
    assert hass.states.get(power_eid).state == "unavailable"
    assert devices.unavailable_reason(dev_uuid) == "state_unknown"
    assert float(hass.states.get(energy_eid).state) == pytest.approx(
        energy_before, abs=1e-6
    )

    # The kettle's fall (inside its fall box) makes the power 0.0 (D-29c).
    _kettle_off(hass)
    await hass.async_block_till_done()
    await _drive_reported(hass, freezer, 5, GRID_S, 500)
    assert hass.states.get(power_eid).state == "0.0"
    assert float(hass.states.get(energy_eid).state) == pytest.approx(
        energy_before, abs=1e-6
    )


async def _setup_entry_minimal(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
) -> ConfigEntry:
    """Set up the entry with standard prefs and power states, no candidates."""
    energy_manager.data = std_prefs
    async_set_power(hass, "sensor.grid_power", 500)
    async_set_power(hass, "sensor.solar_power", 0)
    async_set_power(hass, "sensor.battery_power", 0)
    async_set_power(hass, "sensor.washer_power", 200)
    return await async_setup_via_flow(hass)


def _add_device_subentry(
    hass: HomeAssistant, entry: ConfigEntry, data: dict[str, Any], title: str
) -> Any:
    """Add a device subentry directly (bypassing Review) and let reconcile run."""
    sub = ConfigSubentry(
        data=MappingProxyType(data),
        subentry_type="device",
        title=title,
        unique_id=data.get("device_uuid") or title,
    )
    hass.config_entries.async_add_subentry(entry, sub)
    return sub


async def test_subentry_without_record_is_unavailable_with_issue(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
) -> None:
    """A backup-mismatch subentry (no cold record) gets unavailable sensors
    and a WARNING Repairs issue; deleting it clears the issue."""
    entry = await _setup_entry_minimal(hass, energy_manager, std_prefs)

    sub = _add_device_subentry(
        hass,
        entry,
        {"device_uuid": "uuid-ghost", "name": "Ghost", "signature_id": "sig-ghost"},
        "Ghost",
    )
    await hass.async_block_till_done()

    ent_reg = er.async_get(hass)
    power_eid = ent_reg.async_get_entity_id(
        "sensor", DOMAIN, f"{entry.entry_id}_uuid-ghost_power"
    )
    energy_eid = ent_reg.async_get_entity_id(
        "sensor", DOMAIN, f"{entry.entry_id}_uuid-ghost_energy"
    )
    assert power_eid is not None
    assert energy_eid is not None
    assert hass.states.get(power_eid).state == "unavailable"
    assert hass.states.get(energy_eid).state == "unavailable"

    issue = ir.async_get(hass).async_get_issue(
        DOMAIN, f"device_record_missing_{sub.subentry_id}"
    )
    assert issue is not None

    hass.config_entries.async_remove_subentry(entry, sub.subentry_id)
    await hass.async_block_till_done()
    assert (
        ir.async_get(hass).async_get_issue(
            DOMAIN, f"device_record_missing_{sub.subentry_id}"
        )
        is None
    )


async def test_subentry_without_uuid_key_raises_issue_only(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
) -> None:
    """A subentry whose data lacks device_uuid adds no entities and raises
    the Repairs issue with the subentry title (no crash)."""
    entry = await _setup_entry_minimal(hass, energy_manager, std_prefs)

    sub = _add_device_subentry(hass, entry, {"name": "Ghost"}, "Ghost")
    await hass.async_block_till_done()

    # No entities can be formed for it; the issue carries its title.
    ent_reg = er.async_get(hass)
    sub_entities = [
        entity
        for entity in ent_reg.entities.values()
        if entity.config_subentry_id == sub.subentry_id
    ]
    assert sub_entities == []
    issue = ir.async_get(hass).async_get_issue(
        DOMAIN, f"device_record_missing_{sub.subentry_id}"
    )
    assert issue is not None
    assert issue.translation_key == "device_record_missing"

    hass.config_entries.async_remove_subentry(entry, sub.subentry_id)
    await hass.async_block_till_done()
    assert (
        ir.async_get(hass).async_get_issue(
            DOMAIN, f"device_record_missing_{sub.subentry_id}"
        )
        is None
    )


async def test_startup_reconcile_truth_table(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
) -> None:
    """Startup reconcile: crash/backup mismatches end recoverable (D-05).

    A confirmed record without its subentry becomes unconfirmed (uuid,
    versions and energy kept, candidate reviewable again); a review link to
    a missing subentry is cleared.
    """
    candidate = make_switched_candidate("sig-alpha", "Alpha", rise_w=2000.0)
    artefact = {
        k: v
        for k, v in candidate.items()
        if k not in ("identity", "fingerprint", "first_label")
    }
    hass_storage[DEVICES_STORE_KEY] = {
        "version": 1,
        "minor_version": 1,
        "key": DEVICES_STORE_KEY,
        "data": {
            "devices": {
                "uuid-alpha": {
                    "name": "Alpha",
                    "signature_id": "sig-alpha",
                    "status": "confirmed",
                    "subentry_id": "sub-missing",
                    "versions": [
                        {
                            "v": 1,
                            "artefact": artefact,
                            "trained_at": "2026-11-16T12:00:00Z",
                            "scores": {},
                            "accepted_at": "2026-11-16T12:00:00Z",
                            "source": "confirm",
                        }
                    ],
                    "active_v": 1,
                }
            }
        },
    }
    hass_storage[DEVICE_STATE_STORE_KEY] = {
        "version": 1,
        "minor_version": 1,
        "key": DEVICE_STATE_STORE_KEY,
        "data": {"energy_wh": {"uuid-alpha": 777.0}, "detector": None},
    }
    hass_storage[REVIEW_STORE_KEY] = {
        "version": LEARNING_STORES_VERSION,
        "minor_version": 1,
        "key": REVIEW_STORE_KEY,
        "data": {
            "decisions": {
                "sig-alpha": {
                    "status": "named",
                    "name": "Alpha",
                    "device_uuid": "uuid-alpha",
                    "subentry_id": "sub-missing",
                    "pre_confirm_status": "kept",
                }
            }
        },
    }
    seed_candidates(hass_storage, [candidate])
    energy_manager.data = std_prefs
    async_set_power(hass, "sensor.grid_power", 500)
    async_set_power(hass, "sensor.solar_power", 0)
    async_set_power(hass, "sensor.battery_power", 0)
    async_set_power(hass, "sensor.washer_power", 200)

    entry = await async_setup_via_flow(hass)
    await hass.async_block_till_done()

    devices = entry.runtime_data.devices
    rec = devices.record("uuid-alpha")
    # The record is unconfirmed, keeps uuid/versions/name and its energy.
    assert rec["status"] == "unconfirmed"
    assert rec["signature_id"] == "sig-alpha"
    assert rec["name"] == "Alpha"
    assert len(rec["versions"]) == 1
    assert devices.energy_wh("uuid-alpha") == pytest.approx(777.0, abs=1e-6)
    # The candidate is reviewable again.
    learning = entry.runtime_data.learning
    visible = learning.visible_candidates()
    assert any(c.get("signature_id") == "sig-alpha" for c in visible)
    # The review link to the missing subentry is cleared; the kept status
    # and the device_uuid survive.
    dec = learning.review.decisions["sig-alpha"]
    assert dec["subentry_id"] is None
    # Un-confirm returns the record to its pre-confirm kept status.
    assert dec["status"] == "kept"
    assert dec["pre_confirm_status"] == "kept"
    assert dec["device_uuid"] == "uuid-alpha"


async def test_learning_run_to_device_shows_validation_error(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    aioclient_mock: Any,
    inprocess_worker: Any,
    freezer: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """D-16/OUT-04: a confirmed device from a real learning run shows the learning-split
    energy error on both sensors, labelled validation_basis "learning_split"."""
    monkeypatch.setattr("custom_components.powerguesser.learning.MAX_REFINE_ROUNDS", 0)
    energy_manager.data = std_prefs
    async_set_power(hass, "sensor.grid_power", 500)
    async_set_power(hass, "sensor.solar_power", 1200)
    async_set_power(hass, "sensor.battery_power", -300)
    entry = await async_setup_via_flow(hass)
    _pin_grid(entry)
    await hass.async_block_till_done()

    cand = await _run_learning_to_candidate(
        hass,
        entry,
        SynthSpec(days=22),
        aioclient_mock,
        {
            "label": "Kettle",
            "category": "kettle",
            "target_w": 2000.0,
            "states": (("on", None),),
        },
    )
    assert cand["scores"]["validated"] is True
    dev_uuid, power_eid, energy_eid = await _confirm_and_entities(
        hass, entry, cand["signature_id"], "Kettle"
    )

    # The confirmed device took the candidate's scores into its active version.
    rec = entry.runtime_data.devices.record(dev_uuid)
    version_scores = rec["versions"][rec["active_v"] - 1]["scores"]
    assert version_scores["validation_error_pct"] is not None
    assert version_scores["validation_basis"] == "learning_split"

    for eid in (power_eid, energy_eid):
        state = hass.states.get(eid)
        assert state is not None
        vep = state.attributes.get("validation_error_pct")
        assert isinstance(vep, float), f"{eid} shows no numeric validation_error_pct: {state.attributes}"
        assert state.attributes.get("validation_basis") == "learning_split"


# The published static attribute order: the Phase 4 six-key D-17 contract with
# the Plan 05-04 keys and the Plan 05-05 drift keys appended (D-09/D-16:
# appended, never reordered); the retired key follows on retired records only.
STATIC_ATTR_KEYS = (
    "status",
    "confidence",
    "validation_error_pct",
    "validation_basis",
    "last_trained",
    "preview",
    "validation_class",
    "validation_peers",
    "unvalidated",
    "drift",
    "drift_reason",
    "drift_since",
)


def _entity(hass: HomeAssistant, entity_id: str) -> Any:
    """Return the live sensor entity instance behind an entity id."""
    component = hass.data["entity_components"]["sensor"]
    entity = component.get_entity(entity_id)
    assert entity is not None, f"entity not found: {entity_id}"
    return entity


def _extra(hass: HomeAssistant, entity_id: str) -> dict[str, Any]:
    """Return the entity's published extra_state_attributes."""
    attrs = _entity(hass, entity_id).extra_state_attributes
    assert isinstance(attrs, dict)
    return attrs


async def test_validated_device_attribute_set(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
) -> None:
    """A validated device shows the complete static set on both sensors (D-14..D-17, D-21)."""
    candidate = make_switched_candidate("sig-kettle", "Kettle", rise_w=2000.0)
    candidate["scores"]["validation_error_pct"] = 3.2
    candidate["scores"]["validation_basis"] = "learning_split"
    entry, entities = await _setup_confirmed(
        hass,
        energy_manager,
        std_prefs,
        hass_storage,
        freezer,
        [candidate],
    )
    power_eid, energy_eid, dev_uuid = entities["Kettle"]
    rec = entry.runtime_data.devices.record(dev_uuid)
    trained_at = rec["versions"][rec["active_v"] - 1]["trained_at"]

    for eid in (power_eid, energy_eid):
        attrs = _extra(hass, eid)
        assert attrs["status"] == "provisional"
        assert attrs["confidence"] == 0.9
        assert attrs["validation_error_pct"] == 3.2
        assert attrs["validation_basis"] == "learning_split"
        assert attrs["last_trained"] == trained_at
        assert attrs["preview"] is False


async def test_preview_device_attributes_null_but_present(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
) -> None:
    """A preview device shows null confidence/error (not 0.0) with every key present."""
    candidate = make_switched_candidate(
        "sig-kettle", "Kettle", rise_w=2000.0, validated=False, val_share=None
    )
    entry, entities = await _setup_confirmed(
        hass,
        energy_manager,
        std_prefs,
        hass_storage,
        freezer,
        [candidate],
    )
    power_eid, energy_eid, dev_uuid = entities["Kettle"]
    assert entry.runtime_data.devices.record(dev_uuid)["preview"] is True

    for eid in (power_eid, energy_eid):
        attrs = _extra(hass, eid)
        assert attrs["status"] == "provisional"
        assert attrs["preview"] is True
        assert attrs["confidence"] is None
        assert attrs["validation_error_pct"] is None
        assert attrs["validation_basis"] is None
        assert attrs["last_trained"] is not None
    # Every static key is present on both sensors; the live keys are power-only.
    assert list(_extra(hass, power_eid).keys()) == list(STATIC_ATTR_KEYS) + [
        "match_score",
        "unattributed_edges",
        "update_available",
        "overlap_refused",
    ]
    assert list(_extra(hass, energy_eid).keys()) == list(STATIC_ATTR_KEYS)


@pytest.mark.parametrize(
    ("train_share", "val_share", "expected"),
    [
        (0.05, 0.05, 1.0),  # val == train
        (0.05, 0.06, 1.0),  # val > train clips to 1.0
        (0.05, 0.0, 0.0),  # val 0 gives 0.0
        (0.0, 0.05, None),  # zero train share gives null
    ],
)
async def test_confidence_clip_edges(
    train_share: float,
    val_share: float,
    expected: float | None,
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
) -> None:
    """D-21: confidence = clip(val.share / train.share, 0, 1) rounded to 3 decimals."""
    candidate = make_switched_candidate(
        "sig-kettle",
        "Kettle",
        rise_w=2000.0,
        train_share=train_share,
        val_share=val_share,
    )
    _entry, entities = await _setup_confirmed(
        hass,
        energy_manager,
        std_prefs,
        hass_storage,
        freezer,
        [candidate],
    )
    power_eid, energy_eid, _uuid = entities["Kettle"]
    for eid in (power_eid, energy_eid):
        confidence = _extra(hass, eid)["confidence"]
        if expected is None:
            assert confidence is None
        else:
            assert confidence == expected


async def test_match_score_and_unattributed_power_only(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
) -> None:
    """D-14/D-17: match_score only while ON, on the power sensor only; unattributed_edges too."""
    entry, entities = await _setup_confirmed(
        hass,
        energy_manager,
        std_prefs,
        hass_storage,
        freezer,
        [make_switched_candidate("sig-kettle", "Kettle", rise_w=2000.0)],
    )
    power_eid, energy_eid, _uuid = entities["Kettle"]

    # OFF: no live fit.
    assert _extra(hass, power_eid)["match_score"] is None
    assert "match_score" not in _extra(hass, energy_eid)
    assert "unattributed_edges" not in _extra(hass, energy_eid)

    # ON: the power sensor carries the box fit in [0, 1] and the refused-edge count.
    _kettle_on(hass)
    await hass.async_block_till_done()
    await _drive_grid(hass, freezer, 5)
    power_attrs = _extra(hass, power_eid)
    match = power_attrs["match_score"]
    assert isinstance(match, float)
    assert 0.0 <= match <= 1.0
    assert isinstance(power_attrs["unattributed_edges"], int)
    energy_attrs = _extra(hass, energy_eid)
    assert "match_score" not in energy_attrs
    assert "unattributed_edges" not in energy_attrs

    # OFF again: the live fit is null, the static set unchanged.
    _kettle_off(hass)
    await hass.async_block_till_done()
    await _drive_grid(hass, freezer, 3)
    assert _extra(hass, power_eid)["match_score"] is None
    assert entry.runtime_data.devices.power_w(entities["Kettle"][2]) == 0.0


async def test_overlap_refused_power_only(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
) -> None:
    """D-07/D-41: the ambiguous-overlap refusal count is a power-sensor attribute."""
    _entry, entities = await _setup_confirmed(
        hass,
        energy_manager,
        std_prefs,
        hass_storage,
        freezer,
        [
            make_switched_candidate("sig-kettle-a", "Kettle A", rise_w=2000.0),
            make_switched_candidate("sig-kettle-b", "Kettle B", rise_w=2000.0),
        ],
    )
    power_a, energy_a, _ua = entities["Kettle A"]
    power_b, energy_b, _ub = entities["Kettle B"]
    assert _extra(hass, power_a)["overlap_refused"] == 0

    # two identical boxes: a 2000 W rise fits both, so it is refused and counted
    # for every competing device (D-07)
    _kettle_on(hass)
    await hass.async_block_till_done()
    await _drive_grid(hass, freezer, 5)

    for power_eid, energy_eid in ((power_a, energy_a), (power_b, energy_b)):
        attrs = _extra(hass, power_eid)
        assert attrs["overlap_refused"] == 1
        assert isinstance(attrs["overlap_refused"], int)
        assert attrs["unattributed_edges"] == 1
        assert "overlap_refused" not in _extra(hass, energy_eid)


async def test_unrecorded_attributes_contract(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
) -> None:
    """D-17: match_score/unattributed_edges are unrecorded; the energy sensor records the static set."""
    from custom_components.powerguesser.sensor import (
        DeviceEnergySensor,
        DevicePowerSensor,
    )

    assert DevicePowerSensor._unrecorded_attributes == frozenset(
        {
            "match_score",
            "unattributed_edges",
            "update_available",
            "overlap_refused",
            "validation_session_start",
            "validation_session_end",
        }
    )
    assert not (set(STATIC_ATTR_KEYS) & set(DeviceEnergySensor._unrecorded_attributes))

    _entry, entities = await _setup_confirmed(
        hass,
        energy_manager,
        std_prefs,
        hass_storage,
        freezer,
        [make_switched_candidate("sig-kettle", "Kettle", rise_w=2000.0)],
    )
    power_eid, energy_eid, _uuid = entities["Kettle"]
    assert set(STATIC_ATTR_KEYS) <= set(_extra(hass, energy_eid).keys())
    assert set(STATIC_ATTR_KEYS) <= set(_extra(hass, power_eid).keys())


async def test_attribute_order_stable_and_no_accuracy_wording(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
) -> None:
    """D-17/OUT-04: identical key order on every write, ON or OFF; no "accura" anywhere."""
    _entry, entities = await _setup_confirmed(
        hass,
        energy_manager,
        std_prefs,
        hass_storage,
        freezer,
        [make_switched_candidate("sig-kettle", "Kettle", rise_w=2000.0)],
    )
    power_eid, energy_eid, _uuid = entities["Kettle"]
    expected_power = list(STATIC_ATTR_KEYS) + [
        "match_score",
        "unattributed_edges",
        "update_available",
        "overlap_refused",
    ]
    expected_energy = list(STATIC_ATTR_KEYS)

    orders: list[tuple[list[str], list[str]]] = []

    async def _capture() -> None:
        orders.append((list(_extra(hass, power_eid).keys()), list(_extra(hass, energy_eid).keys())))

    await _capture()
    _kettle_on(hass)
    await hass.async_block_till_done()
    await _drive_grid(hass, freezer, 4)
    await _capture()
    _kettle_off(hass)
    await hass.async_block_till_done()
    await _drive_grid(hass, freezer, 3)
    await _capture()

    assert len(orders) == 3
    for power_keys, energy_keys in orders:
        assert power_keys == expected_power
        assert energy_keys == expected_energy
        for name, attrs in (
            ("power", _extra(hass, power_eid)),
            ("energy", _extra(hass, energy_eid)),
        ):
            for key, value in attrs.items():
                assert "accura" not in str(key).lower(), f"{name} attribute name: {key}"
                assert "accura" not in str(value).lower(), f"{name} attribute {key}: {value}"


async def test_remove_entry_deletes_device_stores(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
) -> None:
    """Removing the config entry deletes both device Stores."""
    freezer = None
    del freezer  # no clock driving needed
    candidate = make_switched_candidate("sig-kettle", "Kettle", rise_w=2000.0)
    seed_candidates(hass_storage, [candidate])
    energy_manager.data = std_prefs
    async_set_power(hass, "sensor.grid_power", 500)
    async_set_power(hass, "sensor.solar_power", 0)
    async_set_power(hass, "sensor.battery_power", 0)
    async_set_power(hass, "sensor.washer_power", 200)
    entry = await async_setup_via_flow(hass)
    await async_confirm_candidate(hass, entry, "sig-kettle", "Kettle")
    await hass.async_block_till_done()

    assert DEVICES_STORE_KEY in hass_storage
    assert DEVICE_STATE_STORE_KEY in hass_storage

    await hass.config_entries.async_remove(entry.entry_id)
    await hass.async_block_till_done()

    assert DEVICES_STORE_KEY not in hass_storage
    assert DEVICE_STATE_STORE_KEY not in hass_storage

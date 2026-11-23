"""Tests for confirmed program devices: confirm, live stage tracking, gates.

Plan 06-04: a program candidate (dishwasher, washing machine, dryer program or
hob session, D-37) is confirmed through Review exactly like a switched one
(D-05) and its power and energy sensors follow a live run through the compiled
stage template (D-01/D-02/D-33): 0 W while it is OFF during the recognition
lag, the stage power afterwards, back-dated monotonic energy, 0 W on the
matched terminal fall. A program device is gated on CYCLE_SHAPE (RESEARCH A5
Gate), so it keeps working at 60 s sampling while a switched device in the same
hub reads disabled_interval.
"""

from datetime import timedelta
import math
from typing import Any, Sequence

import pytest

from homeassistant.components.energy.data import EnergyManager
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers import issue_registry as ir
from homeassistant.util import dt as dt_util
from pytest_homeassistant.common import async_fire_time_changed

from custom_components.powerguesser.const import DOMAIN
from custom_components.powerguesser.engine.intervals import EVENT_TYPE_MIN_S, EventType
from custom_components.powerguesser.engine.learning.artefact import SignatureArtefact

from tests.conftest import (
    async_confirm_candidate,
    async_open_options_step,
    async_set_power,
    async_setup_via_flow,
    make_switched_candidate,
    seed_candidates,
    seed_validation_report,
)
from tests.fixtures.programs import (
    dishwasher_stages,
    dryer_stages,
    hob_session_stages,
    program_artefact,
    washer_stages,
)
from tests.ha.test_device_detection import (
    _publish_profile,
    _state,
    _tick,
)

GRID_S = 30
BASE_W = 500.0
ISSUE_ID = "detection_interval_too_slow"

# The twelve-key DeviceRuntime.attributes() contract (P4 D-17 + P5 D-14/D-16).
ATTRIBUTE_KEYS = (
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


# -- helpers ----------------------------------------------------------------


def _aligned_move(freezer: Any, grid_s: int = GRID_S) -> None:
    """Move the freezer to the next epoch multiple of the detector grid."""
    now = dt_util.utcnow().timestamp()
    freezer.move_to(dt_util.utc_from_timestamp(math.ceil(now / grid_s) * grid_s))


async def _drive_grid_steps(
    hass: HomeAssistant, freezer: Any, n: int, grid_s: int = GRID_S
) -> None:
    """Advance ``n`` full detector grid steps through the 10 s tick path."""
    ticks = max(1, grid_s // 10)
    for _ in range(n * ticks):
        await _tick(hass, freezer, 10)


def _program_candidate(
    sig_id: str,
    label: str,
    stages: Sequence[tuple[float, int]],
    *,
    grid_s: int = GRID_S,
) -> dict[str, Any]:
    """Build a program candidate dict (artefact + scores + fingerprint) for tests.

    ``program_artefact`` writes the v1 artefact with the hand-written stage
    template; the scores and the fingerprint are the fields
    ``make_switched_candidate`` adds, so confirm and identity see the same
    shape a real run produces.
    """
    cand = program_artefact(sig_id, stages, grid_s=grid_s)
    cand["label"] = label
    cand["scores"] = {
        "train": {"share": 0.06, "n": 20},
        "val": {"share": 0.05, "n": 10},
        "validated": True,
        "conflicts": 0,
    }
    cand["fingerprint"] = SignatureArtefact.from_dict(cand).fingerprint()
    return cand


def _device_uuid(entry: ConfigEntry, label: str) -> str:
    """Return the device uuid for a confirmed device title."""
    sub = next(s for s in entry.get_subentries_of_type("device") if s.title == label)
    return sub.data["device_uuid"]


def _entity_ids(
    hass: HomeAssistant, entry: ConfigEntry, label: str
) -> tuple[str, str, str]:
    """Return (power, energy, validation) entity ids of a confirmed device."""
    dev_uuid = _device_uuid(entry, label)
    reg = er.async_get(hass)
    power = reg.async_get_entity_id(
        "sensor", DOMAIN, f"{entry.entry_id}_{dev_uuid}_power"
    )
    energy = reg.async_get_entity_id(
        "sensor", DOMAIN, f"{entry.entry_id}_{dev_uuid}_energy"
    )
    validation = reg.async_get_entity_id(
        "sensor", DOMAIN, f"{entry.entry_id}_{dev_uuid}_validation"
    )
    assert power is not None and energy is not None and validation is not None
    return power, energy, validation


async def _setup(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any | None,
    candidates: list[dict[str, Any]],
    *,
    grid_s: int = GRID_S,
    confirm: bool = True,
    report: dict[str, Any] | None = None,
) -> tuple[ConfigEntry, dict[str, tuple[str, str, str]]]:
    """Set up the entry, pin the grid and confirm every candidate.

    Returns (entry, {label: (power, energy, validation)}); the map is empty
    with ``confirm=False`` (confirm later in the test).
    """
    if report is not None:
        seed_validation_report(hass_storage, report)
    seed_candidates(hass_storage, candidates)
    energy_manager.data = std_prefs
    async_set_power(hass, "sensor.grid_power", BASE_W)
    async_set_power(hass, "sensor.solar_power", 0)
    async_set_power(hass, "sensor.battery_power", 0)
    async_set_power(hass, "sensor.washer_power", 200)
    if freezer is not None:
        _aligned_move(freezer, grid_s)
    entry = await async_setup_via_flow(hass)
    _publish_profile(entry, float(grid_s))
    await hass.async_block_till_done()

    entities: dict[str, tuple[str, str, str]] = {}
    if not confirm:
        return entry, entities
    for cand in candidates:
        label = str(cand["label"])
        await async_confirm_candidate(hass, entry, str(cand["signature_id"]), label)
        entities[label] = _entity_ids(hass, entry, label)
    return entry, entities


def _run_levels(stages: Sequence[tuple[float, int]]) -> list[float]:
    """One grid sample per program sample: each stage level, then the end (0 W).

    The run ends with the load off plus three base-load samples: the edge
    stream reports the closing edge one sample after it happened (the pairing
    lag rule), so the trailing samples carry the matched terminal fall.
    """
    levels = [float(level) for level, count in stages for _ in range(count)]
    levels.extend([0.0, 0.0, 0.0])
    return levels


async def _drive_samples(
    hass: HomeAssistant,
    freezer: Any,
    levels: Sequence[float],
    power_eid: str,
    energy_eid: str,
    *,
    grid_s: int = GRID_S,
) -> tuple[list[float], list[float]]:
    """Push one grid sample per entry of ``levels``; return (power, energy) reads.

    Each sample sets the grid source to ``BASE_W + level`` and advances exactly
    one detector grid step, so a program run is reproduced sample by sample. An
    unavailable power reading is recorded as NaN so a failure shows the whole
    sequence.
    """
    powers: list[float] = []
    energies: list[float] = []
    for level in levels:
        async_set_power(hass, "sensor.grid_power", BASE_W + level)
        await hass.async_block_till_done()
        await _drive_grid_steps(hass, freezer, 1, grid_s)
        raw_power = _state(hass, power_eid)
        powers.append(float("nan") if raw_power == "unavailable" else float(raw_power))
        energies.append(float(_state(hass, energy_eid)))
    return powers, energies


def _issue(hass: HomeAssistant) -> Any:
    return ir.async_get(hass).async_get_issue(DOMAIN, ISSUE_ID)


async def _warm_up_interval(
    hass: HomeAssistant, freezer: Any, grid_s: int, samples: int = 21
) -> None:
    """Report the base load at the target cadence so the interval profiler agrees.

    The hub rebuilds its profile snapshot from the live interval profiler, which
    needs MIN_DELTAS (20) observations before it has a median; below that it
    falls back to the assumed 30 s default and would re-grid the detector away
    from a slower profile. Reporting the base load at the target cadence - with
    the program still idle, so no rebuild can park it unknown - settles the
    median before a run is driven.
    """
    for _ in range(samples):
        async_set_power(hass, "sensor.grid_power", BASE_W)
        await hass.async_block_till_done()
        await _drive_grid_steps(hass, freezer, 1, grid_s)


# -- Task 1: confirm creates the device, sensors follow a live run ----------


async def test_confirm_program_candidate_creates_device_sensors(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
) -> None:
    """A confirmed program candidate gets the same subentry/unique_id path (D-05)."""
    cand = _program_candidate("sig-dish", "Dishwasher", dishwasher_stages())
    entry, entities = await _setup(
        hass, energy_manager, std_prefs, hass_storage, freezer, [cand]
    )
    power_eid, energy_eid, _validation_eid = entities["Dishwasher"]
    dev_uuid = _device_uuid(entry, "Dishwasher")

    # One device subentry, its unique_id the uuid, entities on UUID ids (T-06-10).
    sub = next(
        s for s in entry.get_subentries_of_type("device") if s.title == "Dishwasher"
    )
    assert sub.unique_id == dev_uuid
    assert power_eid == er.async_get(hass).async_get_entity_id(
        "sensor", DOMAIN, f"{entry.entry_id}_{dev_uuid}_power"
    )
    assert energy_eid == er.async_get(hass).async_get_entity_id(
        "sensor", DOMAIN, f"{entry.entry_id}_{dev_uuid}_energy"
    )

    # The record carries the program kind and a version 1 artefact.
    rec = entry.runtime_data.devices.record(dev_uuid)
    assert rec is not None
    assert rec["kind"] == "program"
    assert rec["active_v"] == 1
    assert rec["versions"][0]["artefact"]["session"]["stages"]

    # DeviceInfo model names the program appliance (D-05).
    energy_entry = er.async_get(hass).async_get(energy_eid)
    device = dr.async_get(hass).async_get(energy_entry.device_id)
    assert device is not None
    assert device.model == "Estimated program appliance"

    # Both sensors expose the twelve-key attribute contract in order.
    assert tuple(entry.runtime_data.devices.attributes(dev_uuid)) == ATTRIBUTE_KEYS

    # A fresh program device is OFF at 0 W with a zero energy total.
    assert _state(hass, power_eid) == "0.0"
    assert _state(hass, energy_eid) == "0.0"


async def test_program_candidate_menu_lists_confirm(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
) -> None:
    """The Review candidate menu of a program candidate lists Confirm (D-05)."""
    cand = _program_candidate("sig-dish", "Dishwasher", dishwasher_stages())
    seed_candidates(hass_storage, [cand])
    energy_manager.data = std_prefs
    async_set_power(hass, "sensor.grid_power", BASE_W)
    async_set_power(hass, "sensor.solar_power", 0)
    async_set_power(hass, "sensor.battery_power", 0)
    async_set_power(hass, "sensor.washer_power", 200)
    entry = await async_setup_via_flow(hass)

    res_review = await async_open_options_step(hass, entry, "review")
    assert res_review["type"] is FlowResultType.FORM
    res_cand = await hass.config_entries.options.async_configure(
        res_review["flow_id"], {"candidate": "sig-dish"}
    )
    assert res_cand["type"] is FlowResultType.MENU
    assert res_cand["step_id"] == "candidate"
    assert "confirm" in res_cand["menu_options"]


@pytest.mark.parametrize(
    ("stages", "label"),
    [
        (dishwasher_stages(), "Dishwasher"),
        (washer_stages(), "Washing machine"),
        (dryer_stages(), "Dryer"),
        (hob_session_stages(), "Hob session"),
    ],
    ids=["dishwasher", "washer", "dryer", "hob_session"],
)
async def test_confirmed_program_tracks_stages(
    stages: list[tuple[float, int]],
    label: str,
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
) -> None:
    """A confirmed program device follows a live run through every stage (D-37).

    D-33: 0 W while OFF during the recognition lag, then the stage power with
    the back-dated pending energy, and 0 W on the matched terminal fall.
    """
    cand = _program_candidate("sig-prog", label, stages)
    entry, entities = await _setup(
        hass, energy_manager, std_prefs, hass_storage, freezer, [cand]
    )
    power_eid, energy_eid, _validation_eid = entities[label]
    devices = entry.runtime_data.devices
    dev_uuid = _device_uuid(entry, label)
    assert _state(hass, power_eid) == "0.0"

    # Sample windows of every stage in the driven run.
    windows: list[tuple[int, int, float]] = []
    offset = 0
    for level, count in stages:
        windows.append((offset, offset + count, float(level)))
        offset += count

    powers, energies = await _drive_samples(
        hass, freezer, _run_levels(stages), power_eid, energy_eid
    )

    # D-33: OFF at 0 W through the recognition lag, promoted inside stage 1.
    first_on = next(i for i, power in enumerate(powers) if power > 0.0)
    assert first_on >= 2, f"{label}: no recognition lag"
    assert first_on < windows[0][1], f"{label}: promoted after stage 1"
    assert all(power == 0.0 for power in powers[:first_on])

    # The pending energy is back-dated onto the promotion sample (D-02) and
    # the total never decreases (monotonic kWh).
    assert all(energy == 0.0 for energy in energies[:first_on])
    assert energies[first_on] > 0.0
    assert all(later >= earlier for earlier, later in zip(energies, energies[1:]))

    # Every stage is visited: a reading inside its learned level box (D-01).
    boxes = cand["session"]["stages"]
    for k, (start, end, level_w) in enumerate(windows):
        box = boxes[k]["level_w"]
        assert any(
            box["lo"] <= power <= box["hi"] for power in powers[start:end]
        ), f"{label}: stage {k} at {level_w} W never reported ({powers[start:end]})"

    # The matched terminal fall returns to 0 W and stops the energy (D-42).
    assert powers[-1] == 0.0
    assert powers[-2] == 0.0
    assert energies[-1] == energies[-2]
    assert devices.unavailable_reason(dev_uuid) is None

    # The committed energy is the run's energy (D-01 hybrid output).
    true_kwh = sum(level * count for level, count in stages) * GRID_S / 3600.0 / 1000.0
    assert 0.5 * true_kwh <= energies[-1] <= 1.5 * true_kwh


async def test_program_device_tracks_at_60s(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
) -> None:
    """A 60 s profile leaves a program device available while switched reads disabled_interval.

    The program device is gated on CYCLE_SHAPE (enabled at 60 s), the switched
    one on STEP (disabled above 45 s), and the D-34 interval issue names only
    the switched device with the STEP-derived limit (RESEARCH A5 Gate).
    """
    prog_stages = [(2000.0, 10), (150.0, 15), (2000.0, 8), (200.0, 10)]
    c_prog = _program_candidate("sig-prog", "Program dishwasher", prog_stages, grid_s=60)
    c_slow = make_switched_candidate("sig-slow", "Slow heater", rise_w=1200.0)
    c_slow["extraction"]["grid_s"] = 60
    entry, entities = await _setup(
        hass,
        energy_manager,
        std_prefs,
        hass_storage,
        freezer,
        [c_prog, c_slow],
        grid_s=60,
    )
    prog_power, prog_energy, _prog_val = entities["Program dishwasher"]
    slow_power, slow_energy, _slow_val = entities["Slow heater"]
    devices = entry.runtime_data.devices
    prog_uuid = _device_uuid(entry, "Program dishwasher")
    slow_uuid = _device_uuid(entry, "Slow heater")

    # Both models are compiled on the 60 s grid: the switched one is only
    # gated off by its own event type, not mismatched (D-26 vs D-34).
    await _warm_up_interval(hass, freezer, 60)
    assert devices.unavailable_reason(prog_uuid) is None
    assert devices.unavailable_reason(slow_uuid) == "disabled_interval"
    assert devices.detector_status == "running"
    assert _state(hass, prog_power) == "0.0"
    assert _state(hass, slow_power) == "unavailable"

    # The D-34 issue names the switched device only, with the STEP limit.
    issue = _issue(hass)
    assert issue is not None
    assert issue.translation_placeholders["devices"] == "Slow heater"
    expected_max = f"{EVENT_TYPE_MIN_S[EventType.STEP] / 2:g}"
    assert issue.translation_placeholders["max_interval"] == expected_max

    powers, energies = await _drive_samples(
        hass, freezer, _run_levels(prog_stages), prog_power, prog_energy, grid_s=60
    )
    assert not any(math.isnan(power) for power in powers), (
        f"program power went unavailable at 60 s: {powers} "
        f"(reason {devices.unavailable_reason(prog_uuid)})"
    )

    # The program device still tracks its stages at 60 s sampling.
    first_on = next(i for i, power in enumerate(powers) if power > 0.0)
    assert first_on >= 2
    assert all(power == 0.0 for power in powers[:first_on])
    assert energies[first_on] > 0.0
    assert energies[-1] > energies[first_on]
    assert all(later >= earlier for earlier, later in zip(energies, energies[1:]))
    assert powers[-1] == 0.0
    assert powers[-2] == 0.0
    assert devices.unavailable_reason(prog_uuid) is None

    # The switched device stayed unavailable with a frozen total throughout.
    assert devices.unavailable_reason(slow_uuid) == "disabled_interval"
    assert _state(hass, slow_power) == "unavailable"
    assert _state(hass, slow_energy) == "0.0"
    assert devices.detector_status == "running"


async def test_program_artefact_without_stages_reads_model_error(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
) -> None:
    """A stored program artefact without a stage template is model_error, never a crash (D-33)."""
    cand = _program_candidate("sig-dish", "Dishwasher", dishwasher_stages())
    cand["session"]["stages"] = None
    entry, entities = await _setup(
        hass, energy_manager, std_prefs, hass_storage, freezer, [cand]
    )
    power_eid, energy_eid, _validation_eid = entities["Dishwasher"]
    devices = entry.runtime_data.devices
    dev_uuid = _device_uuid(entry, "Dishwasher")

    assert devices.unavailable_reason(dev_uuid) == "model_error"
    assert _state(hass, power_eid) == "unavailable"
    assert _state(hass, energy_eid) == "0.0"

    # A run does not crash the hub and stays unmodelled.
    async_set_power(hass, "sensor.grid_power", BASE_W + 2000.0)
    await hass.async_block_till_done()
    await _drive_grid_steps(hass, freezer, 5)
    assert devices.unavailable_reason(dev_uuid) == "model_error"
    assert _state(hass, power_eid) == "unavailable"
    assert _state(hass, energy_eid) == "0.0"


# -- Task 2: stage-template identity and the D-40 merge restriction ---------


def test_artefact_states_differ_counts_program_stages() -> None:
    """_artefact_states_differ delegates to the stage-aware detection_differs."""
    from custom_components.powerguesser.devices import _artefact_states_differ
    from tests.fixtures.programs import dishwasher_stages

    active = _program_candidate("sig-p", "Program", dishwasher_stages())
    same = _program_candidate("sig-p", "Program", dishwasher_stages())
    other = _program_candidate("sig-p", "Program", [(900.0, 20), (300.0, 30)])

    assert _artefact_states_differ(active, same) is False
    assert _artefact_states_differ(active, other) is True
    assert _artefact_states_differ(None, active) is True


async def _open_device_menu(
    hass: HomeAssistant, entry: ConfigEntry, dev_uuid: str
) -> dict[str, Any]:
    """Open Review and pick a confirmed device entry; returns the device menu."""
    res_review = await async_open_options_step(hass, entry, "review")
    assert res_review["type"] is FlowResultType.FORM
    return await hass.config_entries.options.async_configure(
        res_review["flow_id"], {"candidate": f"device_{dev_uuid}"}
    )


async def test_program_device_cannot_be_merged(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
) -> None:
    """D-40: a program device is never a merge target or a merge source (Pitfall 9)."""
    c_prog = _program_candidate("sig-dish", "Dishwasher", dishwasher_stages())
    c_kettle = make_switched_candidate("sig-kettle", "Kettle", rise_w=2000.0)
    c_oven = make_switched_candidate("sig-oven", "Oven", rise_w=1800.0)
    entry, _entities = await _setup(
        hass, energy_manager, std_prefs, hass_storage, freezer, [c_prog, c_kettle, c_oven]
    )
    prog_uuid = _device_uuid(entry, "Dishwasher")
    kettle_uuid = _device_uuid(entry, "Kettle")
    oven_uuid = _device_uuid(entry, "Oven")

    # A program target offers no Merge at all, even with two switched devices
    # that could otherwise be sources.
    res_prog = await _open_device_menu(hass, entry, prog_uuid)
    assert res_prog["type"] is FlowResultType.MENU
    assert res_prog["step_id"] == "device"
    assert "merge" not in res_prog["menu_options"]

    # A switched target lists the other switched device, never the program one.
    res_kettle = await _open_device_menu(hass, entry, kettle_uuid)
    assert "merge" in res_kettle["menu_options"]
    res_merge = await hass.config_entries.options.async_configure(
        res_kettle["flow_id"], {"next_step_id": "merge"}
    )
    assert res_merge["type"] is FlowResultType.FORM
    source_opts = res_merge["data_schema"].schema["sources"].config["options"]
    assert [opt["value"] for opt in source_opts] == [f"device_{oven_uuid}"]


# -- Task 3: the D-26 drift window, the stage class and holdout backing ------


_PROG_CLASS = "program_1.5_3kw_multi_level"
_HOLDOUT_START = "2026-11-10T00:00:00+00:00"
_HOLDOUT_END = "2026-11-17T00:00:00+00:00"


def _class_entry(class_name: str, *, passes: bool = True) -> dict[str, Any]:
    """One class entry as Plan 05-02's report builder writes it."""
    return {
        "class": class_name,
        "peers": 3,
        "passing": 2 if passes else 1,
        "median_error_pct": 11.0 if passes else 30.0,
        "max_error_pct": 20.0 if passes else 38.0,
        "median_nde": 0.25,
        "median_mae_w": 180.0,
        "median_baseline_nde": 0.4,
        "n_events": 30,
        "peer_entities": ["sensor.plug_a_power", "sensor.plug_b_power"],
        "passes": passes,
    }


def _ok_report(classes: list[dict[str, Any]]) -> dict[str, Any]:
    """An ok report in the stored v1 shape (Plan 05-04 reads classes + holdout)."""
    return {
        "status": "ok",
        "reason": None,
        "train_cutoff": _HOLDOUT_START,
        "holdout": {"start": _HOLDOUT_START, "end": _HOLDOUT_END},
        "classes": classes,
        "devices": [],
        "refusals": [],
        "not_discovered": [],
    }


def _drift_window(tracker: Any) -> tuple[int, int, float]:
    """The tracker's own D-26 window constants (the kind selection, Plan 06-11)."""
    return (tracker._min_episodes, tracker._baseline_episodes, tracker._window_days)


async def _drive_switched_run(
    hass: HomeAssistant,
    freezer: Any,
    power_eid: str,
    energy_eid: str,
    *,
    rise_w: float = 2000.0,
) -> None:
    """Drive one switched episode (three samples on, three off) through the hub."""
    await _drive_samples(
        hass,
        freezer,
        [rise_w, rise_w, rise_w, 0.0, 0.0, 0.0],
        power_eid,
        energy_eid,
    )


async def test_program_device_uses_the_d26_drift_window(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
) -> None:
    """A program device drifts on the D-26 window (min 3, baseline 3, 42 days).

    Three closed program runs already read drift_evidence sufficient, while a
    switched device in the same hub with six episodes still reads insufficient
    on the D-15 window (8 / 10 / 14). The same window is restored after a
    reload and re-created by a version activation (Plan 06-11 kind=...).
    """
    from custom_components.powerguesser.engine.validation.constants import (
        LIVE_BASELINE_EPISODES,
        LIVE_MIN_EPISODES,
        LIVE_WINDOW_MAX_DAYS,
        PROGRAM_LIVE_BASELINE_EPISODES,
        PROGRAM_LIVE_MIN_EPISODES,
        PROGRAM_LIVE_WINDOW_MAX_DAYS,
    )

    c_prog = _program_candidate("sig-hob", "Hob session", hob_session_stages())
    # A 700 W switched rival: its rise box (630..770) contains none of the
    # hob's stage levels, so the +1800 W stage-1 entry edge is never ambiguous
    # between the program and the switched device (D-07).
    c_kettle = make_switched_candidate("sig-heater", "Heater", rise_w=700.0)
    entry, entities = await _setup(
        hass, energy_manager, std_prefs, hass_storage, freezer, [c_prog, c_kettle]
    )
    prog_power, prog_energy, prog_val = entities["Hob session"]
    kettle_power, kettle_energy, kettle_val = entities["Heater"]
    devices = entry.runtime_data.devices
    prog_uuid = _device_uuid(entry, "Hob session")
    kettle_uuid = _device_uuid(entry, "Heater")

    program_window = (
        PROGRAM_LIVE_MIN_EPISODES,
        PROGRAM_LIVE_BASELINE_EPISODES,
        PROGRAM_LIVE_WINDOW_MAX_DAYS,
    )
    switched_window = (LIVE_MIN_EPISODES, LIVE_BASELINE_EPISODES, LIVE_WINDOW_MAX_DAYS)

    # Three closed program runs and six closed switched episodes.
    for _ in range(3):
        await _drive_samples(
            hass, freezer, _run_levels(hob_session_stages()), prog_power, prog_energy
        )
    for _ in range(6):
        await _drive_switched_run(hass, freezer, kettle_power, kettle_energy, rise_w=700.0)

    assert _drift_window(devices._drift[prog_uuid]) == program_window
    assert _drift_window(devices._drift[kettle_uuid]) == switched_window
    assert hass.states.get(prog_val).attributes["drift_evidence"] == "sufficient"
    assert hass.states.get(kettle_val).attributes["drift_evidence"] == "insufficient"

    # A reload restores the program tracker with the record's kind.
    await devices._async_save_hot()
    await hass.config_entries.async_reload(entry.entry_id)
    await hass.async_block_till_done()
    devices2 = entry.runtime_data.devices
    restored = devices2._drift.get(prog_uuid)
    assert restored is not None
    assert _drift_window(restored) == program_window
    assert _drift_window(devices2._drift[kettle_uuid]) == switched_window
    assert hass.states.get(prog_val).attributes["drift_evidence"] == "sufficient"

    # A version activation starts the program device over on the program window.
    await devices2.async_activate_version(prog_uuid, 1)
    await hass.async_block_till_done()
    reset = devices2._drift[prog_uuid]
    assert reset.to_dict()["evidence"] == []
    assert reset.to_dict()["baseline"] is None
    assert _drift_window(reset) == program_window
    assert hass.states.get(prog_val).attributes["drift_evidence"] == "insufficient"


async def test_program_class_from_stage_template_and_holdout_backed(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
) -> None:
    """A dishwasher reads its stage-based class and is holdout-backed by a passing class.

    The class comes from the stage template (Plan 06-11 derive_class), never
    from the degenerate session rise boxes (RESEARCH Pitfall 1), and a passing
    class makes the device show the class median with basis class_holdout
    (P5 D-09/D-20).
    """
    cand = _program_candidate("sig-dish", "Dishwasher", dishwasher_stages())
    entry, entities = await _setup(
        hass,
        energy_manager,
        std_prefs,
        hass_storage,
        freezer,
        [cand],
        report=_ok_report([_class_entry(_PROG_CLASS)]),
    )
    power_eid, energy_eid, validation_eid = entities["Dishwasher"]
    dev_uuid = _device_uuid(entry, "Dishwasher")
    devices = entry.runtime_data.devices

    assert devices.record(dev_uuid)["validation"]["class"] == _PROG_CLASS
    attrs = hass.states.get(power_eid).attributes
    assert attrs["validation_class"] == _PROG_CLASS
    assert attrs["validation_basis"] == "class_holdout"
    assert attrs["validation_error_pct"] == 11.0
    assert attrs["validation_peers"] == 3
    assert attrs["unvalidated"] is False

    # The energy sensor publishes the same figure, and the diagnostic sensor
    # carries the class provenance.
    assert hass.states.get(energy_eid).attributes["validation_basis"] == "class_holdout"
    diagnostic = hass.states.get(validation_eid).attributes
    assert diagnostic["validation_class"] == _PROG_CLASS
    assert diagnostic["unvalidated"] is False
    assert diagnostic["validated_via"] == _PROG_CLASS


async def test_program_device_failing_class_is_not_holdout_backed(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
) -> None:
    """A failing class leaves a program device unvalidated with no figure (P5 D-20)."""
    cand = _program_candidate("sig-dish", "Dishwasher", dishwasher_stages())
    entry, entities = await _setup(
        hass,
        energy_manager,
        std_prefs,
        hass_storage,
        freezer,
        [cand],
        report=_ok_report([_class_entry(_PROG_CLASS, passes=False)]),
    )
    power_eid, _energy_eid, _validation_eid = entities["Dishwasher"]
    dev_uuid = _device_uuid(entry, "Dishwasher")

    assert entry.runtime_data.devices.record(dev_uuid)["validation"]["class"] == _PROG_CLASS
    attrs = hass.states.get(power_eid).attributes
    assert attrs["validation_class"] == _PROG_CLASS
    assert attrs["unvalidated"] is True
    assert attrs["validation_basis"] is None
    assert attrs["validation_error_pct"] is None

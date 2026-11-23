"""Tests for merging confirmed devices: retire-in-place merge (D-12, D-23, OUT-02, OUT-05).

Plan 04-11 Task 1: merging one confirmed device into another retires the
absorbed device in place — its record becomes status ``retired`` with
``retired_into`` the survivor, its signature and aliases become aliases of the
survivor (identity routing only, never detection), its power sensor becomes
unavailable (reason "retired"), its energy sensor keeps its last value with
the ``retired`` attribute, and neither energy counter ever moves (no summing,
no decreases, no jumps). The device menu offers Merge while another confirmed
device exists as a source; retired devices are never listed in Review.

Plan 04-11 Task 2 adds the retired lifecycle tests (Energy-dashboard Repairs
hint, reload, deletion, later learning runs).
"""

import copy
import json
import math
from datetime import timedelta
from typing import Any

import pytest

from homeassistant.components.energy.data import EnergyManager
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers import issue_registry as ir
from homeassistant.util import dt as dt_util
from pytest_homeassistant.common import async_fire_time_changed

from custom_components.powerguesser.const import DOMAIN
from tests.conftest import (
    async_confirm_candidate,
    async_open_options_step,
    async_set_power,
    async_setup_via_flow,
    make_switched_candidate,
    mock_llm,
    seed_candidates,
)
from tests.fixtures.learning.fake_llm import chat_completion

GRID_S = 30


def _aligned_move(freezer: Any, grid_s: int = GRID_S) -> None:
    """Move the freezer to the next epoch multiple of the grid."""
    now = dt_util.utcnow().timestamp()
    aligned = math.ceil(now / grid_s) * grid_s
    freezer.move_to(dt_util.utc_from_timestamp(aligned))


def _pin_grid(entry: ConfigEntry, grid_s: int = GRID_S) -> None:
    """Publish a profile snapshot pinning the D-26 grid."""
    from custom_components.powerguesser.engine.intervals import (
        ProfileSnapshot,
        gate,
    )

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


async def _tick(hass: HomeAssistant, freezer: Any, seconds: int) -> None:
    """Advance the freezer and fire the time-changed event."""
    freezer.tick(timedelta(seconds=seconds))
    async_fire_time_changed(hass, dt_util.utcnow())
    await hass.async_block_till_done()


async def _drive_grid(hass: HomeAssistant, freezer: Any, n: int, grid_s: int = GRID_S) -> None:
    """Advance n full grid steps via the 10 s tick path."""
    for _ in range(n * 3):
        await _tick(hass, freezer, 10)


async def _load_on(hass: HomeAssistant, freezer: Any, grid_w: float) -> None:
    """Raise grid power by one device's step and settle three grid points.

    Ages the gross sources past the aligner's on-change window (45 s, D-19)
    first: the initial samples carry the real pre-alignment timestamp, so
    without the ageing the settle release can land on the last drive tick,
    where the raised value is overwritten by the base value before any grid
    point drains — the device then never sees the step (flaky energy 0).
    """
    await _drive_grid(hass, freezer, 2)
    async_set_power(hass, "sensor.grid_power", grid_w)
    await hass.async_block_till_done()
    await _drive_grid(hass, freezer, 3)


async def _load_off(hass: HomeAssistant, freezer: Any) -> None:
    """Return grid power to base load and settle three grid points."""
    async_set_power(hass, "sensor.grid_power", 500)
    await hass.async_block_till_done()
    await _drive_grid(hass, freezer, 3)


async def _setup_two_devices(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
) -> tuple[ConfigEntry, dict[str, tuple[str, str, str]]]:
    """Entry with two confirmed devices (Kettle 2000 W, Water heater 3000 W).

    Returns (entry, {label: (power_eid, energy_eid, device_uuid)}).
    """
    seed_candidates(
        hass_storage,
        [
            make_switched_candidate("sig-kettle", "Kettle", rise_w=2000.0),
            make_switched_candidate("sig-heater", "Water heater", rise_w=3000.0),
        ],
    )
    energy_manager.data = std_prefs
    async_set_power(hass, "sensor.grid_power", 500)
    async_set_power(hass, "sensor.solar_power", 0)
    async_set_power(hass, "sensor.battery_power", 0)
    async_set_power(hass, "sensor.washer_power", 200)
    _aligned_move(freezer)
    entry = await async_setup_via_flow(hass)
    _pin_grid(entry)
    await hass.async_block_till_done()

    ent_reg = er.async_get(hass)
    entities: dict[str, tuple[str, str, str]] = {}
    for sig_id, label in (("sig-kettle", "Kettle"), ("sig-heater", "Water heater")):
        await async_confirm_candidate(hass, entry, sig_id, label)
        sub = next(s for s in entry.get_subentries_of_type("device") if s.title == label)
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


async def _open_device_menu(
    hass: HomeAssistant, entry: ConfigEntry, dev_uuid: str
) -> dict[str, Any]:
    """Open Review and pick the device entry; returns the device menu result."""
    res_review = await async_open_options_step(hass, entry, "review")
    assert res_review["type"] is FlowResultType.FORM
    return await hass.config_entries.options.async_configure(
        res_review["flow_id"], {"candidate": f"device_{dev_uuid}"}
    )


async def _merge_devices(
    hass: HomeAssistant,
    entry: ConfigEntry,
    target_uuid: str,
    source_uuid: str,
) -> dict[str, Any]:
    """Open the target's device menu, pick Merge and submit the source device."""
    res_dev = await _open_device_menu(hass, entry, target_uuid)
    assert res_dev["type"] is FlowResultType.MENU
    assert "merge" in res_dev["menu_options"]
    res_merge = await hass.config_entries.options.async_configure(
        res_dev["flow_id"], {"next_step_id": "merge"}
    )
    assert res_merge["type"] is FlowResultType.FORM
    assert res_merge["step_id"] == "merge"
    return await hass.config_entries.options.async_configure(
        res_merge["flow_id"], {"sources": [f"device_{source_uuid}"]}
    )


def _state(hass: HomeAssistant, entity_id: str) -> Any:
    return hass.states.get(entity_id)


def _energy_kwh(hass: HomeAssistant, entity_id: str) -> float:
    return float(_state(hass, entity_id).state)


async def test_merge_confirmed_devices_retires_source(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
) -> None:
    """Merging the water heater into the kettle retires it in place: the target
    keeps versions, name, energy total and unique ids; the source record reads
    retired with retired_into; aliases and review decisions absorb the source."""
    entry, entities = await _setup_two_devices(
        hass, energy_manager, std_prefs, hass_storage, freezer
    )
    devices = entry.runtime_data.devices
    power_a, energy_a, uuid_a = entities["Kettle"]
    power_b, energy_b, uuid_b = entities["Water heater"]

    # The absorbed device carries a pre-existing alias (a previous candidate
    # merge): the survivor must absorb that alias too.
    devices.record(uuid_b)["aliases"].append("sig-heater-old")

    # Both devices accumulate energy before the merge.
    await _load_on(hass, freezer, 2500)
    await _load_off(hass, freezer)
    await _load_on(hass, freezer, 3500)
    await _load_off(hass, freezer)
    energy_a_before = _energy_kwh(hass, energy_a)
    energy_b_before = _energy_kwh(hass, energy_b)
    assert energy_a_before > 0.0 and energy_b_before > 0.0
    rec_a_before = {
        k: v for k, v in devices.record(uuid_a).items() if k != "aliases"
    }

    # The merge source selector lists the other confirmed device only.
    res_dev = await _open_device_menu(hass, entry, uuid_a)
    assert res_dev["type"] is FlowResultType.MENU
    assert "merge" in res_dev["menu_options"]
    res_merge = await hass.config_entries.options.async_configure(
        res_dev["flow_id"], {"next_step_id": "merge"}
    )
    src_opts = res_merge["data_schema"].schema["sources"].config["options"]
    assert [o["value"] for o in src_opts] == [f"device_{uuid_b}"]
    assert src_opts[0]["label"] == "Water heater — confirmed device"

    res_after = await hass.config_entries.options.async_configure(
        res_merge["flow_id"], {"sources": [f"device_{uuid_b}"]}
    )
    assert res_after["type"] is FlowResultType.FORM
    assert res_after["step_id"] == "review"

    # The target is unchanged except for the absorbed aliases.
    rec_a = devices.record(uuid_a)
    assert {k: v for k, v in rec_a.items() if k != "aliases"} == rec_a_before
    assert rec_a["aliases"] == ["sig-heater", "sig-heater-old"]
    assert devices.energy_wh(uuid_a) == pytest.approx(energy_a_before * 1000.0, abs=1.0)
    assert _device_ids(hass, entry, uuid_a) == {
        "power": power_a,
        "energy": energy_a,
    }

    # The source is retired in place.
    rec_b = devices.record(uuid_b)
    assert rec_b["status"] == "retired"
    assert rec_b["retired_into"] == uuid_a
    assert rec_b["pending_proposal"] is None

    # Review: the source decision is merged into the target signature; the
    # target decision carries the absorbed signature as an alias.
    decisions = entry.runtime_data.learning.review.decisions
    assert decisions["sig-heater"]["status"] == "merged"
    assert decisions["sig-heater"]["merged_into"] == "sig-kettle"
    assert "sig-heater" in decisions["sig-kettle"]["aliases"]


def _device_ids(
    hass: HomeAssistant, entry: ConfigEntry, dev_uuid: str
) -> dict[str, str]:
    ent_reg = er.async_get(hass)
    return {
        "power": ent_reg.async_get_entity_id(
            "sensor", DOMAIN, f"{entry.entry_id}_{dev_uuid}_power"
        ),
        "energy": ent_reg.async_get_entity_id(
            "sensor", DOMAIN, f"{entry.entry_id}_{dev_uuid}_energy"
        ),
    }


async def test_retired_power_unavailable_energy_frozen(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
) -> None:
    """After the merge: the retired device's power sensor is unavailable, its
    energy sensor keeps exactly its pre-merge value with retired true, and a
    later run of its load moves neither counter."""
    entry, entities = await _setup_two_devices(
        hass, energy_manager, std_prefs, hass_storage, freezer
    )
    devices = entry.runtime_data.devices
    power_a, energy_a, uuid_a = entities["Kettle"]
    power_b, energy_b, uuid_b = entities["Water heater"]

    await _load_on(hass, freezer, 3500)
    await _load_off(hass, freezer)
    energy_b_before = _energy_kwh(hass, energy_b)
    energy_a_before = _energy_kwh(hass, energy_a)
    assert energy_b_before > 0.0

    await _merge_devices(hass, entry, uuid_a, uuid_b)
    devices = entry.runtime_data.devices

    # Power unavailable, energy frozen at the pre-merge value. (HA does not
    # write custom attributes of an unavailable entity, so the power sensor's
    # retired flag is asserted at the runtime seam; the energy sensor is
    # available and carries it on the state.)
    assert _state(hass, power_b).state == "unavailable"
    energy_state = _state(hass, energy_b)
    assert energy_state.state != "unavailable"
    assert float(energy_state.state) == pytest.approx(energy_b_before, abs=1e-6)
    assert energy_state.attributes["retired"] is True
    assert devices.attributes(uuid_b)["retired"] is True

    # A later run of the absorbed load is not modelled: neither counter moves.
    await _load_on(hass, freezer, 3500)
    await _drive_grid(hass, freezer, 4)
    assert float(_state(hass, energy_a).state) == pytest.approx(
        energy_a_before, abs=1e-6
    )
    assert float(_state(hass, energy_b).state) == pytest.approx(
        energy_b_before, abs=1e-6
    )
    assert _state(hass, power_a).state != "unavailable"
    await _load_off(hass, freezer)

    # The survivor still detects its own load.
    await _load_on(hass, freezer, 2500)
    assert float(_state(hass, power_a).state) == pytest.approx(2000.0, abs=1.0)
    await _load_off(hass, freezer)


async def test_merge_counters_never_summed_or_decreased(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
) -> None:
    """Over the merge both counters stay flat; afterwards they only ever grow:
    the survivor's total never jumps by the absorbed device's total (OUT-02)."""
    entry, entities = await _setup_two_devices(
        hass, energy_manager, std_prefs, hass_storage, freezer
    )
    power_a, energy_a_eid, uuid_a = entities["Kettle"]
    _power_b, energy_b_eid, uuid_b = entities["Water heater"]
    del power_a

    await _load_on(hass, freezer, 2500)
    await _load_off(hass, freezer)
    await _load_on(hass, freezer, 3500)
    await _load_off(hass, freezer)
    a0 = _energy_kwh(hass, energy_a_eid)
    b0 = _energy_kwh(hass, energy_b_eid)

    await _merge_devices(hass, entry, uuid_a, uuid_b)

    # The merge itself never touches either counter.
    assert _energy_kwh(hass, energy_a_eid) == pytest.approx(a0, abs=1e-6)
    assert _energy_kwh(hass, energy_b_eid) == pytest.approx(b0, abs=1e-6)

    # Later runs only add the survivor's own contribution: both counters
    # never decrease, the absorbed counter stays frozen and the survivor's
    # total never jumps by the absorbed total (one kettle run accrues at most
    # ~2000 W x 180 s including back-dating, so 0.125 kWh is a safe bound).
    a_prev = a0
    for i in range(2):
        await _load_on(hass, freezer, 2500)
        await _load_off(hass, freezer)
        a_now = _energy_kwh(hass, energy_a_eid)
        b_now = _energy_kwh(hass, energy_b_eid)
        assert a_prev <= a_now <= a0 + 0.125 * (i + 1)
        assert b_now == pytest.approx(b0, abs=1e-6)
        a_prev = a_now


async def test_device_menu_merge_needs_a_source(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
) -> None:
    """With one confirmed device and no visible candidate the device menu has
    no Merge; with a second confirmed device it offers Merge."""
    seed_candidates(
        hass_storage,
        [make_switched_candidate("sig-kettle", "Kettle", rise_w=2000.0)],
    )
    energy_manager.data = std_prefs
    async_set_power(hass, "sensor.grid_power", 500)
    async_set_power(hass, "sensor.solar_power", 0)
    async_set_power(hass, "sensor.battery_power", 0)
    async_set_power(hass, "sensor.washer_power", 200)
    _aligned_move(freezer)
    entry = await async_setup_via_flow(hass)
    _pin_grid(entry)
    await hass.async_block_till_done()

    await async_confirm_candidate(hass, entry, "sig-kettle", "Kettle")
    sub = next(s for s in entry.get_subentries_of_type("device") if s.title == "Kettle")
    uuid_kettle = sub.data["device_uuid"]

    res_dev = await _open_device_menu(hass, entry, uuid_kettle)
    assert res_dev["type"] is FlowResultType.MENU
    assert "merge" not in res_dev["menu_options"]

    # Confirm a second device: Merge appears on the first device's menu.
    devices = entry.runtime_data.devices
    await devices.async_confirm(
        make_switched_candidate("sig-heater", "Water heater", rise_w=3000.0),
        "Water heater",
        preview=False,
        trained_at=None,
    )
    res_dev = await _open_device_menu(hass, entry, uuid_kettle)
    assert "merge" in res_dev["menu_options"]
    res_dev2 = await _open_device_menu(
        hass, entry, entry.runtime_data.devices.device_for_signature("sig-heater")
    )
    assert "merge" in res_dev2["menu_options"]


async def test_retired_not_listed(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
) -> None:
    """A retired device is not listed in Review: neither as a device entry nor
    through its signature id."""
    entry, entities = await _setup_two_devices(
        hass, energy_manager, std_prefs, hass_storage, freezer
    )
    uuid_a = entities["Kettle"][2]
    uuid_b = entities["Water heater"][2]

    await _merge_devices(hass, entry, uuid_a, uuid_b)

    res_review = await async_open_options_step(hass, entry, "review")
    assert res_review["type"] is FlowResultType.FORM
    opts = res_review["data_schema"].schema["candidate"].config["options"]
    values = [o["value"] for o in opts]
    assert values == [f"device_{uuid_a}"]
    assert "sig-heater" not in values
    assert f"device_{uuid_b}" not in values


async def test_retired_availability_polarity(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
) -> None:
    """Availability polarity (D-12, D-33/D-34): a retired device reads
    "retired" with its power sensor unavailable and its energy sensor available
    at the frozen total — before and while the step type is disabled. A
    confirmed device under a disabled step type reads disabled_interval with
    its power sensor unavailable and its energy available, and the
    detection_interval_too_slow issue names it but not the retired device."""
    entry, entities = await _setup_two_devices(
        hass, energy_manager, std_prefs, hass_storage, freezer
    )
    power_a, energy_a, uuid_a = entities["Kettle"]
    power_b, energy_b, uuid_b = entities["Water heater"]

    await _load_on(hass, freezer, 3500)
    await _load_off(hass, freezer)
    energy_b_before = _energy_kwh(hass, energy_b)
    assert energy_b_before > 0.0

    # Before the merge the retired reason does not exist; the devices are healthy.
    devices = entry.runtime_data.devices
    assert devices.unavailable_reason(uuid_b) is None

    await _merge_devices(hass, entry, uuid_a, uuid_b)
    assert devices.unavailable_reason(uuid_b) == "retired"
    assert devices.power_available(uuid_b) is False
    assert devices.energy_available(uuid_b) is True

    # Disable the step type (a 60 s effective interval gates STEP off, D-34).
    _pin_grid(entry, 60)
    await hass.async_block_till_done()

    # Retired device: still "retired", power unavailable, energy available.
    assert devices.unavailable_reason(uuid_b) == "retired"
    assert _state(hass, power_b).state == "unavailable"
    energy_state = _state(hass, energy_b)
    assert energy_state.state != "unavailable"
    assert float(energy_state.state) == pytest.approx(energy_b_before, abs=1e-6)

    # Confirmed device: disabled_interval, power unavailable, energy available.
    assert devices.unavailable_reason(uuid_a) == "disabled_interval"
    assert _state(hass, power_a).state == "unavailable"
    assert _state(hass, energy_a).state != "unavailable"

    # The D-34 issue names the confirmed device and not the retired one.
    issue_reg = ir.async_get(hass)
    issue = issue_reg.async_get_issue(DOMAIN, "detection_interval_too_slow")
    assert issue is not None
    assert issue.translation_placeholders is not None
    assert "Kettle" in issue.translation_placeholders["devices"]
    assert "Water heater" not in issue.translation_placeholders["devices"]


# --- Task 2: a retired device stays honest over time (TDD, D-12) ------------


def _kettle_and_heater_responder() -> Any:
    """Responder proposing two switched devices: kettle (~2000 W) + water heater (~3000 W)."""

    def _responder(data: Any) -> dict[str, Any]:
        from tests.ha.test_device_proposals import _digest_from_request

        digest = _digest_from_request(data)
        clusters = digest["clusters"]
        ck = min(clusters, key=lambda c: abs(c["rise_w"]["p50"] - 2000))["id"]
        cw = min(clusters, key=lambda c: abs(c["rise_w"]["p50"] - 3000))["id"]
        prop = {
            "analysis": "Kettle and water heater.",
            "schema_version": 1,
            "devices": [
                {
                    "ref": "d1",
                    "label": "Kettle",
                    "category": "kettle",
                    "tracker_kind": "switched",
                    "member_clusters": [ck],
                    "states": [
                        {"name": "heat", "cluster_ref": ck, "role": "main"}
                    ],
                    "phase_hint": "single",
                    "confidence": "high",
                    "rationale": "2kW resistive step typical of a kettle.",
                    "ambiguous_with": [],
                    "reference_plug": "",
                },
                {
                    "ref": "d2",
                    "label": "Water heater",
                    "category": "water_heater",
                    "tracker_kind": "switched",
                    "member_clusters": [cw],
                    "states": [
                        {"name": "tank", "cluster_ref": cw, "role": "main"}
                    ],
                    "phase_hint": "single",
                    "confidence": "high",
                    "rationale": "3kW resistive step typical of a water heater.",
                    "ambiguous_with": [],
                    "reference_plug": "",
                },
            ],
            "rejected_clusters": [],
            "merge_suggestions": [],
        }
        return {"status": 200, "json": chat_completion(json.dumps(prop))}

    return _responder


def _changed_kettle_and_heater_responder() -> Any:
    """Run-2 responder: the kettle device is proposed with a changed state
    (its artefact differs from the confirmed version -> a proposal), and the
    water-heater load is proposed again (its artefact matches the absorbed
    signature). Each device claims its own cluster."""

    def _responder(data: Any) -> dict[str, Any]:
        from tests.ha.test_device_proposals import _digest_from_request

        digest = _digest_from_request(data)
        clusters = digest["clusters"]
        ck = min(clusters, key=lambda c: abs(c["rise_w"]["p50"] - 2000))["id"]
        cw = min(clusters, key=lambda c: abs(c["rise_w"]["p50"] - 3000))["id"]
        prop = {
            "analysis": "Kettle state renamed; water heater still present.",
            "schema_version": 1,
            "devices": [
                {
                    "ref": "d1",
                    "label": "Kettle",
                    "category": "kettle",
                    "tracker_kind": "switched",
                    "member_clusters": [ck],
                    "states": [
                        {"name": "boil", "cluster_ref": ck, "role": "main"}
                    ],
                    "phase_hint": "single",
                    "confidence": "high",
                    "rationale": "2kW resistive step, state renamed after refit.",
                    "ambiguous_with": [],
                    "reference_plug": "",
                },
                {
                    "ref": "d2",
                    "label": "Water heater",
                    "category": "water_heater",
                    "tracker_kind": "switched",
                    "member_clusters": [cw],
                    "states": [
                        {"name": "tank", "cluster_ref": cw, "role": "main"}
                    ],
                    "phase_hint": "single",
                    "confidence": "high",
                    "rationale": "3kW resistive step typical of a water heater.",
                    "ambiguous_with": [],
                    "reference_plug": "",
                },
            ],
            "rejected_clusters": [],
            "merge_suggestions": [],
        }
        return {"status": 200, "json": chat_completion(json.dumps(prop))}

    return _responder


async def test_retired_energy_in_prefs_raises_and_clears_issue(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
) -> None:
    """A retired device whose energy entity is still in Energy prefs
    device_consumption raises retired_device_in_energy_<uuid> naming both
    devices; removing it from the prefs clears the issue; a confirmed device
    never raises it."""
    entry, entities = await _setup_two_devices(
        hass, energy_manager, std_prefs, hass_storage, freezer
    )
    _power_a, energy_a, uuid_a = entities["Kettle"]
    _power_b, energy_b, uuid_b = entities["Water heater"]
    devices = entry.runtime_data.devices
    issue_reg = ir.async_get(hass)

    # The confirmed device's energy entity in the prefs raises nothing.
    prefs = copy.deepcopy(std_prefs)
    prefs["device_consumption"].append(
        {"stat_consumption": energy_b, "name": "Water heater"}
    )
    await energy_manager.async_update(prefs)
    await hass.async_block_till_done()
    assert (
        issue_reg.async_get_issue(DOMAIN, f"retired_device_in_energy_{uuid_b}")
        is None
    )

    await _merge_devices(hass, entry, uuid_a, uuid_b)

    # Retired and still listed: WARNING issue naming both devices.
    issue = issue_reg.async_get_issue(
        DOMAIN, f"retired_device_in_energy_{uuid_b}"
    )
    assert issue is not None
    assert issue.severity == ir.IssueSeverity.WARNING
    assert issue.translation_placeholders["name"] == "Water heater"
    assert issue.translation_placeholders["entity"] == energy_b
    assert issue.translation_placeholders["target"] == "Kettle"
    assert (
        issue_reg.async_get_issue(DOMAIN, f"retired_device_in_energy_{uuid_a}")
        is None
    )

    # Removing the entity from the prefs clears the issue.
    prefs2 = copy.deepcopy(prefs)
    prefs2["device_consumption"] = [
        item
        for item in prefs2["device_consumption"]
        if item.get("stat_consumption") != energy_b
    ]
    await energy_manager.async_update(prefs2)
    await hass.async_block_till_done()
    assert (
        issue_reg.async_get_issue(DOMAIN, f"retired_device_in_energy_{uuid_b}")
        is None
    )


async def test_retired_device_survives_reload(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
) -> None:
    """Reload: the retired device's energy state is the same value, its power
    sensor is still unavailable and the Energy-dashboard issue state is
    unchanged."""
    entry, entities = await _setup_two_devices(
        hass, energy_manager, std_prefs, hass_storage, freezer
    )
    power_b, energy_b, uuid_b = entities["Water heater"]
    uuid_a = entities["Kettle"][2]

    await _load_on(hass, freezer, 3500)
    await _load_off(hass, freezer)
    energy_before = _energy_kwh(hass, energy_b)

    prefs = copy.deepcopy(std_prefs)
    prefs["device_consumption"].append(
        {"stat_consumption": energy_b, "name": "Water heater"}
    )
    await energy_manager.async_update(prefs)
    await hass.async_block_till_done()

    await _merge_devices(hass, entry, uuid_a, uuid_b)
    issue_reg = ir.async_get(hass)
    assert (
        issue_reg.async_get_issue(DOMAIN, f"retired_device_in_energy_{uuid_b}")
        is not None
    )

    await hass.config_entries.async_reload(entry.entry_id)
    await hass.async_block_till_done()

    assert _state(hass, power_b).state == "unavailable"
    assert float(_state(hass, energy_b).state) == pytest.approx(
        energy_before, abs=1e-6
    )
    assert (
        issue_reg.async_get_issue(DOMAIN, f"retired_device_in_energy_{uuid_b}")
        is not None
    )


async def test_delete_retired_device_removes_record_no_candidate(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
) -> None:
    """Deleting the retired device's subentry removes its record entirely (no
    candidate returns to Review), the survivor keeps the alias and the
    retired-device Repairs issue is deleted."""
    entry, entities = await _setup_two_devices(
        hass, energy_manager, std_prefs, hass_storage, freezer
    )
    _power_a, energy_a, uuid_a = entities["Kettle"]
    _power_b, energy_b, uuid_b = entities["Water heater"]
    devices = entry.runtime_data.devices

    prefs = copy.deepcopy(std_prefs)
    prefs["device_consumption"].append(
        {"stat_consumption": energy_b, "name": "Water heater"}
    )
    await energy_manager.async_update(prefs)
    await hass.async_block_till_done()

    await _merge_devices(hass, entry, uuid_a, uuid_b)
    issue_reg = ir.async_get(hass)
    assert (
        issue_reg.async_get_issue(DOMAIN, f"retired_device_in_energy_{uuid_b}")
        is not None
    )

    # The user deletes the retired device under Settings.
    sub = next(
        s
        for s in entry.get_subentries_of_type("device")
        if s.data.get("device_uuid") == uuid_b
    )
    hass.config_entries.async_remove_subentry(entry, sub.subentry_id)
    await hass.async_block_till_done()

    # The record is removed entirely; nothing returns to Review.
    assert devices.record(uuid_b) is None
    res_review = await async_open_options_step(hass, entry, "review")
    assert res_review["type"] is FlowResultType.FORM
    values = [
        o["value"] for o in res_review["data_schema"].schema["candidate"].config["options"]
    ]
    assert f"device_{uuid_b}" not in values
    assert "sig-heater" not in values

    # The survivor keeps the absorbed alias; the issue for B is deleted.
    assert "sig-heater" in devices.record(uuid_a)["aliases"]
    assert (
        issue_reg.async_get_issue(DOMAIN, f"retired_device_in_energy_{uuid_b}")
        is None
    )


# --- Plan 04-12: merging an unconfirmed candidate into a confirmed device
# retrains it locally over the union of both signatures' episodes (D-13) -----


async def test_merge_candidate_into_device_runs_union_refit(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
    aioclient_mock: Any,
    inprocess_worker: Any,
) -> None:
    """Merging a duplicate kettle candidate into the confirmed Kettle device
    records the merge, stores the candidate's artefact as an alias, runs the
    union retrain in the worker, and — accepted — appends version 2 with
    source "union": unique_ids, entity ids and energy totals never move."""
    from tests.conftest import async_fill_synth_buffer
    from tests.fixtures.learning.synth import SynthSpec

    cand_main = make_switched_candidate("sig-kettle-2k", "Kettle", rise_w=2000.0)
    # Narrow, fitted-style boxes (a subset of what the union refit will fit),
    # so the union version claims a superset of the current validation
    # episodes and the keep-or-reject rule deterministically accepts it.
    st = cand_main["states"][0]
    tight_rise = {"lo": 1950.0, "p50": 2000.0, "hi": 2050.0}
    st["rise_w"] = dict(tight_rise)
    st["fall_w"] = dict(tight_rise)
    st["plateau_w"] = dict(tight_rise)
    st["dur_s"] = {"lo": 160.0, "p50": 180.0, "hi": 200.0}
    cand_main["fingerprint"] = {"kind": "switched", "states": [st]}
    seed_candidates(
        hass_storage,
        [
            cand_main,
            make_switched_candidate("sig-kettle-2150", "Kettle 2nd", rise_w=2150.0),
        ],
    )
    energy_manager.data = std_prefs
    async_set_power(hass, "sensor.grid_power", 500)
    async_set_power(hass, "sensor.solar_power", 0)
    async_set_power(hass, "sensor.battery_power", 0)
    async_set_power(hass, "sensor.washer_power", 200)
    _aligned_move(freezer)
    entry = await async_setup_via_flow(hass)
    _pin_grid(entry)
    await async_fill_synth_buffer(hass, entry, spec=SynthSpec(days=22))
    await hass.async_block_till_done()

    await async_confirm_candidate(hass, entry, "sig-kettle-2k", "Kettle")
    sub = next(s for s in entry.get_subentries_of_type("device") if s.title == "Kettle")
    dev_uuid = sub.data["device_uuid"]
    devices = entry.runtime_data.devices
    sup = entry.runtime_data.learning

    # The device runs once so the energy total is non-trivial before the merge.
    await _load_on(hass, freezer, 2500)
    await _load_off(hass, freezer)
    ent_reg = er.async_get(hass)
    power_eid = ent_reg.async_get_entity_id(
        "sensor", DOMAIN, f"{entry.entry_id}_{dev_uuid}_power"
    )
    energy_eid = ent_reg.async_get_entity_id(
        "sensor", DOMAIN, f"{entry.entry_id}_{dev_uuid}_energy"
    )
    energy_before = _energy_kwh(hass, energy_eid)
    assert energy_before > 0.0
    ids_before = _device_ids(hass, entry, dev_uuid)
    versions_before = len(devices.record(dev_uuid)["versions"])

    # Merge the unconfirmed candidate into the device through the Review flow.
    res_dev = await _open_device_menu(hass, entry, dev_uuid)
    assert res_dev["type"] is FlowResultType.MENU
    assert "merge" in res_dev["menu_options"]
    res_merge = await hass.config_entries.options.async_configure(
        res_dev["flow_id"], {"next_step_id": "merge"}
    )
    assert res_merge["type"] is FlowResultType.FORM
    assert res_merge["step_id"] == "merge"
    res_after = await hass.config_entries.options.async_configure(
        res_merge["flow_id"], {"sources": ["sig-kettle-2150"]}
    )
    assert res_after["type"] is FlowResultType.FORM
    assert res_after["step_id"] == "review"

    # The union retrain ran in the background worker and completed.
    assert sup.run_task is not None
    await sup.run_task

    # The merge decision is recorded; the alias carries the candidate artefact.
    decisions = sup.review.decisions
    assert decisions["sig-kettle-2150"]["status"] == "merged"
    assert decisions["sig-kettle-2150"]["merged_into"] == "sig-kettle-2k"
    rec = devices.record(dev_uuid)
    assert "sig-kettle-2150" in rec["aliases"]
    alias_art = rec["alias_artefacts"]["sig-kettle-2150"]
    assert alias_art["signature_id"] == "sig-kettle-2150"
    assert alias_art["states"][0]["rise_w"]["p50"] == pytest.approx(2150.0)

    # An accepted union version became active; identity and counters untouched.
    assert rec["pending_refit"] is False
    assert rec["last_refit"]["accepted"] is True
    assert rec["last_refit"]["new_S"] >= rec["last_refit"]["current_S"]
    assert len(rec["versions"]) == versions_before + 1
    union_v = rec["versions"][-1]
    assert union_v["source"] == "union"
    assert union_v["v"] == rec["active_v"]
    assert union_v["artefact"]["provenance"]["source"] == "union"
    assert union_v["artefact"]["signature_id"] == "sig-kettle-2k"
    assert _device_ids(hass, entry, dev_uuid) == ids_before
    assert _energy_kwh(hass, energy_eid) == pytest.approx(energy_before, abs=1e-6)
    assert _state(hass, power_eid).state != "unavailable"


async def test_confirmed_target_merge_sources_include_unconfirmed_switched(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
) -> None:
    """A confirmed device's merge sources are the other confirmed devices plus
    the visible unconfirmed switched candidates: never a program candidate,
    never a retired device, never the device itself (D-13, C3)."""
    entry, entities = await _setup_two_devices(
        hass, energy_manager, std_prefs, hass_storage, freezer
    )
    uuid_a = entities["Kettle"][2]
    uuid_b = entities["Water heater"][2]
    sup = entry.runtime_data.learning

    # Add a visible switched candidate and a program candidate to the store.
    cand2 = make_switched_candidate("sig-kettle-2", "Kettle 2nd", rise_w=2150.0)
    prog = make_switched_candidate(
        "sig-dish", "Dishwasher", rise_w=800.0, kind="program"
    )
    record = sup._candidates_record
    record["candidates"] = [*record["candidates"], cand2, prog]
    sup.candidates = record["candidates"]
    await sup._candidates_store.async_save(record)

    # The device menu offers Merge while either kind of source exists.
    res_dev = await _open_device_menu(hass, entry, uuid_a)
    assert res_dev["type"] is FlowResultType.MENU
    assert "merge" in res_dev["menu_options"]

    # The selector lists the other confirmed device and the switched
    # candidate, never the program candidate or the device itself.
    res_merge = await hass.config_entries.options.async_configure(
        res_dev["flow_id"], {"next_step_id": "merge"}
    )
    assert res_merge["type"] is FlowResultType.FORM
    src_opts = res_merge["data_schema"].schema["sources"].config["options"]
    values = [o["value"] for o in src_opts]
    assert values == [f"device_{uuid_b}", "sig-kettle-2"]

    # Retire the water heater into the kettle (D-12): the retired device is
    # no longer a source; the switched candidate still is.
    res_after = await hass.config_entries.options.async_configure(
        res_merge["flow_id"], {"sources": [f"device_{uuid_b}"]}
    )
    assert res_after["type"] is FlowResultType.FORM
    res_dev2 = await _open_device_menu(hass, entry, uuid_a)
    assert "merge" in res_dev2["menu_options"]
    res_merge2 = await hass.config_entries.options.async_configure(
        res_dev2["flow_id"], {"next_step_id": "merge"}
    )
    src_opts2 = res_merge2["data_schema"].schema["sources"].config["options"]
    assert [o["value"] for o in src_opts2] == ["sig-kettle-2"]


# --- Plan 04-12 Task 3: retrains never collide with learning runs, survive
# restarts and failures, and keep later runs consistent (TDD) ----------------

import asyncio

from homeassistant.exceptions import ServiceValidationError

from custom_components.powerguesser.learning import (
    LearningStageError,
    LearningSupervisor,
)
from tests.conftest import (
    async_fill_synth_buffer,
    async_seed_consent,
    async_seed_llm,
)
from tests.fixtures.learning.synth import SynthSpec


async def _setup_device_with_candidate(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
    *,
    extra_cands: list[dict[str, Any]] | None = None,
    buffer_fills: list[tuple[SynthSpec, int]] | None = None,
) -> tuple[ConfigEntry, str]:
    """Confirmed Kettle device (sig-kettle-2k) with one visible candidate.

    ``buffer_fills`` is a list of (spec, end_days_ago) tuples; the default is
    one 22-day fill ending today. The confirmed candidate carries narrow,
    fitted-style boxes (a subset of what the union refit fits), so the
    keep-or-reject rule deterministically accepts the retrain. Returns
    (entry, dev_uuid).
    """
    cand_main = make_switched_candidate("sig-kettle-2k", "Kettle", rise_w=2000.0)
    st = cand_main["states"][0]
    tight_rise = {"lo": 1950.0, "p50": 2000.0, "hi": 2050.0}
    st["rise_w"] = dict(tight_rise)
    st["fall_w"] = dict(tight_rise)
    st["plateau_w"] = dict(tight_rise)
    st["dur_s"] = {"lo": 160.0, "p50": 180.0, "hi": 200.0}
    cand_main["fingerprint"] = {"kind": "switched", "states": [st]}
    seed_candidates(
        hass_storage,
        [
            cand_main,
            *(
                extra_cands
                if extra_cands is not None
                else [
                    make_switched_candidate(
                        "sig-kettle-2150", "Kettle 2nd", rise_w=2150.0
                    )
                ]
            ),
        ],
    )
    energy_manager.data = std_prefs
    async_set_power(hass, "sensor.grid_power", 500)
    async_set_power(hass, "sensor.solar_power", 0)
    async_set_power(hass, "sensor.battery_power", 0)
    async_set_power(hass, "sensor.washer_power", 200)
    _aligned_move(freezer)
    entry = await async_setup_via_flow(hass)
    _pin_grid(entry)
    if buffer_fills is None:
        buffer_fills = [(SynthSpec(days=22), 0)]
    today_midnight = dt_util.start_of_local_day(dt_util.now())
    for spec, end_days_ago in buffer_fills:
        end_ms = int(
            round((today_midnight - timedelta(days=end_days_ago)).timestamp() * 1000)
        )
        await async_fill_synth_buffer(hass, entry, spec=spec, end_ms=end_ms)
    await async_seed_llm(hass, entry)
    await async_seed_consent(hass, host="llm.test", names=False)
    await hass.async_block_till_done()
    # The gates were evaluated against the empty buffer at setup; re-evaluate
    # now that the synth buffer and LLM config exist, so the status is the
    # realistic post-gate "idle".
    await entry.runtime_data.learning._async_reevaluate_gates()

    await async_confirm_candidate(hass, entry, "sig-kettle-2k", "Kettle")
    sub = next(s for s in entry.get_subentries_of_type("device") if s.title == "Kettle")
    return entry, sub.data["device_uuid"]


async def _merge_candidate_via_flow(
    hass: HomeAssistant,
    entry: ConfigEntry,
    dev_uuid: str,
    source_sig: str = "sig-kettle-2150",
) -> dict[str, Any]:
    """Open the device menu, pick Merge and submit the candidate source."""
    res_dev = await _open_device_menu(hass, entry, dev_uuid)
    assert res_dev["type"] is FlowResultType.MENU
    assert "merge" in res_dev["menu_options"]
    res_merge = await hass.config_entries.options.async_configure(
        res_dev["flow_id"], {"next_step_id": "merge"}
    )
    assert res_merge["type"] is FlowResultType.FORM
    assert res_merge["step_id"] == "merge"
    return await hass.config_entries.options.async_configure(
        res_merge["flow_id"], {"sources": [source_sig]}
    )


async def test_refit_waits_for_running_learning(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
    aioclient_mock: Any,
    inprocess_worker: Any,
) -> None:
    """Merging while a learning run is in progress records pending_refit and
    starts the retrain exactly once when the run ends (D-13, T-04-25)."""
    entry, dev_uuid = await _setup_device_with_candidate(
        hass, energy_manager, std_prefs, hass_storage, freezer
    )
    sup = entry.runtime_data.learning
    devices = entry.runtime_data.devices

    started = asyncio.Event()
    release = asyncio.Event()

    async def _held_responder(data: Any) -> dict[str, Any]:
        from tests.ha.test_device_proposals import _kettle_responder

        started.set()
        await release.wait()
        return _kettle_responder("Kettle")(data)

    mock_llm(aioclient_mock, "http://llm.test/v1", _held_responder)

    await sup.async_request_run("service")
    learn_task = sup.run_task
    assert learn_task is not None
    await started.wait()

    # Merge while the run is held: queued as pending, no refit starts.
    res_after = await _merge_candidate_via_flow(hass, entry, dev_uuid)
    assert res_after["type"] is FlowResultType.FORM
    rec = devices.record(dev_uuid)
    assert rec["pending_refit"] is True
    assert rec["last_refit"] is None
    assert sup.run_task is learn_task

    release.set()
    await learn_task

    # The queued retrain now ran once and cleared the pending flag.
    refit_task = sup.run_task
    assert refit_task is not None and refit_task is not learn_task
    await refit_task
    rec = devices.record(dev_uuid)
    assert rec["pending_refit"] is False
    assert rec["last_refit"]["accepted"] is True
    assert len(rec["versions"]) == 2


async def test_learn_refused_during_refit(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
    aioclient_mock: Any,
    inprocess_worker: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A learn request during a refit raises already_running (single-flight)."""
    entry, dev_uuid = await _setup_device_with_candidate(
        hass, energy_manager, std_prefs, hass_storage, freezer
    )
    sup = entry.runtime_data.learning

    started = asyncio.Event()
    release = asyncio.Event()
    orig = LearningSupervisor._async_run_stage

    async def _held(self: Any, stage: str, job_path: Any) -> dict[str, Any]:
        if stage == "refit":
            started.set()
            await release.wait()
        return await orig(self, stage, job_path)

    monkeypatch.setattr(LearningSupervisor, "_async_run_stage", _held)

    await sup.async_record_decision(
        "merge", "sig-kettle-2k", sources=["sig-kettle-2150"]
    )
    await started.wait()
    assert sup.coordinator.data["running_job"] == "refit"

    with pytest.raises(ServiceValidationError):
        await sup.async_request_run("service")

    release.set()
    await sup.run_task
    rec = entry.runtime_data.devices.record(dev_uuid)
    assert rec["pending_refit"] is False
    assert rec["last_refit"]["accepted"] is True


async def test_pending_refit_resumes_after_reload(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
    aioclient_mock: Any,
    inprocess_worker: Any,
) -> None:
    """A pending_refit stored before a restart runs after startup."""
    entry, dev_uuid = await _setup_device_with_candidate(
        hass, energy_manager, std_prefs, hass_storage, freezer
    )
    devices = entry.runtime_data.devices
    await devices.async_mark_refit_pending(dev_uuid)
    assert devices.record(dev_uuid)["pending_refit"] is True

    await hass.config_entries.async_reload(entry.entry_id)
    await hass.async_block_till_done()

    sup = entry.runtime_data.learning
    devices = entry.runtime_data.devices
    # The retrain started at startup and may already have finished (the slot
    # is released when the job ends); await it when still referenced.
    task = sup.run_task
    if task is not None:
        await task
    rec = devices.record(dev_uuid)
    assert rec["pending_refit"] is False
    assert rec["last_refit"] is not None
    # Refits never touch the learning run's status metadata.
    assert sup.coordinator.data["last_run"] is None
    assert sup.coordinator.data["last_error"] is None


async def test_refit_failure_keeps_version(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
    aioclient_mock: Any,
    inprocess_worker: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failing refit keeps the current version, records last_refit with the
    error, and leaves the merge decision and alias in place."""
    entry, dev_uuid = await _setup_device_with_candidate(
        hass, energy_manager, std_prefs, hass_storage, freezer
    )
    sup = entry.runtime_data.learning
    devices = entry.runtime_data.devices
    orig = LearningSupervisor._async_run_stage

    async def _fail(self: Any, stage: str, job_path: Any) -> dict[str, Any]:
        if stage == "refit":
            raise LearningStageError("refit: injected failure")
        return await orig(self, stage, job_path)

    monkeypatch.setattr(LearningSupervisor, "_async_run_stage", _fail)

    await sup.async_record_decision(
        "merge", "sig-kettle-2k", sources=["sig-kettle-2150"]
    )
    assert sup.run_task is not None
    await sup.run_task

    rec = devices.record(dev_uuid)
    assert len(rec["versions"]) == 1
    assert rec["active_v"] == 1
    assert "error" in rec["last_refit"]
    assert "LearningStageError" in rec["last_refit"]["error"]
    assert "sig-kettle-2150" in rec["aliases"]
    assert "sig-kettle-2150" in rec["alias_artefacts"]
    assert sup.review.decisions["sig-kettle-2150"]["status"] == "merged"
    # The learning run's status metadata is untouched by the refit failure.
    assert sup.coordinator.data["last_run"] is None
    assert sup.coordinator.data["last_error"] is None
    assert sup.coordinator.data["status"] == "idle"


def _short_kettle_responder() -> Any:
    """Run responder proposing the short-kettle cluster (the merged candidate)."""

    def _responder(data: Any) -> dict[str, Any]:
        from tests.ha.test_device_proposals import _digest_from_request

        digest = _digest_from_request(data)
        clusters = digest["clusters"]
        cs = min(clusters, key=lambda c: abs(c["rise_w"]["p50"] - 2150))["id"]
        prop = {
            "analysis": "Short kettle pulses again.",
            "schema_version": 1,
            "devices": [
                {
                    "ref": "d1",
                    "label": "Kettle 2nd",
                    "category": "kettle",
                    "tracker_kind": "switched",
                    "member_clusters": [cs],
                    "states": [{"name": "on", "cluster_ref": cs, "role": "main"}],
                    "phase_hint": "single",
                    "confidence": "high",
                    "rationale": "Short 2.1kW kettle pulses, same as merged.",
                    "ambiguous_with": [],
                    "reference_plug": "",
                }
            ],
            "rejected_clusters": [],
            "merge_suggestions": [],
        }
        return {"status": 200, "json": chat_completion(json.dumps(prop))}

    return _responder


async def test_rejected_refit_and_later_run_consistency(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
    aioclient_mock: Any,
    inprocess_worker: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rejected union retrain keeps the current version and records
    last_refit accepted false; in a later learning run the merged candidate's
    matching artefact is identity merged (hidden, no proposal) while the
    device's own version is untouched.

    The buffer trains on short kettle pulses only and validates on long
    kettle runs, so the union retrain (fit on the shorts) is genuinely worse
    (D-13 keep-or-reject). The device's-own-signature proposal half of the
    rule rides the 04-11 test_later_run_maps_absorbed_signature_to_merged
    machinery, which passes unchanged.
    """
    from tests.ha.test_device_proposals import _run_learning

    monkeypatch.setattr("custom_components.powerguesser.learning.MAX_REFINE_ROUNDS", 0)
    # Oldest 15 days: short kettle (train); recent 7 days: long kettle (val).
    # The merged candidate carries fitted-style short-pulse boxes so a later
    # run's short-cluster artefact deterministically matches it (Jaccard).
    cand_short = make_switched_candidate(
        "sig-kettle-2150", "Kettle 2nd", rise_w=2150.0
    )
    st_short = cand_short["states"][0]
    short_rise = {"lo": 2020.0, "p50": 2150.0, "hi": 2280.0}
    st_short["rise_w"] = dict(short_rise)
    st_short["fall_w"] = dict(short_rise)
    # Short pulses average to ~1525 W plateau at the 30 s grid — the fitted
    # shape a real candidate of this load would carry.
    st_short["plateau_w"] = {"lo": 1000.0, "p50": 1525.0, "hi": 2200.0}
    st_short["dur_s"] = {"lo": 30.0, "p50": 60.0, "hi": 90.0}
    cand_short["fingerprint"] = {"kind": "switched", "states": [st_short]}
    entry, dev_uuid = await _setup_device_with_candidate(
        hass,
        energy_manager,
        std_prefs,
        hass_storage,
        freezer,
        extra_cands=[cand_short],
        buffer_fills=[
            (
                SynthSpec(
                    days=15, kettle_w=2150.0, kettle_s=60, kettle_per_day=(3, 4)
                ),
                7,
            ),
            (SynthSpec(days=7, kettle_w=2000.0, kettle_s=180), 0),
        ],
    )
    sup = entry.runtime_data.learning
    devices = entry.runtime_data.devices

    await sup.async_record_decision(
        "merge", "sig-kettle-2k", sources=["sig-kettle-2150"]
    )
    assert sup.run_task is not None
    await sup.run_task

    # The union retrain ran and was rejected: version kept, error recorded.
    rec = devices.record(dev_uuid)
    assert rec["last_refit"]["accepted"] is False
    assert rec["last_refit"]["new_S"] < rec["last_refit"]["current_S"]
    assert len(rec["versions"]) == 1
    assert rec["active_v"] == 1
    # The merge decision and alias stay in place either way (D-13).
    assert "sig-kettle-2150" in rec["aliases"]
    assert "sig-kettle-2150" in rec["alias_artefacts"]

    # A later learning run proposing the merged load again maps it to the
    # device's signature as merged: hidden, never proposed.
    mock_llm(aioclient_mock, "http://llm.test/v1", _short_kettle_responder())
    await _run_learning(sup)

    matched = [c for c in sup.candidates if c["signature_id"] == "sig-kettle-2150"]
    assert len(matched) == 1
    assert matched[0]["identity"]["status"] == "merged"
    assert matched[0]["identity"]["merged_into"] == "sig-kettle-2k"
    assert all(
        c["signature_id"] != "sig-kettle-2150" for c in sup.visible_candidates()
    )
    rec = devices.record(dev_uuid)
    assert rec["pending_proposal"] is None
    assert len(rec["versions"]) == 1


async def test_refit_running_job_reported(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
    aioclient_mock: Any,
    inprocess_worker: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The learning status sensor shows running_job "refit" while the retrain
    runs and null afterwards; last_run is unchanged by refits."""
    entry, dev_uuid = await _setup_device_with_candidate(
        hass, energy_manager, std_prefs, hass_storage, freezer
    )
    sup = entry.runtime_data.learning

    started = asyncio.Event()
    release = asyncio.Event()
    stages: list[str] = []
    orig = LearningSupervisor._async_run_stage

    async def _held(self: Any, stage: str, job_path: Any) -> dict[str, Any]:
        stages.append(stage)
        if stage == "refit":
            started.set()
            await release.wait()
        return await orig(self, stage, job_path)

    monkeypatch.setattr(LearningSupervisor, "_async_run_stage", _held)

    await sup.async_record_decision(
        "merge", "sig-kettle-2k", sources=["sig-kettle-2150"]
    )
    await started.wait()

    state = hass.states.get("sensor.powerguesser_learning_status")
    assert state is not None
    assert state.state == "running"
    assert state.attributes.get("running_job") == "refit"
    assert state.attributes.get("last_run") is None

    release.set()
    await sup.run_task

    state = hass.states.get("sensor.powerguesser_learning_status")
    assert state is not None
    assert state.state == "idle"
    assert state.attributes.get("running_job") is None
    assert state.attributes.get("last_run") is None
    # Exactly one stage ran, and it was the refit.
    assert stages == ["refit"]
    rec = entry.runtime_data.devices.record(dev_uuid)
    assert rec["last_refit"]["accepted"] is True


async def test_later_run_maps_absorbed_signature_to_merged(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    aioclient_mock: Any,
    inprocess_worker: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A later learning run whose artefact matches the absorbed signature maps
    it to merged (hidden, no proposal); a match for the survivor's own
    signature still yields a proposal."""
    from tests.ha.test_device_proposals import _run_learning, _setup_learning_env

    entry = await _setup_learning_env(hass, energy_manager, std_prefs, monkeypatch)
    sup = entry.runtime_data.learning
    devices = entry.runtime_data.devices

    # --- Run 1: propose kettle + water heater; confirm both ---
    call_count = 0

    def _responder(data: Any) -> dict[str, Any]:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _kettle_and_heater_responder()(data)
        return _changed_kettle_and_heater_responder()(data)

    mock_llm(aioclient_mock, "http://llm.test/v1", _responder)
    await _run_learning(sup)
    cands = sup.visible_candidates()
    kettle_cand = next(c for c in cands if c.get("label") == "Kettle")
    heater_cand = next(c for c in cands if c.get("label") == "Water heater")
    sig_a = str(kettle_cand["signature_id"])
    sig_b = str(heater_cand["signature_id"])
    await async_confirm_candidate(hass, entry, sig_a, "Kettle")
    await async_confirm_candidate(hass, entry, sig_b, "Water heater")
    sub_a = next(s for s in entry.get_subentries_of_type("device") if s.title == "Kettle")
    sub_b = next(
        s for s in entry.get_subentries_of_type("device") if s.title == "Water heater"
    )
    uuid_a = sub_a.data["device_uuid"]
    uuid_b = sub_b.data["device_uuid"]

    # Merge the water heater into the kettle.
    await _merge_devices(hass, entry, uuid_a, uuid_b)
    assert devices.record(uuid_b)["status"] == "retired"

    # --- Run 2: the kettle load changed (renamed state -> proposal for A);
    # the water-heater load is proposed again (matches the absorbed signature) ---
    await _run_learning(sup)

    # The absorbed signature maps to merged: hidden, never proposed.
    matched_b = [c for c in sup.candidates if c["signature_id"] == sig_b]
    assert len(matched_b) == 1
    assert matched_b[0]["identity"]["status"] == "merged"
    assert matched_b[0]["identity"]["merged_into"] == sig_a
    assert all(c["signature_id"] != sig_b for c in sup.visible_candidates())
    rec_b = devices.record(uuid_b)
    assert rec_b["status"] == "retired"
    assert rec_b["pending_proposal"] is None

    # The survivor's own changed match still yields a proposal.
    rec_a = devices.record(uuid_a)
    assert rec_a["status"] == "confirmed"
    assert rec_a["pending_proposal"] is not None
    assert devices.update_available(uuid_a) is True

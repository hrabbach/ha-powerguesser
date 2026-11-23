"""Tests for the confirmed-device lifecycle: delete is un-confirm (D-05).

Plan 04-05 Task 1: deleting a confirmed device in Settings returns its
candidate to the review list with its name; the device record becomes
unconfirmed and keeps its uuid, parameter versions and energy total; a
re-confirm reuses the same uuid so the sensors get the same unique_ids, HA
restores the user's entity_id and the energy sensor continues from its
retained total (OUT-05, OUT-02, REV-02).

Task 2: renaming a device from the standard Settings UI (subentry
reconfigure flow, RESEARCH Pattern 7) renames the subentry, the device
registry entry, the cold record and the review record, never the sensors'
entity_ids or unique_ids, and never reloads (OUT-05).
"""

import asyncio
from datetime import timedelta
import math
from typing import Any
from unittest.mock import patch

import pytest

import custom_components.powerguesser as pg
from homeassistant.components.energy.data import EnergyManager
from homeassistant.config_entries import ConfigEntry, ConfigEntryState
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers import entity_registry as er
from homeassistant.util import dt as dt_util
from pytest_homeassistant.common import async_fire_time_changed

from custom_components.powerguesser.const import (
    CANDIDATES_STORE_KEY,
    DOMAIN,
    LEARNING_STORES_VERSION,
    REVIEW_STORE_KEY,
)
from custom_components.powerguesser.devices import DeviceRuntime
from custom_components.powerguesser.hub import PowerGuesserHub
from custom_components.powerguesser.learning import LearningSupervisor
from custom_components.powerguesser.review_store import ReviewStore

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
    from custom_components.powerguesser.engine.intervals import (
        ProfileSnapshot,
        gate,
    )

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
    """Advance the freezer and fire the time-changed event."""
    freezer.tick(timedelta(seconds=seconds))
    async_fire_time_changed(hass, dt_util.utcnow())
    await hass.async_block_till_done()


async def _drive_ticks(hass: HomeAssistant, freezer: Any, n: int) -> None:
    """Advance n full grid steps via the 10 s tick path."""
    for _ in range(n * 3):
        await _tick(hass, freezer, 10)


async def _setup_confirmed(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
    candidates: list[dict[str, Any]],
) -> tuple[ConfigEntry, dict[str, tuple[str, str]]]:
    """Set up the entry, pin the grid at 30 s and confirm every candidate.

    Returns (entry, {label: (power_entity_id, energy_entity_id)}).
    """
    seed_candidates(hass_storage, candidates)
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
    entities: dict[str, tuple[str, str]] = {}
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


def _sub_for(entry: ConfigEntry, title: str) -> Any:
    """Return the device subentry with the given title."""
    return next(s for s in entry.get_subentries_of_type("device") if s.title == title)


async def _kettle_run(hass: HomeAssistant, freezer: Any, ticks: int = 5) -> None:
    """Switch the kettle on, run `ticks` grid steps, switch it off, settle."""
    async_set_power(hass, "sensor.grid_power", 2500)
    await hass.async_block_till_done()
    await _drive_ticks(hass, freezer, ticks)
    async_set_power(hass, "sensor.grid_power", 500)
    await hass.async_block_till_done()
    await _drive_ticks(hass, freezer, ticks)


async def test_delete_returns_candidate_and_reconfirm_restores_identity(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
) -> None:
    """Delete is un-confirm: the candidate returns, re-confirm restores identity."""
    entry, entities = await _setup_confirmed(
        hass,
        energy_manager,
        std_prefs,
        hass_storage,
        freezer,
        [make_switched_candidate("sig-kettle", "Kettle", rise_w=2000.0)],
    )
    power_eid, energy_eid = entities["Kettle"]
    ent_reg = er.async_get(hass)
    sub = _sub_for(entry, "Kettle")
    dev_uuid = sub.data["device_uuid"]

    # The user renamed the power entity in the entity registry.
    ent_reg.async_update_entity(power_eid, new_entity_id="sensor.my_kettle_power")
    power_unique_id = f"{entry.entry_id}_{dev_uuid}_power"
    energy_unique_id = f"{entry.entry_id}_{dev_uuid}_energy"

    # Drive one run so the energy total is non-zero.
    await _kettle_run(hass, freezer)
    energy_before = float(hass.states.get(energy_eid).state)
    assert energy_before > 0.0

    # Delete the device in Settings: HA removes the subentry's entities and
    # device, keeping deleted-entity registry rows (re-add restores entity_id).
    hass.config_entries.async_remove_subentry(entry, sub.subentry_id)
    await hass.async_block_till_done()

    assert hass.states.get("sensor.my_kettle_power") is None
    assert ent_reg.async_get_entity_id("sensor", DOMAIN, power_unique_id) is None

    learning = entry.runtime_data.learning
    devices = entry.runtime_data.devices

    # The review list shows the candidate again, with its name (D-05): the
    # stored learning candidate still exists, so the name comes from the
    # review decision.
    visible = learning.visible_candidates()
    back = next((c for c in visible if c.get("signature_id") == "sig-kettle"), None)
    assert back is not None
    assert learning.candidate_label(back).startswith("Kettle")

    # The device record is unconfirmed and keeps its uuid and energy total.
    rec = devices.record(dev_uuid)
    assert rec is not None
    assert rec["status"] == "unconfirmed"
    assert rec["signature_id"] == "sig-kettle"
    assert devices.energy_wh(dev_uuid) > 0.0

    # The review record keeps name and device_uuid, loses the subentry link.
    dec = learning.review.decisions["sig-kettle"]
    assert dec["status"] == "named"
    assert dec["name"] == "Kettle"
    assert dec["device_uuid"] == dev_uuid
    assert dec["subentry_id"] is None

    # Re-confirm: same uuid -> same unique_ids, restored entity_id, energy kept.
    # The kettle was the last visible candidate, so the review step returns to
    # the confirmed device entry (Review contract C2, Plan 04-10).
    res = await async_confirm_candidate(hass, entry, "sig-kettle", "Kettle")
    assert res["type"] is FlowResultType.FORM
    assert res["step_id"] == "review"
    re_opts = res["data_schema"].schema["candidate"].config["options"]
    assert [o["value"] for o in re_opts] == [f"device_{dev_uuid}"]
    new_sub = _sub_for(entry, "Kettle")
    assert new_sub.data["device_uuid"] == dev_uuid
    assert new_sub.unique_id == dev_uuid
    assert (
        ent_reg.async_get_entity_id("sensor", DOMAIN, power_unique_id)
        == "sensor.my_kettle_power"
    )
    assert ent_reg.async_get_entity_id("sensor", DOMAIN, energy_unique_id) == energy_eid
    assert float(hass.states.get(energy_eid).state) == pytest.approx(
        energy_before, abs=1e-6
    )
    assert hass.states.get("sensor.my_kettle_power").state == "0.0"


async def test_deleted_device_candidate_listed_after_new_run_without_it(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
) -> None:
    """A deleted device stays reviewable even when a new run drops its signature."""
    entry, _entities = await _setup_confirmed(
        hass,
        energy_manager,
        std_prefs,
        hass_storage,
        freezer,
        [make_switched_candidate("sig-kettle", "Kettle", rise_w=2000.0)],
    )
    sub = _sub_for(entry, "Kettle")
    dev_uuid = sub.data["device_uuid"]
    hass.config_entries.async_remove_subentry(entry, sub.subentry_id)
    await hass.async_block_till_done()

    # A new learning run replaced the candidates library: the kettle's
    # signature is no longer among the stored candidates.
    new_library = [make_switched_candidate("sig-oven", "Oven", train_share=0.09)]
    seed_candidates(hass_storage, new_library)
    learning = entry.runtime_data.learning
    learning.candidates = list(new_library)
    await hass.async_block_till_done()

    visible = learning.visible_candidates()
    sigs = {c.get("signature_id") for c in visible}
    assert "sig-kettle" in sigs  # listed from the device record
    assert "sig-oven" in sigs
    kettle = next(c for c in visible if c.get("signature_id") == "sig-kettle")
    assert kettle["first_label"] == "Kettle"

    # Re-confirming reuses the same device uuid.
    await async_confirm_candidate(hass, entry, "sig-kettle", "Kettle")
    new_sub = _sub_for(entry, "Kettle")
    assert new_sub.data["device_uuid"] == dev_uuid


async def test_delete_keeps_review_name_and_uuid(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
) -> None:
    """Un-confirm returns kept as kept and named as named, keeping name + uuid."""
    hass_storage[REVIEW_STORE_KEY] = {
        "version": LEARNING_STORES_VERSION,
        "minor_version": 1,
        "key": REVIEW_STORE_KEY,
        "data": {
            "decisions": {
                # A kept merge target and a named candidate before confirm.
                "sig-alpha": {"status": "kept", "name": "Alpha", "aliases": []},
                "sig-beta": {"status": "named", "name": "Beta", "aliases": []},
            }
        },
    }
    entry, _entities = await _setup_confirmed(
        hass,
        energy_manager,
        std_prefs,
        hass_storage,
        freezer,
        [
            make_switched_candidate("sig-alpha", "Alpha", rise_w=2000.0),
            make_switched_candidate("sig-beta", "Beta", rise_w=1800.0),
        ],
    )
    uuids = {}
    for label in ("Alpha", "Beta"):
        sub = _sub_for(entry, label)
        uuids[label] = sub.data["device_uuid"]
        hass.config_entries.async_remove_subentry(entry, sub.subentry_id)
    await hass.async_block_till_done()

    decisions = entry.runtime_data.learning.review.decisions
    alpha = decisions["sig-alpha"]
    assert alpha["status"] == "kept"
    assert alpha["name"] == "Alpha"
    assert alpha["device_uuid"] == uuids["Alpha"]
    assert alpha["subentry_id"] is None

    beta = decisions["sig-beta"]
    assert beta["status"] == "named"
    assert beta["name"] == "Beta"
    assert beta["device_uuid"] == uuids["Beta"]
    assert beta["subentry_id"] is None

    # Both device records are unconfirmed with their energy/uuid kept.
    devices = entry.runtime_data.devices
    assert devices.record(uuids["Alpha"])["status"] == "unconfirmed"
    assert devices.record(uuids["Beta"])["status"] == "unconfirmed"


# ---- Task 2: rename from the Settings UI (TDD) ----


async def _setup_confirmed_plain(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> ConfigEntry:
    """Set up the entry and confirm every candidate (no clock driving)."""
    seed_candidates(hass_storage, candidates)
    energy_manager.data = std_prefs
    async_set_power(hass, "sensor.grid_power", 500)
    async_set_power(hass, "sensor.solar_power", 0)
    async_set_power(hass, "sensor.battery_power", 0)
    async_set_power(hass, "sensor.washer_power", 200)
    entry = await async_setup_via_flow(hass)
    for cand in candidates:
        await async_confirm_candidate(
            hass, entry, str(cand["signature_id"]), str(cand["label"])
        )
    return entry


async def _start_rename(hass: HomeAssistant, entry: ConfigEntry, subentry_id: str) -> Any:
    """Open the Settings rename (reconfigure) flow for a device subentry."""
    return await hass.config_entries.subentries.async_init(
        (entry.entry_id, "device"),
        context={"source": "reconfigure", "subentry_id": subentry_id},
    )


async def test_rename_via_settings_keeps_entity_ids(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reconfiguring the name renames everything visible and nothing that history uses."""
    entry = await _setup_confirmed_plain(
        hass,
        energy_manager,
        std_prefs,
        hass_storage,
        [make_switched_candidate("sig-kettle", "Kettle", rise_w=2000.0)],
    )
    ent_reg = er.async_get(hass)
    sub = _sub_for(entry, "Kettle")
    dev_uuid = sub.data["device_uuid"]
    power_eid = ent_reg.async_get_entity_id(
        "sensor", DOMAIN, f"{entry.entry_id}_{dev_uuid}_power"
    )
    energy_eid = ent_reg.async_get_entity_id(
        "sensor", DOMAIN, f"{entry.entry_id}_{dev_uuid}_energy"
    )
    assert power_eid is not None and energy_eid is not None

    setup_calls: list[None] = []
    orig_setup = pg.async_setup_entry

    async def _counting_setup(hass_: HomeAssistant, entry_: ConfigEntry) -> bool:
        setup_calls.append(None)
        return await orig_setup(hass_, entry_)

    monkeypatch.setattr(pg, "async_setup_entry", _counting_setup)

    with (
        patch.object(PowerGuesserHub, "async_request_rebuild") as mock_rebuild,
        patch.object(
            LearningSupervisor, "async_settings_changed"
        ) as mock_settings_changed,
    ):
        result = await _start_rename(hass, entry, sub.subentry_id)
        assert result["type"] is FlowResultType.FORM
        assert result["step_id"] == "reconfigure"

        result2 = await hass.config_entries.subentries.async_configure(
            result["flow_id"], {"name": "  Küche   Wasserkocher "}
        )
        await hass.async_block_till_done()

    assert result2["type"] is FlowResultType.ABORT
    assert result2["reason"] == "reconfigure_successful"

    # Never reload, never touch learning gates (Pitfall 1).
    assert setup_calls == []
    mock_rebuild.assert_not_called()
    mock_settings_changed.assert_not_called()
    assert entry.state is ConfigEntryState.LOADED

    # Subentry title and data name carry the normalised name.
    sub = entry.subentries[sub.subentry_id]
    assert sub.title == "Küche Wasserkocher"
    assert sub.data["name"] == "Küche Wasserkocher"

    # Device registry name, cold record name and review record name follow.
    device = dr.async_get(hass).async_get_device_by_identifier((DOMAIN, dev_uuid), entry.entry_id)
    assert device is not None
    assert device.name == "Küche Wasserkocher"
    rec = entry.runtime_data.devices.record(dev_uuid)
    assert rec is not None
    assert rec["name"] == "Küche Wasserkocher"
    dec = entry.runtime_data.learning.review.decisions["sig-kettle"]
    assert dec["name"] == "Küche Wasserkocher"

    # The sensors' entity_ids and unique_ids never change (OUT-05, D-04).
    assert (
        ent_reg.async_get_entity_id("sensor", DOMAIN, f"{entry.entry_id}_{dev_uuid}_power")
        == power_eid
    )
    assert (
        ent_reg.async_get_entity_id("sensor", DOMAIN, f"{entry.entry_id}_{dev_uuid}_energy")
        == energy_eid
    )
    assert hass.states.get(power_eid) is not None
    assert hass.states.get(energy_eid) is not None


async def test_rename_blank_name_invalid(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
) -> None:
    """A blank or whitespace-only name shows name_invalid and changes nothing."""
    entry = await _setup_confirmed_plain(
        hass,
        energy_manager,
        std_prefs,
        hass_storage,
        [make_switched_candidate("sig-kettle", "Kettle", rise_w=2000.0)],
    )
    ent_reg = er.async_get(hass)
    sub = _sub_for(entry, "Kettle")
    dev_uuid = sub.data["device_uuid"]
    power_eid = ent_reg.async_get_entity_id(
        "sensor", DOMAIN, f"{entry.entry_id}_{dev_uuid}_power"
    )

    result = await _start_rename(hass, entry, sub.subentry_id)
    result2 = await hass.config_entries.subentries.async_configure(
        result["flow_id"], {"name": "   "}
    )
    assert result2["type"] is FlowResultType.FORM
    assert result2["step_id"] == "reconfigure"
    assert result2["errors"]["base"] == "name_invalid"
    await hass.async_block_till_done()

    # Nothing changed anywhere.
    sub = entry.subentries[sub.subentry_id]
    assert sub.title == "Kettle"
    assert entry.runtime_data.devices.record(dev_uuid)["name"] == "Kettle"
    dec = entry.runtime_data.learning.review.decisions["sig-kettle"]
    assert dec["name"] == "Kettle"
    device = dr.async_get(hass).async_get_device_by_identifier((DOMAIN, dev_uuid), entry.entry_id)
    assert device.name == "Kettle"
    assert (
        ent_reg.async_get_entity_id("sensor", DOMAIN, f"{entry.entry_id}_{dev_uuid}_power")
        == power_eid
    )


async def test_same_name_devices_distinct(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
) -> None:
    """Two devices confirmed with the same name get distinct uuids and entity_ids."""
    entry = await _setup_confirmed_plain(
        hass,
        energy_manager,
        std_prefs,
        hass_storage,
        [
            make_switched_candidate("sig-a", "Kettle", rise_w=2000.0, train_share=0.06),
            make_switched_candidate("sig-b", "Kettle", rise_w=1800.0, train_share=0.05),
        ],
    )
    subs = entry.get_subentries_of_type("device")
    assert len(subs) == 2
    uuids = {sub.data["device_uuid"] for sub in subs}
    assert len(uuids) == 2
    assert {sub.unique_id for sub in subs} == uuids

    ent_reg = er.async_get(hass)
    entity_ids: set[str] = set()
    for dev_uuid in uuids:
        for kind in ("power", "energy"):
            eid = ent_reg.async_get_entity_id(
                "sensor", DOMAIN, f"{entry.entry_id}_{dev_uuid}_{kind}"
            )
            assert eid is not None
            entity_ids.add(eid)
    # Two devices x two sensors, all with HA-distinct entity_ids.
    assert len(entity_ids) == 4


async def test_reconcile_idempotent_and_concurrent(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
) -> None:
    """Reconcile twice, or two listener tasks concurrently, change nothing."""
    entry = await _setup_confirmed_plain(
        hass,
        energy_manager,
        std_prefs,
        hass_storage,
        [make_switched_candidate("sig-kettle", "Kettle", rise_w=2000.0)],
    )
    devices = entry.runtime_data.devices
    sub = _sub_for(entry, "Kettle")

    with (
        patch.object(DeviceRuntime, "_async_save_cold", autospec=True) as mock_save,
        patch.object(ReviewStore, "async_rename_linked", autospec=True) as mock_rename,
    ):
        await devices.async_reconcile()
        await devices.async_reconcile()
        await asyncio.gather(
            devices.async_reconcile(), devices.async_reconcile(), devices.async_reconcile()
        )
        await hass.async_block_till_done()

    # Nothing to write: no cold saves, no renames, no duplicate entities.
    mock_save.assert_not_called()
    mock_rename.assert_not_called()
    ent_reg = er.async_get(hass)
    sub_entities = [
        entity
        for entity in ent_reg.entities.values()
        if entity.config_subentry_id == sub.subentry_id
    ]
    assert sorted(e.unique_id for e in sub_entities) == sorted(
        {
            f"{entry.entry_id}_{sub.data['device_uuid']}_power",
            f"{entry.entry_id}_{sub.data['device_uuid']}_energy",
            # Plan 05-04: the third per-device entity (D-14, VAL-02).
            f"{entry.entry_id}_{sub.data['device_uuid']}_validation",
        }
    )


async def test_lifecycle_sequences_keep_unique_ids(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
) -> None:
    """delete -> re-confirm -> rename and rename -> delete -> re-confirm keep identity."""
    entry = await _setup_confirmed_plain(
        hass,
        energy_manager,
        std_prefs,
        hass_storage,
        [
            make_switched_candidate("sig-kettle", "Kettle", rise_w=2000.0),
            make_switched_candidate("sig-oven", "Oven", rise_w=1800.0),
        ],
    )
    ent_reg = er.async_get(hass)
    devices = entry.runtime_data.devices

    def _unique_ids(dev_uuid: str) -> tuple[str, str]:
        return (
            f"{entry.entry_id}_{dev_uuid}_power",
            f"{entry.entry_id}_{dev_uuid}_energy",
        )

    # Sequence A: delete -> re-confirm -> rename.
    sub = _sub_for(entry, "Kettle")
    kettle_uuid = sub.data["device_uuid"]
    kettle_ids = _unique_ids(kettle_uuid)
    hass.config_entries.async_remove_subentry(entry, sub.subentry_id)
    await hass.async_block_till_done()
    await async_confirm_candidate(hass, entry, "sig-kettle", "Kettle")
    assert _sub_for(entry, "Kettle").data["device_uuid"] == kettle_uuid

    result = await _start_rename(hass, entry, _sub_for(entry, "Kettle").subentry_id)
    result2 = await hass.config_entries.subentries.async_configure(
        result["flow_id"], {"name": "Kettle Two"}
    )
    assert result2["type"] is FlowResultType.ABORT
    await hass.async_block_till_done()
    for unique_id in kettle_ids:
        assert ent_reg.async_get_entity_id("sensor", DOMAIN, unique_id) is not None
    assert devices.record(kettle_uuid)["name"] == "Kettle Two"

    # Sequence B: rename -> delete -> re-confirm.
    sub = _sub_for(entry, "Oven")
    oven_uuid = sub.data["device_uuid"]
    oven_ids = _unique_ids(oven_uuid)
    result = await _start_rename(hass, entry, sub.subentry_id)
    await hass.config_entries.subentries.async_configure(
        result["flow_id"], {"name": "Oven Renamed"}
    )
    await hass.async_block_till_done()
    hass.config_entries.async_remove_subentry(
        entry, _sub_for(entry, "Oven Renamed").subentry_id
    )
    await hass.async_block_till_done()
    await async_confirm_candidate(hass, entry, "sig-oven", "Oven Renamed")
    await hass.async_block_till_done()

    # Both end with the original unique_ids and the latest names.
    for unique_id in kettle_ids + oven_ids:
        assert ent_reg.async_get_entity_id("sensor", DOMAIN, unique_id) is not None
    assert devices.record(kettle_uuid)["name"] == "Kettle Two"
    assert devices.record(oven_uuid)["name"] == "Oven Renamed"
    assert _sub_for(entry, "Kettle Two").data["device_uuid"] == kettle_uuid
    assert _sub_for(entry, "Oven Renamed").data["device_uuid"] == oven_uuid

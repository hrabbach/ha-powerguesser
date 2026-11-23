"""Tests for confirming a review candidate as a device (REV-02, OUT-01, OUT-02)."""

from typing import Any
from unittest.mock import patch

import pytest

from homeassistant.components.energy.data import EnergyManager
from homeassistant.config_entries import ConfigEntry, ConfigEntryState
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers import entity_registry as er

import custom_components.powerguesser as pg
from custom_components.powerguesser.const import (
    DEVICES_STORE_KEY,
    DOMAIN,
    LEARNING_STORES_VERSION,
    REVIEW_STORE_KEY,
)
from custom_components.powerguesser.hub import PowerGuesserHub
from custom_components.powerguesser.learning import LearningSupervisor

from tests.conftest import (
    async_confirm_candidate,
    async_open_options_step,
    async_set_power,
    async_setup_via_flow,
    make_switched_candidate,
    seed_candidates,
)


@pytest.fixture
def seeded_switched(hass_storage: dict[str, Any]) -> list[dict[str, Any]]:
    """Seed two validated switched candidates (Kettle and Oven)."""
    c_kettle = make_switched_candidate(
        "sig-kettle", "Kettle", rise_w=2000.0, train_share=0.052, val_share=0.048
    )
    c_oven = make_switched_candidate(
        "sig-oven", "Oven", rise_w=1800.0, train_share=0.085, val_share=0.079
    )
    seed_candidates(hass_storage, [c_kettle, c_oven])
    return [c_kettle, c_oven]


async def _setup_entry(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
) -> ConfigEntry:
    """Set up PowerGuesser with standard prefs and power states."""
    energy_manager.data = std_prefs
    async_set_power(hass, "sensor.grid_power", 500)
    async_set_power(hass, "sensor.solar_power", 1200)
    async_set_power(hass, "sensor.battery_power", -300)
    async_set_power(hass, "sensor.washer_power", 200)
    return await async_setup_via_flow(hass)


async def test_confirm_creates_device_with_power_and_energy_sensors(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    seeded_switched: list[dict[str, Any]],
) -> None:
    """Confirming a switched candidate creates one device subentry with W/kWh sensors."""
    entry = await _setup_entry(hass, energy_manager, std_prefs)

    res = await async_confirm_candidate(hass, entry, "sig-kettle", "Kettle")
    assert res["type"] is FlowResultType.FORM
    assert res["step_id"] == "review"

    # Exactly one device subentry with exactly the D-03 data contract
    subs = entry.get_subentries_of_type("device")
    assert len(subs) == 1
    sub = subs[0]
    assert set(sub.data.keys()) == {"device_uuid", "name", "signature_id"}
    assert sub.data["name"] == "Kettle"
    assert sub.data["signature_id"] == "sig-kettle"
    assert sub.title == "Kettle"
    device_uuid = sub.data["device_uuid"]
    assert device_uuid == sub.unique_id

    # Entities with the D-04 uuid-based unique_ids and the right classes
    ent_reg = er.async_get(hass)
    power_eid = ent_reg.async_get_entity_id(
        "sensor", DOMAIN, f"{entry.entry_id}_{device_uuid}_power"
    )
    energy_eid = ent_reg.async_get_entity_id(
        "sensor", DOMAIN, f"{entry.entry_id}_{device_uuid}_energy"
    )
    assert power_eid is not None
    assert energy_eid is not None
    power_entry = ent_reg.async_get(power_eid)
    energy_entry = ent_reg.async_get(energy_eid)
    assert power_entry.original_device_class == "power"
    assert power_entry.unit_of_measurement == "W"
    assert energy_entry.original_device_class == "energy"
    assert energy_entry.unit_of_measurement == "kWh"
    # Entity and device carry the subentry id
    assert power_entry.config_subentry_id == sub.subentry_id
    assert energy_entry.config_subentry_id == sub.subentry_id
    assert power_entry.device_id == energy_entry.device_id
    dev_reg = dr.async_get(hass)
    device = dev_reg.async_get(power_entry.device_id)
    assert device is not None
    assert device.name == "Kettle"
    assert device.manufacturer == "PowerGuesser"
    assert device.model == "Estimated switching load"
    assert device.config_subentry_id == sub.subentry_id

    # New device: OFF power 0.0 W, stored energy total 0.0 kWh
    power_state = hass.states.get(power_eid)
    energy_state = hass.states.get(energy_eid)
    assert power_state.state == "0.0"
    assert energy_state.state == "0.0"
    assert power_state.attributes["device_class"] == "power"
    assert power_state.attributes["state_class"] == "measurement"
    assert energy_state.attributes["device_class"] == "energy"
    assert energy_state.attributes["state_class"] == "total_increasing"


async def test_confirm_does_not_reload_or_touch_learning(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    seeded_switched: list[dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Confirming adds a subentry without reload, hub rebuild or learning gates."""
    entry = await _setup_entry(hass, energy_manager, std_prefs)
    hub_before = entry.runtime_data

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
        res = await async_confirm_candidate(hass, entry, "sig-kettle", "Kettle")
        await hass.async_block_till_done()
        assert res["type"] is FlowResultType.FORM
        assert res["step_id"] == "review"
        mock_rebuild.assert_not_called()
        mock_settings_changed.assert_not_called()

    assert setup_calls == []
    assert entry.runtime_data is hub_before
    assert entry.state is ConfigEntryState.LOADED
    assert len(entry.get_subentries_of_type("device")) == 1


async def test_options_change_still_calls_settings_changed(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    seeded_switched: list[dict[str, Any]],
) -> None:
    """A real options change still runs the learning settings hook."""
    entry = await _setup_entry(hass, energy_manager, std_prefs)

    with patch.object(
        LearningSupervisor, "async_settings_changed", autospec=True
    ) as mock_settings_changed:
        hass.config_entries.async_update_entry(
            entry, options={**entry.options, "max_tokens_per_call": 20_480}
        )
        await hass.async_block_till_done()

    assert mock_settings_changed.call_count == 1


async def test_subentry_user_step_aborts_use_review(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    seeded_switched: list[dict[str, Any]],
) -> None:
    """The subentry flow "Add" button cannot create devices outside Review."""
    entry = await _setup_entry(hass, energy_manager, std_prefs)

    result = await hass.config_entries.subentries.async_init(
        (entry.entry_id, "device"), context={"source": "user"}
    )
    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "use_review"
    assert entry.get_subentries_of_type("device") == []


async def test_program_candidate_offers_confirm(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
) -> None:
    """Program/session candidates offer Confirm exactly like switched ones (D-05)."""
    c_prog = make_switched_candidate(
        "sig-dishwasher", "Dishwasher", kind="program", train_share=0.06
    )
    seed_candidates(hass_storage, [c_prog])
    entry = await _setup_entry(hass, energy_manager, std_prefs)

    res_review = await async_open_options_step(hass, entry, "review")
    res_cand = await hass.config_entries.options.async_configure(
        res_review["flow_id"], {"candidate": "sig-dishwasher"}
    )
    assert res_cand["type"] is FlowResultType.MENU
    assert res_cand["menu_options"][0] == "confirm"
    assert "confirm" in res_cand["menu_options"]
    # The former "becomes possible once program detection is available" note
    # is gone with the note placeholder itself.
    assert "note" not in res_cand["description_placeholders"]
    assert entry.get_subentries_of_type("device") == []


# ---- Task 2: confirm is complete and safe (TDD) ----


async def test_preview_confirm_shows_warning_and_marks_preview(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
) -> None:
    """A preview candidate shows the unvalidated warning and records preview true (D-02)."""
    c_preview = make_switched_candidate(
        "sig-toaster",
        "Toaster",
        rise_w=800.0,
        train_share=0.021,
        val_share=None,
        validated=False,
    )
    c_valid = make_switched_candidate(
        "sig-kettle",
        "Kettle",
        rise_w=2000.0,
        train_share=0.052,
        val_share=0.048,
        validated=True,
    )
    seed_candidates(hass_storage, [c_preview, c_valid], validated=False)
    entry = await _setup_entry(hass, energy_manager, std_prefs)

    # Preview candidate: the confirm form carries the warning text
    res_review = await async_open_options_step(hass, entry, "review")
    res_cand = await hass.config_entries.options.async_configure(
        res_review["flow_id"], {"candidate": "sig-toaster"}
    )
    res_form = await hass.config_entries.options.async_configure(
        res_cand["flow_id"], {"next_step_id": "confirm"}
    )
    assert res_form["type"] is FlowResultType.FORM
    warning = res_form["description_placeholders"]["warning"]
    assert "Unvalidated" in warning
    assert "less than 21 days" in warning
    res_done = await hass.config_entries.options.async_configure(
        res_form["flow_id"], {"name": "Toaster"}
    )
    assert res_done["step_id"] == "review"

    # Validated candidate: no warning
    res_review2 = await async_open_options_step(hass, entry, "review")
    res_cand2 = await hass.config_entries.options.async_configure(
        res_review2["flow_id"], {"candidate": "sig-kettle"}
    )
    res_form2 = await hass.config_entries.options.async_configure(
        res_cand2["flow_id"], {"next_step_id": "confirm"}
    )
    assert res_form2["type"] is FlowResultType.FORM
    assert res_form2["description_placeholders"]["warning"] == ""
    res_done2 = await hass.config_entries.options.async_configure(
        res_form2["flow_id"], {"name": "Kettle"}
    )
    # Kettle was the last visible candidate: the review list now holds only the
    # confirmed device entry (Review contract C2, Plan 04-10), shown as a form.
    assert res_done2["type"] is FlowResultType.FORM
    assert res_done2["step_id"] == "review"

    # Device records carry the preview flag accordingly
    devices = entry.runtime_data.devices
    preview_uuid = devices.device_for_signature("sig-toaster")
    valid_uuid = devices.device_for_signature("sig-kettle")
    assert devices.record(preview_uuid)["preview"] is True
    assert devices.record(valid_uuid)["preview"] is False
    opts = res_done2["data_schema"].schema["candidate"].config["options"]
    assert sorted(o["value"] for o in opts) == sorted(
        [f"device_{valid_uuid}", f"device_{preview_uuid}"]
    )


async def test_review_record_gets_device_link_and_keeps_it(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    seeded_switched: list[dict[str, Any]],
) -> None:
    """Confirm links the review record to the device; later renames keep the link."""
    # Kettle already has a "kept" review record before confirm
    hass_storage[REVIEW_STORE_KEY] = {
        "version": LEARNING_STORES_VERSION,
        "minor_version": 1,
        "key": REVIEW_STORE_KEY,
        "data": {
            "decisions": {
                "sig-kettle": {
                    "status": "kept",
                    "name": None,
                    "merged_into": None,
                    "aliases": [],
                    "decided_at": "2026-11-16T12:00:00Z",
                    "fingerprint": {},
                }
            }
        },
    }
    entry = await _setup_entry(hass, energy_manager, std_prefs)

    res = await async_confirm_candidate(hass, entry, "sig-kettle", "Kettle")
    assert res["step_id"] == "review"

    sub = entry.get_subentries_of_type("device")[0]
    rec = hass_storage[REVIEW_STORE_KEY]["data"]["decisions"]["sig-kettle"]
    assert rec["status"] == "named"
    assert rec["name"] == "Kettle"
    assert rec["device_uuid"] == sub.data["device_uuid"]
    assert rec["subentry_id"] == sub.subentry_id
    assert rec["pre_confirm_status"] == "kept"

    # Renaming another candidate afterwards leaves the device link intact
    res_review = await async_open_options_step(hass, entry, "review")
    res_cand = await hass.config_entries.options.async_configure(
        res_review["flow_id"], {"candidate": "sig-oven"}
    )
    res_rename = await hass.config_entries.options.async_configure(
        res_cand["flow_id"], {"next_step_id": "rename"}
    )
    await hass.config_entries.options.async_configure(
        res_rename["flow_id"], {"name": "Backofen"}
    )
    rec = hass_storage[REVIEW_STORE_KEY]["data"]["decisions"]["sig-kettle"]
    assert rec["device_uuid"] == sub.data["device_uuid"]
    assert rec["subentry_id"] == sub.subentry_id


async def test_confirmed_candidate_leaves_review_list(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    seeded_switched: list[dict[str, Any]],
) -> None:
    """Review contract C1: a confirmed signature leaves visible_candidates and pending count."""
    entry = await _setup_entry(hass, energy_manager, std_prefs)
    learning = entry.runtime_data.learning
    pending_before = learning.coordinator.data["pending_candidates"]
    assert pending_before == 2

    res = await async_confirm_candidate(hass, entry, "sig-kettle", "Kettle")
    assert res["step_id"] == "review"

    visible = learning.visible_candidates()
    assert [c["signature_id"] for c in visible] == ["sig-oven"]
    assert learning.coordinator.data["pending_candidates"] == pending_before - 1

    # Reopening the review step offers no option for the confirmed signature
    # (never assert the total option count: Plan 04-10 adds device entries).
    res_review = await async_open_options_step(hass, entry, "review")
    opts = res_review["data_schema"].schema["candidate"].config["options"]
    values = [opt["value"] for opt in opts]
    assert "sig-kettle" not in values
    assert "sig-oven" in values

    # The review record captured the pre-confirm status for an undecided candidate
    rec = hass_storage[REVIEW_STORE_KEY]["data"]["decisions"]["sig-kettle"]
    assert rec["pre_confirm_status"] == "named"


async def test_double_confirm_creates_one_subentry(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    seeded_switched: list[dict[str, Any]],
) -> None:
    """Two option flows confirming the same candidate create one subentry (T-04-08)."""
    entry = await _setup_entry(hass, energy_manager, std_prefs)

    # Flow A opens the confirm form for Kettle
    res_review_a = await async_open_options_step(hass, entry, "review")
    res_cand_a = await hass.config_entries.options.async_configure(
        res_review_a["flow_id"], {"candidate": "sig-kettle"}
    )
    res_form_a = await hass.config_entries.options.async_configure(
        res_cand_a["flow_id"], {"next_step_id": "confirm"}
    )
    assert res_form_a["type"] is FlowResultType.FORM

    # Flow B opens its own confirm form for Kettle before A submits
    res_review_b = await async_open_options_step(hass, entry, "review")
    res_cand_b = await hass.config_entries.options.async_configure(
        res_review_b["flow_id"], {"candidate": "sig-kettle"}
    )
    res_form_b = await hass.config_entries.options.async_configure(
        res_cand_b["flow_id"], {"next_step_id": "confirm"}
    )
    assert res_form_b["type"] is FlowResultType.FORM

    # A submits: back at the review list
    res_a = await hass.config_entries.options.async_configure(
        res_form_a["flow_id"], {"name": "Kettle"}
    )
    assert res_a["type"] is FlowResultType.FORM
    assert res_a["step_id"] == "review"

    # B submits: candidate_gone, and still exactly one subentry
    res_b = await hass.config_entries.options.async_configure(
        res_form_b["flow_id"], {"name": "Kettle"}
    )
    assert res_b["type"] is FlowResultType.FORM
    assert res_b["step_id"] == "review"
    assert res_b["errors"]["base"] == "candidate_gone"

    subs = entry.get_subentries_of_type("device")
    assert len(subs) == 1
    sub = subs[0]
    devices = entry.runtime_data.devices
    assert devices.record(sub.data["device_uuid"])["subentry_id"] == sub.subentry_id
    rec = hass_storage[REVIEW_STORE_KEY]["data"]["decisions"]["sig-kettle"]
    assert rec["subentry_id"] == sub.subentry_id


async def test_confirm_abortflow_backfills_subentry_id(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    seeded_switched: list[dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AbortFlow on duplicate unique_id adopts the existing subentry (T-04-08)."""
    from types import MappingProxyType
    from unittest.mock import MagicMock

    from homeassistant.config_entries import ConfigSubentry

    from custom_components.powerguesser import devices as devices_mod
    from custom_components.powerguesser.const import SUBENTRY_TYPE_DEVICE

    entry = await _setup_entry(hass, energy_manager, std_prefs)

    # A subentry with this unique_id already exists (e.g. Plan 04-05 re-confirm)
    reused_uuid = "a" * 32
    hass.config_entries.async_add_subentry(
        entry,
        ConfigSubentry(
            data=MappingProxyType(
                {
                    "device_uuid": reused_uuid,
                    "name": "Ghost",
                    "signature_id": "sig-ghost",
                }
            ),
            subentry_type=SUBENTRY_TYPE_DEVICE,
            title="Ghost",
            unique_id=reused_uuid,
        ),
    )
    await hass.async_block_till_done()

    mock_uuid = MagicMock()
    mock_uuid.hex = reused_uuid
    with patch.object(devices_mod.uuid, "uuid4", return_value=mock_uuid):
        res = await async_confirm_candidate(hass, entry, "sig-kettle", "Kettle")
    assert res["step_id"] == "review"

    # No second subentry; record and review link carry the existing subentry id
    subs = entry.get_subentries_of_type("device")
    assert len(subs) == 1
    sub = subs[0]
    assert sub.unique_id == reused_uuid
    devices = entry.runtime_data.devices
    assert devices.device_for_signature("sig-kettle") == reused_uuid
    assert devices.record(reused_uuid)["status"] == "confirmed"
    assert devices.record(reused_uuid)["subentry_id"] == sub.subentry_id
    rec = hass_storage[REVIEW_STORE_KEY]["data"]["decisions"]["sig-kettle"]
    assert rec["subentry_id"] == sub.subentry_id

    # Reconcile keeps the record confirmed
    await devices.async_reconcile()
    assert devices.record(reused_uuid)["status"] == "confirmed"


async def test_confirm_invalid_name_or_gone_creates_nothing(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    seeded_switched: list[dict[str, Any]],
) -> None:
    """An invalid name or a replaced candidate creates no subentry and no record."""
    entry = await _setup_entry(hass, energy_manager, std_prefs)

    # Whitespace-only name
    res_review = await async_open_options_step(hass, entry, "review")
    res_cand = await hass.config_entries.options.async_configure(
        res_review["flow_id"], {"candidate": "sig-kettle"}
    )
    res_form = await hass.config_entries.options.async_configure(
        res_cand["flow_id"], {"next_step_id": "confirm"}
    )
    res_bad = await hass.config_entries.options.async_configure(
        res_form["flow_id"], {"name": "   "}
    )
    assert res_bad["type"] is FlowResultType.FORM
    assert res_bad["errors"]["base"] == "name_invalid"
    assert entry.get_subentries_of_type("device") == []

    # Candidate replaced by a learning run between menu and submit
    res_review2 = await async_open_options_step(hass, entry, "review")
    res_cand2 = await hass.config_entries.options.async_configure(
        res_review2["flow_id"], {"candidate": "sig-kettle"}
    )
    res_form2 = await hass.config_entries.options.async_configure(
        res_cand2["flow_id"], {"next_step_id": "confirm"}
    )
    entry.runtime_data.learning.candidates = [
        c
        for c in entry.runtime_data.learning.candidates
        if c.get("signature_id") != "sig-kettle"
    ]
    res_gone = await hass.config_entries.options.async_configure(
        res_form2["flow_id"], {"name": "Kettle"}
    )
    assert res_gone["type"] is FlowResultType.FORM
    assert res_gone["step_id"] == "review"
    assert res_gone["errors"]["base"] == "candidate_gone"

    # Nothing was created in either case
    assert entry.get_subentries_of_type("device") == []
    stored = hass_storage.get(DEVICES_STORE_KEY, {}).get("data", {}).get("devices", {})
    assert stored == {}


async def test_review_decisions_and_learning_create_no_devices(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    aioclient_mock: Any,
    inprocess_worker: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Name, merge and dismiss decisions and a completed learning run create no devices (REV-02)."""
    monkeypatch.setattr(
        "custom_components.powerguesser.learning.MAX_REFINE_ROUNDS", 0
    )
    c_old = make_switched_candidate(
        "sig-old", "Old Device", rise_w=9000.0, train_share=0.05, val_share=0.04
    )
    c_other = make_switched_candidate(
        "sig-other", "Other", rise_w=7000.0, train_share=0.04, val_share=0.035
    )
    seed_candidates(hass_storage, [c_old, c_other])
    entry = await _setup_entry(hass, energy_manager, std_prefs)
    entities_before = len(er.async_get(hass).entities)
    devices_before = len(dr.async_get(hass).devices)

    # Name sig-other
    res_review = await async_open_options_step(hass, entry, "review")
    res_cand = await hass.config_entries.options.async_configure(
        res_review["flow_id"], {"candidate": "sig-other"}
    )
    res_rename = await hass.config_entries.options.async_configure(
        res_cand["flow_id"], {"next_step_id": "rename"}
    )
    await hass.config_entries.options.async_configure(
        res_rename["flow_id"], {"name": "Other Device"}
    )

    # Merge sig-other into sig-old (sig-old becomes kept)
    res_review = await async_open_options_step(hass, entry, "review")
    res_cand = await hass.config_entries.options.async_configure(
        res_review["flow_id"], {"candidate": "sig-old"}
    )
    res_merge = await hass.config_entries.options.async_configure(
        res_cand["flow_id"], {"next_step_id": "merge"}
    )
    await hass.config_entries.options.async_configure(
        res_merge["flow_id"], {"sources": ["sig-other"]}
    )

    # Dismiss sig-old
    res_review = await async_open_options_step(hass, entry, "review")
    res_cand = await hass.config_entries.options.async_configure(
        res_review["flow_id"], {"candidate": "sig-old"}
    )
    await hass.config_entries.options.async_configure(
        res_cand["flow_id"], {"next_step_id": "dismiss"}
    )

    decisions = hass_storage[REVIEW_STORE_KEY]["data"]["decisions"]
    assert decisions["sig-other"]["status"] == "merged"
    assert decisions["sig-old"]["status"] == "dismissed"
    assert entry.get_subentries_of_type("device") == []

    # A completed learning run creates no device either
    from tests.conftest import (
        async_fill_synth_buffer,
        async_seed_consent,
        async_seed_llm,
        mock_llm,
        proposal_responder,
    )
    from tests.fixtures.learning.synth import SynthSpec

    spec = SynthSpec(days=22, oven=True)
    await async_fill_synth_buffer(hass, entry, spec=spec)
    await async_seed_llm(hass, entry)
    await async_seed_consent(hass, host="llm.test", names=False)
    mock_llm(aioclient_mock, "http://llm.test/v1", proposal_responder(1))
    sup = entry.runtime_data.learning
    await sup.async_request_run("service")
    assert sup.run_task is not None
    await sup.run_task

    assert sup.candidates
    assert entry.get_subentries_of_type("device") == []
    assert len(er.async_get(hass).entities) == entities_before
    assert len(dr.async_get(hass).devices) == devices_before


async def test_program_candidate_menu_matches_switched(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
) -> None:
    """A program candidate's menu is the switched menu: no note placeholder left (D-05)."""
    c_prog = make_switched_candidate(
        "sig-dishwasher", "Dishwasher", kind="program", train_share=0.06
    )
    c_kettle = make_switched_candidate(
        "sig-kettle", "Kettle", rise_w=2000.0, train_share=0.052, val_share=0.048
    )
    seed_candidates(hass_storage, [c_prog, c_kettle])
    entry = await _setup_entry(hass, energy_manager, std_prefs)

    res_review = await async_open_options_step(hass, entry, "review")
    res_cand = await hass.config_entries.options.async_configure(
        res_review["flow_id"], {"candidate": "sig-dishwasher"}
    )
    assert res_cand["type"] is FlowResultType.MENU
    assert "note" not in res_cand["description_placeholders"]

    # Switched candidates keep the same menu with Confirm first.
    res_review2 = await async_open_options_step(hass, entry, "review")
    res_cand2 = await hass.config_entries.options.async_configure(
        res_review2["flow_id"], {"candidate": "sig-kettle"}
    )
    assert res_cand2["type"] is FlowResultType.MENU
    assert res_cand2["menu_options"][0] == "confirm"
    assert "note" not in res_cand2["description_placeholders"]

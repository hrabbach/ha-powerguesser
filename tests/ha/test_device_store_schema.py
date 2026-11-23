"""Cross-plan schema-drift guard for the device Stores (D-11 costly contract).

The cold Store (powerguesser.devices v1) record key set and the hot Store
(powerguesser.device_state v1) top-level keys are pinned here. Plans 04-05..04-12
fill values into these keys and never add record keys; this test must be re-run
by every plan that writes device records.
"""

from typing import Any

import pytest

from homeassistant.components.energy.data import EnergyManager
from homeassistant.core import HomeAssistant

from custom_components.powerguesser.const import (
    DEVICES_STORE_KEY,
    DEVICES_STORE_VERSION,
    DEVICE_STATE_STORE_KEY,
)

from tests.conftest import (
    async_confirm_candidate,
    make_switched_candidate,
    seed_candidates,
)
from tests.ha.test_device_confirm import _setup_entry

# The complete v1 device-record key set (Plan 04-03 interfaces block).
# Plan 05-04 deliberately adds exactly one key, "validation" (D-06 per-device
# validation state: the derived class today, drift and rejected proposals in
# 05-05/05-06); plans never add further record keys (RESEARCH Pitfall 8).
DEVICE_RECORD_KEYS = {
    "name",
    "signature_id",
    "aliases",
    "kind",
    "preview",
    "status",
    "retired_into",
    "subentry_id",
    "versions",
    "active_v",
    "pending_proposal",
    "created_at",
    "alias_artefacts",
    "pending_refit",
    "last_refit",
    "validation",
}

VERSION_KEYS = {"v", "artefact", "trained_at", "scores", "accepted_at", "source"}


async def test_device_store_schema(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
) -> None:
    """Cold and hot Stores carry exactly the v1 key sets; missing keys get defaults."""
    c_kettle = make_switched_candidate(
        "sig-kettle", "Kettle", rise_w=2000.0, train_share=0.052, val_share=0.048
    )
    seed_candidates(hass_storage, [c_kettle])

    # A stored record from before alias_artefacts/pending_refit/last_refit existed
    legacy_uuid = "b" * 32
    legacy_rec = {
        "name": "Legacy",
        "signature_id": "sig-legacy",
        "aliases": [],
        "kind": "switched",
        "preview": False,
        "status": "confirmed",
        "retired_into": None,
        "subentry_id": None,
        "versions": [],
        "active_v": 0,
        "pending_proposal": None,
        "created_at": "2026-11-10T12:00:00Z",
    }
    hass_storage[DEVICES_STORE_KEY] = {
        "version": DEVICES_STORE_VERSION,
        "minor_version": 1,
        "key": DEVICES_STORE_KEY,
        "data": {"devices": {legacy_uuid: legacy_rec}},
    }
    entry = await _setup_entry(hass, energy_manager, std_prefs)
    devices = entry.runtime_data.devices

    # The loaded legacy record gains the missing keys with their defaults
    legacy = devices.record(legacy_uuid)
    assert legacy is not None
    assert set(legacy.keys()) == DEVICE_RECORD_KEYS
    assert legacy["alias_artefacts"] == {}
    assert legacy["pending_refit"] is False
    assert legacy["last_refit"] is None
    # Plan 05-04: a record stored before the validation key existed loads
    # with the default empty dict.
    assert legacy["validation"] == {}

    # Confirm Kettle (the only candidate; the review list then holds only the
    # confirmed device entry as a form, Review contract C2 / Plan 04-10)
    res = await async_confirm_candidate(hass, entry, "sig-kettle", "Kettle")
    assert res["type"] == "form"
    uuid = devices.device_for_signature("sig-kettle")
    opts = res["data_schema"].schema["candidate"].config["options"]
    assert [o["value"] for o in opts] == [f"device_{uuid}"]
    stored = hass_storage[DEVICES_STORE_KEY]["data"]["devices"][uuid]
    assert set(stored.keys()) == DEVICE_RECORD_KEYS
    assert set(stored["versions"][0].keys()) == VERSION_KEYS
    assert stored["status"] == "confirmed"
    assert stored["active_v"] == 1
    assert stored["alias_artefacts"] == {}
    assert stored["pending_refit"] is False
    assert stored["last_refit"] is None
    assert stored["pending_proposal"] is None

    # Hot Store: exact top-level keys; both Stores stay at version 1.
    # Plan 05-05 deliberately adds the "drift" key (D-15 live drift evidence,
    # RESEARCH Pattern 4).
    hot = hass_storage[DEVICE_STATE_STORE_KEY]["data"]
    assert set(hot.keys()) == {"energy_wh", "detector", "drift"}
    assert hot["energy_wh"][uuid] == 0.0
    assert hass_storage[DEVICES_STORE_KEY]["version"] == DEVICES_STORE_VERSION
    assert hass_storage[DEVICE_STATE_STORE_KEY]["version"] == DEVICES_STORE_VERSION

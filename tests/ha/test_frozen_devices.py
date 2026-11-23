"""Frozen-device invariant across the whole holdout-validation phase (VAL-05 SC3).

One test proves that nothing this phase added — learning runs (a gated
proposal and a rejected one), live drift set and cleared, class-holdout
drift set through run_validation, and run_validation itself — ever changes a
confirmed device behind the user's back: versions, active_v, signature_id,
name, uuid, subentry title, the power/energy/validation entity ids and the
energy totals stay exactly as confirmed; only the user's explicit Accept
creates version 2 (D-16, D-17, D-23).
"""

import copy
import json
from typing import Any

import pytest

from homeassistant.components.energy.data import EnergyManager
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers import issue_registry as ir
from homeassistant.util import dt as dt_util

from custom_components.powerguesser.const import (
    VALIDATION_STORE_KEY,
    VALIDATION_STORES_VERSION,
)

from tests.conftest import (
    async_confirm_candidate,
    async_fill_synth_buffer,
    async_seed_consent,
    async_seed_llm,
    async_seed_shadow_set,
    async_setup_via_flow,
    mock_llm,
)
from tests.fixtures.learning.synth import SynthSpec
from tests.ha.test_device_detection import _aligned_move, _pin_grid
from tests.ha.test_device_drift import (
    _class_entry,
    _class_stub,
    _over_counted_class_state,
    _run_validation,
)
from tests.ha.test_device_proposals import (
    _device_ids,
    _differing_entry,
    _kettle_responder,
    _kettle_wh_responder,
    _run_learning,
    _set_std_powers,
)

KETTLE_CLASS = "switched_1.5_3kw_1_level"
WINDOW_A = "2026-11-10"
WINDOW_B = "2026-11-17"
WINDOW_C = "2026-11-24"


def _assert_frozen(
    devices: Any,
    entry: ConfigEntry,
    hass: HomeAssistant,
    dev_uuid: str,
    snapshot: dict[str, Any],
) -> None:
    """The whole frozen surface is byte-identical to the post-confirm snapshot."""
    rec = devices.record(dev_uuid)
    assert rec is not None
    assert json.dumps(rec["versions"], sort_keys=True) == snapshot["versions"]
    assert rec["active_v"] == snapshot["active_v"]
    assert rec["signature_id"] == snapshot["signature_id"]
    assert rec["name"] == snapshot["name"]
    assert rec["preview"] is False
    sub = next(s for s in entry.get_subentries_of_type("device") if s.title == "Kettle")
    assert sub.subentry_id == snapshot["subentry_id"]
    assert sub.data["device_uuid"] == dev_uuid
    assert _device_ids(hass, entry, dev_uuid) == snapshot["ids"]
    assert set(er.async_get(hass).entities) == snapshot["entities"]
    assert set(ir.async_get(hass).issues) == snapshot["issues"]
    # The energy total is untouched by learning, drift and validation.
    assert devices._energy_wh.get(dev_uuid, 0.0) == snapshot["energy_wh"]


async def test_confirmed_device_never_silently_changes(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    aioclient_mock: Any,
    inprocess_worker: Any,
    monkeypatch: pytest.MonkeyPatch,
    hass_storage: dict[str, Any],
    freezer: Any,
) -> None:
    """Across gated and rejected proposals, live drift set/clear,
    class-holdout drift and run_validation, a confirmed device changes only
    through the user's explicit Accept (VAL-05 SC3)."""
    from custom_components.powerguesser.learning import LearningSupervisor

    monkeypatch.setattr("custom_components.powerguesser.learning.MAX_REFINE_ROUNDS", 0)
    energy_manager.data = std_prefs
    _set_std_powers(hass)

    # Seed the kettle class's holdout history (the Task 2 path) before load.
    hass_storage[VALIDATION_STORE_KEY] = {
        "version": VALIDATION_STORES_VERSION,
        "minor_version": 1,
        "key": VALIDATION_STORE_KEY,
        "data": {
            "report": None,
            "plug_classes": {},
            "history": {
                f"class:{KETTLE_CLASS}": [
                    {"shadow_run_id": "test-shadow", "holdout_end": WINDOW_A, "energy_error_pct": 8.0},
                    {"shadow_run_id": "test-shadow", "holdout_end": WINDOW_B, "energy_error_pct": 20.0},
                ]
            },
            "holdout_drift": {
                f"class:{KETTLE_CLASS}": _over_counted_class_state(WINDOW_B)
            },
        },
    }

    _aligned_move(freezer)
    entry = await async_setup_via_flow(hass)
    _pin_grid(entry)
    await hass.async_block_till_done()
    await async_fill_synth_buffer(
        hass, entry, spec=SynthSpec(days=22, oven=True, water_heater=True)
    )
    await async_seed_llm(hass, entry)
    await async_seed_consent(hass, host="llm.test", names=False)

    call_count = 0

    def _responder(data: Any) -> dict[str, Any]:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _kettle_responder("Kettle")(data)
        return _kettle_wh_responder("Kettle")(data)

    mock_llm(aioclient_mock, "http://llm.test/v1", _responder)
    sup = entry.runtime_data.learning
    devices = entry.runtime_data.devices

    # --- Run 1: propose and confirm the kettle; take the frozen snapshot ---
    await _run_learning(sup)
    sig_kettle = sup.visible_candidates()[0]["signature_id"]
    await async_confirm_candidate(hass, entry, sig_kettle, "Kettle")
    sub = next(s for s in entry.get_subentries_of_type("device") if s.title == "Kettle")
    dev_uuid = sub.data["device_uuid"]
    rec = devices.record(dev_uuid)
    snapshot = {
        "versions": json.dumps(rec["versions"], sort_keys=True),
        "active_v": rec["active_v"],
        "signature_id": rec["signature_id"],
        "name": rec["name"],
        "subentry_id": sub.subentry_id,
        "ids": _device_ids(hass, entry, dev_uuid),
        "entities": set(er.async_get(hass).entities),
        "issues": set(ir.async_get(hass).issues),
        "energy_wh": devices._energy_wh.get(dev_uuid, 0.0),
    }
    _assert_frozen(devices, entry, hass, dev_uuid, snapshot)

    # --- Run 2: the differing two-state proposal passes the D-17 gate and
    # is offered; nothing else moves ---
    await _run_learning(sup)
    assert devices.record(dev_uuid)["pending_proposal"] is not None
    _assert_frozen(devices, entry, hass, dev_uuid, snapshot)

    # --- A gate-failing proposal is rejected and recorded for diagnosis ---
    art = _differing_entry(
        copy.deepcopy(devices.record(dev_uuid)["versions"][0]["artefact"]),
        sig_kettle,
    )
    art["proposal_gate"] = {
        "passed": False,
        "proposed_S": 0.01234,
        "active_S": 0.05678,
        "basis": "learning_split_val",
    }
    await sup._route_device_matches([art], "run-reject", dt_util.utcnow().isoformat())
    rejected = devices.record(dev_uuid)["validation"]["last_rejected_proposal"]
    assert rejected["reason"] == "worse_than_active"
    assert devices.record(dev_uuid)["pending_proposal"] is not None
    _assert_frozen(devices, entry, hass, dev_uuid, snapshot)

    # --- Live drift set from seeded evidence, then cleared by the D-22
    # activation reset (no episode energy is driven into the device) ---
    devices.record(dev_uuid)["validation"]["drift"] = {
        "drift": True,
        "reason": "match_score_low",
        "since": dt_util.utcnow().isoformat(),
    }
    devices._republish()
    await hass.async_block_till_done()
    assert devices.attributes(dev_uuid)["drift"] is True
    _assert_frozen(devices, entry, hass, dev_uuid, snapshot)

    await devices.async_activate_version(dev_uuid, 1)
    assert devices.attributes(dev_uuid)["drift"] is False
    _assert_frozen(devices, entry, hass, dev_uuid, snapshot)

    # --- Class-holdout drift through run_validation (the Task 2 path) ---
    await async_seed_shadow_set(hass, entry, [], [])
    monkeypatch.setattr(
        LearningSupervisor,
        "_async_run_stage",
        _class_stub([_class_entry(30.0)]),
    )
    res = await _run_validation(hass)
    klass = next(c for c in res["classes"] if c["class"] == KETTLE_CLASS)
    assert klass["drift"] is True
    attrs = devices.attributes(dev_uuid)
    assert attrs["drift"] is True
    assert attrs["drift_reason"] == "validation_error_rising"
    assert attrs["drift_since"] == WINDOW_C
    # The record itself is untouched: class drift lives in the validation
    # Store and is derived at read time (D-26 (3)).
    assert "drift" not in devices.record(dev_uuid)["validation"]
    _assert_frozen(devices, entry, hass, dev_uuid, snapshot)

    # --- Only the user's explicit Accept creates version 2 ---
    v2 = await devices.async_accept_proposal(dev_uuid)
    assert v2 == 2
    rec = devices.record(dev_uuid)
    assert [v["v"] for v in rec["versions"]] == [1, 2]
    assert rec["active_v"] == 2
    # Identity, entities, subentry and energy totals still hold.
    assert rec["signature_id"] == snapshot["signature_id"]
    assert rec["name"] == snapshot["name"]
    sub = next(s for s in entry.get_subentries_of_type("device") if s.title == "Kettle")
    assert sub.subentry_id == snapshot["subentry_id"]
    assert _device_ids(hass, entry, dev_uuid) == snapshot["ids"]
    assert set(er.async_get(hass).entities) == snapshot["entities"]
    assert devices._energy_wh.get(dev_uuid, 0.0) == snapshot["energy_wh"]

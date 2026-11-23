"""Tests for the per-device validation surface (Plan 05-04; VAL-02, VAL-03).

The diagnostic validation sensor per confirmed device, the class-holdout
basis on the power and energy sensors, and the one filterable unvalidated
flag everywhere else. Plan 05-02 owns the report's classes; this file owns
what devices and sensors show of it.
"""

import json
from pathlib import Path
from typing import Any

import pytest

from homeassistant.components.energy.data import EnergyManager
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import EntityCategory
from homeassistant.core import HomeAssistant
from homeassistant.helpers import entity_registry as er

from custom_components.powerguesser.const import DOMAIN

from tests.conftest import (
    async_confirm_candidate,
    async_fill_synth_buffer,
    async_seed_shadow_set,
    async_set_power,
    async_setup_via_flow,
    make_switched_candidate,
    seed_candidates,
    seed_validation_report,
)

KETTLE_CLASS = "switched_1.5_3kw_1_level"

# The published device-sensor attribute order (D-17 with the Plan 05-04 keys
# and the Plan 05-05 drift keys appended, retired last on retired records
# only).
TWELVE_KEYS = [
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
]

# The diagnostic sensor's fixed thirteen-key attribute order (D-14 with the
# Plan 05-05 drift_evidence key; drift_evidence stays recorded: low churn and
# user-relevant, unlike the eight high-churn provenance keys).
DIAGNOSTIC_KEYS = [
    "unvalidated",
    "validation_basis",
    "validation_class",
    "validation_peers",
    "validated_via",
    "nde",
    "mae_w",
    "baseline_nde",
    "n_events",
    "holdout_start",
    "holdout_end",
    "train_cutoff",
    "drift_evidence",
]

DIAGNOSTIC_UNRECORDED = frozenset(
    {
        "validated_via",
        "nde",
        "mae_w",
        "baseline_nde",
        "n_events",
        "holdout_start",
        "holdout_end",
        "train_cutoff",
    }
)

HOLDOUT_START = "2026-11-10T00:00:00+00:00"
HOLDOUT_END = "2026-11-17T00:00:00+00:00"
TRAIN_CUTOFF = "2026-11-10T00:00:00+00:00"


def _passing_class(class_name: str = KETTLE_CLASS, **overrides: Any) -> dict[str, Any]:
    """One D-08-passing class entry as Plan 05-02's report builder writes it."""
    entry: dict[str, Any] = {
        "class": class_name,
        "peers": 3,
        "passing": 2,
        "median_error_pct": 11.0,
        "max_error_pct": 20.0,
        "median_nde": 0.25,
        "median_mae_w": 180.0,
        "median_baseline_nde": 0.4,
        "n_events": 30,
        "peer_entities": [
            "sensor.kettle_power",
            "sensor.plug_two_power",
            "sensor.plug_three_power",
        ],
        "passes": True,
    }
    entry.update(overrides)
    return entry


def _failing_class(class_name: str = KETTLE_CLASS, **overrides: Any) -> dict[str, Any]:
    """The kettle's class with a D-08 failure (one passing peer)."""
    return _passing_class(
        class_name,
        peers=2,
        passing=1,
        median_error_pct=30.0,
        max_error_pct=38.0,
        passes=False,
        **overrides,
    )


def _ok_report(classes: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """An ok report in the stored v1 shape (ISO strings already stamped)."""
    return {
        "status": "ok",
        "reason": None,
        "train_cutoff": TRAIN_CUTOFF,
        "holdout": {"start": HOLDOUT_START, "end": HOLDOUT_END},
        "classes": classes if classes is not None else [_passing_class()],
        "devices": [],
        "refusals": [],
        "not_discovered": [],
    }


async def _setup_confirmed(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    candidates: list[dict[str, Any]],
    report: dict[str, Any] | None = None,
    plug_classes: dict[str, str] | None = None,
) -> ConfigEntry:
    """Seed candidates (+ optional validation report), set up, confirm all."""
    if report is not None:
        seed_validation_report(hass_storage, report, plug_classes)
    seed_candidates(hass_storage, candidates)
    energy_manager.data = std_prefs
    async_set_power(hass, "sensor.grid_power", 500)
    async_set_power(hass, "sensor.solar_power", 0)
    async_set_power(hass, "sensor.battery_power", 0)
    async_set_power(hass, "sensor.washer_power", 200)
    entry = await async_setup_via_flow(hass)
    await hass.async_block_till_done()
    for cand in candidates:
        await async_confirm_candidate(
            hass, entry, str(cand["signature_id"]), str(cand["label"])
        )
    await hass.async_block_till_done()
    return entry


def _device_entity_ids(
    hass: HomeAssistant, entry: ConfigEntry, dev_uuid: str
) -> tuple[str | None, str | None, str | None]:
    """Return (power, energy, validation) entity ids of one device."""
    ent_reg = er.async_get(hass)
    return (
        ent_reg.async_get_entity_id("sensor", DOMAIN, f"{entry.entry_id}_{dev_uuid}_power"),
        ent_reg.async_get_entity_id("sensor", DOMAIN, f"{entry.entry_id}_{dev_uuid}_energy"),
        ent_reg.async_get_entity_id(
            "sensor", DOMAIN, f"{entry.entry_id}_{dev_uuid}_validation"
        ),
    )


def _extra(hass: HomeAssistant, entity_id: str) -> dict[str, Any]:
    """Return the live entity's published extra_state_attributes."""
    component = hass.data["entity_components"]["sensor"]
    entity = component.get_entity(entity_id)
    assert entity is not None, f"entity not found: {entity_id}"
    attrs = entity.extra_state_attributes
    assert isinstance(attrs, dict)
    return attrs


def _entity(hass: HomeAssistant, entity_id: str) -> Any:
    """Return the live sensor entity instance behind an entity id."""
    component = hass.data["entity_components"]["sensor"]
    entity = component.get_entity(entity_id)
    assert entity is not None, f"entity not found: {entity_id}"
    return entity


async def test_class_pass_shows_class_holdout_on_all_sensors(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
) -> None:
    """A confirmed kettle in a class that passed on this installation shows the
    class-holdout error on its power and energy sensors and on the new
    diagnostic sensor (D-09, D-14, D-19)."""
    candidate = make_switched_candidate("sig-kettle", "Kettle", rise_w=2000.0)
    candidate["scores"]["validation_error_pct"] = 3.2
    candidate["scores"]["validation_basis"] = "learning_split"
    entry = await _setup_confirmed(
        hass,
        energy_manager,
        std_prefs,
        hass_storage,
        [candidate],
        report=_ok_report(),
    )
    dev_uuid = entry.runtime_data.devices.device_for_signature("sig-kettle")
    assert dev_uuid is not None
    power_eid, energy_eid, validation_eid = _device_entity_ids(hass, entry, dev_uuid)
    assert validation_eid is not None

    # The class-holdout basis replaces the learning-split figure (D-09) while
    # status stays provisional (no confirmed device has its own plug in v1).
    for eid in (power_eid, energy_eid):
        state = hass.states.get(eid)
        assert state is not None
        assert state.attributes["validation_basis"] == "class_holdout"
        assert state.attributes["validation_error_pct"] == 11.0
        assert state.attributes["validation_class"] == KETTLE_CLASS
        assert state.attributes["validation_peers"] == 3
        assert state.attributes["unvalidated"] is False
        assert state.attributes["status"] == "provisional"

    # The third entity: diagnostic, on the device's DeviceInfo, numeric %.
    val_state = hass.states.get(validation_eid)
    assert val_state is not None
    assert val_state.state == "11.0"
    assert val_state.attributes["unit_of_measurement"] == "%"
    assert val_state.attributes["state_class"] == "measurement"
    val_entry = er.async_get(hass).async_get(validation_eid)
    assert val_entry is not None
    assert val_entry.entity_category == EntityCategory.DIAGNOSTIC
    attrs = val_state.attributes
    assert attrs["unvalidated"] is False
    assert attrs["validation_basis"] == "class_holdout"
    assert attrs["validation_class"] == KETTLE_CLASS
    assert attrs["validation_peers"] == 3
    # The provenance attributes name the class and carry its medians.
    assert attrs["validated_via"] == KETTLE_CLASS
    assert attrs["nde"] == 0.25
    assert attrs["mae_w"] == 180.0
    assert attrs["baseline_nde"] == 0.4
    assert attrs["n_events"] == 30
    assert attrs["holdout_start"] == HOLDOUT_START
    assert attrs["holdout_end"] == HOLDOUT_END
    assert attrs["train_cutoff"] == TRAIN_CUTOFF


async def test_run_validation_refreshes_device_sensors(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    kettle_plug_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    aioclient_mock: Any,
    freezer: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After run_validation the confirmed device's sensors switch to
    class_holdout without waiting for an aligner publish (VAL-03)."""
    from tests.fixtures.learning.synth import SynthSpec

    from custom_components.powerguesser.learning import LearningSupervisor

    from tests.ha.test_device_sensors import _aligned_move, _drive_grid, _pin_grid

    energy_manager.data = kettle_plug_prefs
    async_set_power(hass, "sensor.grid_power", 500)
    async_set_power(hass, "sensor.solar_power", 1200)
    async_set_power(hass, "sensor.battery_power", -300)
    async_set_power(hass, "sensor.washer_power", 200)
    async_set_power(hass, "sensor.kettle_power", 0)

    candidate = make_switched_candidate("sig-kettle", "Kettle", rise_w=2000.0)
    candidate["scores"]["validation_error_pct"] = 3.2
    candidate["scores"]["validation_basis"] = "learning_split"
    seed_candidates(hass_storage, [candidate])
    entry = await async_setup_via_flow(hass)
    await async_fill_synth_buffer(hass, entry, SynthSpec(days=35, kettle_plug=True))

    from tests.fixtures.detect_traces import switched_artefact

    artefact = switched_artefact(
        "sig-shadow-kettle",
        rise=(1800, 2000, 2200),
        fall=(-2200, -2000, -1800),
        dur=(120, 180, 300),
        plateau=(1800, 2000, 2200),
    )
    await async_seed_shadow_set(
        hass,
        entry,
        [artefact],
        [
            {
                "entity_id": "sensor.kettle_power",
                "signature_id": "sig-shadow-kettle",
                "kind": "switched",
                "overlap": 0.8,
            }
        ],
    )

    await async_confirm_candidate(hass, entry, "sig-kettle", "Kettle")
    await hass.async_block_till_done()
    dev_uuid = entry.runtime_data.devices.device_for_signature("sig-kettle")
    assert dev_uuid is not None
    power_eid, energy_eid, validation_eid = _device_entity_ids(hass, entry, dev_uuid)
    assert validation_eid is not None

    # One aligner publish so the coordinator holds a snapshot; from here on
    # nothing may publish again before the assertions.
    _aligned_move(freezer)
    _pin_grid(entry)
    await _drive_grid(hass, freezer, 2)

    # Before validation: the Phase 4 learning-split figure, unvalidated.
    for eid in (power_eid, energy_eid):
        assert hass.states.get(eid).attributes["validation_basis"] == "learning_split"
        assert hass.states.get(eid).attributes["unvalidated"] is True
    assert hass.states.get(validation_eid).state == "unknown"
    assert hass.states.get(validation_eid).attributes["unvalidated"] is True

    # Stub the validate stage with an ok report whose class passes the kettle.
    async def _stub(self: Any, stage: str, job_path: Any) -> dict[str, Any]:
        job = json.loads(Path(str(job_path)).read_text(encoding="utf-8"))
        report = {
            "status": "ok",
            "reason": None,
            "train_cutoff_ms": job["shadow"]["train_cutoff_ms"],
            "holdout": {
                "start_ms": job["shadow"]["train_cutoff_ms"],
                "end_ms": job["shadow"]["train_cutoff_ms"] + 7 * 86_400_000,
                "days": [],
            },
            "classes": [_passing_class()],
            "devices": [],
            "refusals": [],
            "not_discovered": [],
        }
        (Path(job["stage_dir"]) / "report.json").write_text(
            json.dumps(report), encoding="utf-8"
        )
        return {"stats": {"wall_s": 0.0, "max_rss_kb": 0}}

    monkeypatch.setattr(LearningSupervisor, "_async_run_stage", _stub)

    res = await hass.services.async_call(
        DOMAIN, "run_validation", {}, blocking=True, return_response=True
    )
    assert res["status"] == "ok"

    # No aligner publish happened in between: the switch rode apply_validation.
    for eid in (power_eid, energy_eid):
        state = hass.states.get(eid)
        assert state.attributes["validation_basis"] == "class_holdout"
        assert state.attributes["validation_error_pct"] == 11.0
        assert state.attributes["validation_class"] == KETTLE_CLASS
        assert state.attributes["validation_peers"] == 3
        assert state.attributes["unvalidated"] is False
    val_state = hass.states.get(validation_eid)
    assert val_state.state == "11.0"
    assert val_state.attributes["unvalidated"] is False

    # Local and deterministic: no HTTP request was made.
    assert len(aioclient_mock.mock_calls) == 0


@pytest.mark.parametrize("with_figure", [True, False])
async def test_no_report_marks_unvalidated(
    with_figure: bool,
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
) -> None:
    """No report stored: the kettle shows unvalidated with its Phase 4 figure
    (or null where it had none) and the diagnostic sensor reads unknown."""
    candidate = make_switched_candidate("sig-kettle", "Kettle", rise_w=2000.0)
    if with_figure:
        candidate["scores"]["validation_error_pct"] = 3.2
        candidate["scores"]["validation_basis"] = "learning_split"
    entry = await _setup_confirmed(
        hass, energy_manager, std_prefs, hass_storage, [candidate]
    )
    dev_uuid = entry.runtime_data.devices.device_for_signature("sig-kettle")
    assert dev_uuid is not None
    power_eid, energy_eid, validation_eid = _device_entity_ids(hass, entry, dev_uuid)

    for eid in (power_eid, energy_eid):
        attrs = _extra(hass, eid)
        assert attrs["unvalidated"] is True
        assert attrs["validation_peers"] == 0
        assert attrs["validation_class"] == KETTLE_CLASS
        if with_figure:
            assert attrs["validation_error_pct"] == 3.2
            assert attrs["validation_basis"] == "learning_split"
        else:
            assert attrs["validation_error_pct"] is None
            assert attrs["validation_basis"] is None

    val_state = hass.states.get(validation_eid)
    assert val_state is not None
    assert val_state.state == "unknown"
    assert val_state.attributes["unvalidated"] is True
    # D-20: the Phase 4 basis stays visible alongside unvalidated true.
    assert val_state.attributes["validation_basis"] == (
        "learning_split" if with_figure else None
    )


async def test_refused_report_never_validates(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
) -> None:
    """A refused latest report clears the class results even though an older
    ok report had passed the class (insufficient_data, D-12/T-05-11)."""
    entry = await _setup_confirmed(
        hass,
        energy_manager,
        std_prefs,
        hass_storage,
        [make_switched_candidate("sig-kettle", "Kettle", rise_w=2000.0)],
        report=_ok_report(),
    )
    dev_uuid = entry.runtime_data.devices.device_for_signature("sig-kettle")
    assert dev_uuid is not None
    power_eid, energy_eid, validation_eid = _device_entity_ids(hass, entry, dev_uuid)
    devices = entry.runtime_data.devices

    # The seeded ok report validated the class on load.
    assert _extra(hass, power_eid)["validation_basis"] == "class_holdout"

    # No shadow set stored: run_validation refuses, and the refusal is the
    # latest report from here on.
    res = await hass.services.async_call(
        DOMAIN, "run_validation", {}, blocking=True, return_response=True
    )
    assert res["status"] == "refused"

    for eid in (power_eid, energy_eid):
        attrs = _extra(hass, eid)
        assert attrs["unvalidated"] is True
        assert attrs["validation_peers"] == 0
        assert attrs["validation_class"] == KETTLE_CLASS
        assert attrs["validation_basis"] is None
    state, details = devices.validation_details(dev_uuid)
    assert state is None
    assert details["unvalidated"] is True
    assert hass.states.get(validation_eid).state == "unknown"


async def test_failed_class_keeps_learning_split(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
) -> None:
    """The kettle's class exists but fails D-08: unvalidated with the class's
    peer count and the Phase 4 figure kept (D-20)."""
    candidate = make_switched_candidate("sig-kettle", "Kettle", rise_w=2000.0)
    candidate["scores"]["validation_error_pct"] = 3.2
    candidate["scores"]["validation_basis"] = "learning_split"
    entry = await _setup_confirmed(
        hass,
        energy_manager,
        std_prefs,
        hass_storage,
        [candidate],
        report=_ok_report(classes=[_failing_class()]),
    )
    dev_uuid = entry.runtime_data.devices.device_for_signature("sig-kettle")
    assert dev_uuid is not None
    power_eid, energy_eid, validation_eid = _device_entity_ids(hass, entry, dev_uuid)

    for eid in (power_eid, energy_eid):
        attrs = _extra(hass, eid)
        assert attrs["unvalidated"] is True
        assert attrs["validation_peers"] == 2
        assert attrs["validation_class"] == KETTLE_CLASS
        assert attrs["validation_error_pct"] == 3.2
        assert attrs["validation_basis"] == "learning_split"
    assert hass.states.get(validation_eid).state == "unknown"


@pytest.mark.parametrize("scenario", ["class_pass", "class_fail", "refused"])
async def test_status_never_validated(
    scenario: str,
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
) -> None:
    """status is provisional in every case and no attribute value is ever
    "validated" (D-09, D-16 by construction)."""
    if scenario == "class_pass":
        report = _ok_report()
    elif scenario == "class_fail":
        report = _ok_report(classes=[_failing_class()])
    else:
        report = _ok_report()
    entry = await _setup_confirmed(
        hass,
        energy_manager,
        std_prefs,
        hass_storage,
        [make_switched_candidate("sig-kettle", "Kettle", rise_w=2000.0)],
        report=report,
    )
    dev_uuid = entry.runtime_data.devices.device_for_signature("sig-kettle")
    assert dev_uuid is not None
    power_eid, energy_eid, validation_eid = _device_entity_ids(hass, entry, dev_uuid)

    if scenario == "refused":
        res = await hass.services.async_call(
            DOMAIN, "run_validation", {}, blocking=True, return_response=True
        )
        assert res["status"] == "refused"

    for eid in (power_eid, energy_eid, validation_eid):
        attrs = dict(_extra(hass, eid))
        if eid == power_eid:
            # The live keys follow the twelve-key static set on the power sensor.
            assert list(attrs.keys()) == TWELVE_KEYS + [
                "match_score",
                "unattributed_edges",
                "update_available",
                "overlap_refused",
            ]
        elif eid == energy_eid:
            assert list(attrs.keys()) == TWELVE_KEYS
        else:
            assert list(attrs.keys()) == DIAGNOSTIC_KEYS
        for key, value in attrs.items():
            assert value != "validated", f"{eid} attribute {key}: {value}"
    # The status key itself stays provisional on the device sensors.
    assert _extra(hass, power_eid)["status"] == "provisional"
    assert _extra(hass, energy_eid)["status"] == "provisional"


async def test_attribute_key_order_with_retired(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
) -> None:
    """Nine keys on both sensors (power then its live keys); a retired device
    lists the nine keys then retired, unvalidated (D-17, D-12)."""
    candidates = [
        make_switched_candidate("sig-kettle", "Kettle", rise_w=2000.0),
        make_switched_candidate("sig-oven", "Oven", rise_w=1800.0),
    ]
    entry = await _setup_confirmed(
        hass, energy_manager, std_prefs, hass_storage, candidates
    )
    devices = entry.runtime_data.devices
    uuid_kettle = devices.device_for_signature("sig-kettle")
    uuid_oven = devices.device_for_signature("sig-oven")
    assert uuid_kettle is not None and uuid_oven is not None
    power_k, energy_k, _ = _device_entity_ids(hass, entry, uuid_kettle)
    power_o, energy_o, _ = _device_entity_ids(hass, entry, uuid_oven)

    assert list(_extra(hass, power_k).keys()) == TWELVE_KEYS + [
        "match_score",
        "unattributed_edges",
        "update_available",
        "overlap_refused",
    ]
    assert list(_extra(hass, energy_k).keys()) == TWELVE_KEYS

    # Retire the oven in place (merge outcome, D-12): the retired key stays
    # last of the static set (the power sensor's live keys follow, as in
    # Phase 4) and the device is never holdout-backed.
    devices.record(uuid_oven)["status"] = "retired"
    for eid, expected in (
        (
            power_o,
            TWELVE_KEYS
            + [
                "retired",
                "match_score",
                "unattributed_edges",
                "update_available",
                "overlap_refused",
            ],
        ),
        (energy_o, TWELVE_KEYS + ["retired"]),
    ):
        attrs = _extra(hass, eid)
        assert list(attrs.keys()) == expected
        assert attrs["unvalidated"] is True
        assert attrs["retired"] is True


async def test_diagnostic_attributes_and_unrecorded(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
) -> None:
    """The diagnostic sensor publishes the fixed thirteen-key order and keeps
    exactly the eight provenance keys out of the recorder (D-14, T-05-13)."""
    from custom_components.powerguesser.sensor import DeviceValidationSensor

    entry = await _setup_confirmed(
        hass,
        energy_manager,
        std_prefs,
        hass_storage,
        [make_switched_candidate("sig-kettle", "Kettle", rise_w=2000.0)],
        report=_ok_report(),
    )
    dev_uuid = entry.runtime_data.devices.device_for_signature("sig-kettle")
    assert dev_uuid is not None
    _power_eid, _energy_eid, validation_eid = _device_entity_ids(hass, entry, dev_uuid)
    entity = _entity(hass, validation_eid)
    assert isinstance(entity, DeviceValidationSensor)
    assert entity._unrecorded_attributes == DIAGNOSTIC_UNRECORDED
    assert list(entity.extra_state_attributes.keys()) == DIAGNOSTIC_KEYS


async def test_validation_unique_id_stable_and_class_refresh_on_activation(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
) -> None:
    """The validation entity keeps its unique_id across a version activation
    that moves the device's class, and the new class re-evaluates it (D-06)."""
    entry = await _setup_confirmed(
        hass,
        energy_manager,
        std_prefs,
        hass_storage,
        [make_switched_candidate("sig-kettle", "Kettle", rise_w=2000.0)],
        report=_ok_report(),
    )
    devices = entry.runtime_data.devices
    dev_uuid = devices.device_for_signature("sig-kettle")
    assert dev_uuid is not None
    power_eid, _energy_eid, validation_eid = _device_entity_ids(hass, entry, dev_uuid)
    assert _extra(hass, power_eid)["validation_basis"] == "class_holdout"

    # Version 2 draws the class into the >3 kW band: no passing peer class
    # for it in the report, so the device becomes unvalidated again.
    rec = devices.record(dev_uuid)
    big = make_switched_candidate("sig-kettle", "Kettle", rise_w=4000.0)
    artefact2 = {
        k: v for k, v in big.items() if k not in ("identity", "fingerprint", "first_label")
    }
    rec["versions"].append(
        {
            "v": 2,
            "artefact": artefact2,
            "trained_at": "2026-11-18T12:00:00Z",
            "scores": dict(big["scores"]),
            "accepted_at": "2026-11-18T12:00:00Z",
            "source": "proposal",
        }
    )
    await devices.async_activate_version(dev_uuid, 2)

    assert rec["validation"]["class"] == "switched_over_3kw_1_level"
    # The same entity (unique_id unchanged) now reads unvalidated.
    power_eid2, _energy_eid2, validation_eid2 = _device_entity_ids(hass, entry, dev_uuid)
    assert power_eid2 == power_eid
    assert validation_eid2 == validation_eid
    attrs = _extra(hass, power_eid)
    assert attrs["validation_class"] == "switched_over_3kw_1_level"
    assert attrs["unvalidated"] is True
    assert attrs["validation_basis"] is None
    assert hass.states.get(validation_eid).state == "unknown"


async def test_report_lists_confirmed_devices_coverage(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
) -> None:
    """The run_validation report lists confirmed_devices with name, class,
    basis and unvalidated, sorted by name (VAL-03 coverage)."""
    from custom_components.powerguesser.const import VALIDATION_STORE_KEY

    kettle = make_switched_candidate("sig-kettle", "Kettle", rise_w=2000.0)
    kettle["scores"]["validation_error_pct"] = 3.2
    kettle["scores"]["validation_basis"] = "learning_split"
    oven = make_switched_candidate("sig-oven", "Oven", rise_w=1800.0)
    entry = await _setup_confirmed(
        hass,
        energy_manager,
        std_prefs,
        hass_storage,
        [kettle, oven],
        report=_ok_report(),
    )
    devices = entry.runtime_data.devices

    # With the passing class report: both devices carry the class figure.
    assert devices.coverage_summary() == [
        {
            "name": "Kettle",
            "validation_class": KETTLE_CLASS,
            "validation_basis": "class_holdout",
            "unvalidated": False,
        },
        {
            "name": "Oven",
            "validation_class": KETTLE_CLASS,
            "validation_basis": "class_holdout",
            "unvalidated": False,
        },
    ]

    # A refused run_validation (no shadow set) still reports coverage, now
    # with every device unvalidated and back on its Phase 4 basis.
    res = await hass.services.async_call(
        DOMAIN, "run_validation", {}, blocking=True, return_response=True
    )
    assert res["status"] == "refused"
    assert res["confirmed_devices"] == [
        {
            "name": "Kettle",
            "validation_class": KETTLE_CLASS,
            "validation_basis": "learning_split",
            "unvalidated": True,
        },
        {
            "name": "Oven",
            "validation_class": KETTLE_CLASS,
            "validation_basis": None,
            "unvalidated": True,
        },
    ]
    stored = hass_storage[VALIDATION_STORE_KEY]["data"]["report"]
    assert stored["confirmed_devices"] == res["confirmed_devices"]


# --- Plan 06-09 (REV-04, D-14/D-32): the sessions list never touches a class ---


async def test_session_validation_lifecycle_and_no_class_effect(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
) -> None:
    """apply_validation stores a session result, keeps it across reports without
    an entry, only replaces it with a newer one and never touches a class (D-14, D-32)."""
    candidate = make_switched_candidate("sig-kettle", "Kettle", rise_w=2000.0)
    entry = await _setup_confirmed(
        hass,
        energy_manager,
        std_prefs,
        hass_storage,
        [candidate],
        report=_ok_report(),
    )
    devices = entry.runtime_data.devices
    dev_uuid = devices.device_for_signature("sig-kettle")
    assert dev_uuid is not None
    power_eid, energy_eid, _v = _device_entity_ids(hass, entry, dev_uuid)
    active_v = devices.record(dev_uuid)["active_v"]

    def _session_report(window_end: str, *, passes: bool = True) -> dict[str, Any]:
        return {
            **_ok_report(),
            "sessions": [
                {
                    "session_id": "sess-kettle",
                    "device_uuid": dev_uuid,
                    "artefact_v": active_v,
                    "passes": passes,
                    "energy_error_pct": 6.0,
                    "window_start": "2026-11-03T00:00:00+00:00",
                    "window_end": window_end,
                    "holdout_start": "2026-11-11T00:00:00+00:00",
                    "holdout_end": window_end,
                }
            ],
        }

    # D-32: the same report's classes are identical with and without sessions.
    devices.apply_validation(_ok_report())
    classes_without = {k: dict(v) for k, v in devices._class_results.items()}
    peers_without = devices.attributes(dev_uuid)["validation_peers"]
    devices.apply_validation(_session_report("2026-11-18T00:00:00+00:00"))
    assert {k: dict(v) for k, v in devices._class_results.items()} == classes_without
    assert devices.attributes(dev_uuid)["validation_peers"] == peers_without

    attrs = devices.attributes(dev_uuid)
    assert attrs["status"] == "validated"
    assert attrs["validation_basis"] == "plug_session"
    assert attrs["unvalidated"] is False
    assert attrs["validation_error_pct"] == 6.0
    assert attrs["validation_session_start"] == "2026-11-03T00:00:00+00:00"
    assert attrs["validation_session_end"] == "2026-11-18T00:00:00+00:00"
    # The two session keys follow the fixed twelve-key contract.
    keys = list(attrs.keys())
    assert keys[: len(TWELVE_KEYS)] == TWELVE_KEYS
    assert keys[len(TWELVE_KEYS) :] == [
        "validation_session_start",
        "validation_session_end",
    ]
    # Both device sensors carry the session period.
    for eid in (power_eid, energy_eid):
        sensor_keys = list(_extra(hass, eid).keys())
        assert sensor_keys[: len(TWELVE_KEYS)] == TWELVE_KEYS
        assert "validation_session_start" in sensor_keys

    # A report with no sessions key (a shadow-only run) keeps the stored result.
    devices.apply_validation(_ok_report())
    assert devices.attributes(dev_uuid)["status"] == "validated"

    # An older session entry never replaces the stored one.
    devices.apply_validation(_session_report("2026-11-01T00:00:00+00:00", passes=False))
    assert devices.attributes(dev_uuid)["status"] == "validated"

    # A refused report without sessions keeps it too (D-14).
    devices.apply_validation({"status": "refused", "reason": "no_shadow_set"})
    assert devices.attributes(dev_uuid)["status"] == "validated"

    # A newer passing session replaces the stored result and re-validates.
    devices.apply_validation(_session_report("2026-11-25T00:00:00+00:00"))
    assert devices.attributes(dev_uuid)["status"] == "validated"
    assert (
        devices.attributes(dev_uuid)["validation_session_end"]
        == "2026-11-25T00:00:00+00:00"
    )

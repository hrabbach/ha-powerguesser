"""HA tests for the D-03 confirm gate and its logged, recorded override.

Plan 08-06 Task 1: a capture preview backed by fewer than ``CONFIRM_MIN_RUNS``
(3) clean captured runs cannot be confirmed by accident. The confirm form shows
a ``confirm_anyway`` checkbox exactly when the gate applies, an informed
override is logged at WARNING and recorded on the review decision, and a preview
with three clean runs (or any learned candidate) confirms exactly as before.
"""

import copy
import logging
from typing import Any

from homeassistant.components.energy.data import EnergyManager
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType
from homeassistant.util import dt as dt_util

from custom_components.powerguesser.const import DOMAIN
from custom_components.powerguesser.engine.intervals import ProfileSnapshot, gate
from tests.conftest import (
    async_fill_synth_buffer,
    async_open_options_step,
    async_set_power,
    async_setup_via_flow,
    make_switched_candidate,
    seed_candidates,
)
from tests.fixtures.learning.synth import SynthSpec

GRID_S = 30


def _pin_grid(entry: Any) -> None:
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


def _iso_utc(ms: int) -> str:
    return dt_util.utc_from_timestamp(ms / 1000.0).isoformat()


def _schema_fields(result: dict[str, Any]) -> list[str]:
    return [str(key) for key in result["data_schema"].schema]


async def _setup(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    *,
    candidates: list[dict[str, Any]] | None = None,
) -> tuple[Any, Any]:
    if candidates:
        seed_candidates(hass_storage, candidates)
    energy_manager.data = copy.deepcopy(std_prefs)
    async_set_power(hass, "sensor.grid_power", 500)
    async_set_power(hass, "sensor.solar_power", 0)
    async_set_power(hass, "sensor.battery_power", 0)
    async_set_power(hass, "sensor.washer_power", 0)
    entry = await async_setup_via_flow(hass)
    _pin_grid(entry)
    await hass.async_block_till_done()
    data = await async_fill_synth_buffer(hass, entry, SynthSpec(days=3))
    return entry, data


async def _capture(
    hass: HomeAssistant,
    *,
    start_ms: int,
    end_ms: int,
    target_type: str = "new_device",
    target: str = "Kettle",
) -> dict[str, Any]:
    return await hass.services.async_call(
        DOMAIN,
        "capture_window",
        {
            "start": _iso_utc(start_ms),
            "end": _iso_utc(end_ms),
            "label": "run",
            "target_type": target_type,
            "target": target,
        },
        blocking=True,
        return_response=True,
    )


async def _open_confirm_form(hass: HomeAssistant, entry: Any, sig: str) -> dict[str, Any]:
    """Open Review -> candidate -> confirm and return the confirm form."""
    res_review = await async_open_options_step(hass, entry, "review")
    assert res_review["type"] is FlowResultType.FORM
    res_cand = await hass.config_entries.options.async_configure(
        res_review["flow_id"], {"candidate": sig}
    )
    assert res_cand["type"] is FlowResultType.MENU
    assert res_cand["step_id"] == "candidate"
    res_form = await hass.config_entries.options.async_configure(
        res_cand["flow_id"], {"next_step_id": "confirm"}
    )
    assert res_form["type"] is FlowResultType.FORM
    assert res_form["step_id"] == "confirm"
    return res_form


async def test_one_run_preview_refused_without_override(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
) -> None:
    """A 1-run preview shows confirm_anyway and refuses to confirm without it."""
    entry, data = await _setup(hass, energy_manager, std_prefs, hass_storage)
    hub = entry.runtime_data
    on_ms, off_ms = data.kettle_runs[-1]

    cap = await _capture(hass, start_ms=on_ms - 45_000, end_ms=off_ms + 50_000)
    sig = cap["target"]["id"]
    assert sig.startswith("cap_")
    assert hub.learning.visible_candidates()[0]["scores"]["n_examples"] == 1

    form = await _open_confirm_form(hass, entry, sig)
    assert "confirm_anyway" in _schema_fields(form)

    res = await hass.config_entries.options.async_configure(
        form["flow_id"], {"name": "Kettle"}
    )
    assert res["type"] is FlowResultType.FORM
    assert res["step_id"] == "confirm"
    assert res["errors"] == {"base": "preview_needs_runs"}
    assert "based on 1 captured run" in res["description_placeholders"]["warning"]
    # The re-rendered form still offers the override.
    assert "confirm_anyway" in _schema_fields(res)

    # Nothing was confirmed: no device subentry, no device link, no override.
    assert entry.get_subentries_of_type("device") == []
    dec = hub.learning.review.decisions.get(sig, {})
    assert dec.get("subentry_id") is None
    assert "confirm_override" not in dec


async def test_override_confirms_logs_and_records(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    caplog: Any,
) -> None:
    """confirm_anyway confirms; a WARNING names the id and count; the override is on the decision."""
    entry, data = await _setup(hass, energy_manager, std_prefs, hass_storage)
    hub = entry.runtime_data
    on_ms, off_ms = data.kettle_runs[-1]

    cap = await _capture(hass, start_ms=on_ms - 45_000, end_ms=off_ms + 50_000)
    sig = cap["target"]["id"]
    form = await _open_confirm_form(hass, entry, sig)

    caplog.set_level(logging.WARNING)
    await hass.config_entries.options.async_configure(
        form["flow_id"], {"name": "Kettle", "confirm_anyway": True}
    )

    subs = entry.get_subentries_of_type("device")
    assert len(subs) == 1
    uuid = subs[0].data["device_uuid"]
    assert hub.devices.record(uuid)["status"] == "confirmed"
    assert hub.learning.visible_candidates() == []

    warnings = [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING and sig in r.getMessage()
    ]
    assert len(warnings) == 1
    message = warnings[0].getMessage()
    assert "1 of 3" in message
    assert "1 captured run" in message

    dec = hub.learning.review.decisions[sig]
    override = dec["confirm_override"]
    assert override["n_runs"] == 1
    assert override["required"] == 3
    assert dt_util.parse_datetime(override["at"]) is not None
    # The record keeps its other keys (name, status).
    assert dec["name"] == "Kettle"
    assert dec["status"] == "named"


async def test_three_runs_confirm_without_field(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
) -> None:
    """A 3-run preview has no confirm_anyway and confirms, trained at the first capture."""
    entry, data = await _setup(hass, energy_manager, std_prefs, hass_storage)
    hub = entry.runtime_data
    on_ms, off_ms = data.kettle_runs[-1]

    first = await _capture(hass, start_ms=on_ms - 45_000, end_ms=off_ms + 50_000)
    sig = first["target"]["id"]
    for _ in range(2):
        joined = await _capture(
            hass,
            start_ms=on_ms - 45_000,
            end_ms=off_ms + 50_000,
            target_type="candidate",
            target="kettle",
        )
        assert joined["target"]["id"] == sig

    preview = next(
        c for c in hub.learning.visible_candidates() if c["signature_id"] == sig
    )
    assert preview["scores"]["n_examples"] == 3

    form = await _open_confirm_form(hass, entry, sig)
    assert "confirm_anyway" not in _schema_fields(form)

    await hass.config_entries.options.async_configure(form["flow_id"], {"name": "Kettle"})

    subs = entry.get_subentries_of_type("device")
    assert len(subs) == 1
    uuid = subs[0].data["device_uuid"]
    rec = hub.devices.record(uuid)
    assert rec["status"] == "confirmed"

    earliest = (
        await hass.services.async_call(
            DOMAIN,
            "capture_list",
            {"capture_id": first["capture_id"]},
            blocking=True,
            return_response=True,
        )
    )["captures"][0]["resolved_at"]
    assert rec["versions"][0]["trained_at"] == earliest


async def test_learned_candidate_confirm_unchanged(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
) -> None:
    """A learned candidate shows no confirm_anyway and confirms with no override record."""
    cand = make_switched_candidate("sig-kettle", "Kettle")
    entry, _data = await _setup(
        hass, energy_manager, std_prefs, hass_storage, candidates=[cand]
    )
    hub = entry.runtime_data

    form = await _open_confirm_form(hass, entry, "sig-kettle")
    assert "confirm_anyway" not in _schema_fields(form)

    await hass.config_entries.options.async_configure(form["flow_id"], {"name": "Kettle"})

    subs = entry.get_subentries_of_type("device")
    assert len(subs) == 1
    uuid = subs[0].data["device_uuid"]
    assert hub.devices.record(uuid)["status"] == "confirmed"
    assert "confirm_override" not in hub.learning.review.decisions.get("sig-kettle", {})

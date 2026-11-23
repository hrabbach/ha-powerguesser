"""Tests for temporary plug sessions (REV-04, Plan 06-06).

A plug session assigns a spare smart plug to a review candidate or a confirmed
device for a period. These tests drive the real options flow and the real hub so
the session record, the live remainder subtraction, the session buffer channel
and the Review marker are asserted end to end.
"""

import copy
from datetime import timedelta
import math
from pathlib import Path
import sqlite3
from typing import Any

from homeassistant.components.energy.data import EnergyManager
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers import entity_registry as er
from homeassistant.util import dt as dt_util
from pytest_homeassistant.common import async_fire_time_changed
from pytest_homeassistant.common import MockConfigEntry

from custom_components.powerguesser.const import (
    BUFFER_DIR,
    BUFFER_FILE,
    DOMAIN,
    SESSIONS_STORE_KEY,
)
from custom_components.powerguesser.engine.intervals import ProfileSnapshot, gate
from custom_components.powerguesser.review_store import resolve_session_target
from tests.conftest import (
    async_confirm_candidate,
    async_fill_synth_buffer,
    async_open_options_step,
    async_read_buffer,
    async_set_power,
    async_setup_via_flow,
    make_switched_candidate,
    seed_candidates,
)
from tests.fixtures.learning.synth import ENTITY_KETTLE_SESSION, SynthSpec

GRID_S = 30
PLUG = "sensor.test_plug_power"


def _aligned_move(freezer: Any) -> None:
    """Move the freezer to the next epoch multiple of the grid."""
    now = dt_util.utcnow().timestamp()
    aligned = math.ceil(now / GRID_S) * GRID_S
    freezer.move_to(dt_util.utc_from_timestamp(aligned))


def _pin_grid(entry: Any) -> None:
    """Publish a 30 s profile snapshot so the grid is pinned at 30."""
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


async def _drive_grid(hass: HomeAssistant, freezer: Any, n: int) -> None:
    """Advance n full grid steps via the 10 s tick path."""
    for _ in range(n * 3):
        freezer.tick(timedelta(seconds=10))
        async_fire_time_changed(hass, dt_util.utcnow())
        await hass.async_block_till_done()


def _stored_sessions(hass_storage: dict[str, Any]) -> list[dict[str, Any]]:
    """Read the sessions Store payload written to hass_storage."""
    return hass_storage.get(SESSIONS_STORE_KEY, {}).get("data", {}).get("sessions", [])


async def _open_step(hass: HomeAssistant, entry: Any, step_id: str) -> dict[str, Any]:
    """Open the options flow and land on a plug-sessions menu."""
    result = await async_open_options_step(hass, entry, "plug_sessions")
    assert result["type"] is FlowResultType.MENU
    assert result["step_id"] == "plug_sessions"
    return await hass.config_entries.options.async_configure(
        result["flow_id"], {"next_step_id": step_id}
    )


async def test_candidate_session_subtracts_plug_live(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
) -> None:
    """A plug assigned to a candidate leaves the remainder and marks the candidate."""
    seed_candidates(
        hass_storage, [make_switched_candidate("sig01", "Kettle", rise_w=2000.0)]
    )
    energy_manager.data = copy.deepcopy(std_prefs)
    async_set_power(hass, "sensor.grid_power", 1500)
    async_set_power(hass, "sensor.solar_power", 0)
    async_set_power(hass, "sensor.battery_power", 0)
    async_set_power(hass, "sensor.washer_power", 0)
    async_set_power(hass, PLUG, 400)
    _aligned_move(freezer)
    entry = await async_setup_via_flow(hass)
    _pin_grid(entry)
    await hass.async_block_till_done()

    assert float(hass.states.get("sensor.powerguesser_remainder_power").state) == 1500.0

    form = await _open_step(hass, entry, "session_start")
    assert form["type"] is FlowResultType.FORM
    assert form["step_id"] == "session_start"
    await hass.config_entries.options.async_configure(
        form["flow_id"], {"plug": PLUG, "target": "candidate_sig01"}
    )
    await hass.async_block_till_done()

    sessions = _stored_sessions(hass_storage)
    assert len(sessions) == 1
    record = sessions[0]
    assert set(record) == {
        "session_id",
        "plug_entity_id",
        "target_kind",
        "target_id",
        "start_ts",
        "end_ts",
        "planned_end_ts",
        "end_reason",
    }
    assert record["plug_entity_id"] == PLUG
    assert record["target_kind"] == "candidate"
    assert record["target_id"] == "sig01"
    assert record["end_ts"] is None
    assert record["end_reason"] is None
    assert record["planned_end_ts"] is None
    assert len(record["session_id"]) == 32
    assert dt_util.parse_datetime(record["start_ts"]) is not None

    # A fresh grid sample settles and publishes: 1500 gross - 400 plug = 1100.
    async_set_power(hass, "sensor.grid_power", 1500)
    await _drive_grid(hass, freezer, 4)
    assert float(
        hass.states.get("sensor.powerguesser_remainder_power").state
    ) == 1100.0

    hub = entry.runtime_data
    kettle = next(
        e for e in hub.learning.review_entries() if e.get("signature_id") == "sig01"
    )
    assert hub.learning.candidate_label(kettle).endswith(" — plug session active")

    rows = await hass.async_add_executor_job(hub.writer.db.channel_rows)
    assert any(
        r[1] == PLUG and r[2] == "session" and r[3] == "session" for r in rows
    )


# ---------------------------------------------------------------------------
# Task 2: validation, end, list and label survival
# ---------------------------------------------------------------------------


async def _setup_session_home(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
    *,
    prefs: dict[str, Any] | None = None,
) -> Any:
    """Seed three switched candidates, the standard powers and set the entry up."""
    seed_candidates(
        hass_storage,
        [
            make_switched_candidate("sig01", "Kettle", rise_w=2000.0),
            make_switched_candidate("sig02", "Mixer", rise_w=1200.0),
            make_switched_candidate("sig03", "Toaster", rise_w=800.0),
        ],
    )
    energy_manager.data = copy.deepcopy(prefs if prefs is not None else std_prefs)
    async_set_power(hass, "sensor.grid_power", 1500)
    async_set_power(hass, "sensor.solar_power", 0)
    async_set_power(hass, "sensor.battery_power", 0)
    async_set_power(hass, "sensor.washer_power", 0)
    async_set_power(hass, PLUG, 400)
    _aligned_move(freezer)
    entry = await async_setup_via_flow(hass)
    _pin_grid(entry)
    await hass.async_block_till_done()
    return entry


async def _open_start(hass: HomeAssistant, entry: Any) -> dict[str, Any]:
    """Open the Start step and return its form (options rendered now)."""
    form = await _open_step(hass, entry, "session_start")
    assert form["type"] is FlowResultType.FORM
    assert form["step_id"] == "session_start"
    return form


async def _submit_start(
    hass: HomeAssistant,
    entry: Any,
    plug: str,
    target: str,
    end_date: str | None = None,
) -> dict[str, Any]:
    """Open the Start step and submit one plug/target(/end date) triple."""
    form = await _open_start(hass, entry)
    user_input: dict[str, Any] = {"plug": plug, "target": target}
    if end_date is not None:
        user_input["end_date"] = end_date
    result = await hass.config_entries.options.async_configure(
        form["flow_id"], user_input
    )
    # The step only schedules the hub rebuild (config_flow.async_step_session_start
    # calls hub.async_request_rebuild); drain it so callers observe the applied
    # session and the tracker count is settled before they assert.
    await hass.async_block_till_done()
    return result


async def _end_session(
    hass: HomeAssistant, entry: Any, session_id: str
) -> dict[str, Any]:
    """Open the End step and end one active session."""
    form = await _open_step(hass, entry, "session_end")
    assert form["type"] is FlowResultType.FORM
    assert form["step_id"] == "session_end"
    result = await hass.config_entries.options.async_configure(
        form["flow_id"], {"session": session_id}
    )
    # Same as _submit_start: the step schedules the hub rebuild; drain it so the
    # session is really gone when the caller asserts.
    await hass.async_block_till_done()
    return result


async def _list_text(hass: HomeAssistant, entry: Any) -> str:
    """Run the List step and return its {sessions} placeholder, unescaped.

    config_flow markdown-escapes the placeholder (the same _escape_markdown the
    Review labels use), so the raw text carries backslashes before '.'/'_'/'-';
    this undoes that so the assertions can name the plug and the target.
    """
    result = await _open_step(hass, entry, "session_list")
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "session_list"
    return _unescape(str(result["description_placeholders"]["sessions"]))


def _unescape(text: str) -> str:
    """Undo backslash markdown escaping."""
    out: list[str] = []
    i = 0
    while i < len(text):
        if text[i] == "\\" and i + 1 < len(text):
            out.append(text[i + 1])
            i += 2
        else:
            out.append(text[i])
            i += 1
    return "".join(out)


async def test_start_rejects_powerguesser_entity(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
) -> None:
    """A PowerGuesser entity cannot be the session plug (plug_self_reference, D-10)."""
    entry = await _setup_session_home(
        hass, energy_manager, std_prefs, hass_storage, freezer
    )
    result = await _submit_start(
        hass, entry, "sensor.powerguesser_remainder_power", "candidate_sig01"
    )
    assert result["type"] is FlowResultType.FORM
    assert "plug_self_reference" in result["errors"].values()
    assert _stored_sessions(hass_storage) == []


async def test_start_rejects_source_and_metered_plug(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
) -> None:
    """A gross source or an Energy device_consumption power entity is refused (D-10)."""
    entry = await _setup_session_home(
        hass, energy_manager, std_prefs, hass_storage, freezer
    )
    for plug in ("sensor.grid_power", "sensor.solar_power", "sensor.washer_power"):
        result = await _submit_start(hass, entry, plug, "candidate_sig01")
        assert result["type"] is FlowResultType.FORM
        assert "plug_is_source" in result["errors"].values(), plug
    assert _stored_sessions(hass_storage) == []


async def test_start_rejects_sibling_power_sensor_of_metered_device(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
) -> None:
    """A sibling power sensor of a device_consumption entity is refused (D-10).

    The device has two power sensors, so the source model resolves no metered
    power entity for it and the sibling set is the only thing naming them.
    """
    mock_entry = MockConfigEntry(domain="test_domain")
    mock_entry.add_to_hass(hass)
    dev_reg = dr.async_get(hass)
    ent_reg = er.async_get(hass)
    device = dev_reg.async_get_or_create(
        config_entry_id=mock_entry.entry_id,
        identifiers={("test_domain", "ambig_two_sensors")},
        name="Ambiguous Plug",
    )
    ent_reg.async_get_or_create(
        domain="sensor",
        platform="test_domain",
        unique_id="ambig_energy_uid",
        suggested_object_id="ambig_energy",
        device_id=device.id,
        unit_of_measurement="kWh",
    )
    for name in ("ambig_power1", "ambig_power2"):
        ent_reg.async_get_or_create(
            domain="sensor",
            platform="test_domain",
            unique_id=f"{name}_uid",
            suggested_object_id=name,
            device_id=device.id,
            original_device_class="power",
            unit_of_measurement="W",
        )

    prefs = copy.deepcopy(std_prefs)
    prefs["device_consumption"] = [
        {"stat_consumption": "sensor.ambig_energy", "name": "Ambiguous Plug"}
    ]
    async_set_power(hass, "sensor.ambig_power1", 100)
    async_set_power(hass, "sensor.ambig_power2", 200)

    entry = await _setup_session_home(
        hass, energy_manager, std_prefs, hass_storage, freezer, prefs=prefs
    )
    result = await _submit_start(hass, entry, "sensor.ambig_power1", "candidate_sig01")
    assert result["type"] is FlowResultType.FORM
    assert "plug_is_source" in result["errors"].values()
    assert _stored_sessions(hass_storage) == []


async def test_start_rejects_plug_of_active_session(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
) -> None:
    """The plug of an active session cannot start a second session (D-10)."""
    entry = await _setup_session_home(
        hass, energy_manager, std_prefs, hass_storage, freezer
    )
    first = await _submit_start(hass, entry, PLUG, "candidate_sig01")
    assert first["type"] is FlowResultType.MENU
    assert len(_stored_sessions(hass_storage)) == 1

    second = await _submit_start(hass, entry, PLUG, "candidate_sig02")
    assert second["type"] is FlowResultType.FORM
    assert "plug_in_session" in second["errors"].values()
    assert len(_stored_sessions(hass_storage)) == 1


async def test_start_rejects_past_end_date(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
) -> None:
    """An end date that is not in the future is refused (D-10)."""
    entry = await _setup_session_home(
        hass, energy_manager, std_prefs, hass_storage, freezer
    )
    past = (dt_util.utcnow() - timedelta(days=1)).isoformat()
    result = await _submit_start(hass, entry, PLUG, "candidate_sig01", end_date=past)
    assert result["type"] is FlowResultType.FORM
    assert "end_date_past" in result["errors"].values()
    assert _stored_sessions(hass_storage) == []

    future = (dt_util.utcnow() + timedelta(days=1)).isoformat()
    ok = await _submit_start(hass, entry, PLUG, "candidate_sig01", end_date=future)
    assert ok["type"] is FlowResultType.MENU
    assert len(_stored_sessions(hass_storage)) == 1
    assert _stored_sessions(hass_storage)[0]["planned_end_ts"] is not None


async def test_start_rejects_gone_target(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
) -> None:
    """A dismissed candidate no longer resolves: target_gone, nothing stored (D-10)."""
    entry = await _setup_session_home(
        hass, energy_manager, std_prefs, hass_storage, freezer
    )
    hub = entry.runtime_data
    # The form's target list was rendered while the candidate was still there;
    # dismissing it before submitting is the race target_gone covers (D-10).
    form = await _open_start(hass, entry)
    await hub.learning.async_record_decision("dismiss", "sig01")
    await hass.async_block_till_done()
    result = await hass.config_entries.options.async_configure(
        form["flow_id"], {"plug": PLUG, "target": "candidate_sig01"}
    )
    assert result["type"] is FlowResultType.FORM
    assert "target_gone" in result["errors"].values()
    assert _stored_sessions(hass_storage) == []

    # The resolver also refuses the stored record shape directly.
    record = {"target_kind": "candidate", "target_id": "sig01"}
    assert (
        resolve_session_target(
            record, hub.learning.review.decisions, hub.devices.record
        )
        is None
    )


async def test_end_step_stops_the_subtraction(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
) -> None:
    """The End step closes the session with reason user and stops subtracting."""
    entry = await _setup_session_home(
        hass, energy_manager, std_prefs, hass_storage, freezer
    )
    await _submit_start(hass, entry, PLUG, "candidate_sig01")
    async_set_power(hass, "sensor.grid_power", 1500)
    await _drive_grid(hass, freezer, 4)
    assert float(
        hass.states.get("sensor.powerguesser_remainder_power").state
    ) == 1100.0

    session_id = _stored_sessions(hass_storage)[0]["session_id"]
    result = await _end_session(hass, entry, session_id)
    assert result["type"] is FlowResultType.MENU
    record = _stored_sessions(hass_storage)[0]
    assert record["end_ts"] is not None
    assert record["end_reason"] == "user"

    async_set_power(hass, "sensor.grid_power", 1500)
    await _drive_grid(hass, freezer, 4)
    assert float(
        hass.states.get("sensor.powerguesser_remainder_power").state
    ) == 1500.0


async def test_planned_end_date_ends_the_session_on_a_tick(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
) -> None:
    """A planned end date passing ends the session with reason end_date on a tick."""
    entry = await _setup_session_home(
        hass, energy_manager, std_prefs, hass_storage, freezer
    )
    planned = dt_util.utcnow() + timedelta(minutes=30)
    await _submit_start(
        hass, entry, PLUG, "candidate_sig01", end_date=planned.isoformat()
    )
    assert _stored_sessions(hass_storage)[0]["end_reason"] is None

    await _drive_grid(hass, freezer, 1)
    assert _stored_sessions(hass_storage)[0]["end_ts"] is None

    freezer.tick(timedelta(minutes=31))
    async_fire_time_changed(hass, dt_util.utcnow())
    await hass.async_block_till_done()

    record = _stored_sessions(hass_storage)[0]
    assert record["end_ts"] is not None
    assert record["end_reason"] == "end_date"

    async_set_power(hass, "sensor.grid_power", 1500)
    await _drive_grid(hass, freezer, 4)
    assert float(
        hass.states.get("sensor.powerguesser_remainder_power").state
    ) == 1500.0


async def test_list_step_empty_state_and_sessions(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
) -> None:
    """The List step shows an empty state, then one line per session (D-15)."""
    entry = await _setup_session_home(
        hass, energy_manager, std_prefs, hass_storage, freezer
    )
    assert "no plug sessions yet" in await _list_text(hass, entry)

    await _submit_start(hass, entry, PLUG, "candidate_sig01")
    text = await _list_text(hass, entry)
    assert PLUG in text
    assert "Kettle" in text
    assert "active" in text


async def test_second_session_leaves_the_first_byte_identical(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
) -> None:
    """Session A ended, session B started: A stays byte-identical and named (SC3)."""
    entry = await _setup_session_home(
        hass, energy_manager, std_prefs, hass_storage, freezer
    )
    await _submit_start(hass, entry, PLUG, "candidate_sig01")
    first_id = _stored_sessions(hass_storage)[0]["session_id"]
    await _end_session(hass, entry, first_id)
    before = copy.deepcopy(_stored_sessions(hass_storage)[0])

    await _submit_start(hass, entry, PLUG, "candidate_sig02")

    sessions = _stored_sessions(hass_storage)
    assert len(sessions) == 2
    assert sessions[0] == before
    assert sessions[0]["target_id"] == "sig01"
    assert sessions[1]["target_id"] == "sig02"

    first_line = (await _list_text(hass, entry)).split("\n")[0]
    assert "Kettle" in first_line
    assert "user" in first_line


async def test_session_target_label_follows_rename_merge_confirm_and_retire(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
) -> None:
    """Earlier labels follow the target's rename, merge, confirm and retire (SC3)."""
    entry = await _setup_session_home(
        hass, energy_manager, std_prefs, hass_storage, freezer
    )
    hub = entry.runtime_data
    await _submit_start(hass, entry, PLUG, "candidate_sig01")
    record_before = copy.deepcopy(_stored_sessions(hass_storage)[0])

    assert "Kettle" in (await _list_text(hass, entry)).split("\n")[0]

    await hub.learning.async_record_decision("rename", "sig01", name="Renamed load")
    assert "Renamed load" in (await _list_text(hass, entry)).split("\n")[0]
    assert _stored_sessions(hass_storage)[0] == record_before

    await hub.learning.async_record_decision("merge", "sig02", sources=["sig01"])
    assert "Mixer" in (await _list_text(hass, entry)).split("\n")[0]
    assert _stored_sessions(hass_storage)[0] == record_before

    await async_confirm_candidate(hass, entry, "sig02", "Mixer device")
    assert "Mixer device" in (await _list_text(hass, entry)).split("\n")[0]
    assert _stored_sessions(hass_storage)[0] == record_before

    await async_confirm_candidate(hass, entry, "sig03", "Survivor")
    uuid_sig02 = hub.devices.device_for_signature("sig02")
    uuid_sig03 = hub.devices.device_for_signature("sig03")
    await hub.devices.async_retire(uuid_sig02, into=uuid_sig03)
    await hass.async_block_till_done()

    assert "Survivor" in (await _list_text(hass, entry)).split("\n")[0]
    assert _stored_sessions(hass_storage)[0] == record_before


# ---------------------------------------------------------------------------
# Task 3: Energy dashboard edits, plug outages, restarts and the channel
# ---------------------------------------------------------------------------


async def _channel_row(
    hass: HomeAssistant, hub: Any, entity_id: str, role: str
) -> tuple[int, str, str, str] | None:
    """Return the buffer channel row for (entity_id, role), if any."""
    rows = await hass.async_add_executor_job(hub.writer.db.channel_rows)
    for row in rows:
        if row[1] == entity_id and row[2] == role:
            return row
    return None


async def test_available_plug_added_to_energy_dashboard_ends_the_session(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
) -> None:
    """A plug that becomes a metered device ends the session (plug_metered, D-10).

    It is then subtracted once, as metered, never twice (T-06-15).
    """
    entry = await _setup_session_home(
        hass, energy_manager, std_prefs, hass_storage, freezer
    )
    await _submit_start(hass, entry, PLUG, "candidate_sig01")
    async_set_power(hass, "sensor.grid_power", 1500)
    await _drive_grid(hass, freezer, 4)
    assert float(
        hass.states.get("sensor.powerguesser_remainder_power").state
    ) == 1100.0

    prefs = copy.deepcopy(std_prefs)
    prefs["device_consumption"].append(
        {"stat_consumption": "sensor.test_plug_energy", "stat_rate": PLUG}
    )
    await energy_manager.async_update(prefs)
    await hass.async_block_till_done()

    record = _stored_sessions(hass_storage)[0]
    assert record["end_ts"] is not None
    assert record["end_reason"] == "plug_metered"

    async_set_power(hass, "sensor.grid_power", 1500)
    await _drive_grid(hass, freezer, 4)
    remainder = hass.states.get("sensor.powerguesser_remainder_power")
    assert float(remainder.state) == 1100.0
    assert remainder.attributes["metered_w"] == 400.0


async def test_unavailable_plug_is_reported_not_subtracted(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
) -> None:
    """An unavailable session plug is metered_unavailable, never counted as 0 W."""
    entry = await _setup_session_home(
        hass, energy_manager, std_prefs, hass_storage, freezer
    )
    await _submit_start(hass, entry, PLUG, "candidate_sig01")
    hub = entry.runtime_data
    assert hub._session_values.get(PLUG) is not None

    async_set_power(hass, PLUG, "unavailable")
    async_set_power(hass, "sensor.grid_power", 1500)
    await _drive_grid(hass, freezer, 4)

    remainder = hass.states.get("sensor.powerguesser_remainder_power")
    assert float(remainder.state) == 1500.0
    assert remainder.attributes["metered_unavailable"] == [PLUG]
    assert hub._session_values[PLUG] is None


async def test_device_target_session_is_recorded_but_not_subtracted_live(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
) -> None:
    """A device-target session records the plug on its channel, not the remainder (D-25)."""
    entry = await _setup_session_home(
        hass, energy_manager, std_prefs, hass_storage, freezer
    )
    hub = entry.runtime_data
    await async_confirm_candidate(hass, entry, "sig01", "Kettle")
    device_uuid = hub.devices.device_for_signature("sig01")
    assert device_uuid is not None

    async_set_power(hass, "sensor.grid_power", 1500)
    await _drive_grid(hass, freezer, 4)
    before = float(hass.states.get("sensor.powerguesser_remainder_power").state)

    form = await _open_start(hass, entry)
    await hass.config_entries.options.async_configure(
        form["flow_id"], {"plug": PLUG, "target": f"device_{device_uuid}"}
    )
    await hass.async_block_till_done()

    record = _stored_sessions(hass_storage)[0]
    assert record["target_kind"] == "device"
    assert record["target_id"] == device_uuid
    assert hub._session_metered_plugs == set()
    assert hub._session_values.get(PLUG) == 400.0

    async_set_power(hass, "sensor.grid_power", 1500)
    await _drive_grid(hass, freezer, 4)
    after = float(hass.states.get("sensor.powerguesser_remainder_power").state)
    assert after == before

    await entry.runtime_data.writer.async_flush()
    row = await _channel_row(hass, hub, PLUG, "session")
    assert row is not None and row[3] == "session"


async def test_active_session_survives_a_reload(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
) -> None:
    """After a reload the hub re-tracks the session plug and subtracts it again."""
    entry = await _setup_session_home(
        hass, energy_manager, std_prefs, hass_storage, freezer
    )
    await _submit_start(hass, entry, PLUG, "candidate_sig01")
    async_set_power(hass, "sensor.grid_power", 1500)
    await _drive_grid(hass, freezer, 4)
    assert float(
        hass.states.get("sensor.powerguesser_remainder_power").state
    ) == 1100.0

    await hass.config_entries.async_reload(entry.entry_id)
    await hass.async_block_till_done()
    reloaded = hass.config_entries.async_get_entry(entry.entry_id)
    assert reloaded is not None
    hub = reloaded.runtime_data
    assert hub._session_metered_plugs == {PLUG}

    async_set_power(hass, "sensor.grid_power", 1500)
    await _drive_grid(hass, freezer, 4)
    assert float(
        hass.states.get("sensor.powerguesser_remainder_power").state
    ) == 1100.0
    assert hub._session_values.get(PLUG) == 400.0


async def test_session_channel_survives_session_end(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
) -> None:
    """An ended session keeps its channel row and samples but leaves held_channels."""
    entry = await _setup_session_home(
        hass, energy_manager, std_prefs, hass_storage, freezer
    )
    hub = entry.runtime_data
    await _submit_start(hass, entry, PLUG, "candidate_sig01")
    async_set_power(hass, "sensor.grid_power", 1500)
    await _drive_grid(hass, freezer, 4)
    row = await _channel_row(hass, hub, PLUG, "session")
    assert row is not None
    session_channel_id = row[0]
    assert session_channel_id in hub.writer.held_channels

    await entry.runtime_data.writer.async_flush()
    session_id = _stored_sessions(hass_storage)[0]["session_id"]
    await _end_session(hass, entry, session_id)
    await hass.async_block_till_done()
    await entry.runtime_data.writer.async_flush()
    await hass.async_block_till_done()

    still_there = await _channel_row(hass, entry.runtime_data, PLUG, "session")
    assert still_there == row
    assert session_channel_id not in entry.runtime_data.writer.held_channels

    data, _layout = await async_read_buffer(hass)
    samples = data[(PLUG, "session")]
    assert samples, "the session channel kept no samples"
    assert samples[-1][1] is None
    assert any(w == 400 for _ts, w in samples if w is not None)


async def test_session_listeners_do_not_accumulate(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
) -> None:
    """Rebuilds do not pile up trackers, and an ended session's plug is dropped."""
    entry = await _setup_session_home(
        hass, energy_manager, std_prefs, hass_storage, freezer
    )
    hub = entry.runtime_data
    await _submit_start(hass, entry, PLUG, "candidate_sig01")

    await hub.async_rebuild()
    baseline = len(hub._unsub_trackers)
    assert baseline > 0
    for _ in range(3):
        await hub.async_rebuild()
        assert len(hub._unsub_trackers) == baseline

    session_id = _stored_sessions(hass_storage)[0]["session_id"]
    await _end_session(hass, entry, session_id)
    await hass.async_block_till_done()

    assert hub._session_values == {}
    async_set_power(hass, PLUG, 999)
    await hass.async_block_till_done()
    assert PLUG not in hub._session_values
    assert len(hub._unsub_trackers) == baseline


# ---------------------------------------------------------------------------
# Task 1 (Plan 06-09): a device-target session validates the confirmed device
# ---------------------------------------------------------------------------


def _device_power_entity(hass: HomeAssistant, entry: Any, dev_uuid: str) -> str:
    """Return the device's power sensor entity_id from the entity registry."""
    reg = er.async_get(hass)
    for ent in er.async_entries_for_config_entry(reg, entry.entry_id):
        if ent.unique_id == f"{entry.entry_id}_{dev_uuid}_power":
            return ent.entity_id
    raise AssertionError(f"no power sensor for device {dev_uuid}")


async def _seed_device_session(
    hass: HomeAssistant,
    entry: Any,
    *,
    device_uuid: str,
    window_days: int,
) -> int:
    """Seed an active device-target session over the last window_days and return its plug channel."""
    hub = entry.runtime_data
    review = hub.learning.review
    now = dt_util.utcnow()
    review._sessions.append(
        {
            "session_id": "sess-kettle",
            "plug_entity_id": ENTITY_KETTLE_SESSION,
            "target_kind": "device",
            "target_id": device_uuid,
            "start_ts": (now - timedelta(days=window_days)).isoformat(),
            "end_ts": None,
            "planned_end_ts": None,
            "end_reason": None,
        }
    )
    await review._sessions_store.async_save({"sessions": review._sessions})
    # The rebuild registers the session channel (kind session) for the plug.
    await hub.async_rebuild()
    await hass.async_block_till_done()
    return 0


async def _zero_session_baseline(hass: HomeAssistant, channel: int, ts_ms: int) -> None:
    """Set the session channel's samples before ts_ms to 0 W (weak baseline fixture)."""
    buf_path = hass.config.path(BUFFER_DIR, BUFFER_FILE)

    def _run() -> None:
        conn = sqlite3.connect(buf_path, timeout=30)
        try:
            conn.execute(
                "UPDATE samples SET w = 0 WHERE ch = ? AND ts < ? AND w IS NOT NULL",
                (channel, ts_ms),
            )
            conn.commit()
        finally:
            conn.close()

    await hass.async_add_executor_job(_run)


async def test_device_session_validates_device(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
    inprocess_worker: None,
) -> None:
    """A device-target session that clears the P5 bar validates the device (REV-04, D-13).

    No shadow set is seeded, so this exercises the no_shadow_set-plus-sessions
    branch: the report stays refused/no_shadow_set yet carries the session
    verdict, and the device reads status validated via plug_session.
    """
    seed_candidates(
        hass_storage, [make_switched_candidate("sig01", "Kettle", rise_w=2000.0)]
    )
    energy_manager.data = copy.deepcopy(std_prefs)
    async_set_power(hass, "sensor.grid_power", 1500)
    async_set_power(hass, "sensor.solar_power", 0)
    async_set_power(hass, "sensor.battery_power", 0)
    async_set_power(hass, "sensor.washer_power", 0)
    _aligned_move(freezer)
    entry = await async_setup_via_flow(hass)
    _pin_grid(entry)
    await hass.async_block_till_done()
    hub = entry.runtime_data

    await async_confirm_candidate(hass, entry, "sig01", "Kettle")
    device_uuid = hub.devices.device_for_signature("sig01")
    assert device_uuid is not None

    await _seed_device_session(hass, entry, device_uuid=device_uuid, window_days=15)
    data = await async_fill_synth_buffer(
        hass, entry, SynthSpec(days=20, kettle_session=True)
    )
    session_ch = data.channel_ids[ENTITY_KETTLE_SESSION]
    await hub.writer.async_flush()

    # The plug's earlier-window readings are 0 W (the device was idle before the
    # holdout), so the constant-baseline is deliberately weak enough for the
    # D-07 energy beat (an honest v0.x limitation, see the engine test).
    today_midnight = dt_util.start_of_local_day(dt_util.utcnow())
    holdout_start_ms = int(
        round((today_midnight - timedelta(days=7)).timestamp() * 1000)
    )
    await _zero_session_baseline(hass, session_ch, holdout_start_ms)

    res = await hass.services.async_call(
        DOMAIN, "run_validation", {}, blocking=True, return_response=True
    )
    assert res["status"] == "refused"
    assert res["reason"] == "no_shadow_set"
    sessions = res["sessions"]
    assert len(sessions) == 1
    record = sessions[0]
    assert record["passes"] is True
    assert record["device_uuid"] == device_uuid
    assert record["name"] == "Kettle"
    assert record["window_start"] and record["window_end"]
    assert record["holdout_start"] and record["holdout_end"]

    power_eid = _device_power_entity(hass, entry, device_uuid)
    attrs = dict(hass.states.get(power_eid).attributes)
    assert attrs["status"] == "validated"
    assert attrs["validation_basis"] == "plug_session"
    assert attrs["unvalidated"] is False
    assert attrs["validation_error_pct"] == record["energy_error_pct"]
    assert attrs["validation_session_start"] == record["window_start"]
    assert attrs["validation_session_end"] == record["window_end"]


async def test_short_device_session_stays_provisional(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
    inprocess_worker: None,
) -> None:
    """A 5-day session has no baseline/holdout day, so the device stays provisional."""
    seed_candidates(
        hass_storage, [make_switched_candidate("sig01", "Kettle", rise_w=2000.0)]
    )
    energy_manager.data = copy.deepcopy(std_prefs)
    async_set_power(hass, "sensor.grid_power", 1500)
    async_set_power(hass, "sensor.solar_power", 0)
    async_set_power(hass, "sensor.battery_power", 0)
    async_set_power(hass, "sensor.washer_power", 0)
    _aligned_move(freezer)
    entry = await async_setup_via_flow(hass)
    _pin_grid(entry)
    await hass.async_block_till_done()
    hub = entry.runtime_data

    await async_confirm_candidate(hass, entry, "sig01", "Kettle")
    device_uuid = hub.devices.device_for_signature("sig01")
    assert device_uuid is not None

    await _seed_device_session(hass, entry, device_uuid=device_uuid, window_days=5)
    await async_fill_synth_buffer(hass, entry, SynthSpec(days=20, kettle_session=True))
    await hub.writer.async_flush()

    res = await hass.services.async_call(
        DOMAIN, "run_validation", {}, blocking=True, return_response=True
    )
    assert res["status"] == "refused"
    assert res["reason"] == "no_shadow_set"
    sessions = res["sessions"]
    assert len(sessions) == 1
    assert sessions[0]["passes"] is False
    assert sessions[0]["status"] == "refused"

    power_eid = _device_power_entity(hass, entry, device_uuid)
    attrs = dict(hass.states.get(power_eid).attributes)
    assert attrs["status"] == "provisional"
    assert attrs["unvalidated"] is True
    # The session period is still stored and visible.
    assert attrs["validation_session_start"] == sessions[0]["window_start"]
    assert attrs["validation_session_end"] == sessions[0]["window_end"]


# ---------------------------------------------------------------------------
# Task 2 (Plan 06-09): session windows shape learning
# ---------------------------------------------------------------------------


async def _seed_candidate_session(
    hass: HomeAssistant,
    entry: Any,
    *,
    plug: str,
    sig_id: str,
    window_days: int,
) -> str:
    """Seed an active candidate-target session over the last window_days."""
    hub = entry.runtime_data
    review = hub.learning.review
    now = dt_util.utcnow()
    session_id = f"sess-{sig_id}"
    review._sessions.append(
        {
            "session_id": session_id,
            "plug_entity_id": plug,
            "target_kind": "candidate",
            "target_id": sig_id,
            "start_ts": (now - timedelta(days=window_days)).isoformat(),
            "end_ts": None,
            "planned_end_ts": None,
            "end_reason": None,
        }
    )
    await review._sessions_store.async_save({"sessions": review._sessions})
    await hub.async_rebuild()
    await hass.async_block_till_done()
    return session_id


async def test_candidate_session_is_a_labelled_line_named_from_the_record(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
    tmp_path: Path,
) -> None:
    """A candidate session window becomes a plug line named after the record (D-12, SC3)."""
    seed_candidates(
        hass_storage,
        [
            make_switched_candidate("sig01", "Kettle", rise_w=2000.0),
            make_switched_candidate("sig02", "Mixer", rise_w=1200.0),
        ],
    )
    energy_manager.data = copy.deepcopy(std_prefs)
    async_set_power(hass, "sensor.grid_power", 1500)
    async_set_power(hass, "sensor.solar_power", 0)
    async_set_power(hass, "sensor.battery_power", 0)
    async_set_power(hass, "sensor.washer_power", 0)
    _aligned_move(freezer)
    entry = await async_setup_via_flow(hass)
    _pin_grid(entry)
    await hass.async_block_till_done()
    hub = entry.runtime_data

    session_a = await _seed_candidate_session(
        hass, entry, plug=ENTITY_KETTLE_SESSION, sig_id="sig01", window_days=15
    )
    data = await async_fill_synth_buffer(
        hass, entry, SynthSpec(days=20, kettle_session=True)
    )
    await hub.writer.async_flush()
    session_ch = data.channel_ids[ENTITY_KETTLE_SESSION]
    stage_dir = tmp_path / "learning_job"

    def _build() -> tuple[dict[str, Any], dict[str, Any], int]:
        return hub.learning._build_extract_job_data(stage_dir)

    job, cache_key, _end = await hass.async_add_executor_job(_build)
    windows = job["session_windows"]
    assert len(windows) == 1
    assert windows[0]["target_kind"] == "candidate"
    assert windows[0]["channel"] == session_ch
    assert windows[0]["key"] >= 1_000_000
    assert "Kettle" in windows[0]["label"]
    # The prepare cache key carries the same windows.
    assert cache_key["session_windows"] == windows

    # The extract stage turns the window into a plug line keyed by its window.
    def _extract() -> tuple[dict[str, Any], dict[str, Any]]:
        from custom_components.powerguesser.engine.learning.worker import extract

        res = extract(dict(job))
        return res["digest"], res["meta"]

    digest, meta = await hass.async_add_executor_job(_extract)
    window_key = windows[0]["key"]
    assert window_key in meta["plugs"].values()
    assert digest["plugs"], "the session window produced no reference line"
    channel_rows = await hass.async_add_executor_job(hub.writer.db.channel_rows)
    names = hub.learning._plug_names(meta, channel_rows)
    assert any("Kettle" in value for value in names.values())

    # After A ends and B (same plug, target Y) starts, A keeps X's label.
    await hub.learning.review.async_end_session(session_a, "user")
    await hass.async_block_till_done()
    await _seed_candidate_session(
        hass, entry, plug=ENTITY_KETTLE_SESSION, sig_id="sig02", window_days=5
    )
    job2, _ck2, _e2 = await hass.async_add_executor_job(_build)
    by_key = {window["key"]: window for window in job2["session_windows"]}
    assert window_key in by_key
    assert "Kettle" in by_key[window_key]["label"]
    assert any("Mixer" in window["label"] for window in job2["session_windows"])


async def test_candidate_session_produces_no_device_session_entry(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
    inprocess_worker: None,
) -> None:
    """A candidate-target session never produces a report sessions entry (D-25, D-32)."""
    seed_candidates(
        hass_storage, [make_switched_candidate("sig01", "Kettle", rise_w=2000.0)]
    )
    energy_manager.data = copy.deepcopy(std_prefs)
    async_set_power(hass, "sensor.grid_power", 1500)
    async_set_power(hass, "sensor.solar_power", 0)
    async_set_power(hass, "sensor.battery_power", 0)
    async_set_power(hass, "sensor.washer_power", 0)
    _aligned_move(freezer)
    entry = await async_setup_via_flow(hass)
    _pin_grid(entry)
    await hass.async_block_till_done()
    hub = entry.runtime_data

    await _seed_candidate_session(
        hass, entry, plug=ENTITY_KETTLE_SESSION, sig_id="sig01", window_days=15
    )
    await async_fill_synth_buffer(hass, entry, SynthSpec(days=20, kettle_session=True))
    await hub.writer.async_flush()

    res = await hass.services.async_call(
        DOMAIN, "run_validation", {}, blocking=True, return_response=True
    )
    assert res["status"] == "refused"
    assert res["reason"] == "no_shadow_set"
    # No device-target session: no sessions key and no confirmed device touched.
    assert "sessions" not in res
    assert res["confirmed_devices"] == []

"""Tests for registry-backed metered devices, sibling resolution, and hierarchy."""

import copy
from datetime import timedelta
import math
import re
from typing import Any
import pytest

from homeassistant.components.energy.data import EnergyManager
from homeassistant.core import HomeAssistant
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers import issue_registry as ir
from homeassistant.util import dt as dt_util
from pytest_homeassistant.common import (
    MockConfigEntry,
    async_fire_time_changed,
)

from custom_components.powerguesser.const import DOMAIN
from custom_components.powerguesser.engine.intervals import ProfileSnapshot, gate
from tests.conftest import (
    async_confirm_candidate,
    async_open_options_step,
    async_set_power,
    async_setup_via_flow,
    make_switched_candidate,
    seed_candidates,
)

GRID_S = 30


def _aligned_move(freezer: Any) -> None:
    """Move the freezer to the next epoch multiple of the grid."""
    now = dt_util.utcnow().timestamp()
    aligned = math.ceil(now / GRID_S) * GRID_S
    freezer.move_to(dt_util.utc_from_timestamp(aligned))


def _pin_grid(entry: Any) -> None:
    """Publish a 30 s profile snapshot so the D-26 grid is pinned at 30."""
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


@pytest.fixture
def base_grid_prefs() -> dict[str, Any]:
    return {
        "energy_sources": [
            {
                "type": "grid",
                "stat_energy_from": "sensor.grid_import_energy",
                "stat_rate": "sensor.grid_power",
            }
        ],
        "device_consumption": [],
    }


async def test_ac_parent_powered_with_two_powered_children(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    base_grid_prefs: dict[str, Any],
) -> None:
    """1. AC parent (powered) with two powered AC children -> only the parent subtracted."""
    async_set_power(hass, "sensor.grid_power", 2000)
    async_set_power(hass, "sensor.ac_parent_power", 800)
    async_set_power(hass, "sensor.ac_child1_power", 300)
    async_set_power(hass, "sensor.ac_child2_power", 400)

    prefs = dict(base_grid_prefs)
    prefs["device_consumption"] = [
        {
            "stat_consumption": "sensor.ac_parent_energy",
            "stat_rate": "sensor.ac_parent_power",
            "name": "AC Parent",
        },
        {
            "stat_consumption": "sensor.ac_child1_energy",
            "stat_rate": "sensor.ac_child1_power",
            "included_in_stat": "sensor.ac_parent_energy",
            "name": "AC Child 1",
        },
        {
            "stat_consumption": "sensor.ac_child2_energy",
            "stat_rate": "sensor.ac_child2_power",
            "included_in_stat": "sensor.ac_parent_energy",
            "name": "AC Child 2",
        },
    ]
    energy_manager.data = prefs

    entry = MockConfigEntry(domain=DOMAIN, data={})
    entry.add_to_hass(hass)
    await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    # Gross = 2000, only parent (800) subtracted -> remainder = 1200
    assert float(hass.states.get("sensor.powerguesser_gross_power").state) == 2000.0
    remainder = hass.states.get("sensor.powerguesser_remainder_power")
    assert float(remainder.state) == 1200.0
    assert remainder.attributes["metered_w"] == 800.0


async def test_ac_unpowered_parent_with_powered_children(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    base_grid_prefs: dict[str, Any],
) -> None:
    """2. Unpowered parent with powered children -> children subtracted, no warning for parent."""
    async_set_power(hass, "sensor.grid_power", 2000)
    async_set_power(hass, "sensor.ac_child1_power", 300)
    async_set_power(hass, "sensor.ac_child2_power", 400)

    prefs = dict(base_grid_prefs)
    prefs["device_consumption"] = [
        {
            "stat_consumption": "sensor.ac_parent_energy",
            "name": "AC Parent",
        },
        {
            "stat_consumption": "sensor.ac_child1_energy",
            "stat_rate": "sensor.ac_child1_power",
            "included_in_stat": "sensor.ac_parent_energy",
            "name": "AC Child 1",
        },
        {
            "stat_consumption": "sensor.ac_child2_energy",
            "stat_rate": "sensor.ac_child2_power",
            "included_in_stat": "sensor.ac_parent_energy",
            "name": "AC Child 2",
        },
    ]
    energy_manager.data = prefs

    entry = MockConfigEntry(domain=DOMAIN, data={})
    entry.add_to_hass(hass)
    await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    # Children subtracted (300 + 400 = 700) -> remainder = 1300
    remainder = hass.states.get("sensor.powerguesser_remainder_power")
    assert float(remainder.state) == 1300.0
    assert remainder.attributes["metered_w"] == 700.0

    issue_reg = ir.async_get(hass)
    # Parent has powered descendants, so NO warning for parent
    assert issue_reg.async_get_issue(DOMAIN, "metered_no_power_sensor_ac_parent_energy") is None


async def test_plug_without_stat_rate_sibling_resolution(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    base_grid_prefs: dict[str, Any],
) -> None:
    """3. Plug without stat_rate whose device has exactly one power sensor -> that sensor subtracted."""
    mock_entry = MockConfigEntry(domain="test_domain")
    mock_entry.add_to_hass(hass)
    dev_reg = dr.async_get(hass)
    ent_reg = er.async_get(hass)

    device = dev_reg.async_get_or_create(
        config_entry_id=mock_entry.entry_id,
        identifiers={("test_domain", "plug_1")},
        name="Plug 1",
    )
    ent_reg.async_get_or_create(
        domain="sensor",
        platform="test_domain",
        unique_id="plug_1_energy_uid",
        suggested_object_id="plug_1_energy",
        device_id=device.id,
        unit_of_measurement="kWh",
    )
    ent_reg.async_get_or_create(
        domain="sensor",
        platform="test_domain",
        unique_id="plug_1_power_uid",
        suggested_object_id="plug_1_power",
        device_id=device.id,
        original_device_class="power",
        unit_of_measurement="W",
    )

    async_set_power(hass, "sensor.grid_power", 1000)
    async_set_power(hass, "sensor.plug_1_power", 150)

    prefs = dict(base_grid_prefs)
    prefs["device_consumption"] = [
        {
            "stat_consumption": "sensor.plug_1_energy",
            "name": "Plug 1",
        }
    ]
    energy_manager.data = prefs

    entry = MockConfigEntry(domain=DOMAIN, data={})
    entry.add_to_hass(hass)
    await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    remainder = hass.states.get("sensor.powerguesser_remainder_power")
    # Subtracted 150 W automatically -> 850 W
    assert float(remainder.state) == 850.0
    assert remainder.attributes["metered_w"] == 150.0


async def test_two_candidate_sensors_not_subtracted(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    base_grid_prefs: dict[str, Any],
) -> None:
    """4. Device with two candidate power sensors -> not subtracted + metered_no_power issue."""
    mock_entry = MockConfigEntry(domain="test_domain")
    mock_entry.add_to_hass(hass)
    dev_reg = dr.async_get(hass)
    ent_reg = er.async_get(hass)

    device = dev_reg.async_get_or_create(
        config_entry_id=mock_entry.entry_id,
        identifiers={("test_domain", "plug_ambiguous")},
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
    ent_reg.async_get_or_create(
        domain="sensor",
        platform="test_domain",
        unique_id="ambig_power1_uid",
        suggested_object_id="ambig_power1",
        device_id=device.id,
        original_device_class="power",
        unit_of_measurement="W",
    )
    ent_reg.async_get_or_create(
        domain="sensor",
        platform="test_domain",
        unique_id="ambig_power2_uid",
        suggested_object_id="ambig_power2",
        device_id=device.id,
        original_device_class="power",
        unit_of_measurement="W",
    )

    async_set_power(hass, "sensor.grid_power", 1000)
    async_set_power(hass, "sensor.ambig_power1", 100)
    async_set_power(hass, "sensor.ambig_power2", 200)

    prefs = dict(base_grid_prefs)
    prefs["device_consumption"] = [
        {"stat_consumption": "sensor.ambig_energy", "name": "Ambiguous Plug"}
    ]
    energy_manager.data = prefs

    entry = MockConfigEntry(domain=DOMAIN, data={})
    entry.add_to_hass(hass)
    await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    remainder = hass.states.get("sensor.powerguesser_remainder_power")
    # Not subtracted
    assert float(remainder.state) == 1000.0
    assert remainder.attributes["metered_w"] == 0.0

    issue_reg = ir.async_get(hass)
    issue = issue_reg.async_get_issue(DOMAIN, "metered_no_power_sensor_ambig_energy")
    assert issue is not None
    assert issue.severity == ir.IssueSeverity.WARNING


async def test_mw_only_sibling_not_selected(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    base_grid_prefs: dict[str, Any],
) -> None:
    """5. Sibling power sensor reporting in MW is ignored (W/kW-only rule, D-03) -> metered_no_power issue."""
    mock_entry = MockConfigEntry(domain="test_domain")
    mock_entry.add_to_hass(hass)
    dev_reg = dr.async_get(hass)
    ent_reg = er.async_get(hass)

    device = dev_reg.async_get_or_create(
        config_entry_id=mock_entry.entry_id,
        identifiers={("test_domain", "plug_mw")},
        name="MW Plug",
    )
    ent_reg.async_get_or_create(
        domain="sensor",
        platform="test_domain",
        unique_id="mw_energy_uid",
        suggested_object_id="mw_energy",
        device_id=device.id,
        unit_of_measurement="MWh",
    )
    ent_reg.async_get_or_create(
        domain="sensor",
        platform="test_domain",
        unique_id="mw_power_uid",
        suggested_object_id="mw_power",
        device_id=device.id,
        original_device_class="power",
        unit_of_measurement="MW",
    )

    async_set_power(hass, "sensor.grid_power", 1000)
    hass.states.async_set("sensor.mw_power", "0.001", {"unit_of_measurement": "MW", "device_class": "power"})

    prefs = dict(base_grid_prefs)
    prefs["device_consumption"] = [
        {"stat_consumption": "sensor.mw_energy", "name": "MW Device"}
    ]
    energy_manager.data = prefs

    entry = MockConfigEntry(domain=DOMAIN, data={})
    entry.add_to_hass(hass)
    await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    remainder = hass.states.get("sensor.powerguesser_remainder_power")
    assert float(remainder.state) == 1000.0

    issue_reg = ir.async_get(hass)
    issue = issue_reg.async_get_issue(DOMAIN, "metered_no_power_sensor_mw_energy")
    assert issue is not None
    assert issue.severity == ir.IssueSeverity.WARNING


async def test_self_reference_metered_stat_rate(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    base_grid_prefs: dict[str, Any],
) -> None:
    """6. Metered stat_rate pointing at powerguesser_remainder_power -> excluded + self_reference issue."""
    ent_reg = er.async_get(hass)
    ent_reg.async_get_or_create(
        domain="sensor",
        platform=DOMAIN,
        unique_id="powerguesser_remainder_power",
        suggested_object_id="powerguesser_remainder_power",
        unit_of_measurement="W",
    )
    async_set_power(hass, "sensor.grid_power", 1000)

    prefs = dict(base_grid_prefs)
    prefs["device_consumption"] = [
        {
            "stat_consumption": "sensor.loop_plug_energy",
            "stat_rate": "sensor.powerguesser_remainder_power",
            "name": "Loop Plug",
        }
    ]
    energy_manager.data = prefs

    entry = MockConfigEntry(domain=DOMAIN, data={})
    entry.add_to_hass(hass)
    await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    # Self-reference ignored, metered_w is 0
    remainder = hass.states.get("sensor.powerguesser_remainder_power")
    assert remainder.attributes["metered_w"] == 0.0

    issue_reg = ir.async_get(hass)
    issue = issue_reg.async_get_issue(DOMAIN, "self_reference_sensor_loop_plug_energy")
    assert issue is not None
    assert issue.severity == ir.IssueSeverity.WARNING


async def test_plug_unavailable_handling(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    base_grid_prefs: dict[str, Any],
) -> None:
    """7. Plug going unavailable -> remainder stays available, metered_unavailable lists it, plug contributes 0."""
    async_set_power(hass, "sensor.grid_power", 1000)
    async_set_power(hass, "sensor.washer_power", 250)

    prefs = dict(base_grid_prefs)
    prefs["device_consumption"] = [
        {
            "stat_consumption": "sensor.washer_energy",
            "stat_rate": "sensor.washer_power",
            "name": "Washer",
        }
    ]
    energy_manager.data = prefs

    entry = MockConfigEntry(domain=DOMAIN, data={})
    entry.add_to_hass(hass)
    await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    remainder = hass.states.get("sensor.powerguesser_remainder_power")
    assert float(remainder.state) == 750.0
    assert remainder.attributes["metered_unavailable"] == []

    # Washer becomes unavailable
    async_set_power(hass, "sensor.washer_power", "unavailable")
    # Also trigger grid state report to cause emission
    async_set_power(hass, "sensor.grid_power", 1000)
    await hass.async_block_till_done()

    remainder_unavail = hass.states.get("sensor.powerguesser_remainder_power")
    assert remainder_unavail.state != "unavailable"
    assert float(remainder_unavail.state) == 1000.0
    assert remainder_unavail.attributes["metered_w"] == 0.0
    assert remainder_unavail.attributes["metered_unavailable"] == ["sensor.washer_power"]


async def test_plug_updating_alone_does_not_trigger_emission(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    base_grid_prefs: dict[str, Any],
) -> None:
    """8. Plug updating alone does not trigger a gross/remainder emission."""
    async_set_power(hass, "sensor.grid_power", 1000)
    async_set_power(hass, "sensor.washer_power", 200)

    prefs = dict(base_grid_prefs)
    prefs["device_consumption"] = [
        {
            "stat_consumption": "sensor.washer_energy",
            "stat_rate": "sensor.washer_power",
            "name": "Washer",
        }
    ]
    energy_manager.data = prefs

    entry = MockConfigEntry(domain=DOMAIN, data={})
    entry.add_to_hass(hass)
    await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    assert float(hass.states.get("sensor.powerguesser_remainder_power").state) == 800.0

    hub = entry.runtime_data
    call_count = hub.coordinator.async_set_updated_data.call_count if hasattr(hub.coordinator.async_set_updated_data, "call_count") else 0

    # Washer updates to 300 W alone (no grid event)
    async_set_power(hass, "sensor.washer_power", 300)
    await hass.async_block_till_done()

    # Remainder still reads 800 W until gross settles and emits!
    assert float(hass.states.get("sensor.powerguesser_remainder_power").state) == 800.0


async def test_device_target_session_leaves_the_remainder_unchanged(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
) -> None:
    """D-25: a device-target plug session is recorded but never subtracted live.

    The test first proves the plug is not a metered entity, so "remainder
    unchanged" cannot hold for the wrong reason (review concern L161@8b3be9c).
    """
    seed_candidates(
        hass_storage, [make_switched_candidate("sig01", "Kettle", rise_w=2000.0)]
    )
    energy_manager.data = copy.deepcopy(std_prefs)
    async_set_power(hass, "sensor.grid_power", 1500)
    async_set_power(hass, "sensor.solar_power", 0)
    async_set_power(hass, "sensor.battery_power", 0)
    async_set_power(hass, "sensor.washer_power", 0)
    async_set_power(hass, "sensor.test_plug_power", 400)
    _aligned_move(freezer)
    entry = await async_setup_via_flow(hass)
    _pin_grid(entry)
    await hass.async_block_till_done()

    hub = entry.runtime_data
    assert "sensor.test_plug_power" not in hub.model.metered_entity_ids

    await async_confirm_candidate(hass, entry, "sig01", "Kettle")
    sub = next(
        s for s in entry.get_subentries_of_type("device") if s.title == "Kettle"
    )
    device_uuid = sub.data["device_uuid"]

    async_set_power(hass, "sensor.grid_power", 1500)
    await _drive_grid(hass, freezer, 4)
    before = float(hass.states.get("sensor.powerguesser_remainder_power").state)

    menu = await async_open_options_step(hass, entry, "plug_sessions")
    form = await hass.config_entries.options.async_configure(
        menu["flow_id"], {"next_step_id": "session_start"}
    )
    await hass.config_entries.options.async_configure(
        form["flow_id"],
        {"plug": "sensor.test_plug_power", "target": f"device_{device_uuid}"},
    )
    await hass.async_block_till_done()

    sessions = hass_storage["powerguesser.review_sessions"]["data"]["sessions"]
    assert len(sessions) == 1
    assert sessions[0]["target_kind"] == "device"
    assert hub._session_metered_plugs == set()

    async_set_power(hass, "sensor.grid_power", 1500)
    await _drive_grid(hass, freezer, 4)
    after = float(hass.states.get("sensor.powerguesser_remainder_power").state)
    assert after == before


async def test_own_device_energy_in_device_consumption_not_subtracted_no_issue(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    freezer: Any,
) -> None:
    """9. A confirmed device's energy sensor listed in device_consumption is ignored:
    no metered_no_power/self_reference issue, only the washer is subtracted as
    metered, and the device estimate is never subtracted twice (OUT-02, Pitfall 2)."""
    seed_candidates(hass_storage, [make_switched_candidate("sig01", "Kettle", rise_w=2000.0)])
    energy_manager.data = copy.deepcopy(std_prefs)
    async_set_power(hass, "sensor.grid_power", 500)
    async_set_power(hass, "sensor.solar_power", 0)
    async_set_power(hass, "sensor.battery_power", 0)
    async_set_power(hass, "sensor.washer_power", 200)
    _aligned_move(freezer)
    entry = await async_setup_via_flow(hass)
    _pin_grid(entry)
    await hass.async_block_till_done()

    # Confirm the kettle candidate (Plan 04-03 flow) and find its energy entity
    await async_confirm_candidate(hass, entry, "sig01", "Kettle")
    sub = next(s for s in entry.get_subentries_of_type("device") if s.title == "Kettle")
    dev_uuid = sub.data["device_uuid"]
    ent_reg = er.async_get(hass)
    energy_eid = ent_reg.async_get_entity_id(
        "sensor", DOMAIN, f"{entry.entry_id}_{dev_uuid}_energy"
    )
    power_eid = ent_reg.async_get_entity_id(
        "sensor", DOMAIN, f"{entry.entry_id}_{dev_uuid}_power"
    )
    assert energy_eid is not None and power_eid is not None

    # The user adds the device's energy sensor to the Energy dashboard
    prefs = copy.deepcopy(std_prefs)
    prefs["device_consumption"].append(
        {"stat_consumption": energy_eid, "name": "Kettle"}
    )
    await energy_manager.async_update(prefs)
    await hass.async_block_till_done()

    # No metered_no_power or self_reference Repairs issue for the own sensor
    issue_reg = ir.async_get(hass)
    slug = re.sub(r"[^a-z0-9]+", "_", energy_eid.lower()).strip("_")
    assert issue_reg.async_get_issue(DOMAIN, f"metered_no_power_{slug}") is None
    assert issue_reg.async_get_issue(DOMAIN, f"self_reference_{slug}") is None

    # Only the washer (200 W) is subtracted as metered: gross 500 - 200 = 300
    remainder = hass.states.get("sensor.powerguesser_remainder_power")
    assert float(remainder.state) == 300.0
    assert remainder.attributes["metered_w"] == 200.0

    # Drive the kettle on (+2000 W): the estimate follows, metered_w stays 200
    # and the remainder is 2500 - 200 - 2000 = 300 -> not subtracted twice
    async_set_power(hass, "sensor.grid_power", 2500)
    await _drive_grid(hass, freezer, 4)

    assert float(hass.states.get(power_eid).state) == pytest.approx(2000.0, abs=1.0)
    remainder_on = hass.states.get("sensor.powerguesser_remainder_power")
    assert remainder_on.attributes["metered_w"] == 200.0
    assert float(remainder_on.state) == pytest.approx(300.0, abs=1.0)

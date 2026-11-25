"""D-13: no stored device artefact may carry evidence or origin.

One test per copy site of a candidate dict into a device record: confirm,
proposal (set + accept) and the merge alias. The rule lives in the single
``const.CANDIDATE_DECORATION_KEYS`` tuple.
"""

import asyncio
from typing import Any

from homeassistant.components.energy.data import EnergyManager
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.storage import Store

from custom_components.powerguesser.const import (
    DEVICES_STORE_KEY,
    DEVICES_STORE_VERSION,
)
from custom_components.powerguesser.learning import LearningSupervisor

from tests.conftest import (
    async_confirm_candidate,
    async_set_power,
    async_setup_via_flow,
    make_switched_candidate,
    seed_candidates,
)


async def _setup(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> ConfigEntry:
    energy_manager.data = std_prefs
    async_set_power(hass, "sensor.grid_power", 500)
    async_set_power(hass, "sensor.solar_power", 1200)
    async_set_power(hass, "sensor.battery_power", -300)
    async_set_power(hass, "sensor.washer_power", 200)
    seed_candidates(hass_storage, candidates)
    return await async_setup_via_flow(hass)


def _uuid_of(entry: ConfigEntry, title: str) -> str:
    return next(
        s for s in entry.get_subentries_of_type("device") if s.title == title
    ).data["device_uuid"]


async def test_confirm_strips_evidence_and_origin(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
) -> None:
    """Confirming a decorated candidate stores a version artefact with neither key."""
    cand = make_switched_candidate("sig-kettle", "Kettle")
    cand["evidence"] = {"v": 1, "n_runs": 9}
    cand["origin"] = "capture"
    entry = await _setup(hass, energy_manager, std_prefs, hass_storage, [cand])

    await async_confirm_candidate(hass, entry, "sig-kettle", "Kettle")

    devices = entry.runtime_data.devices
    uuid = _uuid_of(entry, "Kettle")
    artefact = devices.record(uuid)["versions"][0]["artefact"]
    assert "evidence" not in artefact
    assert "origin" not in artefact

    stored = await Store(
        hass, DEVICES_STORE_VERSION, DEVICES_STORE_KEY
    ).async_load()
    stored_artefact = stored["devices"][uuid]["versions"][0]["artefact"]
    assert "evidence" not in stored_artefact
    assert "origin" not in stored_artefact


async def test_proposal_strips_evidence_and_origin(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
) -> None:
    """A proposal with both keys stores neither, and accepting it keeps them out."""
    cand = make_switched_candidate("sig-kettle", "Kettle")
    entry = await _setup(hass, energy_manager, std_prefs, hass_storage, [cand])
    await async_confirm_candidate(hass, entry, "sig-kettle", "Kettle")

    devices = entry.runtime_data.devices
    uuid = _uuid_of(entry, "Kettle")

    proposal = dict(cand)
    proposal["evidence"] = {"v": 1, "n_runs": 3}
    proposal["origin"] = "capture"
    await devices.async_set_proposal(
        uuid,
        proposal,
        run_id="run-1",
        proposed_at="2026-11-25T00:00:00+00:00",
        trained_at=None,
    )

    rec = devices.record(uuid)
    assert "evidence" not in rec["pending_proposal"]["artefact"]
    assert "origin" not in rec["pending_proposal"]["artefact"]

    await devices.async_accept_proposal(uuid)
    version = rec["versions"][-1]
    assert "evidence" not in version["artefact"]
    assert "origin" not in version["artefact"]


async def test_merge_alias_strips_evidence_and_origin(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    monkeypatch: Any,
) -> None:
    """Merging a decorated candidate into a device stores a clean alias artefact."""
    target = make_switched_candidate("sig-kettle", "Kettle")
    source = make_switched_candidate("sig-oven", "Oven")
    source["evidence"] = {"v": 1, "n_runs": 2}
    source["origin"] = "capture"
    entry = await _setup(
        hass, energy_manager, std_prefs, hass_storage, [target, source]
    )
    await async_confirm_candidate(hass, entry, "sig-kettle", "Kettle")

    devices = entry.runtime_data.devices
    supervisor = entry.runtime_data.learning
    uuid = _uuid_of(entry, "Kettle")

    async def _fake_refit(self: Any, stage: str, job_path: Any) -> dict[str, Any]:
        await asyncio.sleep(0)
        return {"result": "", "accepted": False, "current_S": 0.0, "new_S": 0.0}

    monkeypatch.setattr(LearningSupervisor, "_async_run_stage", _fake_refit)

    await supervisor._async_merge_confirmed_devices(uuid, "sig-kettle", ["sig-oven"])
    if supervisor.run_task is not None:
        await supervisor.run_task

    alias = devices.record(uuid)["alias_artefacts"]["sig-oven"]
    assert "evidence" not in alias
    assert "origin" not in alias
    assert alias["signature_id"] == "sig-oven"

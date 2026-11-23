"""Read-only Energy dashboard preferences adapter for PowerGuesser."""

from collections.abc import Mapping
import copy
import logging
from typing import Any

from homeassistant.components.energy import async_get_manager
from homeassistant.core import HomeAssistant, valid_entity_id
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers.dispatcher import async_dispatcher_send
from homeassistant.util.unit_conversion import PowerConverter

from .const import DOMAIN, SIGNAL_PREFS_UPDATED
from .engine.model import EntityFacts, RegistryFacts

_LOGGER = logging.getLogger(__name__)
_LISTENER_KEY = "powerguesser_energy_listener"


async def async_get_prefs(hass: HomeAssistant) -> dict[str, Any] | None:
    """Fetch a detached deepcopy of the current Energy dashboard preferences."""
    manager = await async_get_manager(hass)
    if manager is None or not isinstance(manager.data, dict):
        return None
    return copy.deepcopy(manager.data)


def collect_registry_facts(
    hass: HomeAssistant,
    prefs: Mapping[str, Any] | None,
) -> RegistryFacts:
    """Collect entity and registry facts for all entities referenced by Energy preferences."""
    if not isinstance(prefs, Mapping):
        return RegistryFacts.empty()

    entity_ids: set[str] = set()

    energy_sources = prefs.get("energy_sources")
    if isinstance(energy_sources, list):
        for item in energy_sources:
            if not isinstance(item, dict):
                continue
            stat_rate = item.get("stat_rate")
            if isinstance(stat_rate, str) and stat_rate:
                entity_ids.add(stat_rate)
            power_config = item.get("power_config")
            if isinstance(power_config, dict):
                for val in power_config.values():
                    if isinstance(val, str) and val:
                        entity_ids.add(val)

    device_consumption = prefs.get("device_consumption")
    if isinstance(device_consumption, list):
        for item in device_consumption:
            if not isinstance(item, dict):
                continue
            stat_rate = item.get("stat_rate")
            if isinstance(stat_rate, str) and stat_rate:
                entity_ids.add(stat_rate)

    ent_reg = er.async_get(hass)
    facts_dict: dict[str, EntityFacts] = {}

    for eid in entity_ids:
        state = hass.states.get(eid)
        entry = ent_reg.async_get(eid)
        exists = (state is not None) or (entry is not None)

        unit = None
        if state is not None:
            unit = state.attributes.get("unit_of_measurement")
        if unit is None and entry is not None:
            unit = getattr(entry, "unit_of_measurement", None)

        unit_ok = None
        if unit is not None:
            unit_ok = unit in PowerConverter.VALID_UNITS

        platform = entry.platform if entry is not None else None
        device_id = entry.device_id if entry is not None else None

        facts_dict[eid] = EntityFacts(
            exists=exists,
            unit_ok=unit_ok,
            platform=platform,
            device_id=device_id,
        )

    # Collect sibling power sensors for device_consumption entries
    siblings_map: dict[str, tuple[str, ...]] = {}
    if isinstance(device_consumption, list):
        for item in device_consumption:
            if not isinstance(item, dict):
                continue
            stat_consumption = item.get("stat_consumption")
            if not isinstance(stat_consumption, str) or not stat_consumption:
                continue
            # External statistics containing ':' have no siblings
            if ":" in stat_consumption or not valid_entity_id(stat_consumption):
                continue

            entry = ent_reg.async_get(stat_consumption)
            if entry is None or not entry.device_id:
                continue

            device_entries = er.async_entries_for_device(ent_reg, entry.device_id)
            candidates: list[str] = []
            for dev_ent in device_entries:
                if dev_ent.domain != "sensor":
                    continue
                if dev_ent.platform == DOMAIN:
                    continue
                dclass = dev_ent.device_class or dev_ent.original_device_class
                if dclass != "power":
                    continue
                if dev_ent.unit_of_measurement not in {"W", "kW"}:
                    continue
                candidates.append(dev_ent.entity_id)

                if dev_ent.entity_id not in facts_dict:
                    c_state = hass.states.get(dev_ent.entity_id)
                    c_unit = (
                        c_state.attributes.get("unit_of_measurement")
                        if c_state
                        else dev_ent.unit_of_measurement
                    )
                    facts_dict[dev_ent.entity_id] = EntityFacts(
                        exists=(c_state is not None) or (dev_ent is not None),
                        unit_ok=c_unit in PowerConverter.VALID_UNITS if c_unit else None,
                        platform=dev_ent.platform,
                        device_id=dev_ent.device_id,
                    )

            siblings_map[stat_consumption] = tuple(sorted(candidates))

    # Collect the entity-registry platform of each device_consumption
    # stat_consumption. PowerGuesser's own device energy sensors may be
    # listed in the Energy dashboard; the source model skips them silently
    # so their estimates are never subtracted as metered (OUT-02, Pitfall 2).
    # External statistics containing ':' have no entity-registry platform.
    consumption_platforms: dict[str, str | None] = {}
    if isinstance(device_consumption, list):
        for item in device_consumption:
            if not isinstance(item, dict):
                continue
            stat_consumption = item.get("stat_consumption")
            if not isinstance(stat_consumption, str) or not stat_consumption:
                continue
            if ":" in stat_consumption or not valid_entity_id(stat_consumption):
                continue

            entry = ent_reg.async_get(stat_consumption)
            consumption_platforms[stat_consumption] = (
                entry.platform if entry is not None else None
            )

    return RegistryFacts(
        entities=facts_dict,
        siblings=siblings_map,
        consumption_platforms=consumption_platforms,
    )


def device_consumption_power_entities(
    prefs: Mapping[str, Any] | None,
    facts: RegistryFacts,
) -> frozenset[str]:
    """Power entities that already belong to an Energy device_consumption item.

    Every ``stat_rate`` plus every sibling power sensor of a ``stat_consumption``
    entity (facts.siblings). A plug session must never subtract one of these: it
    is already subtracted as a metered device, so a session on it would
    double-count (D-10, T-06-15). The sibling set matters precisely when the
    device's power entity cannot be resolved (two candidate sensors): then
    ``SourceModel.referenced_entity_ids`` names none of them.
    """
    out: set[str] = set()
    if isinstance(prefs, Mapping):
        device_consumption = prefs.get("device_consumption")
        if isinstance(device_consumption, list):
            for item in device_consumption:
                if not isinstance(item, dict):
                    continue
                stat_rate = item.get("stat_rate")
                if isinstance(stat_rate, str) and stat_rate:
                    out.add(stat_rate)
    for siblings in facts.siblings.values():
        for entity_id in siblings:
            out.add(str(entity_id))
    return frozenset(out)


async def async_ensure_prefs_listener(hass: HomeAssistant) -> None:
    """Ensure a singleton update listener is registered on the EnergyManager."""
    if hass.data.get(_LISTENER_KEY):
        return

    manager = await async_get_manager(hass)
    if manager is None:
        return

    async def _async_on_energy_prefs_updated() -> None:
        try:
            async_dispatcher_send(hass, SIGNAL_PREFS_UPDATED)
        except Exception:  # noqa: BLE001
            _LOGGER.exception("PowerGuesser prefs listener failed")

    manager.async_listen_updates(_async_on_energy_prefs_updated)
    hass.data[_LISTENER_KEY] = True

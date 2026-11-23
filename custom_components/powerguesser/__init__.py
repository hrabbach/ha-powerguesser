"""The PowerGuesser integration."""

from collections.abc import Mapping
import logging
import sqlite3
from typing import Any

import voluptuous as vol

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import (
    HomeAssistant,
    ServiceCall,
    ServiceResponse,
    SupportsResponse,
)
from homeassistant.exceptions import ConfigEntryNotReady, ServiceValidationError
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers import issue_registry as ir
from homeassistant.helpers.dispatcher import async_dispatcher_connect
from homeassistant.helpers.service import async_register_admin_service
from homeassistant.helpers.storage import Store
from homeassistant.helpers.typing import ConfigType
from homeassistant.util import dt as dt_util

from .buffer_writer import BufferWriter
from .const import (
    BUFFER_DIR,
    BUFFER_FILE,
    CONF_INVERT_PHASES,
    CONF_PHASE_L1,
    CONF_PHASE_L2,
    CONF_PHASE_L3,
    DOMAIN,
    PLATFORMS,
    PROFILE_STORE_KEY,
    PROFILE_STORE_VERSION,
    SERVICE_LEARN,
    SERVICE_PREVIEW,
    SERVICE_RUN_VALIDATION,
    SIGNAL_PREFS_UPDATED,
)
from .devices import DeviceRuntime, async_remove_device_data
from .energy_prefs import async_ensure_prefs_listener
from .engine.buffer import BufferDB, remove_buffer_files
from .hub import PowerGuesserHub
from .learning import LearningSupervisor, async_remove_learning_data
from .validation import ValidationManager, async_remove_validation_data

_LOGGER = logging.getLogger(__name__)

CONFIG_SCHEMA = cv.config_entry_only_config_schema(DOMAIN)

type PowerGuesserConfigEntry = ConfigEntry[PowerGuesserHub]


async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Set up PowerGuesser services."""
    async def _async_learn_service(call: ServiceCall) -> None:
        entries = hass.config_entries.async_loaded_entries(DOMAIN)
        if not entries:
            raise ServiceValidationError(
                translation_domain=DOMAIN,
                translation_key="not_loaded",
            )
        entry = entries[0]
        await entry.runtime_data.learning.async_request_run("service")

    async_register_admin_service(
        hass,
        DOMAIN,
        SERVICE_LEARN,
        _async_learn_service,
    )

    async def _async_preview_service(call: ServiceCall) -> ServiceResponse:
        entries = hass.config_entries.async_loaded_entries(DOMAIN)
        if not entries:
            raise ServiceValidationError(
                translation_domain=DOMAIN,
                translation_key="not_loaded",
            )
        entry = entries[0]
        return await entry.runtime_data.learning.async_preview_payload()

    async_register_admin_service(
        hass,
        DOMAIN,
        SERVICE_PREVIEW,
        _async_preview_service,
        supports_response=SupportsResponse.ONLY,
    )

    async def _async_run_validation_service(call: ServiceCall) -> ServiceResponse:
        entries = hass.config_entries.async_loaded_entries(DOMAIN)
        if not entries:
            raise ServiceValidationError(
                translation_domain=DOMAIN,
                translation_key="not_loaded",
            )
        entry = entries[0]
        return await entry.runtime_data.validation.async_run_validation(
            synthetic_overlap=call.data["synthetic_overlap"]
        )

    async_register_admin_service(
        hass,
        DOMAIN,
        SERVICE_RUN_VALIDATION,
        _async_run_validation_service,
        schema=vol.Schema(
            {vol.Optional("synthetic_overlap", default=False): cv.boolean}
        ),
        supports_response=SupportsResponse.ONLY,
    )
    return True


def _phase_subset(options: Mapping[str, Any]) -> tuple[Any, Any, Any, bool]:
    """Extract phase mapping subset from options."""
    return (
        options.get(CONF_PHASE_L1),
        options.get(CONF_PHASE_L2),
        options.get(CONF_PHASE_L3),
        bool(options.get(CONF_INVERT_PHASES)),
    )


async def async_setup_entry(
    hass: HomeAssistant, entry: PowerGuesserConfigEntry
) -> bool:
    """Set up PowerGuesser from a config entry."""
    await async_ensure_prefs_listener(hass)
    db = BufferDB(hass.config.path(BUFFER_DIR, BUFFER_FILE))
    try:
        await hass.async_add_executor_job(db.open)
    except (OSError, sqlite3.Error) as err:
        raise ConfigEntryNotReady(f"Cannot open the PowerGuesser buffer: {err}") from err

    if db.reset_backup:
        _LOGGER.warning(
            "PowerGuesser buffer was corrupt and has been reset; backup saved as %s",
            db.reset_backup,
        )
        ir.async_create_issue(
            hass,
            DOMAIN,
            "buffer_reset",
            is_fixable=False,
            is_persistent=False,
            severity=ir.IssueSeverity.WARNING,
            translation_key="buffer_reset",
            translation_placeholders={"backup": db.reset_backup},
        )

    writer = BufferWriter(hass, db)
    await writer.async_start()
    hub = PowerGuesserHub(hass, entry, writer)
    hub.devices = DeviceRuntime(hass, entry, hub)
    await hub.devices.async_load()
    await hub.async_start()
    supervisor = LearningSupervisor(hass, entry, hub)
    await supervisor.async_start()
    hub.learning = supervisor
    # An active plug session resumes after a restart only through a rebuild, and
    # the first rebuild in hub.async_start ran before learning (and therefore
    # the review sessions Store) existed. Request one now so the plug is
    # re-tracked and, for a candidate target, subtracted again without user
    # action (Plan 06-06 D-10/D-11).
    if hub.learning.review.active_sessions(dt_util.utcnow()):
        hub.async_request_rebuild()
    hub.validation = ValidationManager(hass, entry, hub)
    await hub.validation.async_load()
    entry.runtime_data = hub

    entry.async_on_unload(
        async_dispatcher_connect(hass, SIGNAL_PREFS_UPDATED, hub.async_request_rebuild)
    )

    last_phase_subset = _phase_subset(entry.options)
    last_options = dict(entry.options)
    last_data = dict(entry.data)

    async def _async_options_updated(
        hass: HomeAssistant, updated_entry: PowerGuesserConfigEntry
    ) -> None:
        """Handle entry updates (options, data, subentries) without reloading."""
        nonlocal last_phase_subset, last_options, last_data
        # Subentry changes (confirm/rename/delete) also fire this listener:
        # reconcile device entities, then leave learning alone unless the
        # options or data actually changed (RESEARCH Pattern 1, Pitfall 1).
        await updated_entry.runtime_data.devices.async_reconcile()
        new_options = dict(updated_entry.options)
        new_data = dict(updated_entry.data)
        if new_options == last_options and new_data == last_data:
            return
        last_options = new_options
        last_data = new_data
        new_phase_subset = _phase_subset(updated_entry.options)
        if new_phase_subset != last_phase_subset:
            last_phase_subset = new_phase_subset
            updated_entry.runtime_data.async_request_rebuild()
        # Options/data changed: the same save can change the schedule or
        # consent-relevant options (WR-10)
        await supervisor.async_settings_changed()

    entry.async_on_unload(entry.add_update_listener(_async_options_updated))

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    hub.async_schedule_bootstrap()
    return True


async def async_unload_entry(
    hass: HomeAssistant, entry: PowerGuesserConfigEntry
) -> bool:
    """Unload a PowerGuesser config entry."""
    unload_ok = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
    if unload_ok:
        # Flush the device Stores before the hub stops: a pending delay-save
        # of a stale Store instance must never overwrite a reload (Pitfall 5).
        await entry.runtime_data.devices.async_flush()
        if getattr(entry.runtime_data, "learning", None) is not None:
            await entry.runtime_data.learning.async_stop()
        await entry.runtime_data.async_stop()
        await entry.runtime_data.writer.async_close()
    return unload_ok


async def async_remove_entry(
    hass: HomeAssistant, entry: PowerGuesserConfigEntry
) -> None:
    """Remove a PowerGuesser config entry, delete buffer files, and remove profile store."""
    await async_remove_learning_data(hass)
    await async_remove_validation_data(hass)
    await async_remove_device_data(hass)
    path = hass.config.path(BUFFER_DIR, BUFFER_FILE)
    deleted = await hass.async_add_executor_job(remove_buffer_files, path)
    _LOGGER.info("Removed PowerGuesser buffer files: %s", deleted)
    await Store(hass, PROFILE_STORE_VERSION, PROFILE_STORE_KEY).async_remove()


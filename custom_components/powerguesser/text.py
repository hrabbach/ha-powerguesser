"""Text platform for PowerGuesser: the capture label entity (CAP-07, D-02)."""

from homeassistant.components.text import TextEntity
from homeassistant.const import STATE_UNAVAILABLE, STATE_UNKNOWN
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ServiceValidationError
from homeassistant.helpers.dispatcher import async_dispatcher_connect
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.restore_state import RestoreEntity

from . import PowerGuesserConfigEntry
from .const import DOMAIN, SIGNAL_CAPTURE_UPDATED
from .hub import PowerGuesserHub
from .review_store import normalise_name
from .sensor import _device_info


class CaptureLabelText(TextEntity, RestoreEntity):
    """Capture label text entity (D-02).

    The label names a new device, or a candidate / confirmed device by display
    name or signature id (resolved by the manager). It survives a restart through
    RestoreEntity and is cleared by the manager only after a successful start.
    Values are validated by ``review_store.normalise_name`` (NFC, whitespace
    collapsed, at most 64 code points, no control characters).
    """

    _attr_has_entity_name = True
    _attr_translation_key = "capture_label"
    _attr_unique_id = "powerguesser_capture_label"
    _attr_native_min = 0
    _attr_native_max = 64

    def __init__(self, hub: PowerGuesserHub, entry: PowerGuesserConfigEntry) -> None:
        """Initialize the capture label text entity."""
        self.hub = hub
        self._entry_id = entry.entry_id
        self._attr_device_info = _device_info()

    @property
    def native_value(self) -> str:
        """Return the label stored on the capture manager."""
        return self.hub.captures.surface_label

    async def async_added_to_hass(self) -> None:
        """Restore the last label and subscribe to capture state changes."""
        await super().async_added_to_hass()
        last = await self.async_get_last_state()
        if last is not None and last.state not in (STATE_UNKNOWN, STATE_UNAVAILABLE):
            value = last.state or ""
            if value == "":
                self.hub.captures.set_surface_label("")
            else:
                try:
                    self.hub.captures.set_surface_label(normalise_name(value))
                except ValueError:
                    # A stale or malformed restored label is dropped; the entity
                    # keeps the manager's default (empty).
                    pass
        self.async_on_remove(
            async_dispatcher_connect(
                self.hass,
                f"{SIGNAL_CAPTURE_UPDATED}_{self._entry_id}",
                self.async_write_ha_state,
            )
        )

    async def async_set_value(self, value: str) -> None:
        """Store a validated label; refuse an invalid one (D-02)."""
        if not (value or "").strip():
            self.hub.captures.set_surface_label("")
            self.async_write_ha_state()
            return
        try:
            normalised = normalise_name(value)
        except ValueError as err:
            raise ServiceValidationError(
                translation_domain=DOMAIN,
                translation_key="capture_invalid_name",
                translation_placeholders={"reason": str(err)},
            ) from err
        self.hub.captures.set_surface_label(normalised)
        self.async_write_ha_state()


async def async_setup_entry(
    hass: HomeAssistant,
    entry: PowerGuesserConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up the PowerGuesser capture label text entity."""
    hub = entry.runtime_data
    async_add_entities([CaptureLabelText(hub, entry)])

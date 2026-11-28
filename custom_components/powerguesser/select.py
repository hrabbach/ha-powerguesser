"""Select platform for PowerGuesser: the capture target-kind select (CAP-07, D-01)."""

from homeassistant.components.select import SelectEntity
from homeassistant.core import HomeAssistant
from homeassistant.helpers.dispatcher import async_dispatcher_connect
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.restore_state import RestoreEntity

from . import PowerGuesserConfigEntry
from .capture import CAPTURE_TARGET_KINDS
from .const import SIGNAL_CAPTURE_UPDATED
from .hub import PowerGuesserHub
from .sensor import _device_info


class CaptureTargetSelect(SelectEntity, RestoreEntity):
    """Static target-kind select for the capture surface (D-01).

    The three options are ``new_device``, ``candidate`` and ``device`` (the same
    values as the ``capture_start`` service's ``target_type``). Dynamic per-
    candidate options are rejected (registry churn, PITFALLS 18). The chosen
    value survives a restart through RestoreEntity; a restored value outside the
    options falls back to ``new_device``.
    """

    _attr_has_entity_name = True
    _attr_should_poll = False
    _attr_translation_key = "capture_target"
    _attr_unique_id = "powerguesser_capture_target"
    _attr_options = list(CAPTURE_TARGET_KINDS)

    def __init__(self, hub: PowerGuesserHub, entry: PowerGuesserConfigEntry) -> None:
        """Initialize the capture target select."""
        self.hub = hub
        self._entry_id = entry.entry_id
        self._attr_device_info = _device_info()

    @property
    def current_option(self) -> str:
        """Return the selected target kind."""
        return self.hub.captures.surface_kind

    async def async_added_to_hass(self) -> None:
        """Restore the last option and subscribe to capture state changes."""
        await super().async_added_to_hass()
        last = await self.async_get_last_state()
        if last is not None and last.state in self._attr_options:
            self.hub.captures.set_surface_kind(last.state)
        self.async_on_remove(
            async_dispatcher_connect(
                self.hass,
                f"{SIGNAL_CAPTURE_UPDATED}_{self._entry_id}",
                self.async_write_ha_state,
            )
        )

    async def async_select_option(self, option: str) -> None:
        """Store the selected target kind."""
        self.hub.captures.set_surface_kind(option)
        self.async_write_ha_state()


async def async_setup_entry(
    hass: HomeAssistant,
    entry: PowerGuesserConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up the PowerGuesser capture target select."""
    hub = entry.runtime_data
    async_add_entities([CaptureTargetSelect(hub, entry)])

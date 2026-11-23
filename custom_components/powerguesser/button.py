"""Button platform for PowerGuesser."""

from homeassistant.components.button import ButtonEntity
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from . import PowerGuesserConfigEntry
from .hub import PowerGuesserHub
from .sensor import _device_info


class LearnButton(ButtonEntity):
    """Button entity to trigger device learning."""

    _attr_has_entity_name = True
    _attr_translation_key = "learn"
    _attr_unique_id = "powerguesser_learn"

    def __init__(self, hub: PowerGuesserHub) -> None:
        """Initialize the learn button."""
        self.hub = hub
        self._attr_device_info = _device_info()

    async def async_press(self) -> None:
        """Handle button press to start a learning run."""
        await self.hub.learning.async_request_run("button")


async def async_setup_entry(
    hass: object,
    entry: PowerGuesserConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up PowerGuesser buttons from a config entry."""
    hub = entry.runtime_data
    async_add_entities([LearnButton(hub)])

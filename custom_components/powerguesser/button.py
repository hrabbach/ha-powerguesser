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


class CaptureStartButton(ButtonEntity):
    """Button entity to start a live capture from the hub surface (CAP-07)."""

    _attr_has_entity_name = True
    _attr_translation_key = "capture_start"
    _attr_unique_id = "powerguesser_capture_start"

    def __init__(self, hub: PowerGuesserHub) -> None:
        """Initialize the start-capture button."""
        self.hub = hub
        self._attr_device_info = _device_info()

    async def async_press(self) -> None:
        """Start the capture described by the select and label entities."""
        await self.hub.captures.async_start_from_surface()


class CaptureStopButton(ButtonEntity):
    """Button entity to stop the open live capture (CAP-07).

    Stays always available: a press with nothing open surfaces the existing
    ``capture_not_open`` ServiceValidationError rather than being greyed out,
    so an automation that presses the buttons in sequence still works.
    """

    _attr_has_entity_name = True
    _attr_translation_key = "capture_stop"
    _attr_unique_id = "powerguesser_capture_stop"

    def __init__(self, hub: PowerGuesserHub) -> None:
        """Initialize the stop-capture button."""
        self.hub = hub
        self._attr_device_info = _device_info()

    async def async_press(self) -> None:
        """Stop the open capture (returns at once with state resolving)."""
        await self.hub.captures.async_stop()


async def async_setup_entry(
    hass: object,
    entry: PowerGuesserConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up PowerGuesser buttons from a config entry."""
    hub = entry.runtime_data
    async_add_entities(
        [LearnButton(hub), CaptureStartButton(hub), CaptureStopButton(hub)]
    )

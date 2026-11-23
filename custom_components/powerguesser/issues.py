"""Issue reconciler for Home Assistant Repairs."""

from collections.abc import Iterable
import logging

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import issue_registry as ir

from .const import DOMAIN
from .engine.model import Problem

_LOGGER = logging.getLogger(__name__)


class IssueReconciler:
    """Manages Home Assistant Repairs issues for PowerGuesser."""

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        """Initialize the reconciler."""
        self.hass = hass
        self.entry = entry
        self._active_issue_ids: set[str] = set()

    # desired-set diff, portable to the minimum supported HA version
    @callback
    def apply(self, problems: Iterable[Problem]) -> None:
        """Reconcile active issues to match the desired problem set."""
        desired: dict[str, Problem] = {p.issue_id: p for p in problems}

        for issue_id, problem in desired.items():
            ir.async_create_issue(
                self.hass,
                DOMAIN,
                issue_id,
                is_fixable=False,
                is_persistent=False,
                config_entry_id=self.entry.entry_id,
                severity=ir.IssueSeverity(problem.severity.value),
                translation_key=problem.translation_key,
                translation_placeholders=dict(problem.placeholders),
            )

        for issue_id in self._active_issue_ids - desired.keys():
            ir.async_delete_issue(self.hass, DOMAIN, issue_id)

        self._active_issue_ids = set(desired.keys())

    @callback
    def clear(self) -> None:
        """Delete all active issues."""
        for issue_id in self._active_issue_ids:
            ir.async_delete_issue(self.hass, DOMAIN, issue_id)
        self._active_issue_ids.clear()

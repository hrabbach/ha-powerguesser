"""Repairs platform for PowerGuesser."""

from typing import Any
import voluptuous as vol

from homeassistant.components.repairs import ConfirmRepairFlow, FlowType, RepairsFlow
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResult

from .const import (
    CONF_LLM_BASE_URL,
    CONF_SHARE_NAMES,
    ISSUE_LEARNING_CONSENT,
    ISSUE_LLM_AUTH,
)
from .consent import format_origin
from .llm_client import endpoint_origin


class LlmAuthRepairFlow(RepairsFlow):
    """Handler for an LLM auth fix flow that opens options."""

    def __init__(self, entry_id: str, hass: HomeAssistant | None = None) -> None:
        """Initialize the repair flow."""
        self.entry_id = entry_id
        if hass is not None:
            self.hass = hass

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle first step, redirect to confirm."""
        return await self.async_step_confirm()

    async def async_step_confirm(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle confirm step showing an empty form, opening options on submit."""
        if user_input is not None:
            result = await self.hass.config_entries.options.async_init(self.entry_id)
            return self.async_abort(
                reason="open_options",
                next_flow=(FlowType.OPTIONS_FLOW, result["flow_id"]),
            )

        return self.async_show_form(
            step_id="confirm",
            data_schema=vol.Schema({}),
        )


class ConsentRepairFlow(RepairsFlow):
    """Handler for an LLM learning consent fix flow."""

    def __init__(self, data: dict[str, Any], hass: HomeAssistant | None = None) -> None:
        """Initialize the consent repair flow."""
        self.data = data
        if hass is not None:
            self.hass = hass

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle first step, redirect to confirm.

        The init input is not a confirmation: HA's repairs flow manager injects
        {"issue_id": ...} here, so forwarding it would grant consent unseen.
        """
        return await self.async_step_confirm()

    async def async_step_confirm(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle confirm step showing description placeholders from data."""
        if user_input is not None:
            entry = self.hass.config_entries.async_get_entry(self.data["entry_id"])
            if not entry or getattr(entry, "runtime_data", None) is None:
                return self.async_abort(reason="endpoint_changed")
            supervisor = entry.runtime_data.learning

            base_url = entry.data.get(CONF_LLM_BASE_URL)
            if not base_url:
                await supervisor._async_check_consent(immediate=True)
                return self.async_abort(reason="endpoint_changed")

            current_endpoint = format_origin(*endpoint_origin(base_url))
            current_names_flag = bool(entry.options.get(CONF_SHARE_NAMES, False))
            current_names_str = "included" if current_names_flag else "not included"

            if (
                current_endpoint != self.data.get("endpoint")
                or current_names_str != self.data.get("names")
            ):
                await supervisor._async_check_consent(immediate=True)
                return self.async_abort(reason="endpoint_changed")

            await supervisor.consent.async_grant(base_url, current_names_flag)
            await supervisor.async_settings_changed()
            return self.async_create_entry(data={})

        placeholders = {k: v for k, v in self.data.items() if k != "entry_id"}
        return self.async_show_form(
            step_id="confirm",
            data_schema=vol.Schema({}),
            description_placeholders=placeholders,
        )


async def async_create_fix_flow(
    hass: HomeAssistant, issue_id: str, data: dict[str, Any] | None
) -> RepairsFlow:
    """Create a fix flow for PowerGuesser repairs issues."""
    if issue_id == ISSUE_LLM_AUTH and data and "entry_id" in data:
        return LlmAuthRepairFlow(data["entry_id"], hass=hass)
    if issue_id == ISSUE_LEARNING_CONSENT and data and "entry_id" in data:
        return ConsentRepairFlow(data, hass=hass)
    return ConfirmRepairFlow()

"""Tests for PowerGuesser options flow Learning step."""

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType
from homeassistant.helpers import issue_registry as ir

from custom_components.powerguesser.const import (
    CONF_INVERT_PHASES,
    CONF_MAX_TOKENS,
    CONF_MONTH_COST_LIMIT,
    CONF_MONTH_INPUT_TOKENS,
    CONF_MONTH_OUTPUT_TOKENS,
    CONF_PHASE_L1,
    CONF_PRICE_INPUT,
    CONF_PRICE_OUTPUT,
    CONF_RUN_INPUT_TOKENS,
    CONF_RUN_OUTPUT_TOKENS,
    CONF_SCHEDULE,
    CONF_SCHEDULE_TIME,
    CONF_SCHEDULE_WEEKDAY,
    CONF_SHARE_NAMES,
    CONF_TRUNCATION_GUARD,
    CONF_VALIDATE_AFTER_LEARNING,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MONTH_INPUT_TOKENS,
    DEFAULT_MONTH_OUTPUT_TOKENS,
    DEFAULT_RUN_INPUT_TOKENS,
    DEFAULT_RUN_OUTPUT_TOKENS,
    DEFAULT_SCHEDULE,
    DEFAULT_SCHEDULE_TIME,
    DEFAULT_SCHEDULE_WEEKDAY,
    DEFAULT_VALIDATE_AFTER_LEARNING,
    DOMAIN,
    ISSUE_LEARNING_CONSENT,
)
from tests.conftest import (
    async_seed_consent,
    async_seed_llm,
    async_set_power,
    async_setup_via_flow,
)


def _set_std_powers(hass: HomeAssistant) -> None:
    async_set_power(hass, "sensor.grid_power", 500)
    async_set_power(hass, "sensor.solar_power", 1200)
    async_set_power(hass, "sensor.battery_power", -300)
    async_set_power(hass, "sensor.washer_power", 200)


async def test_options_menu_includes_learning(
    hass: HomeAssistant,
    energy_manager: object,
    std_prefs: dict[str, Any],
) -> None:
    """The options menu lists phases, llm_endpoint, learning, review, plug_sessions and capture."""
    energy_manager.data = std_prefs
    _set_std_powers(hass)
    entry = await async_setup_via_flow(hass)

    result = await hass.config_entries.options.async_init(entry.entry_id)
    assert result["type"] is FlowResultType.MENU
    assert result["step_id"] == "init"
    assert result["menu_options"] == [
        "phases",
        "llm_endpoint",
        "learning",
        "review",
        "plug_sessions",
        "capture",
    ]


async def test_learning_form_defaults(
    hass: HomeAssistant,
    energy_manager: object,
    std_prefs: dict[str, Any],
) -> None:
    """The learning form shows defaults when nothing is stored."""
    energy_manager.data = std_prefs
    _set_std_powers(hass)
    entry = await async_setup_via_flow(hass)

    result = await hass.config_entries.options.async_init(entry.entry_id)
    result2 = await hass.config_entries.options.async_configure(
        result["flow_id"],
        {"next_step_id": "learning"},
    )
    assert result2["type"] is FlowResultType.FORM
    assert result2["step_id"] == "learning"

    schema = result2["data_schema"]
    # Check suggested values on schema
    suggested = {
        k.schema: k.description.get("suggested_value")
        for k in schema.schema
        if isinstance(getattr(k, "description", None), dict)
    }
    assert suggested[CONF_SHARE_NAMES] is False
    assert suggested[CONF_MAX_TOKENS] == DEFAULT_MAX_TOKENS
    assert suggested[CONF_RUN_INPUT_TOKENS] == DEFAULT_RUN_INPUT_TOKENS
    assert suggested[CONF_RUN_OUTPUT_TOKENS] == DEFAULT_RUN_OUTPUT_TOKENS
    assert suggested[CONF_MONTH_INPUT_TOKENS] == DEFAULT_MONTH_INPUT_TOKENS
    assert suggested[CONF_MONTH_OUTPUT_TOKENS] == DEFAULT_MONTH_OUTPUT_TOKENS
    assert suggested.get(CONF_PRICE_INPUT) is None
    assert suggested.get(CONF_PRICE_OUTPUT) is None
    assert suggested.get(CONF_MONTH_COST_LIMIT) is None
    assert suggested[CONF_SCHEDULE] == DEFAULT_SCHEDULE
    assert suggested[CONF_SCHEDULE_WEEKDAY] == DEFAULT_SCHEDULE_WEEKDAY
    assert suggested[CONF_SCHEDULE_TIME] == DEFAULT_SCHEDULE_TIME
    assert suggested[CONF_TRUNCATION_GUARD] is True
    assert suggested[CONF_VALIDATE_AFTER_LEARNING] is False


async def test_learning_form_saves_and_merges_options(
    hass: HomeAssistant,
    energy_manager: object,
    std_prefs: dict[str, Any],
) -> None:
    """Saving merges into entry.options and preserves existing phase keys; clearing optional removes them."""
    energy_manager.data = std_prefs
    _set_std_powers(hass)
    entry = await async_setup_via_flow(hass)

    # Set phase options initially
    hass.config_entries.async_update_entry(
        entry,
        options={
            CONF_PHASE_L1: "sensor.grid_power",
            CONF_INVERT_PHASES: True,
            CONF_PRICE_INPUT: 1.5,
            CONF_PRICE_OUTPUT: 3.0,
            CONF_MONTH_COST_LIMIT: 25.0,
        },
    )

    result = await hass.config_entries.options.async_init(entry.entry_id)
    result2 = await hass.config_entries.options.async_configure(
        result["flow_id"],
        {"next_step_id": "learning"},
    )

    # Submit without price and cost limit (clearing them)
    user_input = {
        CONF_SHARE_NAMES: True,
        CONF_MAX_TOKENS: 8000,
        CONF_RUN_INPUT_TOKENS: 200_000,
        CONF_RUN_OUTPUT_TOKENS: 64_000,
        CONF_MONTH_INPUT_TOKENS: 1_000_000,
        CONF_MONTH_OUTPUT_TOKENS: 300_000,
        CONF_SCHEDULE: "weekly",
        CONF_SCHEDULE_WEEKDAY: "tue",
        CONF_SCHEDULE_TIME: "04:30:00",
        CONF_TRUNCATION_GUARD: False,
        CONF_VALIDATE_AFTER_LEARNING: False,
    }

    result3 = await hass.config_entries.options.async_configure(
        result2["flow_id"],
        user_input,
    )
    assert result3["type"] is FlowResultType.CREATE_ENTRY
    options = entry.options

    # Phase keys preserved
    assert options[CONF_PHASE_L1] == "sensor.grid_power"
    assert options[CONF_INVERT_PHASES] is True

    # Learning keys saved
    assert options[CONF_SHARE_NAMES] is True
    assert options[CONF_MAX_TOKENS] == 8000
    assert options[CONF_RUN_INPUT_TOKENS] == 200_000
    assert options[CONF_RUN_OUTPUT_TOKENS] == 64_000
    assert options[CONF_MONTH_INPUT_TOKENS] == 1_000_000
    assert options[CONF_MONTH_OUTPUT_TOKENS] == 300_000
    assert options[CONF_SCHEDULE] == "weekly"
    assert options[CONF_SCHEDULE_WEEKDAY] == "tue"
    assert options[CONF_SCHEDULE_TIME] == "04:30:00"
    assert options[CONF_TRUNCATION_GUARD] is False

    # Cleared keys removed
    assert CONF_PRICE_INPUT not in options
    assert CONF_PRICE_OUTPUT not in options
    assert CONF_MONTH_COST_LIMIT not in options


async def test_learning_form_validation_errors(
    hass: HomeAssistant,
    energy_manager: object,
    std_prefs: dict[str, Any],
) -> None:
    """Cost limit without both prices -> error; max_tokens > run_output -> error."""
    energy_manager.data = std_prefs
    _set_std_powers(hass)
    entry = await async_setup_via_flow(hass)

    result = await hass.config_entries.options.async_init(entry.entry_id)
    result2 = await hass.config_entries.options.async_configure(
        result["flow_id"],
        {"next_step_id": "learning"},
    )

    # 1. cost limit with missing price
    bad_input_1 = {
        CONF_SHARE_NAMES: False,
        CONF_MAX_TOKENS: 16000,
        CONF_RUN_INPUT_TOKENS: 300_000,
        CONF_RUN_OUTPUT_TOKENS: 128_000,
        CONF_MONTH_INPUT_TOKENS: 2_000_000,
        CONF_MONTH_OUTPUT_TOKENS: 600_000,
        CONF_PRICE_INPUT: 1.5,
        # price_output missing
        CONF_MONTH_COST_LIMIT: 10.0,
        CONF_SCHEDULE: "off",
        CONF_SCHEDULE_WEEKDAY: "sun",
        CONF_SCHEDULE_TIME: "03:00:00",
        CONF_TRUNCATION_GUARD: True,
        CONF_VALIDATE_AFTER_LEARNING: False,
    }
    res_err1 = await hass.config_entries.options.async_configure(
        result2["flow_id"],
        bad_input_1,
    )
    assert res_err1["type"] is FlowResultType.FORM
    assert res_err1["errors"] == {"base": "cost_limit_needs_prices"}

    # 2. max_tokens_per_call > run_output_tokens
    bad_input_2 = {
        CONF_SHARE_NAMES: False,
        CONF_MAX_TOKENS: 150_000,
        CONF_RUN_INPUT_TOKENS: 300_000,
        CONF_RUN_OUTPUT_TOKENS: 128_000,
        CONF_MONTH_INPUT_TOKENS: 2_000_000,
        CONF_MONTH_OUTPUT_TOKENS: 600_000,
        CONF_SCHEDULE: "off",
        CONF_SCHEDULE_WEEKDAY: "sun",
        CONF_SCHEDULE_TIME: "03:00:00",
        CONF_TRUNCATION_GUARD: True,
        CONF_VALIDATE_AFTER_LEARNING: False,
    }
    res_err2 = await hass.config_entries.options.async_configure(
        result2["flow_id"],
        bad_input_2,
    )
    assert res_err2["type"] is FlowResultType.FORM
    assert res_err2["errors"] == {"base": "max_tokens_above_run_cap"}


async def test_saving_learning_does_not_rebuild_hub(
    hass: HomeAssistant,
    energy_manager: object,
    std_prefs: dict[str, Any],
) -> None:
    """Saving the learning options does not rebuild the hub."""
    energy_manager.data = std_prefs
    _set_std_powers(hass)
    entry = await async_setup_via_flow(hass)
    hub = entry.runtime_data

    with patch.object(hub, "async_rebuild") as mock_rebuild:
        result = await hass.config_entries.options.async_init(entry.entry_id)
        result2 = await hass.config_entries.options.async_configure(
            result["flow_id"],
            {"next_step_id": "learning"},
        )
        await hass.config_entries.options.async_configure(
            result2["flow_id"],
            {
                CONF_SHARE_NAMES: False,
                CONF_MAX_TOKENS: 16000,
                CONF_RUN_INPUT_TOKENS: 300_000,
                CONF_RUN_OUTPUT_TOKENS: 128_000,
                CONF_MONTH_INPUT_TOKENS: 2_000_000,
                CONF_MONTH_OUTPUT_TOKENS: 600_000,
                CONF_SCHEDULE: "off",
                CONF_SCHEDULE_WEEKDAY: "sun",
                CONF_SCHEDULE_TIME: "03:00:00",
                CONF_TRUNCATION_GUARD: True,
                CONF_VALIDATE_AFTER_LEARNING: False,
            },
        )
        await hass.async_block_till_done()
        mock_rebuild.assert_not_called()


async def test_validate_after_learning_option(
    hass: HomeAssistant,
    energy_manager: object,
    std_prefs: dict[str, Any],
) -> None:
    """The Learning step shows validate_after_learning, defaulting to off, saved as a bool."""
    from homeassistant.data_entry_flow import FlowResultType

    energy_manager.data = std_prefs
    _set_std_powers(hass)
    entry = await async_setup_via_flow(hass)

    result = await hass.config_entries.options.async_init(entry.entry_id)
    result2 = await hass.config_entries.options.async_configure(
        result["flow_id"],
        {"next_step_id": "learning"},
    )
    assert result2["type"] is FlowResultType.FORM
    assert result2["step_id"] == "learning"

    schema = result2["data_schema"]
    suggested = {
        k.schema: k.description.get("suggested_value")
        for k in schema.schema
        if isinstance(getattr(k, "description", None), dict)
    }
    assert suggested[CONF_VALIDATE_AFTER_LEARNING] is DEFAULT_VALIDATE_AFTER_LEARNING

    # The field is a required boolean selector shown after truncation_guard.
    keys = [k.schema for k in schema.schema]
    assert CONF_VALIDATE_AFTER_LEARNING in keys

    result3 = await hass.config_entries.options.async_configure(
        result2["flow_id"],
        {
            CONF_SHARE_NAMES: False,
            CONF_MAX_TOKENS: 16000,
            CONF_RUN_INPUT_TOKENS: 300_000,
            CONF_RUN_OUTPUT_TOKENS: 128_000,
            CONF_MONTH_INPUT_TOKENS: 2_000_000,
            CONF_MONTH_OUTPUT_TOKENS: 600_000,
            CONF_SCHEDULE: "off",
            CONF_SCHEDULE_WEEKDAY: "sun",
            CONF_SCHEDULE_TIME: "03:00:00",
            CONF_TRUNCATION_GUARD: True,
            CONF_VALIDATE_AFTER_LEARNING: True,
        },
    )
    assert result3["type"] is FlowResultType.CREATE_ENTRY
    assert entry.options[CONF_VALIDATE_AFTER_LEARNING] is True


async def test_learning_step_labels_cover_every_schema_key(
    hass: HomeAssistant,
    energy_manager: object,
    std_prefs: dict[str, Any],
) -> None:
    """Every Learning-step field has a label whose key equals its schema key.

    Before the fix four data keys (share_names, max_tokens, price_input,
    price_output) did not match the schema keys, so those fields rendered
    without a label.
    """
    energy_manager.data = std_prefs
    _set_std_powers(hass)
    entry = await async_setup_via_flow(hass)

    result = await hass.config_entries.options.async_init(entry.entry_id)
    result2 = await hass.config_entries.options.async_configure(
        result["flow_id"],
        {"next_step_id": "learning"},
    )
    assert result2["type"] is FlowResultType.FORM
    assert result2["step_id"] == "learning"

    schema_keys = {k.schema for k in result2["data_schema"].schema}

    strings_path = (
        Path(__file__).resolve().parents[2]
        / "custom_components"
        / "powerguesser"
        / "strings.json"
    )
    strings = json.loads(strings_path.read_text(encoding="utf-8"))
    labels = strings["options"]["step"]["learning"]["data"]

    missing = schema_keys - set(labels)
    assert missing == set(), f"Learning fields without a label: {sorted(missing)}"

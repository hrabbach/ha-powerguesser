"""Tests for the llm_auth repair flow driven through HA's repairs flow manager.

CLEAN-03 (D-17): repairs.py needs no change; the gap was the test. A direct call
to LlmAuthRepairFlow.async_step_init is vacuous (PITFALLS 29), so this file starts
the flow exactly as the HA frontend does - through hass.data["repairs"]["flow_manager"]
- against the issue a real 401 run raised.
"""

from typing import Any
import pytest

from homeassistant import data_entry_flow
from homeassistant.components.repairs import FlowType
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType
from homeassistant.helpers import issue_registry as ir
from homeassistant.setup import async_setup_component

from custom_components.powerguesser.const import DOMAIN, ISSUE_LLM_AUTH
from tests.conftest import mock_llm
from tests.ha.test_learning_loop import _setup_ready_supervisor


@pytest.mark.parametrize(
    "start_kwargs",
    [
        # Frontend path: POST /api/repairs/issues/fix -> context only, data=None
        pytest.param({"context": {"issue_id": ISSUE_LLM_AUTH}}, id="context"),
        # Legacy custom-integration path: issue_id carried in data only
        pytest.param({"data": {"issue_id": ISSUE_LLM_AUTH}}, id="data"),
    ],
)
async def test_llm_auth_fix_flow_via_repairs_manager(
    hass: HomeAssistant,
    energy_manager: Any,
    std_prefs: dict,
    aioclient_mock: Any,
    inprocess_worker: Any,
    start_kwargs: dict[str, Any],
) -> None:
    """A real 401 run raises the issue and the manager drives its fix flow end to end.

    The manager injects {"issue_id": ...} into the init step's user_input, so the
    init step must never treat its input as the user's confirmation.
    """
    entry = await _setup_ready_supervisor(hass, energy_manager, std_prefs)

    def _responder(request_data: Any) -> dict[str, Any]:
        return {"status": 401, "text": "Unauthorized key"}

    mock_llm(aioclient_mock, "http://llm.test/v1", _responder)

    supervisor = entry.runtime_data.learning
    await supervisor.async_request_run("service")
    assert supervisor.run_task is not None
    await supervisor.run_task

    # The issue came from production code, so it is fixable with entry data
    issue = ir.async_get(hass).async_get_issue(DOMAIN, ISSUE_LLM_AUTH)
    assert issue is not None
    assert issue.is_fixable is True
    assert issue.data.get("entry_id") == entry.entry_id

    assert await async_setup_component(hass, "repairs", {})
    flow_manager = hass.data["repairs"]["flow_manager"]
    result = await flow_manager.async_init(DOMAIN, **start_kwargs)

    # The confirm form is shown, nothing is opened yet, the issue stays open
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "confirm"
    assert ir.async_get(hass).async_get_issue(DOMAIN, ISSUE_LLM_AUTH) is not None

    # Submitting the (empty) confirm form aborts with next_flow pointing at options
    result2 = await flow_manager.async_configure(result["flow_id"], {})
    assert result2["type"] is FlowResultType.ABORT
    assert result2["reason"] == "open_options"
    assert result2["next_flow"][0] == FlowType.OPTIONS_FLOW

    # The options flow the abort points at is a live, registered flow
    # (async_get raises UnknownFlow when the flow id is not in progress)
    options_flow_id = result2["next_flow"][1]
    options_flow_result = hass.config_entries.options.async_get(options_flow_id)
    assert options_flow_result["flow_id"] == options_flow_id

    # ABORT keeps the issue until a successful run clears it
    assert ir.async_get(hass).async_get_issue(DOMAIN, ISSUE_LLM_AUTH) is not None


async def test_unknown_issue_id_is_rejected_by_flow_manager(
    hass: HomeAssistant,
) -> None:
    """A non-existent issue id is refused by the manager before any flow is created."""
    assert await async_setup_component(hass, "repairs", {})
    flow_manager = hass.data["repairs"]["flow_manager"]

    with pytest.raises(data_entry_flow.UnknownStep):
        await flow_manager.async_init(
            DOMAIN, context={"issue_id": "no_such_issue"}
        )

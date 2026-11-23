"""Tests for PowerGuesser candidate review flow."""

import asyncio
import json
from typing import Any
from unittest.mock import patch
import pytest

from homeassistant.components.energy.data import EnergyManager
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType
from homeassistant.helpers import entity_registry as er

from custom_components.powerguesser.const import (
    CANDIDATES_STORE_KEY,
    DOMAIN,
    LEARNING_STORES_VERSION,
    REVIEW_STORE_KEY,
    SERVICE_RUN_VALIDATION,
)
from tests.conftest import (
    async_confirm_candidate,
    async_fill_synth_buffer,
    async_open_options_step,
    async_seed_consent,
    async_seed_llm,
    async_set_power,
    async_setup_via_flow,
    mock_llm,
)
from tests.fixtures.learning.fake_llm import chat_completion
from tests.fixtures.learning.synth import SynthSpec


def _set_std_powers(hass: HomeAssistant) -> None:
    async_set_power(hass, "sensor.grid_power", 500)
    async_set_power(hass, "sensor.solar_power", 1200)
    async_set_power(hass, "sensor.battery_power", -300)
    async_set_power(hass, "sensor.washer_power", 200)


def _make_candidate(
    sig_id: str,
    label: str,
    train_share: float,
    val_share: float | None = None,
    validated: bool = True,
    fp_lo: int = 1000,
) -> dict[str, Any]:
    box = {"lo": fp_lo, "p50": fp_lo + 200, "hi": fp_lo + 400}
    dur_box = {"lo": 60, "p50": 120, "hi": 300}
    state = {
        "name": "on",
        "rise_w": box,
        "fall_w": box,
        "dur_s": dur_box,
        "plateau_w": box,
        "phase_share": None,
    }
    return {
        "schema_version": 1,
        "signature_id": sig_id,
        "kind": "switched",
        "label": label,
        "category": "appliance",
        "extraction": {"grid_s": 30, "edge_threshold_w": 120.0, "merge_samples": 2, "pair_tol": 0.25},
        "states": [state],
        "session": None,
        "time_prior": {"hour": [1 / 24] * 24, "dow": [1 / 7] * 7},
        "constraints": {"max_concurrent": 1},
        "provenance": {},
        "scores": {
            "train": {"share": train_share, "n": 20},
            "val": {"share": val_share, "n": 10} if val_share is not None else None,
            "validated": validated,
            "conflicts": 0,
        },
        "fingerprint": {
            "kind": "switched",
            "states": [state],
        },
    }


@pytest.fixture
def seeded_candidates(hass_storage: dict[str, Any]) -> list[dict[str, Any]]:
    """Seed the candidates store with three candidates (two validated, one preview)."""
    c_oven = _make_candidate("sig-oven", "Oven", 0.085, 0.079, validated=True, fp_lo=1800)
    c_kettle = _make_candidate("sig-kettle", "Kettle", 0.052, 0.048, validated=True, fp_lo=2000)
    c_toaster = _make_candidate("sig-toaster", "Toaster", 0.021, None, validated=False, fp_lo=800)

    hass_storage[CANDIDATES_STORE_KEY] = {
        "version": LEARNING_STORES_VERSION,
        "minor_version": 1,
        "key": CANDIDATES_STORE_KEY,
        "data": {
            "run_id": "run-test-1",
            "created": "2026-11-16T12:00:00Z",
            "window_end_ms": 1700000000000,
            "validated": True,
            "candidates": [c_kettle, c_oven, c_toaster],
        },
    }
    return [c_oven, c_kettle, c_toaster]


async def test_dismiss_hides_candidate_without_touching_options(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    seeded_candidates: list[dict[str, Any]],
) -> None:
    """The user opens Review candidates, picks Kettle and dismisses it.

    It disappears from the list, decision is stored in review Store, options and hub rebuild untouched.
    """
    energy_manager.data = std_prefs
    _set_std_powers(hass)
    entry = await async_setup_via_flow(hass)
    original_options = dict(entry.options)
    ent_reg = er.async_get(hass)
    entity_count_before = len(ent_reg.entities)

    # 1. Open review step from options menu
    result = await async_open_options_step(hass, entry, "review")
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "review"

    # Verify ranked by train share descending (Oven 8.5%, Kettle 5.2%, Toaster 2.1%)
    selector_opts = result["data_schema"].schema["candidate"].config["options"]
    assert len(selector_opts) == 3
    assert [opt["value"] for opt in selector_opts] == ["sig-oven", "sig-kettle", "sig-toaster"]

    # 2. Select Kettle and proceed to candidate menu
    result_cand = await hass.config_entries.options.async_configure(
        result["flow_id"], {"candidate": "sig-kettle"}
    )
    assert result_cand["type"] is FlowResultType.MENU
    assert result_cand["step_id"] == "candidate"

    with patch.object(entry.runtime_data, "async_request_rebuild") as mock_rebuild:
        # 3. Dismiss Kettle
        result_after_dismiss = await hass.config_entries.options.async_configure(
            result_cand["flow_id"], {"next_step_id": "dismiss"}
        )

        # 4. Review step is shown again with remaining two candidates
        assert result_after_dismiss["type"] is FlowResultType.FORM
        assert result_after_dismiss["step_id"] == "review"
        rem_opts = result_after_dismiss["data_schema"].schema["candidate"].config["options"]
        assert len(rem_opts) == 2
        assert [opt["value"] for opt in rem_opts] == ["sig-oven", "sig-toaster"]

        # Hub rebuild was not requested
        mock_rebuild.assert_not_called()

    # 5. Options remain unchanged
    assert entry.options == original_options

    # 6. Review Store contains dismissed record with fingerprint
    assert REVIEW_STORE_KEY in hass_storage
    decisions = hass_storage[REVIEW_STORE_KEY]["data"]["decisions"]
    assert "sig-kettle" in decisions
    assert decisions["sig-kettle"]["status"] == "dismissed"
    kettle_cand = next(c for c in seeded_candidates if c["signature_id"] == "sig-kettle")
    assert decisions["sig-kettle"]["fingerprint"] == kettle_cand["fingerprint"]

    # 7. Entity registry has no new entities
    assert len(ent_reg.entities) == entity_count_before


async def test_review_aborts_without_candidates(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
) -> None:
    """When no candidates exist, the review step aborts with no_candidates."""
    energy_manager.data = std_prefs
    _set_std_powers(hass)
    entry = await async_setup_via_flow(hass)

    result = await async_open_options_step(hass, entry, "review")
    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "no_candidates"


async def test_labels_hide_figures_for_preview_and_never_say_accuracy(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    seeded_candidates: list[dict[str, Any]],
) -> None:
    """Candidate labels never say accuracy, and preview candidate hides validation/train figures."""
    energy_manager.data = std_prefs
    _set_std_powers(hass)
    entry = await async_setup_via_flow(hass)

    result = await async_open_options_step(hass, entry, "review")
    assert result["type"] is FlowResultType.FORM
    selector_opts = result["data_schema"].schema["candidate"].config["options"]

    for opt in selector_opts:
        label_lower = opt["label"].lower()
        # The word accuracy is never used anywhere
        assert "accura" not in label_lower

    # Toaster is a preview candidate (validated: False)
    toaster_opt = next(opt for opt in selector_opts if opt["value"] == "sig-toaster")
    toaster_label_lower = toaster_opt["label"].lower()
    # Preview label hides train/validation breakdown
    assert "train" not in toaster_label_lower
    assert "preview, not yet validated" in toaster_label_lower


async def test_no_review_services_registered(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
) -> None:
    """Review services are not registered; exactly learn, preview_learning_payload
    and run_validation exist (D-04)."""
    energy_manager.data = std_prefs
    _set_std_powers(hass)
    await async_setup_via_flow(hass)

    services = hass.services.async_services().get(DOMAIN, {})
    assert set(services.keys()) == {"learn", "preview_learning_payload", SERVICE_RUN_VALIDATION}


async def test_rename_candidate_valid_and_invalid(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    seeded_candidates: list[dict[str, Any]],
) -> None:
    """The user names a candidate; whitespace is collapsed and NFC normalised; invalid names show error."""
    energy_manager.data = std_prefs
    _set_std_powers(hass)
    entry = await async_setup_via_flow(hass)

    # 1. Open review and pick Kettle
    res_review = await async_open_options_step(hass, entry, "review")
    res_cand = await hass.config_entries.options.async_configure(
        res_review["flow_id"], {"candidate": "sig-kettle"}
    )
    assert "rename" in res_cand["menu_options"]

    # 2. Open rename step
    res_rename = await hass.config_entries.options.async_configure(
        res_cand["flow_id"], {"next_step_id": "rename"}
    )
    assert res_rename["type"] is FlowResultType.FORM
    assert res_rename["step_id"] == "rename"

    # 3. Test invalid: empty string
    res_empty = await hass.config_entries.options.async_configure(
        res_rename["flow_id"], {"name": "   "}
    )
    assert res_empty["type"] is FlowResultType.FORM
    assert res_empty["errors"]["base"] == "name_invalid"

    # 4. Test invalid: control character
    res_ctrl = await hass.config_entries.options.async_configure(
        res_rename["flow_id"], {"name": "Kettle\x00"}
    )
    assert res_ctrl["type"] is FlowResultType.FORM
    assert res_ctrl["errors"]["base"] == "name_invalid"

    # 5. Test invalid: 65 characters
    res_long = await hass.config_entries.options.async_configure(
        res_rename["flow_id"], {"name": "a" * 65}
    )
    assert res_long["type"] is FlowResultType.FORM
    assert res_long["errors"]["base"] == "name_invalid"

    # 6. Test valid: "  Küche   Wasserkocher "
    res_success = await hass.config_entries.options.async_configure(
        res_rename["flow_id"], {"name": "  Küche   Wasserkocher "}
    )
    # Returns to review list
    assert res_success["type"] is FlowResultType.FORM
    assert res_success["step_id"] == "review"

    # Verify decision stored with NFC text
    assert REVIEW_STORE_KEY in hass_storage
    dec = hass_storage[REVIEW_STORE_KEY]["data"]["decisions"]["sig-kettle"]
    assert dec["status"] == "named"
    assert dec["name"] == "Küche Wasserkocher"

    # Verify label now shows the new name
    selector_opts = res_success["data_schema"].schema["candidate"].config["options"]
    kettle_opt = next(opt for opt in selector_opts if opt["value"] == "sig-kettle")
    assert kettle_opt["label"].startswith("Küche Wasserkocher — ")


async def test_rename_prefill_latest_label_with_first_seen_hint(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    seeded_candidates: list[dict[str, Any]],
) -> None:
    """The Name candidate dialog pre-fills the latest label, not the first-seen one.

    A candidate matched across learning runs keeps first_label as provenance; the
    pre-fill stays the current run's label, and the first-seen label appears as
    history in the description so both theories are visible when naming.
    """
    # The oven candidate was first seen under a different label in an earlier run
    seeded_candidates[0]["first_label"] = "Second oven / grill c11"

    energy_manager.data = std_prefs
    _set_std_powers(hass)
    entry = await async_setup_via_flow(hass)

    res_review = await async_open_options_step(hass, entry, "review")
    res_cand = await hass.config_entries.options.async_configure(
        res_review["flow_id"], {"candidate": "sig-oven"}
    )
    res_rename = await hass.config_entries.options.async_configure(
        res_cand["flow_id"], {"next_step_id": "rename"}
    )
    assert res_rename["type"] is FlowResultType.FORM
    assert res_rename["step_id"] == "rename"

    # Pre-fill is the current label, not the first-seen one
    suggested = {
        key.schema: key.description.get("suggested_value")
        for key in res_rename["data_schema"].schema
    }
    assert suggested["name"] == "Oven"

    # The first-seen label is history in the description placeholder
    first_seen = res_rename["description_placeholders"]["first_seen"]
    assert "Second oven / grill c11" in first_seen


async def test_merge_candidates(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    seeded_candidates: list[dict[str, Any]],
) -> None:
    """The user merges B and C into A; description explains scores update at next run; B and C disappear."""
    energy_manager.data = std_prefs
    _set_std_powers(hass)
    entry = await async_setup_via_flow(hass)

    # 1. Open review and pick Oven ("sig-oven")
    res_review = await async_open_options_step(hass, entry, "review")
    res_cand = await hass.config_entries.options.async_configure(
        res_review["flow_id"], {"candidate": "sig-oven"}
    )
    assert res_cand["menu_options"] == ["confirm", "rename", "merge", "dismiss"]

    # 2. Open merge step
    res_merge = await hass.config_entries.options.async_configure(
        res_cand["flow_id"], {"next_step_id": "merge"}
    )
    assert res_merge["type"] is FlowResultType.FORM
    assert res_merge["step_id"] == "merge"

    # Description explains score updates at the next learning run
    desc = res_merge["description_placeholders"]["candidate"]
    assert desc  # placeholder exists

    # Options offer Kettle and Toaster (other visible candidates)
    src_opts = res_merge["data_schema"].schema["sources"].config["options"]
    assert len(src_opts) == 2
    assert {opt["value"] for opt in src_opts} == {"sig-kettle", "sig-toaster"}

    # 3. Merge both into Oven
    res_after = await hass.config_entries.options.async_configure(
        res_merge["flow_id"], {"sources": ["sig-kettle", "sig-toaster"]}
    )
    assert res_after["type"] is FlowResultType.FORM
    assert res_after["step_id"] == "review"

    # Now only Oven remains visible
    rem_opts = res_after["data_schema"].schema["candidate"].config["options"]
    assert len(rem_opts) == 1
    assert rem_opts[0]["value"] == "sig-oven"

    # Check review decisions
    decisions = hass_storage[REVIEW_STORE_KEY]["data"]["decisions"]
    assert decisions["sig-oven"]["status"] == "kept"
    assert decisions["sig-oven"]["aliases"] == ["sig-kettle", "sig-toaster"]
    assert decisions["sig-kettle"]["status"] == "merged"
    assert decisions["sig-kettle"]["merged_into"] == "sig-oven"
    assert decisions["sig-toaster"]["status"] == "merged"
    assert decisions["sig-toaster"]["merged_into"] == "sig-oven"


async def test_merge_target_label_shows_merge_count(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    seeded_candidates: list[dict[str, Any]],
) -> None:
    """Merging two sources into a target shows "(+2 merged)" in Review and stores the plain label (D-18)."""
    energy_manager.data = std_prefs
    _set_std_powers(hass)
    entry = await async_setup_via_flow(hass)

    # Merge Kettle and Toaster into Oven
    res_review = await async_open_options_step(hass, entry, "review")
    res_cand = await hass.config_entries.options.async_configure(
        res_review["flow_id"], {"candidate": "sig-oven"}
    )
    res_merge = await hass.config_entries.options.async_configure(
        res_cand["flow_id"], {"next_step_id": "merge"}
    )
    res_after = await hass.config_entries.options.async_configure(
        res_merge["flow_id"], {"sources": ["sig-kettle", "sig-toaster"]}
    )
    assert res_after["type"] is FlowResultType.FORM
    assert res_after["step_id"] == "review"

    # The decision record stores the display name without the merge suffix
    decisions = hass_storage[REVIEW_STORE_KEY]["data"]["decisions"]
    assert decisions["sig-oven"]["aliases"] == ["sig-kettle", "sig-toaster"]
    assert decisions["sig-oven"]["label"] == "Oven"

    # The review list shows the merge info
    rem_opts = res_after["data_schema"].schema["candidate"].config["options"]
    assert len(rem_opts) == 1
    assert rem_opts[0]["label"].startswith("Oven (+2 merged) — ")


async def test_rename_keeps_unknown_record_keys(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    seeded_candidates: list[dict[str, Any]],
) -> None:
    """A rename keeps unknown record keys (e.g. a future device link) on the review record."""
    hass_storage[REVIEW_STORE_KEY] = {
        "version": LEARNING_STORES_VERSION,
        "minor_version": 1,
        "key": REVIEW_STORE_KEY,
        "data": {
            "decisions": {
                "sig-kettle": {
                    "status": "named",
                    "name": "Kettle",
                    "merged_into": None,
                    "aliases": [],
                    "decided_at": "2026-11-16T12:00:00Z",
                    "fingerprint": {},
                    "device_uuid": "u1",
                }
            }
        },
    }
    energy_manager.data = std_prefs
    _set_std_powers(hass)
    entry = await async_setup_via_flow(hass)

    res_review = await async_open_options_step(hass, entry, "review")
    res_cand = await hass.config_entries.options.async_configure(
        res_review["flow_id"], {"candidate": "sig-kettle"}
    )
    res_rename = await hass.config_entries.options.async_configure(
        res_cand["flow_id"], {"next_step_id": "rename"}
    )
    await hass.config_entries.options.async_configure(
        res_rename["flow_id"], {"name": "Wasserkocher"}
    )

    dec = hass_storage[REVIEW_STORE_KEY]["data"]["decisions"]["sig-kettle"]
    assert dec["name"] == "Wasserkocher"
    assert dec["label"] == "Wasserkocher"
    # Unknown key from a later schema version survives the rewrite
    assert dec["device_uuid"] == "u1"


async def test_candidate_menu_omits_merge_when_single_candidate(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
) -> None:
    """When exactly one visible candidate exists, merge is omitted from the candidate menu."""
    c_single = _make_candidate("sig-single", "SingleDevice", 0.05, 0.04)
    hass_storage[CANDIDATES_STORE_KEY] = {
        "version": LEARNING_STORES_VERSION,
        "minor_version": 1,
        "key": CANDIDATES_STORE_KEY,
        "data": {
            "run_id": "run-test-single",
            "created": "2026-11-16T12:00:00Z",
            "window_end_ms": 1700000000000,
            "validated": True,
            "candidates": [c_single],
        },
    }
    energy_manager.data = std_prefs
    _set_std_powers(hass)
    entry = await async_setup_via_flow(hass)

    res_review = await async_open_options_step(hass, entry, "review")
    res_cand = await hass.config_entries.options.async_configure(
        res_review["flow_id"], {"candidate": "sig-single"}
    )
    assert res_cand["type"] is FlowResultType.MENU
    # Menu offers Confirm first for switched candidates, not merge (single candidate)
    assert res_cand["menu_options"] == ["confirm", "rename", "dismiss"]


async def test_ordering_tie_break_by_signature_id(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
) -> None:
    """Equal shares are ordered deterministically by signature id ascending."""
    c_b = _make_candidate("sig-b", "Device B", 0.05, 0.05)
    c_a = _make_candidate("sig-a", "Device A", 0.05, 0.05)
    hass_storage[CANDIDATES_STORE_KEY] = {
        "version": LEARNING_STORES_VERSION,
        "minor_version": 1,
        "key": CANDIDATES_STORE_KEY,
        "data": {
            "run_id": "run-test-order",
            "created": "2026-11-16T12:00:00Z",
            "window_end_ms": 1700000000000,
            "validated": True,
            "candidates": [c_b, c_a],
        },
    }
    energy_manager.data = std_prefs
    _set_std_powers(hass)
    entry = await async_setup_via_flow(hass)

    res_review = await async_open_options_step(hass, entry, "review")
    opts = res_review["data_schema"].schema["candidate"].config["options"]
    assert [opt["value"] for opt in opts] == ["sig-a", "sig-b"]


async def test_unconfirmed_merge_sources_exclude_confirmed_devices(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
) -> None:
    """D-23 (C3): an unconfirmed candidate's merge source selector lists other
    visible candidates only — never a confirmed device entry."""
    c_kettle = _make_candidate("sig-kettle", "Kettle", 0.052, 0.048, validated=True, fp_lo=2000)
    c_oven = _make_candidate("sig-oven", "Oven", 0.085, 0.079, validated=True, fp_lo=1800)
    c_toaster = _make_candidate("sig-toaster", "Toaster", 0.021, None, validated=False, fp_lo=800)
    hass_storage[CANDIDATES_STORE_KEY] = {
        "version": LEARNING_STORES_VERSION,
        "minor_version": 1,
        "key": CANDIDATES_STORE_KEY,
        "data": {
            "run_id": "run-test-merge-c3",
            "created": "2026-11-16T12:00:00Z",
            "window_end_ms": 1700000000000,
            "validated": True,
            "candidates": [c_kettle, c_oven, c_toaster],
        },
    }
    energy_manager.data = std_prefs
    _set_std_powers(hass)
    entry = await async_setup_via_flow(hass)

    # Confirm the kettle: it becomes a device and leaves the candidate list.
    await async_confirm_candidate(hass, entry, "sig-kettle", "Kettle")
    sub = next(s for s in entry.get_subentries_of_type("device") if s.title == "Kettle")
    dev_uuid = sub.data["device_uuid"]

    # The oven candidate's merge selector never offers the confirmed device.
    res_review = await async_open_options_step(hass, entry, "review")
    res_cand = await hass.config_entries.options.async_configure(
        res_review["flow_id"], {"candidate": "sig-oven"}
    )
    assert res_cand["type"] is FlowResultType.MENU
    assert "merge" in res_cand["menu_options"]
    res_merge = await hass.config_entries.options.async_configure(
        res_cand["flow_id"], {"next_step_id": "merge"}
    )
    assert res_merge["type"] is FlowResultType.FORM
    assert res_merge["step_id"] == "merge"
    src_opts = res_merge["data_schema"].schema["sources"].config["options"]
    values = [o["value"] for o in src_opts]
    assert values == ["sig-toaster"]
    assert f"device_{dev_uuid}" not in values
    assert "sig-kettle" not in values


async def test_stale_candidate_mid_review_gives_candidate_gone(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    seeded_candidates: list[dict[str, Any]],
) -> None:
    """When candidates are replaced between opening review and submit, candidate_gone is shown."""
    energy_manager.data = std_prefs
    _set_std_powers(hass)
    entry = await async_setup_via_flow(hass)

    # 1. Open review and pick Kettle
    res_review = await async_open_options_step(hass, entry, "review")
    res_cand = await hass.config_entries.options.async_configure(
        res_review["flow_id"], {"candidate": "sig-kettle"}
    )
    assert res_cand["type"] is FlowResultType.MENU

    # 2. Simulate background replacement of candidates (Kettle is gone)
    entry.runtime_data.learning.candidates = [
        c for c in seeded_candidates if c["signature_id"] != "sig-kettle"
    ]

    # 3. Submit dismiss
    res_submit = await hass.config_entries.options.async_configure(
        res_cand["flow_id"], {"next_step_id": "dismiss"}
    )
    # Shows review form with candidate_gone error
    assert res_submit["type"] is FlowResultType.FORM
    assert res_submit["step_id"] == "review"
    assert res_submit["errors"]["base"] == "candidate_gone"

    # Review store has no decision for Kettle
    decisions = hass_storage.get(REVIEW_STORE_KEY, {}).get("data", {}).get("decisions", {})
    assert "sig-kettle" not in decisions


async def test_run_finishing_mid_review_gives_candidate_gone(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    aioclient_mock: Any,
    inprocess_worker: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A real in-flight run finishing mid-review causes candidate_gone on submit."""
    monkeypatch.setattr("custom_components.powerguesser.learning.MAX_REFINE_ROUNDS", 0)

    # Pre-seed with an old candidate (distinct power level so new run doesn't match it)
    c_old = _make_candidate("sig-old-candidate", "Old Device", 0.05, 0.04, fp_lo=10000)
    hass_storage[CANDIDATES_STORE_KEY] = {
        "version": LEARNING_STORES_VERSION,
        "minor_version": 1,
        "key": CANDIDATES_STORE_KEY,
        "data": {
            "run_id": "run-initial",
            "created": "2026-11-16T12:00:00Z",
            "window_end_ms": 1700000000000,
            "validated": True,
            "candidates": [c_old],
        },
    }

    energy_manager.data = std_prefs
    _set_std_powers(hass)
    entry = await async_setup_via_flow(hass)
    spec = SynthSpec(days=22, oven=True)
    await async_fill_synth_buffer(hass, entry, spec=spec)
    await async_seed_llm(hass, entry)
    await async_seed_consent(hass, host="llm.test", names=False)

    # 1. User opens review and selects sig-old-candidate
    res_review = await async_open_options_step(hass, entry, "review")
    res_cand = await hass.config_entries.options.async_configure(
        res_review["flow_id"], {"candidate": "sig-old-candidate"}
    )
    assert res_cand["type"] is FlowResultType.MENU
    assert res_cand["step_id"] == "candidate"

    # 2. Set up responder held on an asyncio.Event for in-flight run
    hold_event = asyncio.Event()

    async def _responder(req_data: Any) -> dict[str, Any]:
        await hold_event.wait()
        # Returns a different simulated proposal
        prop = {
            "analysis": "New proposal.",
            "schema_version": 1,
            "devices": [
                {
                    "ref": "d1",
                    "label": "New Oven",
                    "category": "oven",
                    "tracker_kind": "switched",
                    "member_clusters": ["c01"],
                    "states": [{"name": "on", "cluster_ref": "c01", "role": "main"}],
                    "phase_hint": "single",
                    "confidence": "high",
                    "rationale": "Oven",
                    "ambiguous_with": [],
                    "reference_plug": "",
                }
            ],
            "rejected_clusters": [],
            "merge_suggestions": [],
        }
        return {"status": 200, "json": chat_completion(json.dumps(prop))}

    mock_llm(aioclient_mock, "http://llm.test/v1", _responder)

    # 3. Start learning run while user is in candidate menu
    sup = entry.runtime_data.learning
    await sup.async_request_run("service")
    assert sup.run_task is not None

    # Release the run and await completion
    hold_event.set()
    await sup.run_task

    # Verify run produced new candidates and sig-old-candidate is gone from candidates
    assert sup.candidates
    assert not any(c.get("signature_id") == "sig-old-candidate" for c in sup.candidates)

    # 4. User submits dismiss on the old candidate
    res_submit = await hass.config_entries.options.async_configure(
        res_cand["flow_id"], {"next_step_id": "dismiss"}
    )

    # 5. Form shows candidate_gone and review store has no decision for sig-old-candidate
    assert res_submit["type"] is FlowResultType.FORM
    assert res_submit["step_id"] == "review"
    assert res_submit["errors"]["base"] == "candidate_gone"

    decisions = hass_storage.get(REVIEW_STORE_KEY, {}).get("data", {}).get("decisions", {})
    assert "sig-old-candidate" not in decisions

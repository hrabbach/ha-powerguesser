"""Tests for candidate identity preservation, dismissals, and merges across learning runs."""

import json
from typing import Any
import pytest

from homeassistant.components.energy.data import EnergyManager
from homeassistant.core import HomeAssistant

from custom_components.powerguesser.const import REVIEW_STORE_KEY
from tests.conftest import (
    async_fill_synth_buffer,
    async_open_options_step,
    async_seed_consent,
    async_seed_llm,
    async_set_power,
    async_setup_via_flow,
    mock_llm,
    switched_cluster_ids,
)
from tests.fixtures.learning.fake_llm import chat_completion
from tests.fixtures.learning.synth import SynthSpec


def _set_std_powers(hass: HomeAssistant) -> None:
    async_set_power(hass, "sensor.grid_power", 500)
    async_set_power(hass, "sensor.solar_power", 1200)
    async_set_power(hass, "sensor.battery_power", -300)
    async_set_power(hass, "sensor.washer_power", 200)


def _make_proposal(devices: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "analysis": "Test proposal.",
        "schema_version": 1,
        "devices": devices,
        "rejected_clusters": [],
        "merge_suggestions": [],
    }


def _make_device(ref: str, cluster: str, label: str | None = None) -> dict[str, Any]:
    return {
        "ref": ref,
        "label": label or f"Device {ref}",
        "category": "unknown",
        "tracker_kind": "switched",
        "member_clusters": [cluster],
        "states": [
            {
                "name": "on",
                "cluster_ref": cluster,
                "role": "main",
            }
        ],
        "phase_hint": "single",
        "confidence": "high",
        "rationale": f"Device for {cluster}.",
        "ambiguous_with": [],
        "reference_plug": "",
    }


async def _setup_learning_env(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> Any:
    monkeypatch.setattr("custom_components.powerguesser.learning.MAX_REFINE_ROUNDS", 0)
    energy_manager.data = std_prefs
    _set_std_powers(hass)
    entry = await async_setup_via_flow(hass)
    spec = SynthSpec(days=22, oven=True)
    await async_fill_synth_buffer(hass, entry, spec=spec)
    await async_seed_llm(hass, entry)
    await async_seed_consent(hass, host="llm.test", names=False)
    return entry


async def test_dismissed_candidate_stays_dismissed_in_next_run(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    aioclient_mock: Any,
    inprocess_worker: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run 1 yields candidates; user dismisses X; Run 2 suppresses X and pending_candidates drops."""
    entry = await _setup_learning_env(hass, energy_manager, std_prefs, monkeypatch)

    def _responder(data: Any) -> dict[str, Any]:
        sw = switched_cluster_ids(data)
        prop = _make_proposal([_make_device("d1", sw[0]), _make_device("d2", sw[1])])
        return {"status": 200, "json": chat_completion(json.dumps(prop))}

    mock_llm(aioclient_mock, "http://llm.test/v1", _responder)

    sup = entry.runtime_data.learning

    # --- Run 1 ---
    await sup.async_request_run("service")
    assert sup.run_task is not None
    await sup.run_task

    assert len(sup.candidates) == 2
    assert sup.coordinator.data["pending_candidates"] == 2

    cands_run1 = list(sup.visible_candidates())
    assert len(cands_run1) == 2
    sig_dismiss = cands_run1[0]["signature_id"]
    sig_keep = cands_run1[1]["signature_id"]

    # User dismisses first candidate
    res_review = await async_open_options_step(hass, entry, "review")
    res_cand = await hass.config_entries.options.async_configure(
        res_review["flow_id"], {"candidate": sig_dismiss}
    )
    await hass.config_entries.options.async_configure(
        res_cand["flow_id"], {"next_step_id": "dismiss"}
    )

    # After dismissal, 1 candidate remains visible
    assert len(sup.visible_candidates()) == 1
    assert sup.coordinator.data["pending_candidates"] == 1

    # --- Run 2 on same buffer and proposal ---
    await sup.async_request_run("service")
    assert sup.run_task is not None
    await sup.run_task

    # Visible candidates has only 1, matching the non-dismissed one
    visible_run2 = sup.visible_candidates()
    assert len(visible_run2) == 1
    assert visible_run2[0]["signature_id"] == sig_keep
    assert sup.coordinator.data["pending_candidates"] == 1

    # Stored candidates contains the suppressed artefact for X
    suppressed_cands = [
        c for c in sup.candidates if c.get("identity", {}).get("status") == "suppressed"
    ]
    assert len(suppressed_cands) == 1
    assert suppressed_cands[0]["signature_id"] == sig_dismiss


async def test_matched_candidate_keeps_first_label_across_runs(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    aioclient_mock: Any,
    inprocess_worker: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run 1 labels the kettle cluster "Kettle A"; Run 2's LLM says "Water boiler".

    The matched candidate keeps first_label "Kettle A" as provenance, and the
    review label shows the latest label "Water boiler" — display follows the
    run's best current guess, not the first one (D-18 display precedence
    superseded; first_label carry-over is unchanged).
    """
    entry = await _setup_learning_env(hass, energy_manager, std_prefs, monkeypatch)

    call_count = 0

    def _responder(data: Any) -> dict[str, Any]:
        nonlocal call_count
        call_count += 1
        sw = switched_cluster_ids(data)
        label = "Kettle A" if call_count == 1 else "Water boiler"
        prop = _make_proposal([_make_device("d1", sw[0], label=label)])
        return {"status": 200, "json": chat_completion(json.dumps(prop))}

    mock_llm(aioclient_mock, "http://llm.test/v1", _responder)
    sup = entry.runtime_data.learning

    # --- Run 1 ---
    await sup.async_request_run("service")
    assert sup.run_task is not None
    await sup.run_task

    cand_run1 = sup.visible_candidates()[0]
    assert cand_run1["label"] == "Kettle A"

    # --- Run 2: the LLM now labels the same cluster differently ---
    await sup.async_request_run("service")
    assert sup.run_task is not None
    await sup.run_task

    cand_run2 = sup.visible_candidates()[0]
    assert cand_run2["signature_id"] == cand_run1["signature_id"]
    assert cand_run2["identity"]["status"] == "matched"
    # The fresh LLM label is stored on the artefact but first_label carries over
    assert cand_run2["label"] == "Water boiler"
    assert cand_run2["first_label"] == "Kettle A"
    # The review list shows the latest label; first_label is history only
    assert sup.candidate_label(cand_run2).startswith("Water boiler — ")


async def test_named_candidate_label_survives_label_change(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    aioclient_mock: Any,
    inprocess_worker: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A user-given name beats both the first label and a fresh LLM label (D-18)."""
    entry = await _setup_learning_env(hass, energy_manager, std_prefs, monkeypatch)

    call_count = 0

    def _responder(data: Any) -> dict[str, Any]:
        nonlocal call_count
        call_count += 1
        sw = switched_cluster_ids(data)
        label = "Kettle A" if call_count == 1 else "Water boiler"
        prop = _make_proposal([_make_device("d1", sw[0], label=label)])
        return {"status": 200, "json": chat_completion(json.dumps(prop))}

    mock_llm(aioclient_mock, "http://llm.test/v1", _responder)
    sup = entry.runtime_data.learning

    # --- Run 1 ---
    await sup.async_request_run("service")
    assert sup.run_task is not None
    await sup.run_task

    cand = sup.visible_candidates()[0]
    sig_id = cand["signature_id"]

    # User names it "Wasserkocher"
    res_review = await async_open_options_step(hass, entry, "review")
    res_cand = await hass.config_entries.options.async_configure(
        res_review["flow_id"], {"candidate": sig_id}
    )
    res_rename = await hass.config_entries.options.async_configure(
        res_cand["flow_id"], {"next_step_id": "rename"}
    )
    await hass.config_entries.options.async_configure(
        res_rename["flow_id"], {"name": "Wasserkocher"}
    )

    # --- Run 2: the LLM changes the label again ---
    await sup.async_request_run("service")
    assert sup.run_task is not None
    await sup.run_task

    cand_run2 = sup.visible_candidates()[0]
    assert cand_run2["first_label"] == "Kettle A"
    # The decision name wins over first_label and the fresh LLM label
    assert sup.candidate_label(cand_run2).startswith("Wasserkocher — ")


async def test_first_label_precedence(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    aioclient_mock: Any,
    inprocess_worker: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """first_label precedence: prior first_label > prior label > decision label > own label."""
    entry = await _setup_learning_env(hass, energy_manager, std_prefs, monkeypatch)
    mock_llm(aioclient_mock, "http://llm.test/v1", lambda req: {"status": 200, "json": {}})
    sup = entry.runtime_data.learning

    # Review decision record carrying the label shown at decision time
    await sup.review.async_dismiss("sig-dec", None, label="C")

    # Stored candidates from the previous run
    sup.candidates = [
        {"signature_id": "sig-a", "label": "B", "first_label": "A"},
        {"signature_id": "sig-b", "label": "B"},
    ]

    library = [
        {"signature_id": "sig-a", "label": "X"},
        {"signature_id": "sig-b", "label": "X"},
        {"signature_id": "sig-dec", "label": "X"},
        {"signature_id": "sig-new", "label": "D"},
    ]
    sup._apply_first_labels(library)

    assert library[0]["first_label"] == "A"  # prior first_label wins
    assert library[1]["first_label"] == "B"  # else prior label
    assert library[2]["first_label"] == "C"  # else decision record label
    assert library[3]["first_label"] == "D"  # else the artefact's own label


async def test_near_twin_of_dismissed_stays_hidden(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    aioclient_mock: Any,
    inprocess_worker: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run 2's super-cluster proposal (a strict claims superset of the dismissed candidate, Jaccard < 0.5) is hidden as a near twin (D-19)."""
    entry = await _setup_learning_env(hass, energy_manager, std_prefs, monkeypatch)

    call_count = 0

    def _responder(data: Any) -> dict[str, Any]:
        nonlocal call_count
        call_count += 1
        sw = switched_cluster_ids(data)
        if call_count == 1:
            # Run 1: a candidate on the smaller switched cluster (oven pulses).
            prop = _make_proposal([_make_device("d1", sw[1])])
        else:
            # Run 2: one device over ALL switched clusters — its claims strictly
            # contain the dismissed candidate's (overlap coefficient 1.0) while the
            # one-to-one pass sees only Jaccard 38/129 < 0.5 and does not match.
            # Each state's boxes are refit from its own cluster_ref, so the union
            # needs one state per cluster to claim both episode sets.
            dev = _make_device("d2", sw[0])
            dev["member_clusters"] = list(sw)
            state_b = dict(dev["states"][0])
            state_b["cluster_ref"] = sw[1]
            dev["states"] = [dev["states"][0], state_b]
            prop = _make_proposal([dev])
        return {"status": 200, "json": chat_completion(json.dumps(prop))}

    mock_llm(aioclient_mock, "http://llm.test/v1", _responder)
    sup = entry.runtime_data.learning

    # --- Run 1 ---
    await sup.async_request_run("service")
    assert sup.run_task is not None
    await sup.run_task

    cand_x = sup.visible_candidates()[0]
    sig_x = cand_x["signature_id"]

    # User dismisses X
    res_review = await async_open_options_step(hass, entry, "review")
    res_cand = await hass.config_entries.options.async_configure(
        res_review["flow_id"], {"candidate": sig_x}
    )
    await hass.config_entries.options.async_configure(
        res_cand["flow_id"], {"next_step_id": "dismiss"}
    )

    # --- Run 2: the same load re-proposed under a larger cluster set ---
    await sup.async_request_run("service")
    assert sup.run_task is not None
    await sup.run_task

    # The near twin is not listed and pending_candidates excludes it.
    assert sup.visible_candidates() == []
    assert sup.coordinator.data["pending_candidates"] == 0

    suppressed = [
        c for c in sup.candidates if c.get("identity", {}).get("status") == "suppressed"
    ]
    assert len(suppressed) == 1
    assert suppressed[0]["identity"]["matched"] is None
    assert suppressed[0]["identity"]["twin_of"] == sig_x


async def test_named_candidate_carries_over_in_next_run(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    aioclient_mock: Any,
    inprocess_worker: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A candidate named Wasserkocher in Run 1 keeps its id and label in Run 2."""
    entry = await _setup_learning_env(hass, energy_manager, std_prefs, monkeypatch)

    prop = _make_proposal([_make_device("d1", "c01")])

    def _responder(_data: Any) -> dict[str, Any]:
        return {"status": 200, "json": chat_completion(json.dumps(prop))}

    mock_llm(aioclient_mock, "http://llm.test/v1", _responder)
    sup = entry.runtime_data.learning

    # --- Run 1 ---
    await sup.async_request_run("service")
    assert sup.run_task is not None
    await sup.run_task

    cand = sup.visible_candidates()[0]
    sig_id = cand["signature_id"]

    # User names it "Wasserkocher"
    res_review = await async_open_options_step(hass, entry, "review")
    res_cand = await hass.config_entries.options.async_configure(
        res_review["flow_id"], {"candidate": sig_id}
    )
    res_rename = await hass.config_entries.options.async_configure(
        res_cand["flow_id"], {"next_step_id": "rename"}
    )
    await hass.config_entries.options.async_configure(
        res_rename["flow_id"], {"name": "Wasserkocher"}
    )

    # --- Run 2 ---
    await sup.async_request_run("service")
    assert sup.run_task is not None
    await sup.run_task

    visible_run2 = sup.visible_candidates()
    assert len(visible_run2) == 1
    assert visible_run2[0]["signature_id"] == sig_id

    # Label reflects the name
    label = sup.candidate_label(visible_run2[0])
    assert label.startswith("Wasserkocher — ")


async def test_merged_candidates_carry_over_in_next_run(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    aioclient_mock: Any,
    inprocess_worker: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Candidates B merged into A in Run 1: in Run 2, B is merged into A and only A is listed."""
    entry = await _setup_learning_env(hass, energy_manager, std_prefs, monkeypatch)

    def _responder(data: Any) -> dict[str, Any]:
        sw = switched_cluster_ids(data)
        prop = _make_proposal([_make_device("d1", sw[0]), _make_device("d2", sw[1])])
        return {"status": 200, "json": chat_completion(json.dumps(prop))}

    mock_llm(aioclient_mock, "http://llm.test/v1", _responder)
    sup = entry.runtime_data.learning

    # --- Run 1 ---
    await sup.async_request_run("service")
    assert sup.run_task is not None
    await sup.run_task

    visible1 = sup.visible_candidates()
    cand_a = visible1[0]
    cand_b = visible1[1]
    sig_a = cand_a["signature_id"]
    sig_b = cand_b["signature_id"]

    # Merge B into A
    res_review = await async_open_options_step(hass, entry, "review")
    res_cand = await hass.config_entries.options.async_configure(
        res_review["flow_id"], {"candidate": sig_a}
    )
    res_merge = await hass.config_entries.options.async_configure(
        res_cand["flow_id"], {"next_step_id": "merge"}
    )
    await hass.config_entries.options.async_configure(
        res_merge["flow_id"], {"sources": [sig_b]}
    )

    # --- Run 2 ---
    await sup.async_request_run("service")
    assert sup.run_task is not None
    await sup.run_task

    # Only A is visible
    visible2 = sup.visible_candidates()
    assert len(visible2) == 1
    assert visible2[0]["signature_id"] == sig_a

    # B in stored candidates carries identity.status == "merged" with merged_into == sig_a
    b_artefacts = [c for c in sup.candidates if c.get("signature_id") == sig_b]
    assert len(b_artefacts) == 1
    assert b_artefacts[0]["identity"]["status"] == "merged"
    assert b_artefacts[0]["identity"]["merged_into"] == sig_a


async def test_unmatched_artefact_gets_new_id_and_is_listed(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    aioclient_mock: Any,
    inprocess_worker: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An artefact in Run 2 that matches nothing from Run 1 gets status new and is listed."""
    entry = await _setup_learning_env(hass, energy_manager, std_prefs, monkeypatch)

    call_count = 0

    def _responder(data: Any) -> dict[str, Any]:
        nonlocal call_count
        call_count += 1
        sw = switched_cluster_ids(data)
        if call_count == 1:
            prop = _make_proposal([_make_device("d1", sw[0])])
        else:
            prop = _make_proposal([_make_device("d2", sw[1])])
        return {"status": 200, "json": chat_completion(json.dumps(prop))}

    mock_llm(aioclient_mock, "http://llm.test/v1", _responder)
    sup = entry.runtime_data.learning

    # --- Run 1: device d1 ---
    await sup.async_request_run("service")
    assert sup.run_task is not None
    await sup.run_task
    sig_1 = sup.visible_candidates()[0]["signature_id"]

    # --- Run 2: device d2 (completely different cluster) ---
    await sup.async_request_run("service")
    assert sup.run_task is not None
    await sup.run_task

    visible2 = sup.visible_candidates()
    assert len(visible2) == 1
    new_cand = visible2[0]
    assert new_cand["signature_id"] != sig_1
    assert new_cand["identity"]["status"] == "new"


async def test_learning_runs_never_modify_review_store(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    aioclient_mock: Any,
    inprocess_worker: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Learning runs never write or alter the review Store."""
    entry = await _setup_learning_env(hass, energy_manager, std_prefs, monkeypatch)

    prop = _make_proposal([_make_device("d1", "c01")])

    def _responder(_data: Any) -> dict[str, Any]:
        return {"status": 200, "json": chat_completion(json.dumps(prop))}

    mock_llm(aioclient_mock, "http://llm.test/v1", _responder)
    sup = entry.runtime_data.learning

    # Seed a decision directly in the review store
    await sup.review.async_set_name("sig-custom", "Custom Device", {"states": []})
    expected_decisions = dict(hass_storage[REVIEW_STORE_KEY]["data"]["decisions"])

    # Run learning
    await sup.async_request_run("service")
    assert sup.run_task is not None
    await sup.run_task

    # Review store in storage is unchanged
    actual_decisions = hass_storage[REVIEW_STORE_KEY]["data"]["decisions"]
    assert actual_decisions == expected_decisions

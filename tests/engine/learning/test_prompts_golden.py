"""Capture-free byte-identity golden for the LLM request builders (D-06).

Recorded from the unchanged ``prompts.py`` before any production edit for Phase 9.
The snapshot is frozen: it proves that with no examples every request body stays
byte-identical to the pre-capture behaviour — all four structured-output
strategies with names off and on, the refinement turn, and the repair turn.

Never regenerate this snapshot afterwards (Phase 4/8 golden pattern). A
deliberate change to a capture-free body must be argued in the plan and the
golden updated in its own commit.
"""

import pytest
from syrupy.extensions.amber import AmberSnapshotExtension


@pytest.fixture
def snapshot(snapshot):
    return snapshot.use_extension(AmberSnapshotExtension)


from custom_components.powerguesser.engine.learning.prompts import (
    build_refine_request,
    build_repair_request,
    build_request,
)
from custom_components.powerguesser.engine.llm_wire import Strategy


DIGEST = {
    "header": {
        "grid_s": 30,
        "days": 21,
        "residual_kwh": 210.5,
        "clustered_share": 0.73,
        "edge_threshold_w": 120,
        "phases_mapped": True,
        "plugs": 2,
    },
    "clusters": [
        {
            "id": "c01",
            "kind": "switched",
            "n": 42,
            "rise_w": {"p10": 1960, "p50": 2010, "p90": 2060},
            "fall_w": {"p10": -2060, "p50": -2010, "p90": -1960},
            "dur_min": {"p10": 1.5, "p50": 2.5, "p90": 3.5},
            "phase": {"l1": 0.9, "l2": 0.05, "l3": 0.05},
            "hour": [0] * 24,
            "dow": [0] * 7,
            "energy_share": 0.24,
            "conc_with": [{"id": "c02", "share": 0.31}],
        },
        {
            "id": "c02",
            "kind": "session",
            "n": 12,
            "rise_w": {"p10": 2100, "p50": 2200, "p90": 2300},
            "fall_w": {"p10": -2300, "p50": -2200, "p90": -2100},
            "dur_min": {"p10": 28.0, "p50": 34.0, "p90": 41.0},
            "hour": [0] * 24,
            "dow": [0] * 7,
            "energy_share": 0.41,
            "conc_with": [],
            "exemplars": ["+2200x5 -2200x2 +2180x3"],
        },
    ],
    "plugs": [
        {
            "id": "p1",
            "n": 10,
            "energy_kwh_per_day": 0.8,
            "dur_min": {"p10": 30.0, "p50": 45.0, "p90": 60.0},
            "peak_w": 2200,
            "hour": [0] * 24,
            "dow": [0] * 7,
        },
        {
            "id": "p2",
            "n": 6,
            "energy_kwh_per_day": 1.6,
            "dur_min": {"p10": 40.0, "p50": 55.0, "p90": 70.0},
            "peak_w": 2100,
            "hour": [0] * 24,
            "dow": [0] * 7,
            "exemplars": ["+2100x8 -2100x1"],
        },
    ],
}

NAMES = {"p1": "Kettle", "p2": "Washing machine"}

PREVIOUS = {
    "analysis": 'first pass, a "quoted" line\nsecond line',
    "schema_version": 1,
    "devices": [],
    "rejected_clusters": [],
    "merge_suggestions": [],
}

SCORECARD = {"devices": [{"ref": "d1", "train_share": 0.5, "energy_error_wh": 12.0}]}


def _profile(strategy: Strategy) -> dict:
    return {
        "strategy": strategy.value,
        "token_param": "max_completion_tokens",
        "send_temperature": True,
    }


def test_capture_free_bodies_match_golden(snapshot):
    """Every request body with no examples equals the golden recorded before Phase 9."""
    bodies: dict[str, object] = {}

    for strat in (
        Strategy.JSON_SCHEMA,
        Strategy.TOOL_CALL,
        Strategy.JSON_OBJECT,
        Strategy.FENCED,
    ):
        for label, names in (("none", None), ("names", NAMES)):
            bodies[f"build_request_{strat.value}_{label}"] = build_request(
                DIGEST,
                model="test-model",
                profile=_profile(strat),
                max_tokens=4096,
                names=names,
            )

    bodies["build_refine_request_names"] = build_refine_request(
        model="test-model",
        profile=_profile(Strategy.JSON_SCHEMA),
        max_tokens=4096,
        digest=DIGEST,
        previous=PREVIOUS,
        scorecard=SCORECARD,
        names=NAMES,
    )

    bodies["build_repair_request"] = build_repair_request(
        model="test-model",
        profile=_profile(Strategy.JSON_SCHEMA),
        max_tokens=4096,
        cluster_ids=["c01", "c02"],
        plug_ids=["p1", "p2"],
        invalid_output='{"analysis": "truncated',
        errors=["/devices: expected array", "/analysis: missing required property"],
    )

    assert bodies == snapshot

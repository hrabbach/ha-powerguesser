"""Tests for the shadow extract calendar holdout and the shadow_link stage (Plan 05-03)."""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from custom_components.powerguesser.engine.learning.split import calendar_holdout_start
from custom_components.powerguesser.engine.learning.worker import (
    run_stage,
    stage_outputs,
)
from custom_components.powerguesser.engine.validation.constants import LINK_MIN_OVERLAP
from custom_components.powerguesser.engine.validation.link import assign_links, jaccard, on_mask
from tests.fixtures.learning.fake_llm import kettle_proposal
from tests.fixtures.learning.synth import (
    ENTITY_KETTLE,
    SynthSpec,
    build_buffer,
    utc_day_bounds,
)

_END_MS = int(datetime(2026, 11, 20, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)


def test_on_mask_marks_spans_and_keeps_allowed() -> None:
    """Spans mark inclusive [start, end]; the allowed mask restricts samples."""
    allowed = np.ones(10, dtype=bool)
    allowed[8:] = False
    mask = on_mask([(2, 4), (7, 9)], 10, allowed=allowed)
    assert mask.tolist() == [False, False, True, True, True, False, False, True, False, False]


def test_on_mask_closes_gaps_up_to_gap_samples() -> None:
    """Gaps of at most gap_samples between marked runs are merged; larger gaps are not."""
    mask = on_mask([(0, 1), (4, 5)], 10, gap_samples=2)
    assert mask.tolist() == [True, True, True, True, True, True, False, False, False, False]
    mask_big = on_mask([(0, 1), (5, 6)], 10, gap_samples=2)
    assert mask_big.tolist() == [True, True, False, False, False, True, True, False, False, False]


def test_on_mask_gap_merge_is_transitive_over_runs() -> None:
    """Chained small gaps merge into one run (island semantics)."""
    mask = on_mask([(0, 0), (2, 2), (4, 4)], 7, gap_samples=1)
    assert mask.tolist() == [True, True, True, True, True, False, False]


def test_jaccard_units() -> None:
    """|a and b| / |a or b|; 0.0 for an empty union."""
    a = np.array([True, True, False, False])
    b = np.array([True, False, True, False])
    assert jaccard(a, b) == pytest.approx(1.0 / 3.0)
    assert jaccard(a, a) == 1.0
    assert jaccard(a, ~a) == 0.0
    empty = np.zeros(4, dtype=bool)
    assert jaccard(empty, empty) == 0.0


def test_assign_links_deterministic_tie_break() -> None:
    """Ties break by signature_id then entity_id; each side used at most once."""
    scores = {
        ("sensor.plug_b", "sig02"): 0.9,
        ("sensor.plug_a", "sig01"): 0.9,
        ("sensor.plug_a", "sig02"): 0.95,
        ("sensor.plug_c", "sig03"): 0.4,  # below min_overlap
    }
    entities = ["sensor.plug_a", "sensor.plug_b", "sensor.plug_c"]
    links, unlinked = assign_links(scores, entities, min_overlap=LINK_MIN_OVERLAP)
    # sig02 wins plug_a at 0.95; the 0.9 tie on sig02 goes to the already
    # decided pair, so sig01 takes plug_a? No: plug_a is used by sig02, so
    # the (plug_b, sig02) pair at 0.9 loses its artefact; (plug_a, sig01)
    # cannot use plug_a either. Remaining usable pair: none at 0.9 because
    # both share a used side; only below-min pairs remain.
    assert [(l["entity_id"], l["signature_id"]) for l in links] == [
        ("sensor.plug_a", "sig02")
    ]
    assert unlinked == ["sensor.plug_b", "sensor.plug_c"]


def test_assign_links_tie_prefers_lower_signature_id() -> None:
    """At equal overlap the lower signature_id takes the plug."""
    scores = {
        ("sensor.plug_a", "sig02"): 0.8,
        ("sensor.plug_a", "sig01"): 0.8,
    }
    links, unlinked = assign_links(scores, ["sensor.plug_a"], min_overlap=0.5)
    assert [(l["entity_id"], l["signature_id"]) for l in links] == [
        ("sensor.plug_a", "sig01")
    ]
    assert unlinked == []


def test_assign_links_unlinked_in_input_order() -> None:
    """Unlinked plugs report in the input entity order, not sorted order."""
    scores = {("sensor.plug_b", "sig01"): 0.9}
    entities = ["sensor.plug_z", "sensor.plug_a", "sensor.plug_b"]
    links, unlinked = assign_links(scores, entities, min_overlap=0.5)
    assert [l["entity_id"] for l in links] == ["sensor.plug_b"]
    assert unlinked == ["sensor.plug_z", "sensor.plug_a"]


def _extract_job(buf_path: Path, stage_dir: Path, day_bounds: list[int], days: int) -> dict:
    return {
        "buffer_path": str(buf_path),
        "stage_dir": str(stage_dir),
        "grid_s": 30,
        "day_bounds_ms": day_bounds,
        "first_weekday": datetime.fromtimestamp(day_bounds[0] / 1000.0, tz=timezone.utc).weekday(),
        "holdout_days": 0,
        "holdout_calendar_days": 7,
        "train_fraction": 0.7,
        "enabled_event_types": ["step", "cycle_shape", "continuous"],
        "metered_channels": [],
        "phase_channels": {},
    }


def test_shadow_extract_and_link_on_synth_buffer(tmp_path: Path) -> None:
    """A 35-day kettle-plug buffer: masked extract, shadow compile, link to the plug."""
    buf_path = tmp_path / "buf.sqlite"
    stage_dir = tmp_path / "stage"
    day_bounds = utc_day_bounds(_END_MS, 35)
    data = build_buffer(buf_path, SynthSpec(days=35, kettle_plug=True), _END_MS)
    kettle_channel = data.channel_ids[ENTITY_KETTLE]

    res = run_stage("extract", _extract_job(buf_path, stage_dir, day_bounds, 35))
    outputs = stage_outputs("extract", res)
    assert outputs["insufficient_data"] is False

    meta = json.loads((stage_dir / "digest_meta.json").read_text(encoding="utf-8"))
    first_holdout = calendar_holdout_start(35, 7)
    # The holdout lists the last 7 calendar day indices, gaps included (D-18).
    assert meta["days"]["holdout"] == list(range(first_holdout, 35))
    assert meta["days"]["train"] and meta["days"]["val"]

    digest = json.loads((stage_dir / "digest.json").read_text(encoding="utf-8"))
    # No plug lines at all: build_digest omits the key when plugs_data is
    # empty, which satisfies D-02 just as [] would.
    assert digest.get("plugs", []) == []

    proposal = kettle_proposal(digest)
    (stage_dir / "proposal_1.json").write_text(json.dumps(proposal), encoding="utf-8")
    compile_job = {
        "stage_dir": str(stage_dir),
        "episodes_path": str(stage_dir / "episodes.npz"),
        "proposal_path": str(stage_dir / "proposal_1.json"),
        "run_id": "r-shadow",
        "round": 1,
        "model": "golden-model",
        "incumbent": None,
        "validated": True,
        "prior": [],
        "owned": [],
        "decisions": {},
    }
    compile_res = run_stage("compile_score", compile_job)
    library_path = compile_res["library"]

    link_job = {
        "buffer_path": str(buf_path),
        "stage_dir": str(stage_dir),
        "library_path": str(library_path),
        "masked_channels": [{"channel": kettle_channel, "entity_id": ENTITY_KETTLE}],
        "grid_s": 30,
        "day_bounds_ms": day_bounds,
    }
    link_res = run_stage("shadow_link", link_job)
    assert link_res["n_links"] == 1
    assert link_res["n_not_discovered"] == 0

    link_dict = json.loads((stage_dir / "shadow_link.json").read_text(encoding="utf-8"))
    (link,) = link_dict["links"]
    assert link["entity_id"] == ENTITY_KETTLE
    assert link["channel"] == kettle_channel
    assert link["kind"] == "switched"
    assert link["overlap"] >= LINK_MIN_OVERLAP
    assert link_dict["not_discovered"] == []
    plugs = link_dict["plugs"]
    assert set(plugs) == {ENTITY_KETTLE}
    assert plugs[ENTITY_KETTLE]["train_mean_w"] > 0.0
    # train_cutoff_ms is the start of the first holdout day (D-11/D-18).
    assert link_dict["train_cutoff_ms"] == day_bounds[first_holdout]


def test_shadow_extract_flags_insufficient_days(tmp_path: Path) -> None:
    """A 20-day buffer in a longer window: the shadow extract reports insufficient_data.

    The same buffer extracts normally with holdout_calendar_days 0, so the flag
    belongs to the calendar-holdout path only (RESEARCH Pitfall 5).
    """
    buf_path = tmp_path / "buf.sqlite"
    stage_dir = tmp_path / "stage"
    # Window of 35 calendar days over a 20-day buffer: 15 empty tail days.
    day_bounds = utc_day_bounds(_END_MS, 35)
    build_buffer(buf_path, SynthSpec(days=20, kettle_plug=True), _END_MS)

    job = _extract_job(buf_path, stage_dir, day_bounds, 35)
    res = run_stage("extract", job)
    outputs = stage_outputs("extract", res)
    assert outputs["insufficient_data"] is True
    assert not (stage_dir / "digest.json").exists()

    stage_dir2 = tmp_path / "stage2"
    job0 = {**job, "stage_dir": str(stage_dir2), "holdout_calendar_days": 0}
    res0 = run_stage("extract", job0)
    outputs0 = stage_outputs("extract", res0)
    assert outputs0["insufficient_data"] is False
    assert (stage_dir2 / "digest.json").exists()

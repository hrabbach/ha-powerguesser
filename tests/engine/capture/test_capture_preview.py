"""Engine tests for the capture preview fit percentile branch (D-02, CAP-09).

Plan 08-05 Task 1: with fewer than ``MIN_TRAIN_EPISODES`` clean examples the
preview boxes are the D-02 tolerance boxes spanning the examples; from three
clean examples on they are ``compile.fit_state_boxes`` over the examples
(provenance fit ``"percentile"``); a contaminated example never teaches.
"""

import json

import numpy as np
import pytest

from custom_components.powerguesser.engine.capture import preview
from custom_components.powerguesser.engine.learning import compile as compile_mod
from custom_components.powerguesser.engine.learning.evidence import (
    EVIDENCE_VERSION,
    MIN_RHYTHM_RUNS,
)


def _example(
    capture_id: str,
    *,
    rise: float,
    fall: float,
    plateau: float,
    dur: float,
    verdict: str = "clean",
    threshold: float = 120.0,
    grid: float = 30.0,
) -> dict:
    return {
        "capture_id": capture_id,
        "result": {
            "verdict": verdict,
            "step_w": rise,
            "fall_w": fall,
            "plateau_w": plateau,
            "dur_s": dur,
            "threshold_w": threshold,
            "grid_s": grid,
        },
    }


def _three_examples() -> list[dict]:
    return [
        _example("c1", rise=2000.0, fall=-2000.0, plateau=1980.0, dur=180.0),
        _example("c2", rise=2100.0, fall=-2080.0, plateau=2050.0, dur=120.0),
        _example("c3", rise=1950.0, fall=-1960.0, plateau=1940.0, dur=210.0),
    ]


def _expected_state(examples: list[dict]) -> dict:
    """Build the same rows fit_state_boxes sees, from the example result blocks."""
    rises = np.array([e["result"]["step_w"] for e in examples], dtype=np.float64)
    falls = np.array([e["result"]["fall_w"] for e in examples], dtype=np.float64)
    durs = np.array([e["result"]["dur_s"] for e in examples], dtype=np.float64)
    plats = np.array([e["result"]["plateau_w"] for e in examples], dtype=np.float64)
    nan = np.full(len(examples), np.nan, dtype=np.float64)
    table = {
        "rise_w": rises,
        "fall_w": falls,
        "dur_s": durs,
        "plateau_w": plats,
        "l1_share": nan,
        "l2_share": nan,
        "l3_share": nan,
    }
    return compile_mod.fit_state_boxes(
        table, np.arange(len(examples)), 120.0, name="on"
    ).to_dict()


def test_fit_preview_switches_to_percentile_at_three_clean() -> None:
    """1-2 clean examples use tolerance boxes; 3 clean use fit_state_boxes (percentile)."""
    examples = _three_examples()

    for n in (1, 2):
        art = preview.fit_preview("cap_x", "Kettle", examples[:n])
        assert art is not None
        assert art["provenance"]["fit"] == "tolerance"
        state = art["states"][0]
        rise_vals = [e["result"]["step_w"] for e in examples[:n]]
        assert state["rise_w"]["lo"] <= min(rise_vals)
        assert state["rise_w"]["hi"] >= max(rise_vals)

    art = preview.fit_preview("cap_x", "Kettle", examples)
    assert art is not None
    assert art["provenance"]["fit"] == "percentile"
    assert art["states"][0] == _expected_state(examples)

    # A contaminated example is stored but never teaches: the boxes do not move.
    contaminated = _example(
        "c4",
        rise=9000.0,
        fall=-9000.0,
        plateau=9000.0,
        dur=600.0,
        verdict="contaminated",
    )
    art2 = preview.fit_preview("cap_x", "Kettle", [*examples, contaminated])
    assert art2 is not None
    assert art2["provenance"]["fit"] == "percentile"
    assert art2["states"][0] == _expected_state(examples)
    assert art2["scores"]["n_examples"] == 3
    assert "c4" not in art2["provenance"]["capture_ids"]


# ---------------------------------------------------------------------------
# EVID-05: capture_evidence for a preview's own runs
# ---------------------------------------------------------------------------


def _pair_example(
    capture_id: str,
    rise_ms: int,
    *,
    step_w: float = 2000.0,
    fall_w: float = -2000.0,
    plateau_w: float = 1980.0,
    dur_s: float = 180.0,
    energy_wh: float = 99.0,
    grid_s: float = 30.0,
    hour: int = 7,
    dow: int = 0,
) -> dict:
    return {
        "capture_id": capture_id,
        "result": {
            "verdict": "clean",
            "rise_ms": rise_ms,
            "fall_ms": rise_ms + int(dur_s * 1000),
            "step_w": step_w,
            "fall_w": fall_w,
            "plateau_w": plateau_w,
            "dur_s": dur_s,
            "energy_wh": energy_wh,
            "grid_s": grid_s,
            "threshold_w": 120.0,
        },
        "features": {
            "rise_w": step_w,
            "fall_w": fall_w,
            "dur_s": dur_s,
            "plateau_w": plateau_w,
            "energy_wh": energy_wh,
            "hour": hour,
            "dow": dow,
            "l1_share": None,
            "l2_share": None,
            "l3_share": None,
        },
    }


def test_capture_evidence_shape() -> None:
    """Three captured runs give a v1 capture-sourced block, newest first, all captured."""
    base = 1_700_000_000_000
    examples = [
        _pair_example("c1", base),
        _pair_example("c2", base + 30_000),
        _pair_example("c3", base + 90_000),
    ]
    ev = preview.capture_evidence(
        examples, tz_name="UTC", now_iso="2026-11-24T10:00:00+00:00"
    )
    assert ev is not None
    assert ev["v"] == EVIDENCE_VERSION
    assert ev["source"] == "capture"
    assert ev["n_runs"] == 3
    starts = [occ["start_ms"] for occ in ev["occurrences"]]
    assert starts == sorted(starts, reverse=True)
    assert all(occ["captured"] is True for occ in ev["occurrences"])
    assert all(occ["detected"] is True for occ in ev["occurrences"])
    # Rhythm is suppressed below MIN_RHYTHM_RUNS (5).
    assert ev["rhythm"] is None
    assert MIN_RHYTHM_RUNS == 5
    assert ev["n_captured"] == 3
    assert ev["profile"]["step_w"]["p50"] == pytest.approx(2000.0, abs=1.0)
    assert json.loads(json.dumps(ev)) == ev

    # No example with a pair -> None.
    no_pair = [{"capture_id": "n", "result": {"verdict": "none", "rise_ms": None}}]
    assert preview.capture_evidence(no_pair, tz_name="UTC", now_iso="x") is None

"""Engine tests for the capture snap pair selection and quality verdict (CAP-04).

Plan 08-03 Task 2 replaces Plan 08-01's interim pin (every found pair marked
``clean``) with the full policy: ``clean`` / ``contaminated`` / ``ambiguous`` /
``none`` with their evidence. The hand-built residual arrays call
``select_and_judge`` directly; the ``none`` shape is also asserted through
``snap_run`` on a built buffer.
"""

from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from custom_components.powerguesser.engine.capture import snap
from custom_components.powerguesser.engine.capture.config import CaptureConfig
from tests.fixtures.learning.synth import (
    ENTITY_GRID,
    SynthSpec,
    build_buffer,
)

GRID_MS = 30_000
END_MS = 1_700_000_000_000


def _two_plateau_residual() -> np.ndarray:
    """30 s grid, flat 250 W, two 2000 W plateaus of 6 samples, rises 4 min apart."""
    r = np.full(60, 250.0)
    r[20:26] = 250.0 + 2000.0
    r[28:34] = 250.0 + 2000.0
    return r


def _still_on_residual(last_w: float) -> np.ndarray:
    """A rising run with no fall in range; the final sample sets the still-on test."""
    r = np.full(60, 250.0)
    r[16], r[17], r[18], r[19] = 900.0, 200.0, 250.0, 260.0
    r[20:59] = 2260.0
    r[59] = last_w
    return r


def _judge(r: np.ndarray, *, ref_start_ms: int, ref_end_ms: int, threshold_w: float = 200.0,
           enabled: list[str] | None = None, config: CaptureConfig = CaptureConfig(),
           gross: np.ndarray | None = None, native_gap_s: float | None = None) -> dict:
    return snap.select_and_judge(
        r,
        start_ms=0,
        step_ms=GRID_MS,
        threshold_w=threshold_w,
        rise_range_ms=(0, 2_000_000),
        fall_range_ms=(0, 2_000_000),
        ref_start_ms=ref_start_ms,
        ref_end_ms=ref_end_ms,
        enabled_event_types=enabled if enabled is not None else [],
        config=config,
        gross=gross,
        native_gap_s=native_gap_s,
    )


def test_verdict_clean() -> None:
    """One plateau in range, no inside edges, matching magnitudes -> clean."""
    r = np.full(60, 250.0)
    r[20:26] = 2250.0
    out = _judge(r, ref_start_ms=19 * GRID_MS, ref_end_ms=25 * GRID_MS)
    assert out["verdict"] == "clean"
    assert out["reason_code"] is None
    assert len(out["candidates"]) == 1
    assert out["quality_flags"] == []
    assert out["edges_inside"] == []
    assert out["row"]["start_i"] == 19
    assert out["row"]["end_i"] == 25
    assert out["row"]["rise_w"] == 2000.0
    assert out["row"]["fall_w"] == -2000.0


def test_verdict_contaminated_inside_edge() -> None:
    """An excursion of at least 0.25 x rise inside the run contaminates it; a sub-floor one does not."""
    r = np.full(60, 250.0)
    r[20:30] = 2250.0
    r[24] = 2850.0  # +600 / -600 excursion at samples 23->24 and 24->25
    out = _judge(
        r,
        ref_start_ms=19 * GRID_MS,
        ref_end_ms=29 * GRID_MS,
        enabled=["short_event"],
    )
    assert out["verdict"] == "contaminated"
    assert "inside_edges" in out["quality_flags"]
    assert [e["t_ms"] for e in out["edges_inside"]] == [23 * GRID_MS, 24 * GRID_MS]
    assert [e["delta_w"] for e in out["edges_inside"]] == [600.0, -600.0]

    # The same excursion at 400 W with threshold 300 stays below max(300, 0.25 x 2000) -> clean.
    r2 = np.full(60, 250.0)
    r2[20:30] = 2250.0
    r2[24] = 2650.0
    out2 = _judge(
        r2,
        ref_start_ms=19 * GRID_MS,
        ref_end_ms=29 * GRID_MS,
        threshold_w=200.0,
        enabled=["short_event"],
    )
    assert out2["verdict"] == "clean"
    assert "inside_edges" not in out2["quality_flags"]


def test_verdict_contaminated_magnitude_and_timing() -> None:
    """A fall beyond PAIR_TOL x rise flags magnitude_mismatch; a native gap flags timing_pm_1_sample."""
    # A small rise (400 W) with a 520 W fall: paired by the absolute rule only
    # (|520-400| = 120 <= PAIR_ABS_W = 150, but > PAIR_TOL x 400 = 100).
    r = np.full(60, 250.0)
    r[20:26] = 650.0
    r[26] = 130.0
    out = _judge(r, ref_start_ms=19 * GRID_MS, ref_end_ms=25 * GRID_MS, threshold_w=120.0)
    assert out["verdict"] == "contaminated"
    assert "magnitude_mismatch" in out["quality_flags"]
    assert "inside_edges" not in out["quality_flags"]

    # A clean plateau with native_gap_s 45 at grid 30 flags timing_pm_1_sample.
    r2 = np.full(60, 250.0)
    r2[20:26] = 2250.0
    out2 = _judge(
        r2,
        ref_start_ms=19 * GRID_MS,
        ref_end_ms=25 * GRID_MS,
        native_gap_s=45.0,
    )
    assert out2["verdict"] == "contaminated"
    assert "timing_pm_1_sample" in out2["quality_flags"]


def test_verdict_ambiguous() -> None:
    """Two pairs with scores within the ratio are ambiguous; a distant runner-up stays clean; a tie breaks by rise."""
    r = _two_plateau_residual()

    # Symmetric placement -> both scores tie -> ambiguous, earlier rise first.
    out = _judge(r, ref_start_ms=23 * GRID_MS, ref_end_ms=29 * GRID_MS)
    assert out["verdict"] == "ambiguous"
    assert len(out["candidates"]) == 2
    assert out["candidates"][0]["rise_ms"] == 19 * GRID_MS
    assert out["candidates"][1]["rise_ms"] == 27 * GRID_MS
    assert out["row"]["start_i"] == 19

    # A runner-up far from the reference scores more than 2.0x the best -> clean.
    out2 = _judge(r, ref_start_ms=19 * GRID_MS, ref_end_ms=25 * GRID_MS)
    assert out2["verdict"] == "clean"
    assert len(out2["candidates"]) == 2
    assert out2["row"]["start_i"] == 19


def test_still_on_uses_median_pre_level() -> None:
    """The pre-level is the median of PRE_LEVEL_SAMPLES ending at the rise edge; config varies it."""
    # Pre-level median(900, 200, 250, 260) = 255; rise 2000 -> threshold 255 + 1400 = 1655.
    out = _judge(
        _still_on_residual(1655.0),
        ref_start_ms=19 * GRID_MS,
        ref_end_ms=40 * GRID_MS,
    )
    assert out["verdict"] == "none"
    assert out["reason_code"] == "no_off_edge"
    assert "still_on" in out["quality_flags"]

    out2 = _judge(
        _still_on_residual(1635.0),
        ref_start_ms=19 * GRID_MS,
        ref_end_ms=40 * GRID_MS,
    )
    assert out2["reason_code"] == "no_off_edge"
    assert "still_on" not in out2["quality_flags"]

    # With pre_level_samples = 1 the pre-level is r[19] = 260 -> threshold 260 + 1400 = 1660,
    # so the 0.7 x rise case no longer sets still_on (the parameter path is live).
    out3 = _judge(
        _still_on_residual(1655.0),
        ref_start_ms=19 * GRID_MS,
        ref_end_ms=40 * GRID_MS,
        config=CaptureConfig(pre_level_samples=1),
    )
    assert out3["reason_code"] == "no_off_edge"
    assert "still_on" not in out3["quality_flags"]


def test_verdict_none_reasons() -> None:
    """A quiet residual, a rise without a fall, and a metered load each pick their reason."""
    quiet = np.full(60, 250.0)
    out = _judge(quiet, ref_start_ms=19 * GRID_MS, ref_end_ms=25 * GRID_MS)
    assert out["verdict"] == "none"
    assert out["reason_code"] == "no_edge_pair"
    assert out["quality_flags"] == []

    # A rise in the rise range without a fall -> no_off_edge.
    out2 = _judge(_still_on_residual(1000.0), ref_start_ms=19 * GRID_MS, ref_end_ms=40 * GRID_MS)
    assert out2["reason_code"] == "no_off_edge"

    # No pair in the residual but the gross series shows it -> metered_or_flat.
    gross = np.full(60, 250.0)
    gross[20:26] = 2250.0
    out3 = _judge(
        quiet,
        ref_start_ms=19 * GRID_MS,
        ref_end_ms=25 * GRID_MS,
        gross=gross,
    )
    assert out3["reason_code"] == "metered_or_flat"


def test_none_result_shape_through_snap_run(tmp_path: Path) -> None:
    """A none result keeps every key present, numeric values None and empty lists."""
    path = str(tmp_path / "buffer.sqlite")
    data = build_buffer(path, SynthSpec(days=4), END_MS)
    context = {
        "buffer_path": path,
        "grid_s": 30,
        "enabled_event_types": ["step", "cycle_shape", "continuous"],
        "metered_channels": [],
        "phase_channels": {},
        "session_windows": [],
        "tz_name": "UTC",
    }
    grid_rows = data.rows[ENTITY_GRID]
    base = grid_rows[0][0] + 60 * 60_000
    out = snap.snap_run(
        context,
        rise_range_ms=(base, base),
        fall_range_ms=(base + 300_000, base + 300_000),
        ref_start_ms=base,
        ref_end_ms=base + 300_000,
    )
    result = out["result"]
    assert result["verdict"] == "none"
    assert result["reason_code"] == "no_edge_pair"
    assert result["rise_ms"] is None
    assert result["fall_ms"] is None
    assert result["step_w"] is None
    assert result["energy_wh"] is None
    assert result["candidates"] == []
    assert result["edges_inside"] == []
    assert out["features"] is None
    assert out["trace"] is None

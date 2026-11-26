"""HA-free capture preview fit (D-02 tolerance boxes).

The preview fit turns one or more clean captured runs into a candidate artefact
dict with tolerance boxes: for a small number of examples (n < 3) the boxes are
the captured values plus/minus the D-02 tolerance, never a percentile fit (which
would collapse to near-zero width). Plan 08-05 adds the n >= MIN_TRAIN_EPISODES
percentile branch. numpy only, no Home Assistant import.
"""

from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np

from ..learning import compile as compile_mod
from ..learning import evidence as evidence_mod
from ..learning.artefact import Box, SignatureArtefact, State
from .config import DEFAULT_CONFIG, CaptureConfig


def _features_value(
    example: Mapping[str, Any], key: str, fallback: str | None
) -> float:
    """Read a preview row value from the example's features, else its result block."""
    features = example.get("features")
    if isinstance(features, Mapping) and features.get(key) is not None:
        return float(features[key])
    if fallback is not None:
        result = example.get("result")
        if isinstance(result, Mapping) and result.get(fallback) is not None:
            return float(result[fallback])
    return float("nan")


def _build_table(clean: Sequence[Mapping[str, Any]]) -> dict[str, np.ndarray]:
    """Build the fit_state_boxes episode table from the clean captured examples.

    Rows come from each example's ``features`` snapshot (the snapped episode
    row), falling back to the ``result`` block; phase shares are NaN when absent,
    so a capture without phase data contributes no phase box.
    """
    def column(key: str, fallback: str | None = None) -> np.ndarray:
        return np.asarray(
            [_features_value(ex, key, fallback) for ex in clean], dtype=np.float64
        )

    return {
        "rise_w": column("rise_w", "step_w"),
        "fall_w": column("fall_w", "fall_w"),
        "dur_s": column("dur_s", "dur_s"),
        "plateau_w": column("plateau_w", "plateau_w"),
        "l1_share": column("l1_share"),
        "l2_share": column("l2_share"),
        "l3_share": column("l3_share"),
    }


def _w_box(values: Sequence[float], threshold_w: float, config: CaptureConfig) -> Box:
    """Rise/fall/plateau box: captured value(s) plus/minus the D-02 watt tolerance."""
    arr = np.asarray(values, dtype=np.float64)
    lo_v = float(np.min(arr))
    hi_v = float(np.max(arr))
    lo = lo_v - max(config.preview_w_tol * abs(lo_v), threshold_w / 2.0)
    hi = hi_v + max(config.preview_w_tol * abs(hi_v), threshold_w / 2.0)
    p50 = float(np.median(arr))
    return Box(lo=round(lo, 1), p50=round(p50, 1), hi=round(hi, 1))


def _dur_box(values: Sequence[float], grid_s: float, config: CaptureConfig) -> Box:
    """Duration box: captured duration plus/minus max(30 %, one grid step), lo >= 0."""
    arr = np.asarray(values, dtype=np.float64)
    lo_v = float(np.min(arr))
    hi_v = float(np.max(arr))
    lo = max(0.0, lo_v - max(config.preview_dur_tol * lo_v, grid_s))
    hi = hi_v + max(config.preview_dur_tol * hi_v, grid_s)
    p50 = float(np.median(arr))
    return Box(lo=round(lo, 1), p50=round(p50, 1), hi=round(hi, 1))


def fit_preview(
    signature_id: str,
    label: str,
    examples: Sequence[Mapping[str, Any]],
    *,
    config: CaptureConfig = DEFAULT_CONFIG,
) -> dict[str, Any] | None:
    """Fit a preview artefact dict from the clean captured examples, or None.

    Only examples whose ``result.verdict`` is ``"clean"`` contribute; without one
    the function returns None. The capture ids and count are recorded in the
    provenance so a later merge can see which captures fed the preview.
    """
    clean = [
        ex
        for ex in examples
        if isinstance(ex.get("result"), Mapping)
        and ex["result"].get("verdict") == "clean"
    ]
    if not clean:
        return None

    results = [ex["result"] for ex in clean]
    threshold_w = float(np.median([float(r.get("threshold_w") or 0.0) for r in results]))
    grid_s = float(results[0].get("grid_s") or 30.0)

    if len(clean) >= compile_mod.MIN_TRAIN_EPISODES:
        # n >= 3: the percentile boxes the learner itself fits for a candidate
        # (fit_state_boxes), so a mature capture group matches like a learned one.
        state = compile_mod.fit_state_boxes(
            _build_table(clean), np.arange(len(clean)), threshold_w, name="on"
        )
        fit = "percentile"
    else:
        # n = 1-2: fit_state_boxes collapses to near-zero-width boxes, so use the
        # D-02 tolerance boxes spanning the captured values.
        rise_vals = [float(r["step_w"]) for r in results]
        fall_vals = [float(r["fall_w"]) for r in results]
        plateau_vals = [float(r["plateau_w"]) for r in results]
        dur_vals = [float(r["dur_s"]) for r in results]
        state = State(
            name="on",
            rise_w=_w_box(rise_vals, threshold_w, config),
            fall_w=_w_box(fall_vals, threshold_w, config),
            dur_s=_dur_box(dur_vals, grid_s, config),
            plateau_w=_w_box(plateau_vals, threshold_w, config),
            phase_share=None,
        )
        fit = "tolerance"

    capture_ids = [str(ex.get("capture_id", "")) for ex in clean]
    n = len(clean)
    artefact = SignatureArtefact(
        schema_version=1,
        signature_id=signature_id,
        kind="switched",
        label=label,
        category="unknown",
        extraction={
            "grid_s": grid_s,
            "edge_threshold_w": threshold_w,
            "merge_samples": 2,
            "pair_tol": 0.25,
        },
        states=(state,),
        session=None,
        time_prior={
            "hour": [1.0 / 24.0] * 24,
            "dow": [1.0 / 7.0] * 7,
        },
        constraints={"max_concurrent": 1},
        provenance={
            "source": "capture",
            "capture_ids": capture_ids,
            "n": n,
            "fit": fit,
            "tolerance": {
                "w_frac": config.preview_w_tol,
                "dur_frac": config.preview_dur_tol,
            },
        },
        scores={
            "train": None,
            "val": None,
            "validated": False,
            "conflicts": 0,
            "n_examples": n,
        },
    )
    return artefact.to_dict()


def _int_features_value(example: Mapping[str, Any], key: str) -> int:
    """Read an integer feature (hour/dow) from the example, defaulting to 0."""
    features = example.get("features")
    if isinstance(features, Mapping) and features.get(key) is not None:
        return int(features[key])
    return 0


def _rows_with_pair(
    examples: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    """The examples whose snapped result carries a rise/fall pair."""
    return [
        ex
        for ex in examples
        if isinstance(ex.get("result"), Mapping)
        and ex["result"].get("rise_ms") is not None
    ]


def _local_midnights(first_ms: int, last_ms: int, tz_name: str) -> list[int]:
    """Local-midnight day bounds (ms) covering first..last, plus the day after last.

    The bounds use the HA time zone through ``zoneinfo`` so ``build_evidence``'s
    window and rhythm read the same local days a learned run would (EVID-05). An
    unknown zone falls back to UTC.
    """
    try:
        tz = ZoneInfo(tz_name)
    except (ZoneInfoNotFoundError, ValueError):
        tz = ZoneInfo("UTC")
    start_day = datetime.fromtimestamp(first_ms / 1000.0, tz).date()
    end_day = datetime.fromtimestamp(last_ms / 1000.0, tz).date()
    bounds: list[int] = []
    day = start_day
    while day <= end_day:
        midnight = datetime(day.year, day.month, day.day, tzinfo=tz)
        bounds.append(int(round(midnight.timestamp() * 1000)))
        day += timedelta(days=1)
    after = datetime(end_day.year, end_day.month, end_day.day, tzinfo=tz) + timedelta(days=1)
    bounds.append(int(round(after.timestamp() * 1000)))
    return bounds


def capture_evidence(
    examples: Sequence[Mapping[str, Any]],
    *,
    tz_name: str,
    now_iso: str,
) -> dict[str, Any] | None:
    """Build the EVID-05 evidence block for a capture preview's own runs.

    Every example with a snapped pair contributes one row on the first
    example's grid, and the whole set is claimed for the block, so
    ``build_evidence`` renders the same v1 schema a learned candidate carries
    (``source: "capture"``). Rhythm stays suppressed below ``MIN_RHYTHM_RUNS``.
    Every listed occurrence is marked ``captured`` and ``n_captured`` set, so
    the block reads like the read-time overlay of a learned candidate
    (``overlay_captured``). Returns None without a pair.
    """
    pairs = _rows_with_pair(examples)
    if not pairs:
        return None

    grid_s = float((pairs[0].get("result") or {}).get("grid_s") or 30.0)
    step_ms = int(round(grid_s * 1000))
    if step_ms <= 0:
        step_ms = 30_000
    rises = [int(ex["result"]["rise_ms"]) for ex in pairs]
    base_ms = min(rises)
    n = len(pairs)
    start_i = np.asarray(
        [int(round((rise - base_ms) / step_ms)) for rise in rises], dtype=np.int64
    )

    def column(key: str, fallback: str | None = None) -> np.ndarray:
        return np.asarray(
            [_features_value(ex, key, fallback) for ex in pairs], dtype=np.float64
        )

    table: dict[str, np.ndarray] = {
        "start_i": start_i,
        "end_i": start_i + 1,
        "rise_w": column("rise_w", "step_w"),
        "fall_w": column("fall_w", "fall_w"),
        "dur_s": column("dur_s", "dur_s"),
        "plateau_w": column("plateau_w", "plateau_w"),
        "energy_wh": column("energy_wh", "energy_wh"),
        "kind": np.zeros(n, dtype=np.int32),
        "hour": np.asarray([_int_features_value(ex, "hour") for ex in pairs], dtype=np.int32),
        "dow": np.asarray([_int_features_value(ex, "dow") for ex in pairs], dtype=np.int32),
        "l1_share": column("l1_share"),
        "l2_share": column("l2_share"),
        "l3_share": column("l3_share"),
    }
    bounds = _local_midnights(base_ms, max(rises), tz_name)
    valid_days = list(range(max(0, len(bounds) - 1)))

    result = evidence_mod.build_evidence(
        table,
        {"capture": set(range(n))},
        start_ms=base_ms,
        step_ms=step_ms,
        day_bounds_ms=bounds,
        valid_days=valid_days,
        now_iso=now_iso,
        source="capture",
    )
    evidence = result["capture"]
    for occurrence in evidence["occurrences"]:
        occurrence["captured"] = True
        occurrence["detected"] = True
    evidence["n_captured"] = len(evidence["occurrences"])
    evidence["n_captured_unmatched"] = 0
    return evidence

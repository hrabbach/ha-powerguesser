"""HA-free capture snap: extract pairing on a window, pair selection, result block.

The snap reuses the learner's own residual construction and pairing so a
captured episode equals the row the extract stage would have produced for that
run (Pattern 1). numpy only, no Home Assistant import.
"""

from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta
from math import ceil
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np

from ..learning import episodes
from ..learning.worker import _load_job_residual
from ..reader import load_gross_layout, open_readonly, read_range
from .config import DEFAULT_CONFIG, CaptureConfig


def retro_ranges(
    start_ms: int, end_ms: int, config: CaptureConfig = DEFAULT_CONFIG
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Return the rise and fall search ranges for a retroactive window.

    Each range is the requested instant plus/minus the snap margin (inclusive).
    """
    margin = config.snap_margin_s * 1000
    return ((start_ms - margin, start_ms + margin), (end_ms - margin, end_ms + margin))


def live_ranges(
    press_start_ms: int,
    press_stop_ms: int,
    now_ms: int,
    config: CaptureConfig = DEFAULT_CONFIG,
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Return the live rise/fall search ranges for a started/stopped capture.

    The rise is searched asymmetrically around the start press
    (``live_rise_before_s`` / ``live_rise_after_s``); the fall runs from
    ``live_fall_before_s`` before the stop press up to the newest sample
    (``now_ms``).
    """
    rise_range = (
        press_start_ms - config.live_rise_before_s * 1000,
        press_start_ms + config.live_rise_after_s * 1000,
    )
    fall_range = (press_stop_ms - config.live_fall_before_s * 1000, now_ms)
    return rise_range, fall_range


def _pairs_in_ranges(
    r: np.ndarray,
    *,
    start_ms: int,
    step_ms: int,
    threshold_w: float,
    rise_range_ms: tuple[int, int],
    fall_range_ms: tuple[int, int],
    ref_start_ms: int,
    ref_end_ms: int,
    enabled_event_types: Sequence[str],
) -> tuple[list[tuple[float, int, int]], dict[str, np.ndarray]]:
    """Return the candidate pairs (score, rise_t, index) inside both ranges.

    Uses the learner's own ``detect_edges`` -> ``suppress_alignment_artefacts``
    -> ``pair_switched`` so a captured episode equals the extract row (Pattern 1).
    """
    idx, delta = episodes.detect_edges(r, threshold_w)
    idx, delta = episodes.suppress_alignment_artefacts(
        idx, delta, enabled_event_types=enabled_event_types
    )
    table = episodes.pair_switched(r, idx, delta, step_ms / 1000.0)
    rise_lo, rise_hi = rise_range_ms
    fall_lo, fall_hi = fall_range_ms
    scored: list[tuple[float, int, int]] = []
    for k in range(len(table["start_i"])):
        rise_t = start_ms + int(table["start_i"][k]) * step_ms
        fall_t = start_ms + int(table["end_i"][k]) * step_ms
        if not (rise_lo <= rise_t <= rise_hi):
            continue
        if not (fall_lo <= fall_t <= fall_hi):
            continue
        score = abs(rise_t - ref_start_ms) + abs(fall_t - ref_end_ms)
        scored.append((score, rise_t, k))
    return scored, table


def _pre_level(r: np.ndarray, edge_i: int, n: int) -> float | None:
    """Median of the last ``n`` finite samples ending at the rise edge sample."""
    lo = max(0, edge_i - n + 1)
    window = r[lo : edge_i + 1]
    finite = window[np.isfinite(window)]
    if len(finite) == 0:
        return None
    return float(np.median(finite))


def _none_outcome(
    r: np.ndarray,
    *,
    start_ms: int,
    step_ms: int,
    threshold_w: float,
    rise_range_ms: tuple[int, int],
    fall_range_ms: tuple[int, int],
    ref_start_ms: int,
    ref_end_ms: int,
    idx: np.ndarray,
    delta: np.ndarray,
    gross: np.ndarray | None,
    enabled_event_types: Sequence[str],
    config: CaptureConfig,
) -> dict[str, Any]:
    """Build the ``none`` result, choosing no_off_edge / metered_or_flat / no_edge_pair."""
    rise_lo, rise_hi = rise_range_ms
    rise_edges = [
        (int(i), float(d))
        for i, d in zip(idx, delta)
        if d > 0 and d >= threshold_w and rise_lo <= start_ms + int(i) * step_ms <= rise_hi
    ]
    reason = "no_edge_pair"
    flags: list[str] = []
    if rise_edges:
        reason = "no_off_edge"
        edge_i, rise_delta = rise_edges[0]
        pre = _pre_level(r, edge_i, config.pre_level_samples)
        finite_idx = np.flatnonzero(np.isfinite(r))
        if pre is not None and len(finite_idx):
            last = float(r[int(finite_idx[-1])])
            if last >= pre + config.live_still_on_frac * rise_delta:
                flags.append("still_on")
    elif gross is not None:
        gross_scored, _gross_table = _pairs_in_ranges(
            gross,
            start_ms=start_ms,
            step_ms=step_ms,
            threshold_w=threshold_w,
            rise_range_ms=rise_range_ms,
            fall_range_ms=fall_range_ms,
            ref_start_ms=ref_start_ms,
            ref_end_ms=ref_end_ms,
            enabled_event_types=enabled_event_types,
        )
        if gross_scored:
            reason = "metered_or_flat"
    return {
        "verdict": "none",
        "reason_code": reason,
        "row": None,
        "candidates": [],
        "edges_inside": [],
        "quality_flags": flags,
    }


def select_and_judge(
    r: np.ndarray,
    *,
    start_ms: int,
    step_ms: int,
    threshold_w: float,
    rise_range_ms: tuple[int, int],
    fall_range_ms: tuple[int, int],
    ref_start_ms: int,
    ref_end_ms: int,
    enabled_event_types: Sequence[str],
    config: CaptureConfig = DEFAULT_CONFIG,
    gross: np.ndarray | None = None,
    native_gap_s: float | None = None,
) -> dict[str, Any]:
    """Select the best rise/fall pair in the ranges and judge the verdict (CAP-04).

    Returns ``{"verdict", "reason_code", "row", "candidates", "edges_inside",
    "quality_flags"}``. ``ambiguous`` when a runner-up pair scores within
    ``config.ambiguity_ratio`` of the best (candidates best first, ties by the
    earlier rise). ``contaminated`` when the best pair carries an inside edge of at
    least ``max(threshold_w, config.inside_edge_rel x rise)``, a fall that differs
    from the rise by more than ``PAIR_TOL x rise``, or a missing native sample
    (NaN inside the run, or ``native_gap_s >= 1.5 x grid``). ``clean`` otherwise.
    With no pair the verdict is ``none``: ``no_off_edge`` (a rise in the rise range
    with no fall; flag ``still_on`` when the residual is still up), ``metered_or_flat``
    (only the gross series shows the pair) or ``no_edge_pair``.
    """
    step_s = step_ms / 1000.0
    idx, delta = episodes.detect_edges(r, threshold_w)
    idx, delta = episodes.suppress_alignment_artefacts(
        idx, delta, enabled_event_types=enabled_event_types
    )
    scored, table = _pairs_in_ranges(
        r,
        start_ms=start_ms,
        step_ms=step_ms,
        threshold_w=threshold_w,
        rise_range_ms=rise_range_ms,
        fall_range_ms=fall_range_ms,
        ref_start_ms=ref_start_ms,
        ref_end_ms=ref_end_ms,
        enabled_event_types=enabled_event_types,
    )

    if not scored:
        return _none_outcome(
            r,
            start_ms=start_ms,
            step_ms=step_ms,
            threshold_w=threshold_w,
            rise_range_ms=rise_range_ms,
            fall_range_ms=fall_range_ms,
            ref_start_ms=ref_start_ms,
            ref_end_ms=ref_end_ms,
            idx=idx,
            delta=delta,
            gross=gross,
            enabled_event_types=enabled_event_types,
            config=config,
        )

    # Best = lowest score, ties by earlier rise.
    scored.sort(key=lambda item: (item[0], item[1]))
    best_k = scored[0][2]

    candidates = [
        {
            "rise_ms": start_ms + int(table["start_i"][k]) * step_ms,
            "fall_ms": start_ms + int(table["end_i"][k]) * step_ms,
            "step_w": float(table["rise_w"][k]),
        }
        for _score, _rise_t, k in scored
    ]

    row = {
        "start_i": int(table["start_i"][best_k]),
        "end_i": int(table["end_i"][best_k]),
        "rise_w": float(table["rise_w"][best_k]),
        "fall_w": float(table["fall_w"][best_k]),
        "dur_s": float(table["dur_s"][best_k]),
        "plateau_w": float(table["plateau_w"][best_k]),
        "energy_wh": float(table["energy_wh"][best_k]),
    }

    edges_inside: list[dict[str, Any]] = []
    quality_flags: list[str] = []
    ambiguous = len(scored) >= 2 and scored[1][0] <= config.ambiguity_ratio * max(
        scored[0][0], step_ms
    )
    if ambiguous:
        verdict = "ambiguous"
    else:
        start_i = row["start_i"]
        end_i = row["end_i"]
        rise = abs(row["rise_w"])
        inside_floor = max(threshold_w, config.inside_edge_rel * rise)
        for e_idx, e_delta in zip(idx, delta):
            e_i = int(e_idx)
            if start_i < e_i < end_i and abs(e_delta) >= inside_floor:
                edges_inside.append(
                    {"t_ms": start_ms + e_i * step_ms, "delta_w": float(e_delta)}
                )
        if edges_inside:
            edges_inside.sort(key=lambda item: item["t_ms"])
            quality_flags.append("inside_edges")
        if abs(abs(row["fall_w"]) - rise) > episodes.PAIR_TOL * rise:
            quality_flags.append("magnitude_mismatch")
        segment = r[max(0, start_i - 1) : end_i + 2]
        timing_bad = bool(np.isnan(segment).any())
        if native_gap_s is not None and native_gap_s >= 1.5 * step_s:
            timing_bad = True
        if timing_bad:
            quality_flags.append("timing_pm_1_sample")
        verdict = "contaminated" if quality_flags else "clean"

    return {
        "verdict": verdict,
        "reason_code": None,
        "row": row,
        "candidates": candidates,
        "edges_inside": edges_inside,
        "quality_flags": quality_flags,
    }


def _native_max_gap_s(
    buffer_path: str, rise_ms: int, fall_ms: int, grid_ms: int
) -> float | None:
    """Largest spacing (s) between consecutive native gross rows inside the run.

    Scans the non-solar gross channels (falling back to solar when every gross
    channel is solar), like the window rules and engine/coverage.py. Returns None
    when a channel holds fewer than two rows in the range.
    """
    conn = open_readonly(buffer_path)
    try:
        layout = load_gross_layout(conn)
        roles: dict[int, str] = {}
        for role, _transform, channels in layout:
            for ch in channels:
                roles[ch] = role
        gross_channels = list(roles)
        all_solar = gross_channels and all(
            roles[ch] == "solar" for ch in gross_channels
        )
        chosen = (
            gross_channels
            if all_solar
            else [ch for ch in gross_channels if roles[ch] != "solar"]
        )
        lo = rise_ms - grid_ms
        hi = fall_ms + grid_ms
        max_gap_ms = 0
        for ch in chosen:
            # read_range also returns the last native row before ``lo`` (the
            # seed); keep it as the initial ``prev_ts`` so a gap straddling the
            # run's leading edge is measured, and check the trailing edge from
            # the last row to ``hi`` so a gap straddling the fall boundary is
            # measured too — matching the window rules (a missing boundary
            # sample must flag the run, not slip through).
            prev_ts: int | None = None
            for r in read_range(conn, [ch], lo, hi + 1).get(ch, []):
                ts = r[0]
                if ts < lo:
                    prev_ts = ts
                    continue
                if prev_ts is not None and ts - prev_ts > max_gap_ms:
                    max_gap_ms = ts - prev_ts
                prev_ts = ts
            if prev_ts is not None and hi - prev_ts > max_gap_ms:
                max_gap_ms = hi - prev_ts
        return max_gap_ms / 1000.0 if max_gap_ms > 0 else None
    finally:
        conn.close()


def _align_down(value_ms: int, step_ms: int) -> int:
    return (value_ms // step_ms) * step_ms


def _align_up(value_ms: int, step_ms: int) -> int:
    return -((-value_ms) // step_ms) * step_ms


def _day_bounds_ms(ctx_start_ms: int, ctx_end_ms: int, tz_name: str) -> tuple[list[int], int]:
    """Local-midnight day bounds covering the context, and the first weekday."""
    tz = ZoneInfo(tz_name)
    start_local = datetime.fromtimestamp(ctx_start_ms / 1000.0, tz=tz)
    first_midnight = start_local.replace(hour=0, minute=0, second=0, microsecond=0)
    bounds: list[int] = []
    cur = first_midnight
    while True:
        bounds.append(int(round(cur.timestamp() * 1000)))
        if cur.timestamp() * 1000 > ctx_end_ms:
            break
        # Advance one CALENDAR day, not 86400 s: on an autumn fall-back day
        # (25 h) a +86400 s step lands back inside the same local day / at a
        # repeated local midnight, which would emit a spurious, duplicate
        # midnight and shift every later day_index, so hour/dow come out wrong
        # for any capture crossing the transition (CAP-01 DST-safe).
        next_day = cur.date() + timedelta(days=1)
        cur = datetime(next_day.year, next_day.month, next_day.day, tzinfo=tz)
        if len(bounds) > 400:  # guard against a pathological tz loop
            break
    return bounds, first_midnight.weekday()


def _trace(
    r: np.ndarray, start_ms: int, step_ms: int, run_start_i: int, run_end_i: int,
    config: CaptureConfig,
) -> dict[str, Any]:
    """Return the decimated trace of the residual around the snapped run."""
    pad_samples = int(round(config.trace_pad_s * 1000 / step_ms))
    lo = max(0, run_start_i - pad_samples)
    hi = min(len(r), run_end_i + pad_samples + 1)
    seg = r[lo:hi]
    n = len(seg)
    stride = max(1, ceil(n / config.trace_max_points))
    decimated = seg[::stride]
    w: list[int | None] = [
        None if not np.isfinite(v) else int(round(float(v))) for v in decimated
    ]
    return {
        "start_ms": start_ms + lo * step_ms,
        "step_ms": int(step_ms * stride),
        "w": w,
    }


def snap_run(
    context: Mapping[str, Any],
    *,
    rise_range_ms: tuple[int, int],
    fall_range_ms: tuple[int, int],
    ref_start_ms: int,
    ref_end_ms: int,
    config: CaptureConfig = DEFAULT_CONFIG,
) -> dict[str, Any]:
    """Run the snap for one capture window and return the result/features/trace.

    Builds the residual through the learner's own ``_load_job_residual`` over a
    context window ([min rise - context_before, max fall + context_after]
    aligned to the grid), then selects the pair and computes features and trace.
    """
    grid_s = int(context.get("grid_s", 30))
    step_ms = grid_s * 1000

    raw_start = min(rise_range_ms) - config.context_before_s * 1000
    raw_end = max(fall_range_ms) + config.context_after_s * 1000
    ctx_start = _align_down(raw_start, step_ms)
    ctx_end = _align_up(raw_end, step_ms)

    job = {**dict(context), "day_bounds_ms": [ctx_start, ctx_end]}
    series, _window_by_key = _load_job_residual(job)
    r = series.residual
    phases = series.phases

    finite_d = np.diff(r[np.isfinite(r)])
    threshold_w = episodes.edge_threshold(finite_d)

    enabled_event_types = context.get("enabled_event_types", []) or []

    outcome = select_and_judge(
        r,
        start_ms=ctx_start,
        step_ms=step_ms,
        threshold_w=threshold_w,
        rise_range_ms=rise_range_ms,
        fall_range_ms=fall_range_ms,
        ref_start_ms=ref_start_ms,
        ref_end_ms=ref_end_ms,
        enabled_event_types=enabled_event_types,
        config=config,
        gross=series.gross,
    )

    # The native sample spacing can only be measured once a pair is known
    # (only when a pair exists): re-judge with it so a missing native sample
    # inside the run is reported as contamination (timing_pm_1_sample).
    if outcome["row"] is not None:
        pair_start_i = outcome["row"]["start_i"]
        pair_end_i = outcome["row"]["end_i"]
        pair_rise_ms = ctx_start + pair_start_i * step_ms
        pair_fall_ms = ctx_start + pair_end_i * step_ms
        native_gap_s = _native_max_gap_s(
            str(context.get("buffer_path") or ""), pair_rise_ms, pair_fall_ms, step_ms
        )
        if native_gap_s is not None:
            outcome = select_and_judge(
                r,
                start_ms=ctx_start,
                step_ms=step_ms,
                threshold_w=threshold_w,
                rise_range_ms=rise_range_ms,
                fall_range_ms=fall_range_ms,
                ref_start_ms=ref_start_ms,
                ref_end_ms=ref_end_ms,
                enabled_event_types=enabled_event_types,
                config=config,
                gross=series.gross,
                native_gap_s=native_gap_s,
            )

    row = outcome["row"]
    if row is None:
        result: dict[str, Any] = {
            "rise_ms": None,
            "fall_ms": None,
            "step_w": None,
            "fall_w": None,
            "dur_s": None,
            "plateau_w": None,
            "energy_wh": None,
            "phase_share": None,
            "grid_s": grid_s,
            "threshold_w": round(float(threshold_w), 1),
            "verdict": outcome["verdict"],
            "reason_code": outcome["reason_code"],
            "candidates": outcome["candidates"],
            "edges_inside": outcome["edges_inside"],
            "press_to_rise_s": None,
            "press_to_fall_s": None,
            "quality_flags": outcome["quality_flags"],
            "similar_dismissed": None,
        }
        return {"result": result, "features": None, "trace": None}

    start_i = row["start_i"]
    end_i = row["end_i"]
    rise_ms = ctx_start + start_i * step_ms
    fall_ms = ctx_start + end_i * step_ms

    # Features via the extract pipeline's compute_features on a one-row table.
    tz_name = str(context.get("tz_name") or "UTC")
    day_bounds, first_weekday = _day_bounds_ms(ctx_start, ctx_end, tz_name)
    one_row = {
        "start_i": np.array([start_i], dtype=np.int64),
        "end_i": np.array([end_i], dtype=np.int64),
        "rise_w": np.array([row["rise_w"]], dtype=np.float64),
        "fall_w": np.array([row["fall_w"]], dtype=np.float64),
        "dur_s": np.array([row["dur_s"]], dtype=np.float64),
        "plateau_w": np.array([row["plateau_w"]], dtype=np.float64),
        "energy_wh": np.array([row["energy_wh"]], dtype=np.float64),
        "kind": np.array([0], dtype=np.int32),
    }
    feat_table = episodes.compute_features(
        one_row, ctx_start, step_ms, day_bounds, first_weekday, phases
    )

    def _f(arr: np.ndarray) -> float | None:
        val = float(arr[0])
        return val if np.isfinite(val) else None

    features = {
        "rise_w": row["rise_w"],
        "fall_w": row["fall_w"],
        "dur_s": row["dur_s"],
        "plateau_w": row["plateau_w"],
        "energy_wh": row["energy_wh"],
        "hour": int(feat_table["hour"][0]),
        "dow": int(feat_table["dow"][0]),
        "l1_share": _f(feat_table["l1_share"]),
        "l2_share": _f(feat_table["l2_share"]),
        "l3_share": _f(feat_table["l3_share"]),
    }

    has_phases = bool(phases) and all(
        features[k] is not None for k in ("l1_share", "l2_share", "l3_share")
    )
    phase_share = (
        {
            "l1": features["l1_share"],
            "l2": features["l2_share"],
            "l3": features["l3_share"],
        }
        if has_phases
        else None
    )

    result = {
        "rise_ms": rise_ms,
        "fall_ms": fall_ms,
        "step_w": row["rise_w"],
        "fall_w": row["fall_w"],
        "dur_s": row["dur_s"],
        "plateau_w": row["plateau_w"],
        "energy_wh": row["energy_wh"],
        "phase_share": phase_share,
        "grid_s": grid_s,
        "threshold_w": round(float(threshold_w), 1),
        "verdict": outcome["verdict"],
        "reason_code": outcome["reason_code"],
        "candidates": outcome["candidates"],
        "edges_inside": outcome["edges_inside"],
        "press_to_rise_s": None,
        "press_to_fall_s": None,
        "quality_flags": outcome["quality_flags"],
        "similar_dismissed": None,
    }
    trace = _trace(r, ctx_start, step_ms, start_i, end_i, config)
    return {"result": result, "features": features, "trace": trace}

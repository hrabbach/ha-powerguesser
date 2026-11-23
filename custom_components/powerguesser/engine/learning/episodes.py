"""Event detection, edge merging, and episode extraction from power series."""

from collections.abc import Mapping, Sequence
import numpy as np
import scipy.ndimage

from ..intervals import EventType

MIN_EDGE_W = 120.0
MAD_K = 4.0
MERGE_SAMPLES = 2
PAIR_TOL = 0.25
PAIR_ABS_W = 150.0
MAX_PAIR_S = 21600
SESSION_GAP_S = 180.0
RLE_ROUND_W = 10
RLE_MAX_SEGMENTS = 60
PERCENTILE_WINDOW_S = 3600.0
PLUG_ON_W = 10.0
PLUG_STANDBY_MARGIN_W = 20.0
PLUG_GAP_S = 300.0


def edge_threshold(d: np.ndarray) -> float:
    """Compute adaptive edge threshold from finite diffs."""
    finite = np.isfinite(d)
    finite_d = d[finite]
    if len(finite_d) == 0:
        return MIN_EDGE_W
    med = float(np.median(finite_d))
    mad = float(np.median(np.abs(finite_d - med)))
    return max(MIN_EDGE_W, MAD_K * 1.4826 * mad)


def detect_edges(r: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """Detect power step edges within contiguous finite runs with split-step merging."""
    n = len(r)
    if n < 2:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)

    finite = np.isfinite(r)
    segments: list[tuple[int, int]] = []
    in_seg = False
    seg_start = 0
    for i in range(n):
        if finite[i]:
            if not in_seg:
                in_seg = True
                seg_start = i
        else:
            if in_seg:
                in_seg = False
                if i - seg_start >= 2:
                    segments.append((seg_start, i))
    if in_seg and n - seg_start >= 2:
        segments.append((seg_start, n))

    edge_indices: list[int] = []
    edge_deltas: list[float] = []
    min_split = min(40.0, threshold * 0.25)

    for seg_start, seg_end in segments:
        d = np.diff(r[seg_start:seg_end])
        m = len(d)
        i = 0
        while i < m:
            if (
                i + 1 < m
                and np.sign(d[i]) == np.sign(d[i + 1])
                and np.sign(d[i]) != 0
            ):
                if (
                    abs(d[i]) >= min_split
                    and abs(d[i + 1]) >= min_split
                    and abs(d[i] + d[i + 1]) >= threshold
                ):
                    edge_indices.append(seg_start + i)
                    edge_deltas.append(float(d[i] + d[i + 1]))
                    i += 2
                    continue
            if abs(d[i]) >= threshold:
                edge_indices.append(seg_start + i)
                edge_deltas.append(float(d[i]))
                i += 1
            else:
                i += 1

    return np.array(edge_indices, dtype=np.int64), np.array(edge_deltas, dtype=np.float64)


def suppress_alignment_artefacts(
    idx: np.ndarray,
    delta: np.ndarray,
    *,
    enabled_event_types: Sequence[str | EventType] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Drop single-sample opposite-sign edge excursions when short_event is disabled."""
    # Check if short_event is enabled
    is_short_enabled = False
    if enabled_event_types is not None:
        is_short_enabled = (
            EventType.SHORT_EVENT in enabled_event_types
            or EventType.SHORT_EVENT.value in enabled_event_types
            or "short_event" in enabled_event_types
        )

    if is_short_enabled or len(idx) < 2:
        return idx, delta

    drop_mask = np.zeros(len(idx), dtype=bool)
    i = 0
    while i < len(idx) - 1:
        if idx[i + 1] == idx[i] + 1 and (delta[i] * delta[i + 1] < 0):
            # Opposite signs at consecutive sample indices
            first_abs = abs(delta[i])
            if abs(delta[i] + delta[i + 1]) < 0.25 * first_abs:
                drop_mask[i] = True
                drop_mask[i + 1] = True
                i += 2
                continue
        i += 1

    keep = ~drop_mask
    return idx[keep], delta[keep]


def pair_switched(
    r: np.ndarray,
    idx: np.ndarray,
    delta: np.ndarray,
    step_s: float,
    *,
    return_unpaired: bool = False,
) -> dict[str, np.ndarray] | tuple[dict[str, np.ndarray], int]:
    """Pair rise and fall edges using a LIFO stack into switched episodes."""
    n_edges = len(idx)
    stack: list[tuple[int, float]] = []
    episodes: list[dict[str, float | int]] = []
    n_unpaired = 0

    for k in range(n_edges):
        edge_i = int(idx[k])
        edge_d = float(delta[k])

        if edge_d > 0:
            stack.append((edge_i, edge_d))
        else:
            fall_abs = abs(edge_d)
            matched = False
            for j in range(len(stack) - 1, -1, -1):
                rise_i, rise_w = stack[j]
                dur_s = (edge_i - rise_i) * step_s
                if dur_s <= 0 or dur_s > MAX_PAIR_S:
                    continue
                if np.isnan(r[rise_i : edge_i + 2]).any():
                    continue

                if (abs(fall_abs - rise_w) <= PAIR_TOL * rise_w) or (
                    abs(fall_abs - rise_w) <= PAIR_ABS_W
                ):
                    stack.pop(j)
                    pre_edge = r[rise_i]
                    plateau_slice = r[rise_i + 1 : edge_i + 1]
                    if len(plateau_slice) == 0:
                        plateau_w = (rise_w + fall_abs) / 2.0
                        energy_wh = plateau_w * (dur_s / 3600.0)
                    else:
                        plateau_w = float(np.median(plateau_slice) - pre_edge)
                        energy_wh = float(np.sum(plateau_slice - pre_edge) * (step_s / 3600.0))

                    episodes.append(
                        {
                            "start_i": rise_i,
                            "end_i": edge_i,
                            "rise_w": rise_w,
                            "fall_w": edge_d,
                            "dur_s": dur_s,
                            "plateau_w": plateau_w,
                            "energy_wh": energy_wh,
                            "kind": 0,
                        }
                    )
                    matched = True
                    break

            if not matched:
                n_unpaired += 1

    n_unpaired += len(stack)
    episodes.sort(key=lambda ep: ep["start_i"])

    table: dict[str, np.ndarray] = {
        "start_i": np.array([ep["start_i"] for ep in episodes], dtype=np.int64),
        "end_i": np.array([ep["end_i"] for ep in episodes], dtype=np.int64),
        "rise_w": np.array([ep["rise_w"] for ep in episodes], dtype=np.float64),
        "fall_w": np.array([ep["fall_w"] for ep in episodes], dtype=np.float64),
        "dur_s": np.array([ep["dur_s"] for ep in episodes], dtype=np.float64),
        "plateau_w": np.array([ep["plateau_w"] for ep in episodes], dtype=np.float64),
        "energy_wh": np.array([ep["energy_wh"] for ep in episodes], dtype=np.float64),
        "kind": np.array([ep["kind"] for ep in episodes], dtype=np.int32),
    }

    if return_unpaired:
        return table, n_unpaired
    return table


def _encode_rle(levels: np.ndarray) -> list[list[int]]:
    """Encode 1D level array to run-length encoding with 10 W rounding and max 60 segments."""
    if len(levels) == 0:
        return []

    # Round to RLE_ROUND_W = 10 W
    rounded = (np.round(levels / RLE_ROUND_W) * RLE_ROUND_W).astype(np.int32)

    # Initial runs
    segments: list[list[int]] = []
    curr_level = int(rounded[0])
    curr_n = 1
    for i in range(1, len(rounded)):
        lvl = int(rounded[i])
        if lvl == curr_level:
            curr_n += 1
        else:
            segments.append([curr_level, curr_n])
            curr_level = lvl
            curr_n = 1
    segments.append([curr_level, curr_n])

    # Merge adjacent segments within 10 W
    merged: list[list[int]] = []
    for seg in segments:
        if not merged:
            merged.append(seg)
        else:
            if abs(merged[-1][0] - seg[0]) <= RLE_ROUND_W:
                # Merge into previous
                tot_n = merged[-1][1] + seg[1]
                avg_lvl = int(round((merged[-1][0] * merged[-1][1] + seg[0] * seg[1]) / (tot_n * 10.0)) * 10)
                merged[-1] = [avg_lvl, tot_n]
            else:
                merged.append(seg)
    segments = merged

    # Compress shortest segments until <= RLE_MAX_SEGMENTS
    while len(segments) > RLE_MAX_SEGMENTS:
        # Find segment with minimum n
        min_idx = int(np.argmin([s[1] for s in segments]))
        short_seg = segments.pop(min_idx)
        # Merge into neighbour (prefer adjacent with closest level)
        if min_idx == 0:
            target = 0
        elif min_idx >= len(segments):
            target = len(segments) - 1
        else:
            diff_left = abs(segments[min_idx - 1][0] - short_seg[0])
            diff_right = abs(segments[min_idx][0] - short_seg[0])
            target = min_idx - 1 if diff_left <= diff_right else min_idx

        tot_n = segments[target][1] + short_seg[1]
        avg_lvl = int(round((segments[target][0] * segments[target][1] + short_seg[0] * short_seg[1]) / (tot_n * 10.0)) * 10)
        segments[target] = [avg_lvl, tot_n]

    return segments


def extract_sessions(
    r: np.ndarray,
    step_s: float,
    threshold: float,
    *,
    enabled: bool = True,
    enabled_event_types: Sequence[str | EventType] | None = None,
) -> tuple[dict[str, np.ndarray], list[list[list[int]]]]:
    """Extract above-baseline sessions and RLE plateau encodings for cycle_shape appliances."""
    empty_table = {
        "start_i": np.empty(0, dtype=np.int64),
        "end_i": np.empty(0, dtype=np.int64),
        "rise_w": np.empty(0, dtype=np.float64),
        "fall_w": np.empty(0, dtype=np.float64),
        "dur_s": np.empty(0, dtype=np.float64),
        "plateau_w": np.empty(0, dtype=np.float64),
        "energy_wh": np.empty(0, dtype=np.float64),
        "peak_w": np.empty(0, dtype=np.float64),
        "n_segments": np.empty(0, dtype=np.int32),
        "kind": np.empty(0, dtype=np.int32),
    }
    is_cycle_enabled = enabled
    if enabled_event_types is not None:
        is_cycle_enabled = (
            enabled
            and (
                EventType.CYCLE_SHAPE in enabled_event_types
                or EventType.CYCLE_SHAPE.value in enabled_event_types
                or "cycle_shape" in enabled_event_types
            )
        )
    if not is_cycle_enabled or len(r) == 0:
        return empty_table, []

    finite = np.isfinite(r)
    # Segments of finite samples
    n = len(r)
    segments: list[tuple[int, int]] = []
    in_seg = False
    seg_start = 0
    for i in range(n):
        if finite[i]:
            if not in_seg:
                in_seg = True
                seg_start = i
        else:
            if in_seg:
                in_seg = False
                segments.append((seg_start, i))
    if in_seg:
        segments.append((seg_start, n))

    win_samples = max(3, int(round(PERCENTILE_WINDOW_S / step_s)))
    session_list: list[dict[str, float | int]] = []
    rle_list: list[list[list[int]]] = []
    gap_samples_max = int(round(SESSION_GAP_S / step_s))

    for seg_start, seg_end in segments:
        r_run = r[seg_start:seg_end]
        if len(r_run) < 2:
            continue

        filt_size = min(win_samples, len(r_run))
        baseline = scipy.ndimage.percentile_filter(r_run, percentile=10, size=filt_size, mode="nearest")
        excess = r_run - baseline
        active = excess > threshold

        # Find continuous spans of active
        spans: list[tuple[int, int]] = []
        in_span = False
        span_s = 0
        for idx_rel in range(len(r_run)):
            if active[idx_rel]:
                if not in_span:
                    in_span = True
                    span_s = idx_rel
            else:
                if in_span:
                    in_span = False
                    spans.append((span_s, idx_rel - 1))
        if in_span:
            spans.append((span_s, len(r_run) - 1))

        if not spans:
            continue

        # Merge spans if gap is < SESSION_GAP_S
        merged_spans: list[tuple[int, int]] = []
        for s_span in spans:
            if not merged_spans:
                merged_spans.append(s_span)
            else:
                prev_s, prev_e = merged_spans[-1]
                gap = s_span[0] - prev_e - 1
                if gap <= gap_samples_max:
                    merged_spans[-1] = (prev_s, s_span[1])
                else:
                    merged_spans.append(s_span)

        # Build session features
        for s_rel, e_rel in merged_spans:
            abs_start = seg_start + s_rel
            abs_end = seg_start + e_rel
            span_power = r[abs_start : abs_end + 1]
            b_power = baseline[s_rel : e_rel + 1]
            above = np.maximum(span_power - b_power, 0.0)

            dur_s = float((e_rel - s_rel + 1) * step_s)
            energy_wh = float(np.sum(above) * (step_s / 3600.0))
            peak_w = float(np.max(above)) if len(above) > 0 else 0.0
            plateau_w = float(np.median(above)) if len(above) > 0 else 0.0

            rle = _encode_rle(above)
            rle_list.append(rle)

            session_list.append(
                {
                    "start_i": abs_start,
                    "end_i": abs_end,
                    "rise_w": 0.0,
                    "fall_w": 0.0,
                    "dur_s": dur_s,
                    "plateau_w": plateau_w,
                    "energy_wh": energy_wh,
                    "peak_w": peak_w,
                    "n_segments": len(rle),
                    "kind": 1,
                }
            )

    if not session_list:
        return empty_table, []

    table: dict[str, np.ndarray] = {
        "start_i": np.array([s["start_i"] for s in session_list], dtype=np.int64),
        "end_i": np.array([s["end_i"] for s in session_list], dtype=np.int64),
        "rise_w": np.array([s["rise_w"] for s in session_list], dtype=np.float64),
        "fall_w": np.array([s["fall_w"] for s in session_list], dtype=np.float64),
        "dur_s": np.array([s["dur_s"] for s in session_list], dtype=np.float64),
        "plateau_w": np.array([s["plateau_w"] for s in session_list], dtype=np.float64),
        "energy_wh": np.array([s["energy_wh"] for s in session_list], dtype=np.float64),
        "peak_w": np.array([s["peak_w"] for s in session_list], dtype=np.float64),
        "n_segments": np.array([s["n_segments"] for s in session_list], dtype=np.int32),
        "kind": np.array([s["kind"] for s in session_list], dtype=np.int32),
    }

    return table, rle_list


def plug_sessions(
    p: np.ndarray,
    step_s: float,
) -> tuple[dict[str, np.ndarray], list[list[list[int]]]]:
    """Extract labelled reference sessions and RLE plateau encodings for a plug-metered channel (LRN-02)."""
    empty_table = {
        "start_i": np.empty(0, dtype=np.int64),
        "end_i": np.empty(0, dtype=np.int64),
        "rise_w": np.empty(0, dtype=np.float64),
        "fall_w": np.empty(0, dtype=np.float64),
        "dur_s": np.empty(0, dtype=np.float64),
        "plateau_w": np.empty(0, dtype=np.float64),
        "energy_wh": np.empty(0, dtype=np.float64),
        "peak_w": np.empty(0, dtype=np.float64),
        "n_segments": np.empty(0, dtype=np.int32),
        "kind": np.empty(0, dtype=np.int32),
    }
    if len(p) == 0:
        return empty_table, []

    finite = np.isfinite(p)
    finite_vals = p[finite]
    if len(finite_vals) == 0:
        return empty_table, []

    standby_p10 = float(np.percentile(finite_vals, 10))
    threshold = max(PLUG_ON_W, standby_p10 + PLUG_STANDBY_MARGIN_W)

    n = len(p)
    segments: list[tuple[int, int]] = []
    in_seg = False
    seg_start = 0
    for i in range(n):
        if finite[i]:
            if not in_seg:
                in_seg = True
                seg_start = i
        else:
            if in_seg:
                in_seg = False
                segments.append((seg_start, i))
    if in_seg:
        segments.append((seg_start, n))

    session_list: list[dict[str, float | int]] = []
    rle_list: list[list[list[int]]] = []
    gap_samples_max = int(round(PLUG_GAP_S / step_s))

    for seg_start, seg_end in segments:
        p_run = p[seg_start:seg_end]
        if len(p_run) == 0:
            continue

        active = p_run > threshold
        spans: list[tuple[int, int]] = []
        in_span = False
        span_s = 0
        for idx_rel in range(len(p_run)):
            if active[idx_rel]:
                if not in_span:
                    in_span = True
                    span_s = idx_rel
            else:
                if in_span:
                    in_span = False
                    spans.append((span_s, idx_rel - 1))
        if in_span:
            spans.append((span_s, len(p_run) - 1))

        if not spans:
            continue

        merged_spans: list[tuple[int, int]] = []
        for s_span in spans:
            if not merged_spans:
                merged_spans.append(s_span)
            else:
                prev_s, prev_e = merged_spans[-1]
                gap = s_span[0] - prev_e - 1
                if gap <= gap_samples_max:
                    merged_spans[-1] = (prev_s, s_span[1])
                else:
                    merged_spans.append(s_span)

        for s_rel, e_rel in merged_spans:
            abs_start = seg_start + s_rel
            abs_end = seg_start + e_rel
            span_power = p[abs_start : abs_end + 1]

            dur_s = float((e_rel - s_rel + 1) * step_s)
            energy_wh = float(np.sum(span_power) * (step_s / 3600.0))
            peak_w = float(np.max(span_power)) if len(span_power) > 0 else 0.0
            plateau_w = float(np.median(span_power)) if len(span_power) > 0 else 0.0

            rle = _encode_rle(span_power)
            rle_list.append(rle)

            session_list.append(
                {
                    "start_i": abs_start,
                    "end_i": abs_end,
                    "rise_w": 0.0,
                    "fall_w": 0.0,
                    "dur_s": dur_s,
                    "plateau_w": plateau_w,
                    "energy_wh": energy_wh,
                    "peak_w": peak_w,
                    "n_segments": len(rle),
                    "kind": 1,
                }
            )

    if not session_list:
        return empty_table, []

    table: dict[str, np.ndarray] = {
        "start_i": np.array([s["start_i"] for s in session_list], dtype=np.int64),
        "end_i": np.array([s["end_i"] for s in session_list], dtype=np.int64),
        "rise_w": np.array([s["rise_w"] for s in session_list], dtype=np.float64),
        "fall_w": np.array([s["fall_w"] for s in session_list], dtype=np.float64),
        "dur_s": np.array([s["dur_s"] for s in session_list], dtype=np.float64),
        "plateau_w": np.array([s["plateau_w"] for s in session_list], dtype=np.float64),
        "energy_wh": np.array([s["energy_wh"] for s in session_list], dtype=np.float64),
        "peak_w": np.array([s["peak_w"] for s in session_list], dtype=np.float64),
        "n_segments": np.array([s["n_segments"] for s in session_list], dtype=np.int32),
        "kind": np.array([s["kind"] for s in session_list], dtype=np.int32),
    }

    return table, rle_list


def compute_features(
    table: dict[str, np.ndarray],
    t0_ms: int,
    step_ms: int,
    day_bounds_ms: Sequence[int],
    first_weekday: int,
    phases: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Augment episode table with hour, dow, and per-phase shares."""
    n = len(table["start_i"])
    if n == 0:
        return {
            **table,
            "hour": np.empty(0, dtype=np.int32),
            "dow": np.empty(0, dtype=np.int32),
            "l1_share": np.empty(0, dtype=np.float64),
            "l2_share": np.empty(0, dtype=np.float64),
            "l3_share": np.empty(0, dtype=np.float64),
        }

    start_ts = t0_ms + table["start_i"] * step_ms
    day_bounds = np.array(day_bounds_ms, dtype=np.int64)
    day_indices = np.searchsorted(day_bounds, start_ts, side="right") - 1

    hour_arr = np.zeros(n, dtype=np.int32)
    dow_arr = np.zeros(n, dtype=np.int32)

    for i in range(n):
        d = max(0, min(len(day_bounds_ms) - 2, int(day_indices[i])))
        d_start = day_bounds_ms[d]
        hour_arr[i] = int(min(23, max(0, (start_ts[i] - d_start) // 3_600_000)))
        dow_arr[i] = int((first_weekday + d) % 7)

    has_phases = bool(phases and "l1" in phases and "l2" in phases and "l3" in phases)
    l1_share = np.full(n, np.nan, dtype=np.float64)
    l2_share = np.full(n, np.nan, dtype=np.float64)
    l3_share = np.full(n, np.nan, dtype=np.float64)

    if has_phases:
        l1 = phases["l1"]
        l2 = phases["l2"]
        l3 = phases["l3"]
        for i in range(n):
            idx = int(table["start_i"][i])
            if idx + 1 < len(l1):
                d1 = abs(l1[idx + 1] - l1[idx])
                d2 = abs(l2[idx + 1] - l2[idx])
                d3 = abs(l3[idx + 1] - l3[idx])
                tot = d1 + d2 + d3
                if tot > 0:
                    l1_share[i] = d1 / tot
                    l2_share[i] = d2 / tot
                    l3_share[i] = d3 / tot

    return {
        **table,
        "hour": hour_arr,
        "dow": dow_arr,
        "l1_share": l1_share,
        "l2_share": l2_share,
        "l3_share": l3_share,
    }

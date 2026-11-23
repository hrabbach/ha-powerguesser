"""Local stage-template fitting for program artefacts (D-27, LRN-03).

Stage structure is fitted locally in the niced worker from the train-block
members' run-length-encoded levels: the member with median energy is the
reference, every member is aligned to it with a banded DTW, and each reference
segment collects aligned samples and per-member dwell counts.  The LLM never
emits stage numbers; the proposal schema carries no sequence field.

numpy only.  All constants are initial values tuned post-ship on real data
(D-22).  ``WIDEN_FRAC`` mirrors ``compile.WIDEN_FRAC`` (importing compile here
is circular).
"""

from collections.abc import Sequence
import math

import numpy as np

from .episodes import MIN_EDGE_W

DTW_BAND_FRAC = 0.1  # initial value (D-22): Sakoe-Chiba band as a fraction of the reference
STAGE_MIN_SAMPLES = 2  # initial value (D-22): segments shorter than this merge into a neighbour
STAGE_OPTIONAL_FRACTION = 0.7  # initial value (D-22): below this member share a stage is optional
MAX_STAGES = 12  # initial value (D-22): closest-level neighbours merge beyond this

_WIDEN_FRAC = 0.05  # same as compile.WIDEN_FRAC (circular import if taken from there)


def expand_rle(levels: Sequence[float], counts: Sequence[int]) -> np.ndarray:
    """Expand an RLE (levels, counts) pair into a piecewise-constant array."""
    return np.repeat(np.asarray(levels, dtype=np.float64), np.asarray(counts, dtype=np.int64))


def _segment_reference(ref: np.ndarray, threshold_w: float) -> list[tuple[int, int]]:
    """Split the reference into segments where the level jumps by at least half an edge.

    Total for any non-empty array: a reference shorter than
    STAGE_MIN_SAMPLES cannot be split and cannot be merged into a
    neighbour, so it keeps its single whole-array segment (CR-01).
    """
    seg_min = max(threshold_w / 2.0, MIN_EDGE_W / 2.0)
    boundaries = [0]
    for i in range(1, len(ref)):
        if abs(ref[i] - ref[i - 1]) >= seg_min:
            boundaries.append(i)
    boundaries.append(len(ref))
    segs = [(boundaries[k], boundaries[k + 1]) for k in range(len(boundaries) - 1)]

    # Merge segments shorter than STAGE_MIN_SAMPLES into the neighbour whose
    # level is closer (a 1-sample glitch must never become a stage).
    while True:
        short = next((k for k, (s, e) in enumerate(segs) if e - s < STAGE_MIN_SAMPLES), None)
        if short is None:
            break
        if len(segs) < 2:
            # A degenerate reference (shorter than STAGE_MIN_SAMPLES in
            # total) has no neighbour to merge into: keep its one segment
            # instead of reading a nonexistent segs[1] (CR-01).
            break
        s, e = segs[short]
        level = float(np.mean(ref[s:e]))
        if short == 0:
            target = 1
        elif short == len(segs) - 1:
            target = short - 1
        else:
            ps, pe = segs[short - 1]
            ns, ne = segs[short + 1]
            prev_d = abs(level - float(np.mean(ref[ps:pe])))
            next_d = abs(level - float(np.mean(ref[ns:ne])))
            target = short - 1 if prev_d <= next_d else short + 1
        ts, te = segs[target]
        merged = (min(s, ts), max(e, te))
        lo_i, hi_i = sorted((short, target))
        segs = segs[:lo_i] + [merged] + segs[hi_i + 1 :]

    # Cap the stage count by merging the closest-level adjacent neighbours.
    while len(segs) > MAX_STAGES:
        best_k, best_d = 0, math.inf  # type: ignore[name-defined]
        for k in range(len(segs) - 1):
            s0, e0 = segs[k]
            s1, e1 = segs[k + 1]
            d = abs(float(np.mean(ref[s0:e0])) - float(np.mean(ref[s1:e1])))
            if d < best_d:
                best_k, best_d = k, d
        s0, e0 = segs[best_k]
        _, e1 = segs[best_k + 1]
        segs = segs[:best_k] + [(s0, e1)] + segs[best_k + 2 :]
    return segs


def _banded_dtw_path(ref: np.ndarray, mem: np.ndarray, band: int) -> list[tuple[int, int]]:
    """Banded DTW alignment of mem onto ref; returns the (ref_i, mem_j) path.

    The band is a performance guard, not a correctness bound: when a member
    misses a whole stage it can fall outside the initial band, so the band
    doubles until the path reaches the bottom-right cell.
    """
    n, m = len(ref), len(mem)
    if n == 0 or m == 0:
        return []
    band = min(max(4, int(band)), max(n, m))
    cost: np.ndarray | None = None
    while True:
        cost = np.full((n + 1, m + 1), np.inf)
        cost[0, 0] = 0.0
        for i in range(1, n + 1):
            j_lo = max(1, i - band)
            j_hi = min(m, i + band)
            row_prev = cost[i - 1]
            row = cost[i]
            for j in range(j_lo, j_hi + 1):
                d = abs(ref[i - 1] - mem[j - 1])
                row[j] = d + min(row_prev[j - 1], row_prev[j], row[j - 1])
        if np.isfinite(cost[n, m]) or band >= max(n, m):
            break
        band *= 2
    assert cost is not None
    path: list[tuple[int, int]] = []
    i, j = n, m
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        diag, up, left = cost[i - 1, j - 1], cost[i - 1, j], cost[i, j - 1]
        if diag <= up and diag <= left:
            i, j = i - 1, j - 1
        elif up <= left:
            i -= 1
        else:
            j -= 1
    path.reverse()
    return path


def _widen(p5: float, p50: float, p95: float, threshold_w: float) -> dict[str, float]:
    """p5/p50/p95 box widened by max(threshold_w / 2, WIDEN_FRAC x |p|) per side."""
    lo = p5 - max(threshold_w / 2.0, _WIDEN_FRAC * abs(p5))
    hi = p95 + max(threshold_w / 2.0, _WIDEN_FRAC * abs(p95))
    return {"lo": round(float(lo), 1), "p50": round(float(p50), 1), "hi": round(float(hi), 1)}


def fit_stages(
    members: Sequence[np.ndarray],
    threshold_w: float,
    grid_s: float,
) -> dict[str, object] | None:
    """Fit a stage template from the expanded per-run level arrays.

    Returns ``{"stages": [...], "terminal": {"fall_w": ...}}`` or None when
    fewer than three members exist, a member is empty, or no member is
    long enough (STAGE_MIN_SAMPLES) to serve as a segmentable reference.
    """
    arrays = [np.asarray(m, dtype=np.float64) for m in members]
    if len(arrays) < 3:
        return None
    if any(len(a) == 0 for a in arrays):
        return None
    # Reference: the median-energy member among those with the richest
    # segmentation, so a member that skips a stage cannot become the
    # reference and hide that stage; ties go to the lowest member index.
    seg_lists = [_segment_reference(a, threshold_w) for a in arrays]
    max_segs = max(len(s) for s in seg_lists)
    # A member shorter than STAGE_MIN_SAMPLES has no stage structure (a
    # program session episode may legitimately be one sample long), so it is
    # never the reference — it is still aligned below like every other
    # member. With no segmentable member there is nothing to fit (CR-01).
    candidates = [
        i
        for i, s in enumerate(seg_lists)
        if len(s) == max_segs and len(arrays[i]) >= STAGE_MIN_SAMPLES
    ]
    if not candidates:
        return None
    energies = [float(np.sum(arrays[i])) for i in candidates]
    order = sorted(range(len(candidates)), key=lambda k: (energies[k], k))
    ref_pos = candidates[order[len(order) // 2]]
    reference = arrays[ref_pos]
    segments = seg_lists[ref_pos]
    n_members = len(arrays)
    band = max(4, int(np.ceil(DTW_BAND_FRAC * len(reference))))
    paths = [_banded_dtw_path(reference, a, band) for a in arrays]

    stages: list[dict[str, object]] = []
    prev_p50 = 0.0
    for si, (s_start, s_end) in enumerate(segments):
        # Per member: the distinct own samples aligned into this segment.
        aligned: list[tuple[int, list[float]]] = []
        for arr, path in zip(arrays, paths):
            js = sorted({j for (i, j) in path if s_start <= i < s_end})
            if len(js) >= STAGE_MIN_SAMPLES:
                aligned.append((len(js), [float(arr[j]) for j in js]))
        # Membership by per-member median level: a member that skipped the
        # stage only bleeds tail samples into its DTW alignment, so its
        # median still sits at the neighbouring stage's level.  The stage
        # centre is the median of member medians (robust to that bleed).
        if aligned:
            member_medians = np.asarray(
                [float(np.median(np.asarray(vals))) for _, vals in aligned]
            )
            centre = float(np.median(member_medians))
            half = max(threshold_w, 0.25 * abs(centre))
            qualifying = [
                (dwell, vals)
                for (dwell, vals), med in zip(aligned, member_medians)
                if abs(med - centre) <= half
            ]
        else:
            qualifying = []
        if not qualifying:
            # No member's per-member median lies within ``half`` of the centre:
            # an even number of aligned members with far-apart medians, or a
            # member that skipped the stage and only bled tail samples into its
            # DTW alignment.  Fit the stage from every aligned sample rather
            # than taking a percentile of an empty array, which raises
            # IndexError and fails the whole learning run (WR-02).
            qualifying = aligned
        if not qualifying:
            # Nothing aligned into this segment at all: it carries no evidence,
            # so it is left out instead of being fitted from an empty
            # percentile.
            continue
        n_qualifying = len(qualifying)
        level_samples = [v for _, vals in qualifying for v in vals]
        dwells = [float(dwell) * float(grid_s) for dwell, _ in qualifying]
        lvl = np.asarray(level_samples, dtype=np.float64)
        l_p5, l_p50, l_p95 = np.percentile(lvl, [5, 50, 95])
        dw = np.asarray(dwells, dtype=np.float64)
        d_p5, d_p50, d_p95 = np.percentile(dw, [5, 50, 95])
        entry = float(l_p50) - prev_p50
        e_widen = max(threshold_w / 2.0, _WIDEN_FRAC * abs(entry))
        stages.append(
            {
                "name": f"s{si + 1}",
                "level_w": _widen(float(l_p5), float(l_p50), float(l_p95), threshold_w),
                "entry_w": {
                    "lo": round(entry - e_widen, 1),
                    "p50": round(entry, 1),
                    "hi": round(entry + e_widen, 1),
                },
                "dur_s": {
                    "lo": round(max(float(grid_s), float(d_p5)), 1),
                    "p50": round(float(d_p50), 1),
                    "hi": round(float(d_p95), 1),
                },
                "optional": n_qualifying < STAGE_OPTIONAL_FRACTION * n_members,
            }
        )
        prev_p50 = float(l_p50)

    last_required = max(
        (k for k, st in enumerate(stages) if not st["optional"]),
        default=len(stages) - 1,
    )
    last_level = stages[last_required]["level_w"]
    assert isinstance(last_level, dict)
    return {
        "stages": stages,
        "terminal": {
            "fall_w": {
                "lo": round(-float(last_level["hi"]), 1),
                "p50": round(-float(last_level["p50"]), 1),
                "hi": round(-float(last_level["lo"]), 1),
            }
        },
    }

"""Coverage calculation and span operations for PowerGuesser."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

MS_PER_DAY = 86_400_000


@dataclass(frozen=True, slots=True)
class ChannelCoverage:
    """Coverage summary for a single channel."""

    days: float
    first_ms: int | None
    last_ms: int | None
    gap_count: int


def segments(
    rows: Sequence[tuple[int, int | None]],
    start_ms: int,
    end_ms: int,
) -> list[tuple[int, int]]:
    """Return merged usable spans [s, e) clipped to [start_ms, end_ms).

    Each non-NULL row starts a usable span that ends at the next row if that
    row is NULL, else continues; the last non-NULL row extends to end_ms.
    Spans are clipped to [start_ms, end_ms) and merged.
    """
    if end_ms <= start_ms or not rows:
        return []

    sorted_rows = sorted(rows, key=lambda r: r[0])
    spans: list[tuple[int, int]] = []
    current_start: int | None = None

    for ts, w in sorted_rows:
        if w is not None:
            if current_start is None:
                current_start = ts
        else:
            if current_start is not None:
                spans.append((current_start, ts))
                current_start = None

    if current_start is not None:
        spans.append((current_start, max(current_start, end_ms)))

    # Clip to [start_ms, end_ms)
    clipped: list[tuple[int, int]] = []
    for s, e in spans:
        cs = max(s, start_ms)
        ce = min(e, end_ms)
        if cs < ce:
            clipped.append((cs, ce))

    if not clipped:
        return []

    clipped.sort()
    merged: list[tuple[int, int]] = [clipped[0]]
    for s, e in clipped[1:]:
        prev_s, prev_e = merged[-1]
        if s <= prev_e:
            merged[-1] = (prev_s, max(prev_e, e))
        else:
            merged.append((s, e))

    return merged


def intersect(
    segment_lists: Sequence[Sequence[tuple[int, int]]],
) -> list[tuple[int, int]]:
    """Return the common spans across all segment lists."""
    if not segment_lists or any(not sl for sl in segment_lists):
        return []

    result = list(segment_lists[0])
    for sl in segment_lists[1:]:
        new_result: list[tuple[int, int]] = []
        i = 0
        j = 0
        while i < len(result) and j < len(sl):
            s1, e1 = result[i]
            s2, e2 = sl[j]
            start = max(s1, s2)
            end = min(e1, e2)
            if start < end:
                new_result.append((start, end))
            if e1 < e2:
                i += 1
            else:
                j += 1
        result = new_result
        if not result:
            break
    return result


def channel_coverage(
    rows: Sequence[tuple[int, int | None]],
    start_ms: int,
    end_ms: int,
) -> ChannelCoverage:
    """Compute coverage statistics for a channel inside [start_ms, end_ms)."""
    segs = segments(rows, start_ms, end_ms)
    total_span_ms = sum(e - s for s, e in segs)
    days = total_span_ms / MS_PER_DAY

    window_non_null_ts = [
        r[0]
        for r in rows
        if r[1] is not None and start_ms <= r[0] < end_ms
    ]
    first_ms = window_non_null_ts[0] if window_non_null_ts else None
    last_ms = window_non_null_ts[-1] if window_non_null_ts else None

    gap_count = 0
    prev_w: int | None = None
    sorted_rows = sorted(rows, key=lambda r: r[0])
    for _, w in sorted_rows:
        if prev_w is not None and w is None:
            gap_count += 1
        prev_w = w

    return ChannelCoverage(
        days=days,
        first_ms=first_ms,
        last_ms=last_ms,
        gap_count=gap_count,
    )


def usable_days(
    segments_by_channel: Mapping[int, Sequence[tuple[int, int]]],
    roles: Mapping[int, str],
) -> float:
    """Compute length of usable history in days from gross channels.

    Intersection is taken over the channels whose role is not 'solar'
    (P1 D-20: gross is computed with unavailable solar as 0 W, so solar
    NULLs do not make time unusable), or over the solar channels when
    every gross channel is solar; 0.0 for no channels.

    This is the learning gate's figure (BufferDB.coverage_for). Extract counts
    data days differently: split.data_days flags a calendar day usable when at
    least MIN_DAY_COVERAGE (0.5) of its residual samples are finite. The two
    therefore agree on a whole day that is fully present or fully NULL in every
    non-solar channel, but disagree on a partial day: this sums the intersected
    non-solar segment time over the window, so a day with 60 % coverage counts
    0.6 here, while data_days counts it as 1. A solar channel that is NULL keeps
    a day usable on both sides (gross_on_read maps NULL solar to 0 W), which is
    why the gate never intersects solar. tests/engine/test_coverage.py pins the
    solar-only and grid-only outage parity.
    """
    if not roles or not segments_by_channel:
        return 0.0

    gross_channels = list(roles.keys())
    if not gross_channels:
        return 0.0

    all_solar = all(roles[ch] == "solar" for ch in gross_channels)
    if all_solar:
        target_channels = gross_channels
    else:
        target_channels = [ch for ch in gross_channels if roles[ch] != "solar"]

    if not target_channels:
        return 0.0

    lists_to_intersect = [
        segments_by_channel.get(ch, []) for ch in target_channels
    ]
    common = intersect(lists_to_intersect)
    total_ms = sum(e - s for s, e in common)
    return total_ms / MS_PER_DAY

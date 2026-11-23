"""Recorder history conversion and gap reconstruction for PowerGuesser buffer."""

from collections.abc import Sequence
import math

GAP_FACTOR = 3.0


def rows_from_history(
    states: Sequence[tuple[str, float, float]],
    *,
    ch: int,
    unit_factor: float,
    mode: str,
    interval_s: float,
    live_lr_s: float | None = None,
    end_s: float | None = None,
    gap_markers: bool = True,
) -> list[tuple[int, int, int | None]]:
    """Convert recorder state tuples to buffer rows and gap markers.

    States are (state_str, last_updated_s, last_reported_s).
    gap_markers=False (metered plugs, held with no age limit per P1 D-07) emits no
    synthesized staleness NULLs; unavailable/unknown states still become NULL rows.
    Duplicate timestamps within a single call are collapsed with last-wins,
    so rows returned never contain duplicates for one (ts, ch) and stored
    row equals engine output. INSERT OR IGNORE only arbitrates against rows
    already in the buffer, where live rows win (D-08).
    """
    if not states:
        return []

    emitted: list[tuple[int, int, int | None]] = []
    n = len(states)

    for i, (state_str, lu_s, lr_s) in enumerate(states):
        is_last = i == n - 1
        try:
            val = float(state_str)
            w = round(val * unit_factor) if math.isfinite(val) else None
        except (ValueError, TypeError):
            w = None

        lu_ms = int(round(lu_s * 1000))
        emitted.append((lu_ms, ch, w))

        effective_lr_s = lr_s
        if is_last and live_lr_s is not None and live_lr_s > effective_lr_s:
            effective_lr_s = live_lr_s

        if w is not None and effective_lr_s > lu_s:
            lr_ms = int(round(effective_lr_s * 1000))
            emitted.append((lr_ms, ch, w))

        if gap_markers and mode == "re_reporting" and w is not None and i < n - 1:
            next_lu_s = states[i + 1][1]
            gap_threshold_s = effective_lr_s + GAP_FACTOR * interval_s
            if next_lu_s > gap_threshold_s:
                gap_ms = int(round(gap_threshold_s * 1000))
                emitted.append((gap_ms, ch, None))

    if (
        gap_markers
        and mode == "re_reporting"
        and states
        and emitted
        and emitted[-1][2] is not None
        and end_s is not None
    ):
        gap_threshold_s = effective_lr_s + GAP_FACTOR * interval_s
        if end_s > gap_threshold_s:
            gap_ms = int(round(gap_threshold_s * 1000))
            emitted.append((gap_ms, ch, None))

    # Collapse duplicate timestamps: last wins
    row_by_ts: dict[int, tuple[int, int, int | None]] = {}
    for r in emitted:
        row_by_ts[r[0]] = r

    return [row_by_ts[t] for t in sorted(row_by_ts)]

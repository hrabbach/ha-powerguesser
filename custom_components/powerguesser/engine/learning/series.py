"""Read and reconstruct residual power series from buffer."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import math

import numpy as np

from ..reader import (
    gross_on_read,
    load_gross_layout,
    load_phase_sign,
    open_readonly,
    read_range,
    resample_zoh,
)

# A missing metered value holds its last known value this long, then counts as 0 W in the
# residual and NaN in the plug series (owner pick C, amends P3 A7 / P2 D-03 for metered
# channels in the learning path; debug extract-zero-data-days).
METERED_HOLD_MS = 300_000


@dataclass(frozen=True, slots=True)
class ResidualSeries:
    """Reconstructed residual and channel power series."""

    t0_ms: int
    step_ms: int
    gross: np.ndarray
    residual: np.ndarray
    plugs: dict[int, np.ndarray]
    phases: dict[str, np.ndarray]
    # Plan 06-09 (REV-04, D-25): the per-session plug series, one entry per
    # session channel passed via load_residual(session_windows=...), NaN outside
    # its own windows. Empty (and the residual untouched) when no windows are
    # given, so every existing caller stays byte-identical.
    sessions: dict[int, np.ndarray] = field(default_factory=dict)


def resample_hold(
    rows: Sequence[tuple[int, int | None]],
    start_ms: int,
    end_ms: int,
    step_ms: int,
    hold_ms: int,
) -> np.ndarray:
    """Zero-order hold where a run of NULL rows keeps the last non-NULL value for hold_ms.

    The hold counts from the first NULL row of the run, so later NULL rows of the same run
    do not restart it. Beyond the hold, before the first row, and where no non-NULL row
    precedes the run, the point is NaN. Rows must be sorted by ts and should start at
    least hold_ms before start_ms so a value that went missing just before is still held.
    """
    if end_ms <= start_ms:
        return np.empty(0, dtype=np.float64)
    num_points = math.ceil((end_ms - start_ms) / step_ms)
    out = np.full(num_points, np.nan, dtype=np.float64)
    if not rows:
        return out

    grid = start_ms + np.arange(num_points, dtype=np.int64) * step_ms
    row_ts = np.array([r[0] for r in rows], dtype=np.int64)
    row_w = np.array([np.nan if r[1] is None else float(r[1]) for r in rows], dtype=np.float64)
    pos = np.arange(len(rows))
    is_null = np.isnan(row_w)

    last_value = np.where(~is_null, pos, -1)
    np.maximum.accumulate(last_value, out=last_value)
    held_w = np.where(last_value >= 0, row_w[np.maximum(last_value, 0)], np.nan)

    run_first = np.where(is_null & np.r_[True, ~is_null[:-1]], pos, -1)
    np.maximum.accumulate(run_first, out=run_first)
    run_ts = row_ts[np.maximum(run_first, 0)]

    idx = np.searchsorted(row_ts, grid, side="right") - 1
    valid = idx >= 0
    j = idx[valid]
    keep = ~is_null[j] | (grid[valid] - run_ts[j] <= hold_ms)
    out[valid] = np.where(keep, held_w[j], np.nan)
    return out


def load_residual(
    buffer_path: str,
    start_ms: int,
    end_ms: int,
    step_ms: int,
    metered_channels: Sequence[int],
    phase_channels: Mapping[str, int],
    session_windows: Mapping[int, Sequence[tuple[int, int]]] | None = None,
) -> ResidualSeries:
    """Load gross and metered channels and compute the residual on read.

    Metered channels: 0 W before their first buffered row (A7 lifetime rule). Inside the
    lifetime a missing value (NULL rows) holds the last known value for at most
    METERED_HOLD_MS after the first NULL, then counts as 0 W in the residual, so a flaky
    or offline plug never blanks the residual; the returned plug series stays NaN beyond
    the hold so plug sessions get no fake off edges. Gross NaN still makes the residual NaN.

    Plan 06-09 (REV-04, D-25): ``session_windows`` maps a session channel to the
    half-open [start_ms, end_ms) windows of one or more plug sessions. Each such
    channel is read with the same metered hold rule, masked to NaN outside its
    windows, and subtracted from the residual inside them (a NaN inside a window
    counts as 0 W, like the metered hold rule); the residual outside every window
    is byte-identical to the call without windows. A channel that is also a
    metered channel is skipped (the metered rule wins), so it is never subtracted
    twice. The masked series is exposed under ``ResidualSeries.sessions``.
    """
    conn = open_readonly(buffer_path)
    try:
        layout = load_gross_layout(conn)
        needed_channels: set[int] = set()
        for _, _, ch_tuple in layout:
            needed_channels.update(ch_tuple)

        series: dict[int, np.ndarray] = {}
        for ch in needed_channels:
            rows_dict = read_range(conn, [ch], start_ms, end_ms)
            series[ch] = resample_zoh(rows_dict.get(ch, []), start_ms, end_ms, step_ms)

        gross = gross_on_read(series, layout)
        residual = gross.copy()
        num_points = len(gross)
        grid_ts = start_ms + np.arange(num_points, dtype=np.int64) * step_ms

        plugs: dict[int, np.ndarray] = {}
        for m_ch in metered_channels:
            # Check first row timestamp in the entire buffer for this channel
            cur = conn.execute("SELECT MIN(ts) FROM samples WHERE ch = ?", (m_ch,))
            min_row = cur.fetchone()
            first_buffered_ts = min_row[0] if (min_row and min_row[0] is not None) else None

            if first_buffered_ts is None:
                # No rows exist for this channel; treat as 0 W throughout
                m_held = np.zeros(num_points, dtype=np.float64)
            else:
                # Read one hold span early so a value that went missing just before
                # start_ms is still held into the window.
                rows_dict = read_range(conn, [m_ch], start_ms - METERED_HOLD_MS, end_ms)
                m_held = resample_hold(
                    rows_dict.get(m_ch, []), start_ms, end_ms, step_ms, METERED_HOLD_MS
                )
                # Before first buffered row: device was unmetered -> treat as 0 W
                before_mask = grid_ts < first_buffered_ts
                m_held[before_mask] = np.where(
                    np.isnan(m_held[before_mask]), 0.0, m_held[before_mask]
                )

            plugs[m_ch] = m_held
            residual = residual - np.where(np.isnan(m_held), 0.0, m_held)

        # Plan 06-09: window-bounded session subtraction. A channel already
        # metered is skipped so it is never subtracted twice (the metered rule
        # wins).
        sessions: dict[int, np.ndarray] = {}
        if session_windows:
            metered_set = set(metered_channels)
            for s_ch, windows in session_windows.items():
                if s_ch in metered_set:
                    continue
                cur = conn.execute("SELECT MIN(ts) FROM samples WHERE ch = ?", (s_ch,))
                min_row = cur.fetchone()
                first_buffered_ts = (
                    min_row[0] if (min_row and min_row[0] is not None) else None
                )
                if first_buffered_ts is None:
                    # No samples for this channel: nothing to subtract or expose.
                    continue
                rows_dict = read_range(conn, [s_ch], start_ms - METERED_HOLD_MS, end_ms)
                s_held = resample_hold(
                    rows_dict.get(s_ch, []), start_ms, end_ms, step_ms, METERED_HOLD_MS
                )
                in_window = np.zeros(num_points, dtype=bool)
                for win_start, win_end in windows:
                    in_window |= (grid_ts >= int(win_start)) & (grid_ts < int(win_end))
                s_masked = np.where(in_window, s_held, np.nan)
                sessions[s_ch] = s_masked
                residual = residual - np.where(np.isnan(s_masked), 0.0, s_masked)

        phases: dict[str, np.ndarray] = {}
        if phase_channels:
            phase_sign = load_phase_sign(conn)
            for phase_name, p_ch in phase_channels.items():
                rows_dict = read_range(conn, [p_ch], start_ms, end_ms)
                p_resampled = resample_zoh(rows_dict.get(p_ch, []), start_ms, end_ms, step_ms)
                phases[phase_name] = p_resampled * float(phase_sign)

        return ResidualSeries(
            t0_ms=start_ms,
            step_ms=step_ms,
            gross=gross,
            residual=residual,
            plugs=plugs,
            phases=phases,
            sessions=sessions,
        )
    finally:
        conn.close()

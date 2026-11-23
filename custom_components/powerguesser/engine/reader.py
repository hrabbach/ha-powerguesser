"""Read operations, resampling, and gross reconstruction for the buffer."""

from collections.abc import Mapping, Sequence
import json
import math
import pathlib
import sqlite3

import numpy as np


def open_readonly(path: str) -> sqlite3.Connection:
    """Open a SQLite database in read-only mode."""
    uri = pathlib.Path(path).resolve().as_uri() + "?mode=ro"
    return sqlite3.connect(uri, uri=True)


def read_range(
    conn: sqlite3.Connection,
    channels: Sequence[int],
    start_ms: int,
    end_ms: int,
) -> dict[int, list[tuple[int, int | None]]]:
    """Read samples for each channel in [start_ms, end_ms) plus the last sample before start_ms."""
    result: dict[int, list[tuple[int, int | None]]] = {}
    for ch in channels:
        seed_cursor = conn.execute(
            "SELECT ts, w FROM samples WHERE ch = ? AND ts < ? ORDER BY ts DESC LIMIT 1",
            (ch, start_ms),
        )
        seed = seed_cursor.fetchone()
        range_cursor = conn.execute(
            "SELECT ts, w FROM samples WHERE ch = ? AND ts >= ? AND ts < ? ORDER BY ts ASC",
            (ch, start_ms, end_ms),
        )
        range_rows = range_cursor.fetchall()
        rows: list[tuple[int, int | None]] = []
        if seed is not None:
            rows.append((seed[0], seed[1]))
        rows.extend(range_rows)
        result[ch] = rows
    return result


def resample_zoh(
    rows: Sequence[tuple[int, int | None]],
    start_ms: int,
    end_ms: int,
    step_ms: int,
) -> np.ndarray:
    """Resample samples to a regular grid using zero-order hold.

    Holding stops at the next row of any kind. NaN, never 0, marks missing data or gaps.
    """
    if end_ms <= start_ms:
        return np.empty(0, dtype=np.float64)
    num_points = math.ceil((end_ms - start_ms) / step_ms)
    if not rows:
        return np.full(num_points, np.nan, dtype=np.float64)

    grid = start_ms + np.arange(num_points, dtype=np.int64) * step_ms
    row_ts = np.array([r[0] for r in rows], dtype=np.int64)
    row_w = np.array([np.nan if r[1] is None else float(r[1]) for r in rows], dtype=np.float64)

    idx = np.searchsorted(row_ts, grid, side="right") - 1
    result = np.full(num_points, np.nan, dtype=np.float64)
    valid = idx >= 0
    result[valid] = row_w[idx[valid]]
    return result


def load_gross_layout(
    conn: sqlite3.Connection,
) -> list[tuple[str, str, tuple[int, ...]]]:
    """Load and parse the gross layout from meta table."""
    cursor = conn.execute("SELECT value FROM meta WHERE key = ?", ("gross_layout",))
    row = cursor.fetchone()
    if row is None or not row[0]:
        return []
    raw_data = json.loads(row[0])
    return [
        (item["role"], item["transform"], tuple(item["channels"]))
        for item in raw_data
    ]


def gross_on_read(
    series: Mapping[int, np.ndarray],
    layout: Sequence[tuple[str, str, tuple[int, ...]]],
) -> np.ndarray:
    """Reconstruct gross power from resampled channel series according to layout.

    A NULL solar row counts as 0 W whatever its cause (D-04 stores no gap reason),
    mirroring P1 D-20. This differs from the live aligner for a stale re-reporting solar
    where gross is made unavailable. A NULL grid, battery, or home row makes gross NaN.
    """
    if not layout:
        return np.empty(0, dtype=np.float64)

    first_ch = layout[0][2][0]
    length = len(series[first_ch])
    total = np.zeros(length, dtype=np.float64)

    for role, transform, ch_tuple in layout:
        if transform == "direct":
            val = series[ch_tuple[0]].copy()
        elif transform == "inverted":
            val = -series[ch_tuple[0]]
        elif transform == "from_minus_to":
            val = series[ch_tuple[0]] - series[ch_tuple[1]]
        else:
            raise ValueError(f"Unknown transform: {transform}")

        if role == "solar":
            val = np.where(np.isnan(val), 0.0, val)
            val = np.maximum(val, 0.0)

        total += val

    return total


def load_phase_sign(conn: sqlite3.Connection) -> int:
    """Return phase power multiplier (-1 if inverted, else 1) from meta table."""
    cursor = conn.execute("SELECT value FROM meta WHERE key = ?", ("phase_invert",))
    row = cursor.fetchone()
    if row is not None and row[0] == "1":
        return -1
    return 1


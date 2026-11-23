"""Whole-day train/validation/holdout split engine (D-08)."""

from collections.abc import Sequence
from dataclasses import dataclass
import math
import numpy as np

MIN_DAY_COVERAGE = 0.5


@dataclass(frozen=True, slots=True)
class DaySplit:
    """Whole-day split into train, validation, and holdout blocks (D-08)."""

    train: tuple[int, ...]
    val: tuple[int, ...]
    holdout: tuple[int, ...]


def data_days(
    residual: np.ndarray,
    t0_ms: int,
    step_ms: int,
    day_bounds_ms: Sequence[int],
) -> list[int]:
    """Identify day indices that have at least MIN_DAY_COVERAGE finite samples."""
    n_days = len(day_bounds_ms) - 1
    valid_days: list[int] = []

    for d in range(n_days):
        day_start = day_bounds_ms[d]
        day_end = day_bounds_ms[d + 1]

        idx_start = max(0, int(round((day_start - t0_ms) / step_ms)))
        idx_end = min(len(residual), int(round((day_end - t0_ms) / step_ms)))

        if idx_end <= idx_start:
            continue

        day_slice = residual[idx_start:idx_end]
        n_total = len(day_slice)
        n_finite = int(np.sum(np.isfinite(day_slice)))

        if n_total > 0 and (n_finite / n_total) >= MIN_DAY_COVERAGE:
            valid_days.append(d)

    return valid_days


def calendar_holdout_start(n_days: int, holdout_days: int) -> int:
    """First day index of the D-18 calendar holdout window.

    The holdout is the last holdout_days calendar days of an n_days window,
    gaps included: the window starts at max(0, n_days - holdout_days). Used by
    the validate stage (via the shadow extract in Plan 05-03) to anchor the
    holdout to the calendar, not to the count of valid days (Pitfall 4).
    """
    return max(0, n_days - holdout_days)


def split_days(
    days: Sequence[int],
    *,
    train_fraction: float = 0.7,
    holdout_days: int = 0,
    holdout_from_day: int | None = None,
    n_dropped: int = 0,
) -> DaySplit:
    """Split ordered data days into train, validation, and holdout sets (D-08).

    With holdout_from_day set (D-18, additive), the holdout is every given day
    whose index is at or after that calendar day index — the most recent
    calendar days, gaps included — and the remaining days split 70/30 as
    before. holdout_days (the last N *valid* days) and holdout_from_day are
    mutually exclusive; callers that pass nothing are unchanged.
    """
    if holdout_days > 0 and holdout_from_day is not None:
        raise ValueError("holdout_days and holdout_from_day are mutually exclusive")
    days_list = list(days)
    if holdout_from_day is not None:
        holdout = tuple(d for d in days_list if d >= holdout_from_day)
        remaining = [d for d in days_list if d < holdout_from_day]
    elif holdout_days > 0:
        if holdout_days >= len(days_list):
            holdout = tuple(days_list)
            remaining = []
        else:
            holdout = tuple(days_list[-holdout_days:])
            remaining = days_list[:-holdout_days]
    else:
        holdout = ()
        remaining = days_list

    n = len(remaining)
    if n < 2:
        raise ValueError(
            f"need at least 2 data days, have {n} ({n_dropped} dropped below 50 % coverage)"
        )

    n_train = max(1, min(n - 1, math.floor(train_fraction * n)))
    train = tuple(remaining[:n_train])
    val = tuple(remaining[n_train:])

    return DaySplit(train=train, val=val, holdout=holdout)


def block_of(
    start_ms_array: np.ndarray,
    day_bounds_ms: Sequence[int],
    split: DaySplit,
) -> np.ndarray:
    """Determine block id (0=train, 1=val, 2=holdout, -1=none) for each start timestamp."""
    n = len(start_ms_array)
    blocks = np.full(n, -1, dtype=np.int8)
    if n == 0:
        return blocks

    day_bounds = np.array(day_bounds_ms, dtype=np.int64)
    day_indices = np.searchsorted(day_bounds, start_ms_array, side="right") - 1

    train_set = set(split.train)
    val_set = set(split.val)
    holdout_set = set(split.holdout)

    for i in range(n):
        d = int(day_indices[i])
        if 0 <= d < len(day_bounds_ms) - 1:
            if d in train_set:
                blocks[i] = 0
            elif d in val_set:
                blocks[i] = 1
            elif d in holdout_set:
                blocks[i] = 2

    return blocks

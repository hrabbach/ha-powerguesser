"""Holdout-exclusion mask for capture windows (D-05, Plan 08-04).

numpy only; no Home Assistant import (enforced by
``tests/engine/test_no_ha_imports.py``). The validate and validate_sessions
stages use the mask to drop captured runs from their holdout metric masks so a
taught example can never sit in a validation holdout.
"""

from collections.abc import Sequence

import numpy as np


def holdout_exclusion_mask(
    grid_ts_ms: np.ndarray,
    windows: Sequence[Sequence[int]],
    pad_ms: int,
) -> np.ndarray:
    """Return a bool mask that is True inside any padded capture window.

    Each window is half-open on both sides: ``[start - pad_ms, end + pad_ms)`` —
    the start sample is included and the end sample is excluded. This single
    convention lives here: every caller passes the grid timestamps, the windows
    and the pad only, and never compares timestamps against the window bounds
    itself.
    """
    mask = np.zeros(len(grid_ts_ms), dtype=bool)
    for window in windows:
        start = int(window[0])
        end = int(window[1])
        mask |= (grid_ts_ms >= start - pad_ms) & (grid_ts_ms < end + pad_ms)
    return mask

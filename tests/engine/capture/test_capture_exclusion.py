"""D-05 holdout-exclusion tests: padded mask boundaries and the no-overlap invariant.

Plan 08-04 Task 1. The helper is pure numpy; the invariant test proves that for
random day bounds, random holdout day sets and random capture windows, no grid
sample lies in both a padded capture window and the final holdout metric mask.
"""

import numpy as np

from custom_components.powerguesser.engine.capture.exclusion import (
    holdout_exclusion_mask,
)

GRID_MS = 30_000
T0 = 1_700_000_000_000


def test_exclusion_mask_padding_boundaries() -> None:
    """A 30 s grid, a 300 s window and a 60 s pad: half-open [start - pad, end + pad)."""
    t = T0
    grid = t + np.arange(-10, 40, dtype=np.int64) * GRID_MS
    mask = holdout_exclusion_mask(grid, [[t, t + 300_000]], 60_000)

    assert len(mask) == len(grid)
    assert mask.dtype == bool
    for ts, flag in zip(grid, mask):
        expected = (t - 60_000) <= int(ts) < (t + 360_000)
        assert bool(flag) is expected

    # The boundary samples explicitly: start sample in, end sample out.
    by_ts = {int(ts): bool(flag) for ts, flag in zip(grid, mask)}
    assert by_ts[t - 60_000] is True  # inclusive start
    assert by_ts[t - 90_000] is False  # just before the pad
    assert by_ts[t + 330_000] is True  # last sample before the exclusive end
    assert by_ts[t + 360_000] is False  # exclusive end

    # An empty window list excludes nothing.
    empty = holdout_exclusion_mask(grid, [], 60_000)
    assert not bool(np.any(empty))


def test_no_sample_in_capture_and_holdout() -> None:
    """200 seeded trials: no sample lies in both a padded capture window and the holdout."""
    rng = np.random.default_rng(5)
    step_ms = 30_000
    pad_ms = 60_000

    for _ in range(200):
        n = int(rng.integers(200, 2_000))
        grid = T0 + np.arange(n, dtype=np.int64) * step_ms

        # Random day bounds and a random holdout day set rendered as a sample mask.
        holdout_mask = rng.random(n) < float(rng.uniform(0.1, 0.9))

        n_windows = int(rng.integers(0, 5))
        windows: list[list[int]] = []
        for _w in range(n_windows):
            start = int(rng.integers(int(grid[0]) - step_ms, int(grid[-1]) + step_ms))
            length = int(rng.integers(0, 40)) * step_ms
            windows.append([start, start + length])

        cap = holdout_exclusion_mask(grid, windows, pad_ms)
        final = holdout_mask & ~cap

        # The invariant: nothing survives in both the capture mask and the final mask.
        assert not bool(np.any(final & cap))
        # The final mask is a strict subset of the original holdout mask.
        assert not bool(np.any(final & ~holdout_mask))

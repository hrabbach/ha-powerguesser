"""Unit tests for the overlap metrics (Plan 06-08; D-08, D-28).

The judge is the plug ground truth: an ON tick is a sample where a linked plug
reads above PLUG_ON_W, an overlap tick is a sample where two or more are ON.
Every figure over fewer than OVERLAP_MIN_TICKS ticks is refused (None), never 0.
"""

import numpy as np
import pytest

from custom_components.powerguesser.engine.learning.episodes import PLUG_ON_W
from custom_components.powerguesser.engine.validation import overlap
from custom_components.powerguesser.engine.validation.constants import OVERLAP_MIN_TICKS

STEP_S = 30.0


def test_on_mask_nan_is_off() -> None:
    """A NaN (missing) plug reading counts as off, and the threshold is strict."""
    y = np.array([0.0, 5.0, PLUG_ON_W, PLUG_ON_W + 1.0, np.nan, 100.0])
    assert list(overlap.on_mask(y)) == [False, False, False, True, False, True]


def test_tick_masks_splits_solo_and_overlap() -> None:
    """Each plug's ON ticks split into solo (no other ON) and overlap (>= 2 ON)."""
    a = np.array([True, True, False, True, True, False])
    b = np.array([False, True, True, False, True, False])
    solo, over, n_on = overlap.tick_masks({"a": a, "b": b})
    assert list(n_on) == [1, 2, 1, 1, 2, 0]
    assert list(solo["a"]) == [True, False, False, True, False, False]
    assert list(over["a"]) == [False, True, False, False, True, False]
    assert list(solo["b"]) == [False, False, True, False, False, False]
    assert list(over["b"]) == [False, True, False, False, True, False]


def test_split_errors_refuses_below_min_ticks_never_zero() -> None:
    """A figure under OVERLAP_MIN_TICKS is None; the tick count stays numeric."""
    n = 40
    y = np.zeros(n)
    est = np.zeros(n)
    solo = np.zeros(n, dtype=bool)
    solo[0 : OVERLAP_MIN_TICKS - 1] = True  # 9 ticks, one below the threshold
    over = np.zeros(n, dtype=bool)
    over[20 : 20 + OVERLAP_MIN_TICKS] = True  # exactly the threshold
    y[solo] = 100.0
    est[solo] = 0.0
    y[over] = 100.0
    est[over] = 80.0

    out = overlap.split_errors(y, est, solo, over, STEP_S)

    assert out["solo"]["n_ticks"] == OVERLAP_MIN_TICKS - 1
    assert out["solo"]["energy_error_pct"] is None
    assert out["overlap"]["n_ticks"] == OVERLAP_MIN_TICKS
    assert out["overlap"]["energy_error_pct"] == 20.0


def test_split_errors_solo_figure_at_threshold() -> None:
    """Solo ticks at the threshold do produce a figure (the rule is on the count)."""
    n = OVERLAP_MIN_TICKS
    y = np.full(n, 50.0)
    est = np.full(n, 50.0)
    solo = np.ones(n, dtype=bool)
    over = np.zeros(n, dtype=bool)
    out = overlap.split_errors(y, est, solo, over, STEP_S)
    assert out["solo"]["energy_error_pct"] == 0.0
    assert out["overlap"]["energy_error_pct"] is None


def test_edge_counts_counts_decisions_not_ticks() -> None:
    """One cumulative increase on an overlap sample counts 1, however many overlap
    ticks it spans; a decision one sample before the overlap starts counts nothing
    (review concern L207@8b3be9c)."""
    n = 12
    overlap_any = np.zeros(n, dtype=bool)
    overlap_any[4:9] = True  # five consecutive overlap ticks
    # A joint decision at sample 6 (inside the overlap) and one at sample 3
    # (the sample just before a plug crosses the ON threshold).
    joint_cum = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 2], dtype=np.int64)
    # A refused decision at sample 8 (inside the overlap).
    refused_cum = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)

    counts = overlap.edge_counts(joint_cum, refused_cum, overlap_any)

    assert counts == {"attributed_jointly": 1, "refused": 1}


def test_edge_counts_empty_and_missing_counters() -> None:
    """No overlap tick and absent cumulative counters give 0/0, never an error."""
    empty = np.zeros(5, dtype=bool)
    assert overlap.edge_counts(None, None, empty) == {
        "attributed_jointly": 0,
        "refused": 0,
    }
    assert overlap.edge_counts(np.zeros(0, dtype=np.int64), None, empty) == {
        "attributed_jointly": 0,
        "refused": 0,
    }


def test_remainder_energy_wh_sums_positive_gaps() -> None:
    """energy_left = sum over overlap ticks of max(0, plug - estimate), in Wh."""
    n = 20
    y = np.zeros(n)
    est = np.zeros(n)
    over_a = np.zeros(n, dtype=bool)
    over_a[0:4] = True
    over_b = np.zeros(n, dtype=bool)
    over_b[10:14] = True
    y[over_a] = 100.0
    est[over_a] = 60.0  # 40 W left per tick
    y[over_b] = 50.0
    est[over_b] = 70.0  # over-estimate: contributes nothing, never negative

    total = overlap.remainder_energy_wh(
        {"a": y, "b": y}, {"a": est, "b": est}, {"a": over_a, "b": over_b}, STEP_S
    )

    assert total == pytest.approx(4 * 40.0 * STEP_S / 3600.0)


def test_shifted_copy_moves_both_directions() -> None:
    """The D-28 copy moves a series by whole samples, filling with 0 W and NaN as 0."""
    series = np.array([1.0, 2.0, np.nan, 4.0])
    assert list(overlap.shifted_copy(series, 2)) == [0.0, 0.0, 1.0, 2.0]
    assert list(overlap.shifted_copy(series, -1)) == [2.0, 0.0, 4.0, 0.0]
    assert list(overlap.shifted_copy(series, 0)) == [1.0, 2.0, 0.0, 4.0]


def test_pair_table_sorted_and_thresholded() -> None:
    """One row per pair sharing >= OVERLAP_MIN_TICKS ON ticks, sorted by (a, b);
    each row carries both devices' energy error over the shared ticks."""
    n = 60
    ys = {
        "sensor.a": np.full(n, 100.0),
        "sensor.b": np.full(n, 100.0),
        "sensor.c": np.full(n, 100.0),
        "sensor.d": np.full(n, 100.0),
    }
    ests = {
        "sensor.a": np.full(n, 90.0),
        "sensor.b": np.full(n, 80.0),
        "sensor.c": np.full(n, 80.0),
        "sensor.d": np.full(n, 50.0),
    }
    masks = {key: np.zeros(n, dtype=bool) for key in ys}
    masks["sensor.a"][0:12] = True  # shares 12 ticks with b
    masks["sensor.b"][0:12] = True
    masks["sensor.c"][0:3] = True  # only 3 shared ticks with a: no row
    masks["sensor.a"][30:41] = True  # shares 11 ticks with d
    masks["sensor.d"][30:41] = True

    rows = overlap.pair_table(ys, ests, masks, STEP_S)

    assert [(row["a"], row["b"]) for row in rows] == [
        ("sensor.a", "sensor.b"),
        ("sensor.a", "sensor.d"),
    ]
    first = rows[0]
    assert first["n_ticks"] == 12
    assert first["a_error_pct"] == 10.0
    assert first["b_error_pct"] == 20.0
    assert rows[1]["n_ticks"] == 11
    assert rows[1]["b_error_pct"] == 50.0


def test_pair_table_aggregates_across_windows() -> None:
    """A pair's ticks aggregate across the whole window, all episodes summed
    (review suggestion L212@8b3be9c): three separate 4-tick episodes (12 in all)
    make a row although each episode is below the threshold; two such episodes
    (8 ticks) make none."""
    n = 60
    ys = {"p": np.full(n, 100.0), "q": np.full(n, 100.0)}
    ests = {"p": np.full(n, 80.0), "q": np.full(n, 70.0)}

    def masks_for(episodes: list[int]) -> dict[str, np.ndarray]:
        p = np.zeros(n, dtype=bool)
        q = np.zeros(n, dtype=bool)
        for start in episodes:
            p[start : start + 4] = True
            q[start : start + 4] = True
        return {"p": p, "q": q}

    three = overlap.pair_table(ys, ests, masks_for([0, 20, 40]), STEP_S)
    assert len(three) == 1
    assert three[0]["n_ticks"] == 12
    assert three[0]["a_error_pct"] == 20.0
    assert three[0]["b_error_pct"] == 30.0

    assert overlap.pair_table(ys, ests, masks_for([0, 20]), STEP_S) == []

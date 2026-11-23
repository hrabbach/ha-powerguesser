"""Unit tests for reader engine, zero-order-hold resampling, and gross power reconstruction."""

import sqlite3
import numpy as np
import pytest

from custom_components.powerguesser.engine.buffer import BufferDB, ChannelSpec, GrossLayoutEntry
from custom_components.powerguesser.engine.reader import (
    gross_on_read,
    load_phase_sign,
    resample_zoh,
)


def test_resample_zoh_basic_and_null_semantics() -> None:
    """resample_zoh propagates values until next row; NULL row produces NaNs until next sample."""
    # [(0, 100), (10_000, 200)] on a 5 s grid gives 100, 100, 200
    rows1 = [(0, 100), (10_000, 200)]
    grid1 = resample_zoh(rows1, start_ms=0, end_ms=15_000, step_ms=5000)
    # Grid points: 0, 5000, 10_000
    np.testing.assert_array_equal(grid1, [100.0, 100.0, 200.0])

    # [(0, 100), (10_000, NULL), (20_000, 300)] never holds 100 past 10_000 (NaN until 20_000)
    rows2 = [(0, 100), (10_000, None), (20_000, 300)]
    grid2 = resample_zoh(rows2, start_ms=0, end_ms=25_000, step_ms=5000)
    # Grid points: 0, 5000, 10_000, 15_000, 20_000
    assert grid2[0] == 100.0
    assert grid2[1] == 100.0
    assert np.isnan(grid2[2])
    assert np.isnan(grid2[3])
    assert grid2[4] == 300.0

    # Grid point before the first row is NaN
    grid3 = resample_zoh(rows1, start_ms=-10_000, end_ms=10_000, step_ms=5000)
    # Grid points: -10_000, -5000, 0, 5000
    assert np.isnan(grid3[0])
    assert np.isnan(grid3[1])
    assert grid3[2] == 100.0
    assert grid3[3] == 100.0

    # No output element is 0 unless a row says 0
    rows_zero = [(0, 0), (5000, None)]
    grid_zero = resample_zoh(rows_zero, start_ms=0, end_ms=10_000, step_ms=5000)
    assert grid_zero[0] == 0.0
    assert np.isnan(grid_zero[1])


def test_gross_on_read_layout_transforms() -> None:
    """direct, inverted, and from_minus_to transforms calculate correctly; NaN grid gives NaN."""
    # Channels 1: grid (direct), 2: battery (inverted), 3: solar (direct)
    layout = [
        ("grid", "direct", (1,)),
        ("battery", "inverted", (2,)),
        ("solar", "direct", (3,)),
    ]
    series = {
        1: np.array([500.0, np.nan]),
        2: np.array([200.0, 200.0]),
        3: np.array([1000.0, 1000.0]),
    }
    gross = gross_on_read(series, layout)
    # Index 0: 500 + (-200) + 1000 = 1300
    assert gross[0] == 1300.0
    # Index 1: NaN grid -> gross is NaN
    assert np.isnan(gross[1])

    # Test from_minus_to transform
    layout_diff = [("grid", "from_minus_to", (1, 2))]
    series_diff = {
        1: np.array([1000.0]),
        2: np.array([300.0]),
    }
    assert gross_on_read(series_diff, layout_diff)[0] == 700.0


def test_gross_on_read_null_solar_counts_zero() -> None:
    """A NULL/NaN solar row counts as 0 W, while a NULL/NaN grid or battery makes gross NaN (pinned contract)."""
    layout = [
        ("grid", "direct", (1,)),
        ("battery", "direct", (2,)),
        ("solar", "direct", (3,)),
    ]
    # Point 0: grid 500, battery -300, solar NaN (from NULL marker) -> gross = 500 - 300 + 0 = 200 W
    # Point 1: grid NaN (NULL row), battery -300, solar 1000 -> gross is NaN
    series = {
        1: np.array([500.0, np.nan]),
        2: np.array([-300.0, -300.0]),
        3: np.array([np.nan, 1000.0]),
    }
    gross = gross_on_read(series, layout)
    assert gross[0] == 200.0
    assert np.isnan(gross[1])


def test_load_phase_sign(tmp_path: pytest.TempPathFactory) -> None:
    """load_phase_sign returns -1 when meta phase_invert is '1', and 1 otherwise."""
    db_file = str(tmp_path / "buffer.sqlite")
    db = BufferDB(db_file)
    db.open()

    # Initially phase_invert is not set
    assert load_phase_sign(db._conn) == 1

    # Setting phase_invert via ensure_channels(..., phase_invert=True)
    db.ensure_channels([], [], phase_invert=True)
    assert load_phase_sign(db._conn) == -1

    # Setting phase_invert via ensure_channels(..., phase_invert=False)
    db.ensure_channels([], [], phase_invert=False)
    assert load_phase_sign(db._conn) == 1


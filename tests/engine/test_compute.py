"""Tests for engine.compute channel calculations and gross computation."""

import pytest

from custom_components.powerguesser.engine.compute import (
    OverAttribution,
    channel_value_w,
    compute_snapshot,
    gross_w,
    residual_w,
)
from custom_components.powerguesser.engine.model import (
    GrossChannel,
    Role,
    SourceModel,
    Transform,
)


def test_channel_value_w_direct() -> None:
    ch = GrossChannel("grid:1", Role.GRID, Transform.DIRECT, ("sensor.grid",), "Grid")
    assert channel_value_w(ch, {"sensor.grid": 500.0}) == 500.0
    assert channel_value_w(ch, {"sensor.grid": None}) is None
    assert channel_value_w(ch, {}) is None


def test_channel_value_w_inverted() -> None:
    ch = GrossChannel("grid:exp", Role.GRID, Transform.INVERTED, ("sensor.exp",), "Grid")
    assert channel_value_w(ch, {"sensor.exp": 200.0}) == -200.0
    assert channel_value_w(ch, {"sensor.exp": -300.0}) == 300.0
    assert channel_value_w(ch, {"sensor.exp": None}) is None


def test_channel_value_w_from_minus_to() -> None:
    ch = GrossChannel(
        "batt:two", Role.BATTERY, Transform.FROM_MINUS_TO, ("sensor.dis", "sensor.chg"), "Batt"
    )
    assert channel_value_w(ch, {"sensor.dis": 700.0, "sensor.chg": 200.0}) == 500.0
    assert channel_value_w(ch, {"sensor.dis": 100.0, "sensor.chg": 600.0}) == -500.0
    assert channel_value_w(ch, {"sensor.dis": None, "sensor.chg": 200.0}) is None
    assert channel_value_w(ch, {"sensor.dis": 700.0, "sensor.chg": None}) is None


def test_gross_w_home_source() -> None:
    ch = GrossChannel("home:1", Role.HOME, Transform.DIRECT, ("sensor.home",), "Home")
    assert gross_w([ch], {"sensor.home": 1250.0}) == 1250.0
    assert gross_w([ch], {"sensor.home": None}) is None


def test_gross_w_solar_clamp_and_unclamped_total() -> None:
    grid = GrossChannel("grid:1", Role.GRID, Transform.DIRECT, ("sensor.grid",), "Grid")
    solar = GrossChannel("solar:1", Role.SOLAR, Transform.DIRECT, ("sensor.solar",), "Solar")
    batt = GrossChannel("batt:1", Role.BATTERY, Transform.DIRECT, ("sensor.batt",), "Batt")

    # Solar negative (-20) is clamped to 0.0
    # Net: -500 + 0 + (-100) = -600 (gross is never clamped!)
    values = {"sensor.grid": -500.0, "sensor.solar": -20.0, "sensor.batt": -100.0}
    assert gross_w([grid, solar, batt], values) == -600.0


def test_gross_w_none_propagation() -> None:
    grid = GrossChannel("grid:1", Role.GRID, Transform.DIRECT, ("sensor.grid",), "Grid")
    solar = GrossChannel("solar:1", Role.SOLAR, Transform.DIRECT, ("sensor.solar",), "Solar")

    assert gross_w([grid, solar], {"sensor.grid": 500.0, "sensor.solar": None}) is None


def test_over_attribution_accumulator() -> None:
    """OverAttribution accumulator integrates clipped energy over time intervals."""
    acc = OverAttribution()

    # First add stores state, adds no energy
    acc.add(-500.0, 1000.0)
    assert acc.total_wh == 0.0

    # Non-increasing ts adds nothing
    acc.add(-600.0, 1000.0)
    assert acc.total_wh == 0.0

    # 60s later at -500 W: 500 W * 60s / 3600 = 8.3333 Wh
    acc.add(-500.0, 1060.0)
    assert pytest.approx(acc.total_wh, 0.001) == 8.3333

    # Positive residual (no clipping): signed +200 W adds 500 * (30/3600) = 4.1666 Wh
    acc.add(200.0, 1090.0)
    assert pytest.approx(acc.total_wh, 0.001) == 12.5

    # Another 60s at positive residual: previous was +200, max(0, -200) = 0 -> 0 added
    acc.add(300.0, 1150.0)
    assert pytest.approx(acc.total_wh, 0.001) == 12.5


def test_compute_snapshot_clamping_and_attribution() -> None:
    """Snapshot clamps remainder at 0, flags clipped, and integrates into accumulator."""
    grid = GrossChannel("grid:1", Role.GRID, Transform.DIRECT, ("sensor.grid",), "Grid")
    model = SourceModel(gross_channels=(grid,))
    acc = OverAttribution()

    # Gross 1000 W, metered 1500 W -> signed -500 W, remainder 0, clipped True
    snap1 = compute_snapshot(
        model,
        {"sensor.grid": 1000.0},
        {"sensor.plug": 1500.0},
        ts=1000.0,
        accumulator=acc,
    )
    assert snap1.gross_w == 1000.0
    assert snap1.remainder_w == 0.0
    assert snap1.signed_residual_w == -500.0
    assert snap1.clipped is True
    assert snap1.over_attribution_wh == 0.0

    # Second sample 60s later
    snap2 = compute_snapshot(
        model,
        {"sensor.grid": 1000.0},
        {"sensor.plug": 1500.0},
        ts=1060.0,
        accumulator=acc,
    )
    assert snap2.remainder_w == 0.0
    assert snap2.clipped is True
    assert pytest.approx(snap2.over_attribution_wh, 0.01) == 8.33

"""Tests for windowed sign self-check, hysteresis, and reference day dataset validation."""

import csv
import os
from pathlib import Path

from custom_components.powerguesser.engine.selfcheck import (
    CLEAR_AFTER_S,
    MIN_SAMPLES,
    RAISE_SHARE,
    TOLERANCE_FRAC,
    TOLERANCE_W,
    WINDOW_S,
    SignSelfCheck,
    evaluate_plausibility,
)

FIXTURE_PATH = Path(__file__).parent.parent / "fixtures" / "reference_day.csv"


def test_fixture_hygiene() -> None:
    """Fixture hygiene: header is exactly t_s,grid_w,solar_w,battery_w,metered_w and all cells parse as float."""
    assert FIXTURE_PATH.is_file()
    with open(FIXTURE_PATH, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        assert header == ["t_s", "grid_w", "solar_w", "battery_w", "metered_w"]
        row_count = 0
        for row in reader:
            row_count += 1
            assert len(row) == 5
            for cell in row:
                val = float(cell)
                assert isinstance(val, float)
        assert row_count >= 2880


def test_evaluate_plausibility_tolerances() -> None:
    """Tolerances scale as max(50 W, 3% of |gross|)."""
    # 0 W: tolerance is 50 W
    assert evaluate_plausibility(0.0, 40.0) == ()
    assert evaluate_plausibility(-45.0, 0.0) == ()
    assert "negative_gross" in evaluate_plausibility(-55.0, 0.0)
    assert evaluate_plausibility(0.0, 55.0) == ("metered_above_gross",)

    # 100 W: tolerance is 50 W
    assert evaluate_plausibility(100.0, 140.0) == ()
    assert evaluate_plausibility(100.0, 155.0) == ("metered_above_gross",)

    # 5000 W: tolerance is 3% * 5000 = 150 W
    assert evaluate_plausibility(5000.0, 5140.0) == ()
    assert evaluate_plausibility(5000.0, 5160.0) == ("metered_above_gross",)



def test_under_30_samples_never_activates() -> None:
    """Fewer than 30 samples never activate an issue, even at 100% violation rate."""
    checker = SignSelfCheck()
    for step in range(29):  # 29 samples
        t_s = step * 30.0
        active = checker.observe(t_s, ("negative_gross",), gross_w=-400.0, metered_w=0.0)
        assert len(active) == 0

    # 30th sample triggers activation
    active = checker.observe(29 * 30.0, ("negative_gross",), gross_w=-400.0, metered_w=0.0)
    assert "negative_gross" in active


def test_clear_boundary_11h59m_vs_12h() -> None:
    """Synthetic: violation window activates check; stays active for 11h59m clean, clears after 12h."""
    checker = SignSelfCheck()
    # 45 minutes of violations (90 samples at 30s)
    for step in range(90):
        t_s = step * 30.0
        checker.observe(t_s, ("metered_above_gross",), gross_w=300.0, metered_w=1200.0)

    last_violation_ts = 89 * 30.0  # 2670.0s
    assert "metered_above_gross" in checker.observe(
        last_violation_ts, ("metered_above_gross",), gross_w=300.0, metered_w=1200.0
    )

    # Clean period starts at 2700s.
    # At 11h 59m (43140s) after last violation:
    t_11h59 = last_violation_ts + 43140.0
    active = checker.observe(t_11h59, (), gross_w=1000.0, metered_w=200.0)
    assert "metered_above_gross" in active

    # At 12h (43200s = CLEAR_AFTER_S) after last violation:
    t_12h = last_violation_ts + CLEAR_AFTER_S
    active = checker.observe(t_12h, (), gross_w=1000.0, metered_w=200.0)
    assert "metered_above_gross" not in active


def test_reset_clears_state() -> None:
    """reset() immediately drops all history and active checks."""
    checker = SignSelfCheck()
    for step in range(35):
        checker.observe(step * 30.0, ("negative_gross",), gross_w=-400.0, metered_w=0.0)
    assert "negative_gross" in checker.observe(
        35 * 30.0, ("negative_gross",), gross_w=-400.0, metered_w=0.0
    )

    checker.reset()
    assert len(checker.observe(36 * 30.0, (), gross_w=500.0, metered_w=0.0)) == 0


def test_reference_day_as_configured() -> None:
    """Recorded reference day as configured is quiet throughout all 24 hours.

    Quote from 01-04 SUMMARY:
    grid_charging_day: no
    """
    checker = SignSelfCheck()

    with open(FIXTURE_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t_s = float(row["t_s"])
            grid_w = float(row["grid_w"])
            solar_w = max(float(row["solar_w"]), 0.0)
            battery_w = float(row["battery_w"])
            metered_w = float(row["metered_w"])

            gross = grid_w + solar_w + battery_w
            failing = evaluate_plausibility(gross, metered_w)
            active = checker.observe(t_s, failing, gross_w=gross, metered_w=metered_w)

            assert len(active) == 0, f"Unexpected active check at t_s={t_s}: {active}"


def test_reference_day_with_flipped_battery_sign() -> None:
    """Flipping battery sign on reference day trips negative_gross during battery discharge."""
    checker = SignSelfCheck()
    tripped = False

    with open(FIXTURE_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t_s = float(row["t_s"])
            grid_w = float(row["grid_w"])
            solar_w = max(float(row["solar_w"]), 0.0)
            battery_w = -float(row["battery_w"])  # Flipped sign!
            metered_w = float(row["metered_w"])

            gross = grid_w + solar_w + battery_w
            failing = evaluate_plausibility(gross, metered_w)
            active = checker.observe(t_s, failing, gross_w=gross, metered_w=metered_w)

            if "negative_gross" in active:
                tripped = True
                break

    assert tripped is True, "Flipped battery sign should have tripped negative_gross"


# --- B2 window-mean criterion for metered_above_gross (owner decision, amends D-12) ---
#
# metered_above_gross additionally requires the 45-min window mean of (metered - gross) to
# exceed max(50 W, 3 % of the window mean gross), and clears 12 h after that window condition
# last held. Live trigger (debug session implausible-metered-above-gross): ~20 % per-sample
# violations from metered/gross timing skew while gross exceeded metered by 325-430 W on average.

_STEP_S = 32.0


def _observe(checker: SignSelfCheck, t_s: float, gross_w: float, metered_w: float) -> frozenset[str]:
    failing = evaluate_plausibility(gross_w, metered_w)
    return checker.observe(t_s, failing, gross_w=gross_w, metered_w=metered_w)


def test_transient_skew_with_negative_window_mean_does_not_raise() -> None:
    """Live-like: >= 20 % per-sample violations from timing skew, but gross exceeds metered on average."""
    checker = SignSelfCheck()
    violations = 0
    samples = 120  # 64 min at 32 s
    for step in range(samples):
        if step % 4 == 0:
            # Gross dropped within one poll while slow metered plugs still hold their last value
            gross_w, metered_w = 1300.0, 2100.0
        else:
            # Steady state: ~500 W of real unmetered load
            gross_w, metered_w = 2600.0, 2100.0
        violations += bool(evaluate_plausibility(gross_w, metered_w))
        active = _observe(checker, step * _STEP_S, gross_w, metered_w)
        assert "metered_above_gross" not in active, f"raised at step {step}"

    assert violations / samples >= RAISE_SHARE  # the per-sample share alone would raise


def test_solar_omitted_systematic_fault_raises() -> None:
    """Solar missing from gross: metered exceeds gross on most samples and on average -> raises."""
    checker = SignSelfCheck()
    raised_at: float | None = None
    for step in range(90):
        solar_w = 0.0 if step % 5 == 0 else 1200.0  # passing clouds
        gross_w = 1500.0 - solar_w  # grid only; the solar source is not part of gross
        t_s = step * _STEP_S
        active = _observe(checker, t_s, gross_w, 1100.0)
        if raised_at is None and "metered_above_gross" in active:
            raised_at = t_s

    assert raised_at == (MIN_SAMPLES - 1) * _STEP_S


def test_metered_above_gross_clears_12h_after_window_condition_stops_holding() -> None:
    """Sporadic skew violations after a fixed fault no longer keep the issue alive past 12 h."""
    checker = SignSelfCheck()
    active: frozenset[str] = frozenset()
    t_s = 0.0
    for step in range(90):  # 48 min of a systematic fault: metered 900 W above gross
        t_s = step * _STEP_S
        active = _observe(checker, t_s, 300.0, 1200.0)
    assert "metered_above_gross" in active
    last_fault_ts = t_s

    # Fault fixed. Timing skew still yields a per-sample violation every 10th sample, but the
    # window mean of (metered - gross) is negative once the fault samples leave the window.
    end_ts = last_fault_ts + WINDOW_S + CLEAR_AFTER_S + _STEP_S
    step = 0
    while t_s < end_ts:
        step += 1
        t_s = last_fault_ts + step * _STEP_S
        gross_w = 1300.0 if step % 10 == 0 else 2600.0
        active = _observe(checker, t_s, gross_w, 2100.0)
        if t_s < last_fault_ts + CLEAR_AFTER_S:
            assert "metered_above_gross" in active, f"cleared early at t={t_s}"

    assert "metered_above_gross" not in active


def test_short_burst_after_gap_does_not_extend_metered_above_gross() -> None:
    """The window condition needs MIN_SAMPLES samples; a few skewed samples after a gap do not keep the issue alive."""
    checker = SignSelfCheck()
    active: frozenset[str] = frozenset()
    t_s = 0.0
    for step in range(90):
        t_s = step * _STEP_S
        active = _observe(checker, t_s, 300.0, 1200.0)
    assert "metered_above_gross" in active
    last_fault_ts = t_s

    # Gross unavailable for hours (no emissions), then a short burst of samples with metered above gross
    burst_start = last_fault_ts + CLEAR_AFTER_S - 600.0
    for i in range(5):
        active = _observe(checker, burst_start + i * _STEP_S, 300.0, 1200.0)
        assert "metered_above_gross" in active

    # 12 h after the window condition last held (end of the fault window) the issue clears
    active = _observe(checker, last_fault_ts + CLEAR_AFTER_S, 2600.0, 2100.0)
    assert "metered_above_gross" not in active

"""Unit tests for the interval profiling and event gating engine."""

from collections import deque
import pytest

from custom_components.powerguesser.engine.intervals import (
    EVENT_TYPE_MIN_S,
    FALLBACK_INTERVAL_S,
    MIN_DELTAS,
    PUBLISH_HYSTERESIS_S,
    STALE_FACTOR,
    WINDOW,
    ZERO_W_EPS,
    EventType,
    IntervalProfiler,
    MeasuredFrom,
    Mode,
    SourceProfile,
    effective_interval,
    gate,
)
from custom_components.powerguesser.engine.settle import STALE_AFTER_S, ReReportDetector


def test_constants() -> None:
    """Verify required constants."""
    assert WINDOW == 120
    assert MIN_DELTAS == 20
    assert STALE_FACTOR == 2.5
    assert FALLBACK_INTERVAL_S == 30.0
    assert ZERO_W_EPS == 0.5
    assert PUBLISH_HYSTERESIS_S == 1.0
    assert EVENT_TYPE_MIN_S[EventType.STEP] == 90.0
    assert EVENT_TYPE_MIN_S[EventType.SHORT_EVENT] == 10.0
    assert EVENT_TYPE_MIN_S[EventType.CYCLE_SHAPE] == 180.0
    assert EVENT_TYPE_MIN_S[EventType.CONTINUOUS] == 0.0


def test_single_observation_gives_no_delta() -> None:
    """A single sample yields no deltas and no published median."""
    profiler = IntervalProfiler()
    changed = profiler.observe(100.0, reported=False, w=50.0)
    assert not changed
    assert profiler.delta_count == 0
    assert profiler.median_s is None
    assert profiler.p90_s is None
    assert profiler.last_ts == 100.0


def test_deltas_across_unavailable_skipped() -> None:
    """Deltas spanning unavailable/unknown (w is None) samples are never counted."""
    profiler = IntervalProfiler()
    profiler.observe(10.0, reported=False, w=100.0)
    profiler.observe(40.0, reported=False, w=None)  # unavailable
    profiler.observe(70.0, reported=False, w=100.0)
    assert profiler.delta_count == 0
    assert profiler.median_s is None


def test_window_capped_at_120() -> None:
    """Counted deltas are capped at WINDOW (120)."""
    profiler = IntervalProfiler()
    t = 0.0
    for _ in range(150):
        t += 10.0
        profiler.observe(t, reported=False, w=100.0)
    assert profiler.delta_count == WINDOW
    assert profiler.median_s == 10.0


def test_histogram() -> None:
    """Histogram returns counts of rounded deltas."""
    profiler = IntervalProfiler()
    t = 0.0
    profiler.observe(t, reported=False, w=100.0)
    for _ in range(10):
        t += 10.2
        profiler.observe(t, reported=False, w=100.0)
    for _ in range(15):
        t += 20.4
        profiler.observe(t, reported=False, w=100.0)
    hist = profiler.histogram()
    assert hist == {10: 10, 20: 15}


def test_on_change_publishes_change_cadence() -> None:
    """D-23: 21 changes 31 s apart publish 31.0 with mode ON_CHANGE and measured_from CHANGES."""
    profiler = IntervalProfiler()
    t = 0.0
    # 20 samples -> 19 deltas -> median_s is None
    for i in range(20):
        profiler.observe(t, reported=False, w=100.0 + i)
        t += 31.0
    assert profiler.delta_count == 19
    assert profiler.median_s is None
    assert profiler.mode == Mode.ON_CHANGE
    snap = profiler.snapshot("test_entity")
    assert snap.measured_from is None

    # 21st sample -> 20th delta -> published
    changed = profiler.observe(t, reported=False, w=150.0)
    assert changed
    assert profiler.delta_count == 20
    assert profiler.median_s == 31.0
    assert profiler.mode == Mode.ON_CHANGE
    snap = profiler.snapshot("test_entity")
    assert snap.measured_from == MeasuredFrom.CHANGES


def test_on_change_skips_zero_w_deltas() -> None:
    """D-23: 20 changes 30 s apart, then 0 W sample followed 3600 s later by 50 W -> delta skipped."""
    profiler = IntervalProfiler()
    t = 0.0
    for i in range(21):
        profiler.observe(t, reported=False, w=50.0 + i)
        t += 30.0
    assert profiler.delta_count == 20
    assert profiler.median_s == 30.0

    # Next sample is 0 W
    profiler.observe(t, reported=False, w=0.0)
    assert profiler.delta_count == 21

    # 3600 s later, changes to 50 W -> skipped because previous sample sat at 0 W
    t += 3600.0
    profiler.observe(t, reported=False, w=50.0)
    assert profiler.delta_count == 21
    assert profiler.median_s == 30.0


def test_re_reporting_counts_zero_w_deltas() -> None:
    """D-11/D-24: 0 W change + 22 re-reports of 0 W 30 s apart -> 2 skipped, 20 counted, mode RE_REPORTING."""
    profiler = IntervalProfiler()
    t = 0.0
    # Sample 0: 0 W change
    profiler.observe(t, reported=False, w=0.0)

    # 22 re-reports of 0 W
    for i in range(22):
        t += 30.0
        profiler.observe(t, reported=True, w=0.0)

    # First 2 re-reports occurred while mode was still ON_CHANGE, so deltas were skipped.
    # Third re-report flipped mode to RE_REPORTING, so that delta and all subsequent 19 deltas count.
    assert profiler.delta_count == 20
    assert profiler.mode == Mode.RE_REPORTING
    assert profiler.median_s == 30.0
    snap = profiler.snapshot("battery")
    assert snap.measured_from == MeasuredFrom.REPORTS


def test_on_change_never_stale() -> None:
    """D-15/D-23: An on-change profiler with a 31 s median returns default STALE_AFTER_S."""
    profiler = IntervalProfiler()
    t = 0.0
    for i in range(21):
        profiler.observe(t, reported=False, w=100.0 + i)
        t += 31.0
    assert profiler.mode == Mode.ON_CHANGE
    assert profiler.median_s == 31.0
    assert profiler.stale_after_s() == STALE_AFTER_S
    assert profiler.stale_after_s() == 90.0


def test_single_stray_report_stays_on_change() -> None:
    """D-24: One stray re-report leaves mode ON_CHANGE, 0 W delta still skipped, stale_after 90s."""
    profiler = IntervalProfiler()
    t = 0.0
    for i in range(21):
        profiler.observe(t, reported=False, w=100.0 + i)
        t += 31.0
    assert profiler.mode == Mode.ON_CHANGE

    # One single re-report
    t += 31.0
    profiler.observe(t, reported=True, w=120.0)
    assert profiler.mode == Mode.ON_CHANGE

    # 0 W sample then 50 W sample
    t += 31.0
    profiler.observe(t, reported=False, w=0.0)
    count_before = profiler.delta_count
    t += 3600.0
    profiler.observe(t, reported=False, w=50.0)
    assert profiler.delta_count == count_before
    assert profiler.stale_after_s() == 90.0


def test_profiler_mode_follows_shared_detector() -> None:
    """D-24: Detector flipped outside makes profiler RE_REPORTING; duplicate reports do not recount."""
    detector = ReReportDetector()
    profiler = IntervalProfiler(detector)
    assert profiler.mode == Mode.ON_CHANGE

    detector.observe_report(10.0)
    detector.observe_report(20.0)
    detector.observe_report(30.0)
    assert detector.re_reporting is True
    assert profiler.mode == Mode.RE_REPORTING

    # Observing same timestamp through profiler changes nothing
    profiler.observe(30.0, reported=True, w=100.0)
    assert detector.re_reporting is True


def test_publish_hysteresis() -> None:
    """Review LOW: Unrounded median must move by at least 1.0 s before republishing."""
    profiler = IntervalProfiler()
    t = 0.0
    # 20 deltas of 30.4 s
    profiler.observe(t, reported=False, w=10.0)
    for _ in range(20):
        t += 30.4
        profiler.observe(t, reported=False, w=10.0)
    assert profiler.median_s == pytest.approx(30.4)

    # Deltas of 30.6 filling the window keep median_s 30.4 and observe returns False
    drifted = False
    for _ in range(WINDOW):
        t += 30.6
        if profiler.observe(t, reported=False, w=10.0):
            drifted = True
    assert not drifted
    assert profiler.median_s == pytest.approx(30.4)

    # Deltas of 31.5 filling the window publish 31.5 and observe returns True
    jumped = False
    for _ in range(WINDOW):
        t += 31.5
        if profiler.observe(t, reported=False, w=10.0):
            jumped = True
    assert jumped
    assert profiler.median_s == pytest.approx(31.5)


def test_restored_flag() -> None:
    """Review LOW: restore() sets restored flag, live data replaces it, and re_reporting marks detector."""
    profiler = IntervalProfiler()
    profiler.restore(30.5, 33.0, "on_change")
    assert profiler.restored is True
    assert profiler.median_s == 30.5
    assert profiler.p90_s == 33.0
    assert profiler.mode == Mode.ON_CHANGE
    assert profiler.detector.re_reporting is False

    # 20 live deltas of 30.0 replace restored median even if diff < 1.0
    t = 0.0
    profiler.observe(t, reported=False, w=100.0)
    for _ in range(20):
        t += 30.0
        profiler.observe(t, reported=False, w=100.0)
    assert profiler.median_s == 30.0
    assert profiler.restored is False

    # Restoring re_reporting mode marks detector and updates staleness immediately
    p2 = IntervalProfiler()
    p2.restore(10.0, 11.0, "re_reporting")
    assert p2.detector.re_reporting is True
    assert p2.mode == Mode.RE_REPORTING
    assert p2.stale_after_s() == 25.0


def test_effective_interval() -> None:
    """D-13/D-23: Effective interval takes max of published medians, or (30.0, True)."""
    p1 = SourceProfile("s1", Mode.RE_REPORTING, 30.2, 31.0, 50, False)
    p2 = SourceProfile("s2", Mode.ON_CHANGE, 31.0, 32.0, 50, False)
    assert effective_interval([p1, p2]) == (31.0, False)
    assert effective_interval([p2, p1]) == (31.0, False)

    p_none1 = SourceProfile("s1", Mode.ON_CHANGE, None, None, 5, False)
    p_none2 = SourceProfile("s2", Mode.RE_REPORTING, None, None, 10, False)
    assert effective_interval([p_none1, p_none2]) == (30.0, True)
    assert effective_interval([]) == (30.0, True)


def test_gate() -> None:
    """SRC-07 / D-18: Gating rules, catalogue ordering, inclusive boundary, and assumed reasons."""
    enabled, disabled = gate(31.0, False)
    assert enabled == (EventType.STEP, EventType.CYCLE_SHAPE, EventType.CONTINUOUS)
    assert disabled == {
        EventType.SHORT_EVENT: "short_event disabled: needs ≤5 s interval, effective is 31 s"
    }

    # Adjacency: 45.0 s enables step, 45.001 s disables step
    en45, dis45 = gate(45.0, False)
    assert EventType.STEP in en45

    en45_001, dis45_001 = gate(45.001, False)
    assert EventType.STEP not in en45_001
    assert dis45_001[EventType.STEP] == "step disabled: needs ≤45 s interval, effective is 45 s"

    # Assumed suffix
    _, dis_assumed = gate(30.0, True)
    assert dis_assumed[EventType.SHORT_EVENT] == "short_event disabled: needs ≤5 s interval, effective is 30 s (assumed)"

"""Tests for SettleAligner, temporal alignment, and staleness gating."""

from custom_components.powerguesser.engine.model import Role
from custom_components.powerguesser.engine.settle import (
    ON_CHANGE_AFTER_S,
    REREPORT_MIN_COUNT,
    REREPORT_WINDOW_S,
    STALE_AFTER_S,
    AlignStatus,
    EntityState,
    ReReportDetector,
    SettleAligner,
)


def test_constants() -> None:
    """Verify bootstrap constants match specification."""
    assert STALE_AFTER_S == 90.0
    assert ON_CHANGE_AFTER_S == 45.0
    assert REREPORT_MIN_COUNT == 3
    assert REREPORT_WINDOW_S == 600.0


def test_rereport_detector_window_and_stickiness() -> None:
    """ReReportDetector flips after 3 reports within window and stays re-reporting."""
    detector = ReReportDetector()
    assert detector.re_reporting is False
    assert detector.observe_report(0.0) is False
    assert detector.observe_report(30.0) is False
    assert detector.re_reporting is False
    assert detector.observe_report(60.0) is True
    assert detector.re_reporting is True
    # Later reports return False, stickiness
    assert detector.observe_report(90.0) is False
    assert detector.re_reporting is True


def test_rereport_detector_window_span_and_boundary() -> None:
    """Reports outside 600s window don't flip until recent 3 are within 600s; inclusive boundary."""
    detector = ReReportDetector()
    # Reports at 0, 400, 700 do not flip (700 - 0 = 700 > 600)
    assert detector.observe_report(0.0) is False
    assert detector.observe_report(400.0) is False
    assert detector.observe_report(700.0) is False
    assert detector.re_reporting is False
    # Fourth report at 800 flips (800 - 400 = 400 <= 600)
    assert detector.observe_report(800.0) is True
    assert detector.re_reporting is True

    # Inclusive boundary test: 0, 300, 600 flip (600 - 0 = 600 <= 600)
    boundary_detector = ReReportDetector()
    assert boundary_detector.observe_report(0.0) is False
    assert boundary_detector.observe_report(300.0) is False
    assert boundary_detector.observe_report(600.0) is True
    assert boundary_detector.re_reporting is True


def test_rereport_detector_idempotent_timestamps() -> None:
    """A report whose ts is not later than the last counted report is ignored."""
    detector = ReReportDetector()
    assert detector.observe_report(10.0) is False
    assert detector.observe_report(10.0) is False
    assert detector.observe_report(10.0) is False
    assert detector.re_reporting is False


def test_rereport_detector_mark_re_reporting() -> None:
    """mark_re_reporting sets the flag without counting."""
    detector = ReReportDetector()
    assert detector.re_reporting is False
    detector.mark_re_reporting()
    assert detector.re_reporting is True
    assert detector.observe_report(100.0) is False


def test_phantom_spike_sequence() -> None:
    """SRC-04 / D-05: Grid and battery update at different moments; no single-sample phantom spike is emitted."""
    aligner = SettleAligner({
        "sensor.grid": Role.GRID,
        "sensor.battery": Role.BATTERY,
    })

    # Initial settled state: grid=1500 W, battery=-500 W (gross = 1000 W)
    assert aligner.on_sample("sensor.grid", 1500.0, 100.0) is False
    assert aligner.on_sample("sensor.battery", -500.0, 100.0) is True
    assert aligner.values["sensor.grid"] == 1500.0
    assert aligner.values["sensor.battery"] == -500.0

    # Step change: battery updates to 0 W first (due to load change), grid hasn't updated yet
    # If this emitted, gross would be 1500 + 0 = 1500 W (phantom spike!).
    emitted = aligner.on_sample("sensor.battery", 0.0, 101.0)
    assert emitted is False  # Must NOT emit yet!

    # Grid then updates to 1000 W
    emitted = aligner.on_sample("sensor.grid", 1000.0, 101.5)
    assert emitted is True  # Both fresh -> emits settled state
    assert aligner.values["sensor.grid"] == 1000.0
    assert aligner.values["sensor.battery"] == 0.0
    assert aligner.emit_ts == 101.5


def test_staleness_and_hysteresis() -> None:
    """D-06: A re-reporting entity goes stale after 90s without an update, clears on next sample."""
    aligner = SettleAligner({"sensor.grid": Role.GRID})

    # Entity reports 3 times within 600s with reported=True (state_reported)
    assert aligner.on_sample("sensor.grid", 500.0, 40.0, reported=True) is True
    assert aligner.on_sample("sensor.grid", 500.0, 70.0, reported=True) is True
    assert aligner.on_sample("sensor.grid", 500.0, 100.0, reported=True) is True

    # At 89s after last sample (t=189.0), tick should NOT mark it stale
    assert aligner.on_tick(189.0) is False
    st = aligner.status(189.0)
    assert "sensor.grid" not in st.stale_sources
    assert "sensor.grid" not in st.blocking

    # At 91s after last sample (t=191.0), tick marks it stale
    aligner.on_tick(191.0)
    st = aligner.status(191.0)
    assert "sensor.grid" in st.stale_sources
    assert "sensor.grid" in st.blocking

    # Next sample arrives -> stale is cleared immediately (hysteresis)
    assert aligner.on_sample("sensor.grid", 520.0, 195.0, reported=True) is True
    st = aligner.status(195.0)
    assert "sensor.grid" not in st.stale_sources
    assert "sensor.grid" not in st.blocking


def test_never_reporting_channel_not_stale_by_time() -> None:
    """D-19: A gross entity without reported_seen is never timed out by on_tick."""
    aligner = SettleAligner({"sensor.grid_template": Role.GRID})

    # Initial state without reported=True (e.g. template sensor)
    assert aligner.on_sample("sensor.grid_template", 300.0, 100.0, reported=False) is True

    # Idle for 10 minutes (600s)
    aligner.on_tick(700.0)
    st = aligner.status(700.0)
    assert "sensor.grid_template" not in st.stale_sources
    assert "sensor.grid_template" not in st.blocking


def test_on_change_classification_option_b() -> None:
    """Option (b): Never-reporting entity idle for > 45s becomes on_change, stops gating until update."""
    aligner = SettleAligner({
        "sensor.grid": Role.GRID,
        "sensor.battery_template": Role.BATTERY,
    })

    # Initial settle: grid reports 3 times up to t=10.0 to become re-reporting
    assert aligner.on_sample("sensor.grid", 1000.0, 0.0, reported=True) is False
    assert aligner.on_sample("sensor.grid", 1000.0, 5.0, reported=True) is False
    assert aligner.on_sample("sensor.grid", 1000.0, 10.0, reported=True) is False
    assert aligner.on_sample("sensor.battery_template", 0.0, 10.0, reported=False) is True

    # At t=56.0 (idle 46s > 45s), tick marks battery_template as on_change
    aligner.on_tick(56.0)
    st = aligner.status(56.0)
    assert "sensor.battery_template" in st.on_change_sources
    assert "sensor.battery_template" not in st.stale_sources
    assert len(st.blocking) == 0

    # Grid updates at t=57.0; since battery is on_change (idle), grid does not wait for it
    emitted = aligner.on_sample("sensor.grid", 1200.0, 57.0, reported=True)
    assert emitted is True

    # Battery template updates later -> on_change is cleared and it re-arms gating
    aligner.on_sample("sensor.battery_template", 500.0, 100.0, reported=False)
    st = aligner.status(100.0)
    assert "sensor.battery_template" not in st.on_change_sources


def test_on_tick_releases_pending_settle() -> None:
    """on_tick returns True when an on_change classification releases a pending settle."""
    aligner = SettleAligner({
        "sensor.grid": Role.GRID,
        "sensor.battery_template": Role.BATTERY,
    })

    # Initial settle at t=0
    aligner.on_sample("sensor.grid", 500.0, 0.0, reported=True)
    aligner.on_sample("sensor.battery_template", 0.0, 0.0, reported=False)

    # Grid updates at t=40s (fresh=True), waiting for battery
    assert aligner.on_sample("sensor.grid", 700.0, 40.0, reported=True) is False

    # At t=46s (battery idle > 45s), on_tick classifies battery as on_change and releases grid settle
    assert aligner.on_tick(46.0) is True
    assert aligner.values["sensor.grid"] == 700.0
    assert aligner.values["sensor.battery_template"] == 0.0


def test_solar_unavailable_treated_as_zero_d20() -> None:
    """D-20: Solar entity None/unavailable is treated as 0 W, listed in solar_unavailable, does NOT block."""
    aligner = SettleAligner({
        "sensor.grid": Role.GRID,
        "sensor.solar": Role.SOLAR,
    })

    aligner.on_sample("sensor.grid", 500.0, 10.0, reported=True)
    aligner.on_sample("sensor.solar", None, 10.0, reported=False)

    st = aligner.status(10.0)
    assert "sensor.solar" in st.solar_unavailable
    assert "sensor.solar" not in st.blocking
    assert st.values["sensor.solar"] == 0.0


def test_non_solar_unavailable_blocks() -> None:
    """D-06: Grid or battery None/unavailable blocks gross availability."""
    aligner = SettleAligner({
        "sensor.grid": Role.GRID,
        "sensor.battery": Role.BATTERY,
    })

    aligner.on_sample("sensor.grid", 500.0, 10.0, reported=True)
    aligner.on_sample("sensor.battery", None, 10.0, reported=True)

    st = aligner.status(10.0)
    assert "sensor.battery" in st.blocking


def test_source_age_s_calculation() -> None:
    """source_age_s maps each entity to elapsed time rounded to 0.1s."""
    aligner = SettleAligner({"sensor.grid": Role.GRID})
    aligner.on_sample("sensor.grid", 500.0, 100.0, reported=True)

    st = aligner.status(112.345)
    assert st.source_age_s["sensor.grid"] == 12.3


def test_single_report_not_stale_and_held_as_on_change() -> None:
    """An entity with a single reported=True sample at t=100 is not stale at t=400, is listed in on_change_sources and never blocks."""
    aligner = SettleAligner({"sensor.grid": Role.GRID})
    aligner.on_sample("sensor.grid", 500.0, 100.0, reported=True)

    # At t=400 (idle 300s > 45s, but only 1 report so far -> on_change, not stale)
    aligner.on_tick(400.0)
    st = aligner.status(400.0)
    assert "sensor.grid" in st.on_change_sources
    assert "sensor.grid" not in st.stale_sources
    assert "sensor.grid" not in st.blocking


def test_shared_detectors_survive_rebuild() -> None:
    """Two aligners built over the same detector mapping share the classification."""
    modes = {"sensor.battery": ReReportDetector()}
    aligner1 = SettleAligner({"sensor.battery": Role.BATTERY}, detectors=modes)

    # 3 reports flip detector
    aligner1.on_sample("sensor.battery", 100.0, 0.0, reported=True)
    aligner1.on_sample("sensor.battery", 100.0, 30.0, reported=True)
    aligner1.on_sample("sensor.battery", 100.0, 60.0, reported=True)
    assert modes["sensor.battery"].re_reporting is True

    # Rebuild aligner with same modes
    aligner2 = SettleAligner({"sensor.battery": Role.BATTERY}, detectors=modes)
    aligner2.on_sample("sensor.battery", 100.0, 60.0, reported=False)
    # At 91s after last sample (t=151.0), it goes stale because detector is re-reporting
    aligner2.on_tick(151.0)
    st = aligner2.status(151.0)
    assert "sensor.battery" in st.stale_sources


def test_settle_aligner_custom_stale_after() -> None:
    """SettleAligner with stale_after returning 25.0 marks re-reporting entity stale after 25s (D-15)."""
    detector = ReReportDetector()
    detector.mark_re_reporting()
    aligner = SettleAligner(
        {"sensor.grid": Role.GRID},
        detectors={"sensor.grid": detector},
        stale_after=lambda eid: 25.0,
    )
    aligner.on_sample("sensor.grid", 500.0, 100.0, reported=True)
    # At t=124.0 (idle 24s < 25s) -> not stale
    aligner.on_tick(124.0)
    assert "sensor.grid" not in aligner.status(124.0).stale_sources

    # At t=126.0 (idle 26s > 25s) -> newly stale
    released = aligner.on_tick(126.0)
    assert released is True
    assert "sensor.grid" in aligner.status(126.0).stale_sources

    # Without stale_after, default 90.0 is used
    aligner_default = SettleAligner(
        {"sensor.grid": Role.GRID},
        detectors={"sensor.grid": detector},
    )
    aligner_default.on_sample("sensor.grid", 500.0, 100.0, reported=True)
    aligner_default.on_tick(126.0)
    assert "sensor.grid" not in aligner_default.status(126.0).stale_sources
    aligner_default.on_tick(191.0)
    assert "sensor.grid" in aligner_default.status(191.0).stale_sources

    # An entity not re_reporting is never stale, even with custom stale_after
    detector_on_change = ReReportDetector()
    aligner_on_change = SettleAligner(
        {"sensor.solar": Role.SOLAR},
        detectors={"sensor.solar": detector_on_change},
        stale_after=lambda eid: 25.0,
    )
    aligner_on_change.on_sample("sensor.solar", 100.0, 100.0, reported=False)
    aligner_on_change.on_tick(200.0)
    assert "sensor.solar" not in aligner_on_change.status(200.0).stale_sources


# --- Regression: settle phase lock (debug session implausible-metered-above-gross) ---
#
# Live reference install: one poll chain every ~32 s delivers the inverter (solar) at t,
# battery at t+1.5 s and grid at t+3 s. The settle rule must pair values of the SAME poll
# cycle. A single perturbation (a missed inverter event, or unavailable seeds followed by
# a real sample before the next 10 s tick) used to lock the aligner into emitting on the
# inverter with grid/battery from the previous cycle, permanently.

_INV = "sensor.inverter_input_power"
_BAT = "sensor.battery_real_power"
_GRID = "sensor.real_energy_consumption"
_ROLES = {_GRID: Role.GRID, _INV: Role.SOLAR, _BAT: Role.BATTERY}
_OFFSET = {_INV: 0.0, _BAT: 1.5, _GRID: 3.0}
_PERIOD = 32.0
_TICK = 10.0


def _cycle_value(entity_id: str, k: int) -> float:
    """Encode the poll-cycle index into the sample value (k * 1000 + source tag)."""
    return 1000.0 * k + {_INV: 1.0, _BAT: 2.0, _GRID: 3.0}[entity_id]


def _drive(
    events: list[tuple[float, str, float | None]],
) -> list[tuple[float, dict[str, int]]]:
    """Feed events plus 10 s ticks like the hub; return (time, cycle index per source) per emission."""
    inverter_detector = ReReportDetector()
    inverter_detector.mark_re_reporting()
    aligner = SettleAligner(
        _ROLES,
        detectors={_INV: inverter_detector},
        profiled_interval=lambda _eid: _PERIOD,
    )
    events = sorted(events, key=lambda e: e[0])
    emissions: list[tuple[float, dict[str, int]]] = []

    def _record(ts: float) -> None:
        values = aligner.status(ts).values
        if all(aligner.values[e] is not None for e in _ROLES):
            emissions.append((ts, {e: int(values[e] // 1000) for e in _ROLES}))

    next_tick = _TICK * (int(events[0][0] // _TICK) + 1)
    last_ts = events[-1][0] + 2 * _TICK
    for ts, entity_id, value in events:
        while next_tick < ts:
            if aligner.on_tick(next_tick):
                _record(next_tick)
            next_tick += _TICK
        if aligner.on_sample(entity_id, value, ts):
            _record(ts)
    while next_tick <= last_ts:
        if aligner.on_tick(next_tick):
            _record(next_tick)
        next_tick += _TICK
    return emissions


def _poll_cycles(
    first: int, last: int, *, skip: tuple[tuple[str, int], ...] = ()
) -> list[tuple[float, str, float | None]]:
    """Poll-chain events for cycles first..last, optionally skipping (entity, cycle) pairs."""
    return [
        (_PERIOD * k + _OFFSET[e], e, _cycle_value(e, k))
        for k in range(first, last + 1)
        for e in (_INV, _BAT, _GRID)
        if (e, k) not in skip
    ]


def _assert_same_cycle_after(
    emissions: list[tuple[float, dict[str, int]]], settle_from: float
) -> None:
    late = [(ts, cycles) for ts, cycles in emissions if ts >= settle_from]
    assert len(late) >= 5, f"too few emissions after recovery: {late}"
    for ts, cycles in late:
        assert len(set(cycles.values())) == 1, (
            f"emission at t={ts} pairs values from different poll cycles: {cycles}"
        )


def test_settle_pairs_same_cycle_after_missed_inverter_event() -> None:
    """A single missed inverter event must not lock the settle into cross-cycle pairing."""
    events = _poll_cycles(10, 20, skip=((_INV, 12),))
    emissions = _drive(events)
    # Allow at most one poll cycle to recover after the missed event.
    _assert_same_cycle_after(emissions, settle_from=_PERIOD * 14)


def test_settle_pairs_same_cycle_after_unavailable_seed_before_tick() -> None:
    """Unavailable seeds followed by a real poll before the next tick must still pair same-cycle values."""
    seeds: list[tuple[float, str, float | None]] = [
        (315.0, _INV, None),
        (315.0, _BAT, None),
        (315.0, _GRID, None),
    ]
    events = seeds + _poll_cycles(10, 20)  # cycle 10 starts at 320 s, before the 320 s tick
    emissions = _drive(events)
    _assert_same_cycle_after(emissions, settle_from=_PERIOD * 12)


def test_settle_pairs_same_cycle_in_steady_state() -> None:
    """Baseline: an unperturbed poll chain always pairs same-cycle values."""
    emissions = _drive(_poll_cycles(10, 20))
    _assert_same_cycle_after(emissions, settle_from=_PERIOD * 10)


# --- A1 closer-sample wait (owner decision, debug session implausible-metered-above-gross) ---
#
# When the settle would release, but another gating source's last sample is older than half
# its profiled interval, skip this emission and wait for that source's next sample. Never skip
# twice in a row; no timer. Unknown interval (no profile yet) -> no skip.

_G = "sensor.grid"
_S = "sensor.solar"


def _two_source_aligner(interval: float | None) -> SettleAligner:
    return SettleAligner(
        {_G: Role.GRID, _S: Role.SOLAR},
        profiled_interval=lambda _eid: interval,
    )


def test_settle_skips_and_waits_for_source_older_than_half_interval() -> None:
    """A source 26 s old at a 30 s interval is waited for; its next sample emits the closer pair."""
    aligner = _two_source_aligner(30.0)
    aligner.on_sample(_G, 500.0, 100.0)
    assert aligner.on_sample(_S, 1000.0, 100.0) is True

    assert aligner.on_sample(_G, 600.0, 102.0) is False
    # Solar completes the settle, but grid is 26 s old (> 15 s): skip, wait for grid
    assert aligner.on_sample(_S, 1100.0, 128.0) is False
    assert aligner.on_sample(_G, 700.0, 132.0) is True
    assert aligner.emit_ts == 132.0
    assert aligner.values == {_G: 700.0, _S: 1100.0}


def test_settle_never_skips_twice_in_a_row() -> None:
    """After a skip the next settle emits even if another source is again older than half its interval."""
    aligner = _two_source_aligner(30.0)
    aligner.on_sample(_G, 500.0, 100.0)
    assert aligner.on_sample(_S, 1000.0, 100.0) is True

    aligner.on_sample(_G, 600.0, 102.0)
    assert aligner.on_sample(_S, 1100.0, 128.0) is False  # skip: grid 26 s old
    # Grid arrives late; solar is now 22 s old (> 15 s) but a second skip is not allowed
    assert aligner.on_sample(_G, 700.0, 150.0) is True

    # After an emission the closer-sample wait is armed again
    aligner.on_sample(_G, 800.0, 152.0)
    assert aligner.on_sample(_S, 1200.0, 180.0) is False


def test_settle_without_profiled_interval_does_not_skip() -> None:
    """No profiled interval yet: the settle emits as soon as every gating source is fresh."""
    aligner = _two_source_aligner(None)
    aligner.on_sample(_G, 500.0, 100.0)
    assert aligner.on_sample(_S, 1000.0, 100.0) is True

    aligner.on_sample(_G, 600.0, 102.0)
    assert aligner.on_sample(_S, 1100.0, 128.0) is True


def test_settle_closer_sample_wait_ignores_on_change_sources() -> None:
    """An on-change (non-gating) source is never waited for, however old its last sample."""
    aligner = SettleAligner(
        {_G: Role.GRID, "sensor.battery_template": Role.BATTERY},
        profiled_interval=lambda _eid: 30.0,
    )
    aligner.on_sample(_G, 1000.0, 0.0, reported=True)
    aligner.on_sample(_G, 1000.0, 5.0, reported=True)
    aligner.on_sample(_G, 1000.0, 10.0, reported=True)
    assert aligner.on_sample("sensor.battery_template", 0.0, 10.0) is True

    aligner.on_tick(56.0)  # battery idle 46 s -> on_change
    assert aligner.on_sample(_G, 1200.0, 57.0, reported=True) is True

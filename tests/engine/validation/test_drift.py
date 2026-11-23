"""Tests for the live drift guard (Plan 05-05; D-15, D-22) and the
holdout-basis drift over per-plug and per-class validation history
(Plan 05-06; D-15, D-17, D-23).

Hysteresis (set after 2 consecutive over evaluations, clear after 1 clear
one), LiveDrift windows and baselines, the asymmetric DRIFT_CLEAR_MARGIN
reading (cross-AI review Round 1), refused-delta accounting from cumulative
counts, and the hot Store round trip. Pure stdlib: no HA, no numpy.
"""

import json
import logging
from typing import Any

import pytest

from custom_components.powerguesser.engine.validation.constants import (
    EVIDENCE_MAX,
    LIVE_BASELINE_EPISODES,
    LIVE_MIN_EPISODES,
)
from custom_components.powerguesser.engine.validation.drift import Hysteresis, LiveDrift

BASE_TS = 1_700_000_000.0
DAY_S = 86400.0

# Episodes needed to set drift from an aged-out window: the first
# LIVE_MIN_EPISODES - 1 evaluations are insufficient (counter reset), the 8th
# is the first over evaluation and the 9th sets drift (D-15 hysteresis).
BAD_N = LIVE_MIN_EPISODES + 1


def _tracker_with_baseline(
    score: float = 0.9, *, ts: float = BASE_TS, n: int = LIVE_BASELINE_EPISODES
) -> LiveDrift:
    """A tracker whose baseline (first n episodes, no refused edges) is set."""
    tracker = LiveDrift()
    for i in range(n):
        tracker.record(ts + i, score, 0)
    return tracker


def test_hysteresis_set_after_two_clear_after_one() -> None:
    hyst = Hysteresis(2)
    assert hyst.update(over=True, clear_ok=False) is False
    assert hyst.over_count == 1
    assert hyst.update(over=True, clear_ok=False) is True
    # While active, a clear evaluation clears immediately (clear after 1).
    assert hyst.update(over=True, clear_ok=True) is False
    assert hyst.over_count == 0
    # Re-arms only after two consecutive over evaluations again.
    assert hyst.update(over=True, clear_ok=False) is False
    assert hyst.update(over=True, clear_ok=False) is True
    # A non-over evaluation while inactive resets the counter.
    hyst2 = Hysteresis(2)
    assert hyst2.update(over=True, clear_ok=False) is False
    assert hyst2.update(over=False, clear_ok=False) is False
    assert hyst2.update(over=True, clear_ok=False) is False
    assert hyst2.update(over=True, clear_ok=False) is True


def test_live_drift_insufficient_before_baseline() -> None:
    tracker = LiveDrift()
    for i in range(LIVE_BASELINE_EPISODES - 1):
        tracker.record(BASE_TS + i, 0.1, 0)
    verdict = tracker.evaluate()
    assert verdict.drift is False
    assert verdict.insufficient is True
    assert verdict.changed is False
    assert tracker.evidence_state() == "insufficient"


def test_live_drift_insufficient_when_window_ages_out() -> None:
    tracker = _tracker_with_baseline(0.9)
    verdict = tracker.evaluate()
    assert verdict.insufficient is False
    assert tracker.evidence_state() == "sufficient"
    # One new episode 40 days later: the 10 baseline episodes age out of the
    # 14-day window, leaving fewer than 8 entries -> insufficient, drift false
    # and the hysteresis counter reset.
    tracker.record(BASE_TS + 40 * DAY_S, 0.1, 0)
    verdict = tracker.evaluate()
    assert verdict.insufficient is True
    assert verdict.drift is False
    assert verdict.changed is False
    assert tracker.evidence_state() == "insufficient"


def test_live_drift_program_window_three_plus_three() -> None:
    """D-26: a program device reaches a verdict from 3 baseline + 3 window.

    A washer or dishwasher used a few times a week never yields the switched
    8 episodes in 14 days, so ``LiveDrift(kind="program")`` uses a 3-episode
    baseline and a 42-day window instead; the switched tracker fed the same
    six episodes still reads insufficient.
    """
    program = LiveDrift(kind="program")
    switched = LiveDrift()
    for i in range(6):
        program.record(BASE_TS + i * DAY_S, 0.9, 0)
        switched.record(BASE_TS + i * DAY_S, 0.9, 0)
    assert program.evidence_state() == "sufficient"
    assert program.evaluate().insufficient is False
    assert switched.evidence_state() == "insufficient"
    assert switched.evaluate().insufficient is True


def test_live_drift_program_window_few_episodes_insufficient() -> None:
    """D-26: 42 days hold two of four spans, below the 3-episode minimum."""
    tracker = LiveDrift(kind="program")
    for offset in (-50, -45, -10, 0):
        tracker.record(BASE_TS + offset * DAY_S, 0.9, 0)
    assert tracker.evidence_state() == "insufficient"
    assert tracker.evaluate().insufficient is True


def test_live_drift_program_window_ages_at_42_days() -> None:
    """D-26: the baseline survives ageing; the window does not.

    Ten episodes ~60 days back set both baselines, then three recent closes
    leave the switched 14-day window below its 8-episode minimum while the
    program 42-day window still holds the three recent closes.
    """
    newest = BASE_TS
    program = LiveDrift(kind="program")
    switched = LiveDrift()
    for i in range(10):
        program.record(newest - (60 - i) * DAY_S, 0.9, 0)
        switched.record(newest - (60 - i) * DAY_S, 0.9, 0)
    for i in range(3):
        program.record(newest + i * DAY_S, 0.9, 0)
        switched.record(newest + i * DAY_S, 0.9, 0)
    assert program.evidence_state() == "sufficient"
    assert switched.evidence_state() == "insufficient"


def test_live_drift_from_dict_uses_caller_kind() -> None:
    """D-26/T-05-17: the stored shape has no kind; the caller's kind wins.

    A stray kind-like key inside the stored dict is ignored, so it can never
    override the device record's kind (T-06-30).
    """
    data = {
        "evidence": [{"ts": BASE_TS + i, "score": 0.9, "refused": 0.0} for i in range(6)],
        "last_cum": 0,
        "baseline": {"score": 0.9, "ratio": 0.0},
        "hysteresis": {"active": False, "over_count": 0},
        "since_ts": None,
        "reason": None,
    }
    program = LiveDrift.from_dict(data, kind="program")
    assert program.evidence_state() == "sufficient"
    assert program.evaluate().insufficient is False
    switched = LiveDrift.from_dict(data)
    assert switched.evidence_state() == "insufficient"
    assert (
        LiveDrift.from_dict({**data, "kind": "program"}).evidence_state()
        == "insufficient"
    )


def test_live_drift_to_dict_shape_unchanged() -> None:
    """The Phase 5 key tuple is unchanged for every kind (L290@8b3be9c)."""
    phase5_keys = ("evidence", "last_cum", "baseline", "hysteresis", "since_ts", "reason")
    for tracker in (LiveDrift(), LiveDrift(kind="switched"), LiveDrift(kind="program")):
        assert tuple(tracker.to_dict()) == phase5_keys
    # An explicit "switched" tracker serialises exactly as the default one.
    default = LiveDrift()
    explicit = LiveDrift(kind="switched")
    for i in range(6):
        default.record(BASE_TS + i, 0.9, i)
        explicit.record(BASE_TS + i, 0.9, i)
    assert json.dumps(default.to_dict()) == json.dumps(explicit.to_dict())


def test_live_drift_unknown_kind_uses_switched_window_and_logs(caplog) -> None:
    """A typo'd kind falls back to the switched window with one DEBUG line.

    L285/L289@8b3be9c: a typo'd kind must not silently use the program window.
    """
    with caplog.at_level(logging.DEBUG):
        tracker = LiveDrift(kind="progarm")
    assert sum("progarm" in record.getMessage() for record in caplog.records) == 1
    for i in range(6):
        tracker.record(BASE_TS + i, 0.9, 0)
    assert tracker.evidence_state() == "insufficient"


def test_match_score_low_by_absolute_threshold() -> None:
    tracker = _tracker_with_baseline(0.9)
    verdict = None
    for i in range(BAD_N):
        tracker.record(BASE_TS + 40 * DAY_S + i, 0.1, 0)
        verdict = tracker.evaluate()
    assert verdict is not None
    assert verdict.drift is True
    assert verdict.reason == "match_score_low"
    assert verdict.insufficient is False
    assert verdict.changed is True
    # since_ts is the newest window episode's ts (set on the second over).
    assert verdict.since_ts == pytest.approx(BASE_TS + 40 * DAY_S + BAD_N - 1)


def test_match_score_low_by_baseline_drop() -> None:
    tracker = _tracker_with_baseline(0.9)
    verdict = None
    for i in range(BAD_N):
        # 0.69 is above MATCH_SCORE_MIN (0.6) but more than 0.2 below 0.9.
        tracker.record(BASE_TS + 40 * DAY_S + i, 0.69, 0)
        verdict = tracker.evaluate()
    assert verdict is not None
    assert verdict.drift is True
    assert verdict.reason == "match_score_low"


def test_match_score_drop_boundary_is_strictly_below() -> None:
    # A median exactly at baseline - 0.2 does NOT trigger (rule is "more
    # than 0.2 below").
    tracker = _tracker_with_baseline(0.9)
    verdict = None
    for i in range(BAD_N):
        tracker.record(BASE_TS + 40 * DAY_S + i, 0.7, 0)
        verdict = tracker.evaluate()
    assert verdict is not None
    assert verdict.drift is False


def test_edges_refused_needs_both_absolute_and_relative() -> None:
    # Baseline ratio 0.375 (6 refused over the 10 baseline episodes); scores
    # stay high so only the refused rule can fire.
    tracker = LiveDrift()
    cum = 0
    for i in range(LIVE_BASELINE_EPISODES):
        if 0 < i <= 6:
            cum += 1
        tracker.record(BASE_TS + i, 0.9, cum)
    assert 6 / (LIVE_BASELINE_EPISODES + 6) == pytest.approx(0.375)
    # Phase 1: 8 window episodes with 6 more refused: window ratio 12/24 = 0.5
    # is above 0.30 but not 0.15 above 0.375 -> the AND rule does not fire.
    verdict = None
    for i in range(LIVE_MIN_EPISODES):
        cum += 1 if i < 6 else 0
        tracker.record(BASE_TS + 100 + i, 0.9, cum)
        verdict = tracker.evaluate()
    assert verdict is not None
    assert verdict.drift is False
    assert verdict.reason is None
    # Phase 2: 12 more episodes with a refused burst each push the window
    # ratio past the relative bound -> drift, reason edges_refused.
    for i in range(12):
        cum += 3
        tracker.record(BASE_TS + 200 + i, 0.9, cum)
        verdict = tracker.evaluate()
    assert verdict is not None
    assert verdict.drift is True
    assert verdict.reason == "edges_refused"


def test_refused_relative_without_absolute_does_not_fire() -> None:
    # Baseline ratio ~0.286; window ratio 0.28 is at least 0.15 above it but
    # not above 0.30 -> the AND rule needs both -> no drift.
    tracker = LiveDrift()
    cum = 0
    for i in range(LIVE_BASELINE_EPISODES):
        if 0 < i <= 4:
            cum += 1
        tracker.record(BASE_TS + i, 0.9, cum)
    verdict = None
    for i in range(LIVE_MIN_EPISODES):
        cum += 1 if i < 3 else 0
        tracker.record(BASE_TS + 100 + i, 0.9, cum)
        verdict = tracker.evaluate()
    assert verdict is not None
    assert verdict.drift is False
    assert verdict.reason is None


def test_clear_margin_asymmetric() -> None:
    # With baseline score 0.9, a window median of 0.70 does NOT clear
    # match_score_low (0.70 < 0.9 - 0.2 + 0.05 = 0.75) while 0.76 does. The
    # bad episodes stay inside the 14-day window so the tracker never passes
    # through the insufficient state while drift is active.
    tracker = _tracker_with_baseline(0.9)
    bad_ts = BASE_TS + 40 * DAY_S
    verdict = None
    for i in range(BAD_N):
        tracker.record(bad_ts + i, 0.1, 0)
        verdict = tracker.evaluate()
    assert verdict is not None and verdict.drift is True
    assert verdict.reason == "match_score_low"
    # Recovery at 0.70 starting one day later: the 9 bad episodes are still in
    # the window, keeping it sufficient. After 11 recovery episodes the window
    # is exactly 9 x 0.1 + 11 x 0.70 -> median 0.70, which fails the
    # baseline-relative clear bound (0.75): drift stays set.
    rec_ts = bad_ts + DAY_S
    for i in range(11):
        tracker.record(rec_ts + i, 0.70, 0)
        verdict = tracker.evaluate()
    assert verdict is not None
    assert verdict.drift is True
    # Eleven episodes at 0.76 push the 0.1s out of the 20-entry window:
    # median 0.76 >= 0.75 -> cleared on the first such evaluation.
    for i in range(11):
        tracker.record(rec_ts + 100 + i, 0.76, 0)
        verdict = tracker.evaluate()
    assert verdict is not None
    assert verdict.drift is False
    assert verdict.changed is True


def test_clear_margin_ratio_absolute_bound_alone() -> None:
    # Drift set by edges_refused; on recovery a window ratio of 0.259 clears
    # NEITHER bound (0.259 > 0.25 and > baseline 0 + 0.10) so drift stays,
    # while 0.231 clears through the ABSOLUTE bound alone (0.231 <= 0.25,
    # still above the relative bound). The bad episodes leave the 20-entry
    # window one at a time (deltas of 1), giving exact control of the ratio.
    tracker = _tracker_with_baseline(0.9)
    bad_ts = BASE_TS + 40 * DAY_S
    cum = 0
    verdict = None
    for i in range(BAD_N):
        cum += 1
        tracker.record(bad_ts + i, 0.9, cum)
        verdict = tracker.evaluate()
    assert verdict is not None and verdict.drift is True
    assert verdict.reason == "edges_refused"
    # 13 clean episodes: the window is 7 bad (7 refused) + 13 good = 0.259.
    rec_ts = bad_ts + DAY_S
    for i in range(13):
        tracker.record(rec_ts + i, 0.9, cum)
        verdict = tracker.evaluate()
    assert verdict is not None
    assert verdict.drift is True
    # One more: 6 bad (6 refused) + 14 good -> 6/26 = 0.231 <= 0.25 -> clear.
    tracker.record(rec_ts + 100, 0.9, cum)
    verdict = tracker.evaluate()
    assert verdict is not None
    assert verdict.drift is False
    assert verdict.changed is True


def test_refused_deltas_from_cumulative_counts() -> None:
    tracker = LiveDrift()
    tracker.record(1.0, 0.9, 5)  # first episode: refused 0 by definition
    tracker.record(2.0, 0.9, 5)  # no burst between the closes
    tracker.record(3.0, 0.9, 9)  # burst of 4 between the two closes
    tracker.record(4.0, 0.9, 9)
    evidence = tracker.to_dict()["evidence"]
    assert [e["refused"] for e in evidence] == [0.0, 0.0, 4.0, 0.0]


def test_burst_between_episodes_counts_once_in_next_close() -> None:
    # The review Round 1 edge: a refused burst between two recorded closes
    # lands in exactly one window entry (the next close), not both.
    tracker = _tracker_with_baseline(0.9)
    before = len(tracker.to_dict()["evidence"])
    tracker.record(BASE_TS + 100, 0.9, 7)  # carries a burst of 7
    tracker.record(BASE_TS + 101, 0.9, 7)  # carries none
    evidence = tracker.to_dict()["evidence"]
    assert len(evidence) == before + 2
    assert evidence[-2]["refused"] == 7.0
    assert evidence[-1]["refused"] == 0.0


def test_to_dict_from_dict_round_trip() -> None:
    tracker = _tracker_with_baseline(0.9)
    verdict = None
    for i in range(BAD_N):
        tracker.record(BASE_TS + 40 * DAY_S + i, 0.1, 0)
        verdict = tracker.evaluate()
    assert verdict is not None and verdict.drift is True
    stored = tracker.to_dict()
    restored = LiveDrift.from_dict(stored)
    # The restored tracker keeps the active drift, the reason, the since
    # timestamp and the hysteresis: an evaluation still reports drift without
    # needing two consecutive overs again.
    verdict2 = restored.evaluate()
    assert verdict2.drift is True
    assert verdict2.reason == "match_score_low"
    assert verdict2.since_ts == pytest.approx(verdict.since_ts)
    assert restored.evidence_state() == "sufficient"
    # Refused deltas continue from the restored cumulative count: the previous
    # episodes recorded cum 0, so a close at cum 12 carries a burst of 12.
    restored.record(BASE_TS + 41 * DAY_S, 0.1, 12)
    evidence = restored.to_dict()["evidence"]
    assert evidence[-1]["refused"] == 12.0


def test_from_dict_defensive_on_malformed_entries() -> None:
    for bad in (None, "junk", 42, {"evidence": "junk"}, {"evidence": [{"ts": "x"}]}):
        tracker = LiveDrift.from_dict(bad)
        assert tracker.evidence_state() == "insufficient"
        verdict = tracker.evaluate()
        assert verdict.drift is False
        assert verdict.insufficient is True


def test_evidence_max_bound() -> None:
    tracker = LiveDrift()
    for i in range(100):
        tracker.record(BASE_TS + i, 0.9, 0)
    evidence = tracker.to_dict()["evidence"]
    assert len(evidence) == EVIDENCE_MAX
    # The baseline survived (computed from the first 10) and the tracker still
    # evaluates sufficient over the trimmed tail.
    assert tracker.evidence_state() == "sufficient"


def test_reset_starts_over() -> None:
    tracker = _tracker_with_baseline(0.9)
    verdict = None
    for i in range(BAD_N):
        tracker.record(BASE_TS + 40 * DAY_S + i, 0.1, 0)
        verdict = tracker.evaluate()
    assert verdict is not None and verdict.drift is True
    tracker.reset()
    assert tracker.evidence_state() == "insufficient"
    verdict = tracker.evaluate()
    assert verdict.drift is False
    assert verdict.insufficient is True
    # After the D-22 reset the first 10 episodes rebuild the baseline.
    for i in range(LIVE_BASELINE_EPISODES):
        tracker.record(BASE_TS + 80 * DAY_S + i, 0.2, 0)
    assert tracker.evidence_state() == "sufficient"


def test_reason_precedence_match_score_low_wins() -> None:
    tracker = _tracker_with_baseline(0.9)
    cum = 0
    verdict = None
    for i in range(BAD_N):
        cum += 3
        # Bad score AND refused burst; the baseline aged out of the window.
        tracker.record(BASE_TS + 40 * DAY_S + i, 0.1, cum)
        verdict = tracker.evaluate()
    assert verdict is not None
    assert verdict.drift is True
    assert verdict.reason == "match_score_low"


# --- Holdout-basis drift (Plan 05-06; D-15, D-17, D-23) ---


def _holdout_history(*windows: tuple[str, float], run_id: str = "s1") -> list[dict[str, Any]]:
    """History entries [(holdout_end, error_pct)] tagged with one shadow set."""
    return [
        {"shadow_run_id": run_id, "holdout_end": end, "energy_error_pct": err}
        for end, err in windows
    ]


def _state_roundtrip(state: Any) -> dict[str, Any]:
    """Simulate the validation Store round trip the manager performs."""
    from custom_components.powerguesser.engine.validation.drift import (
        holdout_state_to_dict,
    )

    return holdout_state_to_dict(state)


def test_holdout_drift_needs_two_consecutive_windows() -> None:
    """D-15 on holdout errors: set after 2 consecutive windows at least 5 pp
    above and 1.5x the best earlier error of the same shadow set; clear after
    one window 5 pp under the threshold; a window in between keeps it set."""
    from custom_components.powerguesser.engine.validation.drift import (
        evaluate_holdout_drift,
    )

    result = evaluate_holdout_drift(_holdout_history(("2026-11-10", 10.0)), None, "s1")
    assert result["drift"] is False
    assert result["reason"] is None
    assert result["since"] is None
    assert result["best_error_pct"] is None
    assert result["shadow_run_id"] == "s1"
    assert result["last_end"] == "2026-11-10"

    history = _holdout_history(("2026-11-10", 10.0), ("2026-11-17", 16.0))
    result = evaluate_holdout_drift(history, _state_roundtrip(result), "s1")
    assert result["drift"] is False  # over once (16 >= max(15, 15))
    assert result["hysteresis"].over_count == 1
    assert result["best_error_pct"] == 10.0

    history.append({"shadow_run_id": "s1", "holdout_end": "2026-11-24", "energy_error_pct": 17.0})
    result = evaluate_holdout_drift(history, _state_roundtrip(result), "s1")
    assert result["drift"] is True
    assert result["reason"] == "validation_error_rising"
    assert result["since"] == "2026-11-24"  # the window that set it
    assert result["best_error_pct"] == 10.0

    # 14.0 does not clear (threshold 15, clear needs <= 10): drift stays set.
    history.append({"shadow_run_id": "s1", "holdout_end": "2026-12-01", "energy_error_pct": 14.0})
    result = evaluate_holdout_drift(history, _state_roundtrip(result), "s1")
    assert result["drift"] is True
    assert result["reason"] == "validation_error_rising"
    assert result["since"] == "2026-11-24"  # since stays the setting window

    # 9.0 clears (<= 15 - 5).
    history.append({"shadow_run_id": "s1", "holdout_end": "2026-12-08", "energy_error_pct": 9.0})
    result = evaluate_holdout_drift(history, _state_roundtrip(result), "s1")
    assert result["drift"] is False
    assert result["reason"] is None
    assert result["since"] is None


def test_holdout_drift_scoped_to_shadow_set() -> None:
    """D-17: only the current shadow set's errors feed the comparison; a new
    shadow set starts a fresh window and never reads as decay against an
    older model's best."""
    from custom_components.powerguesser.engine.validation.drift import (
        evaluate_holdout_drift,
    )

    history = _holdout_history(("2026-11-10", 8.0), ("2026-11-17", 9.0), run_id="s1") + [
        {"shadow_run_id": "s2", "holdout_end": "2026-11-24", "energy_error_pct": 20.0},
        {"shadow_run_id": "s2", "holdout_end": "2026-12-01", "energy_error_pct": 21.0},
    ]
    prev_s1 = {
        "shadow_run_id": "s1",
        "drift": True,
        "reason": "validation_error_rising",
        "since": "2026-11-17",
        "last_end": "2026-11-17",
        "hysteresis": {"active": True, "over_count": 2},
    }
    result = evaluate_holdout_drift(history, prev_s1, "s2")
    # Counted against s1's best (8.0) both s2 windows would be over; scoped
    # to s2 the best earlier error is 20.0 and 21.0 is far under the rule.
    assert result["drift"] is False
    assert result["reason"] is None
    assert result["since"] is None
    assert result["best_error_pct"] == 20.0
    assert result["shadow_run_id"] == "s2"
    assert result["hysteresis"].active is False
    assert result["hysteresis"].over_count == 0

    # A single s2 window: drift false, no best earlier error.
    result = evaluate_holdout_drift(
        _holdout_history(("2026-11-24", 20.0), run_id="s2"), None, "s2"
    )
    assert result["drift"] is False
    assert result["best_error_pct"] is None


def test_holdout_drift_same_window_rerun_does_not_advance() -> None:
    """RESEARCH Pitfall 6: a re-measured window replaces its history entry
    and does not advance the consecutive-over counter."""
    from custom_components.powerguesser.engine.validation.drift import (
        evaluate_holdout_drift,
    )

    history = _holdout_history(("2026-11-10", 8.0), ("2026-11-17", 20.0))
    prev = {
        "shadow_run_id": "s1",
        "drift": False,
        "reason": None,
        "since": None,
        "last_end": "2026-11-17",
        "hysteresis": {"active": False, "over_count": 1},
    }
    # The same holdout window measured again: no evaluation, counter stays.
    result = evaluate_holdout_drift(history, prev, "s1")
    assert result["drift"] is False
    assert result["hysteresis"].over_count == 1

    # The next distinct window is the second consecutive over: drift sets
    # with since = the new window (not the re-measured one).
    history.append({"shadow_run_id": "s1", "holdout_end": "2026-11-24", "energy_error_pct": 30.0})
    result = evaluate_holdout_drift(history, _state_roundtrip(result), "s1")
    assert result["drift"] is True
    assert result["reason"] == "validation_error_rising"
    assert result["since"] == "2026-11-24"

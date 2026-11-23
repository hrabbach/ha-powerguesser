"""Live drift guard: a pure state machine fed by episode-close events (D-15).

RESEARCH Pattern 4: no HA imports and no numpy (``statistics.median`` only),
so the DeviceRuntime ``@callback`` detector drain path can record and evaluate
one closed episode without I/O or allocation. The detector's ``ClosedEpisode``
stream feeds ``LiveDrift``; ``Hysteresis`` is the one tested definition of the
D-15 set-after-2 / clear-after-1-better rule (the D-23 class-holdout drift in
Plan 05-06 reuses the same shape).

A close without a known detector state records no evidence upstream
(``OnlineDetector._finish_on_episode``), so no fallback score can enter a
window; refused edges between recorded closes reach the next recorded close
through the cumulative unattributed counter (cross-AI review Round 1).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import logging
import math
import statistics
from typing import Any

from .constants import (
    DRIFT_CLEAR_MARGIN,
    DRIFT_SET_AFTER,
    EVIDENCE_MAX,
    HISTORY_MAX,
    HOLDOUT_DRIFT_FACTOR,
    HOLDOUT_DRIFT_MIN_PP,
    LIVE_BASELINE_EPISODES,
    LIVE_MIN_EPISODES,
    LIVE_WINDOW_EPISODES,
    LIVE_WINDOW_MAX_DAYS,
    MATCH_SCORE_MAX_DROP,
    MATCH_SCORE_MIN,
    PROGRAM_LIVE_BASELINE_EPISODES,
    PROGRAM_LIVE_MIN_EPISODES,
    PROGRAM_LIVE_WINDOW_MAX_DAYS,
    REFUSED_RATIO_MAX,
    REFUSED_RATIO_MAX_RISE,
)

_DAY_S = 86400.0

# Stdlib logging keeps the engine HA-free; used only to make a typo'd kind
# visible instead of silently changing the drift window (L285/L289@8b3be9c).
_LOGGER = logging.getLogger(__name__)

# The D-16 drift_reason vocabulary for the live proxy (the D-23
# validation_error_rising reason arrives with Plan 05-06).
REASON_MATCH_SCORE_LOW = "match_score_low"
REASON_EDGES_REFUSED = "edges_refused"
_REASONS = frozenset({REASON_MATCH_SCORE_LOW, REASON_EDGES_REFUSED})

# The D-16/D-23 drift_reason for the holdout basis (plug devices and the
# class medians that confirmed devices display).
REASON_VALIDATION_ERROR_RISING = "validation_error_rising"


def _refused_ratio(entries: list[dict[str, float]]) -> float:
    """unattributed / (matched + unattributed) over the given window entries."""
    refused = sum(e["refused"] for e in entries)
    denom = len(entries) + refused
    return refused / denom if denom > 0 else 0.0


@dataclass(frozen=True, slots=True)
class DriftVerdict:
    """The outcome of one ``LiveDrift.evaluate()`` call (D-15)."""

    drift: bool
    reason: str | None
    insufficient: bool
    changed: bool
    since_ts: float | None


class Hysteresis:
    """Set after N consecutive over evaluations; clear on one clear evaluation.

    One tested definition of the D-15 hysteresis, shared by every drift path.
    While inactive, ``over`` increments the consecutive-over counter and sets
    ``active`` at ``set_after``; a non-over evaluation resets the counter.
    While active, only ``clear_ok`` clears (one good evaluation, D-15).
    """

    def __init__(self, set_after: int = DRIFT_SET_AFTER) -> None:
        self._set_after = max(1, int(set_after))
        self.active = False
        self.over_count = 0

    def update(self, over: bool, clear_ok: bool) -> bool:
        """Feed one evaluation; return the resulting active state."""
        if self.active:
            if clear_ok:
                self.active = False
                self.over_count = 0
            return self.active
        if over:
            self.over_count += 1
            if self.over_count >= self._set_after:
                self.active = True
        else:
            self.over_count = 0
        return self.active

    def reset(self) -> None:
        """Drop the state (insufficient evidence or a D-22 version reset)."""
        self.active = False
        self.over_count = 0

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe state for the hot Store."""
        return {"active": self.active, "over_count": self.over_count}

    @classmethod
    def from_dict(cls, data: Any, set_after: int = DRIFT_SET_AFTER) -> "Hysteresis":
        """Restore from stored state; anything malformed yields a fresh one."""
        hyst = cls(set_after)
        if isinstance(data, dict):
            hyst.active = bool(data.get("active"))
            try:
                hyst.over_count = int(data.get("over_count") or 0)
            except (TypeError, ValueError):
                hyst.over_count = 0
        return hyst


class LiveDrift:
    """Per-device live drift tracker fed by closed-episode events (D-15, D-26).

    Evaluation runs when an ON episode closes, never per sample. The window is
    the last ``LIVE_WINDOW_EPISODES`` recorded episodes within ``window_days``
    of the newest (at least ``min_episodes`` required, else insufficient
    evidence: drift reads false, T-05-18 shows why instead of a silent false).
    The baseline (median match_score and refused ratio of the first
    ``baseline_episodes`` episodes after the last reset) is computed once;
    devices confirmed before this plan shipped have no stored evidence and
    baseline from their first ``baseline_episodes`` closed episodes after the
    upgrade, which describe the current parameters. Drift sets after 2
    consecutive over evaluations and clears after 1 evaluation clear of
    triggering by ``DRIFT_CLEAR_MARGIN`` in each metric's own units,
    asymmetrically by rule shape.

    ``kind`` selects the window (D-26): a program device (rarely run) uses the
    PROGRAM_LIVE_* constants (min 3, baseline 3, 42 days) so a verdict is
    reachable from a few weeks of normal use, while every other kind keeps the
    D-15 switched window (LIVE_MIN_EPISODES / LIVE_BASELINE_EPISODES /
    LIVE_WINDOW_MAX_DAYS). LIVE_WINDOW_EPISODES is shared by both kinds.
    """

    def __init__(self, kind: str = "switched") -> None:
        if kind == "program":
            self._min_episodes = PROGRAM_LIVE_MIN_EPISODES
            self._baseline_episodes = PROGRAM_LIVE_BASELINE_EPISODES
            self._window_days = PROGRAM_LIVE_WINDOW_MAX_DAYS
        else:
            self._min_episodes = LIVE_MIN_EPISODES
            self._baseline_episodes = LIVE_BASELINE_EPISODES
            self._window_days = LIVE_WINDOW_MAX_DAYS
            if kind != "switched":
                # L285/L289@8b3be9c: a typo'd kind still uses the switched
                # window, but says so once instead of changing the window
                # silently.
                _LOGGER.debug(
                    "LiveDrift kind %r is neither 'switched' nor 'program'; "
                    "using the switched D-15 window",
                    kind,
                )
        self._evidence: list[dict[str, float]] = []
        self._last_cum = 0
        self._baseline: dict[str, float] | None = None
        self._hyst = Hysteresis()
        self._since_ts: float | None = None
        self._reason: str | None = None

    def record(self, end_ts: float, match_score: float, unattributed: int) -> None:
        """Record one closed ON episode (verified or not) with a known state.

        ``unattributed`` is the detector runtime's CUMULATIVE refused-edge
        counter at the close; this episode's refused delta is the difference
        from the previously recorded close (0 for the first), so a refused
        burst between two episodes counts once, in the window of the next
        closing episode.
        """
        cum = int(unattributed)
        refused = max(0, cum - self._last_cum) if self._evidence else 0
        self._evidence.append(
            {"ts": float(end_ts), "score": float(match_score), "refused": float(refused)}
        )
        self._last_cum = cum
        if len(self._evidence) > EVIDENCE_MAX:
            del self._evidence[: len(self._evidence) - EVIDENCE_MAX]
        if self._baseline is None and len(self._evidence) >= self._baseline_episodes:
            base = self._evidence[: self._baseline_episodes]
            self._baseline = {
                "score": float(statistics.median(e["score"] for e in base)),
                "ratio": _refused_ratio(base),
            }

    def evaluate(self) -> DriftVerdict:
        """Evaluate the current window; return the verdict (D-15)."""
        prev_drift = self._hyst.active
        prev_reason = self._reason
        if not self._evidence:
            return DriftVerdict(
                drift=False,
                reason=None,
                insufficient=True,
                changed=False,
                since_ts=self._since_ts,
            )
        newest = self._evidence[-1]["ts"]
        min_ts = newest - self._window_days * _DAY_S
        window = [e for e in self._evidence if e["ts"] >= min_ts]
        window = window[-LIVE_WINDOW_EPISODES:]
        base = self._baseline
        if base is None or len(window) < self._min_episodes:
            # D-15: rare devices sit in insufficient evidence (drift: false);
            # the hysteresis counter resets so the next over evaluations must
            # again be consecutive.
            self._hyst.reset()
            self._reason = None
            changed = (self._hyst.active != prev_drift) or (self._reason != prev_reason)
            return DriftVerdict(
                drift=self._hyst.active,
                reason=None,
                insufficient=True,
                changed=changed,
                since_ts=self._since_ts,
            )
        med = float(statistics.median(e["score"] for e in window))
        ratio = _refused_ratio(window)
        match_low = med < MATCH_SCORE_MIN or med < base["score"] - MATCH_SCORE_MAX_DROP
        refused_high = ratio > REFUSED_RATIO_MAX and ratio >= base["ratio"] + REFUSED_RATIO_MAX_RISE
        over = match_low or refused_high
        reason = (
            REASON_MATCH_SCORE_LOW if match_low else (REASON_EDGES_REFUSED if refused_high else None)
        )
        # D-15 asymmetric clear (cross-AI review Round 1): the OR rule
        # (match_score_low) clears only when BOTH of its bounds are
        # DRIFT_CLEAR_MARGIN clear; the AND rule (edges_refused) clears when
        # EITHER of its bounds is clear.
        score_clear = (
            med >= MATCH_SCORE_MIN + DRIFT_CLEAR_MARGIN
            and med >= base["score"] - MATCH_SCORE_MAX_DROP + DRIFT_CLEAR_MARGIN
        )
        ratio_clear = (
            ratio <= REFUSED_RATIO_MAX - DRIFT_CLEAR_MARGIN
            or ratio <= base["ratio"] + REFUSED_RATIO_MAX_RISE - DRIFT_CLEAR_MARGIN
        )
        clear_ok = score_clear and ratio_clear
        was_active = self._hyst.active
        self._hyst.update(over, clear_ok)
        if self._hyst.active:
            if not was_active:
                self._since_ts = newest
            self._reason = reason
        elif was_active or self._reason is not None:
            self._reason = None
            self._since_ts = None
        changed = (self._hyst.active != prev_drift) or (self._reason != prev_reason)
        return DriftVerdict(
            drift=self._hyst.active,
            reason=self._reason,
            insufficient=False,
            changed=changed,
            since_ts=self._since_ts,
        )

    def evidence_state(self) -> str:
        """Return "sufficient" or "insufficient" for the diagnostic sensor.

        Sufficient means an evaluation has run with a baseline and at least
        ``min_episodes`` window episodes; before any evaluation (or while
        insufficient) the device reads "insufficient" (D-15, T-05-18). The
        window and baseline sizes come from the tracker's kind (D-26): the
        program window is 3 / 3 / 42 days, every other kind the switched
        8 / 10 / 14.
        """
        if not self._evidence:
            return "insufficient"
        newest = self._evidence[-1]["ts"]
        min_ts = newest - self._window_days * _DAY_S
        window = [e for e in self._evidence if e["ts"] >= min_ts]
        window = window[-LIVE_WINDOW_EPISODES:]
        if self._baseline is None or len(window) < self._min_episodes:
            return "insufficient"
        return "sufficient"

    def reset(self) -> None:
        """Start over (D-22): a version activation/accept/rollback resets the
        window, the baseline and the hysteresis; evidence never carries over
        between parameter versions."""
        self._evidence.clear()
        self._last_cum = 0
        self._baseline = None
        self._hyst.reset()
        self._since_ts = None
        self._reason = None

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe state for the hot Store's top-level "drift" key."""
        return {
            "evidence": [dict(e) for e in self._evidence],
            "last_cum": self._last_cum,
            "baseline": dict(self._baseline) if self._baseline is not None else None,
            "hysteresis": self._hyst.to_dict(),
            "since_ts": self._since_ts,
            "reason": self._reason,
        }

    @classmethod
    def from_dict(cls, data: Any, kind: str = "switched") -> "LiveDrift":
        """Restore from stored state; anything malformed loads empty (T-05-17).

        ``kind`` is the caller's kind (the device record), never read from the
        stored dict: the stored shape carries no kind, so a program device
        restored without it would use the switched D-15 window instead of the
        D-26 program one. The stored to_dict shape is unchanged.
        """
        tracker = cls(kind=kind)
        if not isinstance(data, dict):
            return tracker
        raw_evidence = data.get("evidence")
        if isinstance(raw_evidence, list):
            for entry in raw_evidence[-EVIDENCE_MAX:]:
                if not isinstance(entry, dict):
                    continue
                try:
                    ts = float(entry.get("ts"))
                    score = float(entry.get("score"))
                    refused = float(entry.get("refused") or 0.0)
                except (TypeError, ValueError):
                    continue
                if not (math.isfinite(ts) and math.isfinite(score)):
                    continue
                tracker._evidence.append(
                    {"ts": ts, "score": score, "refused": refused}
                )
        try:
            tracker._last_cum = int(data.get("last_cum") or 0)
        except (TypeError, ValueError):
            tracker._last_cum = 0
        raw_baseline = data.get("baseline")
        if isinstance(raw_baseline, dict):
            try:
                tracker._baseline = {
                    "score": float(raw_baseline.get("score")),
                    "ratio": float(raw_baseline.get("ratio")),
                }
            except (TypeError, ValueError):
                tracker._baseline = None
        tracker._hyst = Hysteresis.from_dict(data.get("hysteresis"))
        since = data.get("since_ts")
        try:
            tracker._since_ts = float(since) if since is not None else None
        except (TypeError, ValueError):
            tracker._since_ts = None
        reason = data.get("reason")
        tracker._reason = str(reason) if reason in _REASONS else None
        return tracker


# --- Holdout-basis drift (Plan 05-06; D-15, D-17, D-23) ---


def _coerce_holdout_state(raw: Any, shadow_run_id: str) -> dict[str, Any]:
    """Restore an evaluation state; a missing or foreign-set state starts fresh.

    A fresh state (drift false, since None, a new Hysteresis(DRIFT_SET_AFTER))
    is exactly what a new shadow set needs: a new masked model never reads as
    decay against an older model's best (D-17).
    """
    state: dict[str, Any] = {
        "drift": False,
        "reason": None,
        "since": None,
        "last_end": None,
        "shadow_run_id": shadow_run_id,
        "hysteresis": Hysteresis(DRIFT_SET_AFTER),
    }
    if not isinstance(raw, Mapping) or raw.get("shadow_run_id") != shadow_run_id:
        return state
    state["drift"] = bool(raw.get("drift"))
    if state["drift"]:
        reason = raw.get("reason")
        state["reason"] = str(reason) if reason else None
        since = raw.get("since")
        state["since"] = str(since) if since else None
    last_end = raw.get("last_end")
    state["last_end"] = str(last_end) if last_end else None
    state["hysteresis"] = Hysteresis.from_dict(raw.get("hysteresis"))
    return state


def evaluate_holdout_drift(
    history: Sequence[Mapping[str, Any]],
    prev_state: Mapping[str, Any] | None,
    shadow_run_id: str,
) -> dict[str, Any]:
    """D-15 holdout-basis drift over one plug's or one class's history.

    ``history`` holds every stored entry of one history key (a plug entity id
    or ``class:<class>``), each tagged with its shadow set's run_id. Only the
    entries of ``shadow_run_id`` feed the comparison — errors measured on a
    different shadow set are never compared (D-17: a new shadow set is a new
    masked model, so VAL-05 decay means the same model getting worse). A
    ``prev_state`` of another (or no) shadow set starts fresh.

    The D-15 rule on the current set's entries, oldest first: no evaluation
    with fewer than 2 entries, or when the newest window was already evaluated
    (a re-run replaces its entry and never advances the hysteresis, RESEARCH
    Pitfall 6); otherwise ``best_error_pct`` is the best EARLIER error, the
    threshold is max(best + HOLDOUT_DRIFT_MIN_PP, best x HOLDOUT_DRIFT_FACTOR),
    ``over`` is latest >= threshold, and ``clear_ok`` is latest <= threshold -
    HOLDOUT_DRIFT_MIN_PP (the D-15 0.05 margin in error-percentage points).
    Drift sets after DRIFT_SET_AFTER consecutive over evaluations and clears
    after one clear evaluation through the shared Hysteresis. ``since`` is the
    holdout_end of the window that set drift; drift, since and best_error_pct
    are None/False for a fresh or cleared state.
    """
    current = sorted(
        (
            entry
            for entry in history
            if isinstance(entry, Mapping)
            and entry.get("shadow_run_id") == shadow_run_id
            and isinstance(entry.get("holdout_end"), str)
            and isinstance(entry.get("energy_error_pct"), (int, float))
            and not isinstance(entry.get("energy_error_pct"), bool)
        ),
        key=lambda e: str(e["holdout_end"]),
    )
    state = _coerce_holdout_state(prev_state, shadow_run_id)
    hysteresis: Hysteresis = state["hysteresis"]
    latest_end = str(current[-1]["holdout_end"]) if current else None

    evaluated = len(current) >= 2 and latest_end is not None and (
        latest_end != state["last_end"]
    )
    best_earlier: float | None = None
    if evaluated:
        errors = [float(entry["energy_error_pct"]) for entry in current]
        latest = errors[-1]
        best_earlier = min(errors[:-1])
        threshold = max(
            best_earlier + HOLDOUT_DRIFT_MIN_PP,
            best_earlier * HOLDOUT_DRIFT_FACTOR,
        )
        over = latest >= threshold
        clear_ok = latest <= threshold - DRIFT_CLEAR_MARGIN * 100.0
        was_active = hysteresis.active
        hysteresis.update(over, clear_ok)
        if hysteresis.active:
            state["since"] = latest_end if not was_active else state["since"]
        else:
            state["since"] = None
    state["drift"] = hysteresis.active
    state["reason"] = REASON_VALIDATION_ERROR_RISING if hysteresis.active else None
    if not hysteresis.active:
        state["since"] = None
    state["last_end"] = latest_end
    return {
        "drift": state["drift"],
        "reason": state["reason"],
        "since": state["since"],
        "best_error_pct": best_earlier,
        "last_end": state["last_end"],
        "shadow_run_id": shadow_run_id,
        "hysteresis": hysteresis,
    }


def holdout_state_to_dict(state: Mapping[str, Any]) -> dict[str, Any]:
    """JSON-safe holdout-drift state for the powerguesser.validation Store."""
    hysteresis = state.get("hysteresis")
    if not isinstance(hysteresis, Hysteresis):
        hysteresis = Hysteresis.from_dict(hysteresis)
    drift = bool(state.get("drift"))
    return {
        "drift": drift,
        "reason": str(state.get("reason")) if drift and state.get("reason") else None,
        "since": str(state.get("since")) if drift and state.get("since") else None,
        "last_end": str(state.get("last_end")) if state.get("last_end") else None,
        "shadow_run_id": str(state.get("shadow_run_id") or ""),
        "hysteresis": hysteresis.to_dict(),
    }


def clean_holdout_entry(raw: Any) -> dict[str, Any] | None:
    """One history entry, or None when malformed (T-05-21).

    An entry needs its shadow set's run_id, a holdout end date and a numeric
    energy error; anything else is dropped on load, for plug keys and
    ``class:`` keys alike.
    """
    if not isinstance(raw, Mapping):
        return None
    shadow_run_id = raw.get("shadow_run_id")
    holdout_end = raw.get("holdout_end")
    error = raw.get("energy_error_pct")
    if not isinstance(shadow_run_id, str) or not shadow_run_id:
        return None
    if not isinstance(holdout_end, str) or not holdout_end:
        return None
    if isinstance(error, bool) or not isinstance(error, (int, float)):
        return None
    return {
        "shadow_run_id": shadow_run_id,
        "holdout_end": holdout_end,
        "energy_error_pct": float(error),
    }


def clean_holdout_key_history(raw: Any) -> list[dict[str, Any]]:
    """A per-key history list with malformed entries dropped and trimmed."""
    if not isinstance(raw, list):
        return []
    entries = [entry for entry in (clean_holdout_entry(e) for e in raw) if entry is not None]
    entries.sort(key=lambda e: e["holdout_end"])
    if len(entries) > HISTORY_MAX:
        del entries[: len(entries) - HISTORY_MAX]
    return entries

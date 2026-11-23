"""Windowed sign self-check with hysteresis and threshold-based plausibility evaluation."""

from collections import deque
from collections.abc import Sequence

TOLERANCE_W = 50.0
TOLERANCE_FRAC = 0.03
WINDOW_S = 2700.0  # 45 minutes
RAISE_SHARE = 0.2  # 20%
MIN_SAMPLES = 30
CLEAR_AFTER_S = 43200.0  # 12 hours

NEGATIVE_GROSS = "negative_gross"
METERED_ABOVE_GROSS = "metered_above_gross"
CHECKS: tuple[str, ...] = (NEGATIVE_GROSS, METERED_ABOVE_GROSS)


def evaluate_plausibility(gross_w: float, metered_w: float) -> tuple[str, ...]:
    """Evaluate per-emission plausibility of gross and metered power.

    Returns tuple of failing checks:
    - 'negative_gross' if gross < -max(50 W, 3% of |gross|)
    - 'metered_above_gross' if metered > gross + max(50 W, 3% of |gross|)
    """
    tolerance = max(TOLERANCE_W, TOLERANCE_FRAC * abs(gross_w))
    failing: list[str] = []

    if gross_w < -tolerance:
        failing.append(NEGATIVE_GROSS)

    if metered_w > (max(0.0, gross_w) + tolerance):
        failing.append(METERED_ABOVE_GROSS)

    return tuple(failing)


class SignSelfCheck:
    """Tracks windowed plausibility violations and maintains active issue states with hysteresis.

    negative_gross raises when at least RAISE_SHARE of the samples in the window violate and
    clears CLEAR_AFTER_S after the last violating sample.

    metered_above_gross additionally requires the window condition: the window mean of
    (metered - gross) exceeds max(50 W, 3 % of the window mean |gross|). It clears
    CLEAR_AFTER_S after the window condition last held, so transient metered/gross timing
    skew (positive per-sample spikes, negative mean) neither raises nor keeps it alive.
    """

    def __init__(self) -> None:
        """Initialize sliding windows and violation tracking."""
        self._windows: dict[str, deque[tuple[float, bool]]] = {
            c: deque() for c in CHECKS
        }
        self._last_violation: dict[str, float | None] = {
            c: None for c in CHECKS
        }
        self._balance: deque[tuple[float, float, float]] = deque()
        self._last_surplus_ts: float | None = None
        self._active: set[str] = set()

    def observe(
        self,
        ts: float,
        failing: Sequence[str],
        *,
        gross_w: float,
        metered_w: float,
    ) -> frozenset[str]:
        """Record a sample, its failing checks and power values, returning currently active issues."""
        failing_set = set(failing)
        cutoff = ts - WINDOW_S

        self._balance.append((ts, gross_w, metered_w))
        while self._balance and self._balance[0][0] <= cutoff:
            self._balance.popleft()
        surplus = self._window_surplus_holds()
        if surplus:
            self._last_surplus_ts = ts

        for check in CHECKS:
            violated = check in failing_set
            if violated:
                self._last_violation[check] = ts

            w = self._windows[check]
            w.append((ts, violated))

            while w and w[0][0] <= cutoff:
                w.popleft()

            if len(w) >= MIN_SAMPLES:
                violated_count = sum(1 for _, v in w if v)
                share_met = (violated_count / len(w)) >= RAISE_SHARE
                if share_met and (check != METERED_ABOVE_GROSS or surplus):
                    self._active.add(check)

            if check in self._active:
                last_v = (
                    self._last_surplus_ts
                    if check == METERED_ABOVE_GROSS
                    else self._last_violation[check]
                )
                if last_v is not None and (ts - last_v) >= CLEAR_AFTER_S:
                    self._active.discard(check)

        return frozenset(self._active)

    def _window_surplus_holds(self) -> bool:
        """Window condition: mean(metered - gross) > max(50 W, 3 % of mean |gross|)."""
        n = len(self._balance)
        if n < MIN_SAMPLES:
            return False
        mean_gross = sum(g for _, g, _ in self._balance) / n
        mean_surplus = sum(m - g for _, g, m in self._balance) / n
        return mean_surplus > max(TOLERANCE_W, TOLERANCE_FRAC * abs(mean_gross))

    def reset(self) -> None:
        """Reset all windows, violation timestamps, and active checks."""
        for c in CHECKS:
            self._windows[c].clear()
            self._last_violation[c] = None
        self._balance.clear()
        self._last_surplus_ts = None
        self._active.clear()

"""Per-tick timing ring (SC5 instrumentation, HA-free)."""

from collections import deque
import math


class TickTimer:
    """Ring of per-tick durations with p50/p95/max summary in ms.

    ``ticks`` counts every record ever seen, not just the window.  Percentiles
    use nearest-rank on a sorted copy; an empty timer reports zeros.
    """

    def __init__(self, maxlen: int = 256) -> None:
        self._durations: deque[int] = deque(maxlen=int(maxlen))
        self._ticks = 0

    def record(self, ns: int) -> None:
        """Record one tick duration in nanoseconds."""
        self._durations.append(int(ns))
        self._ticks += 1

    def summary(self) -> dict[str, float | int]:
        """Return {"p50_ms", "p95_ms", "max_ms", "ticks"} rounded to 3 decimals."""
        if not self._durations:
            return {"p50_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0, "ticks": self._ticks}
        s = sorted(self._durations)
        n = len(s)

        def _rank(frac: float) -> int:
            return s[min(n - 1, max(0, math.ceil(frac * n) - 1))]

        return {
            "p50_ms": round(_rank(0.50) / 1e6, 3),
            "p95_ms": round(_rank(0.95) / 1e6, 3),
            "max_ms": round(s[-1] / 1e6, 3),
            "ticks": self._ticks,
        }

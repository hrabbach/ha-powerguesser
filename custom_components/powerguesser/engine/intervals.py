"""Sample interval profiling and resolution gating engine."""

from collections import Counter, deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
import math
import statistics

from .settle import STALE_AFTER_S, ReReportDetector

WINDOW = 120
MIN_DELTAS = 20
STALE_FACTOR = 2.5
FALLBACK_INTERVAL_S = 30.0
ZERO_W_EPS = 0.5
PUBLISH_HYSTERESIS_S = 1.0


class Mode(StrEnum):
    """Classification mode for a power source."""

    RE_REPORTING = "re_reporting"
    ON_CHANGE = "on_change"


class MeasuredFrom(StrEnum):
    """Source of interval measurement (D-23)."""

    REPORTS = "reports"
    CHANGES = "changes"


class EventType(StrEnum):
    """Catalogued event types for NILM decomposition."""

    SHORT_EVENT = "short_event"
    STEP = "step"
    CYCLE_SHAPE = "cycle_shape"
    CONTINUOUS = "continuous"


EVENT_TYPE_MIN_S: dict[EventType, float] = {
    EventType.SHORT_EVENT: 10.0,
    EventType.STEP: 90.0,
    EventType.CYCLE_SHAPE: 180.0,
    EventType.CONTINUOUS: 0.0,
}


@dataclass(frozen=True, slots=True)
class SourceProfile:
    """Live profile data for a single tracked power source."""

    entity_id: str
    mode: Mode
    median_s: float | None
    p90_s: float | None
    delta_count: int
    restored: bool

    @property
    def measured_from(self) -> MeasuredFrom | None:
        """Indicate whether the interval was measured from reports or changes (D-23)."""
        if self.median_s is None:
            return None
        return (
            MeasuredFrom.REPORTS
            if self.mode == Mode.RE_REPORTING
            else MeasuredFrom.CHANGES
        )


@dataclass(frozen=True, slots=True)
class ProfileSource:
    """Metadata for a source tracked by the interval profiler."""

    entity_id: str
    role: str
    label: str


@dataclass(frozen=True, slots=True)
class ProfileSnapshot:
    """Coordinator snapshot combining all source profiles and effective gating."""

    sources: Mapping[str, SourceProfile]
    effective_s: float
    assumed: bool
    enabled: tuple[EventType, ...]
    disabled: Mapping[EventType, str]


class IntervalProfiler:
    """Measures native sample intervals live per source."""

    def __init__(self, detector: ReReportDetector | None = None) -> None:
        """Initialize profiler over a shared or private ReReportDetector."""
        self.detector: ReReportDetector = (
            detector if detector is not None else ReReportDetector()
        )
        self._deltas: deque[float] = deque(maxlen=WINDOW)
        self.prev_ts: float | None = None
        self.prev_w: float | None = None
        self._published_median: float | None = None
        self._restored_median: float | None = None
        self._restored_p90: float | None = None
        self._restored: bool = False

    @property
    def mode(self) -> Mode:
        """Current mode derived from the shared ReReportDetector (D-24)."""
        return Mode.RE_REPORTING if self.detector.re_reporting else Mode.ON_CHANGE

    @property
    def median_s(self) -> float | None:
        """Published median interval held by hysteresis."""
        return self._published_median

    @property
    def p90_s(self) -> float | None:
        """90th percentile interval once MIN_DELTAS deltas exist, or restored p90."""
        if len(self._deltas) >= MIN_DELTAS:
            return float(statistics.quantiles(self._deltas, n=10)[8])
        return self._restored_p90

    @property
    def delta_count(self) -> int:
        """Number of counted deltas currently in the window."""
        return len(self._deltas)

    @property
    def restored(self) -> bool:
        """True if the median comes from restore() and live deltas < MIN_DELTAS."""
        return self._restored

    @property
    def last_ts(self) -> float | None:
        """Timestamp of the most recent observed sample."""
        return self.prev_ts

    def stale_after_s(self) -> float:
        """Staleness timeout: 2.5x median for re-reporting channels, else STALE_AFTER_S (D-15)."""
        if self.mode == Mode.RE_REPORTING and self._published_median is not None:
            return STALE_FACTOR * self._published_median
        return STALE_AFTER_S

    def histogram(self) -> dict[int, int]:
        """Distribution of rounded deltas in the window."""
        counts = Counter(round(d) for d in self._deltas)
        return dict(sorted(counts.items()))

    def restore(
        self,
        median_s: float,
        p90_s: float | None,
        mode: Mode | str,
    ) -> None:
        """Seed profiler from persisted Store data across restarts."""
        if not (
            isinstance(median_s, (int, float))
            and math.isfinite(median_s)
            and median_s > 0
        ):
            raise ValueError(f"Invalid restored median: {median_s}")
        if isinstance(mode, str):
            try:
                mode = Mode(mode)
            except ValueError:
                raise ValueError(f"Unknown mode: {mode}")
        if mode == Mode.RE_REPORTING:
            self.detector.mark_re_reporting()
        self._restored_median = float(median_s)
        self._restored_p90 = (
            float(p90_s)
            if (p90_s is not None and math.isfinite(p90_s) and p90_s > 0)
            else None
        )
        self._restored = True
        self._published_median = self._restored_median

    def observe(self, ts_s: float, *, reported: bool, w: float | None) -> bool:
        """Observe a live sample. Returns True if mode, median_s, or restored status changed."""
        old_mode = self.mode
        old_published = self._published_median
        old_restored = self._restored

        if reported:
            self.detector.observe_report(ts_s)

        current_mode = self.mode

        if self.prev_ts is not None and ts_s > self.prev_ts:
            if self.prev_w is not None and w is not None:
                delta = ts_s - self.prev_ts
                # While ON_CHANGE, skip deltas starting at 0 W (e.g. solar at night)
                if current_mode == Mode.ON_CHANGE and abs(self.prev_w) <= ZERO_W_EPS:
                    pass
                else:
                    self._deltas.append(delta)

        if self.prev_ts is None or ts_s > self.prev_ts:
            self.prev_ts = ts_s
            self.prev_w = w

        # Evaluate raw median
        if len(self._deltas) >= MIN_DELTAS:
            raw_median = float(statistics.median(self._deltas))
            if self._restored:
                # The first live raw median always replaces a restored one
                self._published_median = raw_median
                self._restored = False
                self._restored_median = None
                self._restored_p90 = None
            elif self._published_median is None:
                self._published_median = raw_median
            else:
                if abs(raw_median - self._published_median) >= PUBLISH_HYSTERESIS_S:
                    self._published_median = raw_median
        else:
            if self._restored:
                self._published_median = self._restored_median
            else:
                self._published_median = None

        new_mode = self.mode
        new_published = self._published_median
        new_restored = self._restored

        return (
            (new_mode != old_mode)
            or (new_published != old_published)
            or (new_restored != old_restored)
        )

    def snapshot(self, entity_id: str) -> SourceProfile:
        """Produce an immutable snapshot of current profiler state."""
        return SourceProfile(
            entity_id=entity_id,
            mode=self.mode,
            median_s=self.median_s,
            p90_s=self.p90_s,
            delta_count=self.delta_count,
            restored=self.restored,
        )


def effective_interval(gross_profiles: Iterable[SourceProfile]) -> tuple[float, bool]:
    """Calculate effective gross sample interval (D-13, D-23). Returns (interval_s, assumed)."""
    medians = [p.median_s for p in gross_profiles if p.median_s is not None]
    if medians:
        return (max(medians), False)
    return (FALLBACK_INTERVAL_S, True)


def gate(
    effective_s: float, assumed: bool
) -> tuple[tuple[EventType, ...], dict[EventType, str]]:
    """Determine enabled/disabled event types based on effective sample interval (D-18, SRC-07)."""
    enabled: list[EventType] = []
    disabled: dict[EventType, str] = {}
    assumed_suffix = " (assumed)" if assumed else ""
    eff_rounded = round(effective_s)

    for event_type in EventType:
        if event_type == EventType.CONTINUOUS:
            enabled.append(event_type)
            continue
        min_s = EVENT_TYPE_MIN_S[event_type]
        if min_s >= 2 * effective_s:
            enabled.append(event_type)
        else:
            max_interval = min_s / 2
            max_int_str = f"{max_interval:g}"
            disabled[event_type] = (
                f"{event_type.value} disabled: needs ≤{max_int_str} s interval, "
                f"effective is {eff_rounded} s{assumed_suffix}"
            )

    return (tuple(enabled), disabled)

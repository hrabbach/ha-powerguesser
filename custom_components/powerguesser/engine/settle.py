"""Settle-based temporal alignment for multi-source gross power."""

from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from .model import Role

STALE_AFTER_S = 90.0
ON_CHANGE_AFTER_S = 45.0
REREPORT_MIN_COUNT = 3
REREPORT_WINDOW_S = 600.0


@dataclass(slots=True)
class ReReportDetector:
    """Tracks state_reported events to classify an entity as re-reporting (D-24)."""

    re_reporting: bool = False
    _reports: deque[float] = field(
        default_factory=lambda: deque(maxlen=REREPORT_MIN_COUNT),
        init=False,
        repr=False,
    )

    def observe_report(self, ts: float) -> bool:
        """Observe a state_reported timestamp. Returns True only on flip."""
        if self.re_reporting:
            return False
        if self._reports and ts <= self._reports[-1]:
            return False
        self._reports.append(ts)
        if (
            len(self._reports) == REREPORT_MIN_COUNT
            and (self._reports[-1] - self._reports[0]) <= REREPORT_WINDOW_S
        ):
            self.re_reporting = True
            return True
        return False

    def mark_re_reporting(self) -> None:
        """Mark as re-reporting directly without counting."""
        self.re_reporting = True


@dataclass(slots=True)
class EntityState:
    """State tracking for a single gross channel entity."""

    value_w: float | None = None
    last_ts: float | None = None
    fresh: bool = False
    available: bool = False
    stale: bool = False
    on_change: bool = False


@dataclass(frozen=True, slots=True)
class AlignStatus:
    """Status summary produced by SettleAligner."""

    values: Mapping[str, float]
    blocking: tuple[str, ...]
    stale_sources: tuple[str, ...]
    solar_unavailable: tuple[str, ...]
    on_change_sources: tuple[str, ...]
    source_age_s: Mapping[str, float]
    emit_ts: float


class SettleAligner:
    """Aligns multiple power readings under D-05 settle mode."""

    def __init__(
        self,
        entity_roles: Mapping[str, Role],
        *,
        detectors: Mapping[str, ReReportDetector] | None = None,
        stale_after: Callable[[str], float] | None = None,
        profiled_interval: Callable[[str], float | None] | None = None,
    ) -> None:
        """Initialize aligner for a set of gross entities."""
        self.entity_roles: dict[str, Role] = dict(entity_roles)
        self.detectors: dict[str, ReReportDetector] = (
            {e: detectors[e] for e in self.entity_roles if e in detectors}
            if detectors is not None
            else {}
        )
        for e in self.entity_roles:
            if e not in self.detectors:
                self.detectors[e] = ReReportDetector()

        self._stale_after: Callable[[str], float] | None = stale_after
        self._profiled_interval: Callable[[str], float | None] | None = (
            profiled_interval
        )
        self._entities: dict[str, EntityState] = {
            e: EntityState() for e in self.entity_roles
        }
        self._emit_ts: float = 0.0
        self._init_ts: float | None = None
        # Closer-sample wait (A1): True after a skipped settle until the next emission,
        # so a settle is never skipped twice in a row.
        self._skipped: bool = False

    @property
    def values(self) -> Mapping[str, float | None]:
        """Current held values per entity."""
        return {e: s.value_w for e, s in self._entities.items()}

    @property
    def emit_ts(self) -> float | None:
        """Latest timestamp among the emitting sample set."""
        return self._emit_ts if self._emit_ts > 0.0 else None

    def on_sample(
        self,
        entity_id: str,
        value_w: float | None,
        ts: float,
        *,
        reported: bool = False,
    ) -> bool:
        """Record an entity update. Returns True if all gross sources settled."""
        if entity_id not in self.entity_roles:
            return False

        if self._init_ts is None:
            self._init_ts = ts

        state = self._entities[entity_id]
        state.value_w = value_w
        state.last_ts = ts
        state.fresh = True
        state.available = value_w is not None
        if reported:
            self.detectors[entity_id].observe_report(ts)
        state.stale = False
        state.on_change = False

        if not self.entity_roles:
            return False

        # If any gross entity becomes unavailable/None, trigger immediate publication
        if not state.available:
            self._emit_ts = ts
            return True


        gating = [
            e
            for e, s in self._entities.items()
            if not s.on_change
        ]
        if not gating:
            self._emit_ts = ts
            self._reset_after_emission()
            return True

        if all(self._entities[e].fresh for e in gating):
            if not self._skipped:
                behind = self._behind_sources(entity_id, gating, ts)
                if behind:
                    # A1 closer-sample wait: another source's sample belongs to an earlier
                    # poll cycle; wait for its next sample instead of pairing across cycles.
                    for e in behind:
                        self._entities[e].fresh = False
                    self._skipped = True
                    return False
            valid_ts = [self._entities[e].last_ts for e in gating if self._entities[e].last_ts is not None]
            self._emit_ts = max(valid_ts) if valid_ts else ts
            self._reset_after_emission()
            return True

        return False

    def _behind_sources(
        self, entity_id: str, gating: list[str], ts: float
    ) -> list[str]:
        """Other gating sources whose last sample is older than half their profiled interval."""
        if self._profiled_interval is None:
            return []
        behind: list[str] = []
        for e in gating:
            if e == entity_id:
                continue
            last_ts = self._entities[e].last_ts
            interval = self._profiled_interval(e)
            if last_ts is None or interval is None or interval <= 0:
                continue
            if (ts - last_ts) > 0.5 * interval:
                behind.append(e)
        return behind

    def _reset_after_emission(self) -> None:
        """Clear fresh flags and re-arm the closer-sample wait after a settle emission."""
        for state in self._entities.values():
            state.fresh = False
        self._skipped = False

    def on_tick(self, now: float) -> bool:
        """Periodic staleness and on-change evaluation tick. Returns True if settle released."""
        if self._init_ts is None:
            self._init_ts = now

        newly_stale = False

        for entity_id, state in self._entities.items():
            ref_ts = state.last_ts if state.last_ts is not None else self._init_ts
            is_re_reporting = self.detectors[entity_id].re_reporting
            stale_threshold = (
                self._stale_after(entity_id)
                if self._stale_after is not None
                else STALE_AFTER_S
            )
            # Staleness check (D-06, D-15): only re-reporting entities time out
            if is_re_reporting and (now - ref_ts) > stale_threshold:
                if not state.stale:
                    state.stale = True
                    newly_stale = True

            # On-change classification (Option b / D-19):
            # Entities not re_reporting idle for > ON_CHANGE_AFTER_S become on_change
            if not is_re_reporting and (now - ref_ts) > ON_CHANGE_AFTER_S:
                state.on_change = True

        gating = [
            e
            for e, s in self._entities.items()
            if not s.on_change
        ]

        # Check if pending settle was released
        if any(self._entities[e].fresh for e in self._entities) and all(
            self._entities[e].fresh for e in gating
        ):
            valid_ts = [self._entities[e].last_ts for e in gating if self._entities[e].last_ts is not None]
            self._emit_ts = max(valid_ts) if valid_ts else now
            self._reset_after_emission()
            return True

        if newly_stale:
            return True

        return False

    def status(self, now: float) -> AlignStatus:
        """Calculate the current alignment, staleness, and blocking status."""
        values: dict[str, float] = {}
        solar_unavailable: list[str] = []
        blocking: list[str] = []
        stale_sources: list[str] = []
        on_change_sources: list[str] = []
        source_age_s: dict[str, float] = {}

        for eid, state in self._entities.items():
            role = self.entity_roles.get(eid)
            age = round(now - state.last_ts, 1) if state.last_ts is not None else 0.0
            source_age_s[eid] = age

            if state.stale:
                stale_sources.append(eid)
                blocking.append(eid)

            if state.on_change:
                on_change_sources.append(eid)

            if state.value_w is None or not state.available:
                if role == Role.SOLAR:
                    solar_unavailable.append(eid)
                    values[eid] = 0.0
                else:
                    blocking.append(eid)
                    values[eid] = 0.0
            else:
                values[eid] = state.value_w

        return AlignStatus(
            values=values,
            blocking=tuple(sorted(set(blocking))),
            stale_sources=tuple(sorted(stale_sources)),
            solar_unavailable=tuple(sorted(solar_unavailable)),
            on_change_sources=tuple(sorted(on_change_sources)),
            source_age_s=source_age_s,
            emit_ts=self._emit_ts,
        )

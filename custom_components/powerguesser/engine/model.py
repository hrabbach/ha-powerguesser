"""Data models and enums for PowerGuesser engine."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum


class Role(StrEnum):
    """Energy role of a power channel."""

    GRID = "grid"
    SOLAR = "solar"
    BATTERY = "battery"
    HOME = "home"


class Transform(StrEnum):
    """Channel transform semantics."""

    DIRECT = "direct"
    INVERTED = "inverted"
    FROM_MINUS_TO = "from_minus_to"


class Severity(StrEnum):
    """Problem severity."""

    ERROR = "error"
    WARNING = "warning"


@dataclass(frozen=True, slots=True)
class EntityFacts:
    """Facts about an entity queried from registries and states."""

    exists: bool
    unit_ok: bool | None
    platform: str | None
    device_id: str | None


@dataclass(frozen=True, slots=True)
class RegistryFacts:
    """Collection of facts across all referenced entities."""

    entities: Mapping[str, EntityFacts] = field(default_factory=dict)
    siblings: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    # Platform of each device_consumption stat_consumption entity (None when
    # not registered), so the source model can skip PowerGuesser's own
    # device energy sensors listed in the Energy dashboard (OUT-02).
    consumption_platforms: Mapping[str, str | None] = field(default_factory=dict)

    @classmethod
    def empty(cls) -> "RegistryFacts":
        """Return an empty RegistryFacts instance."""
        return cls(entities={}, siblings={})


@dataclass(frozen=True, slots=True)
class GrossChannel:
    """A configured gross power input channel."""

    key: str
    role: Role
    transform: Transform
    entity_ids: tuple[str, ...]
    label: str


@dataclass(frozen=True, slots=True)
class MeteredDevice:
    """A metered consumer device from Energy preferences."""

    stat_consumption: str
    name: str | None = None
    included_in_stat: str | None = None
    stat_rate: str | None = None
    power_entity: str | None = None


@dataclass(frozen=True, slots=True)
class Problem:
    """A diagnostic problem identified during source ingestion."""

    issue_id: str
    kind: str
    severity: Severity
    translation_key: str
    placeholders: Mapping[str, str] = field(default_factory=dict)
    structural: bool = False

    def __post_init__(self) -> None:
        """Ensure placeholders is a plain copied dict."""
        object.__setattr__(self, "placeholders", dict(self.placeholders))


@dataclass(frozen=True, slots=True)
class SourceModel:
    """The complete digested source configuration for PowerGuesser."""

    gross_channels: tuple[GrossChannel, ...] = ()
    metered_roots: tuple[MeteredDevice, ...] = ()
    problems: tuple[Problem, ...] = ()

    @property
    def gross_entity_ids(self) -> tuple[str, ...]:
        """All unique entity IDs needed to compute gross power."""
        return tuple(dict.fromkeys(e for ch in self.gross_channels for e in ch.entity_ids))

    @property
    def metered_entity_ids(self) -> tuple[str, ...]:
        """All unique entity IDs needed to measure metered consumption."""
        return tuple(
            dict.fromkeys(
                d.power_entity
                for d in self.metered_roots
                if d.power_entity is not None
            )
        )

    @property
    def referenced_entity_ids(self) -> tuple[str, ...]:
        """All unique entity IDs referenced by any gross source or metered device."""
        refs: list[str] = list(self.gross_entity_ids)
        for d in self.metered_roots:
            if d.stat_rate:
                refs.append(d.stat_rate)
            if d.power_entity and d.power_entity != d.stat_rate:
                refs.append(d.power_entity)
        return tuple(dict.fromkeys(refs))

    @property
    def fatal(self) -> bool:
        """Return True if any non-structural ERROR problem exists."""
        return any(p.severity == Severity.ERROR and not p.structural for p in self.problems)


@dataclass(frozen=True, slots=True)
class Snapshot:
    """A calculated power snapshot for coordinator publication."""

    available: bool
    gross_w: float | None
    remainder_w: float | None
    signed_residual_w: float | None
    metered_w: float
    metered_unavailable: tuple[str, ...] = ()
    over_attribution_wh: float = 0.0
    clipped: bool = False
    ts: float | None = None
    stale_sources: tuple[str, ...] = ()
    source_age_s: Mapping[str, float] = field(default_factory=dict)
    solar_unavailable: tuple[str, ...] = ()
    on_change_sources: tuple[str, ...] = ()
    plausibility: str = "ok"
    estimated_w: float = 0.0
    device_w: Mapping[str, float] = field(default_factory=dict)



"""Energy dashboard preferences ingestion for PowerGuesser engine."""

from collections.abc import Mapping
import re
from typing import Any

from .hierarchy import power_roots, unresolved_devices
from .model import (
    EntityFacts,
    GrossChannel,
    MeteredDevice,
    Problem,
    RegistryFacts,
    Role,
    Severity,
    SourceModel,
    Transform,
)

ABORT_ORDER: tuple[str, ...] = (
    "no_energy_prefs",
    "missing_stat_rate",
    "unsupported_power_config",
    "source_not_found",
    "invalid_unit",
)

OWN_PLATFORM = "powerguesser"


def source_slug(source: Mapping[str, Any], role: Role, index: int) -> str:
    """Derive a stable issue slug from an energy source configuration."""
    stat_id: str | None = None
    if role == Role.GRID:
        stat_id = source.get("stat_energy_from") or source.get("stat_energy_to")
    elif role == Role.SOLAR:
        stat_id = source.get("stat_energy_from")
    elif role == Role.BATTERY:
        stat_id = source.get("stat_energy_from") or source.get("stat_energy_to")
    elif role == Role.HOME:
        stat_id = source.get("stat_energy_to") or source.get("stat_energy_from")

    if not isinstance(stat_id, str) or not stat_id:
        return f"{role.value}_{index}"

    slug = re.sub(r"[^a-z0-9]+", "_", stat_id.lower()).strip("_")
    return slug or f"{role.value}_{index}"


def _source_label(source: Mapping[str, Any], role: Role) -> str:
    """Derive a human-readable source label from an energy source configuration."""
    stat_id: str | None = None
    if role == Role.GRID:
        stat_id = source.get("stat_energy_from") or source.get("stat_energy_to")
    elif role == Role.SOLAR:
        stat_id = source.get("stat_energy_from")
    elif role == Role.BATTERY:
        stat_id = source.get("stat_energy_from") or source.get("stat_energy_to")
    elif role == Role.HOME:
        stat_id = source.get("stat_energy_to") or source.get("stat_energy_from")

    if isinstance(stat_id, str) and stat_id:
        return f"{role.value.capitalize()} ({stat_id})"
    return role.value.capitalize()


def classify_power_config(
    source: Mapping[str, Any],
    role: Role,
    slug: str,
    label: str,
) -> GrossChannel | Problem:
    """Classify an energy source power_config into a GrossChannel or Problem."""
    power_config = source.get("power_config")
    if isinstance(power_config, dict) and power_config:
        keys = set(power_config.keys())
        if keys == {"stat_rate"}:
            val = power_config["stat_rate"]
            if isinstance(val, str) and val:
                return GrossChannel(
                    key=f"{role.value}:{val}",
                    role=role,
                    transform=Transform.DIRECT,
                    entity_ids=(val,),
                    label=label,
                )
        elif keys == {"stat_rate_inverted"}:
            val = power_config["stat_rate_inverted"]
            if isinstance(val, str) and val:
                return GrossChannel(
                    key=f"{role.value}:{val}",
                    role=role,
                    transform=Transform.INVERTED,
                    entity_ids=(val,),
                    label=label,
                )
        elif keys == {"stat_rate_from", "stat_rate_to"}:
            val_from = power_config["stat_rate_from"]
            val_to = power_config["stat_rate_to"]
            if isinstance(val_from, str) and val_from and isinstance(val_to, str) and val_to:
                return GrossChannel(
                    key=f"{role.value}:{val_from}_{val_to}",
                    role=role,
                    transform=Transform.FROM_MINUS_TO,
                    entity_ids=(val_from, val_to),
                    label=label,
                )

        return Problem(
            issue_id=f"unsupported_power_config_{slug}",
            kind="unsupported_power_config",
            severity=Severity.ERROR,
            translation_key="unsupported_power_config",
            placeholders={"source": label},
        )

    # Absent or empty power_config -> fallback to top-level stat_rate
    stat_rate = source.get("stat_rate")
    if isinstance(stat_rate, str) and stat_rate:
        return GrossChannel(
            key=f"{role.value}:{stat_rate}",
            role=role,
            transform=Transform.DIRECT,
            entity_ids=(stat_rate,),
            label=label,
        )

    return Problem(
        issue_id=f"missing_stat_rate_{slug}",
        kind="missing_stat_rate",
        severity=Severity.ERROR,
        translation_key="missing_stat_rate",
        placeholders={"source": label},
    )


def _check_source_self_ref(
    source: Mapping[str, Any],
    facts: RegistryFacts | None,
) -> str | None:
    """Return the entity id if any power entity in source belongs to powerguesser."""
    if facts is None:
        return None
    stat_rate = source.get("stat_rate")
    if isinstance(stat_rate, str) and stat_rate:
        ef = facts.entities.get(stat_rate)
        if ef and ef.platform == OWN_PLATFORM:
            return stat_rate
    power_config = source.get("power_config")
    if isinstance(power_config, dict):
        for val in power_config.values():
            if isinstance(val, str) and val:
                ef = facts.entities.get(val)
                if ef and ef.platform == OWN_PLATFORM:
                    return val
    return None


def build_source_model(
    prefs: Mapping[str, Any] | None,
    facts: RegistryFacts | None = None,
) -> SourceModel:
    """Build a SourceModel from raw Energy dashboard preferences and optional registry facts."""
    if not isinstance(prefs, Mapping):
        return SourceModel(
            problems=(
                Problem(
                    issue_id="no_energy_prefs",
                    kind="no_energy_prefs",
                    severity=Severity.ERROR,
                    translation_key="no_energy_prefs",
                ),
            )
        )

    gross_channels: list[GrossChannel] = []
    problems: list[Problem] = []

    energy_sources = prefs.get("energy_sources")
    if not isinstance(energy_sources, list):
        energy_sources = []

    # Check for whole-home source first
    home_src: dict[str, Any] | None = None
    home_idx: int = 0
    for idx, item in enumerate(energy_sources):
        if isinstance(item, dict) and item.get("type") in (Role.HOME, "home"):
            has_rate = bool(
                item.get("stat_rate")
                or (isinstance(item.get("power_config"), dict) and item.get("power_config"))
            )
            if has_rate:
                home_src = item
                home_idx = idx
                break

    if home_src is not None:
        self_ref_eid = _check_source_self_ref(home_src, facts)
        if self_ref_eid:
            slug = source_slug(home_src, Role.HOME, home_idx)
            label = _source_label(home_src, Role.HOME)
            problems.append(
                Problem(
                    issue_id=f"self_reference_{slug}",
                    kind="self_reference",
                    severity=Severity.WARNING,
                    translation_key="self_reference",
                    placeholders={"entity": self_ref_eid, "source": label},
                )
            )
            home_src = None  # drop home source, formula path applies

    if home_src is not None:
        slug = source_slug(home_src, Role.HOME, home_idx)
        label = _source_label(home_src, Role.HOME)
        classified = classify_power_config(home_src, Role.HOME, slug, label)
        if isinstance(classified, GrossChannel):
            gross_channels.append(classified)
        else:
            problems.append(classified)
    else:
        # Standard formula path (grid, solar, battery)
        for idx, item in enumerate(energy_sources):
            if not isinstance(item, dict):
                continue
            src_type = item.get("type")
            if src_type not in (Role.GRID, Role.SOLAR, Role.BATTERY, "grid", "solar", "battery"):
                continue
            role = Role(src_type)
            slug = source_slug(item, role, idx)
            label = _source_label(item, role)

            self_ref_eid = _check_source_self_ref(item, facts)
            if self_ref_eid:
                problems.append(
                    Problem(
                        issue_id=f"self_reference_{slug}",
                        kind="self_reference",
                        severity=Severity.ERROR,
                        translation_key="self_reference",
                        placeholders={"entity": self_ref_eid, "source": label},
                    )
                )
                continue

            classified = classify_power_config(item, role, slug, label)
            if isinstance(classified, GrossChannel):
                gross_channels.append(classified)
            else:
                problems.append(classified)

    # Empty gross sources check
    if not gross_channels and not problems:
        problems.append(
            Problem(
                issue_id="no_energy_prefs",
                kind="no_energy_prefs",
                severity=Severity.ERROR,
                translation_key="no_energy_prefs",
            )
        )

    # Facts validation on configured gross channels
    if facts is not None:
        for channel in gross_channels:
            for entity_id in channel.entity_ids:
                ef = facts.entities.get(entity_id)
                e_slug = re.sub(r"[^a-z0-9]+", "_", entity_id.lower()).strip("_")
                if ef is None or not ef.exists:
                    problems.append(
                        Problem(
                            issue_id=f"source_not_found_{e_slug}",
                            kind="source_not_found",
                            severity=Severity.ERROR,
                            translation_key="source_not_found",
                            placeholders={"source": entity_id},
                            structural=True,
                        )
                    )
                elif ef.unit_ok is False:
                    problems.append(
                        Problem(
                            issue_id=f"invalid_unit_{e_slug}",
                            kind="invalid_unit",
                            severity=Severity.ERROR,
                            translation_key="invalid_unit",
                            placeholders={"source": entity_id},
                            structural=False,
                        )
                    )

    # Device consumption (metered devices hierarchy)
    devices: list[MeteredDevice] = []
    device_consumption = prefs.get("device_consumption")
    if isinstance(device_consumption, list):
        for item in device_consumption:
            if not isinstance(item, dict):
                continue
            stat_consumption = str(item.get("stat_consumption") or "")
            name = item.get("name")
            included_in_stat = item.get("included_in_stat")
            stat_rate = item.get("stat_rate")
            power_entity: str | None = None

            # PowerGuesser's own device energy sensors listed in
            # device_consumption are read-only dashboard entries: skip them
            # silently (no metered_no_power, no self_reference) so the
            # device estimate is never subtracted twice (OUT-02, Pitfall 2).
            # They are neither MeteredDevice roots nor hierarchy parents.
            if (
                facts is not None
                and facts.consumption_platforms.get(stat_consumption) == OWN_PLATFORM
            ):
                continue

            if isinstance(stat_rate, str) and stat_rate:
                if facts is not None:
                    ef = facts.entities.get(stat_rate)
                    if ef and ef.platform == OWN_PLATFORM:
                        dev_slug = re.sub(r"[^a-z0-9]+", "_", stat_consumption.lower()).strip("_")
                        problems.append(
                            Problem(
                                issue_id=f"self_reference_{dev_slug}",
                                kind="self_reference",
                                severity=Severity.WARNING,
                                translation_key="self_reference",
                                placeholders={"entity": stat_rate, "source": name or stat_consumption},
                            )
                        )
                        stat_rate = None
                if stat_rate:
                    power_entity = stat_rate

            if power_entity is None and facts is not None:
                siblings = facts.siblings.get(stat_consumption)
                if siblings and len(siblings) == 1:
                    power_entity = siblings[0]

            devices.append(
                MeteredDevice(
                    stat_consumption=stat_consumption,
                    name=name,
                    included_in_stat=included_in_stat,
                    stat_rate=stat_rate,
                    power_entity=power_entity,
                )
            )

    metered_roots = power_roots(devices)
    unresolved = unresolved_devices(devices)
    for d in unresolved:
        dev_slug = re.sub(r"[^a-z0-9]+", "_", d.stat_consumption.lower()).strip("_")
        problems.append(
            Problem(
                issue_id=f"metered_no_power_{dev_slug}",
                kind="metered_no_power",
                severity=Severity.WARNING,
                translation_key="metered_no_power",
                placeholders={"device": d.name or d.stat_consumption},
            )
        )

    return SourceModel(
        gross_channels=tuple(gross_channels),
        metered_roots=tuple(metered_roots),
        problems=tuple(problems),
    )

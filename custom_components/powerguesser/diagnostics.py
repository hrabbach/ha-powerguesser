"""Diagnostics platform for PowerGuesser."""

from typing import Any

from homeassistant.components.diagnostics import async_redact_data
from homeassistant.core import HomeAssistant

from . import PowerGuesserConfigEntry
from .const import CONF_LLM_API_KEY
from .engine.intervals import MeasuredFrom, Mode


async def async_get_config_entry_diagnostics(
    hass: HomeAssistant,
    entry: PowerGuesserConfigEntry,
) -> dict[str, Any]:
    """Return diagnostics for a config entry (aggregates only, no raw samples)."""
    hub = entry.runtime_data

    # Buffer statistics from SQLite
    buffer_stats = await hass.async_add_executor_job(hub.writer.db.stats)

    # Interval profiles for current tracked sources
    profiles: dict[str, Any] = {}
    for entity_id, source in hub.profile_sources.items():
        profiler = hub.profilers.get(entity_id)
        if profiler is None:
            continue

        measured_from: str | None = None
        if profiler.median_s is not None:
            measured_from = (
                MeasuredFrom.REPORTS.value
                if profiler.mode == Mode.RE_REPORTING
                else MeasuredFrom.CHANGES.value
            )

        profiles[entity_id] = {
            "role": source.role,
            "mode": profiler.mode.value,
            "measured_from": measured_from,
            "restored": profiler.restored,
            "median_s": profiler.median_s,
            "p90_s": profiler.p90_s,
            "delta_count": profiler.delta_count,
            "histogram": profiler.histogram(),
        }

    # Effective gating snapshot
    prof_data = hub.profile_coordinator.data
    if prof_data is not None:
        effective = {
            "interval_s": prof_data.effective_s,
            "assumed": prof_data.assumed,
            "enabled": [e.value for e in prof_data.enabled],
            "disabled": {
                e.value: reason for e, reason in prof_data.disabled.items()
            },
        }
    else:
        effective = {
            "interval_s": 30.0,
            "assumed": True,
            "enabled": ["step", "cycle_shape", "continuous"],
            "disabled": {},
        }

    # Bootstrap status and report
    bootstrap = dict(hub.bootstrap_coordinator.data or {})

    # Entry options
    options = dict(entry.options)

    # Entry data (redacted)
    data = async_redact_data(dict(entry.data), {CONF_LLM_API_KEY})

    return {
        "buffer": buffer_stats,
        "profiles": profiles,
        "effective": effective,
        "bootstrap": bootstrap,
        "options": options,
        "data": data,
    }

"""Diagnostics platform for PowerGuesser."""

from typing import Any

from homeassistant.components.diagnostics import async_redact_data
from homeassistant.core import HomeAssistant

from . import PowerGuesserConfigEntry
from .const import CONF_LLM_API_KEY, CONF_LLM_BASE_URL, CONF_LLM_PROBE
from .engine.intervals import MeasuredFrom, Mode

# CLEAN-01 (D-16): the explicit allow-list copied out of entry.data["llm_probe"].
# The probe dict is built by llm_client.probe with exactly these keys (no secrets);
# copying by allow-list means a future extra key never reaches a shared dump.
LLM_PROBE_DIAG_KEYS = (
    "strategy",
    "token_param",
    "send_temperature",
    "model_returned",
    "model_mismatch",
    "probed_at",
)


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

    # Entry data (redacted: the key and the whole base URL, which may carry
    # userinfo or a key in the query, CLEAN-01 / T-07-04)
    data = async_redact_data(dict(entry.data), {CONF_LLM_API_KEY, CONF_LLM_BASE_URL})

    # The probe result, copied once from a snapshot of entry.data with no await in
    # between, so a probe re-run racing this download yields the old or the new
    # six-field record, never a mix (CLEAN-01). A missing probe (or a non-dict)
    # reads None; model_mismatch is copied as stored, never recomputed.
    probe = entry.data.get(CONF_LLM_PROBE)
    llm_probe = (
        {k: probe.get(k) for k in LLM_PROBE_DIAG_KEYS}
        if isinstance(probe, dict)
        else None
    )
    # WR-01: async_redact_data only replaces values by key NAME, so the raw probe
    # (every stored key, including a future secret-bearing one) would still ship
    # inside ``data``. Replace it there with the same allow-listed copy, so the
    # allow-list guarantee holds for the whole dump, not just the ``llm_probe`` key.
    if CONF_LLM_PROBE in data:
        data[CONF_LLM_PROBE] = llm_probe

    return {
        "buffer": buffer_stats,
        "profiles": profiles,
        "effective": effective,
        "bootstrap": bootstrap,
        "options": options,
        "data": data,
        "llm_probe": llm_probe,
    }

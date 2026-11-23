"""Recorder bootstrap runner and history queries for PowerGuesser."""

import json
import logging
from typing import TYPE_CHECKING, Any

from homeassistant.components.recorder import get_instance, is_entity_recorded
from homeassistant.components.recorder.history import state_changes_during_period
from homeassistant.const import ATTR_UNIT_OF_MEASUREMENT, UnitOfPower
from homeassistant.core import HomeAssistant, callback
from homeassistant.util import dt as dt_util
from homeassistant.util.unit_conversion import PowerConverter

from .engine.bootstrap import rows_from_history
from .engine.buffer import (
    KIND_GROSS,
    KIND_METERED,
    KIND_PHASE,
    RETENTION_DAYS,
    BufferDB,
)
from .engine.intervals import FALLBACK_INTERVAL_S

if TYPE_CHECKING:
    from .hub import PowerGuesserHub

_LOGGER = logging.getLogger(__name__)

BOOTSTRAP_DAY_MS = 86_400_000
DONE_IMPORTED = "1"
DONE_INVALID_UNIT = "invalid_unit"


def _query(
    hass: HomeAssistant,
    entity_id: str,
    start_s: float,
    end_s: float,
) -> list[tuple[str, float, float]]:
    """Query state changes during period from the recorder on its executor."""
    start_dt = dt_util.utc_from_timestamp(start_s)
    end_dt = dt_util.utc_from_timestamp(end_s)
    res = state_changes_during_period(
        hass,
        start_time=start_dt,
        end_time=end_dt,
        entity_id=entity_id,
        no_attributes=True,
        include_start_time_state=False,
    )
    states = res.get(entity_id, [])
    return [
        (s.state, s.last_updated_timestamp, s.last_reported_timestamp)
        for s in states
    ]


def load_stored_report(db: BufferDB) -> dict[str, Any]:
    """Read the stored bootstrap report from buffer meta; run in the executor."""
    stored_report_json = db.get_meta("bootstrap_report")
    if stored_report_json:
        try:
            parsed = json.loads(stored_report_json)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
    return {}


class BootstrapRunner:
    """Runner for importing historical recorder data into PowerGuesser buffer."""

    def __init__(self, hass: HomeAssistant, hub: "PowerGuesserHub") -> None:
        """Initialize the runner."""
        self.hass = hass
        self.hub = hub

    @callback
    def _publish(self, **fields: Any) -> None:
        """Publish updated status fields to the coordinator."""
        current = dict(self.hub.bootstrap_coordinator.data or {})
        current.update(fields)
        self.hub.bootstrap_coordinator.async_set_updated_data(current)

    async def async_run(self) -> None:
        """Run the bootstrap import process."""
        # Loaded by the awaited hub setup, so waiting and the stored report publish together
        stored_report = dict(self.hub._stored_bootstrap_report)
        self._publish(status="waiting", **stored_report)

        await self.hub._bootstrap_gate.wait()
        self._publish(status="running")
        recorder_instance = get_instance(self.hass)
        ready = await recorder_instance.async_db_ready
        if not ready:
            self._publish(status="recorder_unavailable")
            return

        await self.hub.writer.async_flush()

        channels_imported_this_run = False

        while True:
            all_channels = await self.hass.async_add_executor_job(
                self.hub.writer.db.channel_rows
            )
            active_rows = [
                row for row in all_channels
                if row[0] in self.hub._channel_entities
            ]
            kind_order = {KIND_GROSS: 0, KIND_METERED: 1, KIND_PHASE: 2}
            active_rows.sort(key=lambda r: (kind_order.get(r[3], 99), r[0]))

            pending: list[tuple[int, str, str, str]] = []
            for ch, entity_id, role, kind in active_rows:
                done_val = await self.hass.async_add_executor_job(
                    self.hub.writer.db.get_meta, f"bootstrap_done:{ch}"
                )
                if not done_val:
                    pending.append((ch, entity_id, role, kind))

            if not pending:
                break

            processed_any = False
            for ch, entity_id, role, kind in pending:
                live_start_str = await self.hass.async_add_executor_job(
                    self.hub.writer.db.get_meta, f"live_start_ms:{ch}"
                )
                if live_start_str is None:
                    continue

                end_ms = int(live_start_str)

                retention_start_ms = end_ms - RETENTION_DAYS * BOOTSTRAP_DAY_MS
                oldest_ts = getattr(
                    recorder_instance.states_manager, "oldest_ts", None
                )
                if oldest_ts is not None:
                    start_ms = max(
                        retention_start_ms, int(round(oldest_ts * 1000))
                    )
                else:
                    start_ms = retention_start_ms

                state_obj = self.hass.states.get(entity_id)
                if state_obj is None:
                    continue

                unit = state_obj.attributes.get(ATTR_UNIT_OF_MEASUREMENT)
                if unit not in PowerConverter.VALID_UNITS:
                    _LOGGER.warning(
                        "Channel %s (%s) has unconvertible unit %s; skipping bootstrap",
                        ch,
                        entity_id,
                        unit,
                    )
                    await self.hass.async_add_executor_job(
                        self.hub.writer.db.set_meta,
                        f"bootstrap_done:{ch}",
                        DONE_INVALID_UNIT,
                    )
                    processed_any = True
                    channels_imported_this_run = True
                    continue

                try:
                    unit_factor = float(
                        PowerConverter.convert(1.0, unit, UnitOfPower.WATT)
                    )
                except Exception:
                    _LOGGER.warning(
                        "Failed to convert unit %s for %s", unit, entity_id
                    )
                    await self.hass.async_add_executor_job(
                        self.hub.writer.db.set_meta,
                        f"bootstrap_done:{ch}",
                        DONE_INVALID_UNIT,
                    )
                    processed_any = True
                    channels_imported_this_run = True
                    continue

                profiler = self.hub.profilers.get(entity_id)
                mode = profiler.mode.value if profiler else "on_change"
                interval_s = (
                    profiler.median_s
                    if (profiler and profiler.median_s is not None)
                    else FALLBACK_INTERVAL_S
                )

                cursor_str = await self.hass.async_add_executor_job(
                    self.hub.writer.db.get_meta, f"bootstrap_cursor_ms:{ch}"
                )
                if cursor_str is not None:
                    stored_cursor = int(cursor_str)
                    current_start = max(start_ms, stored_cursor - BOOTSTRAP_DAY_MS)
                else:
                    current_start = start_ms

                total_imported_rows = 0
                last_tuple: tuple[str, float, float] | None = None

                while current_start < end_ms:
                    chunk_end = min(current_start + BOOTSTRAP_DAY_MS, end_ms)
                    is_final_chunk = chunk_end == end_ms

                    q_start_s = (current_start - 1) / 1000.0
                    q_end_s = chunk_end / 1000.0

                    queried_states = await recorder_instance.async_add_executor_job(
                        _query, self.hass, entity_id, q_start_s, q_end_s
                    )

                    states: list[tuple[str, float, float]] = []
                    if last_tuple is not None:
                        states.append(last_tuple)
                    states.extend(queried_states)

                    if states:
                        last_tuple = states[-1]

                    live_lr_s = None
                    if is_final_chunk and states and state_obj is not None:
                        last_tuple_lu = states[-1][1]
                        if (
                            abs(
                                state_obj.last_updated_timestamp
                                - last_tuple_lu
                            )
                            < 1e-4
                        ):
                            live_lr_s = state_obj.last_reported_timestamp

                    rows = rows_from_history(
                        states,
                        ch=ch,
                        unit_factor=unit_factor,
                        mode=mode,
                        interval_s=interval_s,
                        live_lr_s=live_lr_s,
                        end_s=(chunk_end / 1000.0) if is_final_chunk else None,
                        gap_markers=kind != KIND_METERED,
                    )

                    rows = [r for r in rows if r[0] <= end_ms]

                    await self.hass.async_add_executor_job(
                        self.hub.writer.db.insert_bootstrap,
                        rows,
                        ch,
                        chunk_end,
                    )
                    total_imported_rows += len(rows)
                    current_start = chunk_end

                await self.hass.async_add_executor_job(
                    self.hub.writer.db.set_meta,
                    f"bootstrap_done:{ch}",
                    DONE_IMPORTED,
                )
                _LOGGER.info(
                    "Bootstrapped channel %s (%s): %d rows",
                    ch,
                    entity_id,
                    total_imported_rows,
                )
                processed_any = True
                channels_imported_this_run = True

            if not processed_any:
                break

        if channels_imported_this_run or not stored_report:
            report = await self._async_compute_and_store_report(recorder_instance)
            self._publish(status="done", **report)
        else:
            self._publish(status="done", **stored_report)

    async def _async_compute_and_store_report(
        self, recorder_instance: Any
    ) -> dict[str, Any]:
        """Compute coverage report across active channels and save to meta."""
        oldest_ts = getattr(recorder_instance.states_manager, "oldest_ts", None)
        all_channels = await self.hass.async_add_executor_job(
            self.hub.writer.db.channel_rows
        )
        active_rows = [
            row for row in all_channels
            if row[0] in self.hub._channel_entities
        ]

        windows: dict[int, tuple[int, int]] = {}
        done_statuses: dict[int, str | None] = {}
        for ch, entity_id, role, kind in active_rows:
            done_val = await self.hass.async_add_executor_job(
                self.hub.writer.db.get_meta, f"bootstrap_done:{ch}"
            )
            done_statuses[ch] = done_val
            live_start_str = await self.hass.async_add_executor_job(
                self.hub.writer.db.get_meta, f"live_start_ms:{ch}"
            )
            if live_start_str is not None:
                end_ms = int(live_start_str)
                retention_start_ms = end_ms - RETENTION_DAYS * BOOTSTRAP_DAY_MS
                if oldest_ts is not None:
                    start_ms = max(
                        retention_start_ms, int(round(oldest_ts * 1000))
                    )
                else:
                    start_ms = retention_start_ms
                windows[ch] = (start_ms, end_ms)

        def _get_coverage() -> tuple[float, dict[int, Any]]:
            gross_layout_raw = self.hub.writer.db.get_meta("gross_layout")
            gross_roles: dict[int, str] = {}
            if gross_layout_raw:
                try:
                    layout_data = json.loads(gross_layout_raw)
                    for item in layout_data:
                        r = item["role"]
                        for c in item["channels"]:
                            gross_roles[c] = r
                except Exception:
                    pass
            return self.hub.writer.db.coverage_for(windows, gross_roles)

        usable_days_val, coverages = await self.hass.async_add_executor_job(
            _get_coverage
        )

        report_channels: dict[str, Any] = {}
        for ch, entity_id, role, kind in active_rows:
            is_recorded = is_entity_recorded(self.hass, entity_id)
            done_val = done_statuses.get(ch)
            if done_val == DONE_INVALID_UNIT:
                report_channels[entity_id] = {
                    "role": role,
                    "days": 0.0,
                    "first": None,
                    "last": None,
                    "gap_count": 0,
                    "excluded_by_recorder": not is_recorded,
                    "skipped": DONE_INVALID_UNIT,
                }
            else:
                cov = coverages.get(ch)
                days = round(cov.days, 2) if cov else 0.0
                first_iso = (
                    dt_util.utc_from_timestamp(cov.first_ms / 1000.0).isoformat()
                    if (cov and cov.first_ms is not None)
                    else None
                )
                last_iso = (
                    dt_util.utc_from_timestamp(cov.last_ms / 1000.0).isoformat()
                    if (cov and cov.last_ms is not None)
                    else None
                )
                gap_count = cov.gap_count if cov else 0
                report_channels[entity_id] = {
                    "role": role,
                    "days": days,
                    "first": first_iso,
                    "last": last_iso,
                    "gap_count": gap_count,
                    "excluded_by_recorder": not is_recorded,
                    "skipped": None,
                }

        report = {
            "usable_days": round(usable_days_val, 2),
            "computed_at": dt_util.utcnow().isoformat(),
            "channels": report_channels,
        }

        await self.hass.async_add_executor_job(
            self.hub.writer.db.set_meta, "bootstrap_report", json.dumps(report)
        )
        self.hub._stored_bootstrap_report = report
        _LOGGER.info(
            "Bootstrap report complete: usable_days=%.2f, channels=%d",
            report["usable_days"],
            len(report_channels),
        )
        _LOGGER.debug(
            "Bootstrap report channels: %s", list(report_channels.keys())
        )
        return report

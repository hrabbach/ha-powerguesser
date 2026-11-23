"""BufferWriter adapter managing in-memory batches and executor flushes."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import asyncio
import logging
import sqlite3
from typing import Any

from homeassistant.const import EVENT_HOMEASSISTANT_STOP
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.event import async_call_later, async_track_time_interval
from homeassistant.util import dt as dt_util

from .const import FLUSH_INTERVAL, RETENTION_FIRST_RUN_S, RETENTION_INTERVAL
from .engine.buffer import RETENTION_DAYS, BufferDB
from .engine.settle import STALE_AFTER_S

_LOGGER = logging.getLogger(__name__)

MAX_PENDING_ROWS = 20_000


@dataclass(slots=True)
class ChannelState:
    """Tracking state for a single buffered channel."""

    last_ts_ms: int | None = None
    last_null: bool = False
    stale_marked: bool = False


class BufferWriter:
    """Manages in-memory sample batching and periodic SQLite executor flushes."""

    def __init__(self, hass: HomeAssistant, db: BufferDB) -> None:
        """Initialize the buffer writer."""
        self.hass = hass
        self.db = db
        self._pending: list[tuple[int, int, int | None]] = []
        self._channel_states: dict[int, ChannelState] = {}
        self._active_channels: set[int] | None = None
        self._lock = asyncio.Lock()
        self._unsub_flush: Callable[[], None] | None = None
        self._unsub_stop: Callable[[], None] | None = None
        self._unsub_retention_first: Callable[[], None] | None = None
        self._unsub_retention: Callable[[], None] | None = None
        self._flush_failing: bool = False
        self._closed: bool = False

        self.stale_after_ms: Callable[[int], int] = lambda _ch: round(STALE_AFTER_S * 1000)
        self.re_reporting: Callable[[int], bool] = lambda _ch: False
        # Metered plug channels: held with no age limit (P1 D-07), never stale-marked
        self.held_channels: frozenset[int] = frozenset()

    @callback
    def set_active(self, channel_ids: Sequence[int] | set[int], now_ms: int) -> set[int]:
        """Update active channel set, writing activation and removal NULL markers."""
        new_set = set(channel_ids)
        prev_set = self._active_channels if self._active_channels is not None else set()
        newly_active = new_set - prev_set
        removed = prev_set - new_set

        for ch in sorted(newly_active):
            self.mark_gap(ch, now_ms)
        for ch in sorted(removed):
            self.mark_gap(ch, now_ms)

        self._active_channels = new_set
        return newly_active

    @callback
    def record(self, ch: int, w: float | None, ts_s: float) -> None:
        """Record a power sample in memory for later batch flushing."""
        # The hub clears its trackers at the start of async_rebuild, before set_active,
        # so only an already-scheduled callback can race the removal marker and dropping
        # it keeps that marker the channel's last row (D-03).
        if self._active_channels is not None and ch not in self._active_channels:
            return

        ts_ms = round(ts_s * 1000)
        w_val = None if w is None else round(w)
        self._pending.append((ts_ms, ch, w_val))
        state = self._channel_states.setdefault(ch, ChannelState())
        state.last_ts_ms = ts_ms
        state.last_null = (w_val is None)
        state.stale_marked = False

    @callback
    def mark_gap(self, ch: int, ts_ms: int) -> None:
        """Record a gap (NULL sample) at a specific timestamp in milliseconds."""
        self._pending.append((ts_ms, ch, None))
        state = self._channel_states.setdefault(ch, ChannelState())
        state.last_ts_ms = ts_ms
        state.last_null = True

    @callback
    def mark_stale(self, now_ms: int) -> None:
        """Record a NULL stale marker for any silent re-reporting non-metered channel."""
        if self._active_channels is None:
            return
        for ch in sorted(self._active_channels):
            if ch in self.held_channels or not self.re_reporting(ch):
                continue
            state = self._channel_states.get(ch)
            if state is None or state.last_ts_ms is None or state.last_null or state.stale_marked:
                continue
            threshold = self.stale_after_ms(ch)
            if now_ms - state.last_ts_ms > threshold:
                marker_ts = state.last_ts_ms + threshold
                self._pending.append((marker_ts, ch, None))
                state.last_ts_ms = marker_ts
                state.last_null = True
                state.stale_marked = True

    async def async_start(self) -> None:
        """Start periodic flush timer, retention timers, and stop listener."""
        self._unsub_flush = async_track_time_interval(
            self.hass,
            self._async_on_flush_timer,
            FLUSH_INTERVAL,
            name="powerguesser buffer flush",
        )
        self._unsub_retention_first = async_call_later(
            self.hass,
            RETENTION_FIRST_RUN_S,
            self._async_on_retention_timer,
        )
        self._unsub_retention = async_track_time_interval(
            self.hass,
            self._async_on_retention_timer,
            RETENTION_INTERVAL,
            name="powerguesser buffer retention",
        )
        self._unsub_stop = self.hass.bus.async_listen_once(
            EVENT_HOMEASSISTANT_STOP,
            self._async_on_stop,
        )

    async def _async_on_flush_timer(self, _now: Any) -> None:
        """Handle periodic flush timer tick."""
        await self.async_flush()

    async def _async_on_retention_timer(self, _now: Any) -> None:
        """Handle periodic retention cleanup."""
        now_ms = round(dt_util.utcnow().timestamp() * 1000)
        cutoff_ms = now_ms - RETENTION_DAYS * 86_400_000
        try:
            deleted = await self.hass.async_add_executor_job(
                self.db.delete_older_than,
                cutoff_ms,
            )
            _LOGGER.debug("Deleted %d buffer rows older than %s", deleted, cutoff_ms)
        except sqlite3.Error as err:
            _LOGGER.warning("Buffer retention cleanup failed: %s", err)

    async def _async_on_stop(self, _event: Any) -> None:
        """Handle Home Assistant stop event."""
        self._unsub_stop = None
        await self.async_close()

    async def async_flush(self) -> None:
        """Flush pending buffered samples to SQLite in executor."""
        async with self._lock:
            if not self._pending:
                return
            rows = self._pending
            self._pending = []
            try:
                await self.hass.async_add_executor_job(self.db.write_batch, rows)
            except sqlite3.Error as err:
                self._pending = rows + self._pending
                if not self._flush_failing:
                    self._flush_failing = True
                    _LOGGER.warning("Buffer write failed: %s", err)
                if len(self._pending) > MAX_PENDING_ROWS:
                    overflow = len(self._pending) - MAX_PENDING_ROWS
                    del self._pending[:overflow]
                    _LOGGER.warning(
                        "Dropped %d oldest pending buffer rows due to overflow",
                        overflow,
                    )
                return

            if self._flush_failing:
                self._flush_failing = False
                _LOGGER.info("Buffer write recovered successfully")

    async def async_close(self) -> None:
        """Flush remaining samples and close database cleanly."""
        if self._closed:
            return
        self._closed = True
        if self._unsub_flush is not None:
            self._unsub_flush()
            self._unsub_flush = None
        if self._unsub_retention_first is not None:
            self._unsub_retention_first()
            self._unsub_retention_first = None
        if self._unsub_retention is not None:
            self._unsub_retention()
            self._unsub_retention = None
        if self._active_channels:
            now_ms = round(dt_util.utcnow().timestamp() * 1000)
            for ch in sorted(self._active_channels):
                self.mark_gap(ch, now_ms)
        await self.async_flush()
        await self.hass.async_add_executor_job(self.db.close_clean)
        if self._unsub_stop is not None:
            self._unsub_stop()
            self._unsub_stop = None

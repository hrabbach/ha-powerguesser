"""HA-free capture window rules: local-time resolution, coverage and gap refusals.

A retroactive window is validated before any snap runs (CAP-01 success criterion
1: a window outside the buffer, across a sample gap or too short for the meter's
grid is refused with its stated reason, never snapped to some nearby run). This
module is HA-free: stdlib ``zoneinfo`` for DST handling (RESEARCH Pitfall 1) and
``sqlite3`` through :mod:`..reader` for the buffer bounds and gap scan.
"""

from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

from ..learning.episodes import MAX_PAIR_S
from ..reader import load_gross_layout, open_readonly, read_range
from .config import DEFAULT_CONFIG, CaptureConfig

# Every refusal code a retroactive window can raise (SC1). The admin service maps
# each to translation key ``capture_<code>``.
REFUSAL_CODES: tuple[str, ...] = (
    "invalid_window",
    "future_end",
    "window_too_long",
    "too_short_for_rate",
    "interval_too_coarse",
    "outside_buffer",
    "across_gap",
    "ambiguous_local_time",
    "nonexistent_local_time",
)


class CaptureRefused(Exception):
    """A retroactive window was refused, with a stated reason.

    ``code`` is one of :data:`REFUSAL_CODES`; ``details`` are the translation
    placeholders for the ``capture_<code>`` message.
    """

    def __init__(self, code: str, details: dict[str, str] | None = None) -> None:
        """Store the refusal code and its translation placeholders."""
        self.code = code
        self.details = details or {}
        super().__init__(code)


def _iso(ms: int) -> str:
    """Format a millisecond epoch as an ISO-8601 UTC string."""
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).isoformat()


def _as_datetime(value: Any) -> datetime:
    """Return an aware or naive datetime from a datetime or ISO string."""
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value))


def local_to_utc_ms(value: Any, tz_name: str) -> int:
    """Resolve a naive local time in ``tz_name`` to a UTC millisecond epoch.

    An aware datetime (or string) is used as given. A naive one is resolved at
    ``fold=0`` and ``fold=1``: two different UTC instants mean the wall time is in
    the repeated (ambiguous) DST hour; a wall time whose fold-0 round trip does
    not reproduce it is in the skipped (nonexistent) hour. Both raise
    :class:`CaptureRefused`.
    """
    parsed = _as_datetime(value)
    if parsed.tzinfo is not None:
        return int(round(parsed.timestamp() * 1000))

    tz = ZoneInfo(tz_name)
    u0 = parsed.replace(tzinfo=tz, fold=0).timestamp()
    u1 = parsed.replace(tzinfo=tz, fold=1).timestamp()
    if u0 == u1:
        return int(round(u0 * 1000))

    wall = parsed.replace(tzinfo=None)
    ok0 = datetime.fromtimestamp(u0, tz=tz).replace(tzinfo=None) == wall
    ok1 = datetime.fromtimestamp(u1, tz=tz).replace(tzinfo=None) == wall
    if ok0 or ok1:
        raise CaptureRefused("ambiguous_local_time", {"value": wall.isoformat()})
    raise CaptureRefused("nonexistent_local_time", {"value": wall.isoformat()})


def _has_step(enabled_event_types: Sequence[Any]) -> bool:
    """Return True when STEP is among the enabled event types (enum or string)."""
    for event_type in enabled_event_types:
        value = getattr(event_type, "value", event_type)
        if value == "step":
            return True
    return False


def resolve_window(
    start: Any,
    end: Any,
    *,
    tz_name: str,
    now_ms: int,
    grid_s: int,
    enabled_event_types: Sequence[Any],
    buffer_path: str,
    config: CaptureConfig = DEFAULT_CONFIG,
) -> tuple[int, int]:
    """Validate a retroactive window and return its UTC millisecond bounds.

    Checks, in order: ``invalid_window`` (end <= start), ``future_end`` (end lies
    after ``now_ms``), ``window_too_long`` (longer than ``MAX_PAIR_S``),
    ``too_short_for_rate`` (shorter than ``config.min_run_samples`` grid steps),
    ``interval_too_coarse`` (STEP disabled at this interval), then — opening its
    own read-only buffer connection, closed before returning — ``outside_buffer``
    (the window starts before the first or ends after the last buffered gross row
    plus one grid step) and ``across_gap`` (a gross sample gap longer than
    ``max(gap_max_floor_s, gap_max_samples x grid_s)`` or a NULL gross row inside
    the window plus the snap margin).

    Solar is exempt from the gap/NULL scan: the residual reads a NULL solar row as
    0 W whatever its cause (engine/reader.py, P1 D-20) and engine/coverage.py never
    intersects solar for the same reason, so a night-time or unavailable solar
    NULL does not make the window unusable, while a NULL grid or battery row does.
    """
    start_ms = local_to_utc_ms(start, tz_name)
    end_ms = local_to_utc_ms(end, tz_name)
    grid_ms = int(grid_s) * 1000

    if end_ms <= start_ms:
        raise CaptureRefused("invalid_window", {"reason": "end must be after start"})
    if end_ms > now_ms:
        raise CaptureRefused("future_end", {"end": _iso(end_ms)})
    if end_ms - start_ms > MAX_PAIR_S * 1000:
        raise CaptureRefused("window_too_long", {"max_hours": str(MAX_PAIR_S // 3600)})
    if end_ms - start_ms < config.min_run_samples * grid_ms:
        min_seconds = config.min_run_samples * int(grid_s)
        raise CaptureRefused("too_short_for_rate", {"min_seconds": str(min_seconds)})
    if not _has_step(enabled_event_types):
        raise CaptureRefused(
            "interval_too_coarse",
            {"reason": "the meter's sample interval is too coarse for step detection"},
        )

    margin_ms = config.snap_margin_s * 1000
    conn = open_readonly(buffer_path)
    try:
        layout = load_gross_layout(conn)
        roles: dict[int, str] = {}
        for role, _transform, channels in layout:
            for ch in channels:
                roles[ch] = role
        if not roles:
            raise CaptureRefused("outside_buffer", {"buffer_start": "", "buffer_end": ""})

        gross_channels = list(roles)
        all_solar = all(roles[ch] == "solar" for ch in gross_channels)
        chosen = (
            gross_channels
            if all_solar
            else [ch for ch in gross_channels if roles[ch] != "solar"]
        )
        gap_channels = [ch for ch in chosen if roles[ch] != "solar"]

        first_ms: int | None = None
        last_ms: int | None = None
        for ch in chosen:
            row = conn.execute(
                "SELECT MIN(ts), MAX(ts) FROM samples WHERE ch = ?", (ch,)
            ).fetchone()
            if row and row[0] is not None:
                first_ms = row[0] if first_ms is None else min(first_ms, row[0])
                last_ms = row[1] if last_ms is None else max(last_ms, row[1])
        if first_ms is None or last_ms is None:
            raise CaptureRefused("outside_buffer", {"buffer_start": "", "buffer_end": ""})
        if start_ms < first_ms or end_ms > last_ms + grid_ms:
            raise CaptureRefused(
                "outside_buffer",
                {"buffer_start": _iso(first_ms), "buffer_end": _iso(last_ms)},
            )

        win_lo = max(first_ms, start_ms - margin_ms)
        win_hi = min(last_ms + grid_ms, end_ms + margin_ms)
        gap_max_ms = max(config.gap_max_floor_s, config.gap_max_samples * int(grid_s)) * 1000
        for ch in gap_channels:
            # read_range returns the in-window rows plus the last sample before
            # win_lo (the seed). Seed prev_ts from it so a gap straddling the
            # leading edge is measured; the seed's own NULL weight is not
            # refused (it may legitimately sit in an earlier gap, outside the
            # window). The trailing edge from the last row to win_hi is checked
            # too, so a gap straddling the window end is refused.
            prev_ts: int | None = None
            for ts, w in read_range(conn, [ch], win_lo, win_hi).get(ch, []):
                if ts < win_lo:
                    prev_ts = ts
                    continue
                if w is None:
                    raise CaptureRefused(
                        "across_gap", {"gap_start": _iso(ts), "gap_end": _iso(ts)}
                    )
                if prev_ts is not None and (ts - prev_ts) > gap_max_ms:
                    raise CaptureRefused(
                        "across_gap",
                        {"gap_start": _iso(prev_ts), "gap_end": _iso(ts)},
                    )
                prev_ts = ts
            if prev_ts is not None and (win_hi - prev_ts) > gap_max_ms:
                raise CaptureRefused(
                    "across_gap",
                    {"gap_start": _iso(prev_ts), "gap_end": _iso(win_hi)},
                )
    finally:
        conn.close()

    return start_ms, end_ms

"""Epoch-aligned grid resampling and causal edge streaming (HA-free).

StreamEdges is a causal re-implementation of learning-time ``detect_edges``
followed by ``suppress_alignment_artefacts`` using plain floats.  It imports
the episode constants instead of copying them so tuning stays in one place
(DET-01, RESEARCH Pitfall 3).
"""

from collections import deque
import math
from typing import NamedTuple

from ..learning.episodes import MERGE_SAMPLES, MIN_EDGE_W


class GridPoint(NamedTuple):
    """One epoch-aligned grid sample."""

    grid_ts: float
    value: float  # NaN when unknown
    phases: tuple[float, float, float] | None


class GridResampler:
    """Zero-order-hold resampler onto epoch multiples of ``grid_s``.

    ``push`` stores the latest raw row; ``drain(now)`` yields every grid
    multiple ``g`` of ``grid_s`` with ``last_drained < g <= now - guard_s``,
    valued by the last pushed row with ``ts <= g`` (the reader.resample_zoh
    rule).  A grid point with no covering row yields NaN.  Rows are consumed
    as grid points pass them, so memory stays O(1) for in-order feeds.
    """

    def __init__(self, grid_s: int, guard_s: float = 0.0) -> None:
        self._grid_s = int(grid_s)
        self._guard_s = float(guard_s)
        self._rows: deque[tuple[float, float | None, tuple | None]] = deque()
        self._last_drained: float | None = None

    def push(self, ts: float, value: float | None, phases: tuple | None = None) -> None:
        """Store one raw row."""
        self._rows.append((float(ts), value, phases))

    def drain(self, now: float) -> list[GridPoint]:
        """Yield all grid points up to ``now - guard_s``."""
        grid = self._grid_s
        limit = float(now) - self._guard_s
        if self._last_drained is None:
            if not self._rows:
                return []
            first_ts = self._rows[0][0]
            g = math.ceil(first_ts / grid - 1e-9) * grid
            if g < first_ts:
                g += grid
            g = float(g)
        else:
            g = self._last_drained + grid

        out: list[GridPoint] = []
        cur: tuple[float, float | None, tuple | None] | None = None
        while g <= limit + 1e-9:
            while self._rows and self._rows[0][0] <= g + 1e-9:
                cur = self._rows.popleft()
            value = float(cur[1]) if (cur is not None and cur[1] is not None) else math.nan
            phases = cur[2] if cur is not None else None
            out.append(GridPoint(g, value, phases))
            self._last_drained = g
            g += grid
        return out


class Edge(NamedTuple):
    """One finalised power step edge."""

    index: int  # pre-edge sample index of the pushed stream
    delta_w: float


class StreamEdges:
    """Causal edge detector with parity to detect_edges + suppression.

    ``push(value)`` returns the finalised edges.  The merge decision for a
    diff needs the next diff (lag 1 sample, the D-09 confirmation).  With
    ``suppress_alignment`` a finalised edge is held until its pairing window
    closes: an edge at index ``i`` can only pair with an edge at ``i + 1``,
    and no edge can start at ``i + 1`` once its diff fell below ``min_split``
    or was consumed by a merge, so the held edge is emitted as soon as the
    batch outcome is decidable.  A NaN value flushes the finite segment
    exactly as detect_edges ends a segment, then resets.
    """

    def __init__(self, threshold_w: float, suppress_alignment: bool) -> None:
        self._threshold = float(threshold_w)
        self._min_split = min(40.0, self._threshold * 0.25)
        self._suppress = bool(suppress_alignment)
        self._prev: float | None = None
        self._pending: tuple[int, float] | None = None  # (pre-edge index, diff)
        self._held: Edge | None = None
        self._next_index = 0

    def _emit(self, edge: Edge, out: list[Edge]) -> None:
        if not self._suppress:
            out.append(edge)
            return
        held = self._held
        if held is not None:
            if (
                held.index + 1 == edge.index
                and held.delta_w * edge.delta_w < 0
                and abs(held.delta_w + edge.delta_w) < 0.25 * abs(held.delta_w)
            ):
                self._held = None  # drop both; skip-two leaves nothing held
                return
            out.append(held)
        self._held = edge

    def _no_edge_at(self, idx: int, out: list[Edge]) -> None:
        """The diff at ``idx`` sealed without an edge: close any held window."""
        held = self._held
        if held is not None and held.index + 1 == idx:
            out.append(held)
            self._held = None

    def _probe(self, nxt_idx: int, nxt_d: float, out: list[Edge]) -> None:
        """Close the held edge's window if no edge can start at ``nxt_idx``."""
        held = self._held
        if held is not None and held.index + 1 == nxt_idx and abs(nxt_d) < self._min_split:
            out.append(held)
            self._held = None

    def _decide_pending_with(self, nxt_idx: int, nxt_d: float, out: list[Edge]) -> None:
        """Apply the detect_edges merge rule to the pending diff and the next."""
        assert self._pending is not None
        p_idx, p_d = self._pending
        self._pending = None
        if (
            (p_d > 0 and nxt_d > 0) or (p_d < 0 and nxt_d < 0)
        ) and abs(p_d) >= self._min_split and abs(nxt_d) >= self._min_split \
                and abs(p_d + nxt_d) >= self._threshold:
            # Merged edge consumes both diffs: no separate edge at nxt_idx.
            self._emit(Edge(p_idx, p_d + nxt_d), out)
            self._probe(nxt_idx, 0.0, out)
            return
        if abs(p_d) >= self._threshold:
            self._emit(Edge(p_idx, p_d), out)
        else:
            self._no_edge_at(p_idx, out)
        self._pending = (nxt_idx, nxt_d)
        # The held edge (sealed at p_idx) pairs only with an edge at
        # nxt_idx == p_idx + 1, impossible when the next diff is below
        # min_split (single needs >= threshold, merge needs >= min_split).
        self._probe(nxt_idx, nxt_d, out)

    def push(self, value: float) -> list[Edge]:
        """Feed one grid sample; return newly finalised edges."""
        out: list[Edge] = []
        v = float(value)
        if math.isnan(v):
            self.flush(out)
            self._prev = None
            self._pending = None
            self._next_index += 1
            return out
        if self._prev is not None:
            d = v - self._prev
            idx = self._next_index - 1
            if self._pending is None:
                self._pending = (idx, d)
            else:
                self._decide_pending_with(idx, d, out)
        self._prev = v
        self._next_index += 1
        return out

    def flush(self, out: list[Edge] | None = None) -> list[Edge]:
        """Finalise the trailing pending diff and emit the held edge."""
        if out is None:
            out = []
        pending = self._pending
        self._pending = None
        if pending is not None:
            if abs(pending[1]) >= self._threshold:
                self._emit(Edge(pending[0], pending[1]), out)
            else:
                self._no_edge_at(pending[0], out)
        held = self._held
        self._held = None
        if held is not None:
            out.append(held)
        return out

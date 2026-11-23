"""Property test: StreamEdges is provably the learning-time edge detection.

For 300 seeded random series (noise, steps, split steps, single-sample
opposite spikes, NaN runs), run twice -- once with E = (EventType.STEP,)
and once with E = (EventType.STEP, EventType.SHORT_EVENT) -- the
concatenated StreamEdges output must equal
suppress_alignment_artefacts(*detect_edges(r, threshold_w),
enabled_event_types=E) exactly (same indices, same deltas).  The real gate
decides the expected edges, locking the mapping
suppress_alignment = (SHORT_EVENT not in E) to episodes.py lines 98-108,
including the enabled-SHORT_EVENT case where the gate returns the edges
unchanged (review concern L59@4e11826).
"""

import numpy as np
import pytest

from custom_components.powerguesser.engine.detect.stream import StreamEdges
from custom_components.powerguesser.engine.intervals import EventType
from custom_components.powerguesser.engine.learning.episodes import (
    detect_edges,
    suppress_alignment_artefacts,
)
from tests.fixtures.detect_traces import random_series

THRESHOLD = 120.0
EVENT_SETS = [
    (EventType.STEP,),
    (EventType.STEP, EventType.SHORT_EVENT),
]


def _stream_edges(r: np.ndarray, suppress_alignment: bool) -> list[tuple[int, float]]:
    stream = StreamEdges(THRESHOLD, suppress_alignment)
    out: list[tuple[int, float]] = []
    for v in r:
        out.extend((e.index, e.delta_w) for e in stream.push(float(v)))
    out.extend((e.index, e.delta_w) for e in stream.flush())
    return out


@pytest.mark.parametrize("event_types", EVENT_SETS)
@pytest.mark.parametrize("seed", range(300))
def test_stream_edges_parity_with_learning_detection(seed: int, event_types) -> None:
    """Streaming output equals detect_edges + the real suppression gate."""
    r = random_series(seed, n=200)
    expected_idx, expected_delta = detect_edges(r, THRESHOLD)
    expected_idx, expected_delta = suppress_alignment_artefacts(
        expected_idx, expected_delta, enabled_event_types=event_types
    )
    expected = [(int(i), float(d)) for i, d in zip(expected_idx, expected_delta, strict=True)]

    suppress_alignment = EventType.SHORT_EVENT not in event_types
    assert _stream_edges(r, suppress_alignment) == expected

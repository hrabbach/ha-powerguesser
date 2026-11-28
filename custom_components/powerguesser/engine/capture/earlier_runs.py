"""HA-free earlier-runs replay count for a snapped capture (CAP-11, D-12).

``earlier_runs`` answers one question: *how many earlier runs in the buffer would
the captured signature have matched?* It fits the capture's own single-run D-02
tolerance artefact (so the number means "runs that look like this capture"),
rebuilds the residual over a bounded lookback window ending at the captured rise
through the learner's own ``_load_job_residual`` (the same executor-only context
the capture snap uses) and replays exactly that artefact through the production
detector.

The window ends at the grid-aligned captured rise, so the captured run itself and
every later run sit at or after the end and are never counted; the start is
``EARLIER_RUNS_LOOKBACK_DAYS`` before it, clipped to the buffer. The result is
stored once at settle and never recomputed, because the buffer rolls (D-12). The
count is shown with its basis (the capture's own tolerance box over a stated
number of days), never as an accuracy figure. numpy only, no Home Assistant
import.
"""

from collections.abc import Mapping
from typing import Any

import numpy as np

from ..detect.replay import replay_online
from ..learning.worker import _load_job_residual
from . import preview as preview_mod
from .config import DEFAULT_CONFIG, CaptureConfig

_MS_PER_DAY = 86_400_000
_BASIS_TEMPLATE = "this capture's tolerance box on the last {days} days"
_UNAVAILABLE = "unavailable"


def _align_down(value_ms: int, step_ms: int) -> int:
    return (value_ms // step_ms) * step_ms


def earlier_runs(
    context: Mapping[str, Any],
    capture: Mapping[str, Any],
    *,
    config: CaptureConfig = DEFAULT_CONFIG,
) -> dict[str, Any]:
    """Return ``{n, basis}`` for the capture's own tolerance artefact.

    ``n`` is the number of earlier buffered runs the capture's single-run D-02
    tolerance artefact matches by replay; ``basis`` names the span actually
    covered (the lookback clipped to the buffer), as days.
    """
    result = capture.get("result") or {}
    rise_ms = result.get("rise_ms")
    if rise_ms is None:
        return {"n": 0, "basis": _BASIS_TEMPLATE.format(days=0)}

    grid_s = int(context.get("grid_s", 30) or 30)
    step_ms = grid_s * 1000

    marked = {**dict(capture), "result": {**dict(result), "verdict": "clean"}}
    artefact = preview_mod.fit_preview("cap_earlier", "earlier", [marked], config=config)
    if not artefact:
        return {"n": None, "basis": _UNAVAILABLE}

    end_ms = _align_down(int(rise_ms), step_ms)
    start_ms = _align_down(
        int(rise_ms) - config.earlier_runs_lookback_days * _MS_PER_DAY, step_ms
    )

    job = {**dict(context), "day_bounds_ms": [start_ms, end_ms]}
    series, _window_by_key = _load_job_residual(job)
    residual = series.residual

    enabled = context.get("enabled_event_types", []) or []
    out = replay_online(
        [artefact],
        residual,
        float(grid_s),
        phases=series.phases or None,
        suppress_alignment="short_event" not in enabled,
    )
    sig = str(artefact.get("signature_id", "cap_earlier"))
    n = len(out.episodes.get(sig, []))

    finite = int(np.count_nonzero(np.isfinite(residual)))
    covered_days = round(finite * grid_s / 86_400.0, 1)
    return {"n": n, "basis": _BASIS_TEMPLATE.format(days=f"{covered_days:g}")}

"""HA-free D-04 read-only replay check for a device-target capture.

``replay_check`` answers one question: *would the device's active version alone
have claimed this captured run?* It rebuilds the residual over the capture
window (a 600 s pre-roll before the rise and two grid steps after the fall,
aligned to the grid) through the learner's own ``_load_job_residual`` — the same
context the capture snap uses — and replays exactly one artefact through the
production detector.

The result is deliberately honest about its scope: competition from other
devices is **not** modelled, so the block says so in ``basis``. The check is
strictly read-only: it never touches the device record, writes no proposal and
changes no version (D-04). A program device is never matched to a single
rise/fall run. numpy only, no Home Assistant import.
"""

from collections.abc import Mapping
from typing import Any

from ..detect.replay import replay_online
from ..learning.worker import _load_job_residual
from .config import DEFAULT_CONFIG, CaptureConfig

_PROGRAM_BASIS = "program devices are not matched to a single run"
_BASIS = "active version alone on the capture window"


def _align_down(value_ms: int, step_ms: int) -> int:
    return (value_ms // step_ms) * step_ms


def _align_up(value_ms: int, step_ms: int) -> int:
    return -((-value_ms) // step_ms) * step_ms


def replay_check(
    context: Mapping[str, Any],
    artefact: Mapping[str, Any] | None,
    *,
    rise_ms: int,
    fall_ms: int,
    config: CaptureConfig = DEFAULT_CONFIG,
) -> dict[str, Any]:
    """Return ``{would_claim, n_episodes, basis}`` for the device's active artefact.

    A non-switched (program) or missing artefact claims nothing. ``would_claim``
    is True when at least one replayed episode overlaps the capture's
    ``[rise, fall]`` sample indices; ``n_episodes`` is that count. ``context``
    carries the same keys as the capture snap's channel context (buffer_path,
    grid_s, enabled_event_types, metered_channels, phase_channels,
    session_windows).
    """
    if not isinstance(artefact, Mapping) or artefact.get("kind") != "switched":
        return {"would_claim": False, "n_episodes": 0, "basis": _PROGRAM_BASIS}

    grid_s = int(context.get("grid_s", 30))
    step_ms = grid_s * 1000
    # Grid convention: load_residual grids [start, end) at step_ms, so the last
    # sample sits at end - step. Aligning the end to fall + 3 steps makes the last
    # grid sample fall + 2 grid steps, i.e. two baseline samples after the fall
    # even when the snapped fall_ms is up to one step before the true off edge.
    start_ms = _align_down(int(rise_ms) - config.replay_pre_roll_s * 1000, step_ms)
    end_ms = _align_up(int(fall_ms), step_ms) + 3 * step_ms

    job = {**dict(context), "day_bounds_ms": [start_ms, end_ms]}
    series, _window_by_key = _load_job_residual(job)
    r = series.residual
    phases = series.phases

    enabled = context.get("enabled_event_types", []) or []
    art = dict(artefact)
    sig = str(art.get("signature_id", "sig0"))
    out = replay_online(
        [art],
        r,
        float(grid_s),
        phases=phases or None,
        suppress_alignment="short_event" not in enabled,
    )

    rise_i = int(round((int(rise_ms) - start_ms) / step_ms))
    fall_i = int(round((int(fall_ms) - start_ms) / step_ms))
    episodes = out.episodes.get(sig, [])
    n_episodes = sum(
        1 for ep in episodes if ep.start_i <= fall_i and ep.end_i >= rise_i
    )
    return {"would_claim": n_episodes >= 1, "n_episodes": n_episodes, "basis": _BASIS}

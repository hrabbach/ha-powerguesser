#!/usr/bin/env python3
"""Offline tuning harness for the PowerGuesser engine (Plan 06-07; D-22).

Dev-only: this script is NOT part of the integration package and is never
imported by it. It opens a COPY of the reference-install buffer through
``engine.reader.open_readonly`` (SQLite ``mode=ro``), measures, and prints one
sorted-key JSON document per run. Nothing here assigns an engine constant: the
values that ship this phase are conservative initial values, and the tuned ones
come from running these commands on the real buffer copy as a post-ship UAT
item (D-22, D-03, CR-03, the folded 60 s todo).

Subcommands
-----------
session-scaling
    Re-check the CR-03 session clustering scaling. The session rows of the
    window are clustered once with ``digest.session_features`` and then for a
    3 x 3 grid of duration/rise multipliers ``m_dur`` x ``m_rise``: the ln
    energy, ln duration and ln segment-count columns (0, 1, 3) are multiplied
    by ``DUR_SCALE / (DUR_SCALE x m_dur)`` and the ln peak column (2) by
    ``RISE_SCALE / (RISE_SCALE x m_rise)``, so the printed grid says what a
    different scaling would do to the clusters. Per setting: n_sessions,
    n_clusters, the cluster sizes, and the median within-cluster spread of ln
    energy and ln duration (spread = max - min within a cluster of >= 2
    members; the median over those clusters, 0.0 when there is none).

interval-60s
    Evaluate device detection at 60 s sampling. The 30 s residual of the
    window is compared with its two 60 s decimations (every second sample, and
    the mean of each pair); each variant reports its edge count (detect_edges
    with that variant's own edge_threshold, alignment suppression as in the
    learning path), its paired switched-episode count (pair_switched rows) and
    its explained energy share (paired energy / positive residual energy). The
    STEP gate stays unchanged until native 60 s data is measured.

lag-cap
    Measure the D-03 recognition lag of one or more program artefacts for a
    list of ``PROGRAM_PROMOTE_FRACTION`` values. See ``_cmd_lag_cap``.

Usage (the reference-install run is post-ship, on a copy of the buffer):

    cp buffer.sqlite /tmp/pg-buffer.sqlite
    python scripts/tune_offline.py session-scaling --buffer /tmp/pg-buffer.sqlite --days 35
    python scripts/tune_offline.py interval-60s --buffer /tmp/pg-buffer.sqlite --days 14
    python scripts/tune_offline.py lag-cap --buffer /tmp/pg-buffer.sqlite --days 7 \
        --artefacts washer_program.json --plug-entity sensor.washer_power \
        --fractions 0.5,1.0,1.5
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import argparse
from dataclasses import replace
from datetime import datetime, timezone
import itertools
import json
import math
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
_ENGINE_ROOT = _REPO_ROOT / "custom_components" / "powerguesser"
if str(_ENGINE_ROOT) not in sys.path:
    sys.path.insert(0, str(_ENGINE_ROOT))

import numpy as np  # noqa: E402

from engine.detect.online import OnlineDetector, compile_model  # noqa: E402
from engine.detect.program import PROGRAM_PROMOTE_MIN_STEPS  # noqa: E402
from engine.learning.digest import (  # noqa: E402
    DUR_SCALE,
    RISE_SCALE,
    dbscan,
    session_features,
)
from engine.learning.episodes import (  # noqa: E402
    MIN_EDGE_W,
    detect_edges,
    edge_threshold,
    pair_switched,
    plug_sessions,
    suppress_alignment_artefacts,
)
from engine.learning.series import load_residual  # noqa: E402
from engine.learning.worker import build_episode_table  # noqa: E402
from engine.reader import open_readonly  # noqa: E402
from engine.validation.metrics import energy_error_pct  # noqa: E402

DAY_MS = 86_400_000
DEFAULT_EVENT_TYPES: tuple[str, ...] = ("step", "cycle_shape", "continuous")
MULTIPLIERS: tuple[float, ...] = (0.5, 1.0, 2.0)
DEFAULT_FRACTIONS = "0.5,1.0,1.5"


# --- shared helpers ---------------------------------------------------------


def _first_weekday(start_ms: int) -> int:
    return datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc).weekday()


def _window(buffer_path: str | Path, days: int, grid_s: int) -> tuple[int, int, list[int]]:
    """(start_ms, end_ms, day_bounds_ms) for the last N whole UTC days.

    The window ends at the UTC midnight at or before the newest buffered
    sample and starts ``days`` whole days earlier, so every day in it is
    complete and the runs of two consecutive invocations are comparable.
    """
    del grid_s  # the window itself is grid-independent
    conn = open_readonly(str(buffer_path))
    try:
        row = conn.execute("SELECT MAX(ts) FROM samples").fetchone()
    finally:
        conn.close()
    newest = int(row[0]) if row and row[0] is not None else None
    if newest is None:
        raise SystemExit(f"buffer {buffer_path} holds no samples")
    end_ms = (newest // DAY_MS) * DAY_MS
    start_ms = end_ms - int(days) * DAY_MS
    bounds = [start_ms + i * DAY_MS for i in range(int(days) + 1)]
    return start_ms, end_ms, bounds


def _channels(buffer_path: str | Path) -> list[tuple[int, str, str, str]]:
    """(channel id, entity_id, role, kind) for every buffered channel."""
    conn = open_readonly(str(buffer_path))
    try:
        rows = conn.execute(
            "SELECT id, entity_id, role, kind FROM channels ORDER BY id"
        ).fetchall()
    finally:
        conn.close()
    return [(int(r[0]), str(r[1]), str(r[2]), str(r[3])) for r in rows]


def _metered_channels(buffer_path: str | Path, explicit: str | None) -> list[int]:
    """The metered channels to subtract: the CLI list, else every kind 'metered'."""
    known = _channels(buffer_path)
    if explicit is None:
        return [cid for cid, _e, _r, kind in known if kind == "metered"]
    out: list[int] = []
    for token in str(explicit).split(","):
        token = token.strip()
        if not token:
            continue
        if token.lstrip("-").isdigit():
            out.append(int(token))
            continue
        match = [cid for cid, entity, _r, _k in known if entity == token]
        if not match:
            raise SystemExit(f"entity {token!r} is not a channel of this buffer")
        out.append(match[0])
    return sorted(set(out))


def _resolve_channel(buffer_path: str | Path, entity_id: str) -> int:
    match = [cid for cid, entity, _r, _k in _channels(buffer_path) if entity == entity_id]
    if not match:
        raise SystemExit(f"entity {entity_id!r} is not a channel of this buffer")
    return match[0]


def _detector_threshold(models: Sequence) -> float:
    """The replay threshold: the smallest artefact edge threshold, floored."""
    thresholds = [m.threshold_w for m in models if m.threshold_w > 0.0]
    return max(MIN_EDGE_W, min(thresholds) if thresholds else MIN_EDGE_W)


def _decimate_every_second(r: np.ndarray, factor: int = 2) -> np.ndarray:
    """Keep every ``factor``-th sample (the 60 s 'every second sample' variant)."""
    return np.asarray(r, dtype=np.float64)[::factor]


def _decimate_mean(r: np.ndarray, factor: int = 2) -> np.ndarray:
    """Average each group of ``factor`` samples (the 60 s 'mean of pairs' variant)."""
    arr = np.asarray(r, dtype=np.float64)
    usable = len(arr) - (len(arr) % factor)
    if usable == 0:
        return np.empty(0, dtype=np.float64)
    return arr[:usable].reshape(-1, factor).mean(axis=1)


def _median_spread(values: np.ndarray, labels: np.ndarray) -> float:
    """Median max-minus-min of ``values`` within clusters of at least two members."""
    spreads: list[float] = []
    for cluster in sorted({int(v) for v in labels} - {-1}):
        members = values[labels == cluster]
        if len(members) < 2:
            continue
        spreads.append(float(np.max(members) - np.min(members)))
    if not spreads:
        return 0.0
    return round(float(np.median(spreads)), 4)


# --- subcommands ------------------------------------------------------------


def _cmd_session_scaling(args: argparse.Namespace) -> dict:
    """Cluster the window's session rows over a 3 x 3 scaling grid (CR-03)."""
    grid_s = int(args.grid_s)
    step_ms = grid_s * 1000
    start_ms, end_ms, bounds = _window(args.buffer, args.days, grid_s)
    metered = _metered_channels(args.buffer, args.metered)
    series = load_residual(str(args.buffer), start_ms, end_ms, step_ms, metered, {})
    table, _rle, _threshold = build_episode_table(
        series.residual,
        series.phases,
        start_ms,
        step_ms,
        bounds,
        _first_weekday(start_ms),
        list(DEFAULT_EVENT_TYPES),
    )

    sess_idx = np.where(table["kind"] == 1)[0]
    X = session_features(
        {k: table[k][sess_idx] for k in ("energy_wh", "dur_s", "peak_w", "n_segments")}
    )
    # The natural-log features behind the scaled columns, for the spread report.
    ln_energy = X[:, 0] * DUR_SCALE
    ln_duration = X[:, 1] * DUR_SCALE

    settings: list[dict] = []
    for m_dur, m_rise in itertools.product(MULTIPLIERS, MULTIPLIERS):
        scaled = X.copy()
        scaled[:, [0, 1, 3]] *= DUR_SCALE / (DUR_SCALE * m_dur)
        scaled[:, 2] *= RISE_SCALE / (RISE_SCALE * m_rise)
        labels = dbscan(scaled)
        sizes = [
            int(np.sum(labels == cluster))
            for cluster in sorted({int(v) for v in labels} - {-1})
        ]
        settings.append(
            {
                "m_dur": m_dur,
                "m_rise": m_rise,
                "n_sessions": int(len(X)),
                "n_clusters": len(sizes),
                "sizes": sizes,
                "median_spread_ln_energy": _median_spread(ln_energy, labels),
                "median_spread_ln_duration": _median_spread(ln_duration, labels),
            }
        )

    return {
        "subcommand": "session-scaling",
        "grid_s": grid_s,
        "days": int(args.days),
        "metered_channels": metered,
        "window": {"start_ms": start_ms, "end_ms": end_ms},
        "n_sessions": int(len(X)),
        "settings": settings,
    }


def _cmd_interval_60s(args: argparse.Namespace) -> dict:
    """Compare the 30 s residual with its two 60 s decimations (folded 60 s todo)."""
    grid_s = int(args.grid_s)
    step_ms = grid_s * 1000
    start_ms, end_ms, _bounds = _window(args.buffer, args.days, grid_s)
    metered = _metered_channels(args.buffer, args.metered)
    series = load_residual(str(args.buffer), start_ms, end_ms, step_ms, metered, {})
    r = np.asarray(series.residual, dtype=np.float64)

    variants = (
        ("30s", r, float(grid_s)),
        ("60s_every_second", _decimate_every_second(r), float(grid_s * 2)),
        ("60s_mean", _decimate_mean(r), float(grid_s * 2)),
    )

    out: list[dict] = []
    for name, arr, variant_step_s in variants:
        finite = np.isfinite(arr)
        threshold = edge_threshold(np.diff(arr[finite]))
        idx, delta = detect_edges(arr, threshold)
        idx, delta = suppress_alignment_artefacts(
            idx, delta, enabled_event_types=list(DEFAULT_EVENT_TYPES)
        )
        table = pair_switched(arr, idx, delta, variant_step_s)
        paired_wh = float(np.sum(table["energy_wh"]))
        positive_wh = float(np.sum(np.maximum(arr[finite], 0.0)) * variant_step_s / 3600.0)
        out.append(
            {
                "variant": name,
                "step_s": variant_step_s,
                "n_samples": int(len(arr)),
                "threshold_w": round(float(threshold), 1),
                "edges": int(len(idx)),
                "pairs": int(len(table["start_i"])),
                "paired_energy_wh": round(paired_wh, 3),
                "positive_energy_wh": round(positive_wh, 3),
                "explained_share": (
                    round(paired_wh / positive_wh, 4) if positive_wh > 0.0 else None
                ),
            }
        )

    return {
        "subcommand": "interval-60s",
        "grid_s": grid_s,
        "days": int(args.days),
        "metered_channels": metered,
        "window": {"start_ms": start_ms, "end_ms": end_ms},
        "variants": out,
    }


# --- entry point -----------------------------------------------------------


def _cmd_lag_cap(args: argparse.Namespace) -> dict:
    """Measure the D-03 recognition lag per PROGRAM_PROMOTE_FRACTION value.

    The gross residual is read with no metered channel subtracted, so the
    plug's own load stays in the series (exactly like the validate stage), and
    the plug's ON runs come from ``plug_sessions`` on its own channel. A run
    the window cuts — still ON at the window end, or already ON at its first
    sample — is counted in ``runs_truncated`` and excluded from runs_plug,
    runs_detected, runs_missed and every lag figure, because its start (or its
    recognition window) is not observable.

    A program enters ``StepResult.on`` only at promotion (Plan 06-01, D-33), so
    the recognition lag of a plug run is the grid time of the first step inside
    that run whose ``StepResult.on`` holds the program key, minus the run's
    start time. The promotion threshold for a fraction is
    ``max(PROGRAM_PROMOTE_MIN_STEPS, ceil(fraction x stage-1 dur_s.lo / grid_s))``
    applied with ``dataclasses.replace`` to the compiled model; the shipped
    ``PROGRAM_PROMOTE_FRACTION`` is never assigned.
    """
    grid_s = int(args.grid_s)
    step_ms = grid_s * 1000
    start_ms, end_ms, _bounds = _window(args.buffer, args.days, grid_s)

    fractions = [
        float(token)
        for token in str(args.fractions).split(",")
        if token.strip()
    ]
    if not fractions:
        raise SystemExit("--fractions must name at least one value")

    raw_artefacts = json.loads(Path(args.artefacts).read_text(encoding="utf-8"))
    if not isinstance(raw_artefacts, list) or not raw_artefacts:
        raise SystemExit("--artefacts must hold a JSON list of program artefact dicts")

    plug_channel = _resolve_channel(args.buffer, args.plug_entity)

    # The gross residual (the plug is NOT subtracted: its load must stay in the
    # series so the program tracker sees the run), plus the plug's own series.
    gross = load_residual(str(args.buffer), start_ms, end_ms, step_ms, [], {})
    r = np.asarray(gross.gross, dtype=np.float64)
    plug_series = load_residual(str(args.buffer), start_ms, end_ms, step_ms, [plug_channel], {})
    y = np.asarray(plug_series.plugs[plug_channel], dtype=np.float64)
    n = len(r)

    table, _rle = plug_sessions(y, float(grid_s))
    complete: list[tuple[int, int]] = []
    runs_truncated = 0
    for start_i, end_i in zip(table["start_i"], table["end_i"]):
        first, last = int(start_i), int(end_i)
        if first == 0 or last >= n - 1:
            runs_truncated += 1
            continue
        complete.append((first, last))

    compiled = [
        compile_model(str(artefact.get("signature_id", "")), artefact)
        for artefact in raw_artefacts
        if isinstance(artefact, Mapping)
    ]
    if not compiled:
        raise SystemExit("--artefacts holds no usable program artefact")
    # The first artefact is the program the figures are measured for; any
    # further ones are compiled and replayed alongside it as competitors.
    primary = compiled[0].key
    primary_model = compiled[0]
    stage1_dur_lo = float(primary_model.stages[0].dur[0])

    out: list[dict] = []
    for fraction in fractions:
        promote = max(
            PROGRAM_PROMOTE_MIN_STEPS,
            int(math.ceil(fraction * stage1_dur_lo / float(grid_s))),
        )
        models = [
            replace(model, promote_steps=promote) if model.key == primary else model
            for model in compiled
        ]
        detector = OnlineDetector(
            models, grid_s=grid_s, threshold_w=_detector_threshold(models)
        )
        on_flags = np.zeros(n, dtype=bool)
        energy_delta = np.zeros(n, dtype=np.float64)
        for i in range(n):
            result = detector.step(float(i) * float(grid_s), float(r[i]))
            on_flags[i] = primary in result.on
            energy_delta[i] = float(result.energy_wh_delta.get(primary, 0.0))
        detector.flush()

        lags: list[float] = []
        detected = 0
        est_wh = 0.0
        true_wh = 0.0
        for first, last in complete:
            hits = np.nonzero(on_flags[first : last + 1])[0]
            if len(hits):
                detected += 1
                lags.append(float(hits[0]) * float(grid_s))
            est_wh += float(np.sum(energy_delta[first : last + 1]))
            true_wh += float(np.sum(y[first : last + 1])) * float(grid_s) / 3600.0

        out.append(
            {
                "fraction": fraction,
                "promote_steps": promote,
                "runs_plug": len(complete),
                "runs_truncated": runs_truncated,
                "runs_detected": detected,
                "runs_missed": len(complete) - detected,
                "lag_s_median": round(float(np.median(lags)), 1) if lags else None,
                "lag_s_max": round(max(lags), 1) if lags else None,
                "energy_true_wh": round(true_wh, 1),
                "energy_est_wh": round(est_wh, 1),
                "energy_error_pct": energy_error_pct(true_wh, est_wh),
            }
        )

    return {
        "subcommand": "lag-cap",
        "grid_s": grid_s,
        "days": int(args.days),
        "window": {"start_ms": start_ms, "end_ms": end_ms},
        "plug_entity": str(args.plug_entity),
        "plug_channel": plug_channel,
        "program_keys": [str(model.key) for model in compiled],
        "primary_program_key": str(primary),
        "stage1_dur_lo_s": round(stage1_dur_lo, 1),
        "fractions": out,
    }


def build_parser() -> argparse.ArgumentParser:
    """The subcommand layout (buffer/days/grid-s shared, metered where it applies)."""
    parser = argparse.ArgumentParser(
        description="Offline engine tuning harness (dev-only; reads a buffer copy read-only)"
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    def add_common(sub: argparse.ArgumentParser, *, with_metered: bool) -> None:
        sub.add_argument(
            "--buffer",
            required=True,
            help="path to a COPY of the buffer SQLite file (opened read-only)",
        )
        sub.add_argument(
            "--days",
            type=int,
            default=14,
            help="window = the last N whole UTC days before the newest buffered sample",
        )
        sub.add_argument(
            "--grid-s",
            type=int,
            default=30,
            help="residual grid in seconds (every load_residual call uses grid_s x 1000 ms)",
        )
        if with_metered:
            sub.add_argument(
                "--metered",
                default=None,
                help=(
                    "comma-separated channel ids or entity ids to subtract as metered; "
                    "default: every channel whose kind is 'metered'"
                ),
            )

    add_common(subparsers.add_parser("session-scaling", help="CR-03 session scaling grid"), with_metered=True)
    add_common(subparsers.add_parser("interval-60s", help="30 s vs 60 s detection quality"), with_metered=True)
    lag = subparsers.add_parser("lag-cap", help="D-03 recognition lag per promote fraction")
    add_common(lag, with_metered=False)
    lag.add_argument(
        "--artefacts",
        required=True,
        help=(
            "JSON file holding a list of program artefact dicts; the first one's "
            "signature_id is the program the figures are measured for"
        ),
    )
    lag.add_argument(
        "--plug-entity",
        required=True,
        help="entity id of the plug channel the washer/dryer runs are read from",
    )
    lag.add_argument(
        "--fractions",
        default=DEFAULT_FRACTIONS,
        help=f"comma-separated PROGRAM_PROMOTE_FRACTION values (default {DEFAULT_FRACTIONS})",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one subcommand and print its sorted-key JSON document."""
    args = build_parser().parse_args(argv)
    if args.subcommand == "session-scaling":
        payload = _cmd_session_scaling(args)
    elif args.subcommand == "interval-60s":
        payload = _cmd_interval_60s(args)
    else:
        payload = _cmd_lag_cap(args)
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
